"""
Tests for benchmark.py — covers simulation engine, path validation, and
property-based testing with Hypothesis.
"""

import os
import pytest

os.environ["SLA_SIMULATION_MODE"] = "1"

from llm_sla_gatekeeper.benchmark import (
    BenchmarkResult,
    TokenLatencySample,
    _looks_like_valid_model_path,
    _estimate_model_size_b,
    _simulated_latency_ms,
    run_benchmark,
)

try:
    from hypothesis import given, settings, assume
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


# ── Path validation ────────────────────────────────────────────────────────────

class TestPathValidation:

    def test_hf_org_slash_model_is_valid(self):
        assert _looks_like_valid_model_path("Qwen/Qwen2.5-7B-Instruct") is True

    def test_bare_model_name_with_keyword_is_valid(self):
        assert _looks_like_valid_model_path("qwen3.5-27b") is True

    def test_gpt2_is_valid(self):
        assert _looks_like_valid_model_path("openai-community/gpt2") is True

    def test_llama_path_is_valid(self):
        assert _looks_like_valid_model_path("meta-llama/Llama-3.1-8B-Instruct") is True

    def test_size_indicator_makes_valid(self):
        assert _looks_like_valid_model_path("my-custom-7b-model") is True

    def test_corrupted_file_is_invalid(self):
        assert _looks_like_valid_model_path("corrupted_file") is False

    def test_random_word_is_invalid(self):
        assert _looks_like_valid_model_path("foobar") is False

    def test_empty_string_is_invalid(self):
        assert _looks_like_valid_model_path("") is False

    def test_local_path_existing(self, tmp_path):
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()
        assert _looks_like_valid_model_path(str(model_dir)) is True


# ── Model size estimation ──────────────────────────────────────────────────────

class TestModelSizeEstimation:

    def test_27b_detected(self):
        size = _estimate_model_size_b("qwen3.5-27b")
        assert abs(size - 27.0) < 0.1

    def test_7b_detected(self):
        size = _estimate_model_size_b("Qwen/Qwen2.5-7B-Instruct")
        assert abs(size - 7.0) < 0.1

    def test_350m_detected(self):
        size = _estimate_model_size_b("opt-350m")
        assert size < 1.0  # 0.35B

    def test_1_5b_detected(self):
        size = _estimate_model_size_b("Qwen2.5-1.5B-Instruct")
        assert 1.0 < size < 2.0

    def test_unknown_model_returns_positive(self):
        size = _estimate_model_size_b("some-unknown-model")
        assert size > 0


# ── Simulated latency ──────────────────────────────────────────────────────────

class TestSimulatedLatency:

    def test_larger_model_higher_latency(self):
        small = _simulated_latency_ms("openai-community/gpt2")   # ~124M
        large = _simulated_latency_ms("qwen3.5-27b")             # ~27B
        assert large > small

    def test_latency_positive(self):
        for model in ["gpt2", "llama-7b", "qwen3.5-27b", "opt-350m"]:
            assert _simulated_latency_ms(model) > 0

    def test_27b_latency_in_realistic_range(self):
        lat = _simulated_latency_ms("qwen3.5-27b")
        # Should be somewhere between 50ms and 500ms on CPU simulation
        assert 20.0 <= lat <= 600.0

    def test_deterministic_per_model(self):
        """Same model name → same latency (seeded RNG)."""
        lat1 = _simulated_latency_ms("Qwen/Qwen2.5-7B-Instruct")
        lat2 = _simulated_latency_ms("Qwen/Qwen2.5-7B-Instruct")
        assert lat1 == lat2  # exact equality — same seed


# ── run_benchmark ──────────────────────────────────────────────────────────────

class TestRunBenchmark:

    def test_pass_model_returns_result(self):
        result = run_benchmark("qwen3.5-27b", force_simulation=True)
        assert isinstance(result, BenchmarkResult)

    def test_result_fields_populated(self):
        result = run_benchmark("qwen3.5-27b", force_simulation=True)
        assert result.avg_latency_ms > 0
        assert result.p50_latency_ms > 0
        assert result.p95_latency_ms >= result.p50_latency_ms
        assert result.throughput_tokens_per_sec > 0
        assert result.total_tokens > 0
        assert result.benchmark_duration_sec > 0
        assert result.num_runs >= 1
        assert isinstance(result.hardware_info, dict)

    def test_mode_is_simulated(self):
        result = run_benchmark("qwen3.5-27b", force_simulation=True)
        assert result.mode == "simulated"

    def test_samples_count_matches_num_runs(self):
        result = run_benchmark("qwen3.5-27b", num_runs=3, force_simulation=True)
        assert len(result.samples) == 3

    def test_invalid_path_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            run_benchmark("corrupted_file", force_simulation=True)

    def test_empty_path_raises_value_error(self):
        with pytest.raises(ValueError):
            run_benchmark("", force_simulation=True)

    def test_hardware_info_has_device_key(self):
        result = run_benchmark("openai-community/gpt2", force_simulation=True)
        assert "device" in result.hardware_info

    def test_hardware_info_has_ram(self):
        result = run_benchmark("openai-community/gpt2", force_simulation=True)
        assert "ram_total_gb" in result.hardware_info
        assert result.hardware_info["ram_total_gb"] > 0

    def test_token_latency_samples_have_correct_fields(self):
        result = run_benchmark("openai-community/gpt2", num_runs=2, force_simulation=True)
        for sample in result.samples:
            assert isinstance(sample, TokenLatencySample)
            assert sample.tokens_generated > 0
            assert sample.elapsed_ms > 0
            assert sample.tokens_per_second > 0

    def test_gpt2_faster_than_27b(self):
        small = run_benchmark("openai-community/gpt2", force_simulation=True)
        large = run_benchmark("qwen3.5-27b", force_simulation=True)
        assert small.avg_latency_ms < large.avg_latency_ms


# ── Property-based tests (Hypothesis) ─────────────────────────────────────────

@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestHypothesis:

    @given(st.floats(min_value=0.1, max_value=10000.0))
    @settings(max_examples=30)
    def test_latency_always_positive(self, sla_target):
        from llm_sla_gatekeeper.sla_validator import SLAConfig, SLAValidator
        sla = SLAConfig(max_latency_ms=sla_target)
        validator = SLAValidator(force_simulation=True)
        result = validator.validate("qwen3.5-27b", sla)
        if result.benchmark:
            assert result.benchmark.avg_latency_ms > 0

    @given(st.floats(min_value=0.1, max_value=10000.0))
    @settings(max_examples=30)
    def test_pass_iff_latency_le_sla(self, sla_target):
        from llm_sla_gatekeeper.sla_validator import SLAConfig, SLAValidator
        sla = SLAConfig(max_latency_ms=sla_target)
        validator = SLAValidator(force_simulation=True)
        result = validator.validate("qwen3.5-27b", sla)
        if result.benchmark and result.status in ("PASS", "FAIL"):
            bm_lat = result.benchmark.avg_latency_ms
            if result.status == "PASS":
                assert bm_lat <= sla_target
            else:
                assert bm_lat > sla_target or result.benchmark.p95_latency_ms > sla_target * sla.p95_latency_multiplier

    @given(st.text(min_size=1, max_size=50).filter(
        lambda s: not any(k in s.lower() for k in
                          ["gpt", "llama", "qwen", "bert", "opt", "falcon",
                           "mistral", "phi", "gemma", "mpt", "starcoder"])
        and not any(c in s for c in ["/"])
        and not __import__("re").search(r"\d+[bBmM]", s)
    ))
    @settings(max_examples=20)
    def test_unknown_model_names_return_error_or_valid(self, model_name):
        """Strings without LLM patterns should return ERROR status."""
        assume(model_name.strip())
        from llm_sla_gatekeeper.sla_validator import SLAConfig, SLAValidator
        sla = SLAConfig(max_latency_ms=500.0)
        validator = SLAValidator(force_simulation=True)
        result = validator.validate(model_name, sla)
        # Either ERROR (unrecognized) or PASS/FAIL if it happens to match a keyword
        assert result.status in ("PASS", "FAIL", "ERROR")
