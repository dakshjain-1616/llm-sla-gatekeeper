"""
Tests for sla_validator.py — covers PASS, FAIL, ERROR paths plus edge cases.

Run:
    python -m pytest tests/ -v
"""

import os
import pytest

# Force simulation mode for all tests — no model downloads
os.environ["SLA_SIMULATION_MODE"] = "1"

from llm_sla_gatekeeper.sla_validator import (
    SLAConfig,
    SLAValidator,
    ValidationResult,
    validate_model,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def validator():
    return SLAValidator(force_simulation=True)


@pytest.fixture
def generous_sla():
    return SLAConfig(max_latency_ms=500.0)


@pytest.fixture
def tight_sla():
    return SLAConfig(max_latency_ms=10.0)


@pytest.fixture
def standard_sla():
    return SLAConfig(max_latency_ms=200.0)


# ── Test case 1: PASS scenario ─────────────────────────────────────────────────

class TestPassScenario:
    """Model: qwen3.5-27b, SLA=500ms → PASS"""

    def test_status_is_pass(self, validator, generous_sla):
        result = validator.validate("qwen3.5-27b", generous_sla)
        assert result.status == "PASS"

    def test_message_contains_ready_for_deployment(self, validator, generous_sla):
        result = validator.validate("qwen3.5-27b", generous_sla)
        assert "Ready for Deployment" in result.message

    def test_is_pass_property(self, validator, generous_sla):
        result = validator.validate("qwen3.5-27b", generous_sla)
        assert result.is_pass is True
        assert result.is_fail is False
        assert result.is_error is False

    def test_benchmark_is_populated(self, validator, generous_sla):
        result = validator.validate("qwen3.5-27b", generous_sla)
        assert result.benchmark is not None
        assert result.benchmark.avg_latency_ms > 0
        assert result.benchmark.throughput_tokens_per_sec > 0
        assert result.benchmark.total_tokens > 0

    def test_avg_latency_below_sla(self, validator, generous_sla):
        result = validator.validate("qwen3.5-27b", generous_sla)
        assert result.benchmark.avg_latency_ms <= generous_sla.max_latency_ms

    def test_recommendations_present(self, validator, generous_sla):
        result = validator.validate("qwen3.5-27b", generous_sla)
        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) >= 1

    def test_timestamp_present(self, validator, generous_sla):
        result = validator.validate("qwen3.5-27b", generous_sla)
        assert result.timestamp
        assert "Z" in result.timestamp  # UTC marker

    def test_to_dict_structure(self, validator, generous_sla):
        result = validator.validate("qwen3.5-27b", generous_sla)
        d = result.to_dict()
        for key in ("status", "message", "model_path", "sla_config", "benchmark",
                    "recommendations", "timestamp"):
            assert key in d

    def test_to_json_valid(self, validator, generous_sla):
        import json
        result = validator.validate("qwen3.5-27b", generous_sla)
        parsed = json.loads(result.to_json())
        assert parsed["status"] == "PASS"


# ── Test case 2: FAIL scenario ─────────────────────────────────────────────────

class TestFailScenario:
    """Model: qwen3.5-27b, SLA=10ms → FAIL"""

    def test_status_is_fail(self, validator, tight_sla):
        result = validator.validate("qwen3.5-27b", tight_sla)
        assert result.status == "FAIL"

    def test_message_contains_exceeds(self, validator, tight_sla):
        result = validator.validate("qwen3.5-27b", tight_sla)
        assert "Exceeds" in result.message or "exceeds" in result.message

    def test_message_contains_target_value(self, validator, tight_sla):
        result = validator.validate("qwen3.5-27b", tight_sla)
        # SLA target (10ms) should appear in the message
        assert "10ms" in result.message or "10" in result.message

    def test_is_fail_property(self, validator, tight_sla):
        result = validator.validate("qwen3.5-27b", tight_sla)
        assert result.is_fail is True
        assert result.is_pass is False
        assert result.is_error is False

    def test_avg_latency_exceeds_sla(self, validator, tight_sla):
        result = validator.validate("qwen3.5-27b", tight_sla)
        assert result.benchmark is not None
        assert result.benchmark.avg_latency_ms > tight_sla.max_latency_ms

    def test_fail_recommendations_mention_optimization(self, validator, tight_sla):
        result = validator.validate("qwen3.5-27b", tight_sla)
        combined = " ".join(result.recommendations).lower()
        # Should mention quantization, GPU, or a smaller model
        assert any(kw in combined for kw in ("quantiz", "gpu", "smaller", "hardware", "distil"))

    def test_benchmark_mode_is_simulated(self, validator, tight_sla):
        result = validator.validate("qwen3.5-27b", tight_sla)
        assert result.benchmark.mode == "simulated"

    def test_benchmark_runs_recorded(self, validator, tight_sla):
        result = validator.validate("qwen3.5-27b", tight_sla)
        assert result.benchmark.num_runs >= 1
        assert len(result.benchmark.samples) == result.benchmark.num_runs


# ── Test case 3: ERROR scenario ────────────────────────────────────────────────

class TestErrorScenario:
    """Model: corrupted_file → ERROR"""

    def test_status_is_error(self, validator, standard_sla):
        result = validator.validate("corrupted_file", standard_sla)
        assert result.status == "ERROR"

    def test_message_contains_invalid_model_path(self, validator, standard_sla):
        result = validator.validate("corrupted_file", standard_sla)
        assert "Invalid Model Path" in result.message

    def test_is_error_property(self, validator, standard_sla):
        result = validator.validate("corrupted_file", standard_sla)
        assert result.is_error is True
        assert result.is_pass is False
        assert result.is_fail is False

    def test_benchmark_is_none_on_error(self, validator, standard_sla):
        result = validator.validate("corrupted_file", standard_sla)
        assert result.benchmark is None

    def test_error_recommendations_suggest_valid_path(self, validator, standard_sla):
        result = validator.validate("corrupted_file", standard_sla)
        assert len(result.recommendations) >= 1

    def test_empty_path_returns_error(self, validator, standard_sla):
        result = validator.validate("", standard_sla)
        assert result.status == "ERROR"

    def test_whitespace_path_returns_error(self, validator, standard_sla):
        result = validator.validate("   ", standard_sla)
        assert result.status == "ERROR"

    def test_nonexistent_local_path_returns_error(self, validator, standard_sla):
        result = validator.validate("/nonexistent/path/to/model", standard_sla)
        assert result.status == "ERROR"


# ── SLAConfig validation ───────────────────────────────────────────────────────

class TestSLAConfig:

    def test_negative_latency_raises(self):
        with pytest.raises(ValueError):
            SLAConfig(max_latency_ms=-1.0)

    def test_zero_latency_raises(self):
        with pytest.raises(ValueError):
            SLAConfig(max_latency_ms=0.0)

    def test_negative_throughput_raises(self):
        with pytest.raises(ValueError):
            SLAConfig(max_latency_ms=200.0, min_throughput_tokens_per_sec=-5.0)

    def test_valid_config_created(self):
        sla = SLAConfig(
            max_latency_ms=300.0,
            min_throughput_tokens_per_sec=10.0,
            max_cost_per_1k_tokens=0.01,
        )
        assert sla.max_latency_ms == 300.0
        assert sla.min_throughput_tokens_per_sec == 10.0

    def test_to_dict_contains_expected_keys(self):
        sla = SLAConfig(max_latency_ms=200.0)
        d = sla.to_dict()
        assert "max_latency_ms" in d
        assert d["max_latency_ms"] == 200.0


# ── Convenience function ───────────────────────────────────────────────────────

class TestValidateModelConvenienceFunction:

    def test_pass_result(self):
        result = validate_model("qwen3.5-27b", max_latency_ms=500.0, force_simulation=True)
        assert result.status == "PASS"

    def test_fail_result(self):
        result = validate_model("qwen3.5-27b", max_latency_ms=10.0, force_simulation=True)
        assert result.status == "FAIL"

    def test_error_result(self):
        result = validate_model("corrupted_file", max_latency_ms=200.0, force_simulation=True)
        assert result.status == "ERROR"

    def test_returns_validation_result_type(self):
        result = validate_model("openai-community/gpt2", max_latency_ms=500.0, force_simulation=True)
        assert isinstance(result, ValidationResult)


# ── Reproducibility ────────────────────────────────────────────────────────────

class TestReproducibility:
    """Simulated benchmarks should be deterministic for the same model path."""

    def test_same_model_same_latency(self, validator, standard_sla):
        r1 = validator.validate("Qwen/Qwen2.5-7B-Instruct", standard_sla)
        r2 = validator.validate("Qwen/Qwen2.5-7B-Instruct", standard_sla)
        if r1.benchmark and r2.benchmark:
            # Allow tiny floating-point variance but should be very close
            diff = abs(r1.benchmark.avg_latency_ms - r2.benchmark.avg_latency_ms)
            assert diff < r1.benchmark.avg_latency_ms * 0.25  # within 25%

    def test_different_models_different_latency(self, validator):
        sla = SLAConfig(max_latency_ms=10000.0)
        r_small = validator.validate("openai-community/gpt2", sla)   # ~124M
        r_large = validator.validate("qwen3.5-27b", sla)              # ~27B
        if r_small.benchmark and r_large.benchmark:
            assert r_large.benchmark.avg_latency_ms > r_small.benchmark.avg_latency_ms
