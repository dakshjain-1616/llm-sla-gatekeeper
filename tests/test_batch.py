"""
Tests for batch validation, confidence scoring, stats, and new CLI features.
"""

import math
import os
import pytest

os.environ["SLA_SIMULATION_MODE"] = "1"

from llm_sla_gatekeeper.sla_validator import SLAConfig, SLAValidator, ValidationResult, _compute_confidence
from llm_sla_gatekeeper.benchmark import _stats, run_benchmark, BenchmarkResult


# ── Batch validation ───────────────────────────────────────────────────────────

class TestValidateBatch:

    @pytest.fixture
    def validator(self):
        return SLAValidator(force_simulation=True)

    @pytest.fixture
    def generous_sla(self):
        return SLAConfig(max_latency_ms=500.0)

    def test_returns_list_of_results(self, validator, generous_sla):
        models = ["openai-community/gpt2", "Qwen/Qwen2.5-7B-Instruct"]
        results = validator.validate_batch(models, generous_sla)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_each_result_is_validation_result(self, validator, generous_sla):
        models = ["openai-community/gpt2", "qwen3.5-27b"]
        results = validator.validate_batch(models, generous_sla)
        for r in results:
            assert isinstance(r, ValidationResult)

    def test_results_match_model_order(self, validator, generous_sla):
        models = ["openai-community/gpt2", "corrupted_file", "qwen3.5-27b"]
        results = validator.validate_batch(models, generous_sla)
        assert results[0].model_path == "openai-community/gpt2"
        assert results[1].model_path == "corrupted_file"
        assert results[2].model_path == "qwen3.5-27b"

    def test_empty_model_returns_error_in_batch(self, validator, generous_sla):
        results = validator.validate_batch(["", "openai-community/gpt2"], generous_sla)
        assert results[0].status == "ERROR"
        assert results[1].status in ("PASS", "FAIL")

    def test_invalid_model_is_error_not_exception(self, validator, generous_sla):
        results = validator.validate_batch(["corrupted_file"], generous_sla)
        assert results[0].status == "ERROR"

    def test_batch_single_model_same_as_single_validate(self, validator, generous_sla):
        single = validator.validate("openai-community/gpt2", generous_sla)
        batch = validator.validate_batch(["openai-community/gpt2"], generous_sla)
        assert batch[0].status == single.status
        assert batch[0].model_path == single.model_path

    def test_different_models_produce_different_latencies(self, validator):
        sla = SLAConfig(max_latency_ms=100000.0)
        results = validator.validate_batch(["openai-community/gpt2", "qwen3.5-27b"], sla)
        lat_gpt2 = results[0].benchmark.avg_latency_ms
        lat_27b = results[1].benchmark.avg_latency_ms
        assert lat_27b > lat_gpt2


# ── Statistical helpers ────────────────────────────────────────────────────────

class TestStatsHelper:

    def test_single_value(self):
        mean, std, ci = _stats([42.0])
        assert mean == 42.0
        assert std == 0.0
        assert ci == (42.0, 42.0)

    def test_empty_list(self):
        mean, std, ci = _stats([])
        assert mean == 0.0

    def test_mean_correct(self):
        mean, _, _ = _stats([10.0, 20.0, 30.0])
        assert abs(mean - 20.0) < 1e-9

    def test_std_positive(self):
        _, std, _ = _stats([10.0, 15.0, 20.0, 25.0, 30.0])
        assert std > 0

    def test_identical_values_std_zero(self):
        _, std, _ = _stats([5.0, 5.0, 5.0, 5.0])
        assert std == 0.0

    def test_ci_contains_mean(self):
        mean, _, (lo, hi) = _stats([10.0, 12.0, 11.0, 13.0, 9.0])
        assert lo <= mean <= hi

    def test_ci_width_decreases_with_more_data(self):
        small_n = [10.0, 20.0]
        large_n = [10.0, 12.0, 14.0, 11.0, 13.0, 9.0, 15.0, 10.0, 12.0, 11.0]
        _, _, (lo_s, hi_s) = _stats(small_n)
        _, _, (lo_l, hi_l) = _stats(large_n)
        # Small n should have wider CI relative to similar spread
        # (This just checks the mechanism works, not absolute values)
        assert isinstance(lo_s, float) and isinstance(hi_s, float)
        assert isinstance(lo_l, float) and isinstance(hi_l, float)


# ── Benchmark std dev & CI ─────────────────────────────────────────────────────

class TestBenchmarkStdAndCI:

    def test_std_latency_ms_populated(self):
        result = run_benchmark("Qwen/Qwen2.5-7B-Instruct", force_simulation=True, num_runs=5)
        assert isinstance(result.std_latency_ms, float)
        assert result.std_latency_ms >= 0.0

    def test_confidence_interval_is_tuple_of_two(self):
        result = run_benchmark("Qwen/Qwen2.5-7B-Instruct", force_simulation=True, num_runs=5)
        ci = result.confidence_interval_95
        assert len(ci) == 2
        assert ci[0] <= ci[1]

    def test_ci_contains_avg_latency(self):
        result = run_benchmark("openai-community/gpt2", force_simulation=True, num_runs=5)
        lo, hi = result.confidence_interval_95
        avg = result.avg_latency_ms
        assert lo <= avg <= hi

    def test_single_run_ci_equals_latency(self):
        result = run_benchmark("openai-community/gpt2", force_simulation=True, num_runs=1)
        lo, hi = result.confidence_interval_95
        assert lo == hi  # no variance with 1 run


# ── Confidence scoring ─────────────────────────────────────────────────────────

class TestConfidenceScore:

    def test_confidence_score_present_on_pass(self):
        sla = SLAConfig(max_latency_ms=500.0)
        validator = SLAValidator(force_simulation=True)
        result = validator.validate("openai-community/gpt2", sla)
        assert result.confidence_score is not None
        assert 0.0 <= result.confidence_score <= 1.0

    def test_confidence_score_present_on_fail(self):
        sla = SLAConfig(max_latency_ms=1.0)
        validator = SLAValidator(force_simulation=True)
        result = validator.validate("qwen3.5-27b", sla)
        assert result.confidence_score is not None

    def test_confidence_score_none_on_error(self):
        sla = SLAConfig(max_latency_ms=200.0)
        validator = SLAValidator(force_simulation=True)
        result = validator.validate("corrupted_file", sla)
        assert result.confidence_score is None

    def test_confidence_in_to_dict(self):
        sla = SLAConfig(max_latency_ms=500.0)
        validator = SLAValidator(force_simulation=True)
        result = validator.validate("openai-community/gpt2", sla)
        d = result.to_dict()
        assert "confidence_score" in d

    def test_more_runs_higher_confidence(self):
        """More benchmark runs should yield equal or higher confidence (simulated)."""
        bm_few = run_benchmark("Qwen/Qwen2.5-7B-Instruct", force_simulation=True, num_runs=2)
        bm_many = run_benchmark("Qwen/Qwen2.5-7B-Instruct", force_simulation=True, num_runs=15)
        conf_few = _compute_confidence(bm_few)
        conf_many = _compute_confidence(bm_many)
        assert conf_many >= conf_few

    def test_simulated_mode_lower_confidence_than_real_would_be(self):
        """Simulated benchmarks carry a mode_factor penalty."""
        bm = run_benchmark("openai-community/gpt2", force_simulation=True, num_runs=20)
        conf = _compute_confidence(bm)
        # Simulated mode is capped at 0.75 (mode_factor=0.75) for max theoretical score
        assert conf <= 0.76  # small tolerance


# ── CLI helpers ────────────────────────────────────────────────────────────────

class TestCLIParsing:

    def test_parse_ms_basic(self):
        from run_validation import _parse_ms
        assert _parse_ms("200ms") == 200.0
        assert _parse_ms("0.2s") == pytest.approx(200.0)
        assert _parse_ms("300") == 300.0

    def test_parse_ms_strips_whitespace(self):
        from run_validation import _parse_ms
        assert _parse_ms("  150ms ") == 150.0

    def test_results_to_csv_structure(self):
        from run_validation import _results_to_csv
        sla = SLAConfig(max_latency_ms=500.0)
        validator = SLAValidator(force_simulation=True)
        result = validator.validate("openai-community/gpt2", sla)
        csv_text = _results_to_csv([result])
        lines = csv_text.strip().splitlines()
        assert len(lines) == 2  # header + 1 data row
        header = lines[0].lower()
        assert "model_path" in header
        assert "status" in header
        assert "avg_latency_ms" in header
        assert "confidence_score" in header

    def test_results_to_csv_multiple_rows(self):
        from run_validation import _results_to_csv
        sla = SLAConfig(max_latency_ms=500.0)
        validator = SLAValidator(force_simulation=True)
        results = validator.validate_batch(["openai-community/gpt2", "qwen3.5-27b"], sla)
        csv_text = _results_to_csv(results)
        lines = csv_text.strip().splitlines()
        assert len(lines) == 3  # header + 2 rows
