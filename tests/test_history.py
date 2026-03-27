"""
Tests for history.py — validation history persistence.
"""

import json
import os
import pytest
from pathlib import Path

os.environ["SLA_SIMULATION_MODE"] = "1"


@pytest.fixture
def tmp_history_file(tmp_path):
    return tmp_path / "history.jsonl"


@pytest.fixture
def sample_record():
    return {
        "status": "PASS",
        "message": "PASS - Ready for Deployment",
        "model_path": "Qwen/Qwen3-8B",
        "timestamp": "2026-03-27T10:00:00.000Z",
        "sla_config": {"max_latency_ms": 500.0},
        "benchmark": {
            "avg_latency_ms": 55.0,
            "throughput_tokens_per_sec": 18.2,
        },
        "confidence_score": 0.72,
    }


class TestAppendResult:

    def test_creates_file_on_first_write(self, tmp_history_file, sample_record):
        from llm_sla_gatekeeper.history import append_result
        assert not tmp_history_file.exists()
        append_result(sample_record, history_file=tmp_history_file)
        assert tmp_history_file.exists()

    def test_file_contains_json_line(self, tmp_history_file, sample_record):
        from llm_sla_gatekeeper.history import append_result
        append_result(sample_record, history_file=tmp_history_file)
        lines = tmp_history_file.read_text().strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["status"] == "PASS"
        assert parsed["model_path"] == "Qwen/Qwen3-8B"

    def test_multiple_appends_produce_multiple_lines(self, tmp_history_file, sample_record):
        from llm_sla_gatekeeper.history import append_result
        for _ in range(5):
            append_result(sample_record, history_file=tmp_history_file)
        lines = tmp_history_file.read_text().strip().splitlines()
        assert len(lines) == 5

    def test_creates_parent_directories(self, tmp_path, sample_record):
        from llm_sla_gatekeeper.history import append_result
        deep_path = tmp_path / "a" / "b" / "c" / "history.jsonl"
        append_result(sample_record, history_file=deep_path)
        assert deep_path.exists()

    def test_write_error_does_not_raise(self, sample_record):
        """append_result should silently swallow OS errors."""
        from llm_sla_gatekeeper.history import append_result
        # Pass a path inside a non-existent root that can't be created
        bad_path = Path("/nonexistent_root_xyz/history.jsonl")
        # Should not raise
        append_result(sample_record, history_file=bad_path)


class TestLoadHistory:

    def test_returns_empty_list_when_no_file(self, tmp_history_file):
        from llm_sla_gatekeeper.history import load_history
        result = load_history(history_file=tmp_history_file)
        assert result == []

    def test_loads_written_records(self, tmp_history_file, sample_record):
        from llm_sla_gatekeeper.history import append_result, load_history
        for _ in range(3):
            append_result(sample_record, history_file=tmp_history_file)
        records = load_history(history_file=tmp_history_file)
        assert len(records) == 3

    def test_limit_respected(self, tmp_history_file, sample_record):
        from llm_sla_gatekeeper.history import append_result, load_history
        for _ in range(10):
            append_result(sample_record, history_file=tmp_history_file)
        records = load_history(history_file=tmp_history_file, limit=3)
        assert len(records) == 3

    def test_limit_zero_loads_all(self, tmp_history_file, sample_record):
        from llm_sla_gatekeeper.history import append_result, load_history
        for _ in range(15):
            append_result(sample_record, history_file=tmp_history_file)
        records = load_history(history_file=tmp_history_file, limit=0)
        assert len(records) == 15

    def test_skips_malformed_lines(self, tmp_history_file, sample_record):
        from llm_sla_gatekeeper.history import append_result, load_history
        tmp_history_file.parent.mkdir(parents=True, exist_ok=True)
        with tmp_history_file.open("w") as f:
            f.write('{"status":"PASS","model_path":"gpt2"}\n')
            f.write("not json at all\n")
            f.write('{"status":"FAIL","model_path":"llama"}\n')
        records = load_history(history_file=tmp_history_file)
        assert len(records) == 2
        assert records[0]["status"] == "PASS"
        assert records[1]["status"] == "FAIL"

    def test_records_are_dicts(self, tmp_history_file, sample_record):
        from llm_sla_gatekeeper.history import append_result, load_history
        append_result(sample_record, history_file=tmp_history_file)
        records = load_history(history_file=tmp_history_file)
        assert isinstance(records[0], dict)


class TestHistoryForModel:

    def test_filters_by_model_path(self, tmp_history_file):
        from llm_sla_gatekeeper.history import append_result, history_for_model
        for model in ["gpt2", "llama-7b", "gpt2", "qwen-7b", "gpt2"]:
            record = {"status": "PASS", "model_path": model, "timestamp": "2026-01-01T00:00:00Z"}
            append_result(record, history_file=tmp_history_file)

        gpt2_records = history_for_model("gpt2", history_file=tmp_history_file)
        assert len(gpt2_records) == 3
        assert all(r["model_path"] == "gpt2" for r in gpt2_records)

    def test_returns_empty_for_unseen_model(self, tmp_history_file, sample_record):
        from llm_sla_gatekeeper.history import append_result, history_for_model
        append_result(sample_record, history_file=tmp_history_file)
        result = history_for_model("nonexistent-model", history_file=tmp_history_file)
        assert result == []


class TestHistorySummary:

    def test_counts_statuses(self, tmp_history_file):
        from llm_sla_gatekeeper.history import append_result, history_summary
        for status in ["PASS", "PASS", "FAIL", "ERROR", "PASS"]:
            record = {"status": status, "model_path": "some-model", "timestamp": "2026-01-01T00:00:00Z"}
            append_result(record, history_file=tmp_history_file)
        s = history_summary(history_file=tmp_history_file)
        assert s["total"] == 5
        assert s["pass_count"] == 3
        assert s["fail_count"] == 1
        assert s["error_count"] == 1

    def test_models_seen_unique(self, tmp_history_file):
        from llm_sla_gatekeeper.history import append_result, history_summary
        for model in ["gpt2", "llama", "gpt2", "qwen"]:
            record = {"status": "PASS", "model_path": model, "timestamp": "2026-01-01T00:00:00Z"}
            append_result(record, history_file=tmp_history_file)
        s = history_summary(history_file=tmp_history_file)
        assert len(s["models_seen"]) == 3  # gpt2, llama, qwen

    def test_empty_file_returns_zeros(self, tmp_history_file):
        from llm_sla_gatekeeper.history import history_summary
        s = history_summary(history_file=tmp_history_file)
        assert s["total"] == 0
        assert s["pass_count"] == 0


class TestClearHistory:

    def test_clears_existing_file(self, tmp_history_file, sample_record):
        from llm_sla_gatekeeper.history import append_result, clear_history, load_history
        for _ in range(5):
            append_result(sample_record, history_file=tmp_history_file)
        n = clear_history(history_file=tmp_history_file)
        assert n == 5
        assert not tmp_history_file.exists()

    def test_returns_zero_for_nonexistent_file(self, tmp_history_file):
        from llm_sla_gatekeeper.history import clear_history
        n = clear_history(history_file=tmp_history_file)
        assert n == 0


class TestHistoryIntegration:
    """Full round-trip: validate → history → reload."""

    def test_validator_results_round_trip_through_history(self, tmp_history_file):
        from llm_sla_gatekeeper.sla_validator import SLAConfig, SLAValidator
        from llm_sla_gatekeeper.history import append_result, load_history

        sla = SLAConfig(max_latency_ms=500.0)
        validator = SLAValidator(force_simulation=True)
        result = validator.validate("Qwen/Qwen3-8B", sla)
        append_result(result.to_dict(), history_file=tmp_history_file)

        records = load_history(history_file=tmp_history_file)
        assert len(records) == 1
        assert records[0]["status"] == result.status
        assert records[0]["model_path"] == "Qwen/Qwen3-8B"
        assert "confidence_score" in records[0]
