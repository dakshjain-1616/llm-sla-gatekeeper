#!/usr/bin/env python3
"""Quick smoke test — verifies all new features work end-to-end without pytest."""
import os
import tempfile
import pathlib

os.environ["SLA_SIMULATION_MODE"] = "1"

from llm_sla_gatekeeper.benchmark import run_benchmark, _stats, BenchmarkResult
from llm_sla_gatekeeper.sla_validator import SLAConfig, SLAValidator, ValidationResult, _compute_confidence
from llm_sla_gatekeeper.sla_profiles import get_profile, list_profiles, profile_to_sla_config, PROFILES
from llm_sla_gatekeeper.history import append_result, load_history, history_for_model, clear_history, history_summary

print("All imports OK")

mean, std, ci = _stats([10.0, 20.0, 15.0, 12.0, 18.0])
assert 10.0 < mean < 20.0
assert std > 0
assert ci[0] <= mean <= ci[1]
print("_stats OK")

bm = run_benchmark("openai-community/gpt2", force_simulation=True, num_runs=5)
assert bm.std_latency_ms >= 0
assert len(bm.confidence_interval_95) == 2
assert bm.confidence_interval_95[0] <= bm.avg_latency_ms <= bm.confidence_interval_95[1]
print(f"benchmark OK: avg={bm.avg_latency_ms:.2f}ms std={bm.std_latency_ms:.3f}ms ci={bm.confidence_interval_95}")

sla_chatbot = profile_to_sla_config("chatbot")
assert sla_chatbot.max_latency_ms == 150.0
assert sla_chatbot.min_throughput_tokens_per_sec == 10.0
profiles_list = list_profiles()
assert len(profiles_list) == 5
print(f"profiles OK: {[p[0] for p in profiles_list]}")

try:
    get_profile("bogus")
    assert False, "Should have raised"
except KeyError:
    pass
print("profile KeyError OK")

validator = SLAValidator(force_simulation=True)
result = validator.validate("openai-community/gpt2", SLAConfig(max_latency_ms=500.0))
assert result.status == "PASS"
assert result.confidence_score is not None
assert 0.0 <= result.confidence_score <= 1.0
d = result.to_dict()
assert "confidence_score" in d
assert "std_latency_ms" in d["benchmark"]
assert "confidence_interval_95" in d["benchmark"]
print(f"single validate OK: {result.status}, confidence={result.confidence_score:.3f}")

result_fail = validator.validate("qwen3.5-27b", SLAConfig(max_latency_ms=1.0))
assert result_fail.status == "FAIL"
print("fail scenario OK")

results = validator.validate_batch(
    ["openai-community/gpt2", "qwen3.5-27b", "corrupted_file"],
    SLAConfig(max_latency_ms=500.0)
)
assert len(results) == 3
assert results[0].model_path == "openai-community/gpt2"
assert results[2].status == "ERROR"
print(f"batch validate OK: {[r.status for r in results]}")

with tempfile.TemporaryDirectory() as tmpdir:
    hfile = pathlib.Path(tmpdir) / "test.jsonl"
    append_result(result.to_dict(), history_file=hfile)
    records = load_history(history_file=hfile)
    assert len(records) == 1
    assert records[0]["status"] == "PASS"
    s = history_summary(history_file=hfile)
    assert s["total"] == 1
    assert s["pass_count"] == 1
    n = clear_history(history_file=hfile)
    assert n == 1
    assert not hfile.exists()
print("history round-trip OK")

bm_few = run_benchmark("Qwen/Qwen2.5-7B-Instruct", force_simulation=True, num_runs=2)
bm_many = run_benchmark("Qwen/Qwen2.5-7B-Instruct", force_simulation=True, num_runs=15)
conf_few = _compute_confidence(bm_few)
conf_many = _compute_confidence(bm_many)
assert conf_many >= conf_few
print(f"confidence scaling OK: 2 runs={conf_few:.3f}, 15 runs={conf_many:.3f}")

from run_validation import _parse_ms, _results_to_csv
assert _parse_ms("200ms") == 200.0
assert abs(_parse_ms("0.2s") - 200.0) < 0.001
assert _parse_ms("300") == 300.0
csv_text = _results_to_csv([result])
lines = csv_text.strip().splitlines()
assert len(lines) == 2
assert "confidence_score" in lines[0].lower()
print("CLI helpers OK")

print()
print("=" * 50)
print("ALL SMOKE TESTS PASSED")
print("=" * 50)
