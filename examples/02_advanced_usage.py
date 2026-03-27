#!/usr/bin/env python3
"""
02_advanced_usage.py — Advanced features.

Demonstrates:
  - SLAConfig with multiple thresholds (latency + throughput + cost)
  - Batch validation of several models
  - Inspecting per-run samples and confidence interval
  - SLAValidator with custom run count

Run:
    python examples/02_advanced_usage.py
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("SLA_SIMULATION_MODE", "1")

from llm_sla_gatekeeper import SLAConfig, SLAValidator

# ── 1. Multi-threshold SLA ─────────────────────────────────────────────────────
sla = SLAConfig(
    max_latency_ms=300,
    min_throughput_tokens_per_sec=5.0,
    p95_latency_multiplier=1.5,
)

validator = SLAValidator(force_simulation=True, num_runs=10)

models = [
    "openai-community/gpt2",        # ~124M — fast
    "Qwen/Qwen2.5-7B-Instruct",     # ~7B  — medium
    "qwen3.5-27b",                  # ~27B — slow on CPU
]

print("=== Batch Validation (300 ms SLA, ≥5 tok/s) ===\n")
results = validator.validate_batch(models, sla)

for r in results:
    bm = r.benchmark
    if bm:
        ci_lo, ci_hi = bm.confidence_interval_95
        print(
            f"  {r.model_path:<40} {r.status:<5}"
            f"  avg={bm.avg_latency_ms:6.1f} ms"
            f"  tps={bm.throughput_tokens_per_sec:6.1f}"
            f"  95%CI=[{ci_lo:.1f},{ci_hi:.1f}]"
            f"  conf={r.confidence_score:.0%}"
        )
    else:
        print(f"  {r.model_path:<40} {r.status:<5}  {r.message}")

# ── 2. Inspect per-run samples ─────────────────────────────────────────────────
print("\n=== Per-run samples for GPT-2 ===")
r_gpt2 = validator.validate("openai-community/gpt2", sla)
for s in r_gpt2.benchmark.samples[:3]:
    print(f"  run {s.run_index}: {s.elapsed_ms:.1f} ms total, {s.tokens_per_second:.1f} tok/s")

# ── 3. Recommendations ────────────────────────────────────────────────────────
print("\n=== Recommendations for 27B model ===")
r_27b = results[2]
for rec in r_27b.recommendations:
    print(f"  • {rec}")
