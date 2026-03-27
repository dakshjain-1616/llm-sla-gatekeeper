#!/usr/bin/env python3
"""
04_full_pipeline.py — End-to-end deployment gate pipeline.

Demonstrates the full workflow:
  1. Define an SLA config (or use a profile)
  2. Validate a shortlist of candidate models
  3. Save results to history
  4. Load history and print a trend summary
  5. Export results to JSON and a Markdown table

Run:
    python examples/04_full_pipeline.py
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import tempfile
from pathlib import Path

os.environ.setdefault("SLA_SIMULATION_MODE", "1")

from llm_sla_gatekeeper import (
    SLAConfig,
    SLAValidator,
    append_result,
    load_history,
    history_summary,
    clear_history,
)

# ── 1. Setup ───────────────────────────────────────────────────────────────────
tmp_dir = Path(tempfile.mkdtemp())
history_file = tmp_dir / "pipeline_history.jsonl"
output_json  = tmp_dir / "pipeline_results.json"

sla = SLAConfig(
    max_latency_ms=200,
    min_throughput_tokens_per_sec=5.0,
)

candidates = [
    "openai-community/gpt2",
    "facebook/opt-125m",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "qwen3.5-27b",
]

# ── 2. Validate all candidates ─────────────────────────────────────────────────
print("=== Step 1 — Running SLA validation pipeline ===\n")
validator = SLAValidator(force_simulation=True, num_runs=5)
results = validator.validate_batch(candidates, sla)

for r in results:
    icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "⚠️ "}.get(r.status, "?")
    lat = f"{r.benchmark.avg_latency_ms:.1f} ms" if r.benchmark else "N/A"
    print(f"  {icon} {r.status:<5}  {r.model_path:<40}  {lat}")

# ── 3. Persist to history ─────────────────────────────────────────────────────
print("\n=== Step 2 — Saving to history ===")
for r in results:
    append_result(r.to_dict(), history_file=history_file)
print(f"  Saved {len(results)} records to {history_file}")

# ── 4. Load history and summarise ─────────────────────────────────────────────
print("\n=== Step 3 — History summary ===")
s = history_summary(history_file=history_file)
print(f"  Total runs : {s['total']}")
print(f"  PASS       : {s['pass_count']}")
print(f"  FAIL       : {s['fail_count']}")
print(f"  ERROR      : {s['error_count']}")
print(f"  Models seen: {', '.join(s['models_seen'])}")

# ── 5. Export to JSON ─────────────────────────────────────────────────────────
print("\n=== Step 4 — Exporting results to JSON ===")
records = [r.to_dict() for r in results]
output_json.write_text(json.dumps(records, indent=2))
print(f"  Written to {output_json}")

# ── 6. Print Markdown table ───────────────────────────────────────────────────
print("\n=== Step 5 — Markdown summary table ===\n")
print("| Model | Status | Avg Latency | Throughput | Confidence |")
print("|-------|--------|-------------|------------|------------|")
for r in results:
    bm = r.benchmark
    lat  = f"{bm.avg_latency_ms:.1f} ms"   if bm else "N/A"
    tps  = f"{bm.throughput_tokens_per_sec:.1f} tok/s" if bm else "N/A"
    conf = f"{r.confidence_score:.0%}"      if r.confidence_score is not None else "—"
    print(f"| `{r.model_path}` | **{r.status}** | {lat} | {tps} | {conf} |")

# ── 7. Gate decision ──────────────────────────────────────────────────────────
passing = [r for r in results if r.status == "PASS"]
print(f"\n=== Gate Decision ===")
print(f"  {len(passing)}/{len(results)} models passed the SLA.")
if passing:
    best = min(passing, key=lambda r: r.benchmark.avg_latency_ms)
    print(f"  Recommended: {best.model_path}  ({best.benchmark.avg_latency_ms:.1f} ms avg)")
else:
    print("  No models passed. Review SLA thresholds or hardware.")

# Cleanup
clear_history(history_file=history_file)
