#!/usr/bin/env python3
"""
01_quick_start.py — Minimal working example.

Validates a single model against a latency SLA in simulation mode.
No model downloads or GPU required.

Run:
    python examples/01_quick_start.py
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("SLA_SIMULATION_MODE", "1")

from llm_sla_gatekeeper import validate_model

result = validate_model(
    "Qwen/Qwen3-8B",
    max_latency_ms=200,
    force_simulation=True,
)

print(f"Status : {result.status}")
print(f"Message: {result.message}")
print(f"Latency: {result.benchmark.avg_latency_ms:.1f} ms/tok")
print(f"Confidence: {result.confidence_score:.0%}")
