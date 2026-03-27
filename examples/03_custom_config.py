#!/usr/bin/env python3
"""
03_custom_config.py — Customising behaviour via env vars and profiles.

Demonstrates:
  - Using built-in SLA profiles (chatbot, realtime, batch, edge, dev)
  - Overriding profile thresholds via environment variables
  - Listing all available profiles

Run:
    python examples/03_custom_config.py

Override example:
    SLA_PROFILE_CHATBOT_LATENCY_MS=200 python examples/03_custom_config.py
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("SLA_SIMULATION_MODE", "1")

from llm_sla_gatekeeper import (
    get_profile,
    list_profiles,
    profile_to_sla_config,
    SLAValidator,
)

# ── 1. List all profiles ───────────────────────────────────────────────────────
print("=== Available SLA Profiles ===\n")
for key, name, desc in list_profiles():
    p = get_profile(key)
    lat = p["max_latency_ms"]
    tps = p["min_throughput_tokens_per_sec"]
    tps_str = f"{tps:.0f} tok/s" if tps else "—"
    print(f"  {key:<10} {name:<25} {lat:>6.0f} ms   {tps_str}")

# ── 2. Validate GPT-2 against each profile ────────────────────────────────────
print("\n=== GPT-2 vs Each Profile ===\n")
validator = SLAValidator(force_simulation=True)

for key, name, _ in list_profiles():
    sla = profile_to_sla_config(key)
    result = validator.validate("openai-community/gpt2", sla)
    print(f"  [{key:<10}] {result.status}  —  {result.message}")

# ── 3. Show active chatbot profile values (env-var override demo) ──────────────
print("\n=== Active chatbot profile thresholds ===")
chatbot = get_profile("chatbot")
print(f"  max_latency_ms             : {chatbot['max_latency_ms']}")
print(f"  min_throughput_tokens_per_sec: {chatbot['min_throughput_tokens_per_sec']}")
print()
print("  Tip: set SLA_PROFILE_CHATBOT_LATENCY_MS=200 to override the latency threshold.")
