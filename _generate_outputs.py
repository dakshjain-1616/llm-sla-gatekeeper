#!/usr/bin/env python3
"""
Internal helper: generates the outputs/ files directly using only stdlib.
This is used when external deps (psutil, torch) are not yet installed.
Produces the exact same output structure as demo.py.
"""

import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

VALID_MODEL_KEYWORDS = [
    "gpt", "llama", "qwen", "bert", "opt", "falcon", "mistral",
    "phi", "gemma", "vicuna", "alpaca", "bloom", "t5", "bart",
    "roberta", "electra", "mpt", "codegen", "starcoder", "deepseek",
    "yi", "internlm", "baichuan", "chatglm", "rwkv", "mamba",
]


def _looks_valid(model_path):
    if not model_path:
        return False
    p = model_path.lower()
    if any(kw in p for kw in VALID_MODEL_KEYWORDS):
        return True
    if re.search(r"\d+(?:\.\d+)?[bBmM]", p):
        return True
    if "/" in model_path:
        parts = model_path.split("/")
        if len(parts) == 2 and all(x.strip() for x in parts):
            return True
    return False


def _estimate_size_b(model_path):
    p = model_path.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*([bm])", p)
    if m:
        v, u = float(m.group(1)), m.group(2)
        return v / 1000.0 if u == "m" else v
    if "large" in p: return 7.0
    if "xl" in p: return 13.0
    if "small" in p: return 0.5
    if "base" in p: return 0.1
    return 3.0


def _sim_latency(model_path):
    size_b = _estimate_size_b(model_path)
    base_ms = 5.0 * size_b + 15.0
    seed = int(hashlib.md5(model_path.encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(seed)
    jitter = rng.uniform(-base_ms * 0.08, base_ms * 0.08)
    return max(1.0, base_ms + jitter)


def run_scenario(model_path, max_latency_ms, min_throughput=None, num_runs=5, num_tokens=50):
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    sla_cfg = {
        "max_latency_ms": max_latency_ms,
        "min_throughput_tokens_per_sec": min_throughput,
        "max_cost_per_1k_tokens": None,
        "cost_per_token_usd": 0.0,
        "p95_latency_multiplier": 1.5,
    }

    if not model_path.strip():
        return {
            "status": "ERROR",
            "message": "ERROR - Invalid Model Path",
            "detail": "model_path must not be empty",
            "model_path": model_path,
            "sla_config": sla_cfg,
            "benchmark": None,
            "recommendations": ["Provide a valid HuggingFace model ID or local path."],
            "timestamp": now,
        }

    if not _looks_valid(model_path):
        return {
            "status": "ERROR",
            "message": "ERROR - Invalid Model Path",
            "detail": f"Model path '{model_path}' does not match any known LLM identifier pattern.",
            "model_path": model_path,
            "sla_config": sla_cfg,
            "benchmark": None,
            "recommendations": [
                "Verify the model ID or path exists and is accessible.",
                "For HuggingFace models ensure you have network access or the model is cached.",
                "Check spelling: valid examples are 'Qwen/Qwen2.5-7B-Instruct' or 'openai-community/gpt2'.",
            ],
            "timestamp": now,
        }

    base_lat = _sim_latency(model_path)
    seed2 = hash(model_path) & 0xFFFFFFFF
    rng2 = random.Random(seed2)

    latencies = []
    for _ in range(num_runs):
        jitter = rng2.gauss(0, base_lat * 0.05)
        latencies.append(max(0.5, base_lat + jitter))

    latencies_s = sorted(latencies)
    n = len(latencies_s)
    avg_lat = sum(latencies) / n
    p50 = latencies_s[n // 2]
    p95 = latencies_s[int(n * 0.95)] if n > 1 else latencies_s[-1]
    tps = (num_tokens * 1000.0) / avg_lat

    benchmark = {
        "model_path": model_path,
        "mode": "simulated",
        "avg_latency_ms": round(avg_lat, 2),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "throughput_tokens_per_sec": round(tps, 2),
        "total_tokens": num_tokens * num_runs,
        "benchmark_duration_sec": round(num_runs * 0.05, 3),
        "num_runs": num_runs,
        "hardware": {
            "device": "cpu",
            "platform": "Linux",
            "ram_total_gb": 0.0,
            "ram_available_gb": 0.0,
        },
    }

    latency_ok = avg_lat <= max_latency_ms
    p95_limit = max_latency_ms * 1.5
    p95_ok = p95 <= p95_limit
    tput_ok = True
    if min_throughput:
        tput_ok = tps >= min_throughput

    all_ok = latency_ok and p95_ok and tput_ok

    if all_ok:
        status = "PASS"
        message = (
            f"PASS - Ready for Deployment "
            f"(avg {avg_lat:.1f} ms/tok ≤ {max_latency_ms:.0f} ms SLA)"
        )
        recs = ["All SLA thresholds met. Model is ready for production deployment."]
        if avg_lat < max_latency_ms * 0.5:
            recs.append(
                "Latency headroom is >50%. You may be over-provisioning; "
                "consider scaling down compute to reduce costs."
            )
    else:
        failures = []
        recs = []
        if not latency_ok:
            gap = avg_lat - max_latency_ms
            pct = (gap / max_latency_ms) * 100
            failures.append(
                f"avg latency {avg_lat:.1f} ms exceeds {max_latency_ms:.0f} ms target"
            )
            recs.append(
                f"Latency is {pct:.0f}% above threshold. "
                "Consider quantizing the model (INT8/INT4) to reduce per-token latency."
            )
            recs.append(
                "Evaluate deploying on a GPU instance (e.g., T4 or A10G) "
                "to achieve sub-100 ms per-token latency."
            )
            if avg_lat > max_latency_ms * 3:
                recs.append(
                    "Latency is severely over target. Consider a smaller/distilled model "
                    "variant for this SLA tier."
                )
        if not tput_ok and min_throughput:
            gap2 = min_throughput - tps
            failures.append(f"throughput {tps:.1f} tok/s below {min_throughput:.0f} tok/s minimum")
            recs.append(
                f"Throughput is {gap2:.1f} tok/s below the required "
                f"{min_throughput:.0f} tok/s minimum. "
                "Enable continuous batching or increase batch size."
            )

        detail = "; ".join(failures)
        status = "FAIL"
        message = (
            f"FAIL - Exceeds {max_latency_ms:.0f}ms target on Current Hardware ({detail})"
        )

    return {
        "status": status,
        "message": message,
        "detail": None if all_ok else f"avg_ms={avg_lat:.1f}; p95_ms={p95:.1f}; tps={tps:.1f}",
        "model_path": model_path,
        "sla_config": sla_cfg,
        "benchmark": benchmark,
        "recommendations": recs,
        "timestamp": now,
    }


SCENARIOS = [
    {"description": "Qwen3-8B against generous 500ms SLA  → PASS",
     "model": "Qwen/Qwen3-8B", "max_latency_ms": 500.0, "min_throughput": None},
    {"description": "Qwen3-8B against tight 10ms SLA  → FAIL",
     "model": "Qwen/Qwen3-8B", "max_latency_ms": 10.0, "min_throughput": None},
    {"description": "Invalid / corrupted model path  → ERROR",
     "model": "corrupted_file", "max_latency_ms": 200.0, "min_throughput": None},
    {"description": "GPT-2 (124M) against chatbot SLA (150ms / 10 tok/s)  → PASS",
     "model": "openai-community/gpt2", "max_latency_ms": 150.0, "min_throughput": 10.0},
    {"description": "Qwen2.5-7B against 200ms SLA + 0.5 tok/s  → scenario varies",
     "model": "Qwen/Qwen2.5-7B-Instruct", "max_latency_ms": 200.0, "min_throughput": 0.5},
    {"description": "Qwen3-1.7B edge device scenario (500ms / 2 tok/s)  → PASS",
     "model": "Qwen/Qwen3-1.7B", "max_latency_ms": 500.0, "min_throughput": 2.0},
    {"description": "OPT-125M real-time API scenario (50ms / 50 tok/s)  → PASS",
     "model": "facebook/opt-125m", "max_latency_ms": 50.0, "min_throughput": 50.0},
]

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>LLM SLA Gatekeeper — Validation Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background:#f8fafc; color:#1e293b; margin:0; padding:24px; }}
    h1   {{ color:#4f46e5; margin-bottom:4px; }}
    .sub {{ color:#64748b; margin-bottom:24px; font-size:0.9rem; }}
    .card {{ background:#fff; border-radius:10px; box-shadow:0 1px 3px rgba(0,0,0,.1);
             padding:20px 24px; margin-bottom:18px; }}
    .badge {{ display:inline-block; padding:4px 14px; border-radius:20px;
              font-weight:700; font-size:0.85rem; }}
    .PASS  {{ background:#dcfce7; color:#16a34a; }}
    .FAIL  {{ background:#fee2e2; color:#dc2626; }}
    .ERROR {{ background:#fef3c7; color:#d97706; }}
    table {{ border-collapse:collapse; width:100%; font-size:0.9rem; }}
    th    {{ background:#f1f5f9; text-align:left; padding:8px 12px; color:#475569; }}
    td    {{ padding:7px 12px; border-bottom:1px solid #f1f5f9; }}
    ul    {{ margin:0; padding-left:18px; }}
    li    {{ margin:5px 0; line-height:1.5; }}
    .verdict {{ font-size:1.05rem; font-weight:600; margin:8px 0; }}
  </style>
</head>
<body>
  <h1>&#x1F6A6; LLM SLA Gatekeeper</h1>
  <div class="sub">Validation Report &middot; Generated {timestamp}</div>
  {cards}
</body>
</html>
"""

CARD_TEMPLATE = """\
<div class="card">
  <span class="badge {status}">{status}</span>
  <p class="verdict">{message}</p>
  <p><strong>Model:</strong> {model_path}</p>
  {benchmark_table}
  <h4 style="margin:14px 0 6px;">Recommendations</h4>
  <ul>{rec_items}</ul>
</div>
"""


def _bm_table(bm):
    if not bm:
        return "<p style='color:#9ca3af;'>No benchmark data.</p>"
    rows = [
        ("Avg latency", f"{bm['avg_latency_ms']:.2f} ms/tok"),
        ("P50 latency", f"{bm['p50_latency_ms']:.2f} ms/tok"),
        ("P95 latency", f"{bm['p95_latency_ms']:.2f} ms/tok"),
        ("Throughput", f"{bm['throughput_tokens_per_sec']:.2f} tok/s"),
        ("Total tokens", str(bm["total_tokens"])),
        ("Duration", f"{bm['benchmark_duration_sec']:.3f} s"),
        ("Mode", bm["mode"].upper()),
        ("Device", bm["hardware"].get("device", "cpu").upper()),
    ]
    trs = "".join(
        f"<tr><td><strong>{k}</strong></td><td style='font-family:monospace;'>{v}</td></tr>"
        for k, v in rows
    )
    return f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{trs}</tbody></table>"


def _build_card(r):
    rec_items = "".join(f"<li>{x}</li>" for x in r.get("recommendations", []))
    return CARD_TEMPLATE.format(
        status=r["status"],
        message=r["message"],
        model_path=r["model_path"],
        benchmark_table=_bm_table(r.get("benchmark")),
        rec_items=rec_items or "<li>No recommendations.</li>",
    )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  🚦 LLM SLA Gatekeeper — Output Generator")
    print("=" * 70 + "\n")

    results = []
    for i, sc in enumerate(SCENARIOS, 1):
        print(f"[{i}/{len(SCENARIOS)}] {sc['description']}")
        r = run_scenario(sc["model"], sc["max_latency_ms"], sc["min_throughput"])
        sym = {"PASS": "✅", "FAIL": "❌", "ERROR": "⚠️ "}.get(r["status"], "?")
        print(f"       {sym} {r['message']}\n")
        results.append(r)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # 1. results.json
    jp = OUTPUTS_DIR / "results.json"
    jp.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "simulation_mode": True,
        "scenarios": results,
    }, indent=2))
    print(f"📄 Saved JSON results  → {jp}")

    # 2. report.html
    cards = "\n".join(_build_card(r) for r in results)
    html = HTML_TEMPLATE.format(timestamp=now_str, cards=cards)
    hp = OUTPUTS_DIR / "report.html"
    hp.write_text(html)
    print(f"🌐 Saved HTML report   → {hp}")

    # 3. summary.md
    lines = [
        "# LLM SLA Gatekeeper — Validation Summary\n",
        f"**Generated:** {now_str}  \n",
        "**Mode:** Simulation  \n\n",
        "| Model | SLA (ms) | Status | Avg Latency | Throughput |",
        "|-------|----------|--------|-------------|------------|",
    ]
    for r in results:
        bm = r.get("benchmark") or {}
        lines.append(
            f"| `{r['model_path']}` | {r['sla_config']['max_latency_ms']} "
            f"| **{r['status']}** "
            f"| {bm.get('avg_latency_ms', 'N/A')} ms "
            f"| {bm.get('throughput_tokens_per_sec', 'N/A')} tok/s |"
        )
    mp = OUTPUTS_DIR / "summary.md"
    mp.write_text("\n".join(lines) + "\n")
    print(f"📝 Saved summary       → {mp}")

    print(f"\n{'=' * 70}\n  Done. Files in: {OUTPUTS_DIR.resolve()}\n{'=' * 70}\n")
