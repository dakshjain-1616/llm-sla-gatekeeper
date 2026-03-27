#!/usr/bin/env python3
"""
Demo script for LLM SLA Gatekeeper.

Runs representative validation scenarios (PASS / FAIL / ERROR / profile-based),
prints results to stdout, and saves structured output files to outputs/.

No API keys or model downloads required — runs entirely in simulation mode.

Usage:
    python demo.py
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

# Force simulation mode so the demo works without any model downloads
os.environ.setdefault("SLA_SIMULATION_MODE", "1")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "WARNING"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

OUTPUTS_DIR = Path(os.getenv("SLA_OUTPUTS_DIR", "outputs"))
OUTPUTS_DIR.mkdir(exist_ok=True)


# ── HTML Report Template ───────────────────────────────────────────────────────

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
    .confidence {{ font-size:0.8rem; color:#6b7280; margin-top:4px; }}
  </style>
</head>
<body>
  <h1>🚦 LLM SLA Gatekeeper</h1>
  <div class="sub">Validation Report · Generated {timestamp}</div>
  {cards}
</body>
</html>
"""

CARD_TEMPLATE = """\
<div class="card">
  <span class="badge {status}">{status}</span>
  <p class="verdict">{message}</p>
  <p><strong>Model:</strong> {model_path}</p>
  {confidence_line}
  {benchmark_table}
  <h4 style="margin:14px 0 6px;">Recommendations</h4>
  <ul>{rec_items}</ul>
</div>
"""


def _benchmark_table(bm) -> str:
    if not bm:
        return "<p style='color:#9ca3af;'>No benchmark data.</p>"
    ci = bm.get("confidence_interval_95", [0, 0])
    rows = [
        ("Avg latency", f"{bm['avg_latency_ms']:.2f} ms/tok"),
        ("Std dev", f"± {bm.get('std_latency_ms', 0):.2f} ms"),
        ("95% CI", f"[{ci[0]:.2f}, {ci[1]:.2f}] ms"),
        ("P50 latency", f"{bm['p50_latency_ms']:.2f} ms/tok"),
        ("P95 latency", f"{bm['p95_latency_ms']:.2f} ms/tok"),
        ("Throughput", f"{bm['throughput_tokens_per_sec']:.2f} tok/s"),
        ("Total tokens", str(bm["total_tokens"])),
        ("Runs", str(bm.get("num_runs", "—"))),
        ("Duration", f"{bm['benchmark_duration_sec']:.3f} s"),
        ("Mode", bm["mode"].upper()),
        ("Device", bm["hardware"].get("device", "cpu").upper()),
    ]
    trs = "".join(
        f"<tr><td><strong>{k}</strong></td><td style='font-family:monospace;'>{v}</td></tr>"
        for k, v in rows
    )
    return f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{trs}</tbody></table>"


def _build_card(result_dict: dict) -> str:
    rec_items = "".join(f"<li>{r}</li>" for r in result_dict.get("recommendations", []))
    conf = result_dict.get("confidence_score")
    conf_line = (
        f"<p class='confidence'>Confidence score: {conf*100:.0f}%</p>"
        if conf is not None
        else ""
    )
    return CARD_TEMPLATE.format(
        status=result_dict["status"],
        message=result_dict["message"],
        model_path=result_dict["model_path"],
        confidence_line=conf_line,
        benchmark_table=_benchmark_table(result_dict.get("benchmark")),
        rec_items=rec_items or "<li>No recommendations.</li>",
    )


# ── Demo scenarios ─────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "description": "Qwen3-8B against generous 500ms SLA  → PASS",
        "model": "Qwen/Qwen3-8B",
        "max_latency_ms": 500.0,
        "min_throughput": None,
    },
    {
        "description": "Qwen3-8B against tight 10ms SLA  → FAIL",
        "model": "Qwen/Qwen3-8B",
        "max_latency_ms": 10.0,
        "min_throughput": None,
    },
    {
        "description": "Invalid / corrupted model path  → ERROR",
        "model": "corrupted_file",
        "max_latency_ms": 200.0,
        "min_throughput": None,
    },
    {
        "description": "GPT-2 (124M) against chatbot SLA (150ms / 10 tok/s)  → PASS",
        "model": "openai-community/gpt2",
        "max_latency_ms": 150.0,
        "min_throughput": 10.0,
        "profile": "chatbot",
    },
    {
        "description": "Qwen2.5-7B against 200ms SLA + 0.5 tok/s  → scenario varies",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "max_latency_ms": 200.0,
        "min_throughput": 0.5,
    },
    {
        "description": "Qwen3-1.7B edge device scenario (500ms / 2 tok/s)  → PASS",
        "model": "Qwen/Qwen3-1.7B",
        "max_latency_ms": 500.0,
        "min_throughput": 2.0,
        "profile": "edge",
    },
    {
        "description": "OPT-125M real-time API scenario (50ms / 50 tok/s)  → PASS",
        "model": "facebook/opt-125m",
        "max_latency_ms": 50.0,
        "min_throughput": 50.0,
        "profile": "realtime",
    },
]


def run_demo():
    from llm_sla_gatekeeper import SLAConfig, SLAValidator

    print("\n" + "=" * 70)
    print("  🚦 LLM SLA Gatekeeper — Demo Run")
    print("  (simulation mode — no model downloads needed)")
    print("=" * 70 + "\n")

    validator = SLAValidator(force_simulation=True)
    results = []

    for i, scenario in enumerate(SCENARIOS, 1):
        profile_tag = f" [{scenario.get('profile', 'custom')}]" if scenario.get("profile") else ""
        print(f"[{i}/{len(SCENARIOS)}] {scenario['description']}{profile_tag}")
        print(f"       Model : {scenario['model']}")
        print(f"       SLA   : {scenario['max_latency_ms']} ms/tok", end="")
        if scenario.get("min_throughput"):
            print(f"  |  Min throughput: {scenario['min_throughput']} tok/s", end="")
        print()

        sla = SLAConfig(
            max_latency_ms=scenario["max_latency_ms"],
            min_throughput_tokens_per_sec=scenario.get("min_throughput"),
        )
        result = validator.validate(scenario["model"], sla)

        symbol = {"PASS": "✅", "FAIL": "❌", "ERROR": "⚠️ "}.get(result.status, "?")
        conf_str = ""
        if result.confidence_score is not None:
            conf_str = f"  [confidence: {result.confidence_score*100:.0f}%]"
        print(f"       {symbol} {result.message}{conf_str}\n")

        # Save to history
        try:
            from llm_sla_gatekeeper import append_result
            append_result(result.to_dict())
        except Exception:
            pass

        results.append(result.to_dict())

    # ── Save outputs ───────────────────────────────────────────────────────────

    # 1. results.json
    json_path = OUTPUTS_DIR / "results.json"
    json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "simulation_mode": True,
                "scenario_count": len(results),
                "scenarios": results,
            },
            indent=2,
        )
    )
    print(f"📄 Saved JSON results    → {json_path}")

    # 2. report.html
    cards_html = "\n".join(_build_card(r) for r in results)
    html_content = HTML_TEMPLATE.format(
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        cards=cards_html,
    )
    html_path = OUTPUTS_DIR / "report.html"
    html_path.write_text(html_content)
    print(f"🌐 Saved HTML report     → {html_path}")

    # 3. summary.md
    md_lines = [
        "# LLM SLA Gatekeeper — Validation Summary\n",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  \n",
        f"**Mode:** Simulation  \n",
        f"**Scenarios:** {len(results)}  \n\n",
        "| Model | SLA (ms) | Status | Avg Latency | Throughput | Confidence |",
        "|-------|----------|--------|-------------|------------|------------|",
    ]
    pass_count = fail_count = error_count = 0
    for r in results:
        bm = r.get("benchmark") or {}
        model = r["model_path"]
        sla_ms = r["sla_config"]["max_latency_ms"]
        status = r["status"]
        avg_lat = f"{bm.get('avg_latency_ms', 0):.1f} ms" if bm else "N/A"
        tps = f"{bm.get('throughput_tokens_per_sec', 0):.1f} tok/s" if bm else "N/A"
        conf = r.get("confidence_score")
        conf_str = f"{conf*100:.0f}%" if conf is not None else "—"
        md_lines.append(f"| `{model}` | {sla_ms} | **{status}** | {avg_lat} | {tps} | {conf_str} |")
        if status == "PASS":
            pass_count += 1
        elif status == "FAIL":
            fail_count += 1
        else:
            error_count += 1

    md_lines.append(f"\n**Results:** {pass_count} PASS · {fail_count} FAIL · {error_count} ERROR")

    md_path = OUTPUTS_DIR / "summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"📝 Saved Markdown summary → {md_path}")

    print(f"\n{'=' * 70}")
    print(f"  Demo complete. Output files saved to: {OUTPUTS_DIR.resolve()}")
    print(f"  Results: {pass_count} PASS · {fail_count} FAIL · {error_count} ERROR")
    print(f"{'=' * 70}\n")

    return results


if __name__ == "__main__":
    run_demo()
