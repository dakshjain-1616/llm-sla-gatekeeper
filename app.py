#!/usr/bin/env python3
"""
Gradio web UI for the LLM SLA Gatekeeper.

Provides an interactive deployment gate dashboard with:
  - Single-model validation with live streaming output
  - Multi-model comparison (batch)
  - Validation history viewer
  - SLA scenario preset cards
  - Real-time stats panel (token count, timing, confidence)

Launch:
    python app.py
    python app.py --model=Qwen/Qwen3-8B --slatarget=300ms
"""

import json
import logging
import os
import sys
import argparse
from typing import Optional, Tuple, Iterator

logger = logging.getLogger(__name__)

GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() in ("1", "true", "yes")
DEFAULT_MODEL = os.getenv("SLA_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_LATENCY = float(os.getenv("SLA_MAX_LATENCY_MS", "200"))
SIMULATION_MODE = os.getenv("SLA_SIMULATION_MODE", "1").lower() in ("1", "true", "yes")

# ── Preset models shown in the dropdown ───────────────────────────────────────
PRESET_MODELS = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "openai-community/gpt2",
    "facebook/opt-125m",
]


# ── HTML helpers ───────────────────────────────────────────────────────────────

def _status_badge(status: str) -> str:
    colors = {
        "PASS": ("#22c55e", "#f0fdf4", "✅"),
        "FAIL": ("#ef4444", "#fef2f2", "❌"),
        "ERROR": ("#f59e0b", "#fffbeb", "⚠️"),
        "RUNNING": ("#6366f1", "#eef2ff", "⏳"),
    }
    bg, text_bg, icon = colors.get(status, ("#6b7280", "#f9fafb", "?"))
    label = "RUNNING…" if status == "RUNNING" else f"DEPLOYMENT GATE: {status}"
    return f"""
    <div style="
        background:{text_bg};
        border:2px solid {bg};
        border-radius:12px;
        padding:20px 28px;
        text-align:center;
        margin-bottom:12px;
    ">
        <div style="font-size:2.4rem;">{icon}</div>
        <div style="font-size:1.6rem;font-weight:700;color:{bg};">{label}</div>
    </div>
    """


def _metrics_table(benchmark) -> str:
    if not benchmark:
        return "<p style='color:#9ca3af;'>No benchmark data available.</p>"
    ci = benchmark.confidence_interval_95
    rows = [
        ("Avg Latency", f"{benchmark.avg_latency_ms:.2f} ms/tok"),
        ("Std Dev", f"± {benchmark.std_latency_ms:.2f} ms"),
        ("95% CI", f"[{ci[0]:.2f}, {ci[1]:.2f}] ms"),
        ("P50 Latency", f"{benchmark.p50_latency_ms:.2f} ms/tok"),
        ("P95 Latency", f"{benchmark.p95_latency_ms:.2f} ms/tok"),
        ("Throughput", f"{benchmark.throughput_tokens_per_sec:.2f} tok/s"),
        ("Total Tokens", str(benchmark.total_tokens)),
        ("Runs", str(benchmark.num_runs)),
        ("Duration", f"{benchmark.benchmark_duration_sec:.3f} s"),
        ("Bench Mode", benchmark.mode.upper()),
        ("Device", benchmark.hardware_info.get("device", "unknown").upper()),
        ("RAM Available", f"{benchmark.hardware_info.get('ram_available_gb', 0):.1f} GB"),
    ]
    html_rows = "".join(
        f"<tr><td style='padding:6px 12px;font-weight:600;color:#374151;'>{k}</td>"
        f"<td style='padding:6px 12px;font-family:monospace;'>{v}</td></tr>"
        for k, v in rows
    )
    return f"""
    <table style="border-collapse:collapse;width:100%;font-size:0.92rem;">
        <thead>
            <tr style="background:#f3f4f6;">
                <th style="padding:8px 12px;text-align:left;color:#6b7280;">Metric</th>
                <th style="padding:8px 12px;text-align:left;color:#6b7280;">Value</th>
            </tr>
        </thead>
        <tbody>{html_rows}</tbody>
    </table>
    """


def _stats_panel(result) -> str:
    """Compact real-time stats shown next to the verdict."""
    if not result or not result.benchmark:
        return ""
    bm = result.benchmark
    conf = result.confidence_score
    conf_pct = f"{conf * 100:.0f}%" if conf is not None else "—"
    conf_color = "#22c55e" if conf and conf > 0.7 else ("#f59e0b" if conf and conf > 0.4 else "#ef4444")
    return f"""
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;">
        <div style="background:#f3f4f6;border-radius:8px;padding:8px 14px;text-align:center;min-width:90px;">
            <div style="font-size:1.3rem;font-weight:700;color:#4f46e5;">{bm.total_tokens}</div>
            <div style="font-size:0.75rem;color:#6b7280;">tokens</div>
        </div>
        <div style="background:#f3f4f6;border-radius:8px;padding:8px 14px;text-align:center;min-width:90px;">
            <div style="font-size:1.3rem;font-weight:700;color:#4f46e5;">{bm.throughput_tokens_per_sec:.1f}</div>
            <div style="font-size:0.75rem;color:#6b7280;">tok/s</div>
        </div>
        <div style="background:#f3f4f6;border-radius:8px;padding:8px 14px;text-align:center;min-width:90px;">
            <div style="font-size:1.3rem;font-weight:700;color:#4f46e5;">{bm.num_runs}</div>
            <div style="font-size:0.75rem;color:#6b7280;">runs</div>
        </div>
        <div style="background:#f3f4f6;border-radius:8px;padding:8px 14px;text-align:center;min-width:90px;">
            <div style="font-size:1.3rem;font-weight:700;color:{conf_color};">{conf_pct}</div>
            <div style="font-size:0.75rem;color:#6b7280;">confidence</div>
        </div>
        <div style="background:#f3f4f6;border-radius:8px;padding:8px 14px;text-align:center;min-width:90px;">
            <div style="font-size:1.1rem;font-weight:700;color:#4f46e5;">{bm.mode.upper()}</div>
            <div style="font-size:0.75rem;color:#6b7280;">mode</div>
        </div>
    </div>
    """


def _recommendations_html(recs: list, status: str) -> str:
    if not recs:
        return ""
    icon_map = {"PASS": "💡", "FAIL": "🔧", "ERROR": "⚠️"}
    icon = icon_map.get(status, "•")
    items = "".join(
        f"<li style='margin:6px 0;line-height:1.55;'>{icon} {r}</li>" for r in recs
    )
    return f"<ul style='padding-left:20px;margin:0;'>{items}</ul>"


def _compare_table(results: list) -> str:
    """HTML table comparing multiple model validation results."""
    if not results:
        return "<p style='color:#9ca3af;'>No results yet.</p>"
    rows_html = ""
    for r in results:
        bm = r.benchmark
        status_colors = {"PASS": "#22c55e", "FAIL": "#ef4444", "ERROR": "#f59e0b"}
        color = status_colors.get(r.status, "#6b7280")
        avg_lat = f"{bm.avg_latency_ms:.1f}" if bm else "—"
        tps = f"{bm.throughput_tokens_per_sec:.1f}" if bm else "—"
        ci = f"[{bm.confidence_interval_95[0]:.1f}, {bm.confidence_interval_95[1]:.1f}]" if bm else "—"
        conf = f"{r.confidence_score*100:.0f}%" if r.confidence_score is not None else "—"
        rows_html += f"""
        <tr>
            <td style='padding:7px 10px;font-family:monospace;font-size:0.85rem;'>{r.model_path}</td>
            <td style='padding:7px 10px;font-weight:700;color:{color};'>{r.status}</td>
            <td style='padding:7px 10px;font-family:monospace;'>{avg_lat} ms</td>
            <td style='padding:7px 10px;font-family:monospace;'>{tps} tok/s</td>
            <td style='padding:7px 10px;font-family:monospace;font-size:0.8rem;'>{ci}</td>
            <td style='padding:7px 10px;'>{conf}</td>
        </tr>"""
    return f"""
    <table style="border-collapse:collapse;width:100%;font-size:0.9rem;">
        <thead>
            <tr style="background:#f3f4f6;">
                <th style="padding:8px 10px;text-align:left;color:#6b7280;">Model</th>
                <th style="padding:8px 10px;text-align:left;color:#6b7280;">Status</th>
                <th style="padding:8px 10px;text-align:left;color:#6b7280;">Avg Latency</th>
                <th style="padding:8px 10px;text-align:left;color:#6b7280;">Throughput</th>
                <th style="padding:8px 10px;text-align:left;color:#6b7280;">95% CI</th>
                <th style="padding:8px 10px;text-align:left;color:#6b7280;">Confidence</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """


def _history_table(records: list) -> str:
    """HTML table for the history tab."""
    if not records:
        return "<p style='color:#9ca3af;padding:20px;'>No history yet. Run a validation to start tracking.</p>"
    rows_html = ""
    status_colors = {"PASS": "#22c55e", "FAIL": "#ef4444", "ERROR": "#f59e0b"}
    for r in reversed(records[-50:]):  # most recent first, cap at 50
        color = status_colors.get(r.get("status", ""), "#6b7280")
        bm = r.get("benchmark") or {}
        avg_lat = f"{bm.get('avg_latency_ms', 0):.1f}" if bm else "—"
        tps = f"{bm.get('throughput_tokens_per_sec', 0):.1f}" if bm else "—"
        ts = r.get("timestamp", "")[:19].replace("T", " ")
        model = r.get("model_path", "")
        sla_ms = r.get("sla_config", {}).get("max_latency_ms", "—")
        rows_html += f"""
        <tr>
            <td style='padding:6px 10px;font-size:0.8rem;color:#6b7280;'>{ts}</td>
            <td style='padding:6px 10px;font-family:monospace;font-size:0.82rem;'>{model}</td>
            <td style='padding:6px 10px;font-weight:700;color:{color};'>{r.get("status","")}</td>
            <td style='padding:6px 10px;font-family:monospace;'>{avg_lat} ms</td>
            <td style='padding:6px 10px;font-family:monospace;'>{tps} tok/s</td>
            <td style='padding:6px 10px;font-family:monospace;'>{sla_ms}</td>
        </tr>"""
    return f"""
    <table style="border-collapse:collapse;width:100%;font-size:0.88rem;">
        <thead>
            <tr style="background:#f3f4f6;">
                <th style="padding:8px 10px;text-align:left;color:#6b7280;">Time (UTC)</th>
                <th style="padding:8px 10px;text-align:left;color:#6b7280;">Model</th>
                <th style="padding:8px 10px;text-align:left;color:#6b7280;">Status</th>
                <th style="padding:8px 10px;text-align:left;color:#6b7280;">Avg Latency</th>
                <th style="padding:8px 10px;text-align:left;color:#6b7280;">Throughput</th>
                <th style="padding:8px 10px;text-align:left;color:#6b7280;">SLA (ms)</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """


# ── Validation runners ─────────────────────────────────────────────────────────

def run_validation_stream(
    model_path: str,
    max_latency_ms: float,
    min_throughput: Optional[float],
    simulate: bool,
    profile: str,
) -> Iterator[Tuple]:
    """
    Generator that streams live status updates to the UI while running validation.
    Yields (badge_html, message, stats_html, metrics_html, recommendations_html).
    """
    from llm_sla_gatekeeper import SLAConfig, SLAValidator

    # Step 1: show running state immediately
    yield (
        _status_badge("RUNNING"),
        f"⏳ Benchmarking {model_path.strip() or '(no model)'}…",
        "",
        "<p style='color:#9ca3af;'>Running benchmark, please wait…</p>",
        "",
    )

    # Step 2: build SLA config (from profile or manual values)
    if profile and profile != "— custom —":
        try:
            from llm_sla_gatekeeper import profile_to_sla_config
            sla = profile_to_sla_config(profile)
        except (KeyError, Exception) as exc:
            yield (
                _status_badge("ERROR"),
                f"Configuration error: {exc}",
                "", "", "",
            )
            return
    else:
        try:
            sla = SLAConfig(
                max_latency_ms=float(max_latency_ms),
                min_throughput_tokens_per_sec=float(min_throughput) if min_throughput else None,
            )
        except (ValueError, TypeError) as exc:
            yield (
                _status_badge("ERROR"),
                f"Configuration error: {exc}",
                "", "", "",
            )
            return

    # Step 3: run validation
    validator = SLAValidator(force_simulation=simulate)
    result = validator.validate(model_path.strip(), sla)

    # Step 4: auto-save to history
    try:
        from llm_sla_gatekeeper import append_result
        append_result(result.to_dict())
    except Exception:
        pass

    # Step 5: yield final results
    yield (
        _status_badge(result.status),
        result.message,
        _stats_panel(result),
        _metrics_table(result.benchmark),
        _recommendations_html(result.recommendations, result.status),
    )


def run_comparison(
    models_text: str,
    max_latency_ms: float,
    min_throughput: Optional[float],
    simulate: bool,
    profile: str,
) -> str:
    """Run batch validation on multiple models (newline-separated) and return comparison table HTML."""
    from llm_sla_gatekeeper import SLAConfig, SLAValidator

    model_paths = [m.strip() for m in models_text.splitlines() if m.strip()]
    if not model_paths:
        return "<p style='color:#9ca3af;'>Enter at least one model path above.</p>"

    if profile and profile != "— custom —":
        try:
            from llm_sla_gatekeeper import profile_to_sla_config
            sla = profile_to_sla_config(profile)
        except Exception as exc:
            return f"<p style='color:#ef4444;'>Profile error: {exc}</p>"
    else:
        try:
            sla = SLAConfig(
                max_latency_ms=float(max_latency_ms),
                min_throughput_tokens_per_sec=float(min_throughput) if min_throughput else None,
            )
        except (ValueError, TypeError) as exc:
            return f"<p style='color:#ef4444;'>Config error: {exc}</p>"

    validator = SLAValidator(force_simulation=simulate)
    results = validator.validate_batch(model_paths, sla)

    # Save to history
    try:
        from llm_sla_gatekeeper import append_result
        for r in results:
            append_result(r.to_dict())
    except Exception:
        pass

    return _compare_table(results)


def load_history_html() -> str:
    """Load and render the history table."""
    try:
        from llm_sla_gatekeeper import load_history
        records = load_history(limit=100)
        return _history_table(records)
    except Exception as exc:
        return f"<p style='color:#ef4444;'>Could not load history: {exc}</p>"


# ── UI Layout ──────────────────────────────────────────────────────────────────

def build_ui():
    import gradio as gr

    profile_choices = ["— custom —", "chatbot", "realtime", "batch", "edge", "dev"]
    profile_labels = {
        "— custom —": "— custom —",
        "chatbot": "💬 chatbot (150ms / 10 tok/s)",
        "realtime": "⚡ realtime (50ms / 50 tok/s)",
        "batch": "📦 batch (2000ms / 1 tok/s)",
        "edge": "📱 edge (500ms / 2 tok/s)",
        "dev": "🛠️ dev (5000ms)",
    }

    with gr.Blocks(
        title="LLM SLA Gatekeeper",
        theme=gr.themes.Glass(),
        css="""
            .status-box { border-radius:8px; }
            footer { display:none !important; }
            .neo-header { text-align:center; padding:8px 0 4px; }
            .neo-header h1 { font-size:2rem; margin-bottom:4px; }
            .neo-header p  { color:#6b7280; margin:0; }
        """,
    ) as demo:
        gr.HTML("""
            <div class="neo-header">
                <h1>🚦 LLM SLA Gatekeeper</h1>
                <p><strong>Automated Deployment Gate for Language Models</strong> &mdash;
                deterministic PASS / FAIL / ERROR verdict before your model touches users.</p>
                <p style="font-size:0.82rem;margin-top:6px;">
                    Built autonomously by
                    <a href="https://heyneo.so" target="_blank" style="color:#7B61FF;font-weight:600;">NEO</a>
                    &nbsp;·&nbsp;
                    <a href="https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo"
                       target="_blank"
                       style="background:#7B61FF;color:#fff;padding:2px 10px;border-radius:4px;
                              text-decoration:none;font-size:0.78rem;">Install NEO for VS&nbsp;Code</a>
                </p>
            </div>
        """)

        with gr.Tabs():
            # ── Tab 1: Validate ────────────────────────────────────────────────
            with gr.Tab("🚦 Validate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Configuration")

                        model_input = gr.Dropdown(
                            label="Model Path / HuggingFace ID",
                            choices=PRESET_MODELS,
                            value=DEFAULT_MODEL if DEFAULT_MODEL in PRESET_MODELS else PRESET_MODELS[0],
                            allow_custom_value=True,
                            info="Select a preset or type your own HF model ID / local path",
                        )

                        scenario_radio = gr.Radio(
                            label="Quick Scenario Preset",
                            choices=[
                                "💬 Chatbot — 150 ms / 10 tok/s  (interactive chat)",
                                "⚡ Real-time — 50 ms / 50 tok/s  (streaming API)",
                                "📦 Batch — 2 000 ms / 1 tok/s  (offline jobs)",
                                "📱 Edge — 500 ms / 2 tok/s  (mobile / IoT)",
                            ],
                            value=None,
                            info="Select a preset to auto-fill model + SLA profile below",
                        )

                        profile_input = gr.Dropdown(
                            label="SLA Profile",
                            choices=list(profile_labels.values()),
                            value="— custom —",
                            info="Choose a preset or '— custom —' to set thresholds manually",
                        )
                        latency_input = gr.Number(
                            label="SLA Max Latency (ms/token) — used when profile = custom",
                            value=DEFAULT_LATENCY,
                            minimum=1,
                            maximum=100000,
                        )
                        throughput_input = gr.Number(
                            label="Min Throughput (tok/s) — optional, custom only",
                            value=None,
                            minimum=0,
                        )
                        simulate_input = gr.Checkbox(
                            label="Simulation Mode (no model download)",
                            value=SIMULATION_MODE,
                        )
                        validate_btn = gr.Button("🚀 Run Validation", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Gate Decision")
                        badge_out = gr.HTML(
                            value="<p style='color:#9ca3af;text-align:center;padding:40px;'>"
                            "Click <b>Run Validation</b> to check deployment readiness.</p>"
                        )
                        verdict_out = gr.Textbox(
                            label="Verdict",
                            interactive=False,
                            show_copy_button=True,
                        )
                        stats_out = gr.HTML(label="Live Stats")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📈 Benchmark Metrics")
                        metrics_out = gr.HTML()

                    with gr.Column():
                        gr.Markdown("### 💡 Recommendations")
                        recs_out = gr.HTML()

                # Scenario radio — pre-fill model + SLA profile
                def _set_scenario(choice):
                    """Map a scenario radio selection to (model, profile_label, latency)."""
                    if not choice:
                        return gr.update(), gr.update(), gr.update()
                    if "Chatbot" in choice:
                        return "Qwen/Qwen2.5-7B-Instruct", profile_labels["chatbot"], DEFAULT_LATENCY
                    if "Real-time" in choice:
                        return "openai-community/gpt2", profile_labels["realtime"], DEFAULT_LATENCY
                    if "Batch" in choice:
                        return "Qwen/Qwen3-8B", profile_labels["batch"], DEFAULT_LATENCY
                    if "Edge" in choice:
                        return "Qwen/Qwen3-1.7B", profile_labels["edge"], DEFAULT_LATENCY
                    return gr.update(), gr.update(), gr.update()

                scenario_radio.change(
                    _set_scenario,
                    inputs=[scenario_radio],
                    outputs=[model_input, profile_input, latency_input],
                )

                def _profile_key_from_label(label: str) -> str:
                    """Map display label back to profile key."""
                    for k, v in profile_labels.items():
                        if v == label:
                            return k
                    return "— custom —"

                def _run_stream(model, latency, throughput, simulate, profile_label):
                    profile_key = _profile_key_from_label(profile_label)
                    yield from run_validation_stream(model, latency, throughput, simulate, profile_key)

                validate_btn.click(
                    fn=_run_stream,
                    inputs=[model_input, latency_input, throughput_input, simulate_input, profile_input],
                    outputs=[badge_out, verdict_out, stats_out, metrics_out, recs_out],
                )

            # ── Tab 2: Compare Models ──────────────────────────────────────────
            with gr.Tab("⚡ Compare Models"):
                gr.Markdown(
                    "### Compare Multiple Models Against the Same SLA\n"
                    "Enter one model ID per line."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        compare_models_input = gr.Textbox(
                            label="Models (one per line)",
                            value="Qwen/Qwen3-8B\nQwen/Qwen3-1.7B\nQwen/Qwen3-0.6B\nopenai-community/gpt2\nfacebook/opt-125m",
                            lines=6,
                        )
                        compare_profile_input = gr.Dropdown(
                            label="SLA Profile",
                            choices=list(profile_labels.values()),
                            value=profile_labels["chatbot"],
                        )
                        compare_latency_input = gr.Number(
                            label="Max Latency (ms) — custom profile only",
                            value=DEFAULT_LATENCY,
                        )
                        compare_throughput_input = gr.Number(
                            label="Min Throughput (tok/s) — optional",
                            value=None,
                        )
                        compare_simulate_input = gr.Checkbox(
                            label="Simulation Mode",
                            value=SIMULATION_MODE,
                        )
                        compare_btn = gr.Button("⚡ Compare All", variant="primary")

                    with gr.Column(scale=2):
                        compare_out = gr.HTML(
                            value="<p style='color:#9ca3af;'>Results will appear here.</p>"
                        )

                def _run_compare(models, latency, throughput, simulate, profile_label):
                    profile_key = _profile_key_from_label(profile_label)
                    return run_comparison(models, latency, throughput, simulate, profile_key)

                compare_btn.click(
                    fn=_run_compare,
                    inputs=[
                        compare_models_input, compare_latency_input,
                        compare_throughput_input, compare_simulate_input,
                        compare_profile_input,
                    ],
                    outputs=[compare_out],
                )

            # ── Tab 3: History ─────────────────────────────────────────────────
            with gr.Tab("📜 History"):
                gr.Markdown(
                    "### Recent Validations\n"
                    "All results are persisted to `outputs/history.jsonl`."
                )
                history_out = gr.HTML(value=load_history_html())
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh")
                    clear_btn = gr.Button("🗑️ Clear History", variant="stop")

                def _refresh_history():
                    return load_history_html()

                def _clear_history():
                    try:
                        from llm_sla_gatekeeper import clear_history
                        n = clear_history()
                        return f"<p style='color:#22c55e;'>Cleared {n} records.</p>"
                    except Exception as exc:
                        return f"<p style='color:#ef4444;'>Error: {exc}</p>"

                refresh_btn.click(_refresh_history, outputs=[history_out])
                clear_btn.click(_clear_history, outputs=[history_out])

            # ── Tab 4: About ───────────────────────────────────────────────────
            with gr.Tab("ℹ️ About"):
                gr.Markdown(
                    """
                    ## LLM SLA Gatekeeper

                    An automated deployment gatekeeper that validates whether a model meets a
                    specific **Service Level Agreement (SLA)** on target hardware before allowing deployment.

                    ### Quick Start
                    ```bash
                    # CLI — simulate validation (no model download)
                    python run_validation.py --model=Qwen/Qwen3-8B --slatarget=300ms --simulate

                    # Use a preset SLA profile
                    python run_validation.py --model=Qwen/Qwen3-8B --profile=chatbot --simulate

                    # Compare multiple models
                    python run_validation.py --batch="Qwen/Qwen3-8B,openai-community/gpt2" --profile=edge --simulate

                    # Run demo (5 scenarios, simulation mode)
                    python demo.py
                    ```

                    ### SLA Profiles
                    | Profile | Max Latency | Min Throughput | Use case |
                    |---------|-------------|----------------|----------|
                    | `chatbot` | 150 ms | 10 tok/s | Interactive chat |
                    | `realtime` | 50 ms | 50 tok/s | Streaming API |
                    | `batch` | 2000 ms | 1 tok/s | Offline jobs |
                    | `edge` | 500 ms | 2 tok/s | Mobile / IoT |
                    | `dev` | 5000 ms | — | Development |

                    ### Exit Codes (CLI)
                    - `0` → PASS (ready for deployment)
                    - `1` → FAIL (does not meet SLA)
                    - `2` → ERROR (validation could not complete)

                    ### Simulation Mode
                    Latency is estimated as `5 ms × model_size_in_B + 15 ms` (CPU baseline).
                    No model downloads or GPU required. Ideal for CI/CD pipelines.
                    """
                )

        gr.HTML("""
            <div style="text-align:center;padding:14px 0 6px;color:#9ca3af;font-size:0.82rem;border-top:1px solid #e5e7eb;margin-top:8px;">
                Built autonomously using
                <a href="https://heyneo.so" target="_blank" style="color:#7B61FF;">NEO</a>
                — Your Autonomous AI Agent &nbsp;·&nbsp;
                Simulation mode estimates latency from model parameter count
            </div>
        """)

    return demo


# ── Launch helpers ─────────────────────────────────────────────────────────────

def launch_ui(initial_result=None):
    """Build and launch the Gradio UI."""
    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=GRADIO_SHARE,
        show_error=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="LLM SLA Gatekeeper UI")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--slatarget", default=f"{DEFAULT_LATENCY}ms")
    parser.add_argument("--port", type=int, default=GRADIO_SERVER_PORT)
    parser.add_argument("--share", action="store_true", default=GRADIO_SHARE)
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=SIMULATION_MODE,
    )
    args = parser.parse_args(argv)

    global GRADIO_SERVER_PORT, GRADIO_SHARE, SIMULATION_MODE, DEFAULT_MODEL, DEFAULT_LATENCY
    GRADIO_SERVER_PORT = args.port
    GRADIO_SHARE = args.share
    SIMULATION_MODE = args.simulate

    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=GRADIO_SHARE,
        show_error=True,
    )


if __name__ == "__main__":
    main()
