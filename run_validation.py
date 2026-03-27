#!/usr/bin/env python3
"""
CLI entry point for LLM SLA Gatekeeper.

Usage:
    python run_validation.py --model=Qwen/Qwen3-8B --slatarget=200ms
    python run_validation.py --model=openai-community/gpt2 --slatarget=500ms --throughput=10
    python run_validation.py --config=sla_config.json --model=./local_model
    python run_validation.py --model=Qwen/Qwen3-8B --profile=chatbot --simulate
    python run_validation.py --batch="Qwen/Qwen3-8B,Qwen/Qwen3-1.7B" --profile=edge --simulate
    python run_validation.py --model=Qwen/Qwen3-8B --slatarget=300ms --output=result.json --output-format=csv
"""

import argparse
import csv
import io
import json
import logging
import os
import re
import sys
from pathlib import Path

from llm_sla_gatekeeper import SLAConfig, SLAValidator, ValidationResult

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ANSI colour codes — disabled automatically when output is not a TTY
_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR", "") == ""

_C = {
    "PASS":   "\033[1;32m",   # bold green
    "FAIL":   "\033[1;31m",   # bold red
    "ERROR":  "\033[1;33m",   # bold yellow
    "RESET":  "\033[0m",
    "DIM":    "\033[2m",
    "BOLD":   "\033[1m",
}


def _color(text: str, code: str) -> str:
    if not _USE_COLOR:
        return text
    return f"{_C.get(code, '')}{text}{_C['RESET']}"


def _parse_ms(value: str) -> float:
    """Parse latency string like '200ms', '0.2s', or '200' into milliseconds."""
    value = value.strip().lower()
    if value.endswith("ms"):
        return float(value[:-2])
    if value.endswith("s"):
        return float(value[:-1]) * 1000.0
    return float(value)


def _load_sla_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def _print_result(result: ValidationResult, verbose: bool = False) -> None:
    """Pretty-print a ValidationResult to stdout."""
    width = 72
    bar = "═" * width

    status_symbol = {"PASS": "✅", "FAIL": "❌", "ERROR": "⚠️ "}.get(result.status, "?")
    colored_status = _color(result.status, result.status)

    print(f"\n{_color(bar, 'BOLD')}")
    print(f"  {status_symbol}  DEPLOYMENT GATE: {colored_status}")
    print(_color(bar, 'BOLD'))
    print(f"  Model   : {result.model_path}")
    print(f"  Verdict : {result.message}")
    print(f"  Time    : {_color(result.timestamp, 'DIM')}")
    if result.confidence_score is not None:
        conf_pct = f"{result.confidence_score * 100:.0f}%"
        print(f"  Confidence: {conf_pct}")

    if result.benchmark:
        bm = result.benchmark
        ci = bm.confidence_interval_95
        print(f"\n  ── Benchmark Metrics ({bm.mode} mode) ─────────────────────")
        print(f"  Avg latency   : {bm.avg_latency_ms:>8.2f} ms/tok")
        print(f"  Std dev       : {bm.std_latency_ms:>8.2f} ms")
        print(f"  95% CI        :  [{ci[0]:.2f}, {ci[1]:.2f}] ms")
        print(f"  P50 latency   : {bm.p50_latency_ms:>8.2f} ms/tok")
        print(f"  P95 latency   : {bm.p95_latency_ms:>8.2f} ms/tok")
        print(f"  Throughput    : {bm.throughput_tokens_per_sec:>8.2f} tok/s")
        print(f"  Total tokens  : {bm.total_tokens}")
        print(f"  Runs          : {bm.num_runs}")
        print(f"  Duration      : {bm.benchmark_duration_sec:.3f} s")
        print(f"  Device        : {bm.hardware_info.get('device', 'unknown')}")

        if verbose and bm.samples:
            print(f"\n  ── Per-run breakdown ───────────────────────────────────")
            for s in bm.samples:
                print(
                    f"    Run {s.run_index + 1:>2}: "
                    f"{s.elapsed_ms / max(s.tokens_generated, 1):.2f} ms/tok  "
                    f"({s.tokens_per_second:.1f} tok/s,  {s.tokens_generated} tokens)"
                )

    if result.sla_config:
        sla = result.sla_config
        print(f"\n  ── SLA Thresholds ───────────────────────────────────────")
        print(f"  Max latency   : {sla.max_latency_ms:.0f} ms/tok")
        if sla.min_throughput_tokens_per_sec:
            print(f"  Min throughput: {sla.min_throughput_tokens_per_sec:.0f} tok/s")
        if sla.max_cost_per_1k_tokens:
            print(f"  Max cost/1k   : ${sla.max_cost_per_1k_tokens:.4f}")

    if result.recommendations:
        print(f"\n  ── Recommendations ──────────────────────────────────────")
        for i, rec in enumerate(result.recommendations, 1):
            words = rec.split()
            line = f"  {i}. "
            col = len(line)
            for word in words:
                if col + len(word) + 1 > width:
                    print(line)
                    line = "     " + word + " "
                    col = len(line)
                else:
                    line += word + " "
                    col += len(word) + 1
            print(line.rstrip())

    print(f"\n{_color(bar, 'BOLD')}\n")

    if verbose and result.detail:
        print(f"Detail: {result.detail}")


def _print_batch_summary(results: list) -> None:
    """Print a compact summary table for batch validation."""
    width = 72
    bar = "─" * width
    print(f"\n  {'Model':<40} {'Status':<8} {'Avg ms/tok':>10} {'tok/s':>8}")
    print(f"  {bar}")
    for r in results:
        bm = r.benchmark
        avg = f"{bm.avg_latency_ms:.1f}" if bm else "—"
        tps = f"{bm.throughput_tokens_per_sec:.1f}" if bm else "—"
        col = {"PASS": "PASS", "FAIL": "FAIL", "ERROR": "ERROR"}.get(r.status, r.status)
        colored = _color(col, r.status)
        model_display = r.model_path[:38] + ".." if len(r.model_path) > 40 else r.model_path
        print(f"  {model_display:<40} {colored:<8} {avg:>10} {tps:>8}")
    print()


def _results_to_csv(results: list) -> str:
    """Convert a list of ValidationResult to a CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "timestamp", "model_path", "status", "sla_max_latency_ms",
        "avg_latency_ms", "p50_latency_ms", "p95_latency_ms",
        "std_latency_ms", "throughput_tokens_per_sec", "total_tokens",
        "benchmark_duration_sec", "num_runs", "mode", "confidence_score",
    ])
    for r in results:
        bm = r.benchmark
        d = r.sla_config.to_dict() if r.sla_config else {}
        writer.writerow([
            r.timestamp,
            r.model_path,
            r.status,
            d.get("max_latency_ms", ""),
            f"{bm.avg_latency_ms:.3f}" if bm else "",
            f"{bm.p50_latency_ms:.3f}" if bm else "",
            f"{bm.p95_latency_ms:.3f}" if bm else "",
            f"{bm.std_latency_ms:.3f}" if bm else "",
            f"{bm.throughput_tokens_per_sec:.3f}" if bm else "",
            bm.total_tokens if bm else "",
            f"{bm.benchmark_duration_sec:.3f}" if bm else "",
            bm.num_runs if bm else "",
            bm.mode if bm else "",
            r.confidence_score if r.confidence_score is not None else "",
        ])
    return output.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="LLM SLA Gatekeeper — validate model deployment readiness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default=os.getenv("SLA_MODEL_PATH", ""),
        help="HuggingFace model ID or local path (env: SLA_MODEL_PATH)",
    )
    parser.add_argument(
        "--batch",
        default=None,
        help="Comma-separated list of model IDs to compare against the same SLA",
    )
    parser.add_argument(
        "--slatarget",
        default=os.getenv("SLA_MAX_LATENCY_MS", "200ms"),
        help="SLA latency target, e.g. '200ms' or '0.2s' (env: SLA_MAX_LATENCY_MS)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="SLA profile preset: chatbot | realtime | batch | edge | dev",
    )
    parser.add_argument(
        "--throughput",
        type=float,
        default=None,
        help="Minimum throughput in tokens/sec (optional)",
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=None,
        help="Maximum cost per 1k tokens in USD (optional)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to SLA config JSON file (overrides individual flags)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=int(os.getenv("SLA_BENCH_RUNS", "5")),
        help="Number of benchmark iterations (env: SLA_BENCH_RUNS, default: 5)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=int(os.getenv("SLA_BENCHMARK_TOKENS", "50")),
        help="Tokens to generate per run (env: SLA_BENCHMARK_TOKENS, default: 50)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save result(s) to this file path (JSON or CSV depending on --output-format)",
    )
    parser.add_argument(
        "--output-format",
        default="json",
        choices=["json", "csv", "both"],
        help="Output file format: json (default), csv, or both",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print recent validation history and exit",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=os.getenv("SLA_SIMULATION_MODE", "").lower() in ("1", "true", "yes"),
        help="Force simulation mode (no real model loading)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print extra debug details and per-run breakdown",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        default=False,
        help="Launch Gradio UI after validation",
    )

    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── History mode ───────────────────────────────────────────────────────────
    if args.history:
        try:
            from llm_sla_gatekeeper import load_history, history_summary
            records = load_history()
            summary = history_summary()
            print(f"\n  📜 Validation History  ({summary['total']} total records)")
            print(f"  PASS: {summary['pass_count']}  FAIL: {summary['fail_count']}  ERROR: {summary['error_count']}")
            if records:
                print(f"\n  {'Time':<20} {'Model':<35} {'Status':<8} {'Avg ms/tok':>10}")
                print(f"  {'─'*75}")
                for r in reversed(records[-20:]):
                    ts = r.get("timestamp", "")[:19].replace("T", " ")
                    model = r.get("model_path", "")[:33]
                    status = r.get("status", "")
                    bm = r.get("benchmark") or {}
                    avg = f"{bm.get('avg_latency_ms', 0):.1f}" if bm else "—"
                    colored = _color(status, status)
                    print(f"  {ts:<20} {model:<35} {colored:<8} {avg:>10}")
            print()
        except Exception as exc:
            print(f"Could not load history: {exc}", file=sys.stderr)
        return 0

    # ── Build SLA config ───────────────────────────────────────────────────────
    if args.profile:
        try:
            from llm_sla_gatekeeper import profile_to_sla_config, get_profile
            sla = profile_to_sla_config(args.profile)
            p = get_profile(args.profile)
            logger.info(
                "Using SLA profile '%s': max_latency=%sms, min_throughput=%s tok/s",
                args.profile, p["max_latency_ms"], p.get("min_throughput_tokens_per_sec"),
            )
        except KeyError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    elif args.config:
        cfg_data = _load_sla_config(args.config)
        sla = SLAConfig(
            max_latency_ms=float(cfg_data.get("max_latency_ms", 200)),
            min_throughput_tokens_per_sec=cfg_data.get("min_throughput_tokens_per_sec"),
            max_cost_per_1k_tokens=cfg_data.get("max_cost_per_1k_tokens"),
        )
    else:
        try:
            max_latency = _parse_ms(args.slatarget)
        except ValueError:
            print(f"ERROR: Cannot parse --slatarget '{args.slatarget}'. Use e.g. '200ms'.", file=sys.stderr)
            return 2
        sla = SLAConfig(
            max_latency_ms=max_latency,
            min_throughput_tokens_per_sec=args.throughput,
            max_cost_per_1k_tokens=args.cost,
        )

    validator = SLAValidator(
        force_simulation=args.simulate,
        num_tokens=args.tokens,
        num_runs=args.runs,
    )

    # ── Batch mode ─────────────────────────────────────────────────────────────
    if args.batch:
        model_paths = [m.strip() for m in args.batch.split(",") if m.strip()]
        print(f"\n  ⚡ Batch validation — {len(model_paths)} model(s)")
        results = validator.validate_batch(model_paths, sla)
        _print_batch_summary(results)
        for r in results:
            _print_result(r, verbose=args.verbose)

        # Auto-save to history
        try:
            from llm_sla_gatekeeper import append_result
            for r in results:
                append_result(r.to_dict())
        except Exception:
            pass

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _save_output(results, out_path, args.output_format)

        if args.ui:
            from app import launch_ui
            launch_ui()

        # Batch exit code: 0 if all PASS, 1 if any FAIL, 2 if any ERROR
        statuses = {r.status for r in results}
        if "ERROR" in statuses:
            return 2
        if "FAIL" in statuses:
            return 1
        return 0

    # ── Single model mode ──────────────────────────────────────────────────────
    result = validator.validate(args.model, sla)
    _print_result(result, verbose=args.verbose)

    # Auto-save to history
    try:
        from llm_sla_gatekeeper import append_result
        append_result(result.to_dict())
    except Exception:
        pass

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _save_output([result], out_path, args.output_format)

    if args.ui:
        from app import launch_ui
        launch_ui(initial_result=result)

    # Exit code: 0=PASS, 1=FAIL, 2=ERROR/usage
    return {"PASS": 0, "FAIL": 1, "ERROR": 2}.get(result.status, 2)


def _save_output(results: list, out_path: Path, fmt: str) -> None:
    """Save results in the requested format(s)."""
    if fmt in ("json", "both"):
        if len(results) == 1:
            out_path.with_suffix(".json").write_text(results[0].to_json())
            print(f"Result saved to: {out_path.with_suffix('.json')}")
        else:
            data = [r.to_dict() for r in results]
            out_path.with_suffix(".json").write_text(json.dumps(data, indent=2))
            print(f"Results saved to: {out_path.with_suffix('.json')}")

    if fmt in ("csv", "both"):
        csv_path = out_path.with_suffix(".csv")
        csv_path.write_text(_results_to_csv(results))
        print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    sys.exit(main())
