"""
SLA Validator — core business logic for deployment gating.

Given a model and an SLAConfig, this module:
  1. Runs a benchmark (real or simulated)
  2. Compares results to the SLA thresholds
  3. Returns a GO / NO-GO ValidationResult with actionable recommendations
"""

import os
import json
import logging
import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple

from .benchmark import BenchmarkResult, run_benchmark

logger = logging.getLogger(__name__)


# ── SLA Configuration ──────────────────────────────────────────────────────────

@dataclass
class SLAConfig:
    """Thresholds that a model must meet to receive a deployment GO decision."""

    max_latency_ms: float
    """Maximum acceptable average per-token latency in milliseconds."""

    min_throughput_tokens_per_sec: Optional[float] = None
    """Minimum required tokens per second (optional)."""

    max_cost_per_1k_tokens: Optional[float] = None
    """Maximum cost in USD per 1 000 tokens (optional, requires cost_per_token_usd)."""

    cost_per_token_usd: Optional[float] = float(os.getenv("SLA_COST_PER_TOKEN_USD", "0"))
    """Cost per token in USD; used to compute cost-based SLA."""

    p95_latency_multiplier: float = float(os.getenv("SLA_P95_MULTIPLIER", "1.5"))
    """Allowed p95 latency = max_latency_ms × this multiplier."""

    def __post_init__(self):
        if self.max_latency_ms <= 0:
            raise ValueError("max_latency_ms must be positive")
        if self.min_throughput_tokens_per_sec is not None and self.min_throughput_tokens_per_sec <= 0:
            raise ValueError("min_throughput_tokens_per_sec must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


# ── Validation Result ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Complete output of one SLA validation run."""

    status: str
    """'PASS', 'FAIL', or 'ERROR'"""

    message: str
    """Human-readable one-line verdict."""

    model_path: str
    sla_config: SLAConfig
    benchmark: Optional[BenchmarkResult]
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"
    )
    detail: Optional[str] = None
    confidence_score: Optional[float] = None
    """0–1 confidence in the result based on run count and variance (None if unavailable)."""

    @property
    def is_pass(self) -> bool:
        return self.status == "PASS"

    @property
    def is_fail(self) -> bool:
        return self.status == "FAIL"

    @property
    def is_error(self) -> bool:
        return self.status == "ERROR"

    def to_dict(self) -> dict:
        bm = None
        if self.benchmark:
            bm = {
                "model_path": self.benchmark.model_path,
                "mode": self.benchmark.mode,
                "avg_latency_ms": round(self.benchmark.avg_latency_ms, 2),
                "p50_latency_ms": round(self.benchmark.p50_latency_ms, 2),
                "p95_latency_ms": round(self.benchmark.p95_latency_ms, 2),
                "throughput_tokens_per_sec": round(self.benchmark.throughput_tokens_per_sec, 2),
                "total_tokens": self.benchmark.total_tokens,
                "benchmark_duration_sec": self.benchmark.benchmark_duration_sec,
                "num_runs": self.benchmark.num_runs,
                "std_latency_ms": round(self.benchmark.std_latency_ms, 3),
                "confidence_interval_95": list(self.benchmark.confidence_interval_95),
                "hardware": self.benchmark.hardware_info,
            }
        return {
            "status": self.status,
            "message": self.message,
            "detail": self.detail,
            "model_path": self.model_path,
            "sla_config": self.sla_config.to_dict(),
            "benchmark": bm,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
            "confidence_score": self.confidence_score,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ── Confidence Scoring ─────────────────────────────────────────────────────────

def _compute_confidence(benchmark: BenchmarkResult) -> float:
    """
    Compute a 0–1 confidence score for the benchmark result.

    Higher scores mean the result is more statistically reliable.
    Factors: number of runs (more = better) and coefficient of variation (lower = better).
    """
    n = benchmark.num_runs
    avg = benchmark.avg_latency_ms
    std = benchmark.std_latency_ms

    # Run-count contribution: saturates toward 1.0 at ~20 runs
    run_score = min(n / 20.0, 1.0)

    # Variance contribution: coefficient of variation (std/mean), lower is better
    if avg > 0 and std >= 0:
        cv = std / avg
        var_score = max(0.0, 1.0 - cv * 2)  # cv=0 → 1.0; cv=0.5 → 0.0
    else:
        var_score = 1.0

    # Simulated benchmarks are inherently less certain than real ones
    mode_factor = 0.75 if benchmark.mode == "simulated" else 1.0

    return round(min(1.0, (run_score * 0.5 + var_score * 0.5) * mode_factor), 3)


# ── Recommendation Engine ──────────────────────────────────────────────────────

def _build_recommendations(
    benchmark: BenchmarkResult,
    sla: SLAConfig,
    latency_ok: bool,
    throughput_ok: bool,
    cost_ok: bool,
) -> List[str]:
    recs = []

    if not latency_ok:
        gap = benchmark.avg_latency_ms - sla.max_latency_ms
        pct_over = (gap / sla.max_latency_ms) * 100
        recs.append(
            f"Latency is {pct_over:.0f}% above threshold. "
            "Consider quantizing the model (INT8/INT4) to reduce per-token latency."
        )
        recs.append(
            "Evaluate deploying on a GPU instance (e.g., T4 or A10G) "
            "to achieve sub-100 ms per-token latency."
        )
        if benchmark.avg_latency_ms > sla.max_latency_ms * 3:
            recs.append(
                "Latency is severely over target. Consider a smaller/distilled model "
                "variant for this SLA tier."
            )

    if not throughput_ok and sla.min_throughput_tokens_per_sec:
        gap = sla.min_throughput_tokens_per_sec - benchmark.throughput_tokens_per_sec
        recs.append(
            f"Throughput is {gap:.1f} tok/s below the required "
            f"{sla.min_throughput_tokens_per_sec:.0f} tok/s minimum. "
            "Enable continuous batching or increase batch size."
        )

    if not cost_ok and sla.max_cost_per_1k_tokens and sla.cost_per_token_usd:
        recs.append(
            "Cost-per-1k-tokens exceeds budget. "
            "Switch to a more cost-efficient inference backend (vLLM, TensorRT-LLM)."
        )

    if latency_ok and throughput_ok and cost_ok:
        recs.append("All SLA thresholds met. Model is ready for production deployment.")
        if benchmark.avg_latency_ms < sla.max_latency_ms * 0.5:
            recs.append(
                "Latency headroom is >50%. You may be over-provisioning; "
                "consider scaling down compute to reduce costs."
            )

    # Low-confidence advisory
    if benchmark.std_latency_ms > benchmark.avg_latency_ms * 0.15:
        recs.append(
            f"High latency variance detected (std={benchmark.std_latency_ms:.1f} ms). "
            f"Increase --runs for a more reliable measurement."
        )

    return recs


# ── Core Validator ─────────────────────────────────────────────────────────────

class SLAValidator:
    """Validates a model against an SLAConfig and returns a GO/NO-GO decision."""

    def __init__(
        self,
        force_simulation: bool = False,
        num_tokens: int = int(os.getenv("SLA_BENCHMARK_TOKENS", "50")),
        num_runs: int = int(os.getenv("SLA_BENCH_RUNS", "5")),
    ):
        self.force_simulation = force_simulation
        self.num_tokens = num_tokens
        self.num_runs = num_runs

    def validate(self, model_path: str, sla_config: SLAConfig) -> "ValidationResult":
        """
        Run a benchmark and produce a ValidationResult.

        Args:
            model_path: HuggingFace ID or local directory path.
            sla_config: Threshold configuration.

        Returns:
            ValidationResult with status PASS / FAIL / ERROR.
        """
        if not model_path or not model_path.strip():
            return ValidationResult(
                status="ERROR",
                message="ERROR - Invalid Model Path",
                detail="model_path must not be empty",
                model_path=model_path or "",
                sla_config=sla_config,
                benchmark=None,
                recommendations=["Provide a valid HuggingFace model ID or local path."],
            )

        # Run benchmark
        try:
            benchmark = run_benchmark(
                model_path=model_path,
                num_tokens=self.num_tokens,
                num_runs=self.num_runs,
                force_simulation=self.force_simulation,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Benchmark error for '%s': %s", model_path, exc)
            return ValidationResult(
                status="ERROR",
                message="ERROR - Invalid Model Path",
                detail=str(exc),
                model_path=model_path,
                sla_config=sla_config,
                benchmark=None,
                recommendations=[
                    "Verify the model ID or path exists and is accessible.",
                    "For HuggingFace models ensure you have network access or the model is cached.",
                    "Check spelling: valid examples are 'Qwen/Qwen3-8B' or 'openai-community/gpt2'.",
                ],
            )
        except Exception as exc:
            logger.exception("Unexpected benchmark error for '%s'", model_path)
            return ValidationResult(
                status="ERROR",
                message=f"ERROR - Benchmark failed: {type(exc).__name__}",
                detail=str(exc),
                model_path=model_path,
                sla_config=sla_config,
                benchmark=None,
                recommendations=["Check logs for details. Ensure sufficient memory is available."],
            )

        # Evaluate SLA gates
        latency_ok = benchmark.avg_latency_ms <= sla_config.max_latency_ms
        p95_limit = sla_config.max_latency_ms * sla_config.p95_latency_multiplier
        p95_ok = benchmark.p95_latency_ms <= p95_limit

        throughput_ok = True
        if sla_config.min_throughput_tokens_per_sec is not None:
            throughput_ok = (
                benchmark.throughput_tokens_per_sec >= sla_config.min_throughput_tokens_per_sec
            )

        cost_ok = True
        if sla_config.max_cost_per_1k_tokens and sla_config.cost_per_token_usd:
            cost_per_1k = sla_config.cost_per_token_usd * 1000
            cost_ok = cost_per_1k <= sla_config.max_cost_per_1k_tokens

        all_ok = latency_ok and p95_ok and throughput_ok and cost_ok

        recs = _build_recommendations(
            benchmark, sla_config, latency_ok, throughput_ok, cost_ok
        )

        confidence = _compute_confidence(benchmark)

        if all_ok:
            status = "PASS"
            message = (
                f"PASS - Ready for Deployment "
                f"(avg {benchmark.avg_latency_ms:.1f} ms/tok ≤ {sla_config.max_latency_ms:.0f} ms SLA)"
            )
        else:
            failures = []
            if not latency_ok:
                failures.append(
                    f"avg latency {benchmark.avg_latency_ms:.1f} ms exceeds "
                    f"{sla_config.max_latency_ms:.0f} ms target"
                )
            if not p95_ok:
                failures.append(
                    f"p95 latency {benchmark.p95_latency_ms:.1f} ms exceeds "
                    f"{p95_limit:.0f} ms p95 limit"
                )
            if not throughput_ok:
                failures.append(
                    f"throughput {benchmark.throughput_tokens_per_sec:.1f} tok/s below "
                    f"{sla_config.min_throughput_tokens_per_sec:.0f} tok/s minimum"
                )
            if not cost_ok:
                failures.append("cost exceeds budget threshold")

            detail = "; ".join(failures)
            status = "FAIL"
            message = (
                f"FAIL - Exceeds {sla_config.max_latency_ms:.0f}ms target on Current Hardware"
                f" ({detail})"
            )

        return ValidationResult(
            status=status,
            message=message,
            detail=None if all_ok else "; ".join(
                [f"{k}={v}" for k, v in [
                    ("avg_ms", f"{benchmark.avg_latency_ms:.1f}"),
                    ("p95_ms", f"{benchmark.p95_latency_ms:.1f}"),
                    ("tps", f"{benchmark.throughput_tokens_per_sec:.1f}"),
                ]]
            ),
            model_path=model_path,
            sla_config=sla_config,
            benchmark=benchmark,
            recommendations=recs,
            confidence_score=confidence,
        )

    def validate_batch(
        self,
        model_paths: List[str],
        sla_config: SLAConfig,
    ) -> List["ValidationResult"]:
        """
        Validate multiple models against the same SLA config.

        Args:
            model_paths: List of HuggingFace IDs or local paths.
            sla_config: Shared threshold configuration.

        Returns:
            List of ValidationResult objects, one per model (same order).
        """
        return [self.validate(mp, sla_config) for mp in model_paths]


# ── Convenience function ───────────────────────────────────────────────────────

def validate_model(
    model_path: str,
    max_latency_ms: float,
    min_throughput_tokens_per_sec: Optional[float] = None,
    max_cost_per_1k_tokens: Optional[float] = None,
    force_simulation: bool = False,
) -> ValidationResult:
    """
    High-level convenience wrapper around SLAValidator.

    Example:
        result = validate_model("Qwen/Qwen3-8B", max_latency_ms=300)
        print(result.message)
    """
    sla = SLAConfig(
        max_latency_ms=max_latency_ms,
        min_throughput_tokens_per_sec=min_throughput_tokens_per_sec,
        max_cost_per_1k_tokens=max_cost_per_1k_tokens,
    )
    validator = SLAValidator(force_simulation=force_simulation)
    return validator.validate(model_path, sla)
