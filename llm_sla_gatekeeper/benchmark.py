"""
Benchmarking engine for LLM inference latency and throughput measurement.

Supports:
  - Real model benchmarking via HuggingFace transformers
  - Simulated benchmarking for CI/CD and environments without GPUs
"""

import math
import os
import re
import time
import logging
import hashlib
import random
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

SIMULATION_MODE_ENV = "SLA_SIMULATION_MODE"
DEFAULT_PROMPT = "The following is a technical description of a machine learning model deployment pipeline:"
DEFAULT_NUM_TOKENS = int(os.getenv("SLA_BENCHMARK_TOKENS", "50"))
DEFAULT_WARMUP_RUNS = int(os.getenv("SLA_WARMUP_RUNS", "2"))
DEFAULT_BENCH_RUNS = int(os.getenv("SLA_BENCH_RUNS", "5"))
DEFAULT_MAX_RETRIES = int(os.getenv("SLA_BENCH_MAX_RETRIES", "2"))

# Model keywords that indicate a valid LLM path
VALID_MODEL_KEYWORDS = [
    "gpt", "llama", "qwen", "bert", "opt", "falcon", "mistral",
    "phi", "gemma", "vicuna", "alpaca", "bloom", "t5", "bart",
    "roberta", "electra", "mpt", "codegen", "starcoder", "deepseek",
    "yi", "internlm", "baichuan", "chatglm", "rwkv", "mamba",
    # Newer models (2024-2026)
    "mimo", "minimax", "reka", "glm", "grok", "nemotron",
    "granite", "command", "orca", "zephyr", "solar", "mixtral",
]

# t-distribution critical values for 95% CI (df = n-1)
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 20: 2.086, 30: 2.042,
}


def _t95(df: int) -> float:
    """Return the 95% two-tailed t-critical value for the given degrees of freedom."""
    if df <= 0:
        return 12.706
    # Exact lookup for common values
    for threshold, val in sorted(_T95.items(), reverse=True):
        if df >= threshold:
            return val
    return 1.96  # fallback for large n


def _stats(values: List[float]) -> Tuple[float, float, Tuple[float, float]]:
    """Return (mean, std_dev, confidence_interval_95) for a list of values."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, (0.0, 0.0)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, (mean, mean)
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    margin = _t95(n - 1) * std / math.sqrt(n)
    ci = (round(mean - margin, 3), round(mean + margin, 3))
    return mean, round(std, 3), ci


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class TokenLatencySample:
    run_index: int
    tokens_generated: int
    elapsed_ms: float
    tokens_per_second: float


@dataclass
class BenchmarkResult:
    model_path: str
    mode: str                          # "real" | "simulated"
    avg_latency_ms: float              # average ms per token
    p50_latency_ms: float
    p95_latency_ms: float
    throughput_tokens_per_sec: float
    total_tokens: int
    benchmark_duration_sec: float
    num_runs: int
    hardware_info: dict
    samples: List[TokenLatencySample] = field(default_factory=list)
    error: Optional[str] = None
    # Statistical enrichment (added 2026-03)
    std_latency_ms: float = 0.0
    confidence_interval_95: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))


# ── Path validation ────────────────────────────────────────────────────────────

def _looks_like_valid_model_path(model_path: str) -> bool:
    """
    Heuristic check: does this string look like a real model identifier?

    Accepts:
      - Paths that exist on disk
      - HuggingFace-style IDs (org/model or just model) with known keywords
      - Names containing size indicators like 7b, 27b, 1.5b
    """
    from pathlib import Path

    # Empty or whitespace-only strings are invalid
    if not model_path or not model_path.strip():
        return False

    # Absolute/relative paths that exist locally
    if Path(model_path).exists():
        return True

    path_lower = model_path.lower()

    # Contains a known LLM keyword
    if any(kw in path_lower for kw in VALID_MODEL_KEYWORDS):
        return True

    # Contains a parameter-size indicator (e.g. 7b, 27b, 1.5b, 350m)
    if re.search(r"\d+(?:\.\d+)?[bBmM]", path_lower):
        return True

    # HuggingFace org/model format that still has some model-like tokens
    if "/" in model_path:
        parts = model_path.split("/")
        if len(parts) == 2 and all(p.strip() for p in parts):
            return True

    return False


# ── Latency estimation (simulation) ───────────────────────────────────────────

def _estimate_model_size_b(model_path: str) -> float:
    """Parse model name for parameter count in billions."""
    path_lower = model_path.lower()
    # e.g. "27b", "3.5b", "1.5b", "350m"
    match = re.search(r"(\d+(?:\.\d+)?)\s*([bBmM])", path_lower)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit == "m":
            return value / 1000.0
        return value  # billions
    # Fallback guesses by keyword
    if "large" in path_lower:
        return 7.0
    if "xl" in path_lower:
        return 13.0
    if "xxl" in path_lower:
        return 30.0
    if "small" in path_lower:
        return 0.5
    if "base" in path_lower:
        return 0.1
    return 3.0  # default unknown


def _simulated_latency_ms(model_path: str) -> float:
    """
    Return a realistic simulated per-token latency in milliseconds.

    Empirical formula for CPU inference:
        latency ≈ 5ms × size_in_billions + baseline_overhead

    Seeded from model path for reproducibility within a run.
    """
    size_b = _estimate_model_size_b(model_path)
    # ~5 ms per billion params on modern CPU (rough but realistic)
    base_ms = 5.0 * size_b + 15.0

    # Deterministic jitter per model name so test results are stable
    seed = int(hashlib.md5(model_path.encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(seed)
    jitter = rng.uniform(-base_ms * 0.08, base_ms * 0.08)
    return max(1.0, base_ms + jitter)


def _run_simulated_benchmark(
    model_path: str,
    num_tokens: int,
    num_runs: int,
    hardware_info: dict,
) -> BenchmarkResult:
    """Produce a realistic benchmark result without loading a real model."""
    base_latency = _simulated_latency_ms(model_path)
    rng = random.Random(hash(model_path) & 0xFFFFFFFF)

    samples = []
    latencies = []

    start_wall = time.perf_counter()
    for i in range(num_runs):
        jitter = rng.gauss(0, base_latency * 0.05)
        token_latency = max(0.5, base_latency + jitter)
        elapsed_ms = token_latency * num_tokens
        tps = (num_tokens * 1000.0) / elapsed_ms
        samples.append(
            TokenLatencySample(
                run_index=i,
                tokens_generated=num_tokens,
                elapsed_ms=elapsed_ms,
                tokens_per_second=tps,
            )
        )
        latencies.append(token_latency)

    total_elapsed = time.perf_counter() - start_wall
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    avg_latency = sum(latencies) / n
    total_simulated_ms = avg_latency * num_tokens * num_runs
    benchmark_duration_sec = max(total_elapsed, total_simulated_ms / 1000.0 * 0.05)

    _, std_lat, ci95 = _stats(latencies)

    return BenchmarkResult(
        model_path=model_path,
        mode="simulated",
        avg_latency_ms=avg_latency,
        p50_latency_ms=latencies_sorted[n // 2],
        p95_latency_ms=latencies_sorted[int(n * 0.95)] if n > 1 else latencies_sorted[-1],
        throughput_tokens_per_sec=sum(s.tokens_per_second for s in samples) / n,
        total_tokens=num_tokens * num_runs,
        benchmark_duration_sec=round(benchmark_duration_sec, 3),
        num_runs=num_runs,
        hardware_info=hardware_info,
        samples=samples,
        std_latency_ms=std_lat,
        confidence_interval_95=ci95,
    )


# ── Real model benchmark ───────────────────────────────────────────────────────

def _run_real_benchmark(
    model_path: str,
    num_tokens: int,
    num_runs: int,
    warmup_runs: int,
    hardware_info: dict,
) -> BenchmarkResult:
    """Load a real HuggingFace model and benchmark it."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(f"transformers/torch not importable: {exc}") from exc

    device = hardware_info.get("device", "cpu")
    logger.info("Loading model %s on %s …", model_path, device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    input_ids = tokenizer.encode(DEFAULT_PROMPT, return_tensors="pt").to(device)

    # Warm-up
    for _ in range(warmup_runs):
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=num_tokens, do_sample=False)

    samples = []
    latencies = []
    start_wall = time.perf_counter()

    for i in range(num_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=num_tokens, do_sample=False)
        t1 = time.perf_counter()

        generated = output.shape[-1] - input_ids.shape[-1]
        elapsed_ms = (t1 - t0) * 1000.0
        token_latency = elapsed_ms / max(generated, 1)
        tps = generated / (t1 - t0)

        samples.append(
            TokenLatencySample(
                run_index=i,
                tokens_generated=generated,
                elapsed_ms=elapsed_ms,
                tokens_per_second=tps,
            )
        )
        latencies.append(token_latency)

    total_elapsed = time.perf_counter() - start_wall
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    _, std_lat, ci95 = _stats(latencies)

    return BenchmarkResult(
        model_path=model_path,
        mode="real",
        avg_latency_ms=sum(latencies) / n,
        p50_latency_ms=latencies_sorted[n // 2],
        p95_latency_ms=latencies_sorted[int(n * 0.95)] if n > 1 else latencies_sorted[-1],
        throughput_tokens_per_sec=sum(s.tokens_per_second for s in samples) / n,
        total_tokens=sum(s.tokens_generated for s in samples),
        benchmark_duration_sec=round(total_elapsed, 3),
        num_runs=num_runs,
        hardware_info=hardware_info,
        samples=samples,
        std_latency_ms=std_lat,
        confidence_interval_95=ci95,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def run_benchmark(
    model_path: str,
    hardware_info: Optional[dict] = None,
    num_tokens: int = DEFAULT_NUM_TOKENS,
    num_runs: int = DEFAULT_BENCH_RUNS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    force_simulation: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> BenchmarkResult:
    """
    Run a latency/throughput benchmark for the given model.

    Args:
        model_path: HuggingFace model ID or local path.
        hardware_info: Pre-collected hardware dict (collected if None).
        num_tokens: Number of tokens to generate per run.
        num_runs: Number of benchmark iterations.
        warmup_runs: Warm-up iterations before measurement.
        force_simulation: Skip real model loading even if possible.
        max_retries: Number of retry attempts for real benchmark failures.

    Returns:
        BenchmarkResult with latency statistics.

    Raises:
        ValueError: If model_path is empty or obviously invalid.
        FileNotFoundError: If model_path is a non-existent local path.
    """
    if not model_path or not model_path.strip():
        raise ValueError("model_path must not be empty")

    model_path = model_path.strip()

    if hardware_info is None:
        from .hardware_info import get_hardware_info
        hardware_info = get_hardware_info()

    # Validate path
    if not _looks_like_valid_model_path(model_path):
        raise FileNotFoundError(
            f"Model path '{model_path}' does not exist and does not match any "
            "known LLM identifier pattern."
        )

    sim_env = os.getenv(SIMULATION_MODE_ENV, "").lower()
    use_simulation = force_simulation or sim_env in ("1", "true", "yes")

    if not use_simulation:
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info("Retrying real benchmark (attempt %d/%d)…", attempt + 1, max_retries + 1)
                logger.info("Attempting real model benchmark for '%s'", model_path)
                return _run_real_benchmark(
                    model_path, num_tokens, num_runs, warmup_runs, hardware_info
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Real benchmark attempt %d failed (%s).", attempt + 1, exc
                )
        logger.warning(
            "All %d real benchmark attempts failed. Falling back to simulation.", max_retries + 1
        )

    logger.info("Running simulated benchmark for '%s'", model_path)
    return _run_simulated_benchmark(model_path, num_tokens, num_runs, hardware_info)
