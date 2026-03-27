# LLM SLA Gatekeeper – Block bad model deploys before they hit production

> *Made autonomously using [NEO](https://heyneo.so) · [![Install NEO Extension](https://img.shields.io/badge/VS%20Code-Install%20NEO-7B61FF?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-134%20passed-brightgreen.svg)]()

> Gate any LLM deployment on measurable latency and throughput SLAs — get a GO/NO-GO verdict with actionable recommendations before you ship.

## Install

```bash
git clone https://github.com/dakshjain-1616/llm-sla-gatekeeper
cd llm-sla-gatekeeper
pip install -r requirements.txt
```

## What problem this solves

You merge a new quantized model, deploy it to staging, and only discover it's 3× slower than the previous version when users start complaining. Tools like `mlflow` and `wandb` track accuracy metrics but have no concept of per-token latency SLAs or throughput floors. There's no standard way to say "this model must serve ≤200ms p95 latency at ≥50 tok/s or the deploy is blocked." LLM SLA Gatekeeper runs a benchmark against configurable thresholds and returns a hard GO/NO-GO verdict with specific remediation steps before you cut traffic.

## What problem this solves

You merge a new quantized model, deploy it to staging, and only discover it's 3× slower than the previous version when users start complaining. Tools like `mlflow` and `wandb` track accuracy metrics but have no concept of per-token latency SLAs or throughput floors. LLM SLA Gatekeeper runs a real benchmark and returns a hard `PASS`/`FAIL` with specific fix recommendations.

## Real world examples

```python
from llm_sla_gatekeeper import validate_model

# Gate a Qwen3 deploy — must hit 200ms latency and 50 tok/s
result = validate_model("Qwen/Qwen3-8B", max_latency_ms=200, min_throughput_tokens_per_sec=50)
print(result.status)   # FAIL
print(result.message)  # FAIL - Exceeds 200ms target (throughput 18.4 tok/s below 50 tok/s minimum)
for rec in result.recommendations:
    print("-", rec)
# - Throughput is 31.6 tok/s below the required 50 tok/s minimum.
#   Enable continuous batching or increase batch size.
```

```python
# Use a named SLA profile — no need to hard-code numbers
from llm_sla_gatekeeper import get_profile, profile_to_sla_config, validate_model, list_profiles

print(list_profiles())
# [('chatbot', 'Interactive Chatbot', 'Tight latency for real-time conversational interfaces'),
#  ('realtime', 'Real-time API', 'Ultra-low latency for high-throughput streaming API endpoints'),
#  ('batch', 'Batch Processing', 'Relaxed latency for offline batch inference jobs'),
#  ('edge', 'Edge Device', 'Moderate latency for on-device inference (mobile / IoT)'),
#  ('dev', 'Development / Debug', 'Very generous thresholds for local development and testing')]

cfg = profile_to_sla_config("chatbot")
result = validate_model("meta-llama/Llama-3-8B", cfg.max_latency_ms)
print(result.status)  # PASS or FAIL
```

```python
# Track validation history across model versions
from llm_sla_gatekeeper import validate_model, history_summary

validate_model("model-v1", max_latency_ms=300)
validate_model("model-v2", max_latency_ms=300)

summary = history_summary()
print(summary)
# {'total_runs': 2, 'pass_rate': 0.5, 'models_tested': ['model-v1', 'model-v2']}
```

```bash
# Run in simulation mode (no GPU / model download required)
SLA_SIMULATION_MODE=1 python scripts/demo.py
# Saves validation report to outputs/sla_report_<timestamp>.html
```

## Who it's for

ML engineers who run model rollouts — especially those managing multiple quantized variants (Q4, Q8, F16) and need a lightweight gate in their CI/CD pipeline before flipping traffic. If you've ever been paged because a model swap made your p95 latency spike, this is the pre-deploy check you needed.

## Key features

- Named SLA profiles: `chatbot`, `realtime`, `batch`, `edge`, `dev` — no hard-coded thresholds per project
- Confidence score on each validation result based on run count and variance
- Full validation history with `history_summary()` for tracking model regressions over time
- Simulation mode (`SLA_SIMULATION_MODE=1`) — runs without GPU or model download for CI
- HTML report output with per-metric breakdown and recommendations

## Run tests

```
$ pytest tests/ -v --tb=no -q --no-header

tests/test_batch.py ............................                         [ 20%]
tests/test_benchmark.py ...............................                  [ 44%]
tests/test_history.py ...................                                [ 58%]
tests/test_profiles.py ....................                              [ 73%]
tests/test_validator.py ....................................             [100%]

134 passed in 2.52s
```

## Project structure

```
llm-sla-gatekeeper/
├── llm_sla_gatekeeper/   ← core library
│   ├── sla_validator.py  ← validate_model(), SLAConfig, ValidationResult
│   ├── sla_profiles.py   ← named profiles (chatbot, realtime, batch…)
│   ├── benchmark.py      ← token latency benchmarking engine
│   ├── history.py        ← validation history tracking
│   └── hardware_info.py  ← device detection
├── tests/                ← 134 tests
├── scripts/demo.py       ← runnable demo, saves HTML report
└── requirements.txt
```
