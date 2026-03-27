# LLM SLA Gatekeeper — Validation Summary

**Generated:** 2026-03-27 12:00 UTC
**Mode:** Simulation
**Scenarios:** 7

| Model | SLA (ms) | Status | Avg Latency | Throughput | Confidence |
|-------|----------|--------|-------------|------------|------------|
| `Qwen/Qwen3-8B` | 500.0 | **PASS** | 54.8 ms | 18.2 tok/s | 75% |
| `Qwen/Qwen3-8B` | 10.0 | **FAIL** | 54.8 ms | 18.2 tok/s | 75% |
| `corrupted_file` | 200.0 | **ERROR** | N/A | N/A | — |
| `openai-community/gpt2` | 150.0 | **PASS** | 29.6 ms | 33.8 tok/s | 75% |
| `Qwen/Qwen2.5-7B-Instruct` | 200.0 | **PASS** | 51.3 ms | 19.5 tok/s | 75% |
| `Qwen/Qwen3-1.7B` | 500.0 | **PASS** | 23.3 ms | 42.9 tok/s | 75% |
| `facebook/opt-125m` | 50.0 | **PASS** | 15.6 ms | 64.1 tok/s | 75% |

**Results:** 5 PASS · 1 FAIL · 1 ERROR
