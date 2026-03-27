# Examples

All examples run in simulation mode — no GPU, no model downloads.

```bash
cd llm-sla-gatekeeper       # project root
python examples/01_quick_start.py
```

## Scripts

| Script | What it demonstrates |
|--------|----------------------|
| [01_quick_start.py](01_quick_start.py) | Minimal single-model validation — `validate_model()` in ~15 lines |
| [02_advanced_usage.py](02_advanced_usage.py) | Multi-threshold SLA, batch validation, per-run samples, confidence interval |
| [03_custom_config.py](03_custom_config.py) | Built-in profiles (`chatbot`, `realtime`, `batch`, `edge`, `dev`), env-var overrides |
| [04_full_pipeline.py](04_full_pipeline.py) | End-to-end pipeline: validate → history → JSON export → Markdown table → gate decision |

## Quick reference

```python
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llm_sla_gatekeeper import validate_model, SLAConfig, SLAValidator
from llm_sla_gatekeeper import get_profile, profile_to_sla_config
from llm_sla_gatekeeper import append_result, load_history, history_summary
```
