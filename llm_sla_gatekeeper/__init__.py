"""
LLM SLA Gatekeeper — automated deployment gate for language models.

Public API:
    from llm_sla_gatekeeper import SLAConfig, SLAValidator, ValidationResult, validate_model
    from llm_sla_gatekeeper import get_profile, list_profiles, profile_to_sla_config
    from llm_sla_gatekeeper import append_result, load_history, history_for_model, clear_history, history_summary
    from llm_sla_gatekeeper import run_benchmark, BenchmarkResult
    from llm_sla_gatekeeper import get_hardware_info
"""

from .sla_validator import SLAConfig, SLAValidator, ValidationResult, validate_model, _compute_confidence
from .sla_profiles import get_profile, list_profiles, profile_to_sla_config, PROFILES
from .history import append_result, load_history, history_for_model, clear_history, history_summary
from .benchmark import run_benchmark, BenchmarkResult, TokenLatencySample, _stats
from .hardware_info import get_hardware_info, get_device

__all__ = [
    # Core validation
    "SLAConfig",
    "SLAValidator",
    "ValidationResult",
    "validate_model",
    "_compute_confidence",
    # Profiles
    "get_profile",
    "list_profiles",
    "profile_to_sla_config",
    "PROFILES",
    # History
    "append_result",
    "load_history",
    "history_for_model",
    "clear_history",
    "history_summary",
    # Benchmark
    "run_benchmark",
    "BenchmarkResult",
    "TokenLatencySample",
    "_stats",
    # Hardware
    "get_hardware_info",
    "get_device",
]
