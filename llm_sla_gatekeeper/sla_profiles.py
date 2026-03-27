"""
Pre-built SLA profiles for common deployment scenarios.

Usage:
    from llm_sla_gatekeeper.sla_profiles import get_profile, list_profiles, profile_to_sla_config
    params = get_profile("chatbot")
    sla = profile_to_sla_config("chatbot")
"""

import os

# ── Profile definitions ────────────────────────────────────────────────────────
# All numeric values can be overridden via environment variables.

PROFILES: dict = {
    "chatbot": {
        "name": "Interactive Chatbot",
        "description": "Tight latency for real-time conversational interfaces",
        "max_latency_ms": float(os.getenv("SLA_PROFILE_CHATBOT_LATENCY_MS", "150")),
        "min_throughput_tokens_per_sec": float(os.getenv("SLA_PROFILE_CHATBOT_THROUGHPUT", "10")),
        "max_cost_per_1k_tokens": None,
        "emoji": "💬",
    },
    "realtime": {
        "name": "Real-time API",
        "description": "Ultra-low latency for high-throughput streaming API endpoints",
        "max_latency_ms": float(os.getenv("SLA_PROFILE_REALTIME_LATENCY_MS", "50")),
        "min_throughput_tokens_per_sec": float(os.getenv("SLA_PROFILE_REALTIME_THROUGHPUT", "50")),
        "max_cost_per_1k_tokens": None,
        "emoji": "⚡",
    },
    "batch": {
        "name": "Batch Processing",
        "description": "Relaxed latency for offline batch inference jobs",
        "max_latency_ms": float(os.getenv("SLA_PROFILE_BATCH_LATENCY_MS", "2000")),
        "min_throughput_tokens_per_sec": float(os.getenv("SLA_PROFILE_BATCH_THROUGHPUT", "1")),
        "max_cost_per_1k_tokens": None,
        "emoji": "📦",
    },
    "edge": {
        "name": "Edge Device",
        "description": "Moderate latency for on-device inference (mobile / IoT)",
        "max_latency_ms": float(os.getenv("SLA_PROFILE_EDGE_LATENCY_MS", "500")),
        "min_throughput_tokens_per_sec": float(os.getenv("SLA_PROFILE_EDGE_THROUGHPUT", "2")),
        "max_cost_per_1k_tokens": None,
        "emoji": "📱",
    },
    "dev": {
        "name": "Development / Debug",
        "description": "Very generous thresholds for local development and testing",
        "max_latency_ms": float(os.getenv("SLA_PROFILE_DEV_LATENCY_MS", "5000")),
        "min_throughput_tokens_per_sec": None,
        "max_cost_per_1k_tokens": None,
        "emoji": "🛠️",
    },
}


def get_profile(name: str) -> dict:
    """Return a copy of the profile config dict for the given name (case-insensitive).

    Raises KeyError if the profile is not found.
    """
    key = name.strip().lower()
    if key not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise KeyError(f"Unknown SLA profile '{name}'. Available: {available}")
    return dict(PROFILES[key])


def list_profiles() -> list:
    """Return a list of (key, name, description) tuples for all built-in profiles."""
    return [(k, v["name"], v["description"]) for k, v in PROFILES.items()]


def profile_to_sla_config(name: str):
    """Convert a profile name to an SLAConfig instance.

    Args:
        name: One of 'chatbot', 'realtime', 'batch', 'edge', 'dev'.

    Returns:
        SLAConfig populated with the profile's thresholds.
    """
    from .sla_validator import SLAConfig

    p = get_profile(name)
    return SLAConfig(
        max_latency_ms=p["max_latency_ms"],
        min_throughput_tokens_per_sec=p.get("min_throughput_tokens_per_sec"),
        max_cost_per_1k_tokens=p.get("max_cost_per_1k_tokens"),
    )
