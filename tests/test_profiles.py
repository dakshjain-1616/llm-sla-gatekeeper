"""
Tests for sla_profiles.py — SLA preset profile system.
"""

import os
import pytest

os.environ["SLA_SIMULATION_MODE"] = "1"

from llm_sla_gatekeeper.sla_profiles import get_profile, list_profiles, profile_to_sla_config, PROFILES


class TestGetProfile:

    def test_chatbot_profile_exists(self):
        p = get_profile("chatbot")
        assert p["max_latency_ms"] == 150.0
        assert p["min_throughput_tokens_per_sec"] == 10.0

    def test_realtime_profile_exists(self):
        p = get_profile("realtime")
        assert p["max_latency_ms"] == 50.0
        assert p["min_throughput_tokens_per_sec"] == 50.0

    def test_batch_profile_exists(self):
        p = get_profile("batch")
        assert p["max_latency_ms"] == 2000.0
        assert p["min_throughput_tokens_per_sec"] == 1.0

    def test_edge_profile_exists(self):
        p = get_profile("edge")
        assert p["max_latency_ms"] == 500.0

    def test_dev_profile_exists(self):
        p = get_profile("dev")
        assert p["max_latency_ms"] == 5000.0
        assert p["min_throughput_tokens_per_sec"] is None

    def test_case_insensitive(self):
        p1 = get_profile("chatbot")
        p2 = get_profile("CHATBOT")
        p3 = get_profile("ChatBot")
        assert p1["max_latency_ms"] == p2["max_latency_ms"] == p3["max_latency_ms"]

    def test_unknown_profile_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown SLA profile"):
            get_profile("nonexistent")

    def test_returns_copy_not_reference(self):
        p1 = get_profile("chatbot")
        p2 = get_profile("chatbot")
        p1["max_latency_ms"] = 99999.0
        p2_fresh = get_profile("chatbot")
        assert p2_fresh["max_latency_ms"] == 150.0  # mutation did not affect source

    def test_profile_has_name_and_description(self):
        for key in PROFILES:
            p = get_profile(key)
            assert "name" in p and p["name"]
            assert "description" in p and p["description"]


class TestListProfiles:

    def test_returns_list(self):
        result = list_profiles()
        assert isinstance(result, list)

    def test_all_five_profiles_present(self):
        keys = [item[0] for item in list_profiles()]
        assert set(keys) >= {"chatbot", "realtime", "batch", "edge", "dev"}

    def test_each_entry_has_three_fields(self):
        for item in list_profiles():
            assert len(item) == 3, f"Expected (key, name, description), got {item}"

    def test_descriptions_are_nonempty(self):
        for key, name, desc in list_profiles():
            assert desc.strip(), f"Profile '{key}' has empty description"


class TestProfileToSlaConfig:

    def test_returns_sla_config(self):
        from llm_sla_gatekeeper.sla_validator import SLAConfig
        sla = profile_to_sla_config("chatbot")
        assert isinstance(sla, SLAConfig)

    def test_chatbot_thresholds_applied(self):
        sla = profile_to_sla_config("chatbot")
        assert sla.max_latency_ms == 150.0
        assert sla.min_throughput_tokens_per_sec == 10.0

    def test_dev_no_throughput_requirement(self):
        sla = profile_to_sla_config("dev")
        assert sla.min_throughput_tokens_per_sec is None

    def test_all_profiles_create_valid_sla_config(self):
        from llm_sla_gatekeeper.sla_validator import SLAConfig
        for key in PROFILES:
            sla = profile_to_sla_config(key)
            assert isinstance(sla, SLAConfig)
            assert sla.max_latency_ms > 0

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError):
            profile_to_sla_config("bogus")


class TestProfileIntegration:
    """Use profiles end-to-end with the validator."""

    def test_dev_profile_always_passes_gpt2(self):
        """Dev profile is so generous that GPT-2 should always PASS."""
        from llm_sla_gatekeeper.sla_validator import SLAValidator
        sla = profile_to_sla_config("dev")
        validator = SLAValidator(force_simulation=True)
        result = validator.validate("openai-community/gpt2", sla)
        assert result.status == "PASS"

    def test_realtime_profile_fails_large_model(self):
        """50ms limit is too tight for a 27B model on CPU simulation."""
        from llm_sla_gatekeeper.sla_validator import SLAValidator
        sla = profile_to_sla_config("realtime")
        validator = SLAValidator(force_simulation=True)
        result = validator.validate("qwen3.5-27b", sla)
        assert result.status == "FAIL"
