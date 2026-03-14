from __future__ import annotations

from helios_modal_example.config import HeliosRuntimeConfig


def test_runtime_config_from_env_reads_helios_env(monkeypatch) -> None:
    monkeypatch.setenv("HELIOS_MODEL_SOURCE", "/models/Helios-Distilled")
    monkeypatch.setenv("HELIOS_DEFAULT_PROMPT", "custom prompt")
    monkeypatch.setenv("HELIOS_NEGATIVE_PROMPT", "avoid blur")
    monkeypatch.setenv("HELIOS_CHUNK_FRAMES", "66")
    monkeypatch.setenv("HELIOS_GUIDANCE_SCALE", "1.5")
    monkeypatch.setenv("HELIOS_PYRAMID_STEPS", "3,4,5")
    monkeypatch.setenv("HELIOS_AMPLIFY_FIRST_CHUNK", "0")
    monkeypatch.setenv("HELIOS_ENABLE_GROUP_OFFLOADING", "1")
    monkeypatch.setenv("HELIOS_GROUP_OFFLOADING_TYPE", "block_level")
    monkeypatch.setenv("HELIOS_MAX_SEQUENCE_LENGTH", "256")

    config = HeliosRuntimeConfig.from_env()

    assert config.backend == "real"
    assert config.helios_model_source == "/models/Helios-Distilled"
    assert config.helios_default_prompt == "custom prompt"
    assert config.helios_negative_prompt == "avoid blur"
    assert config.helios_chunk_frames == 66
    assert config.helios_guidance_scale == 1.5
    assert config.helios_pyramid_steps == (3, 4, 5)
    assert config.helios_amplify_first_chunk is False
    assert config.helios_enable_group_offloading is True
    assert config.helios_group_offloading_type == "block_level"
    assert config.helios_max_sequence_length == 256
