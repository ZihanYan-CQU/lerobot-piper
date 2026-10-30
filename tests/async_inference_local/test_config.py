from types import SimpleNamespace

import pytest

from lerobot.async_inference_local.shared.configs import InferenceConfig, LocalDeploymentConfig
from lerobot.policies.rtc import RTCConfig


@pytest.mark.parametrize(
    "overrides",
    [
        {"fps": 0},
        {"fps": float("nan")},
        {"actions_per_chunk": 1},
        {"observation_max_age_s": -1},
        {"inference_timeout_s": float("inf")},
        {"interpolation_multiplier": 0},
        {"latency_quantile": 1.1},
        {"action_keys": ["x", "x"]},
        {"duration_s": 0},
        {"trained_delay_steps": 2},
        {"reset_pose": {"x": float("nan")}},
        {"rtc": RTCConfig(enabled=False)},
        {"reset_gripper_tolerance": 0},
        {"reset_settle_s": -0.1},
        {"reset_settle_s": 5.0},
        {"reset_duration_s": float("nan")},
        {"reset_gripper_open_position": float("nan")},
    ],
)
def test_invalid_runtime_settings_fail_before_hardware_initialization(overrides):
    with pytest.raises(ValueError):
        LocalDeploymentConfig(
            robot=SimpleNamespace(), pretrained_name_or_path="model", task="pick", **overrides
        )


def test_model_overrides_are_applied_at_construction_not_after_loading():
    options = InferenceConfig(tokenizer_path="/models/tokenizer").policy_overrides()
    assert "--compile_model=true" in options
    assert "--device=cuda:0" in options
    assert "--dtype=bfloat16" in options
    assert "--text_tokenizer_name=/models/tokenizer" in options
    assert not any(
        v.startswith("--dtype=") for v in InferenceConfig(precision_profile="checkpoint").policy_overrides()
    )


def test_compiled_deployment_requires_warmup_and_known_image_semantics():
    with pytest.raises(ValueError, match="warmup"):
        InferenceConfig(warmup_runs=0)
    with pytest.raises(ValueError, match="image_mode"):
        InferenceConfig(image_mode="jpeg")
