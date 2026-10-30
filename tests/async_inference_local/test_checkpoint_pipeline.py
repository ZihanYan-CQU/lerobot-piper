import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from lerobot.async_inference_local.inference import engine as engine_module
from lerobot.async_inference_local.inference.engine import LocalPolicyEngine
from lerobot.async_inference_local.shared.configs import InferenceConfig, LocalDeploymentConfig
from lerobot.async_inference_local.shared.types import Observation, PlanSlice, Prefix, Request
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.policies.rtc import RTCConfig
from lerobot.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep


@pytest.mark.parametrize("mode", ["guided", "trained"])
@pytest.mark.parametrize("visual_normalization", [NormalizationMode.IDENTITY, NormalizationMode.MEAN_STD])
@pytest.mark.parametrize("saved_names", ["current", "missing", "stale_processor"])
def test_saved_pi05_processors_reload_rename_normalize_and_reanchor_without_real_weights(
    tmp_path, monkeypatch, mode, visual_normalization, saved_names
):
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    tokenizer.save_pretrained(tokenizer_path)
    cfg = PI05Config(
        device="cpu",
        dtype="float32",
        precision_profile="checkpoint",
        compile_model=False,
        chunk_size=6,
        n_action_steps=6,
        max_state_dim=2,
        max_action_dim=2,
        image_resolution=(8, 8),
        empty_cameras=1,
        tokenizer_max_length=32,
        text_tokenizer_name=str(tokenizer_path),
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
        action_feature_names=None if saved_names == "missing" else ["joint", "gripper"],
        rtc_training_max_delay=2,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 8, 8)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        normalization_mapping={
            "VISUAL": visual_normalization,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
    )
    cfg.validate_features()
    cfg.save_pretrained(tmp_path)
    pre, post = make_pi05_pre_post_processors(
        cfg,
        {
            "observation.state": {"mean": [0.0, 0.0], "std": [20.0, 1.0]},
            "action": {"mean": [1.0, 2.0], "std": [2.0, 4.0]},
            "observation.images.front": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
        },
    )
    if saved_names == "stale_processor":
        # The current policy config is authoritative, as when metadata was repaired
        # after these older processor files were saved.
        next(step for step in pre.steps if isinstance(step, RelativeActionsProcessorStep)).action_names = [
            "joint",
            "unnamed_dimension",
        ]
    pre.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    post.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")
    calls = []

    class Policy:
        config_class = PI05Config

        @classmethod
        def from_pretrained(cls, path, *, config):
            instance = cls()
            instance.config = config
            instance.model = SimpleNamespace(rtc_processor=None)
            instance.rtc_processor = None
            return instance

        def eval(self):
            return self

        def predict_action_chunk(self, batch, **kwargs):
            calls.append((batch, kwargs))
            values = torch.zeros((1, 6, 2))
            if mode == "trained" and kwargs["prev_chunk_left_over"] is not None:
                delay = kwargs["inference_delay"]
                values[0, :delay] = kwargs["prev_chunk_left_over"][:delay]
            return values

        def reset(self):
            pass

    monkeypatch.setattr(engine_module, "get_policy_class", lambda _: Policy)
    config = LocalDeploymentConfig(
        robot=SimpleNamespace(),
        pretrained_name_or_path=str(tmp_path),
        task="pick",
        actions_per_chunk=6,
        rtc=RTCConfig(mode=mode, execution_horizon=4),
        inference=InferenceConfig(device="cpu", precision_profile="checkpoint", compile_model=False),
        rename_map={"observation.images.camera": "observation.images.front"},
    )
    if visual_normalization != NormalizationMode.IDENTITY:
        with pytest.raises(ValueError, match="identity visual normalization"):
            LocalPolicyEngine(
                config,
                {"joint.pos": float, "gripper.pos": float, "camera": (8, 8, 3)},
                ["joint.pos", "gripper.pos"],
            )
        config.inference.image_mode = "lerobot_float32"
    engine = LocalPolicyEngine(
        config, {"joint.pos": float, "gripper.pos": float, "camera": (8, 8, 3)}, ["joint.pos", "gripper.pos"]
    )
    absolute_step = next(
        step for step in engine.postprocessor.steps if isinstance(step, AbsoluteActionsProcessorStep)
    )
    assert absolute_step.relative_step is engine.relative
    assert engine.prefix_normalizer is not next(
        step for step in engine.preprocessor.steps if type(step) is type(engine.prefix_normalizer)
    )

    def request(state, prefix):
        now = time.monotonic()
        observation = Observation(
            1,
            now,
            now,
            {
                "joint.pos": state,
                "gripper.pos": 0.3,
                "camera": np.full((8, 8, 3), 127, dtype=np.uint8),
            },
        )
        return Request(1, observation, prefix, "pick", now)

    first = engine.predict(request(10.0, Prefix(0, ())), 0.01)
    np.testing.assert_allclose(first.chunk.cpu, np.tile([11.0, 2.0], (6, 1)), atol=1e-6)
    second = engine.predict(request(20.0, Prefix(0, (PlanSlice(first.chunk, 0, 6),))), 0.01)
    torch.testing.assert_close(calls[1][1]["prev_chunk_left_over"], torch.tensor([[-5.0, 0.0]]).repeat(6, 1))
    pixels = 127 / 255
    if visual_normalization != NormalizationMode.IDENTITY:
        pixels = (pixels - 0.5) / 0.25
    torch.testing.assert_close(calls[0][0]["observation.images.front"], torch.full((1, 3, 8, 8), pixels))
    assert "observation.images.camera" not in calls[0][0]
    assert "observation.images.empty_camera_0" not in calls[0][0]
    expected = [11.0, 2.0] if mode == "trained" else [21.0, 2.0]
    np.testing.assert_allclose(second.chunk.cpu[:2], np.tile(expected, (2, 1)), atol=1e-6)
