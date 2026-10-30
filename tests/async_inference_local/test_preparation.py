from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from lerobot.async_inference_local.inference.observation import ObservationPreparer, resize_openpi
from lerobot.async_inference_local.inference.rtc import LocalGuidedRTCProcessor
from lerobot.async_inference_local.shared.configs import InferenceConfig
from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc import RTCConfig, RTCProcessor


def test_image_resize_preserves_openpi_pixels_and_source_buffer():
    raw = np.random.default_rng(7).integers(0, 256, size=(13, 21, 3), dtype=np.uint8)
    original = raw.copy()
    expected = Image.new("RGB", (10, 10), 0)
    expected.paste(Image.fromarray(raw).resize((10, 6), Image.Resampling.BILINEAR), (0, 2))
    np.testing.assert_array_equal(resize_openpi(raw, (10, 10)), np.asarray(expected))
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize("mode", ["openpi_uint8", "lerobot_float32"])
def test_preparer_preserves_ordering_and_reuses_staging_without_mutating_input(mode):
    config = SimpleNamespace(
        inference=InferenceConfig(device="cpu", compile_model=False, image_mode=mode),
        state_keys=["b.pos", "a.pos"],
        rename_map={"observation.images.camera": "observation.images.front"},
    )
    policy = SimpleNamespace(
        image_resolution=(8, 8),
        input_features={"observation.state": SimpleNamespace(shape=(2,))},
        image_features={"observation.images.front": SimpleNamespace(shape=(3, 8, 8))},
    )
    prepare = ObservationPreparer(config, policy, {"a.pos": float, "b.pos": float, "camera": (8, 8, 3)})
    raw = np.full((8, 8, 3), 123, dtype=np.uint8)
    raw.setflags(write=False)
    first = prepare({"a.pos": 1, "b.pos": 2, "camera": raw}, "pick")
    second = prepare({"a.pos": 3, "b.pos": 4, "camera": raw}, "place")
    assert first["task"] == ["pick"]
    torch.testing.assert_close(first["observation.state"], torch.tensor([[2.0, 1.0]]))
    torch.testing.assert_close(second["observation.state"], torch.tensor([[4.0, 3.0]]))
    torch.testing.assert_close(first["observation.images.camera"], torch.full((1, 3, 8, 8), 123 / 255))
    assert raw.min() == 123


@pytest.mark.parametrize("schedule", list(RTCAttentionSchedule))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_local_guidance_matches_common_reference_including_prepared_sampler(schedule, dtype):
    config = RTCConfig(execution_horizon=4, prefix_attention_schedule=schedule)
    local, reference = LocalGuidedRTCProcessor(config), RTCProcessor(config)
    x = torch.ones((1, 6, 3), dtype=dtype)
    previous = torch.arange(6, dtype=dtype).reshape(2, 3) / 10

    def velocity(value):
        # The native identity-VJP path evaluates velocity without parameter gradients.
        with torch.no_grad():
            return value.square() * 0.37

    padded, prepared = local.prepare_for_compilation(
        previous, 1, None, shape=x.shape, device=x.device, num_steps=10
    )
    for step in range(10):
        time = 1.0 + step * (-1.0 / 10)
        expected = reference.denoise_step(x, previous, 1, time, velocity)
        torch.testing.assert_close(
            local.denoise_step(x, previous, 1, time, velocity), expected, rtol=0, atol=0
        )
        torch.testing.assert_close(
            prepared.denoise_step(x, padded, 0, time, velocity), expected, rtol=0, atol=0
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_float_images_are_rejected(value):
    with pytest.raises(ValueError):
        resize_openpi(np.full((5, 5, 3), value, dtype=np.float32), (8, 8))


def test_explicit_empty_camera_is_left_for_pi05_padding_but_real_missing_camera_fails():
    config = SimpleNamespace(
        inference=InferenceConfig(device="cpu", compile_model=False), state_keys=None, rename_map={}
    )
    policy = SimpleNamespace(
        image_resolution=(8, 8),
        empty_cameras=1,
        input_features={"observation.state": SimpleNamespace(shape=(1,))},
        image_features={
            "observation.images.front": SimpleNamespace(shape=(3, 8, 8)),
            "observation.images.empty_camera_0": SimpleNamespace(shape=(3, 8, 8)),
        },
    )
    features = {"joint.pos": float, "front": (8, 8, 3)}
    prepare = ObservationPreparer(config, policy, features)
    batch = prepare({"joint.pos": 1.0, "front": np.zeros((8, 8, 3), dtype=np.uint8)}, "pick")
    assert "observation.images.front" in batch
    assert "observation.images.empty_camera_0" not in batch
    policy.image_features["observation.images.wrist"] = SimpleNamespace(shape=(3, 8, 8))
    with pytest.raises(ValueError, match="camera"):
        ObservationPreparer(config, policy, features)


def test_float_mode_keeps_checkpoint_dimensions_until_saved_processors_have_run():
    config = SimpleNamespace(
        inference=InferenceConfig(device="cpu", compile_model=False, image_mode="lerobot_float32"),
        state_keys=None,
        rename_map={},
    )
    policy = SimpleNamespace(
        image_resolution=(8, 8),
        input_features={"observation.state": SimpleNamespace(shape=(1,))},
        image_features={"observation.images.front": SimpleNamespace(shape=(3, 6, 10))},
    )
    prepare = ObservationPreparer(config, policy, {"joint.pos": float, "front": (13, 21, 3)})
    batch = prepare({"joint.pos": 1.0, "front": np.full((13, 21, 3), 255, dtype=np.uint8)}, "pick")
    # Early padding would insert pixels before checkpoint normalization/cropping.
    torch.testing.assert_close(batch["observation.images.front"], torch.ones(1, 3, 6, 10))
