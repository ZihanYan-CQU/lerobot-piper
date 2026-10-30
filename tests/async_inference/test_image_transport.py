# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Verify each image mode's pixels, negotiation, and RTC snapshots."""

import hashlib
import json
import pickle  # nosec
import threading
import time
import zlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.async_inference.client.robot_client_deployment import _RTCClient
from lerobot.async_inference.server.policy_server_deployment import (
    PolicyServerDeployment,
    PolicyServerDeploymentConfig,
)
from lerobot.async_inference.shared.helpers import (
    RemotePolicyConfig,
    TimedObservation,
    raw_observation_to_observation,
)
from lerobot.async_inference.shared.image_transport import (
    IMAGE_TRANSPORT_METADATA_KEY,
    IMAGE_TRANSPORT_MODES,
    ImageResizeSpec,
    PackedPolicyImage,
    encode_observation_images,
    image_transport_metadata,
    negotiate_image_transport,
    parse_image_transport_metadata,
    prepare_remote_observation,
)
from lerobot.async_inference.shared.rtc_remote import (
    DEPLOYMENT_RTC_PROTOCOL_VERSION,
    RemoteRTCActionChunk,
    RemoteRTCObservation,
    RemoteRTCPolicySetup,
)
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.rtc import RTCConfig
from lerobot.processor import DeviceProcessorStep, NormalizerProcessorStep, RenameObservationsProcessorStep


class _Identity:
    steps = ()

    def __call__(self, value):
        return value

    def reset(self):
        pass


def _features(shape=(480, 640, 3)):
    return {
        "observation.images.top": {"dtype": "image", "shape": shape},
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
    }


def _config(feature_shape=(3, 480, 640), resolution=(224, 224)):
    return PI05Config(
        input_features={"observation.images.top": PolicyFeature(FeatureType.VISUAL, feature_shape)},
        image_resolution=resolution,
        rtc_training_max_delay=10,
    )


def _raw(shape=(480, 640, 3)):
    return {
        "top": np.random.default_rng(10).integers(0, 256, shape, dtype=np.uint8),
        "joint.pos": 1.25,
        "task": "pick the block",
    }


def _vision_input(config, observation, device="cpu"):
    # Use the production Pi05 preprocessing, without loading any model weights.
    policy = torch.nn.Module()
    policy.config = config
    policy.register_parameter("anchor", torch.nn.Parameter(torch.zeros(1, device=device)))
    return PI05Policy._preprocess_images(policy, observation)[0][0]


@pytest.fixture(autouse=True, scope="module")
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "shape,feature_shape,resolution",
    [
        ((480, 640, 3), (3, 480, 640), (224, 224)),
        ((640, 481, 3), (3, 640, 481), (224, 224)),
        ((480, 640, 3), (3, 224, 224), (224, 224)),
        ((480, 640, 3), (3, 360, 480), (224, 224)),
        ((960, 1280, 3), (3, 960, 1280), (384, 384)),
    ],
)
def test_actual_pi05_vision_input_matches_bit_for_bit_on_cpu(shape, feature_shape, resolution):
    config, raw, features = _config(feature_shape, resolution), _raw(shape), _features(shape)
    untouched = raw["top"].copy()
    plan = negotiate_image_transport("pi05", config, features, _Identity())
    encoded, raw_bytes, wire_bytes = encode_observation_images(raw, plan)
    assert isinstance(encoded["top"], PackedPolicyImage)
    encoded = pickle.loads(pickle.dumps(encoded))  # nosec
    before = raw_observation_to_observation(raw, features, config.image_features)
    after = prepare_remote_observation(encoded, features, config.image_features, plan)
    assert torch.equal(_vision_input(config, before), _vision_input(config, after))
    assert torch.equal(before["observation.state"], after["observation.state"])
    assert after["task"] == raw["task"]
    np.testing.assert_array_equal(raw["top"], untouched)
    assert wire_bytes < raw_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_resize_matches_server_cuda_vision_input_within_float32_roundoff():
    config, raw, features = _config(), _raw(), _features()
    plan = negotiate_image_transport("pi05", config, features, _Identity())
    encoded, _, _ = encode_observation_images(raw, plan)
    before = raw_observation_to_observation(raw, features, config.image_features)
    after = prepare_remote_observation(encoded, features, config.image_features, plan)
    torch.testing.assert_close(
        _vision_input(config, before, "cuda"), _vision_input(config, after, "cuda"), atol=2e-6, rtol=0
    )


@pytest.mark.parametrize(
    "shape,feature_shape,resolution,expected_sha256",
    [
        (
            (480, 640, 3),
            (3, 480, 640),
            (224, 224),
            "0448807d60c0da2a05a8dd93cc6c5fd06caf18e718c056e9fb177ce872403bda",
        ),
        (
            (640, 481, 3),
            (3, 640, 481),
            (224, 224),
            "04ee2d804e2a9e7fee6585afb749fd7c428c7eb3a12664c914a3d13f706b2bae",
        ),
        (
            (480, 640, 3),
            (3, 224, 224),
            (224, 224),
            "0448807d60c0da2a05a8dd93cc6c5fd06caf18e718c056e9fb177ce872403bda",
        ),
        (
            (960, 1280, 3),
            (3, 960, 1280),
            (384, 384),
            "93c8739de9f5a7a722243f487354a6e3c3ad9f089d50df052a1ff3076409e887",
        ),
        (
            (224, 224, 3),
            (3, 480, 640),
            (224, 224),
            "81d0d248228b42499b7dcc13d78dd1067df3f32bb1b13f79f82c79d7cb103c0f",
        ),
        (
            (480, 641, 3),
            (3, 360, 480),
            (192, 256),
            "e181b9652e17f0187f14202799e293991fb2148a4559d630eb01a02964102123",
        ),
        (
            (24, 32, 3),
            (3, 24, 32),
            (224, 224),
            "87667f77b425bd2c656ac601dd52edc4d22c1158729775922865deb60ed6a37f",
        ),
    ],
)
def test_openpi_uint8_matches_reference_pixels_without_server_resize(
    shape, feature_shape, resolution, expected_sha256
):
    # Golden hashes generated with openpi_client.image_tools.resize_with_pad
    # at OpenPI commit 215abfb, Pillow 12.3.0, and _raw()'s fixed NumPy seed.
    # No OpenPI installation is needed to run this regression.
    config, raw, features = _config(feature_shape, resolution), _raw(shape), _features(shape)
    untouched = raw["top"].copy()
    plan = negotiate_image_transport("pi05", config, features, _Identity(), mode="openpi_uint8")
    encoded, raw_bytes, wire_bytes = encode_observation_images(raw, plan)
    encoded = pickle.loads(pickle.dumps(encoded))  # nosec
    packed = encoded["top"]
    assert isinstance(packed, PackedPolicyImage)
    assert packed.spec.mode == "openpi_uint8"
    assert hashlib.sha256(packed.data).hexdigest() == expected_sha256
    assert raw_bytes == raw["top"].nbytes
    assert wire_bytes == 3 * resolution[0] * resolution[1]
    pixels = np.frombuffer(packed.data, dtype=np.uint8).reshape(*resolution, 3).copy()
    expected = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float() / 255
    prepared = prepare_remote_observation(encoded, features, config.image_features, plan)
    assert torch.equal(prepared["observation.images.top"], expected)
    assert prepared["observation.images.top"].dtype == torch.float32
    assert torch.equal(_vision_input(config, prepared), expected * 2 - 1)
    assert prepared["observation.state"].item() == raw["joint.pos"]
    assert prepared["task"] == raw["task"]
    np.testing.assert_array_equal(raw["top"], untouched)


def test_openpi_uint8_accepts_normalized_float_and_noncontiguous_camera_images():
    config, raw, features = _config(), _raw(), _features()
    raw["top"] = raw["top"][:, ::-1]
    plan = negotiate_image_transport("pi05", config, features, _Identity(), mode="openpi_uint8")
    encoded, _, _ = encode_observation_images(raw, plan)
    normalized = {**raw, "top": raw["top"].astype(np.float32) / 255}
    from_float, _, _ = encode_observation_images(normalized, plan)
    assert encoded["top"].data == from_float["top"].data
    assert normalized["top"].dtype == np.float32
    for invalid in (np.nan, np.inf, -0.01, 1.01):
        normalized["top"][0, 0, 0] = invalid
        with pytest.raises(ValueError, match="finite pixels"):
            encode_observation_images(normalized, plan)


@pytest.mark.parametrize(
    "pixels",
    [
        np.zeros((10, 10, 3), dtype=np.float64),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((0, 0, 3), dtype=np.uint8),
        np.zeros((3, 10, 10), dtype=np.uint8),
    ],
)
def test_openpi_rejects_invalid_pixels_instead_of_silently_uploading_raw(pixels):
    plan = {"top": ImageResizeSpec((3, 480, 640), (224, 224), mode="openpi_uint8")}
    with pytest.raises(ValueError, match="RGB"):
        encode_observation_images({"top": pixels}, plan)


@pytest.mark.parametrize("image_mode", IMAGE_TRANSPORT_MODES)
@pytest.mark.parametrize("reason", ["other_policy", "normalize", "cast", "rename", "custom", "memory"])
def test_unknown_or_image_altering_pipeline_does_not_negotiate_resize(reason, image_mode):
    config, processor, policy_type = _config(), _Identity(), "pi05"
    if reason == "other_policy":
        policy_type = "smolvla"
    elif reason == "normalize":
        processor.steps = [
            NormalizerProcessorStep(config.input_features, {FeatureType.VISUAL: NormalizationMode.MEAN_STD})
        ]
    elif reason == "cast":
        processor.steps = [DeviceProcessorStep(device="cpu", float_dtype="bfloat16")]
    elif reason == "rename":
        processor.steps = [
            RenameObservationsProcessorStep({"observation.images.top": "observation.images.wrist"})
        ]
    elif reason == "custom":
        processor.steps = [object()]
    else:
        config.use_visual_memory = True
    assert not negotiate_image_transport(policy_type, config, _features(), processor, mode=image_mode)


def test_raw_and_prepared_images_can_coexist_without_double_resize():
    config, raw, features = _config(), _raw(), _features()
    plan = negotiate_image_transport("pi05", config, features, _Identity())
    plan["wrist"] = ImageResizeSpec((3, 224, 224), (224, 224))
    config.input_features["observation.images.wrist"] = PolicyFeature(FeatureType.VISUAL, (3, 224, 224))
    features["observation.images.wrist"] = {"dtype": "image", "shape": (224, 224, 3)}
    raw["wrist"] = _raw((224, 224, 3))["top"]
    encoded, _, _ = encode_observation_images(raw, plan)
    assert encoded["wrist"] is raw["wrist"]
    result = prepare_remote_observation(encoded, features, config.image_features, plan)
    expected = raw_observation_to_observation(raw, features, config.image_features)
    assert torch.equal(result["observation.images.wrist"], expected["observation.images.wrist"])


def test_larger_native_resolution_falls_back_when_lossless_pixels_cost_more_than_raw():
    config, raw, features = _config(resolution=(384, 384)), _raw(), _features()
    plan = negotiate_image_transport("pi05", config, features, _Identity())
    encoded, raw_bytes, wire_bytes = encode_observation_images(raw, plan)
    assert plan["top"].model_resolution == (384, 384)
    assert encoded["top"] is raw["top"]
    assert wire_bytes == raw_bytes


@pytest.mark.parametrize("image_mode", IMAGE_TRANSPORT_MODES)
def test_invalid_or_unnegotiated_payload_is_rejected(image_mode):
    config, raw, features = _config(), _raw(), _features()
    plan = negotiate_image_transport("pi05", config, features, _Identity(), mode=image_mode)
    encoded, _, _ = encode_observation_images(raw, plan)
    with pytest.raises(ValueError, match="without a negotiated"):
        prepare_remote_observation(encoded, features, config.image_features)
    packed = encoded["top"]
    invalid = [
        replace(packed, data=packed.data[:-1]),
        replace(packed, data=packed.data + b"extra"),
        replace(packed, data=zlib.compress(b"\0" * (3 * 224 * 224 * 4 + 1))),
        replace(packed, spec=ImageResizeSpec((3, 480, 640), (384, 384))),
        replace(
            packed,
            spec=replace(
                packed.spec, mode="openpi_uint8" if image_mode == "lerobot_float32" else "lerobot_float32"
            ),
        ),
    ]
    for value in invalid:
        with pytest.raises(ValueError):
            prepare_remote_observation({**encoded, "top": value}, features, config.image_features, plan)


@pytest.mark.parametrize("mode", ["guided", "trained"])
@pytest.mark.parametrize("image_mode", IMAGE_TRANSPORT_MODES)
def test_setup_upload_and_both_rtc_inference_paths_preserve_vision_input(mode, image_mode, monkeypatch):
    config, raw, features = _config(), _raw(), _features()
    observed = []

    class Policy:
        def __init__(self):
            self.config = config

        def supports_rtc(self):
            return True

        def eval(self):
            return self

        def reset(self):
            pass

        def predict_action_chunk(self, observation, *, inference_delay, prev_chunk_left_over, **kwargs):
            observed.append(_vision_input(config, observation))
            return torch.zeros(1, 50, 1)

    server = PolicyServerDeployment(
        PolicyServerDeploymentConfig(
            rtc=RTCConfig(enabled=True, mode=mode),
            inference_latency=0,
        )
    )

    def load(specs):
        server.policy, server.preprocessor, server.postprocessor = Policy(), _Identity(), _Identity()
        server.lerobot_features, server.device, server.actions_per_chunk = features, "cpu", 50
        return False

    monkeypatch.setattr(server, "_load_or_reuse_policy", load)
    metadata = []
    setup = RemoteRTCPolicySetup(
        policy=RemotePolicyConfig("pi05", "server-only-checkpoint", features, 50, "cpu"),
        control_frequency_hz=30,
        protocol_version=DEPLOYMENT_RTC_PROTOCOL_VERSION,
        rtc_mode=mode,
        trained_queue_threshold=25 if mode == "trained" else None,
        client_image_resize=True,
        image_transport_mode=image_mode,
    )
    # Exercise the client's actual start/serialization without connecting a robot.
    client = object.__new__(_RTCClient)
    client.policy_setup, client.shutdown_event = setup, threading.Event()
    client.config = SimpleNamespace(request_timeout_s=5, debug_rtc_diagnostics=False, log_rtc_timing=False)
    client.logger = SimpleNamespace(info=lambda *a: None, error=lambda *a: None)
    client._on_observation_transport_phase = lambda *a: None
    captured = []

    def setup_call(request):
        response = server.SendPolicyInstructions(
            request, SimpleNamespace(send_initial_metadata=metadata.extend)
        )
        return response, SimpleNamespace(initial_metadata=lambda: metadata)

    def upload(stream, **kwargs):
        captured.append(pickle.loads(b"".join(part.data for part in stream)))  # nosec

    client.stub = SimpleNamespace(
        Ready=lambda _: None,
        SendPolicyInstructions=SimpleNamespace(with_call=setup_call),
        SendObservations=upload,
    )
    request = RemoteRTCObservation(
        observation=TimedObservation(time.time(), 0, raw),
        observation_id=0,
        executed_action_step=-1,
        trained_prefix=torch.empty(0, 1) if mode == "trained" else None,
    )
    try:
        assert client.start()
        assert client._image_transport_plan["top"].model_resolution == (224, 224)
        assert client._image_transport_plan["top"].mode == image_mode
        assert client.send_observation(request)
        assert isinstance(captured[0].observation.observation["top"], PackedPolicyImage)
        assert captured[0].observation.timestamp == request.observation.timestamp
        assert captured[0].executed_action_step == request.executed_action_step
        assert isinstance(request.observation.observation["top"], np.ndarray)
        result = server._predict_remote_chunk(captured[0])
        assert isinstance(result, RemoteRTCActionChunk)
        if image_mode == "openpi_uint8":
            # Pixel parity with OpenPI is covered above; here verify both RTC
            # routes feed that exact prepared image to the policy.
            baseline = prepare_remote_observation(
                captured[0].observation.observation,
                features,
                config.image_features,
                client._image_transport_plan,
            )
        else:
            baseline = raw_observation_to_observation(raw, features, config.image_features)
        assert torch.equal(observed[0], _vision_input(config, baseline))
    finally:
        server.stop()


@pytest.mark.parametrize("image_mode", IMAGE_TRANSPORT_MODES)
def test_metadata_roundtrip_and_legacy_server_fallback(image_mode):
    plan = {"top": ImageResizeSpec((3, 480, 640), (224, 224), mode=image_mode)}
    assert (
        parse_image_transport_metadata(((IMAGE_TRANSPORT_METADATA_KEY, image_transport_metadata(plan)),))
        == plan
    )
    assert parse_image_transport_metadata(()) == {}
    raw = _raw()
    encoded, _, _ = encode_observation_images(raw, {})
    assert encoded["top"] is raw["top"]


@pytest.mark.parametrize("server_mode", [None, "lerobot_float32", "empty_openpi", "unknown"])
def test_client_rejects_unacknowledged_openpi_mode_before_upload(server_mode):
    metadata = []
    if server_mode is not None:
        payload = json.loads(image_transport_metadata({"top": ImageResizeSpec((3, 480, 640), (224, 224))}))
        if server_mode == "empty_openpi":
            payload = {"codec": "pi05-openpi-u8-v1", "images": {}}
        elif server_mode == "unknown":
            payload["codec"] = "unsupported-codec"
        metadata.append((IMAGE_TRANSPORT_METADATA_KEY, json.dumps(payload)))
    client = object.__new__(_RTCClient)
    client.policy_setup = RemoteRTCPolicySetup(
        RemotePolicyConfig("pi05", "unused", _features(), 50),
        30,
        client_image_resize=True,
        image_transport_mode="openpi_uint8",
    )
    client.stub = SimpleNamespace(
        SendPolicyInstructions=SimpleNamespace(
            with_call=lambda _: (None, SimpleNamespace(initial_metadata=lambda: metadata))
        )
    )
    with pytest.raises(ValueError, match="server|codec"):
        client._send_policy_instructions(None)


def test_default_metadata_retains_legacy_wire_format():
    metadata = image_transport_metadata({"top": ImageResizeSpec((3, 480, 640), (224, 224))})
    assert json.loads(metadata) == {
        "codec": "pi05-f32-shuffle-zlib-v1",
        "images": {"top": {"feature_shape": [3, 480, 640], "model_resolution": [224, 224]}},
    }
    assert parse_image_transport_metadata((), expected_mode="lerobot_float32") == {}
