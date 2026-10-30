# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Negotiate Pi05 image preparation before upload.

The default preserves LeRobot's float32 pixels with lossless compression.
The opt-in OpenPI mode uses PIL resize/pad and uncompressed RGB uint8 pixels.
Both use the loaded checkpoint's model resolution and bypass server resizing.
"""

import json
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.configs import FeatureType, NormalizationMode
from lerobot.policies.common.vla_utils import resize_with_pad_torch
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
)
from lerobot.utils.constants import OBS_IMAGES

from .helpers import prepare_image, raw_observation_to_observation, resize_robot_observation_image

IMAGE_TRANSPORT_METADATA_KEY = "lerobot-image-transport"
_CODECS = {
    "lerobot_float32": "pi05-f32-shuffle-zlib-v1",
    "openpi_uint8": "pi05-openpi-u8-v1",
}
IMAGE_TRANSPORT_MODES = tuple(_CODECS)
_IMAGE_PREFIX = f"{OBS_IMAGES}."
_KNOWN_PROCESSORS = {
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    Pi05PrepareStateTokenizerProcessorStep,
}


@dataclass(frozen=True)
class ImageResizeSpec:
    feature_shape: tuple[int, int, int]
    model_resolution: tuple[int, int]
    mode: str = "lerobot_float32"

    def __post_init__(self) -> None:
        if self.mode not in IMAGE_TRANSPORT_MODES:
            raise ValueError(f"unsupported image_transport_mode: {self.mode!r}")
        if (
            len(self.feature_shape) != 3
            or self.feature_shape[0] != 3
            or len(self.model_resolution) != 2
            or any(
                type(v) is not int or not 0 < v <= 8192 for v in (*self.feature_shape, *self.model_resolution)
            )
        ):
            raise ValueError("invalid negotiated RGB image dimensions")


@dataclass(frozen=True)
class PackedPolicyImage:
    spec: ImageResizeSpec
    data: bytes


def negotiate_image_transport(
    policy_type: str,
    config: Any,
    features: dict,
    preprocessor: Any,
    *,
    mode: str = "lerobot_float32",
) -> dict[str, ImageResizeSpec]:
    """Only bypass resize for a known, image-preserving Pi05 processor chain."""
    if mode not in IMAGE_TRANSPORT_MODES:
        raise ValueError(f"unsupported image_transport_mode: {mode!r}")
    if policy_type != "pi05" or getattr(config, "use_visual_memory", False):
        return {}
    resolution = getattr(config, "image_resolution", None)
    if resolution is None or getattr(preprocessor, "steps", None) is None:
        return {}
    for step in preprocessor.steps:
        if type(step) not in _KNOWN_PROCESSORS:
            return {}
        if (
            isinstance(step, NormalizerProcessorStep)
            and step.norm_map.get(FeatureType.VISUAL, NormalizationMode.IDENTITY)
            != NormalizationMode.IDENTITY
        ):
            return {}
        if isinstance(step, DeviceProcessorStep) and step.float_dtype not in (None, "float32", "float"):
            return {}
        if isinstance(step, RenameObservationsProcessorStep) and any(
            old != new and (old.startswith(_IMAGE_PREFIX) or new.startswith(_IMAGE_PREFIX))
            for old, new in step.rename_map.items()
        ):
            return {}
    result = {}
    for key, feature in features.items():
        if not key.startswith(_IMAGE_PREFIX) or feature.get("dtype") not in ("image", "video"):
            continue
        policy_feature = config.image_features.get(key)
        if policy_feature is None:
            return {}
        result[key.removeprefix(_IMAGE_PREFIX)] = ImageResizeSpec(
            tuple(policy_feature.shape), tuple(resolution), mode=mode
        )
    return result


def image_transport_metadata(plan: dict[str, ImageResizeSpec]) -> str:
    modes = {spec.mode for spec in plan.values()}
    if len(modes) > 1:
        raise ValueError("all negotiated images must use the same transport mode")
    mode = next(iter(modes), "lerobot_float32")
    # Keep the legacy float32 metadata layout; the codec identifies the mode.
    return json.dumps(
        {
            "codec": _CODECS[mode],
            "images": {
                key: {"feature_shape": spec.feature_shape, "model_resolution": spec.model_resolution}
                for key, spec in plan.items()
            },
        }
    )


def parse_image_transport_metadata(
    metadata: Any, *, expected_mode: str | None = None
) -> dict[str, ImageResizeSpec]:
    for key, value in metadata or ():
        if key != IMAGE_TRANSPORT_METADATA_KEY:
            continue
        payload = json.loads(value)
        mode = next((name for name, codec in _CODECS.items() if codec == payload["codec"]), None)
        if mode is None:
            raise ValueError("unsupported negotiated image transport codec")
        if expected_mode is not None and mode != expected_mode:
            raise ValueError(
                f"server negotiated {mode}, but client requested {expected_mode}; "
                "update both client and server"
            )
        plan = {
            name: ImageResizeSpec(tuple(spec["feature_shape"]), tuple(spec["model_resolution"]), mode=mode)
            for name, spec in payload["images"].items()
        }
        if plan:
            return plan
        break
    if expected_mode == "openpi_uint8":
        raise ValueError(
            "server did not negotiate openpi_uint8; update both client and server "
            "and use a supported Pi05 RGB preprocessing pipeline"
        )
    return {}


def _prepare_openpi_uint8(raw: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Match openpi_client.image_tools: PIL bilinear, floor sizing, centered black padding."""
    if raw.dtype == np.float32:
        if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1):
            raise ValueError("OpenPI float images must contain finite pixels in [0, 1]")
        raw = (raw * 255).astype(np.uint8)
    height, width = resolution
    if raw.shape[:2] == resolution:
        return raw
    image = Image.fromarray(raw)
    ratio = max(image.width / width, image.height / height)
    resized_width, resized_height = int(image.width / ratio), int(image.height / ratio)
    if min(resized_width, resized_height) < 1:
        raise ValueError("OpenPI resize would produce an empty image dimension")
    resized = image.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (width, height), 0)
    padded.paste(resized, ((width - resized_width) // 2, (height - resized_height) // 2))
    return np.asarray(padded)


def encode_observation_images(
    observation: dict[str, Any], plan: dict[str, ImageResizeSpec]
) -> tuple[dict[str, Any], int, int]:
    """Copy the upload dictionary, retaining original camera/recording buffers."""
    result = observation.copy()
    raw_bytes = wire_bytes = 0
    for key, spec in plan.items():
        raw = observation.get(key)
        if not isinstance(raw, np.ndarray) or raw.dtype not in (np.uint8, np.float32):
            if raw is not None and spec.mode == "openpi_uint8":
                raise ValueError(f"{key}: OpenPI images must be RGB uint8 or normalized float32 arrays")
            continue
        if raw.ndim != 3 or raw.shape[-1] != 3 or min(raw.shape[:2]) < 1:
            raise ValueError(f"{key}: expected an HWC RGB image, got {raw.shape}")
        raw_bytes += raw.nbytes
        if spec.mode == "openpi_uint8":
            pixels = _prepare_openpi_uint8(raw, spec.model_resolution)
            data = pixels.tobytes()
            # Always mark prepared images, even when raw and model sizes match:
            # checkpoint feature sizes may otherwise trigger a second resize.
            result[key] = PackedPolicyImage(spec, data)
            wire_bytes += len(data)
            continue
        if tuple(raw.shape[:2]) == spec.model_resolution == spec.feature_shape[1:]:
            wire_bytes += raw.nbytes
            continue
        # Replay the existing server helper first (including its uint8 rounding),
        # then Pi05's float32 centered resize/pad. Do not replace this with PIL.
        img = prepare_image(resize_robot_observation_image(torch.from_numpy(raw), spec.feature_shape))
        if tuple(img.shape[1:]) != spec.model_resolution:
            img = resize_with_pad_torch(img.permute(1, 2, 0).unsqueeze(0), *spec.model_resolution)
            img = img.squeeze(0).permute(2, 0, 1)
        pixels = np.asarray(img.contiguous().numpy(), dtype="<f4")
        shuffled = pixels.view(np.uint8).reshape(-1, 4).T.copy().tobytes()
        data = zlib.compress(shuffled, level=1)
        # An incompressible/small frame must not make its upload larger.
        if len(data) + 256 < raw.nbytes:
            result[key] = PackedPolicyImage(spec, data)
            wire_bytes += len(data)
        else:
            wire_bytes += raw.nbytes
    return result, raw_bytes, wire_bytes


def _decode_image(image: PackedPolicyImage, spec: ImageResizeSpec) -> torch.Tensor:
    if image.spec != spec:
        raise ValueError("uploaded image does not match the loaded checkpoint's resize specification")
    height, width = spec.model_resolution
    if spec.mode == "openpi_uint8":
        if len(image.data) != 3 * height * width:
            raise ValueError("invalid uint8 image payload size")
        pixels = np.frombuffer(image.data, dtype=np.uint8).reshape(height, width, 3).copy()
        return prepare_image(torch.from_numpy(pixels).permute(2, 0, 1)).unsqueeze(0)
    expected = 3 * height * width * 4
    decoder = zlib.decompressobj()
    data = decoder.decompress(image.data, expected + 1)
    if len(data) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("invalid lossless image payload size or stream")
    pixels = np.frombuffer(data, dtype=np.uint8).reshape(4, -1).T.copy().reshape(-1).view("<f4")
    if not np.isfinite(pixels).all() or np.any(pixels < 0) or np.any(pixels > 1):
        raise ValueError("prepared image must contain finite float32 pixels in [0, 1]")
    return torch.from_numpy(pixels.reshape(3, height, width)).unsqueeze(0)


def prepare_remote_observation(
    observation: dict[str, Any],
    features: dict,
    policy_image_features: dict,
    plan: dict[str, ImageResizeSpec] | None = None,
) -> dict:
    """Prepare raw images normally; negotiated images bypass both resize stages."""
    prepared = {}
    for key, value in observation.items():
        if isinstance(value, PackedPolicyImage):
            if not plan or key not in plan:
                raise ValueError("received a prepared image without a negotiated resize specification")
            prepared[f"{_IMAGE_PREFIX}{key}"] = _decode_image(value, plan[key])
    raw_features = {key: value for key, value in features.items() if key not in prepared}
    result = raw_observation_to_observation(observation, raw_features, policy_image_features)
    return {**result, **prepared}
