"""Prepare owned camera buffers directly, without a serialization round trip."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from lerobot.utils.constants import OBS_STATE


def resize_openpi(image: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Match OpenPI PIL bilinear resizing and centered uint8 padding."""
    if image.dtype == np.float32:
        if not np.isfinite(image).all() or np.any(image < 0) or np.any(image > 1):
            raise ValueError("Float camera images must be finite and in [0, 1]")
        image = (image * 255).astype(np.uint8)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("Camera images must be HWC RGB uint8 or float32")
    if image.shape[:2] == resolution:
        return image
    height, width = resolution
    source = Image.fromarray(image)
    ratio = max(source.width / width, source.height / height)
    size = (int(source.width / ratio), int(source.height / ratio))
    if min(size) < 1:
        raise ValueError("Image aspect ratio produces an empty resized dimension")
    resized = source.resize(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, height), 0)
    canvas.paste(resized, ((width - size[0]) // 2, (height - size[1]) // 2))
    return np.asarray(canvas)


class ObservationPreparer:
    """Prepare images once and leave state tokenization on the CPU."""

    def __init__(self, config, policy_config, observation_features: dict):
        """Resolve camera mapping, state order and reusable input staging buffers."""
        self.mode = config.inference.image_mode
        self.device = torch.device(config.inference.device)
        self.resolution = tuple(policy_config.image_resolution)
        self.state_keys = config.state_keys or [
            key for key, value in observation_features.items() if value is float
        ]
        if any(key not in observation_features for key in self.state_keys):
            raise ValueError("state_keys contains an unknown robot feature")
        expected_state = policy_config.input_features.get(OBS_STATE)
        if expected_state is None or expected_state.shape[-1] != len(self.state_keys):
            raise ValueError("Robot state dimension differs from checkpoint; specify the correct state_keys")
        self.images = {}
        for key, value in observation_features.items():
            if isinstance(value, tuple):
                source = f"observation.images.{key}"
                destination = config.rename_map.get(source, source)
                if destination in policy_config.image_features:
                    self.images[key] = (source, policy_config.image_features[destination])
        mapped = {config.rename_map.get(source, source) for source, _ in self.images.values()}
        if len(mapped) != len(self.images):
            raise ValueError("Multiple robot cameras map to the same checkpoint image feature")
        empty_cameras = {
            f"observation.images.empty_camera_{index}"
            for index in range(getattr(policy_config, "empty_cameras", 0))
        }
        required = set(policy_config.image_features) - empty_cameras
        if not required.issubset(mapped) or not self.images:
            raise ValueError("Checkpoint camera features do not match robot cameras/rename_map")
        # Reused pinned host staging is owned exclusively by the inference worker.
        self._staging: dict[str, torch.Tensor] = {}

    def __call__(self, values: dict, task: str) -> dict:
        """Build a single-observation batch without modifying capture buffers."""
        state = np.asarray([values[key] for key in self.state_keys], dtype=np.float32)
        if not np.isfinite(state).all():
            raise ValueError("Non-finite observation state")
        # Keep state/tokenization on CPU until the checkpoint's final device step.
        batch = {OBS_STATE: torch.from_numpy(state).unsqueeze(0), "task": [task]}
        for key, (feature_name, feature) in self.images.items():
            raw = values[key]
            if self.mode == "openpi_uint8":
                pixels = resize_openpi(raw, self.resolution)
                staging = self._staging.get(key)
                if staging is None or tuple(staging.shape) != pixels.shape:
                    staging = torch.empty(
                        pixels.shape, dtype=torch.uint8, pin_memory=self.device.type == "cuda"
                    )
                    self._staging[key] = staging
                np.copyto(staging.numpy(), pixels)
                image = staging.to(self.device, non_blocking=self.device.type == "cuda")
                image = image.permute(2, 0, 1).to(torch.float32).div_(255).unsqueeze(0)
            else:
                if (
                    not isinstance(raw, np.ndarray)
                    or raw.ndim != 3
                    or raw.shape[-1] != 3
                    or raw.dtype not in (np.uint8, np.float32)
                    or min(raw.shape[:2]) < 1
                ):
                    raise ValueError("Camera images must be HWC RGB arrays")
                if raw.dtype == np.float32 and (
                    not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1)
                ):
                    raise ValueError("Float camera images must be finite and in [0, 1]")
                # Preserve the previous float32 preparation's integer resize rounding.
                image = torch.tensor(raw).permute(2, 0, 1).unsqueeze(0)
                image = torch.nn.functional.interpolate(
                    image, size=feature.shape[1:], mode="bilinear", align_corners=False
                )
                if image.dtype == torch.uint8:
                    image = image.float().div_(255)
                # The saved processor must see checkpoint feature dimensions.
                # PI05 performs model-resolution resize/pad after that pipeline;
                # moving it earlier changes non-identity image transforms.
            batch[feature_name] = image.contiguous()
        return batch
