# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Exercise the public image-mode CLI on all deployment client configurations."""

from dataclasses import replace

import draccus
import pytest

from lerobot.async_inference.client.robot_client_dagger import RobotClientDaggerConfig
from lerobot.async_inference.client.robot_client_deployment import (
    RobotClientDeployment,
    RobotClientDeploymentConfig,
)
from lerobot.async_inference.client.robot_client_recap import RobotClientRecapConfig
from lerobot.async_inference.shared.image_transport import IMAGE_TRANSPORT_MODES
from tests.mocks.mock_robot import MockRobotConfig


@pytest.mark.parametrize(
    "config_class", [RobotClientDeploymentConfig, RobotClientDaggerConfig, RobotClientRecapConfig]
)
@pytest.mark.parametrize("mode", [None, *IMAGE_TRANSPORT_MODES])
def test_image_mode_cli_is_inherited_and_recorded(config_class, mode, tmp_path):
    args = [
        "--robot.type=mock_robot",
        f"--robot.calibration_dir={tmp_path}",
        "--policy_type=pi05",
        "--pretrained_name_or_path=unused-checkpoint",
        "--actions_per_chunk=50",
    ]
    if config_class is RobotClientRecapConfig:
        args += ["--recap_dataset_repo_id=test/images", f"--recap_dataset_root={tmp_path / 'recap'}"]
    if mode is not None:
        args.append(f"--image_transport_mode={mode}")
    config = draccus.parse(config_class, args=args)
    assert config.image_transport_mode == (mode or "lerobot_float32")
    assert config.to_dict()["image_transport_mode"] == config.image_transport_mode
    assert config.resize_images_on_client


def test_image_mode_is_validated_and_carried_into_policy_setup(tmp_path):
    config = RobotClientDeploymentConfig(
        policy_type="pi05",
        pretrained_name_or_path="unused-checkpoint",
        actions_per_chunk=50,
        robot=MockRobotConfig(calibration_dir=tmp_path),
        image_transport_mode="openpi_uint8",
    )
    with pytest.raises(ValueError, match="image_transport_mode"):
        replace(config, image_transport_mode="unit8")
    with pytest.raises(ValueError, match="policy_type=pi05"):
        replace(config, policy_type="smolvla")
    client = RobotClientDeployment(config)
    try:
        assert client.policy_setup.client_image_resize
        assert client.policy_setup.image_transport_mode == "openpi_uint8"
    finally:
        client.stop()
    client = RobotClientDeployment(replace(config, resize_images_on_client=False))
    try:
        assert not client.policy_setup.client_image_resize
    finally:
        client.stop()
