# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Timestamped peeks use fake buffers only; never connect to a camera or CAN bus."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.zmq.camera_zmq import ZMQCamera
from lerobot.robots.piper_follower.piper_follower_PyAgxArm import PIPERFollower
from lerobot.utils.errors import DeviceNotConnectedError


@pytest.fixture(params=[OpenCVCamera, RealSenseCamera, ZMQCamera])
def camera(request, monkeypatch):
    cls = request.param
    monkeypatch.setattr(cls, "is_connected", property(lambda self: True))
    cam = object.__new__(cls)
    cam.index_or_path = "fake"
    cam.serial_number = "fake"
    cam.camera_name = "fake"
    cam.server_address = "fake"
    cam.port = 0
    cam.use_rgb = True
    cam.thread = SimpleNamespace(is_alive=lambda: True)
    cam.frame_lock = threading.Lock()
    cam.new_frame_event = threading.Event()
    cam.new_frame_event.set()
    cam.latest_frame = np.ones((4, 6, 3), dtype=np.uint8)
    cam.latest_color_frame = cam.latest_frame
    cam.latest_depth_frame = np.full((4, 6, 1), 123, dtype=np.uint16)
    cam.latest_timestamp = time.perf_counter()
    return cam


def test_peek_preserves_camera_timestamp_and_recording_event(camera):
    first_frame, first_time = camera.read_latest_with_timestamp()
    second_frame, second_time = camera.read_latest_with_timestamp()
    assert first_frame is second_frame is camera.latest_frame
    assert first_time == second_time == camera.latest_timestamp
    assert camera.new_frame_event.is_set()
    assert camera.read_latest() is first_frame
    if isinstance(camera, RealSenseCamera):
        assert camera._read_latest(500, read_depth=True) is camera.latest_depth_frame


def test_peek_rejects_stalled_camera_instead_of_redating_frame(camera):
    camera.latest_timestamp = time.perf_counter() - 1
    for _ in range(2):
        with pytest.raises(TimeoutError, match="too old"):
            camera.read_latest_with_timestamp(max_age_ms=100)
    assert camera.new_frame_event.is_set()


def test_peek_rejects_unavailable_camera(camera, monkeypatch):
    camera.thread = None
    with pytest.raises(RuntimeError, match="not running"):
        camera.read_latest_with_timestamp()
    monkeypatch.setattr(type(camera), "is_connected", property(lambda self: False))
    with pytest.raises(DeviceNotConnectedError):
        camera.read_latest_with_timestamp()


def test_piper_latest_observation_does_not_consume_camera_event(camera):
    robot = object.__new__(PIPERFollower)
    robot._is_connected = True
    robot.cameras = {"front": camera, "wrist": camera}
    robot.bus = SimpleNamespace(
        read_follower=lambda: "feedback",
        read_action_positions=lambda _: {"joint_1.pos": 0.25},
    )
    observation, times = robot.get_observation_with_timestamps()
    assert observation["joint_1.pos"] == 0.25
    assert observation["front"] is camera.latest_frame
    assert times == {"front": camera.latest_timestamp, "wrist": camera.latest_timestamp}
    assert camera.new_frame_event.is_set()


def test_piper_propagates_stale_camera_error(camera):
    robot = object.__new__(PIPERFollower)
    robot._is_connected = True
    camera.latest_timestamp = time.perf_counter() - 1
    robot.cameras = {"front": camera}
    with pytest.raises(TimeoutError):
        robot.get_observation_with_timestamps(max_age_ms=100)
