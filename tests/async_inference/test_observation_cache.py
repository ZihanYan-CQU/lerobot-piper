# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Latest-only capture and lifecycle tests, using mock hardware and blocked RPCs."""

import threading
import time
from dataclasses import replace
from queue import Empty, Queue

import numpy as np
import pytest
import torch

from lerobot.async_inference.client.robot_client_deployment import (
    RobotClientDeployment,
    RobotClientDeploymentConfig,
    _CapturedObservation,
    _InterpolatedClient,
)
from lerobot.async_inference.shared.rtc_trained import TrainedRTCClientConfig
from tests.async_inference.test_robot_client_dagger import _make_client as make_dagger_client
from tests.async_inference.test_trained_rtc_deployment import _chunk
from tests.mocks.mock_robot import MockRobotConfig


@pytest.fixture
def client_factory(tmp_path):
    clients = []

    def make(**overrides):
        client = RobotClientDeployment(
            RobotClientDeploymentConfig(
                policy_type="pi05",
                pretrained_name_or_path="checkpoint",
                robot=MockRobotConfig(calibration_dir=tmp_path, random_values=False, static_values=[0.0] * 3),
                actions_per_chunk=8,
                server_address="localhost:9999",
                interpolation_multiplier=2,
                trained=TrainedRTCClientConfig(queue_threshold=4),
                reset_to_initial_position=False,
                task="pick",
                **overrides,
            )
        )
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.stop()
        if client._observation_collector is not None:
            assert not client._observation_collector.is_alive()


def _wait_sample(client, predicate=lambda sample: True):
    with client._lifecycle_changed:
        assert client._lifecycle_changed.wait_for(
            lambda: client._latest_observation is not None and predicate(client._latest_observation),
            timeout=2,
        ), client.worker_error
        return client._latest_observation


def _start_sender(client):
    client.start_barrier = threading.Barrier(1)
    worker = threading.Thread(target=client.observation_sender)
    worker.start()
    return worker


def test_production_sender_skips_tracking_and_phase_measurement(client_factory, monkeypatch):
    client = client_factory(continuous_observation=False)
    uploads = Queue()

    def forbidden(*args, **kwargs):
        pytest.fail("disabled observation diagnostics ran")

    monkeypatch.setattr(client, "_log_observation_tracking", forbidden)
    monkeypatch.setattr(client, "command_diagnostics_snapshot", forbidden)
    monkeypatch.setattr(client, "_phase_ms", forbidden)
    monkeypatch.setattr(
        _InterpolatedClient, "send_observation", lambda self, envelope: uploads.put(envelope) or True
    )
    assert client.start_rollout()
    worker = _start_sender(client)
    try:
        assert client._request_observation_if_ready()
        request = uploads.get(timeout=2)
        timing = client._request_timings[request.observation_id]
        assert timing.snapshot_at is not None  # Required for monotonic RTT feedback.
        assert timing.capture_started_at is None
        assert timing.capture_finished_at is None
        assert timing.mailbox_dequeued_at is None
        assert timing.observation_selected_at is None
    finally:
        client.stop()
        worker.join(timeout=2)
        assert not worker.is_alive()


@pytest.mark.parametrize("mode", ["guided", "trained"])
def test_capture_continues_during_upload_and_next_request_uses_live_prefix(client_factory, monkeypatch, mode):
    client = client_factory(rtc_mode=mode)
    uploads = Queue()
    release_upload = threading.Event()
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    capture_count = 0

    def capture():
        nonlocal capture_count
        capture_count += 1
        image.fill(capture_count)
        return {**dict.fromkeys(client.robot.action_features, float(capture_count)), "camera": image}

    def upload(self, envelope):
        uploads.put(envelope)
        assert release_upload.wait(timeout=2)
        return True

    client.robot.get_observation = capture
    monkeypatch.setattr(_InterpolatedClient, "send_observation", upload)
    assert client.start_rollout()
    sender = _start_sender(client)
    try:
        first_sample = _wait_sample(client)
        # Capture is independent of any request, and does not upload on its own.
        _wait_sample(client, lambda sample: sample.sequence > first_sample.sequence)
        assert uploads.empty()
        assert client._request_observation_if_ready()
        first = uploads.get(timeout=2)
        first_value = first.observation.observation["motor_1.pos"]
        np.testing.assert_array_equal(first.observation.observation["camera"], first_value)
        newer = _wait_sample(client, lambda sample: sample.observation["motor_1.pos"] >= first_value + 3)
        # One inference owns the request slot even as the sensor cache is replaced.
        assert not client._request_observation_if_ready()
        assert client._inflight_request_id == first.observation_id
        assert "task" not in newer.observation
        np.testing.assert_array_equal(first.observation.observation["camera"], first_value)

        release_upload.set()
        assert client._handle_action_chunk(_chunk(first, mode=mode))
        for _ in range(8):
            client.control_loop_action()
        live = client._snapshot_execution_state()
        assert live.executed_action_step >= 0
        assert client._request_observation_if_ready()
        second = uploads.get(timeout=2)
        assert second.observation.observation["motor_1.pos"] >= newer.observation["motor_1.pos"]
        assert second.executed_action_step == live.executed_action_step
        if mode == "guided":
            assert second.action_refs == live.action_refs
        else:
            torch.testing.assert_close(second.trained_prefix, live.trained_prefix)
        assert uploads.empty()
    finally:
        release_upload.set()
        client.stop()
        sender.join(timeout=2)
        assert not sender.is_alive()


def test_repeated_camera_frame_keeps_its_age_until_fresh_frame_arrives(client_factory, monkeypatch):
    client = client_factory()
    uploads = Queue()
    fresh_frame = threading.Event()
    camera_time = time.perf_counter() - 1.0

    def capture(*, max_age_ms):
        # Deliberately tolerate stale frames here: client must check their real
        # timestamps again instead of dating them by each successful cache read.
        timestamp = time.perf_counter() if fresh_frame.is_set() else camera_time
        return dict.fromkeys(client.robot.action_features, 3.0), {"camera": timestamp}

    client.robot.get_observation_with_timestamps = capture
    client._timestamped_observation_supported = True
    monkeypatch.setattr(_InterpolatedClient, "send_observation", lambda self, req: uploads.put(req) or True)
    assert client.start_rollout()
    assert client._request_observation_if_ready()
    sender = _start_sender(client)
    try:
        sample = _wait_sample(client)
        later = _wait_sample(client, lambda value: value.sequence >= sample.sequence + 3)
        assert later.oldest_camera_at == pytest.approx(sample.oldest_camera_at, abs=0.001)
        assert later.age(time.monotonic()) >= 1.0
        assert uploads.empty()
        fresh_frame.set()
        assert uploads.get(timeout=2).observation.observation["motor_1.pos"] == 3.0
    finally:
        client.stop()
        sender.join(timeout=2)


def test_recent_frame_from_before_reset_boundary_is_not_admitted(client_factory):
    client = client_factory()
    assert client.start_rollout()
    now = time.monotonic()
    sample = _CapturedObservation(
        generation=client.rollout_generation,
        sequence=1,
        observation={},
        started_at=now,
        finished_at=now,
        started_action_step=-1,
        oldest_camera_at=client._observation_valid_after - 0.001,
    )
    with client._lifecycle_lock:
        assert sample.age(now) < client.config.observation_max_age_s
        assert not client._observation_is_usable_locked(sample)
        assert client._observation_is_usable_locked(replace(sample, oldest_camera_at=now))


def test_slow_capture_age_starts_before_blocking_read(client_factory):
    client = client_factory(observation_max_age_s=0.01, observation_capture_fps=1)
    assert client.start_rollout()
    started = threading.Event()
    release = threading.Event()

    def capture():
        started.set()
        assert release.wait(timeout=2)
        return dict.fromkeys(client.robot.action_features, 1.0)

    client.robot.get_observation = capture
    client._start_observation_collector()
    try:
        assert started.wait(timeout=2)
        time.sleep(0.02)
        release.set()
        sample = _wait_sample(client)
        assert sample.finished_at - sample.started_at >= 0.02
        with client._lifecycle_lock:
            assert not client._observation_is_usable_locked(sample)
    finally:
        release.set()


def test_on_demand_opt_out_does_not_start_collector(client_factory, monkeypatch):
    client = client_factory(continuous_observation=False)
    uploads = Queue()
    count = 0

    def capture():
        nonlocal count
        count += 1
        return dict.fromkeys(client.robot.action_features, 1.0)

    client.robot.get_observation = capture
    monkeypatch.setattr(_InterpolatedClient, "send_observation", lambda self, req: uploads.put(req) or True)
    assert client.start_rollout()
    sender = _start_sender(client)
    try:
        assert client._request_observation_if_ready()
        uploads.get(timeout=2)
        assert count == 1
        assert client._observation_collector is None
    finally:
        client.stop()
        sender.join(timeout=2)


def test_sender_reselects_newer_sample_when_prefix_lock_is_acquired(client_factory, monkeypatch):
    client = client_factory()
    uploads = Queue()
    assert client.start_rollout()
    now = time.monotonic()
    first = _CapturedObservation(
        client.rollout_generation, 1, dict.fromkeys(client.robot.action_features, 1.0), now, now, -1
    )
    newer = replace(first, sequence=2, observation=dict.fromkeys(client.robot.action_features, 2.0))
    client._latest_observation = first
    original_select = client._observation_for_request

    def select_then_refresh(request):
        sample = original_select(request)
        with client._lifecycle_lock:
            client._latest_observation = newer
        return sample

    monkeypatch.setattr(client, "_start_observation_collector", lambda: None)
    monkeypatch.setattr(client, "_observation_for_request", select_then_refresh)
    monkeypatch.setattr(_InterpolatedClient, "send_observation", lambda self, req: uploads.put(req) or True)
    assert client._request_observation_if_ready()
    sender = _start_sender(client)
    try:
        assert uploads.get(timeout=2).observation.observation["motor_1.pos"] == 2.0
        assert client._last_used_observation_sequence == 2
    finally:
        client.stop()
        sender.join(timeout=2)


def test_unsupported_timestamp_api_falls_back_once(client_factory):
    client = client_factory()
    peeks = []

    def unsupported(**kwargs):
        peeks.append(kwargs)
        raise NotImplementedError

    client.robot.get_observation_with_timestamps = unsupported
    client._timestamped_observation_supported = True
    assert client.start_rollout()
    client._start_observation_collector()
    sample = _wait_sample(client)
    later = _wait_sample(client, lambda value: value.sequence > sample.sequence)
    assert len(peeks) == 1
    assert later.oldest_camera_at is None
    with client._lifecycle_lock:
        assert client._observation_is_usable_locked(later)


def test_unexpected_collector_failure_wakes_owner(client_factory, monkeypatch):
    client = client_factory()

    def fail():
        raise RuntimeError("collector failed")

    monkeypatch.setattr(client, "_collect_observations", fail)
    client._start_observation_collector()
    client._observation_collector.join(timeout=2)
    assert not client._observation_collector.is_alive()
    assert client.shutdown_event.is_set()
    assert "collector failed" in client.worker_error


def test_collector_pauses_during_dagger_intervention_and_discards_old_capture(tmp_path, monkeypatch):
    client, _, _ = make_dagger_client(tmp_path)
    started = threading.Event()
    release = threading.Event()
    reads = Queue()

    def capture():
        reads.put(client.rollout_generation)
        started.set()
        assert release.wait(timeout=2)
        return dict.fromkeys(client.robot.action_features, 1.0)

    client._follower_robot.get_observation = capture
    try:
        client._start_observation_collector()
        assert client._latest_observation is None
        assert client.start_rollout()
        assert started.wait(timeout=2)
        reads.get(timeout=2)
        assert client.pause_autonomy()
        release.set()
        with pytest.raises(Empty):
            reads.get(timeout=0.15)
        assert client._latest_observation is None
        assert client.start_correction()
        with pytest.raises(Empty):
            reads.get(timeout=0.1)
        assert client._finish_correction(save=False)
        monkeypatch.setattr(client, "_reset_robot_pose", lambda: True)
        assert client.reset_rollout()
        assert client.start_rollout()
        sample = _wait_sample(client)
        assert sample.generation == client.rollout_generation
    finally:
        release.set()
        client.stop()
        assert not client._observation_collector.is_alive()


@pytest.mark.parametrize("name", ["observation_capture_fps", "observation_max_age_s"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_observation_config_rejects_invalid_values(client_factory, name, value):
    with pytest.raises(ValueError, match=name):
        client_factory(**{name: value})
