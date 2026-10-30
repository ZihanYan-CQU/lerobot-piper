# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Ensure production paths bypass diagnostic work, not just its log output."""

import logging
import time
from types import SimpleNamespace

import pytest
import torch

from lerobot.async_inference.server.policy_server_deployment import (
    PolicyServerDeployment,
    PolicyServerDeploymentConfig,
)
from lerobot.async_inference.shared.deployment_diagnostics import configure_deployment_logging
from lerobot.async_inference.shared.helpers import TimedObservation
from lerobot.async_inference.shared.rtc_remote import RemoteRTCObservation, encode_rollout_observation_id
from tests.async_inference.test_policy_server_deployment import _FakePipeline, _FakePolicy
from tests.async_inference.test_robot_client_deployment import _chunk, _make_client


def _forbidden(*args, **kwargs):
    pytest.fail("disabled diagnostics executed optional work")


def test_command_dispatch_without_debug_allocates_no_diagnostic_tensor(tmp_path, monkeypatch):
    client = _make_client(tmp_path)
    try:
        assert not client.config.debug_rtc_diagnostics
        assert not client.config.log_rtc_timing
        assert client.start_rollout()
        source = encode_rollout_observation_id(client.rollout_generation, 0)
        assert client._handle_action_chunk(_chunk(source, [0.0, 2.0, 4.0]))
        monkeypatch.setattr(client, "_record_command_dispatch", _forbidden)
        monkeypatch.setattr(client, "_max_abs", _forbidden)
        with monkeypatch.context() as patch:
            patch.setattr(torch, "tensor", _forbidden)
            values = [client.control_loop_action()["motor_1.pos"] for _ in range(5)]
        assert values == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert client.latest_action == 2
        assert client.command_diagnostics_snapshot() is None
        assert client.control_loop_action() is None
        assert client._starvation_started_at is None
    finally:
        client.stop()


def test_command_diagnostics_can_be_enabled_without_changing_dispatched_actions(tmp_path):
    client = _make_client(tmp_path)
    client.config.debug_rtc_diagnostics = True
    try:
        assert client.start_rollout()
        source = encode_rollout_observation_id(client.rollout_generation, 0)
        assert client._handle_action_chunk(_chunk(source, [0.0, 2.0, 4.0]))
        assert [client.control_loop_action()["motor_1.pos"] for _ in range(5)] == [0, 1, 2, 3, 4]
        snapshot = client.command_diagnostics_snapshot()
        assert snapshot.policy_step == client.latest_action == 2
        torch.testing.assert_close(snapshot.command, torch.full((3,), 4.0))
        torch.testing.assert_close(snapshot.delta, torch.ones(3))
    finally:
        client.stop()


@pytest.mark.parametrize("timing,debug", [(False, False), (True, False), (False, True)])
def test_transport_only_collects_phase_timestamps_when_enabled(tmp_path, monkeypatch, timing, debug):
    client = _make_client(tmp_path)
    client.config.log_rtc_timing = timing
    client.config.debug_rtc_diagnostics = debug
    phases = []
    uploads = []
    monkeypatch.setattr(
        client, "_on_observation_transport_phase", lambda _, phase, stamp: phases.append(phase)
    )
    monkeypatch.setattr(
        client, "stub", SimpleNamespace(SendObservations=lambda chunks, **kw: uploads.extend(chunks))
    )
    try:
        assert client.start_rollout()
        source = encode_rollout_observation_id(client.rollout_generation, 0)
        request = RemoteRTCObservation(TimedObservation(time.time(), 0, {}), source, -1)
        assert client.send_observation(request)
        assert uploads
        assert phases == (
            ["serialization_started", "serialization_finished", "upload_started", "upload_finished"]
            if timing or debug
            else []
        )
    finally:
        client.stop()


def test_server_without_debug_skips_phase_instrumentation_but_keeps_latency_and_actions(monkeypatch):
    server = PolicyServerDeployment(PolicyServerDeploymentConfig())
    server.policy = _FakePolicy("checkpoint")
    server.preprocessor = _FakePipeline()
    server.postprocessor = _FakePipeline()
    server.device = "cpu"
    server.actions_per_chunk = 20
    server.lerobot_features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]}
    }
    monkeypatch.setattr(server, "_logged_diagnostic_phase", _forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", _forbidden)
    monkeypatch.setattr(time, "sleep", _forbidden)
    try:
        result = server._predict_remote_chunk(
            RemoteRTCObservation(TimedObservation(time.time(), 0, {"joint.pos": 0.0}), 0, -1)
        )
        assert len(result.actions) == 20
        assert result.inference_seconds > 0
        assert server._current_latency() == result.inference_seconds
        assert server._chunk_cache.get(0) is not None
    finally:
        server.stop()


@pytest.mark.parametrize("debug", [False, True])
def test_process_logging_does_not_silently_write_library_debug_records(debug):
    root = logging.getLogger()
    previous = root.level
    child = logging.getLogger("deployment_diagnostics_off.child")
    try:
        configure_deployment_logging(debug=debug)
        assert child.isEnabledFor(logging.DEBUG) is debug
        assert child.isEnabledFor(logging.WARNING)
    finally:
        root.setLevel(previous)
