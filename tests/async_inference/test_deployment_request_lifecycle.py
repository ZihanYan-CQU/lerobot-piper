# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Single-request refill, diagnostic lifetime and deployment-only configuration."""

import inspect
import time
import weakref
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

import lerobot.async_inference.client.robot_client_dagger as dagger_module
import lerobot.async_inference.client.robot_client_deployment as deployment_module
from lerobot.async_inference.client.robot_client_dagger import RobotClientDaggerConfig
from lerobot.async_inference.client.robot_client_deployment import (
    RobotClientDeploymentConfig,
    _InterpolatedClient,
)
from lerobot.async_inference.server.policy_server_dagger import PolicyServerDaggerConfig
from lerobot.async_inference.server.policy_server_deployment import PolicyServerDeploymentConfig
from lerobot.async_inference.shared.configs import PolicyServerConfig, RobotClientConfig
from lerobot.async_inference.shared.helpers import TimedObservation
from lerobot.async_inference.shared.rtc_remote import RemoteRTCObservation
from tests.async_inference.test_robot_client_dagger import _make_client as make_dagger
from tests.async_inference.test_robot_client_deployment import _chunk, _make_client as make_deployment
from tests.mocks.mock_robot import MockRobotConfig


@pytest.mark.parametrize("dagger", [False, True])
def test_five_identical_observations_refill_without_delay_or_retained_images(tmp_path, monkeypatch, dagger):
    client = make_dagger(tmp_path)[0] if dagger else make_deployment(tmp_path)
    uploads = []

    def upload(self, envelope):
        assert envelope.observation.must_go
        uploads.append(envelope.observation_id)
        return True

    monkeypatch.setattr(_InterpolatedClient, "send_observation", upload)
    try:
        assert client.start_rollout()
        for _ in range(5):
            # No execution, waiting or cooldown tick is needed between responses
            # while the short queue remains below the refill threshold.
            assert client._request_observation_if_ready()
            request = client._request_mailbox.get_nowait()
            snapshot = client._snapshot_execution_state()
            observation = TimedObservation(time.time(), 0, client.robot.get_observation())
            reference = weakref.ref(observation)
            envelope = RemoteRTCObservation(
                observation,
                request.observation_id,
                snapshot.executed_action_step,
                action_refs=snapshot.action_refs,
            )
            assert client.send_observation(envelope)
            with monkeypatch.context() as patch:
                patch.setattr(
                    client.action_schedule,
                    "qsize",
                    lambda _: pytest.fail("an occupied request slot should bypass the queue scan"),
                )
                assert not client._request_observation_if_ready()
            assert client._handle_action_chunk(_chunk(request.observation_id, [0.0, 0.0]))
            assert client._inflight_request_id is None
            del envelope, observation
            assert reference() is None, "completed observations must not remain cached as filter baselines"
        assert len(uploads) == len(set(uploads)) == 5
        assert uploads == sorted(uploads)
    finally:
        client.stop()


@pytest.mark.parametrize("dagger", [False, True])
def test_queue_history_is_bounded_and_survives_pause_reset_and_stop(tmp_path, monkeypatch, dagger):
    monkeypatch.setattr(deployment_module, "_MAX_QUEUE_HISTORY_SAMPLES", 3)
    client = make_dagger(tmp_path)[0] if dagger else make_deployment(tmp_path)
    try:
        assert client.start_rollout()
        client.action_queue_size.extend(range(6))
        assert list(client.action_queue_size) == [3, 4, 5]
        if dagger:
            assert client.pause_autonomy()
        assert client.reset_rollout()
        assert list(client.action_queue_size) == [3, 4, 5]
        client.stop()
        assert list(client.action_queue_size) == [3, 4, 5]
    finally:
        client.stop()


@pytest.mark.parametrize("dagger", [False, True])
def test_entrypoint_plots_collected_history_after_hardware_shutdown(tmp_path, monkeypatch, dagger):
    module = dagger_module if dagger else deployment_module
    client = make_dagger(tmp_path)[0] if dagger else make_deployment(tmp_path)
    client.config.debug_visualize_queue_size = True
    events = []
    stop = client.stop

    def tracked_stop():
        stop()
        events.append("stop")

    def plot(values):
        assert not client.running
        assert not client.robot.is_connected
        assert values == [3, 2, 1]
        events.append("plot")

    monkeypatch.setattr(
        module, "RobotClientDagger" if dagger else "RobotClientDeployment", lambda cfg: client
    )
    monkeypatch.setattr(client, "start", lambda: True)
    monkeypatch.setattr(client, "stop", tracked_stop)
    for name in ("receive_actions", "observation_sender", "control_loop"):
        monkeypatch.setattr(client, name, lambda: None)
    monkeypatch.setattr(
        module, "_run_interactive_prompt", lambda *_: client.action_queue_size.extend([3, 2, 1])
    )
    monkeypatch.setattr(module, "visualize_action_queue_size", plot)
    if dagger:
        client.config.dagger_input_device = "prompt"
        monkeypatch.setattr(
            module, "_make_collection_progress", lambda _: SimpleNamespace(close=lambda: None)
        )
        monkeypatch.setattr(module, "_refresh_collection_progress", lambda *_: None)
        monkeypatch.setattr(module, "_monitor_collection_progress", lambda *_: None)
    try:
        inspect.unwrap(module.async_client)(client.config)
        assert events == ["stop", "plot"]
    finally:
        stop()


def _client_kwargs(tmp_path):
    return {
        "policy_type": "smolvla",
        "pretrained_name_or_path": "checkpoint",
        "robot": MockRobotConfig(calibration_dir=tmp_path),
        "actions_per_chunk": 8,
    }


@pytest.mark.parametrize("config_class", [RobotClientDeploymentConfig, RobotClientDaggerConfig])
@pytest.mark.parametrize(
    "removed,value",
    [("aggregate_fn_name", "average"), ("observation_similarity_atol", 1.0), ("cooldown_margin_steps", 1)],
)
def test_deployment_client_rejects_removed_options(tmp_path, config_class, removed, value):
    config = config_class(**_client_kwargs(tmp_path))
    assert removed not in asdict(config)
    assert removed not in config.to_dict()
    with pytest.raises(TypeError, match=removed):
        config_class(**_client_kwargs(tmp_path), **{removed: value})


@pytest.mark.parametrize("config_class", [PolicyServerDeploymentConfig, PolicyServerDaggerConfig])
def test_deployment_server_uses_action_wait_timeout_without_legacy_observation_timeout(config_class):
    config = config_class(action_wait_timeout=0.5)
    assert config.action_wait_timeout == 0.5
    assert "obs_queue_timeout" not in asdict(config)
    with pytest.raises(TypeError, match="obs_queue_timeout"):
        config_class(obs_queue_timeout=0.5)


def test_legacy_async_options_still_control_the_legacy_configurations(tmp_path):
    client = RobotClientConfig(**_client_kwargs(tmp_path), aggregate_fn_name="average")
    torch.testing.assert_close(client.aggregate_fn(torch.zeros(3), torch.ones(3)), torch.full((3,), 0.5))
    assert client.to_dict()["aggregate_fn_name"] == "average"
    assert PolicyServerConfig(obs_queue_timeout=2.5).obs_queue_timeout == 2.5
    with pytest.raises(ValueError, match="Unknown aggregate function"):
        RobotClientConfig(**_client_kwargs(tmp_path), aggregate_fn_name="missing")
    with pytest.raises(ValueError, match="obs_queue_timeout"):
        PolicyServerConfig(obs_queue_timeout=-1)
