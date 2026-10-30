# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Regression coverage for opt-in diagnostics and the common server lifecycle."""

import logging
import threading
from types import SimpleNamespace

import pytest
import torch

import lerobot.async_inference.client.robot_client_deployment as client_module
import lerobot.async_inference.server.policy_server_deployment as server_module
from lerobot.async_inference.server.policy_server_dagger import PolicyServerDagger, PolicyServerDaggerConfig
from lerobot.async_inference.server.policy_server_deployment import (
    PolicyServerDeployment,
    PolicyServerDeploymentConfig,
)
from tests.async_inference.test_robot_client_dagger import _make_client as make_dagger
from tests.async_inference.test_robot_client_deployment import _chunk, _make_client as make_deployment


def test_disabled_splice_diagnostics_skip_copies_and_clear_previous_snapshot(tmp_path, monkeypatch):
    client = make_deployment(tmp_path)
    monkeypatch.setattr(client, "logger", logging.Logger("splice-diagnostics", level=logging.DEBUG))
    try:
        assert client.start_rollout()
        assert client._request_observation_if_ready()
        request = client._request_mailbox.get_nowait()
        client.config.debug_rtc_diagnostics = True
        assert client._handle_action_chunk(_chunk(request.observation_id, [0.0, 1.0, 2.0]))
        assert client.action_schedule.last_merge_diagnostics(request.observation_id) is not None
        client.control_loop_action()  # Endpoint 0 is immutable after dispatch.
        client.config.debug_rtc_diagnostics = False

        def forbidden(*args, **kwargs):
            pytest.fail("disabled splice diagnostics must not copy tensors or materialize the action plan")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", forbidden)
            patch.setattr(torch.Tensor, "clone", forbidden)
            patch.setattr(client.action_schedule, "_execution_plan_locked", forbidden)
            assert client._handle_action_chunk(_chunk(request.observation_id + 1, [9.0, 3.0, 4.0]))

        assert client.latest_action == 0
        assert client.action_schedule.last_merge_diagnostics(request.observation_id) is None
        assert client.action_schedule.last_merge_diagnostics(request.observation_id + 1) is None
        assert client.action_schedule.qsize(0) == 2
        endpoint = client.action_schedule.consume_next(0)
        torch.testing.assert_close(endpoint.timed_action.action, torch.full((3,), 3.0))
    finally:
        client.stop()


@pytest.mark.parametrize("dagger", [False, True])
@pytest.mark.parametrize("visualize", [False, True])
def test_control_loop_collects_queue_history_only_on_request(tmp_path, monkeypatch, dagger, visualize):
    client = make_dagger(tmp_path)[0] if dagger else make_deployment(tmp_path)
    client.config.debug_visualize_queue_size = visualize
    client.start_barrier = threading.Barrier(1)
    assert client.start_rollout()
    assert client._request_observation_if_ready()
    request = client._request_mailbox.get_nowait()
    assert client._handle_action_chunk(_chunk(request.observation_id, [0.0, 0.1, 0.2]))
    monkeypatch.setattr(client, "_request_observation_if_ready", lambda: False)

    class TwoTicks:
        def __init__(self, **kwargs):
            self.remaining = 2

        def tick(self, **kwargs):
            pass

        def wait(self):
            self.remaining -= 1
            if not self.remaining:
                client.shutdown_event.set()

        def log_run_summary(self):
            pass

    monkeypatch.setattr(client_module, "CycleTimer", TwoTicks)
    try:
        client.control_loop()
        assert client.worker_error is None
        assert client.latest_action == 0
        # DAgger seeds interpolation from the aligned arm pose, so endpoint 0
        # remains in flight on tick 1; deployment dispatches it immediately.
        expected = [3 if dagger else 2, 2] if visualize else []
        assert list(client.action_queue_size) == expected
    finally:
        client.stop()


@pytest.mark.parametrize(
    "server_class,config_class",
    [(PolicyServerDeployment, PolicyServerDeploymentConfig), (PolicyServerDagger, PolicyServerDaggerConfig)],
)
@pytest.mark.parametrize("fault_handler", [False, True])
def test_shared_server_lifecycle_cleans_up_on_termination_error(
    tmp_path, monkeypatch, server_class, config_class, fault_handler
):
    fault_path = tmp_path / "server_fault.log"
    service = server_class(
        config_class(
            host="localhost", port=9999, enable_fault_handler=fault_handler, fault_log_path=fault_path
        )
    )
    events = []

    def register(servicer, server):
        assert servicer is service and server is grpc_server
        events.append("register")

    def wait():
        assert service.running
        raise RuntimeError("termination interrupted")

    grpc_server = SimpleNamespace(
        add_insecure_port=lambda address: events.append(address),
        start=lambda: events.append("start"),
        wait_for_termination=wait,
    )
    monkeypatch.setattr(server_module.grpc, "server", lambda executor: grpc_server)
    monkeypatch.setattr(server_module.services_pb2_grpc, "add_AsyncInferenceServicer_to_server", register)
    try:
        with pytest.raises(RuntimeError, match="termination interrupted"):
            server_module._serve_policy_server(service)
        assert events == ["register", "localhost:9999", "start"]
        assert not service.running
        assert fault_path.exists() is fault_handler
        if fault_handler:
            assert "RTC runtime torch=" in fault_path.read_text()
    finally:
        service.stop()
