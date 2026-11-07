"""Deployment entrypoint delegation, single-owner transport and motor/reset guards."""

import pickle  # nosec
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from uuid import uuid4

import grpc
import pytest
import torch

import lerobot.async_inference.server.rlt_runtime as runtime_module
from lerobot.async_inference.client.rlt_runtime import RemoteRLTClient, load_reward_provider
from lerobot.async_inference.client.robot_client_deployment import (
    RobotClientDeployment,
    RobotClientDeploymentConfig,
)
from lerobot.async_inference.server.policy_server_deployment import (
    PolicyServerDeployment,
    PolicyServerDeploymentConfig,
)
from lerobot.async_inference.shared.helpers import RemotePolicyConfig
from lerobot.async_inference.shared.rlt_config import RLTClientConfig, RLTServerConfig
from lerobot.async_inference.shared.rlt_remote import RLTFailure, RLTPlan, RLTRequest, RLTSetup
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.rl.rlt.replay import FixedWindowReplayBuffer
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks
from lerobot.utils.constants import OBS_STATE
from tests.async_inference.test_robot_client_deployment import _make_client
from tests.mocks.mock_robot import MockRobotConfig
from tests.rl.test_rlt_rtc import SESSION, Features, next_request


class Context:
    active = True
    metadata = ()

    def peer(self):
        return "test-rlt-client"

    def is_active(self):
        return self.active

    def send_initial_metadata(self, metadata):
        self.metadata = metadata


def server_runtime(tmp_path, monkeypatch):
    source = SimpleNamespace(
        type="pi05",
        backend="rlinf",
        use_rlt=True,
        use_peft=False,
        rlt_embed_dim=4,
        chunk_size=14,
        action_feature_names=["j1", "j2"],
        input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (3,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
    )
    monkeypatch.setattr(
        runtime_module.FrozenRLTFeatures,
        "from_pretrained",
        lambda checkpoint, *, config_factory, **kwargs: Features(
            config_factory(source, "immutable-features")
        ),
    )
    monkeypatch.setattr(runtime_module, "prepare_remote_observation", lambda raw, *args: raw)
    cfg = PolicyServerDeploymentConfig(rlt=RLTServerConfig(enabled=True, replay_dir=str(tmp_path)))
    server = PolicyServerDeployment(cfg)
    context = Context()
    server.Ready(services_pb2.Empty(), context)
    policy = RemotePolicyConfig("pi05", "stage1", {}, 14, "cpu")
    setup = RLTSetup(SESSION, policy, ("j1.pos", "j2.pos"), cfg.fps)
    server.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(setup)), context)
    return server, context


def exchange(server, context, request):
    payload = pickle.dumps(request)
    server.SendObservations(send_bytes_in_chunks(payload, services_pb2.Observation, silent=True), context)
    return pickle.loads(server.GetActions(services_pb2.Empty(), context).data)


def test_existing_server_entrypoints_complete_rlt_replay_and_clean_restart(tmp_path, monkeypatch):
    server, context = server_runtime(tmp_path, monkeypatch)
    try:
        dry = exchange(server, context, RLTRequest(SESSION, 0, 0, 0, {"value": 0}, dry_run=True))
        assert isinstance(dry, RLTPlan) and len(server._rlt_runtime.replay) == 0
        first = exchange(server, context, RLTRequest(SESSION, 1, 0, 0, {"value": 0}))
        second = exchange(server, context, next_request(first))
        assert second.start_step == 10 and len(server._rlt_runtime.replay) == 1
        exchange(server, context, next_request(second, count=4, truncated=True))
        replay = FixedWindowReplayBuffer.load(tmp_path / SESSION, server._rlt_runtime.features.config)
        assert len(replay) == 2 and replay._data[-1].truncated
        runtime = server._rlt_runtime
        server.Ready(services_pb2.Empty(), context)
        assert not runtime.thread.is_alive() and server._rlt_runtime is None
    finally:
        server.stop()


def test_server_fails_closed_on_missing_receipt_and_no_duplicate_replay(tmp_path, monkeypatch):
    server, context = server_runtime(tmp_path, monkeypatch)
    try:
        exchange(server, context, RLTRequest(SESSION, 1, 0, 0, {"value": 0}))
        response = exchange(server, context, RLTRequest(SESSION, 1, 1, 10, {"value": 10}))
        assert isinstance(response, RLTFailure) and len(server._rlt_runtime.replay) == 0
        with pytest.raises(RuntimeError, match="failed"):
            exchange(server, context, RLTRequest(SESSION, 2, 0, 0, {"value": 0}))
    finally:
        server.stop()


def test_rlt_fatal_device_context_requires_process_restart(tmp_path, monkeypatch):
    server, context = server_runtime(tmp_path, monkeypatch)

    def failed(value):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    server._rlt_runtime.features.hook = failed
    try:
        response = exchange(server, context, RLTRequest(SESSION, 1, 0, 0, {"value": 0}))
        assert isinstance(response, RLTFailure)
        assert server._fatal_cuda_error and not server.running
        with pytest.raises(RuntimeError, match="restart the server"):
            server.Ready(services_pb2.Empty(), None)
    finally:
        server.stop()


def test_missing_reward_provider_is_rejected_before_constructing_or_connecting_robot(tmp_path, monkeypatch):
    import lerobot.async_inference.client.robot_client_deployment as module

    monkeypatch.setattr(
        module,
        "make_robot_from_config",
        lambda _: pytest.fail("robot constructed before callback validation"),
    )
    cfg = RobotClientDeploymentConfig(
        policy_type="pi05",
        pretrained_name_or_path="stage1",
        robot=MockRobotConfig(calibration_dir=tmp_path),
        actions_per_chunk=50,
        server_address="localhost:9999",
        rlt=RLTClientConfig(enabled=True, reward_provider="nonexistent_rlt_callback_module:reward"),
    )
    with pytest.raises(ModuleNotFoundError):
        RobotClientDeployment(cfg)


def test_impossible_rlt_timing_budget_is_rejected_in_config_before_hardware(tmp_path):
    with pytest.raises(ValueError, match="timing margin"):
        RobotClientDeploymentConfig(
            policy_type="pi05",
            pretrained_name_or_path="stage1",
            robot=MockRobotConfig(calibration_dir=tmp_path),
            actions_per_chunk=50,
            server_address="localhost:9999",
            fps=30,
            rlt=RLTClientConfig(enabled=True, reward_provider="manual", budget_margin_s=0.2),
        )


def test_rlt_command_cancelled_while_waiting_motor_lock_never_dispatches(tmp_path):
    client = _make_client(tmp_path)
    client.config.rlt = RLTClientConfig(enabled=True, reward_provider="manual")
    runtime = RemoteRLTClient(client, load_reward_provider("manual"))
    client._rlt_runtime = runtime
    assert client.start_rollout()
    generation = client.rollout_generation
    cancelled = threading.Event()
    commands, errors = [], []
    client.robot.send_action = lambda action: commands.append(action)
    client.action_interpolator.add(torch.zeros(3))
    while client.action_interpolator.get() is not None:
        pass
    client._robot_command_lock.acquire()

    def step():
        try:
            runtime._step(torch.ones(3), generation, cancelled)
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=step)
    worker.start()
    try:
        time.sleep(0.02)
        cancelled.set()
        client._robot_command_lock.release()
        worker.join(1)
        assert not commands and isinstance(errors[0], InterruptedError)
    finally:
        client.stop()


def test_ordinary_client_keeps_rlt_disabled_and_reset_cancels_rlt_owner(tmp_path):
    client = _make_client(tmp_path)
    assert client._rlt_runtime is None and not client.config.rlt.enabled
    calls = []
    client._rlt_runtime = SimpleNamespace(
        cancel=lambda: calls.append("cancel"),
        begin_rollout=lambda: calls.append("begin"),
        close=lambda: calls.append("close"),
    )
    try:
        assert client.start_rollout()
        assert client.reset_rollout()
        assert calls.count("cancel") == 2
    finally:
        client.stop()
    assert "close" in calls


def test_rlt_reuses_background_observation_without_polling_sensors_on_control_thread(tmp_path, monkeypatch):
    client = _make_client(tmp_path)
    client.config.rlt = RLTClientConfig(enabled=True, reward_provider="manual")
    runtime = RemoteRLTClient(client, load_reward_provider("manual"))
    client._rlt_runtime = runtime
    assert client.start_rollout()
    sample = client._capture_observation(client.rollout_generation, buffered_cameras=False)
    client._latest_observation = sample
    monkeypatch.setattr(
        client, "_capture_observation", lambda *args, **kwargs: pytest.fail("control thread polled sensors")
    )
    try:
        raw, selected = runtime._capture(client.rollout_generation, after=sample.started_at - 0.001)
        assert selected is sample and raw is not sample.observation
        assert raw["task"] == client.task
    finally:
        client.stop()


def test_reset_can_fence_generation_while_rlt_is_waiting_for_motor_lock(tmp_path):
    client = _make_client(tmp_path)
    client.config.rlt = RLTClientConfig(enabled=True, reward_provider="manual")
    runtime = RemoteRLTClient(client, load_reward_provider("manual"))
    client._rlt_runtime = runtime
    assert client.start_rollout()
    generation = client.rollout_generation
    runtime.cancelled = threading.Event()
    cancelled = runtime.cancelled
    client._reset_robot_pose = lambda: True
    commands, errors = [], []
    client.robot.send_action = lambda action: commands.append(action)
    client._robot_command_lock.acquire()

    def step():
        try:
            runtime._step(torch.ones(3), generation, cancelled)
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=step)
    worker.start()
    try:
        time.sleep(0.02)
        # This must complete without waiting for the action thread's motor lock.
        assert client.reset_rollout()
        assert cancelled.is_set()
        client._robot_command_lock.release()
        worker.join(1)
        assert not worker.is_alive() and not commands
        assert isinstance(errors[0], InterruptedError)
    finally:
        client.stop()


def test_real_grpc_client_server_warmup_overlap_and_interpolated_receipts(tmp_path, monkeypatch):
    source = SimpleNamespace(
        type="pi05",
        backend="rlinf",
        use_rlt=True,
        use_peft=False,
        rlt_embed_dim=4,
        chunk_size=14,
        action_feature_names=["motor_1", "motor_2", "motor_3"],
        input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (3,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (3,))},
    )

    class RobotFeatures(Features):
        def encode(self, raw, **kwargs):
            return super().encode({"value": 0}, **kwargs)

    monkeypatch.setattr(
        runtime_module.FrozenRLTFeatures,
        "from_pretrained",
        lambda checkpoint, *, config_factory, **kwargs: RobotFeatures(
            config_factory(source, "immutable-features")
        ),
    )
    monkeypatch.setattr(runtime_module, "prepare_remote_observation", lambda raw, *args: raw)
    server = PolicyServerDeployment(
        PolicyServerDeploymentConfig(rlt=RLTServerConfig(enabled=True, replay_dir=str(tmp_path)))
    )
    address = f"unix:/tmp/rlt_stage3_{uuid4().hex[:12]}.sock"
    grpc_server = grpc.server(ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server, grpc_server)
    grpc_server.add_insecure_port(address)
    grpc_server.start()
    client = RobotClientDeployment(
        RobotClientDeploymentConfig(
            policy_type="pi05",
            pretrained_name_or_path="stage1",
            policy_device="cpu",
            robot=MockRobotConfig(
                calibration_dir=tmp_path, random_values=False, static_values=[0.0, 0.0, 0.0]
            ),
            server_address=address,
            actions_per_chunk=14,
            interpolation_multiplier=2,
            rlt=RLTClientConfig(enabled=True, reward_provider="manual"),
            startup_calibration_runs=1,
            startup_calibration_timeout_s=3,
            request_timeout_s=3,
            reset_duration_s=0,
            reset_settle_s=0,
        )
    )
    motor_commands = []

    def send(action):
        motor_commands.append(torch.tensor(list(action.values())))
        return action

    client.robot.send_action = send
    rewards = []

    def reward(raw, actual):
        from lerobot.rl.rlt.rollout import RewardSignal

        rewards.append(actual.clone())
        return RewardSignal(float(len(rewards) == 22), terminated=len(rewards) == 22)

    client._rlt_runtime.reward = reward
    worker = None
    try:
        assert client.start(), client.worker_error
        assert not motor_commands and len(server._rlt_runtime.replay) == 0
        worker = threading.Thread(target=client._rlt_runtime.run, daemon=True)
        worker.start()
        assert client.start_rollout()
        deadline = time.monotonic() + 4
        while len(server._rlt_runtime.replay) < 3 and not client.worker_error and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not client.worker_error, client.worker_error
        assert len(rewards) == 22 and len(motor_commands) == 44
        replay = server._rlt_runtime.replay
        assert [row.valid_steps for row in replay._data] == [10, 10, 2]
        torch.testing.assert_close(replay._data[0].action, torch.stack(rewards[:10]) / 2)
        assert client.latest_action == 21
    finally:
        client.stop()
        if worker is not None:
            worker.join(2)
        grpc_server.stop(0).wait(timeout=2)
        server.stop()
