# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared RTC protocol and transport behavior exercised through current deployment."""

from __future__ import annotations

import pickle
import threading
import time
from concurrent import futures

import pytest
import torch

grpc = pytest.importorskip("grpc")

from lerobot.async_inference.client.robot_client_deployment import (  # noqa: E402
    RobotClientDeployment,
    RobotClientDeploymentConfig,
)
from lerobot.async_inference.server.policy_server_deployment import (  # noqa: E402
    PolicyServerDeployment,
    PolicyServerDeploymentConfig,
)
from lerobot.async_inference.shared.helpers import (  # noqa: E402
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
)
from lerobot.async_inference.shared.rtc_remote import (  # noqa: E402
    DEPLOYMENT_RTC_PROTOCOL_VERSION,
    RemoteRTCActionChunk,
    RemoteRTCObservation,
    RemoteRTCPolicySetup,
    RTCActionRef,
    decode_rollout_generation,
    decode_rollout_sequence,
    encode_rollout_observation_id,
    normalize_rtc_prefix_length,
)
from lerobot.policies.rtc import RTCConfig  # noqa: E402
from lerobot.transport import services_pb2, services_pb2_grpc  # noqa: E402
from lerobot.transport.utils import send_bytes_in_chunks  # noqa: E402
from tests.mocks.mock_robot import MockRobotConfig  # noqa: E402


def _make_server(**kwargs) -> PolicyServerDeployment:
    return PolicyServerDeployment(
        PolicyServerDeploymentConfig(host="localhost", port=9999, action_wait_timeout=0, **kwargs)
    )


def _actions(start: int, count: int, source_timestamp: float | None = None) -> list[TimedAction]:
    timestamp = time.time() if source_timestamp is None else source_timestamp
    return [
        TimedAction(
            timestamp=timestamp + index / 30,
            timestep=start + index,
            action=torch.tensor([float(start + index), float(index)]),
        )
        for index in range(count)
    ]


def test_get_actions_returns_one_complete_chunk_per_rpc():
    server = _make_server()
    chunk = RemoteRTCActionChunk(
        source_observation_id=4,
        chunk_start_step=10,
        observation_timestamp=time.time(),
        actions=_actions(10, 8),
    )
    server._action_mailbox.put_nowait(chunk)

    response = server.GetActions(services_pb2.Empty(), None)
    decoded = pickle.loads(response.data)

    assert isinstance(decoded, RemoteRTCActionChunk)
    assert len(decoded.actions) == 8
    assert [action.get_timestep() for action in decoded.actions] == list(range(10, 18))
    assert server.GetActions(services_pb2.Empty(), None).data == b""
    server.stop()


def test_get_actions_surfaces_fatal_inference_worker_failure():
    server = _make_server()
    server._worker_failed.set()

    with pytest.raises(RuntimeError, match="inference worker failed"):
        server.GetActions(services_pb2.Empty(), None)
    server.stop()


def test_observation_piggybacks_progress_and_latency_in_one_existing_rpc():
    server = _make_server()
    observation = TimedObservation(
        timestamp=time.time(),
        timestep=7,
        observation={"joint.pos": 1.5, "task": "pick"},
        must_go=True,
    )
    envelope = RemoteRTCObservation(
        observation=observation,
        observation_id=12,
        executed_action_step=7,
        action_refs=(RTCActionRef(source_observation_id=3, action_index=5),),
        latency_feedback_seconds=0.12,
        latency_feedback_source_id=3,
    )
    request_iterator = send_bytes_in_chunks(pickle.dumps(envelope), services_pb2.Observation, silent=True)

    server.SendObservations(request_iterator, None)
    stale = RemoteRTCObservation(
        observation=observation,
        observation_id=11,
        executed_action_step=6,
    )
    server.SendObservations(
        send_bytes_in_chunks(pickle.dumps(stale), services_pb2.Observation, silent=True),
        None,
    )

    assert server._observation_mailbox.get_nowait() == envelope
    assert server._latest_received_observation_id == envelope.observation_id
    assert server._current_latency() == pytest.approx(0.12)
    server.stop()


def test_server_reconstructs_raw_prefix_from_client_references():
    server = _make_server()
    raw = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    processed = raw + 100
    server._chunk_cache.put(2, raw, processed)
    envelope = RemoteRTCObservation(
        observation=TimedObservation(time.time(), 1, {}),
        observation_id=3,
        executed_action_step=1,
        action_refs=(
            RTCActionRef(2, 1),
            RTCActionRef(2, 2),
            RTCActionRef(2, 3),
        ),
    )

    rebuilt_raw, rebuilt_processed = server._reconstruct_prefix(envelope)

    assert torch.equal(rebuilt_raw, raw[1:])
    assert torch.equal(rebuilt_processed, processed[1:])
    server.stop()


def test_short_model_prefix_holds_last_action_instead_of_padding_zero():
    prefix = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    normalized = normalize_rtc_prefix_length(prefix, target_steps=5)

    assert torch.equal(
        normalized,
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [5.0, 6.0],
                [5.0, 6.0],
            ]
        ),
    )


def test_empty_model_prefix_cannot_be_hold_padded():
    with pytest.raises(ValueError, match="cannot hold-pad an empty RTC prefix"):
        normalize_rtc_prefix_length(torch.empty(0, 3), target_steps=5)


def test_server_reconstructs_unmodified_rows_from_different_chunks():
    server = _make_server()
    old_raw = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    old_processed = old_raw * 10
    new_raw = torch.tensor([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]])
    new_processed = new_raw * 10
    server._chunk_cache.put(1, old_raw, old_processed)
    server._chunk_cache.put(2, new_raw, new_processed)
    envelope = RemoteRTCObservation(
        observation=TimedObservation(time.time(), 1, {}),
        observation_id=3,
        executed_action_step=0,
        action_refs=(RTCActionRef(1, 1), RTCActionRef(2, 2)),
    )
    rebuilt_raw, rebuilt_processed = server._reconstruct_prefix(envelope)
    torch.testing.assert_close(rebuilt_raw, torch.stack([old_raw[1], new_raw[2]]))
    torch.testing.assert_close(rebuilt_processed, torch.stack([old_processed[1], new_processed[2]]))
    server.stop()


def test_action_ref_rejects_invalid_indices_and_removed_offset_metadata():
    with pytest.raises(ValueError, match="source_observation_id"):
        RTCActionRef(-1, 0)
    with pytest.raises(ValueError, match="action_index"):
        RTCActionRef(0, -1)
    with pytest.raises(TypeError, match="bridge_weight"):
        RTCActionRef(2, 1, bridge_weight=0.5)


def _make_client(tmp_path) -> RobotClientDeployment:
    return RobotClientDeployment(
        RobotClientDeploymentConfig(
            policy_type="smolvla",
            pretrained_name_or_path="checkpoint",
            robot=MockRobotConfig(
                calibration_dir=tmp_path,
                random_values=False,
                static_values=[0.0, 0.0, 0.0],
            ),
            actions_per_chunk=8,
            server_address="localhost:9999",
            interpolation_multiplier=1,
            continuous_observation=False,
            reset_duration_s=0,
            reset_settle_s=0,
        )
    )


def test_control_loop_executes_multiple_local_actions_without_rpc(tmp_path):
    client = _make_client(tmp_path)
    assert client.start_rollout()

    class NoNetworkInControlLoop:
        calls = 0

        def GetActions(self, request):  # noqa: N802
            del request
            self.calls += 1
            raise AssertionError("control_loop_action must not call GetActions")

        def SendObservations(self, request):  # noqa: N802
            del request
            self.calls += 1
            raise AssertionError("control_loop_action must not upload observations")

    stub = NoNetworkInControlLoop()
    client.stub = stub
    timestamp = time.time()
    chunk = RemoteRTCActionChunk(
        source_observation_id=encode_rollout_observation_id(client.rollout_generation, 0),
        chunk_start_step=0,
        observation_timestamp=timestamp,
        actions=[
            TimedAction(timestamp, 0, torch.tensor([1.0, 2.0, 3.0])),
            TimedAction(timestamp + 1 / 30, 1, torch.tensor([4.0, 5.0, 6.0])),
        ],
    )
    assert client._handle_action_chunk(chunk)

    first = client.control_loop_action()
    second = client.control_loop_action()

    assert first == {"motor_1.pos": 1.0, "motor_2.pos": 2.0, "motor_3.pos": 3.0}
    assert second == {"motor_1.pos": 4.0, "motor_2.pos": 5.0, "motor_3.pos": 6.0}
    assert client.latest_action == 1
    assert stub.calls == 0
    client.stop()


def test_client_action_chunk_watermark_absorbs_duplicates(tmp_path):
    client = _make_client(tmp_path)
    assert client.start_rollout()
    timestamp = time.time()
    chunk = RemoteRTCActionChunk(
        source_observation_id=encode_rollout_observation_id(client.rollout_generation, 3),
        chunk_start_step=0,
        observation_timestamp=timestamp,
        actions=[TimedAction(timestamp, 0, torch.tensor([1.0, 2.0, 3.0]))],
    )

    assert client._handle_action_chunk(chunk)
    assert not client._handle_action_chunk(chunk)
    assert client.action_schedule.qsize(-1) == 1
    client.stop()


def test_remote_rtc_end_to_end_keeps_rpc_off_the_action_path(monkeypatch, tmp_path):
    class FakePolicy:
        def __init__(self):
            self.config = type(
                "Config",
                (),
                {
                    "action_feature_names": ["motor_1.pos", "motor_2.pos", "motor_3.pos"],
                    "image_features": {},
                    "rtc_training_max_delay": 0,
                },
            )()
            self.calls = 0

        def supports_rtc(self):
            return True

        def predict_action_chunk(self, observation, *, inference_delay, prev_chunk_left_over):
            del observation, inference_delay, prev_chunk_left_over
            self.calls += 1
            value = float(self.calls)
            return torch.full((1, 8, 3), value)

        def reset(self):
            pass

        def init_rtc_processor(self):
            return None

        def eval(self):
            return self

    class IdentityPipeline:
        steps = []

        def reset(self):
            pass

        def __call__(self, value):
            return value

    policy = FakePolicy()

    def fake_load_policy(self, specs):
        self.policy_type = specs.policy_type
        self.policy = policy
        self.preprocessor = IdentityPipeline()
        self.postprocessor = IdentityPipeline()
        self.device = specs.device
        self.lerobot_features = specs.lerobot_features
        self.actions_per_chunk = specs.actions_per_chunk
        return False

    monkeypatch.setattr(PolicyServerDeployment, "_load_or_reuse_policy", fake_load_policy)
    policy_server = PolicyServerDeployment(
        PolicyServerDeploymentConfig(
            host="127.0.0.1",
            port=9999,
            fps=100,
            inference_latency=0,
            action_wait_timeout=0.1,
            rtc=RTCConfig(execution_horizon=2),
        )
    )
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, grpc_server)
    server_address = f"unix:{tmp_path}/rtc.sock"
    grpc_server.add_insecure_port(server_address)
    grpc_server.start()

    client = RobotClientDeployment(
        RobotClientDeploymentConfig(
            policy_type="smolvla",
            pretrained_name_or_path="checkpoint",
            robot=MockRobotConfig(
                calibration_dir=tmp_path,
                random_values=False,
                static_values=[0.0, 0.0, 0.0],
            ),
            actions_per_chunk=8,
            server_address=server_address,
            fps=100,
            interpolation_multiplier=1,
            reset_duration_s=0,
            reset_settle_s=0,
        )
    )
    receiver = threading.Thread(target=client.receive_actions, daemon=True)
    sender = threading.Thread(target=client.observation_sender, daemon=True)

    try:
        assert client.start()
        assert client.start_rollout("pick")
        receiver.start()
        sender.start()
        client.start_barrier.wait(timeout=2)

        deadline = time.monotonic() + 3
        while client.latest_action < 2 and time.monotonic() < deadline:
            client.control_loop_action()
            client._request_observation_if_ready("pick")
            time.sleep(client.config.environment_dt)

        assert client.latest_action >= 2
        assert policy.calls >= 1
    finally:
        client.stop()
        if receiver.ident is not None:
            receiver.join(timeout=2)
        if sender.ident is not None:
            sender.join(timeout=2)
        policy_server.stop()
        grpc_server.stop(None)


def test_trigger_only_enqueues_background_observation_work(tmp_path):
    client = _make_client(tmp_path)
    assert client.start_rollout()
    capture_calls = 0

    def capture():
        nonlocal capture_calls
        capture_calls += 1
        return {}

    client.robot.get_observation = capture

    assert client._request_observation_if_ready("pick")
    request = client._request_mailbox.get_nowait()
    assert decode_rollout_generation(request.observation_id) == client.rollout_generation
    assert decode_rollout_sequence(request.observation_id) == 0
    assert request.triggered_action_step == -1
    assert capture_calls == 0
    client.stop()


def test_observation_sender_snapshots_execution_after_capture(tmp_path):
    client = _make_client(tmp_path)
    assert client.start_rollout()
    timestamp = time.time()
    assert client._handle_action_chunk(
        RemoteRTCActionChunk(
            source_observation_id=encode_rollout_observation_id(client.rollout_generation, 0),
            chunk_start_step=0,
            observation_timestamp=timestamp,
            actions=[
                TimedAction(
                    timestamp + index / 30,
                    index,
                    torch.tensor([value, value, value]),
                )
                for index, value in enumerate([0.0, 1.0, 2.0])
            ],
        )
    )

    capture_started = threading.Event()
    release_capture = threading.Event()
    envelope_ready = threading.Event()
    envelopes = []

    def delayed_capture():
        capture_started.set()
        assert release_capture.wait(timeout=2)
        return {"captured_latest_action": client.latest_action}

    def collect_envelope(envelope):
        envelopes.append(envelope)
        envelope_ready.set()
        return True

    client.robot.get_observation = delayed_capture
    client.send_observation = collect_envelope
    client.start_barrier = threading.Barrier(1)

    client.control_loop_action()  # step 0
    assert client._request_observation_if_ready("pick")
    sender = threading.Thread(target=client.observation_sender)
    sender.start()

    try:
        assert capture_started.wait(timeout=2)
        client.control_loop_action()  # step 1 advances while capture is blocked
        release_capture.set()
        assert envelope_ready.wait(timeout=2)
        envelope = envelopes[0]

        assert envelope.observation.get_observation()["captured_latest_action"] == 1
        assert envelope.executed_action_step == 1
        assert envelope.action_refs == (
            RTCActionRef(encode_rollout_observation_id(client.rollout_generation, 0), 2),
        )
        assert envelope.observation.get_timestep() == 2
    finally:
        release_capture.set()
        client.shutdown_event.set()
        sender.join(timeout=2)
        client.stop()


def test_policy_setup_enables_rtc_and_starts_server_worker(monkeypatch):
    class FakePolicy:
        def __init__(self):
            self.config = type(
                "Config",
                (),
                {"action_feature_names": ["joint.pos"], "rtc_training_max_delay": 0},
            )()
            self.rtc_initialized = False
            self.eval_called = False

        def supports_rtc(self):
            return True

        def predict_action_chunk(self, observation, *, inference_delay, prev_chunk_left_over):
            del observation, inference_delay, prev_chunk_left_over

        def reset(self):
            pass

        def init_rtc_processor(self):
            self.rtc_initialized = True

        def eval(self):
            self.eval_called = True

    class FakePipeline:
        steps = []

    policy = FakePolicy()

    def fake_load_policy(self, specs):
        assert isinstance(specs, RemotePolicyConfig)
        self.policy_type = specs.policy_type
        self.policy = policy
        self.preprocessor = FakePipeline()
        self.postprocessor = FakePipeline()
        self.device = "cpu"
        self.actions_per_chunk = specs.actions_per_chunk
        return False

    monkeypatch.setattr(PolicyServerDeployment, "_load_or_reuse_policy", fake_load_policy)
    server = _make_server()
    setup = RemoteRTCPolicySetup(
        protocol_version=DEPLOYMENT_RTC_PROTOCOL_VERSION,
        policy=RemotePolicyConfig(
            policy_type="smolvla",
            pretrained_name_or_path="checkpoint",
            lerobot_features={},
            actions_per_chunk=20,
        ),
        control_frequency_hz=30,
        action_feature_names=("joint.pos",),
    )
    server.SendPolicyInstructions(
        services_pb2.PolicySetup(data=pickle.dumps(setup)),
        None,
    )

    assert policy.config.rtc_config is server.config.rtc
    assert policy.rtc_initialized and policy.eval_called
    assert server._inference_thread is not None and server._inference_thread.is_alive()
    server.stop()


def test_policy_setup_rejects_mismatched_client_server_fps():
    server = _make_server()
    setup = RemoteRTCPolicySetup(
        protocol_version=DEPLOYMENT_RTC_PROTOCOL_VERSION,
        policy=RemotePolicyConfig(
            policy_type="smolvla",
            pretrained_name_or_path="checkpoint",
            lerobot_features={},
            actions_per_chunk=20,
        ),
        control_frequency_hz=20,
    )

    with pytest.raises(ValueError, match="control frequencies must match"):
        server.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(setup)), None)
    server.stop()


@pytest.mark.parametrize(
    "field,value",
    [("action_wait_timeout", -0.1), ("action_cache_size", 0)],
)
def test_rtc_server_config_rejects_invalid_transport_settings(field, value):
    with pytest.raises(ValueError):
        PolicyServerDeploymentConfig(**{field: value})


def test_rtc_server_config_requires_enabled_rtc():
    with pytest.raises(ValueError, match="rtc.enabled=true"):
        PolicyServerDeploymentConfig(rtc=RTCConfig(enabled=False))
