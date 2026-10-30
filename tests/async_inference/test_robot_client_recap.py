# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import json
import threading
from types import SimpleNamespace

import pytest

import lerobot.async_inference.client.robot_client_recap as recap_module
from lerobot.async_inference.client.robot_client_dagger import DAggerPhase
from lerobot.async_inference.client.robot_client_deployment import RolloutState
from lerobot.async_inference.client.robot_client_recap import RobotClientRecap, RobotClientRecapConfig
from lerobot.async_inference.shared.metadata_recap import IDENTITY_METADATA_KEY
from lerobot.async_inference.shared.rtc_remote import encode_rollout_observation_id
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.light import NullStatusIndicator
from tests.async_inference.test_piper_hil_recap import IDENTITY, _records
from tests.async_inference.test_robot_client_dagger import _chunk, _FakeController
from tests.async_inference.test_robot_client_deployment import _assert_entrypoint_cleans_up_while_input_blocks
from tests.mocks.mock_robot import MockRobotConfig


@pytest.fixture
def client(tmp_path, monkeypatch):
    config = RobotClientRecapConfig(
        policy_type="smolvla",
        pretrained_name_or_path="checkpoint",
        robot=MockRobotConfig(calibration_dir=tmp_path, random_values=False, static_values=[0.0] * 3),
        actions_per_chunk=8,
        server_address="localhost:9999",
        interpolation_multiplier=2,
        reset_duration_s=0,
        reset_settle_s=0,
        task="pick cube",
        dagger_require_safe_firmware=False,
        recap_dataset_repo_id="test/client_recap",
        recap_dataset_root=tmp_path / "dataset",
        recap_dataset_video=False,
    )
    instance = RobotClientRecap(
        config, dagger_controller=_FakeController(), status_indicator=NullStatusIndicator()
    )
    instance.policy_identity = dict(IDENTITY)
    monkeypatch.setattr(instance, "_reset_robot_pose", lambda: True)
    yield instance
    instance.stop()


def _autonomous_frame(client):
    source = encode_rollout_observation_id(client.rollout_generation, 0)
    assert client._handle_action_chunk(_chunk(source, [1.0, 2.0]))
    assert client.control_loop_action() is not None
    client.rollout_recorder._record_one()
    return source


def test_recap_records_action_metadata_with_diagnostics_disabled(client, monkeypatch):
    assert not client.config.debug_rtc_diagnostics
    monkeypatch.setattr(client, "_max_abs", lambda *_: pytest.fail("recording ran diagnostic reductions"))
    assert client.start_rollout()
    source = _autonomous_frame(client)
    sample = client._read_recording_sample()
    # DAgger seeds interpolation at the aligned pose: the first command is
    # halfway to the policy endpoint, and RECAP must record that sent value.
    assert sample.action == dict.fromkeys(client.robot.action_features, 0.5)
    assert sample.source_observation_id == source
    snapshot = client.command_diagnostics_snapshot()
    assert snapshot.delta is None
    assert snapshot.send_duration_seconds == 0


def test_auto_and_intervention_are_one_episode_and_stale_policy_is_fenced(client):
    assert client.start_rollout()
    old_source = _autonomous_frame(client)
    assert client.pause_autonomy()
    client.rollout_recorder._record_one()
    assert client.start_correction()
    client.rollout_recorder._record_one()
    assert not client._handle_action_chunk(_chunk(old_source, [99.0]))
    assert client.stop_correction()
    assert client.dagger_phase is DAggerPhase.PAUSED
    assert client.rollout_state is RolloutState.RUNNING
    assert client.rollout_recorder.saved_episodes == 0
    assert not client.set_task("different task")
    assert client.finish_rollout("success")
    assert client.rollout_state is RolloutState.READY
    client.stop()

    (record,) = _records(client.config.recap_dataset_root)
    assert record["frame_count"] == 3
    assert record["outcome"] == "success"
    assert record["model"] == IDENTITY
    dataset = LeRobotDataset("test/client_recap", root=client.config.recap_dataset_root)
    assert [dataset[i]["intervention"].item() for i in range(3)] == [False, False, True]
    assert dataset[0]["recap.source_observation_id"].item() == old_source
    assert len(dataset) == 3
    assert "recap.terminal_observation" not in dataset.features


def test_pure_autonomous_failure_is_saved_without_correction(client):
    assert client.start_rollout()
    _autonomous_frame(client)
    assert client.finish_rollout("failure")
    (record,) = _records(client.config.recap_dataset_root)
    assert record["outcome"] == "failure"
    assert record["success"] is False
    assert client.rollout_recorder.saved_episodes == 1


def test_finish_after_manual_home_does_not_request_an_extra_observation(client, monkeypatch):
    assert client.start_rollout()
    _autonomous_frame(client)
    assert client.pause_autonomy()
    assert client.start_correction()
    client.rollout_recorder._record_one()  # The operator's final normal sample, e.g. at home.
    monkeypatch.setattr(
        client._follower_robot,
        "get_observation",
        lambda: pytest.fail("finishing must not capture an extra observation"),
    )
    assert client.finish_rollout("success")
    (record,) = _records(client.config.recap_dataset_root)
    assert record["frame_count"] == 2


@pytest.mark.parametrize("phase", ["paused", "correcting", "fault"])
def test_reset_discards_unconfirmed_data_and_next_episode_starts_clean(client, phase):
    assert client.start_rollout()
    _autonomous_frame(client)
    assert client.pause_autonomy()
    if phase == "correcting":
        assert client.start_correction()
        client.rollout_recorder._record_one()
    elif phase == "fault":
        client._set_fault("test handover", RuntimeError("bridge failed"))
        assert not client.finish_rollout("failure")
    assert client._handle_operator_event("reset")
    assert not _records(client.config.recap_dataset_root)
    assert not client.rollout_recorder.dataset.has_pending_frames()
    assert not client.rollout_recorder.has_rollout
    assert client.rollout_recorder.saved_episodes == 0
    assert client.rollout_state is RolloutState.READY
    assert client.start_rollout("next task")
    _autonomous_frame(client)
    assert client.finish_rollout("failure")
    (record,) = _records(client.config.recap_dataset_root)
    assert record["episode_index"] == 0
    assert record["frame_count"] == 1
    assert record["task"] == "next task"


def test_stop_disconnects_before_discarding_unconfirmed_data(client, monkeypatch):
    assert client.start_rollout()
    _autonomous_frame(client)
    original_discard = client.rollout_recorder.dataset.clear_episode_buffer

    def discard():
        assert not client.robot.is_connected
        assert "close" in client.dagger_controller.calls
        return original_discard()

    monkeypatch.setattr(client.rollout_recorder.dataset, "clear_episode_buffer", discard)
    client.stop()
    assert not _records(client.config.recap_dataset_root)
    assert client.rollout_recorder.saved_episodes == 0
    assert not client.rollout_recorder.dataset.has_pending_frames()


def test_esc_disconnects_while_video_save_is_blocked(client, monkeypatch):
    assert client.start_rollout()
    _autonomous_frame(client)
    saving = threading.Event()
    release = threading.Event()
    disconnected = threading.Event()
    original_save = client.rollout_recorder.dataset.save_episode
    original_disconnect = client.robot.disconnect

    def save(**kwargs):
        saving.set()
        assert release.wait(timeout=5)
        return original_save(**kwargs)

    def disconnect():
        original_disconnect()
        disconnected.set()

    monkeypatch.setattr(client.rollout_recorder.dataset, "save_episode", save)
    monkeypatch.setattr(client.robot, "disconnect", disconnect)
    monkeypatch.setattr(client, "_reset_robot_pose", lambda: pytest.fail("must not reset after Esc"))
    worker = threading.Thread(target=lambda: client.finish_rollout("success"))
    stopper = threading.Thread(target=client.stop)
    worker.start()
    try:
        assert saving.wait(timeout=2)
        assert not client.start_rollout()
        assert not client.start_correction()
        assert not client.reset_rollout()
        assert client.submit_operator_event("stop")
        stopper.start()
        assert disconnected.wait(timeout=2), "encoding must not delay hardware shutdown"
    finally:
        release.set()
        worker.join(timeout=3)
        if stopper.ident is not None:
            stopper.join(timeout=3)
    assert not worker.is_alive()
    assert not stopper.is_alive()
    (record,) = _records(client.config.recap_dataset_root)
    assert record["outcome"] == "success"
    assert client.rollout_recorder.saved_episodes == 1


def test_recording_fault_stops_policy_and_discards_prefix(client):
    assert client.start_rollout()
    _autonomous_frame(client)
    client.rollout_recorder._last_error = OSError("camera timeout")
    assert client.control_loop_action() is None
    assert not client.running
    client.stop()
    assert not _records(client.config.recap_dataset_root)
    assert client.rollout_recorder.saved_episodes == 0


def test_stop_just_before_save_preserves_already_submitted_result(client, monkeypatch):
    assert client.start_rollout()
    _autonomous_frame(client)
    before_save = threading.Event()
    release = threading.Event()
    original_finish = client.rollout_recorder.finish_rollout

    def finish(outcome, *, reason="operator"):
        if outcome == "success":
            before_save.set()
            assert release.wait(timeout=5)
        return original_finish(outcome, reason=reason)

    monkeypatch.setattr(client.rollout_recorder, "finish_rollout", finish)
    worker = threading.Thread(target=lambda: client.finish_rollout("success"))
    worker.start()
    try:
        assert before_save.wait(timeout=2)
        client.stop()  # Shutdown saves the already accepted result before the operator worker resumes.
    finally:
        release.set()
        worker.join(timeout=3)
    assert not worker.is_alive()
    (record,) = _records(client.config.recap_dataset_root)
    assert record["outcome"] == "success"
    assert record["termination_reason"] == "operator"
    assert client.rollout_recorder.saved_episodes == 1


@pytest.mark.parametrize("valid_identity", [False, True])
def test_requires_server_identity_and_rejects_old_server(client, valid_identity):
    client.policy_identity = None
    assert not client.start_rollout()
    client.stub = SimpleNamespace(
        Ready=lambda _: None,
        SendPolicyInstructions=SimpleNamespace(
            with_call=lambda _: (
                None,
                SimpleNamespace(
                    trailing_metadata=lambda: (
                        ((IDENTITY_METADATA_KEY, json.dumps(IDENTITY).encode()),) if valid_identity else ()
                    )
                ),
            )
        ),
    )
    assert client.start() is valid_identity
    if valid_identity:
        assert client.policy_identity == IDENTITY
    else:
        assert "policy_server_recap" in client.worker_error
        assert not client.start_rollout()
        assert not _records(client.config.recap_dataset_root)


def test_result_keyboard_uses_explicit_bindings(client, monkeypatch):
    captured = {}

    def listener(callback, **kwargs):
        captured["callback"] = callback
        return SimpleNamespace(stop=lambda: None)

    monkeypatch.setattr("lerobot.utils.keyboard_input.create_key_listener", listener)
    events = []
    monkeypatch.setattr(client, "submit_operator_event", events.append)
    client.start_operator_input()
    for key in ("right", "s", "f", "u", "left", "esc"):
        captured["callback"](key)
    assert events == ["advance", "success", "failure", "reset", "stop"]


def test_prompt_cleanup_is_responsive_to_background_worker_failure(client, monkeypatch):
    client.config.dagger_input_device = "prompt"
    _assert_entrypoint_cleans_up_while_input_blocks(client, recap_module, monkeypatch)


def test_episode_limit_counts_complete_tasks_not_interventions(client):
    client.config.recap_num_episodes = 1
    assert client.start_rollout()
    _autonomous_frame(client)
    assert client.pause_autonomy()
    assert client.start_correction()
    client.rollout_recorder._record_one()
    assert client.stop_correction()
    assert not client.operator_stop_event.is_set()
    assert client.finish_rollout("success")
    assert client.operator_stop_event.is_set()
    assert client.rollout_recorder.saved_episodes == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("dagger_dataset_repo_id", "test/dagger"),
        ("dagger_input_device", "pedal"),
        ("recap_success_key", "right"),
    ],
)
def test_recap_config_rejects_ambiguous_collection_and_controls(client, field, value):
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(client.config, **{field: value})
