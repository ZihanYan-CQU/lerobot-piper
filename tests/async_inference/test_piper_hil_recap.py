# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import json
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.async_inference.client.piper_hil_recap import (
    RecapControlSample,
    RecapDatasetConfig,
    RecapOutcome,
    RecapPhase,
    RecapRolloutRecorder,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

IDENTITY = {
    "schema_version": 1,
    "model_version": "sha256:" + "a" * 64,
    "identity_source": "server_checkpoint_files",
    "server_load_id": "load-one",
    "policy_type": "smolvla",
}


def _records(root):
    return sorted(
        (json.loads(path.read_text()) for path in (root / "meta/recap").glob("*.json")),
        key=lambda record: record["started_at_utc"],
    )


def _make_recorder(tmp_path):
    robot = SimpleNamespace(
        name="recap_test",
        observation_features={"joint.pos": float},
        action_features={"joint.pos": float},
        cameras={},
        get_observation=lambda: {"joint.pos": 1.0},
    )
    sample = [RecapControlSample(RecapPhase.AUTONOMOUS)]
    config = RecapDatasetConfig(repo_id="test/recap", root=tmp_path / "dataset", use_videos=False)
    recorder = RecapRolloutRecorder(robot, lambda: sample[0], config)
    return recorder, sample


def test_full_rollout_parquet_roundtrip_and_outcome_sidecar(tmp_path):
    recorder, sample = _make_recorder(tmp_path)
    recorder.begin_rollout("pick cube", IDENTITY, {"rtc_mode": "trained"})
    recorder.resume_capture()
    recorder._record_one()  # Waiting for the first policy command is NOT an expert action.
    sample[0] = RecapControlSample(RecapPhase.AUTONOMOUS, {"joint.pos": 2.0}, 1, 4, 17, time.monotonic())
    recorder._record_one()
    recorder.pause_capture()
    sample[0] = RecapControlSample(RecapPhase.PAUSED, generation=2)
    recorder.resume_capture()
    recorder._record_one()
    sample[0] = RecapControlSample(RecapPhase.INTERVENTION, {"joint.pos": 9.0}, generation=2)
    recorder._record_one()
    assert recorder.finish_rollout("success") == 4
    recorder.close()

    dataset = LeRobotDataset("test/recap", root=recorder.config.root)
    assert dataset.meta.total_episodes == 1
    assert len(dataset) == 4
    assert [dataset[i]["intervention"].item() for i in range(4)] == [False, False, False, True]
    assert [dataset[i]["recap.action_valid"].item() for i in range(4)] == [False, True, False, True]
    assert [dataset[i]["recap.segment_id"].item() for i in range(4)] == [0, 0, 1, 2]
    assert dataset[1]["recap.policy_step"].item() == 4
    assert dataset[1]["recap.source_observation_id"].item() == 17
    assert dataset[3]["action"].item() == 9.0
    assert dataset[3]["recap.elapsed_s"] >= dataset[0]["recap.elapsed_s"]
    (record,) = _records(recorder.config.root)
    assert record["status"] == "committed"
    assert record["success"] is True
    assert record["model"] == IDENTITY
    assert record["task"] == "pick cube"
    assert record["frame_count"] == 4
    assert record["episode_index"] == 0


@pytest.mark.parametrize("outcome,success", [("success", True), ("failure", False)])
def test_results_follow_explicit_label_even_with_intervention(tmp_path, outcome, success):
    recorder, sample = _make_recorder(tmp_path)
    recorder.begin_rollout("task", IDENTITY, {})
    recorder.resume_capture()
    sample[0] = RecapControlSample(RecapPhase.INTERVENTION, {"joint.pos": 2.0})
    recorder._record_one()
    recorder.finish_rollout(outcome, reason="operator_test")
    recorder.close()
    (record,) = _records(recorder.config.root)
    assert record["outcome"] == outcome
    assert record["success"] is success
    assert record["termination_reason"] == "operator_test"


def test_resume_retains_each_saved_episode_model(tmp_path):
    recorder, _ = _make_recorder(tmp_path)
    recorder.begin_rollout("task one", IDENTITY, {})
    recorder.resume_capture()
    recorder._record_one()
    recorder.finish_rollout("success")
    recorder.close()
    (first,) = _records(recorder.config.root)
    assert first["outcome"] == "success"

    resumed = RecapRolloutRecorder(
        recorder.robot, recorder.sample_reader, replace(recorder.config, resume=True)
    )
    updated = {**IDENTITY, "model_version": "sha256:" + "b" * 64, "server_load_id": "load-two"}
    resumed.begin_rollout("task two", updated, {})
    resumed.resume_capture()
    resumed._record_one()
    resumed.finish_rollout(RecapOutcome.FAILURE)
    resumed.close()
    records = _records(recorder.config.root)
    assert [r["episode_index"] for r in records] == [0, 1]
    assert [r["model"]["server_load_id"] for r in records] == ["load-one", "load-two"]
    assert records[0]["rollout_id"] != records[1]["rollout_id"]


def test_close_discards_unconfirmed_rollout_and_empty_dataset_can_resume(tmp_path):
    recorder, _ = _make_recorder(tmp_path)
    recorder.begin_rollout("discarded task", IDENTITY, {})
    recorder.resume_capture()
    recorder._record_one()
    recorder.close()
    assert not _records(recorder.config.root)
    assert not recorder.dataset.has_pending_frames()
    assert recorder.saved_episodes == recorder.dataset.meta.total_episodes == 0

    resumed = RecapRolloutRecorder(
        recorder.robot, recorder.sample_reader, replace(recorder.config, resume=True)
    )
    resumed.begin_rollout("kept task", IDENTITY, {})
    resumed.resume_capture()
    resumed._record_one()
    resumed.finish_rollout("failure")
    resumed.close()
    (record,) = _records(recorder.config.root)
    assert record["episode_index"] == 0
    assert record["task"] == "kept task"


@pytest.mark.parametrize("outcome", ["unknown", "aborted"])
def test_unsupported_outcomes_cannot_save_a_rollout(tmp_path, outcome):
    recorder, _ = _make_recorder(tmp_path)
    recorder.begin_rollout("task", IDENTITY, {})
    recorder.resume_capture()
    recorder._record_one()
    with pytest.raises(ValueError):
        recorder.finish_rollout(outcome)
    recorder.close()
    assert not _records(recorder.config.root)
    assert recorder.dataset.meta.total_episodes == 0


def test_save_error_retains_journal_and_never_retries_or_appends(tmp_path, monkeypatch):
    recorder, _ = _make_recorder(tmp_path)
    recorder.begin_rollout("task", IDENTITY, {})
    recorder.resume_capture()
    recorder._record_one()
    calls = []

    def fail(*, parallel_encoding):
        calls.append(parallel_encoding)
        raise OSError("disk full")

    monkeypatch.setattr(recorder.dataset, "save_episode", fail)
    with pytest.raises(OSError, match="disk full"):
        recorder.finish_rollout("success")
    with pytest.raises(RuntimeError, match="retained journal"):
        recorder.discard_rollout()
    recorder.close()
    assert calls == [False]
    (record,) = _records(recorder.config.root)
    assert record["status"] == "saving"
    with pytest.raises(ValueError, match="Unfinished"):
        RecapRolloutRecorder(recorder.robot, recorder.sample_reader, replace(recorder.config, resume=True))


def test_resume_rejects_wrong_fps_and_missing_result_metadata(tmp_path):
    recorder, _ = _make_recorder(tmp_path)
    recorder.begin_rollout("task", IDENTITY, {})
    recorder.resume_capture()
    recorder._record_one()
    recorder.finish_rollout("success")
    recorder.close()
    with pytest.raises(ValueError, match="fps"):
        RecapRolloutRecorder(
            recorder.robot, recorder.sample_reader, replace(recorder.config, resume=True, fps=15)
        )
    (path,) = (recorder.config.root / "meta/recap").glob("*.json")
    record = json.loads(path.read_text())
    record["episode_index"] = 5
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="do not match"):
        RecapRolloutRecorder(recorder.robot, recorder.sample_reader, replace(recorder.config, resume=True))


def test_empty_rollout_is_discarded_and_does_not_reach_episode_target(tmp_path):
    recorder, _ = _make_recorder(tmp_path)
    recorder.begin_rollout("task", IDENTITY, {})
    assert recorder.finish_rollout("success") == 0
    assert recorder.saved_episodes == 0
    recorder.close()
    assert not _records(recorder.config.root)


def test_recording_error_rejects_result_and_discards_prefix_on_close(tmp_path, monkeypatch):
    recorder, _ = _make_recorder(tmp_path)
    recorder.begin_rollout("task", IDENTITY, {})
    recorder.resume_capture()
    recorder._record_one()

    def fail():
        raise RuntimeError("camera disconnected")

    monkeypatch.setattr(recorder.robot, "get_observation", fail)
    with pytest.raises(RuntimeError, match="camera disconnected"):
        recorder._record_one()
    assert not recorder.active
    assert str(recorder.last_error) == "camera disconnected"
    with pytest.raises(RuntimeError, match="incomplete"):
        recorder.finish_rollout("failure")
    recorder.close()
    assert not _records(recorder.config.root)
    assert not recorder.dataset.has_pending_frames()
    assert recorder.dataset.meta.total_episodes == 0


def test_discard_cleans_partial_first_frame_images_and_allows_new_episode(tmp_path, monkeypatch):
    robot = SimpleNamespace(
        name="recap_test",
        observation_features={"joint.pos": float, "front": (8, 8, 3)},
        action_features={"joint.pos": float},
        cameras={"front": object()},
        get_observation=lambda: {"joint.pos": 1.0, "front": np.zeros((8, 8, 3), dtype=np.uint8)},
    )
    config = RecapDatasetConfig(
        repo_id="test/recap", root=tmp_path / "dataset", use_videos=False, image_writer_threads_per_camera=0
    )
    recorder = RecapRolloutRecorder(robot, lambda: RecapControlSample(RecapPhase.AUTONOMOUS), config)
    recorder.begin_rollout("discarded task", IDENTITY, {})
    recorder.resume_capture()
    original_save = recorder.dataset.writer._save_image

    def fail_after_image(*args, **kwargs):
        original_save(*args, **kwargs)
        raise OSError("frame write interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(recorder.dataset.writer, "_save_image", fail_after_image)
        with pytest.raises(OSError, match="interrupted"):
            recorder._record_one()
    assert not recorder.dataset.has_pending_frames()
    assert list(config.root.rglob("*.png"))
    recorder.discard_rollout()
    assert not list(config.root.rglob("*.png"))
    assert not _records(config.root)

    recorder.begin_rollout("kept task", IDENTITY, {})
    recorder.resume_capture()
    recorder._record_one()
    assert recorder.finish_rollout("success") == 1
    recorder.close()
    dataset = LeRobotDataset(config.repo_id, root=config.root)
    assert len(dataset) == 1
    assert dataset[0]["task"] == "kept task"


def test_sidecar_publish_failure_after_dataset_commit_does_not_duplicate_episode(tmp_path, monkeypatch):
    import lerobot.async_inference.client.piper_hil_recap as recorder_module

    recorder, _ = _make_recorder(tmp_path)
    recorder.begin_rollout("task", IDENTITY, {})
    recorder.resume_capture()
    recorder._record_one()
    original_write = recorder_module.atomic_write_json

    def fail_committed(path, record):
        if record["status"] == "committed":
            raise OSError("sidecar publish failed")
        original_write(path, record)

    monkeypatch.setattr(recorder_module, "atomic_write_json", fail_committed)
    with pytest.raises(OSError, match="publish failed"):
        recorder.finish_rollout("success")
    assert recorder.dataset.meta.total_episodes == 1
    recorder.close()
    (record,) = _records(recorder.config.root)
    assert record["status"] == "saving"
    assert recorder.dataset.meta.total_episodes == 1


def test_resume_checks_schema_before_opening_a_writer(tmp_path, monkeypatch):
    recorder, _ = _make_recorder(tmp_path)
    recorder.begin_rollout("task", IDENTITY, {})
    recorder.resume_capture()
    recorder._record_one()
    recorder.finish_rollout("success")
    recorder.close()
    changed_robot = SimpleNamespace(**vars(recorder.robot))
    changed_robot.action_features = {"renamed_joint.pos": float}
    monkeypatch.setattr(LeRobotDataset, "resume", lambda *args, **kwargs: pytest.fail("opened writer"))
    with pytest.raises(ValueError, match="Incompatible RECAP feature action"):
        RecapRolloutRecorder(changed_robot, recorder.sample_reader, replace(recorder.config, resume=True))
