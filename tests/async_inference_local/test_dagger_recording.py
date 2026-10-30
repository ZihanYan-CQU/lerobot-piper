import shutil
import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_dagger_runtime import Robot, config
from test_piper_dagger import _FakeDataset, _pose

from lerobot.async_inference_local.control.piper_dagger import PiperDaggerHardwareConfig
from lerobot.async_inference_local.recording.correction import DaggerCorrectionRecorder, DaggerDatasetConfig


@pytest.mark.parametrize(
    "change",
    [
        {"image_writer_processes": 1},
        {"image_writer_processes": -1},
        {"fps": -1},
        {"fps": float("nan")},
        {"fps": 2.5},
    ],
)
def test_recording_config_prevents_fork_and_invalid_timestamps(change):
    with pytest.raises(ValueError):
        DaggerDatasetConfig(**change)


def test_dataset_fps_inherits_policy_only_when_integer():
    assert config(dataset=DaggerDatasetConfig(fps=0)).dataset.fps == 30
    with pytest.raises(ValueError, match="integer"):
        config(fps=29.97, dataset=DaggerDatasetConfig(fps=0))
    assert config(fps=29.97, dataset=DaggerDatasetConfig(fps=30)).dataset.fps == 30


@pytest.mark.parametrize(
    "change",
    [
        {"num_episodes": -1},
        {"input_device": "unknown"},
        {"advance_key": "esc"},
        {"pedal_reset": "KEY_A"},
        {"robot": SimpleNamespace(type="so101_follower")},
        {"hardware": PiperDaggerHardwareConfig(follower_can_name="wrong_bus")},
        {"dataset": DaggerDatasetConfig(), "num_episodes": 3},
    ],
)
def test_dagger_configuration_rejects_unusable_capture_settings(change):
    with pytest.raises(ValueError):
        config(**change)


@pytest.mark.parametrize(
    "field", ["feedback_timeout_s", "handover_joint_tolerance", "handover_gripper_settle_s"]
)
def test_hardware_validation_rejects_nonfinite_deadlines(field):
    with pytest.raises(ValueError, match="finite"):
        PiperDaggerHardwareConfig(**{field: float("nan")})


def make_recorder(robot, expert_reader):
    cfg = DaggerDatasetConfig(use_videos=False, fps=30)
    template = DaggerCorrectionRecorder(robot, expert_reader, cfg)
    dataset = _FakeDataset(template._dataset_features())
    return DaggerCorrectionRecorder(robot, expert_reader, cfg, dataset=dataset), dataset


def test_capture_owns_recycled_images_before_reading_expert_labels():
    robot = Robot()
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    robot.observation_features = {**robot.observation_features, "camera": (4, 4, 3)}
    robot.get_observation = lambda: {**robot.pose, "camera": image}

    def expert():
        image[:] = 255
        return _pose(1.0)

    recorder, dataset = make_recorder(robot, expert)
    recorder.begin("recover")
    recorder._record_one()
    assert not dataset.frames[0]["observation.images.camera"].any()
    np.testing.assert_equal(dataset.frames[0]["action"], np.ones(7))
    recorder.close()


@pytest.mark.parametrize("source", ["action", "observation"])
def test_nonfinite_capture_is_rejected_before_writing_a_frame(source):
    robot = Robot()
    expert = _pose(1.0)
    (expert if source == "action" else robot.pose)["joint_1.pos"] = float("nan")
    recorder, dataset = make_recorder(robot, lambda: expert)
    recorder.begin("recover")
    with pytest.raises(ValueError, match="non-finite"):
        recorder._record_one()
    assert not dataset.frames
    recorder.close()


def test_stopping_a_sample_waiting_for_observation_prevents_hardware_access():
    robot = Robot()
    recorder, dataset = make_recorder(robot, lambda: _pose(1.0))
    recorder.begin("recover")
    entered = threading.Event()
    recorder._observation_lock.acquire()

    def record():
        entered.set()
        recorder._record_one()

    worker = threading.Thread(target=record, daemon=True)
    worker.start()
    assert entered.wait(1)
    recorder.request_pause()
    robot.closed.set()
    recorder._observation_lock.release()
    worker.join(1)
    assert not worker.is_alive()
    assert not dataset.frames
    recorder.close()


@pytest.mark.parametrize("mismatch", ["fps", "shape", "names", "dtype", "extra", "missing"])
def test_resume_rejects_incompatible_schema_or_timebase(mismatch):
    recorder, dataset = make_recorder(Robot(), lambda: _pose(1.0))
    meta = SimpleNamespace(fps=30, features={key: dict(v) for key, v in dataset.meta.features.items()})
    if mismatch == "fps":
        meta.fps = 15
    elif mismatch == "extra":
        meta.features["reward"] = {"dtype": "float32", "shape": (1,)}
    elif mismatch == "missing":
        meta.features.pop("intervention")
    else:
        meta.features["action"][mismatch] = {"shape": (8,), "names": ["wrong"], "dtype": "int32"}[mismatch]
    with pytest.raises(ValueError):
        recorder._validate_resume(meta)


@pytest.mark.parametrize("use_videos", [False, True])
def test_real_dataset_can_save_finalize_and_resume_corrections(tmp_path, monkeypatch, use_videos):
    if use_videos and shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is needed for video capture")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    robot = Robot()
    robot.cameras = {"top": None, "wrist": None}
    robot.observation_features = {**robot.observation_features, "top": (64, 64, 3), "wrist": (64, 64, 3)}
    frame = np.full((64, 64, 3), 96, dtype=np.uint8)
    robot.get_observation = lambda: {**robot.pose, "top": frame, "wrist": frame}
    cfg = DaggerDatasetConfig(
        repo_id="local/real_corrections", root=tmp_path / "data", fps=30, use_videos=use_videos
    )
    recorder = DaggerCorrectionRecorder(robot, lambda: _pose(0.8), cfg)
    for _ in range(2):
        recorder.begin("recover cube")
        for _ in range(4):
            recorder._record_one()
        recorder.end(save=True)
    recorder.close()
    meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.root)
    assert meta.total_episodes == 2 and meta.total_frames == 8
    assert meta.features["intervention"]["dtype"] == "bool"
    if use_videos:
        assert len(list(cfg.root.rglob("*.mp4"))) >= 2
    resumed = DaggerCorrectionRecorder(robot, lambda: _pose(0.9), replace(cfg, resume=True))
    assert resumed.saved_episodes == 0
    resumed.begin("recover again")
    for _ in range(4):
        resumed._record_one()
    resumed.end(save=True)
    resumed.close()
    assert resumed.saved_episodes == 1
    meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.root)
    assert meta.total_episodes == 3 and meta.total_frames == 12
