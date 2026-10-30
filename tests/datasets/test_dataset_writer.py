#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Contract tests for DatasetWriter."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs import VideoEncoderConfig
from lerobot.datasets.dataset_writer import _encode_video_worker
from lerobot.datasets.io_utils import EPISODE_SAVE_MARKER, validate_recorded_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_IMAGE_PATH
from tests.fixtures.constants import DEFAULT_FPS, DUMMY_REPO_ID

SIMPLE_FEATURES = {
    "state": {"dtype": "float32", "shape": (6,), "names": None},
    "action": {"dtype": "float32", "shape": (6,), "names": None},
}


def _make_frame(features: dict, task: str = "Dummy task") -> dict:
    """Build a valid frame dict for the given features."""
    frame = {"task": task}
    for key, ft in features.items():
        if ft["dtype"] in ("image", "video"):
            frame[key] = np.random.randint(0, 256, size=ft["shape"], dtype=np.uint8)
        elif ft["dtype"] in ("float32", "float64"):
            frame[key] = torch.randn(ft["shape"])
        elif ft["dtype"] == "int64":
            frame[key] = torch.zeros(ft["shape"], dtype=torch.int64)
    return frame


# ── Existing encode_video_worker tests ───────────────────────────────


def test_encode_video_worker_forwards_video_encoder(tmp_path):
    """_encode_video_worker forwards video_encoder to encode_video_frames."""
    video_key = "observation.images.laptop"
    fpath = DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=0, frame_index=0)
    img_dir = tmp_path / Path(fpath).parent
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color="red").save(img_dir / "frame-000000.png")

    captured_kwargs = {}

    def mock_encode(imgs_dir, video_path, fps, **kwargs):
        captured_kwargs.update(kwargs)
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(video_path).touch()

    with patch("lerobot.datasets.dataset_writer.encode_video_frames", side_effect=mock_encode):
        _encode_video_worker(
            video_key,
            0,
            tmp_path,
            fps=30,
            video_encoder=VideoEncoderConfig(vcodec="h264", preset=None),
            encoder_threads=4,
        )

    assert captured_kwargs["video_encoder"].vcodec == "h264"
    assert captured_kwargs["encoder_threads"] == 4


def test_encode_video_worker_default_video_encoder(tmp_path):
    """_encode_video_worker passes None video_encoder which encode_video_frames defaults."""
    video_key = "observation.images.laptop"
    fpath = DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=0, frame_index=0)
    img_dir = tmp_path / Path(fpath).parent
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color="red").save(img_dir / "frame-000000.png")

    captured_kwargs = {}

    def mock_encode(imgs_dir, video_path, fps, **kwargs):
        captured_kwargs.update(kwargs)
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(video_path).touch()

    with patch("lerobot.datasets.dataset_writer.encode_video_frames", side_effect=mock_encode):
        _encode_video_worker(video_key, 0, tmp_path, fps=30)

    assert captured_kwargs["video_encoder"] is None
    assert captured_kwargs["encoder_threads"] is None


# ── add_frame contracts ──────────────────────────────────────────────


def test_add_frame_increments_buffer_size(tmp_path):
    """Each add_frame() call increases episode_buffer['size'] by 1."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    assert dataset.writer.episode_buffer["size"] == 0

    dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    assert dataset.writer.episode_buffer["size"] == 1

    dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    assert dataset.writer.episode_buffer["size"] == 2


def test_add_frame_rejects_missing_feature(tmp_path):
    """add_frame() raises ValueError when a required feature is missing."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    with pytest.raises(ValueError, match="Missing features"):
        dataset.add_frame({"task": "Dummy task", "state": torch.randn(6)})
        # missing 'action'


# ── save_episode contracts ───────────────────────────────────────────


def test_save_episode_writes_parquet(tmp_path):
    """After save_episode(), at least one .parquet file exists under data/."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(3):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()

    parquet_files = list((tmp_path / "ds" / "data").rglob("*.parquet"))
    assert len(parquet_files) > 0


def test_save_episode_updates_counters(tmp_path):
    """After save_episode(), metadata counters are updated."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(5):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()

    assert dataset.meta.total_episodes == 1
    assert dataset.meta.total_frames == 5


def test_save_episode_resets_buffer(tmp_path):
    """After save_episode(), the episode buffer is reset."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(3):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()

    assert dataset.writer.episode_buffer["size"] == 0


def test_save_multiple_episodes(tmp_path):
    """Recording 3 episodes results in correct total counts."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    total_frames = 0
    for ep in range(3):
        n_frames = ep + 2  # 2, 3, 4
        for _ in range(n_frames):
            dataset.add_frame(_make_frame(SIMPLE_FEATURES))
        dataset.save_episode()
        total_frames += n_frames

    assert dataset.meta.total_episodes == 3
    assert dataset.meta.total_frames == total_frames


# ── clear / lifecycle ────────────────────────────────────────────────


def test_clear_resets_buffer(tmp_path):
    """clear_episode_buffer() resets the buffer size to 0."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    assert dataset.writer.episode_buffer["size"] == 1

    dataset.clear_episode_buffer()
    assert dataset.writer.episode_buffer["size"] == 0


def test_clear_removes_video_frame_staging_dir(tmp_path):
    """clear_episode_buffer() removes PNG staging dirs for video features."""
    video_key = "observation.images.cam"
    features = {
        video_key: {
            "dtype": "video",
            "shape": (64, 96, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID,
        fps=DEFAULT_FPS,
        features=features,
        root=tmp_path / "ds",
        use_videos=True,
    )

    dataset.add_frame(_make_frame(features))
    video_staging_dir = (
        dataset.root
        / Path(DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=0, frame_index=0)).parent
    )
    assert video_staging_dir.is_dir()

    dataset.clear_episode_buffer()

    assert dataset.writer.episode_buffer["size"] == 0
    assert not video_staging_dir.exists()


def test_batched_encoding_staging_survives_save(tmp_path):
    """The post-save clear must NOT delete video staging frames.

    With ``batch_encoding_size > 1`` the frames of already-saved episodes stay
    on disk until the batch encode runs; the encoder deletes them afterwards.
    A blanket switch of the post-save cleanup to ``camera_keys`` (as done in the
    discard path) would silently break batched encoding.
    """
    video_key = "observation.images.cam"
    features = {
        video_key: {
            "dtype": "video",
            "shape": (64, 96, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID,
        fps=DEFAULT_FPS,
        features=features,
        root=tmp_path / "ds",
        use_videos=True,
        batch_encoding_size=2,
    )
    for _ in range(3):
        dataset.add_frame(_make_frame(features))

    staging_dir = dataset.writer._get_image_file_dir(0, video_key)
    assert staging_dir.is_dir()

    dataset.save_episode()  # first of a batch of 2: no encoding yet

    assert staging_dir.is_dir() and any(staging_dir.iterdir())


def test_finalize_is_idempotent(tmp_path):
    """Calling finalize() twice does not raise."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(3):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()

    dataset.finalize()
    dataset.finalize()  # second call should not raise


def test_finalize_then_read_roundtrip(tmp_path):
    """Write data, finalize, re-open, and verify data matches."""
    root = tmp_path / "roundtrip"
    features = {"state": {"dtype": "float32", "shape": (2,), "names": None}}
    dataset = LeRobotDataset.create(repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=features, root=root)

    # Record known values
    known_states = []
    for i in range(5):
        state = torch.tensor([float(i), float(i * 10)])
        known_states.append(state)
        dataset.add_frame({"task": "Test task", "state": state})
    dataset.save_episode()
    dataset.finalize()

    # Read back
    for i in range(5):
        item = dataset[i]
        assert torch.allclose(item["state"], known_states[i], atol=1e-5)


@pytest.mark.parametrize("parallel_encoding", [False, True])
def test_second_camera_failure_does_not_append_data_or_videos(tmp_path, monkeypatch, parallel_encoding):
    """A discarded correction must not shift observations/actions of later episodes."""
    from lerobot.datasets import dataset_writer

    features = {
        **SIMPLE_FEATURES,
        **{
            f"observation.images.{camera}": {
                "dtype": "video",
                "shape": (64, 96, 3),
                "names": ["height", "width", "channels"],
            }
            for camera in ("top", "wrist")
        },
    }
    dataset = LeRobotDataset.create(
        DUMMY_REPO_ID,
        DEFAULT_FPS,
        features,
        root=tmp_path / "ds",
        rgb_encoder=VideoEncoderConfig(vcodec="h264", preset="ultrafast"),
        encoder_threads=1,
    )
    # Exercise both scheduling branches without forking pytest or hardware threads.
    monkeypatch.setattr(dataset_writer.concurrent.futures, "ProcessPoolExecutor", ThreadPoolExecutor)

    def record(value):
        for _ in range(3):
            frame = _make_frame(features)
            frame["state"] = np.full(6, value, dtype=np.float32)
            frame["action"] = np.full(6, value + 1, dtype=np.float32)
            dataset.add_frame(frame)

    record(10)
    dataset.save_episode(parallel_encoding=parallel_encoding)
    videos_before = {path: path.read_bytes() for path in dataset.root.glob("videos/*/*/*.mp4")}
    assert len(videos_before) == 2
    encode = dataset_writer.encode_video_frames
    staged_paths = []

    def fail_wrist(img_dir, video_path, fps, **kwargs):
        staged_paths.append(video_path.parent)
        if "observation.images.wrist" in str(img_dir):
            video_path.touch()  # The encoder may leave partial output before failing.
            raise RuntimeError("injected wrist encoding failure")
        return encode(img_dir, video_path, fps, **kwargs)

    record(99)
    with monkeypatch.context() as fault:
        fault.setattr(dataset_writer, "encode_video_frames", fail_wrist)
        with pytest.raises(RuntimeError, match="injected wrist encoding failure"):
            dataset.save_episode(parallel_encoding=parallel_encoding)
    assert dataset.meta.total_frames == 3
    assert dataset.meta.total_episodes == 1
    assert dataset.writer._recorded_frames == 3
    assert dataset.has_pending_frames()
    assert dataset.writer.episode_buffer["size"] == 3
    assert not (dataset.root / EPISODE_SAVE_MARKER).exists()
    assert all(not path.exists() for path in staged_paths)
    assert {path: path.read_bytes() for path in dataset.root.glob("videos/*/*/*.mp4")} == videos_before

    dataset.clear_episode_buffer()
    assert not list(dataset.root.glob("images/*/episode-000001/*.png"))
    record(20)
    dataset.save_episode(parallel_encoding=parallel_encoding)
    dataset.finalize()

    table = pq.read_table(dataset.root / "data")
    assert table["index"].to_pylist() == list(range(6))
    assert table["episode_index"].to_pylist() == [0] * 3 + [1] * 3
    assert table["frame_index"].to_pylist() == [0, 1, 2] * 2
    assert table["action"].to_pylist() == [[11.0] * 6] * 3 + [[21.0] * 6] * 3
    assert dataset.meta.total_frames == 6
    resumed = LeRobotDataset.resume(DUMMY_REPO_ID, root=dataset.root)
    validate_recorded_dataset(resumed.root, resumed.meta.episodes, 6, 2)
    resumed.finalize()


@pytest.mark.parametrize("failure_stage", ["data", "metadata"])
def test_interrupted_commit_blocks_more_recording_and_resume(tmp_path, monkeypatch, failure_stage):
    dataset = LeRobotDataset.create(DUMMY_REPO_ID, DEFAULT_FPS, SIMPLE_FEATURES, root=tmp_path / "ds")
    dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()
    dataset.add_frame(_make_frame(SIMPLE_FEATURES))

    def fail(*args, **kwargs):
        raise OSError("injected persistence failure")

    if failure_stage == "data":
        monkeypatch.setattr(dataset.writer, "_save_episode_data", fail)
    else:
        monkeypatch.setattr(dataset.meta, "save_episode", fail)
    with pytest.raises(OSError, match="injected persistence failure"):
        dataset.save_episode()
    marker = dataset.root / EPISODE_SAVE_MARKER
    assert json.loads(marker.read_text()) == {
        "episode_index": 1,
        "length": 1,
        "previous_total_frames": 1,
    }
    assert dataset.has_pending_frames()
    dataset.clear_episode_buffer()
    with pytest.raises(RuntimeError, match="interrupted episode save"):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    with pytest.raises(RuntimeError, match="interrupted episode save"):
        dataset.save_episode()
    dataset.finalize()
    with pytest.raises(ValueError, match="interrupted episode save"):
        LeRobotDataset.resume(DUMMY_REPO_ID, root=dataset.root)
    assert marker.exists()


def test_resume_rejects_legacy_unregistered_rows(tmp_path):
    dataset = LeRobotDataset.create(DUMMY_REPO_ID, DEFAULT_FPS, SIMPLE_FEATURES, root=tmp_path / "ds")
    for _ in range(2):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
        dataset.save_episode()
    dataset.finalize()
    data_path = next(dataset.root.glob("data/*/*.parquet"))
    table = pq.read_table(data_path)
    pq.write_table(table.take([0, 1, 1]), data_path)
    with pytest.raises(ValueError, match="Parquet has 3 rows but metadata declares 2 frames"):
        LeRobotDataset.resume(DUMMY_REPO_ID, root=dataset.root)
