#!/usr/bin/env python

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

from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest

from lerobot.yzh import lerobot_review_dataset as review


@pytest.fixture
def video_dataset(tmp_path):
    root = tmp_path / "source"
    video_path = root / "videos" / "camera.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video")
    camera_keys = ["observation.images.top", "observation.images.wrist"]
    episodes = [
        {
            f"videos/{key}/{boundary}_timestamp": timestamp
            for key in camera_keys
            for boundary, timestamp in (("from", float(index * 2)), ("to", float((index + 1) * 2)))
        }
        for index in range(3)
    ]
    meta = SimpleNamespace(
        total_episodes=3,
        video_keys=camera_keys,
        episodes=episodes,
        get_video_file_path=lambda episode_index, key: Path("videos/camera.mp4"),
    )
    return SimpleNamespace(root=root, repo_id="test/source", meta=meta)


def test_video_segment_uses_episode_range_for_each_camera(video_dataset):
    for key in video_dataset.meta.video_keys:
        path, start, duration = review.video_segment(video_dataset, 1, key)
        assert path == video_dataset.root / "videos/camera.mp4"
        assert (start, duration) == (2.0, 2.0)


def test_decoder_seeks_exact_episode_and_can_seek_backward(tmp_path):
    video_path = tmp_path / "shared.mp4"
    with av.open(str(video_path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(8):
            pixels = np.zeros((16, 16, 3), dtype=np.uint8)
            pixels[:, :, 0 if index < 4 else 1] = 255
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    decoder = review.EpisodeVideoDecoder(video_path, start_s=1.0, duration_s=1.0, fps=4)
    try:
        for position in (0.0, 0.75, 0.0):
            pixels = np.asarray(decoder.frame_at(position))
            assert pixels[:, :, 1].mean() > 200
            assert pixels[:, :, 0].mean() < 30
    finally:
        decoder.close()


def test_deletion_marks_can_be_reversed_but_not_cover_every_episode():
    marked = review.update_deletions(set(), 1, 3, True)
    assert marked == {1}
    assert review.update_deletions(marked, 1, 3, False) == set()
    with pytest.raises(ValueError, match="至少需要保留一个"):
        review.update_deletions({0, 1}, 2, 3, True)


def test_default_output_avoids_existing_dataset_and_infers_repo_id(monkeypatch, tmp_path):
    home = tmp_path / "lerobot"
    root = home / "block" / "shape-matching_70-80"
    root.mkdir(parents=True)
    (root.parent / "shape-matching_70-80_filtered").mkdir()
    monkeypatch.setattr(review, "HF_LEROBOT_HOME", home)

    target = review.default_output_path(root)

    assert review.infer_repo_id(root) == "block/shape-matching_70-80"
    assert target.name == "shape-matching_70-80_filtered_2"
    assert (
        review.output_repo_id("block/shape-matching_70-80", target) == "block/shape-matching_70-80_filtered_2"
    )


def test_save_filtered_dataset_writes_once_to_new_location(video_dataset, tmp_path, monkeypatch):
    calls = []

    def fake_delete(dataset, indices, output_dir, repo_id):
        calls.append((dataset, indices, repo_id))
        output_dir.mkdir()
        (output_dir / "result.txt").write_text("filtered")

    monkeypatch.setattr(review, "delete_episodes", fake_delete)
    target = tmp_path / "filtered"
    result = review.save_filtered_dataset(video_dataset, {2, 0}, "test/filtered", target)

    assert result == target
    assert calls == [(video_dataset, [0, 2], "test/filtered")]
    assert (target / "result.txt").read_text() == "filtered"
    assert (video_dataset.root / "videos/camera.mp4").is_file()


def test_output_path_rejects_existing_or_nested_dataset(video_dataset, tmp_path):
    with pytest.raises(ValueError, match="输出目录"):
        review.check_output_path(video_dataset.root, video_dataset.root / "filtered")

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="已存在"):
        review.check_output_path(video_dataset.root, existing)
