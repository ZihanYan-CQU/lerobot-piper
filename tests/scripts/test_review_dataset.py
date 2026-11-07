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
from unittest.mock import Mock

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
        fps=4,
        video_keys=camera_keys,
        episodes=episodes,
        get_video_file_path=lambda episode_index, key: Path("videos/camera.mp4"),
    )
    return SimpleNamespace(root=root, repo_id="test/source", meta=meta)


@pytest.fixture
def panel(video_dataset, monkeypatch):
    """Exercise panel navigation and playback without opening a desktop window."""

    def make_button(*args, **kwargs):
        button = Mock()
        button.options = kwargs
        return button

    monkeypatch.setattr(review, "ctk", SimpleNamespace(CTkButton=make_button))
    monkeypatch.setattr(review, "EpisodeVideoDecoder", Mock(side_effect=lambda *args: Mock()))
    instance = review.ReviewPanel.__new__(review.ReviewPanel)
    instance.root = Mock()
    instance.dataset = video_dataset
    instance.current_episode = 0
    instance.episode_page = 0
    instance.episode_buttons = {}
    instance.deletions = set()
    instance.saved_deletions = set()
    instance.dirty = False
    instance.decoders = {key: Mock() for key in video_dataset.meta.video_keys}
    instance.photos = {}
    instance.position_s = 1.9
    instance.duration_s = 2.0
    instance.playing = True
    instance.busy = False
    instance.scrubbing = False
    instance.resume_after_scrub = False
    instance.last_tick = 9.8
    instance.font_latin = None
    interpreter = review.tk.Tcl()
    for name, value in (
        ("episode_var", "0"),
        ("episode_page_var", ""),
        ("playback_mode_var", "单集循环"),
        ("speed_var", "1×"),
        ("status_var", ""),
        ("review_var", ""),
    ):
        setattr(instance, name, review.tk.StringVar(master=interpreter, value=value))
    for name in (
        "episode_entry",
        "jump_button",
        "previous_button",
        "next_button",
        "previous_page_button",
        "next_page_button",
        "play_button",
        "timeline",
        "playback_modes",
        "keep_button",
        "delete_button",
        "save_button",
        "open_button",
        "review_badge",
    ):
        setattr(instance, name, Mock())
    instance.episode_buttons_frame = Mock()
    instance.episode_buttons_frame.winfo_children.return_value = []
    instance._render = Mock()
    instance._refresh_deleted_episodes = Mock()
    instance._show_message = Mock()
    return instance


@pytest.mark.parametrize(
    "mode, current, total, expected, playing",
    [
        ("播完暂停", 0, 3, 0, False),
        ("单集循环", 2, 3, 2, True),
        ("连续播放", 0, 3, 1, True),
        ("连续播放", 2, 3, 0, True),
        ("连续播放", 0, 1, 0, True),
    ],
)
def test_playback_end_modes(panel, monkeypatch, mode, current, total, expected, playing):
    panel.playback_mode_var.set(mode)
    panel.current_episode = current
    panel.dataset.meta.total_episodes = total
    panel.deletions = {0} if total > 1 else set()
    original_decoders = panel.decoders.copy()
    monkeypatch.setattr(review.time, "monotonic", lambda: 10.0)

    panel._tick()

    assert panel.current_episode == expected
    assert panel.playing is playing
    assert panel.position_s == (0.0 if playing else 2.0)
    assert panel.deletions == ({0} if total > 1 else set())
    if current != expected:
        assert panel.episode_var.get() == str(expected)
        assert panel.episode_buttons[expected].options["border_width"] > 0
        for decoder in original_decoders.values():
            decoder.close.assert_called_once()
    else:
        for decoder in original_decoders.values():
            decoder.close.assert_not_called()
    panel.root.after.assert_called_once_with(review.TICK_MS, panel._tick)


@pytest.mark.parametrize("blocked", ["paused", "busy", "scrubbing", "no_video"])
def test_timer_does_not_advance_while_playback_is_blocked(panel, monkeypatch, blocked):
    panel.playback_mode_var.set("连续播放")
    if blocked == "paused":
        panel.playing = False
    elif blocked == "no_video":
        panel.decoders.clear()
    else:
        setattr(panel, blocked, True)
    monkeypatch.setattr(review.time, "monotonic", lambda: 10.0)

    panel._tick()

    assert panel.current_episode == 0
    assert panel.position_s == 1.9
    panel._render.assert_not_called()
    review.EpisodeVideoDecoder.assert_not_called()


def test_next_episode_starts_clock_after_loading(panel, monkeypatch):
    panel.playback_mode_var.set("连续播放")
    monkeypatch.setattr(review.time, "monotonic", Mock(side_effect=[10.0, 15.0]))

    panel._tick()

    assert panel.current_episode == 1
    assert panel.position_s == 0.0
    assert panel.last_tick == 15.0


def test_decoder_failure_stops_continuous_playback(panel, monkeypatch):
    panel.playback_mode_var.set("连续播放")
    opened = Mock()
    review.EpisodeVideoDecoder.side_effect = [opened, RuntimeError("broken video")]
    monkeypatch.setattr(review.time, "monotonic", lambda: 10.0)

    panel._tick()
    panel._tick()

    assert not panel.playing
    assert not panel.decoders
    opened.close.assert_called_once()
    panel._show_message.assert_called_once()
    assert review.EpisodeVideoDecoder.call_count == 2


def test_initial_render_error_does_not_restart_playback(panel):
    panel._render.side_effect = lambda: setattr(panel, "playing", False)

    panel.load_episode(1)

    assert not panel.playing
    panel.play_button.configure.assert_called_with(text="播放")


@pytest.mark.parametrize("value", ["", "abc", "-1", "3", "1.5"])
def test_invalid_episode_jump_keeps_current_playback(panel, value):
    panel.episode_var.set(value)
    panel.load_episode = Mock()

    assert panel._jump_episode() == "break"

    panel.load_episode.assert_not_called()
    assert panel.playing
    assert "0 到 2" in panel.status_var.get()
    panel.episode_entry.focus_set.assert_called_once()


def test_episode_jump_loads_submitted_number(panel):
    panel.episode_var.set(" 2 ")

    panel._jump_episode()

    assert panel.current_episode == 2
    assert panel.episode_var.get() == "2"
    assert panel.playing


def test_number_pages_can_be_browsed_without_changing_playback(panel):
    panel.dataset.meta.total_episodes = 23
    panel.load_episode = Mock()
    panel._build_episode_buttons()
    assert list(panel.episode_buttons) == list(range(10))

    panel._move_episode_page(1)
    assert list(panel.episode_buttons) == list(range(10, 20))
    panel._move_episode_page(1)
    assert list(panel.episode_buttons) == [20, 21, 22]
    assert panel.episode_page_var.get() == "3 / 3 页"
    panel.next_page_button.configure.assert_called_with(state="disabled")
    panel._move_episode_page(1)
    assert panel.episode_page == 2
    assert panel.current_episode == 0
    assert panel.playing
    panel.load_episode.assert_not_called()

    panel.episode_buttons[21].options["command"]()
    panel.load_episode.assert_called_once_with(21)
    panel._move_episode_page(-10)
    assert panel.episode_page == 0
    panel.previous_page_button.configure.assert_called_with(state="disabled")


def test_loading_episode_reveals_its_number_page(panel, monkeypatch):
    panel.dataset.meta.total_episodes = 23
    monkeypatch.setattr(review, "video_segment", lambda *args: (Path("video.mp4"), 0.0, 2.0))

    panel.load_episode(22)

    assert panel.current_episode == 22
    assert panel.episode_page == 2
    assert list(panel.episode_buttons) == [20, 21, 22]
    assert panel.episode_buttons[22].options["border_width"] > 0


def test_number_buttons_reflect_deletion_and_disable_during_save(panel):
    panel.mark_current(True)

    assert panel.deletions == {0}
    assert panel.episode_buttons[0].options["fg_color"] == review.UI_COLORS["danger"]
    panel.mark_current(False)
    assert panel.deletions == set()
    assert panel.episode_buttons[0].options["fg_color"] == review.UI_COLORS["accent"]

    panel.busy = True
    panel._update_action_state()
    for button in panel.episode_buttons.values():
        button.configure.assert_called_with(state="disabled")
    panel.episode_entry.configure.assert_called_with(state="disabled")
    panel.play_button.configure.assert_called_with(state="disabled")


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
