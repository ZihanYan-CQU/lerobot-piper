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

"""Correction-only recording in the local GPU/control process."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DaggerDatasetConfig:
    """Correction-only LeRobot dataset settings.

    ``repo_id=None`` disables recording while retaining the complete takeover
    state machine.  This is useful for hardware validation before collecting a
    real DAgger dataset.
    """

    repo_id: str | None = None
    root: str | Path | None = None
    # Keep the nested dataclass default consistent with CLI inheritance. Draccus
    # reconstructs this class when any dataset.* option is provided.
    fps: int = 0
    use_videos: bool = True
    resume: bool = False
    push_to_hub: bool = False
    private: bool | None = None
    image_writer_processes: int = 0
    image_writer_threads_per_camera: int = 4

    def __post_init__(self) -> None:
        """Reject invalid handover or recording settings before opening hardware."""
        if not isinstance(self.fps, int) or self.fps < 0:
            raise ValueError("DAgger dataset fps must be a nonnegative integer (0 inherits policy fps)")
        if self.image_writer_processes != 0:
            raise ValueError("Local DAgger requires image_writer_processes=0: do not fork a CUDA/CAN process")
        if self.image_writer_threads_per_camera < 0:
            raise ValueError("image_writer_threads_per_camera must be non-negative")
        if self.resume and self.root is None:
            raise ValueError("resuming a DAgger dataset requires an explicit root")


class DaggerCorrectionRecorder:
    """Sample correction frames away from the CAN/control path.

    Each ``begin``/``end(save=True)`` window becomes one LeRobot episode.  The
    label is the master command, while the observation is captured from the
    follower, matching the official corrections-only DAgger strategy.
    """

    def __init__(
        self,
        follower_robot: Any,
        expert_action_reader: Callable[[], dict[str, float]],
        config: DaggerDatasetConfig,
        *,
        observation_lock: threading.Lock | threading.RLock | None = None,
        observation_reader: Callable[[], dict] | None = None,
        dataset: Any | None = None,
    ) -> None:
        """Create the adapter and its synchronization state."""
        if config.fps <= 0:
            raise ValueError("Resolve dataset.fps before constructing the recorder")
        self.robot = follower_robot
        self._observation_reader = observation_reader or follower_robot.get_observation
        self._owns_observation = observation_reader is not None
        self._closed = False
        self.expert_action_reader = expert_action_reader
        self.config = config
        self._observation_lock = observation_lock or threading.Lock()
        self._sample_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._active = threading.Event()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._task = ""
        self._frames_in_window = 0
        self._saved_episodes = 0
        self._last_error: BaseException | None = None
        self.features: dict[str, dict] = {}
        self.dataset = dataset

        if config.repo_id is not None and self.dataset is None:
            self.dataset = self._create_dataset()
        if self.dataset is not None:
            self.features = dict(self.dataset.meta.features)

    @property
    def enabled(self) -> bool:
        """Return whether a dataset is configured for writing."""
        return self.dataset is not None

    @property
    def active(self) -> bool:
        """Return whether new correction samples may be captured."""
        return self._active.is_set()

    @property
    def frames_in_window(self) -> int:
        """Return the number of frames in the current correction window."""
        with self._state_lock:
            return self._frames_in_window

    @property
    def saved_episodes(self) -> int:
        """Return the number of windows committed during this run."""
        with self._state_lock:
            return self._saved_episodes

    @property
    def last_error(self) -> BaseException | None:
        """Return the most recent capture or save failure."""
        with self._state_lock:
            return self._last_error

    def _dataset_features(self) -> dict[str, dict]:
        observation_features = hw_to_dataset_features(
            self.robot.observation_features,
            OBS_STR,
            use_video=self.config.use_videos,
        )
        action_features = hw_to_dataset_features(
            self.robot.action_features,
            ACTION,
            use_video=self.config.use_videos,
        )
        return {
            **observation_features,
            **action_features,
            "intervention": {"dtype": "bool", "shape": (1,), "names": None},
        }

    def _create_dataset(self) -> Any:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:  # pragma: no cover - optional dataset stack
            raise RuntimeError("LeRobot dataset dependencies are required for DAgger recording") from exc

        camera_count = len(getattr(self.robot, "cameras", {}))
        writer_threads = self.config.image_writer_threads_per_camera * camera_count
        if self.config.resume:
            from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

            # Validate before opening a writer or allocating image worker threads.
            meta = LeRobotDatasetMetadata(self.config.repo_id, root=self.config.root)
            self._validate_resume(meta)
            return LeRobotDataset.resume(
                self.config.repo_id,
                root=self.config.root,
                image_writer_processes=self.config.image_writer_processes,
                image_writer_threads=writer_threads,
            )

        return LeRobotDataset.create(
            self.config.repo_id,
            self.config.fps,
            root=self.config.root,
            robot_type=getattr(self.robot, "name", "piper_follower"),
            features=self._dataset_features(),
            use_videos=self.config.use_videos,
            image_writer_processes=self.config.image_writer_processes,
            image_writer_threads=writer_threads,
        )

    def _validate_resume(self, meta: Any) -> None:
        if meta.fps != self.config.fps:
            raise ValueError(f"Dataset fps {meta.fps} does not match recording fps {self.config.fps}")
        expected = self._dataset_features()
        automatic = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
        if set(meta.features) - automatic != set(expected):
            raise ValueError("Resumed dataset features do not match correction capture")
        for key, feature in expected.items():
            actual = meta.features[key]
            if (
                actual["dtype"] != feature["dtype"]
                or tuple(actual["shape"]) != tuple(feature["shape"])
                or actual.get("names") != feature.get("names")
            ):
                raise ValueError(f"Resumed dataset feature does not match: {key}")

    def request_pause(self) -> None:
        """Revoke new samples immediately, even while an episode is encoding."""
        self._active.clear()

    def acknowledge_error(self) -> None:
        """Clear a handled failure only after the operator discarded its window and reset."""
        with self._sample_lock, self._state_lock:
            if self._active.is_set() or self._frames_in_window:
                raise RuntimeError("Cannot acknowledge an active correction window")
            self._last_error = None

    def start(self) -> None:
        """Start the idle sampling worker when recording is enabled."""
        if not self.enabled or self._thread is not None:
            return
        self._shutdown.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="PiperDaggerRecorder")
        self._thread.start()

    def begin(self, task: str) -> None:
        """Open a clean correction window with one immutable task label."""
        if not self.enabled:
            return
        with self._sample_lock, self._state_lock:
            if self.dataset.has_pending_frames():
                self.dataset.clear_episode_buffer()
            self._task = task
            self._frames_in_window = 0
            self._last_error = None
            self._active.set()

    def end(self, *, save: bool) -> int:
        """Pause sampling, then save or discard the current window."""
        if not self.enabled:
            return 0
        self.pause_capture()
        return self.finish_window(save=save)

    def pause_capture(self) -> None:
        """Stop sampling and wait until a possibly active sample has completed."""
        if not self.enabled:
            return
        self._active.clear()
        with self._sample_lock:
            pass

    def finish_window(self, *, save: bool) -> int:
        """Persist or discard an already-paused correction window."""
        if not self.enabled:
            return 0
        with self._sample_lock:
            with self._state_lock:
                frames = self._frames_in_window
            if self.dataset.has_pending_frames():
                if save and frames > 0 and self.last_error is None:
                    # ProcessPoolExecutor forks the live process when multiple
                    # cameras are encoded in parallel.  Forking a process that
                    # owns Piper CAN reader threads can invalidate the SDK
                    # communication context.  Sequential camera encoding still
                    # delegates the heavy work to ffmpeg without forking this
                    # hardware process.
                    try:
                        self.dataset.save_episode(parallel_encoding=False)
                    except BaseException as exc:
                        self._active.clear()
                        with self._state_lock:
                            self._last_error = exc
                            self._frames_in_window = 0
                        # Do not retry a failed save as a partial episode during
                        # shutdown. The writer retains a marker if commit began.
                        try:
                            self.dataset.clear_episode_buffer()
                        except Exception:
                            logger.exception("Failed to discard DAgger frames after a save error")
                        raise
                    with self._state_lock:
                        self._saved_episodes += 1
                else:
                    self.dataset.clear_episode_buffer()
            with self._state_lock:
                self._frames_in_window = 0
        return frames

    def _run(self) -> None:
        period = 1.0 / self.config.fps
        next_tick = time.monotonic()
        while not self._shutdown.is_set():
            if not self._active.wait(timeout=0.05):
                next_tick = time.monotonic()
                continue
            if self._shutdown.is_set():
                break
            next_tick = max(next_tick + period, time.monotonic())
            try:
                self._record_one()
            except Exception:
                logger.exception("DAgger correction recording stopped after an error")
            precise_sleep(max(0.0, next_tick - time.monotonic()))

    def _record_one(self) -> None:
        with self._sample_lock:
            if not self._active.is_set():
                return
            try:
                self._capture_frame()
            except Exception as exc:
                # Publish the failure before releasing the capture barrier.
                # Otherwise a delayed error could invalidate the next window
                # after pause/discard/reset has already acknowledged this one.
                with self._state_lock:
                    self._last_error = exc
                self._active.clear()
                raise

    def _capture_frame(self) -> None:
        """Capture one frame while the caller holds the sample lock."""
        with self._observation_lock:
            # Shutdown may have revoked capture while this sample waited for I/O.
            if not self._active.is_set():
                return
            observation = self._observation_reader()
            # Own camera buffers before reading another SDK stream.
            if not self._owns_observation:
                observation = {
                    k: v.copy() if isinstance(v, np.ndarray) else v for k, v in observation.items()
                }
            expert_action = self.expert_action_reader()
        if not all(math.isfinite(float(expert_action[k])) for k in self.robot.action_features):
            raise ValueError("Expert action contains non-finite positions")
        if not all(math.isfinite(float(observation[k])) for k in self.robot.action_features):
            raise ValueError("Follower observation contains non-finite positions")
        observation_frame = build_dataset_frame(self.features, observation, prefix=OBS_STR)
        action_frame = build_dataset_frame(self.features, expert_action, prefix=ACTION)
        with self._state_lock:
            task = self._task
        self.dataset.add_frame(
            {
                **observation_frame,
                **action_frame,
                "task": task,
                "intervention": np.array([True], dtype=bool),
            }
        )
        with self._state_lock:
            self._frames_in_window += 1

    def close(self, *, save_partial: bool = False) -> None:
        """Stop sampling and finalize the dataset once, optionally saving a partial window."""
        if self._closed or not self.enabled:
            return
        self._closed = True
        try:
            self.end(save=save_partial)
        finally:
            self._shutdown.set()
            self._active.set()
            thread = self._thread
            if thread is not None:
                thread.join(timeout=3.0)
            self._thread = None
            self._active.clear()
            self.dataset.finalize()
        if self.config.push_to_hub and self.last_error is None:
            self.dataset.push_to_hub(private=self.config.private)
