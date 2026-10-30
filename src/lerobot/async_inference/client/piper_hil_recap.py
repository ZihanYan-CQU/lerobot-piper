# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Record whole task rollouts separately from correction-only DAgger datasets."""

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.utils.constants import ACTION, DEFAULT_FEATURES, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

from ..shared.metadata_recap import atomic_write_json, validate_policy_identity
from .piper_hil_dagger import DaggerCorrectionRecorder, DaggerDatasetConfig

logger = logging.getLogger(__name__)
RECAP_METADATA_SCHEMA_VERSION = 2


class RecapPhase(IntEnum):
    AUTONOMOUS = 0
    INTERVENTION = 1
    PAUSED = 2


class RecapOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class RecapControlSample:
    phase: RecapPhase
    action: dict[str, float] | None = None
    generation: int = -1
    policy_step: int = -1
    source_observation_id: int = -1
    command_monotonic: float | None = None


@dataclass(frozen=True)
class RecapDatasetConfig(DaggerDatasetConfig):
    def __post_init__(self):
        super().__post_init__()
        if not self.repo_id or self.root is None:
            raise ValueError("RECAP requires both dataset repo_id and an explicit local root")
        if self.image_writer_processes:
            raise ValueError("RECAP uses image writer threads; do not fork the live CAN process")


class RecapRolloutRecorder(DaggerCorrectionRecorder):
    """Reuse dataset construction, never DAgger's correction-window semantics.

    The independent sampler stores the latest dispatched command at observation
    completion, not the raw policy chunk. Handovers can leave gaps; elapsed times
    and segment ids expose them. Paused/no-command frames have action_valid=False.
    Result/model metadata live in atomic JSON sidecars keyed by rollout UUID.
    """

    def __init__(
        self,
        follower_robot: Any,
        sample_reader: Callable[[], RecapControlSample],
        config: RecapDatasetConfig,
        *,
        observation_lock: Any = None,
        dataset: Any = None,
    ):
        root = Path(config.root)
        # Validate a local resume before opening writer threads or changing files.
        self.robot = follower_robot
        self.config = config
        if config.resume:
            info = json.loads((root / "meta/info.json").read_text())
            self._check_sidecars(root, total_episodes=info["total_episodes"])
            self._validate_schema(info["features"], info["fps"])
        self.sample_reader = sample_reader
        self._rollout: dict | None = None
        self._sidecar: Path | None = None
        self._origin = 0.0
        self._segment = -1
        self._previous_phase: RecapPhase | None = None
        self._sealed_result: tuple[RecapOutcome, str] | None = None
        self._sealed_at: tuple[str, float] | None = None
        self._closed = False
        self._save_failed = False
        self._finish_lock = threading.Lock()
        super().__init__(
            follower_robot, lambda: {}, config, observation_lock=observation_lock, dataset=dataset
        )
        self._validate_schema(self.features, self.dataset.meta.fps)

    def _validate_schema(self, features: dict, fps: int) -> None:
        expected = {**self._dataset_features(), **DEFAULT_FEATURES}
        if set(features) != set(expected):
            raise ValueError("RECAP dataset feature keys do not match the robot/recording schema")
        for key, value in expected.items():
            actual = features.get(key, {})
            if (
                actual.get("dtype") != value["dtype"]
                or tuple(actual.get("shape", ())) != tuple(value["shape"])
                or actual.get("names") != value.get("names")
            ):
                raise ValueError(f"Incompatible RECAP feature {key}: {actual!r}")
        if fps != self.config.fps:
            raise ValueError("RECAP resume fps does not match the dataset")

    @staticmethod
    def _check_sidecars(root: Path, total_episodes: int | None = None) -> None:
        records = [json.loads(path.read_text()) for path in (root / "meta/recap").glob("*.json")]
        if not records and total_episodes != 0:
            raise ValueError("Not a RECAP dataset: missing meta/recap rollout metadata")
        if any(record.get("schema_version") != RECAP_METADATA_SCHEMA_VERSION for record in records):
            raise ValueError("Unsupported RECAP metadata schema")
        if any(record.get("status") != "committed" for record in records):
            raise ValueError("Unfinished RECAP rollout/save found; inspect meta/recap before resuming")
        for record in records:
            validate_policy_identity(record.get("model", {}))
            RecapOutcome(record.get("outcome"))
        indices = [r["episode_index"] for r in records if r["status"] == "committed"]
        if total_episodes is not None and sorted(indices) != list(range(total_episodes)):
            raise ValueError("RECAP sidecars do not match dataset episodes; refusing to append")

    def _dataset_features(self) -> dict[str, dict]:
        features = super()._dataset_features()
        for key, dtype in {
            "phase": "int64",
            "segment_id": "int64",
            "generation": "int64",
            "policy_step": "int64",
            "source_observation_id": "int64",
            "action_valid": "bool",
            "elapsed_s": "float64",
            "wall_time_s": "float64",
            "observation_duration_s": "float64",
            "command_age_s": "float64",
        }.items():
            features[f"recap.{key}"] = {"dtype": dtype, "shape": (1,), "names": None}
        return features

    @property
    def has_rollout(self) -> bool:
        return self._rollout is not None

    def begin_rollout(self, task: str, policy_identity: dict, client_config: dict) -> None:
        validate_policy_identity(policy_identity)
        with self._sample_lock:
            if self._closed or self.has_rollout or self.dataset.has_pending_frames() or self._save_failed:
                raise RuntimeError("Cannot replace an unfinished RECAP rollout")
            rollout_id = str(uuid.uuid4())
            self._origin = time.monotonic()
            self._task = task
            self._frames_in_window = 0
            self._last_error = None
            self._segment = -1
            self._previous_phase = None
            self._sealed_result = None
            self._sealed_at = None
            self._sidecar = Path(self.config.root) / "meta/recap" / f"{rollout_id}.json"
            self._rollout = {
                "schema_version": RECAP_METADATA_SCHEMA_VERSION,
                "rollout_id": rollout_id,
                "episode_index": self.dataset.meta.total_episodes,
                "status": "recording",
                "task": task,
                "started_at_utc": datetime.now(UTC).isoformat(),
                "model": dict(policy_identity),
                "client_config": client_config,
                "sampling": "latest dispatched command at observation completion; nominal fps timestamps",
                "phase_codes": {phase.name.lower(): int(phase) for phase in RecapPhase},
            }
            atomic_write_json(self._sidecar, self._rollout)

    def resume_capture(self) -> None:
        with self._sample_lock:
            if not self.has_rollout or self._last_error is not None or self._closed or self._sealed_result:
                raise RuntimeError("No healthy RECAP rollout to sample")
            self._previous_phase = None  # A new segment after any physical handover.
            self._active.set()

    def request_pause(self) -> None:
        """Nonblocking fence used before physical shutdown/slow-save waits."""
        self._active.clear()

    def _record_one(self) -> None:
        with self._sample_lock:
            try:
                self._capture_frame()
            except Exception as exc:
                # Publish failure before releasing the sample barrier so that a
                # simultaneous result/save cannot mark an incomplete capture good.
                with self._state_lock:
                    self._last_error = exc
                self.request_pause()
                raise

    def _capture_frame(self) -> None:
        # Caller owns _sample_lock throughout capture and failure publication.
        if self._closed or self._sealed_result or not self.has_rollout:
            return
        if not self._active.is_set():
            return
        started = time.monotonic()
        with self._observation_lock:
            if not self._active.is_set():
                return
            observation = self.robot.get_observation()
            sample = self.sample_reader()
        captured = time.monotonic()
        if not self._active.is_set():
            return
        if sample.phase != self._previous_phase:
            self._segment += 1
            self._previous_phase = sample.phase
        valid = sample.action is not None and sample.phase is not RecapPhase.PAUSED
        # A measured hold is a placeholder, explicitly excluded by action_valid.
        action = sample.action if valid else {key: observation[key] for key in self.robot.action_features}
        values = {
            "phase": int(sample.phase),
            "segment_id": self._segment,
            "generation": sample.generation,
            "policy_step": sample.policy_step,
            "source_observation_id": sample.source_observation_id,
            "action_valid": valid,
            "elapsed_s": captured - self._origin,
            "wall_time_s": time.time(),
            "observation_duration_s": captured - started,
            "command_age_s": -1.0
            if sample.command_monotonic is None
            else max(0.0, captured - sample.command_monotonic),
        }
        self.dataset.add_frame(
            {
                **build_dataset_frame(self.features, observation, prefix=OBS_STR),
                **build_dataset_frame(self.features, action, prefix=ACTION),
                "task": self._task,
                "intervention": np.array([sample.phase is RecapPhase.INTERVENTION], dtype=bool),
                **{
                    f"recap.{key}": np.array([value], dtype=self.features[f"recap.{key}"]["dtype"])
                    for key, value in values.items()
                },
            }
        )
        with self._state_lock:
            self._frames_in_window += 1

    def _run(self) -> None:
        period = 1.0 / self.config.fps
        while not self._shutdown.is_set():
            if not self._active.wait(timeout=0.05):
                continue
            started = time.monotonic()
            try:
                self._record_one()
            except Exception:
                logger.exception("RECAP recording failed; unsaved rollout will be discarded")
            # No catch-up bursts after camera stalls. Real times expose any gaps.
            self._shutdown.wait(max(0.0, period - (time.monotonic() - started)))

    def seal_rollout(self, outcome: RecapOutcome | str, *, reason: str = "operator") -> None:
        """Freeze the result before releasing the client's handover lock.

        Shutdown must finish saving a result the operator already submitted.
        """
        outcome = RecapOutcome(outcome)
        self.pause_capture()
        with self._sample_lock:
            if self.has_rollout and self._sealed_result is None:
                if self.last_error is not None:
                    raise RuntimeError("Cannot save an incomplete RECAP capture; reset or stop to discard it")
                self._sealed_result = (outcome, reason)
                self._sealed_at = (datetime.now(UTC).isoformat(), time.monotonic() - self._origin)

    def finish_rollout(self, outcome: RecapOutcome | str, *, reason: str = "operator") -> int:
        self.seal_rollout(outcome, reason=reason)
        return self._finish_sealed_rollout()

    def _finish_sealed_rollout(self) -> int:
        with self._finish_lock, self._sample_lock:
            if not self.has_rollout:
                return 0
            if self._save_failed:
                raise RuntimeError("RECAP save already failed; retained its journal for manual recovery")
            frames = self._frames_in_window
            if not frames:
                self._discard_rollout_locked()
                return 0
            if self._sealed_result is None:
                raise RuntimeError("RECAP requires an explicit success/failure result before saving")
            outcome, reason = self._sealed_result
            record = dict(self._rollout)
            record.update(
                status="saving",
                outcome=str(outcome),
                success=outcome is RecapOutcome.SUCCESS,
                termination_reason=reason,
                ended_at_utc=self._sealed_at[0],
                duration_s=self._sealed_at[1],
                frame_count=frames,
            )
            try:
                atomic_write_json(self._sidecar, record)
                self.dataset.save_episode(parallel_encoding=False)
                record["status"] = "committed"
                atomic_write_json(self._sidecar, record)
            except Exception:
                self._save_failed = True
                # Do not retry a possibly committed episode or erase its data.
                raise
            self._clear_rollout_state()
            self._saved_episodes += 1
            return frames

    def _clear_rollout_state(self) -> None:
        self._rollout = None
        self._sidecar = None
        self._frames_in_window = 0
        self._sealed_result = None
        self._sealed_at = None

    def _discard_rollout_locked(self) -> int:
        frames = self._frames_in_window
        # The first add_frame can fail after creating images/list entries but
        # before advancing its size, so has_pending_frames alone is insufficient.
        if self.has_rollout or self.dataset.has_pending_frames():
            self.dataset.clear_episode_buffer()
        if self._sidecar is not None:
            self._sidecar.unlink(missing_ok=True)
        self._clear_rollout_state()
        return frames

    def discard_rollout(self) -> int:
        """Drop an unconfirmed attempt, including its temporary images and journal."""
        self.pause_capture()
        with self._finish_lock, self._sample_lock:
            if self._save_failed:
                raise RuntimeError("RECAP save failed; inspect the retained journal before discarding data")
            if self._sealed_result is not None:
                raise RuntimeError("Cannot discard a RECAP result already submitted for saving")
            return self._discard_rollout_locked()

    def close(self) -> None:
        if self._closed:
            return
        self.request_pause()
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        try:
            if not self._save_failed:
                if self._sealed_result is not None:
                    self._finish_sealed_rollout()
                else:
                    self.discard_rollout()
        finally:
            self._closed = True
            self.dataset.finalize()
        if self.config.push_to_hub and not self._save_failed and self.last_error is None:
            self.dataset.push_to_hub(private=self.config.private)
