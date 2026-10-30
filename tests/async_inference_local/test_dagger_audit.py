import threading
from types import SimpleNamespace

import draccus
import pytest
from test_dagger_recording import make_recorder
from test_dagger_runtime import Robot, correct, setup
from test_piper_dagger import _FakeBridge, _FakeBus, _pose
from test_runtime import wait_for

from lerobot.async_inference_local.control.dagger_io import DaggerRobotIO
from lerobot.async_inference_local.control.piper_dagger import (
    PiperControlMode,
    PiperDaggerController,
    PiperDaggerHardwareError,
)
from lerobot.async_inference_local.runtime.dagger import DaggerPhase
from lerobot.async_inference_local.runtime.deployment import State
from lerobot.async_inference_local.scripts.dagger import LocalDaggerConfig


@pytest.mark.parametrize("policy_fps", [15, 20, 50])
@pytest.mark.parametrize(
    "dataset_args",
    [
        ["--dataset.repo_id=user/corrections"],
        ["--dataset.repo_id=user/corrections", "--dataset.use_videos=false"],
    ],
)
def test_cli_dataset_options_preserve_default_fps_inheritance(policy_fps, dataset_args):
    cfg = draccus.parse(
        LocalDaggerConfig,
        args=[
            "--robot.type=piper_follower",
            "--robot.cameras={}",
            "--pretrained_name_or_path=unused",
            "--task=pick cube",
            f"--fps={policy_fps}",
            *dataset_args,
        ],
    )
    assert cfg.dataset.fps == policy_fps


def test_cli_still_allows_explicit_independent_dataset_fps():
    cfg = draccus.parse(
        LocalDaggerConfig,
        args=[
            "--robot.type=piper_follower",
            "--robot.cameras={}",
            "--pretrained_name_or_path=unused",
            "--task=pick cube",
            "--fps=20",
            "--dataset.repo_id=user/corrections",
            "--dataset.fps=15",
        ],
    )
    assert cfg.dataset.fps == 15


@pytest.mark.parametrize("recover", [False, True])
def test_bridge_fault_keeps_real_controller_in_leader_mode_until_explicit_reset(recover):
    runtime, robot, _, dataset = setup(start=False, save_partial_on_stop=True)
    events = []

    class TrackingBus(_FakeBus):
        def write_js(self, values):
            super().write_js(values)
            self.pose = dict(zip(self.pose, values, strict=True))

    follower = TrackingBus("follower", events)
    master = TrackingBus("master", events)
    bridge = _FakeBridge(events, master_bus=master, follower_bus=follower)
    failed = False

    def health():
        if failed:
            raise PiperDaggerHardwareError("physical CAN forwarding failed")

    bridge.raise_if_unhealthy = health
    robot.bus = follower
    robot.config = SimpleNamespace(teleop_joint_alpha=1.0, teleop_gripper_alpha=1.0)
    runtime.io = DaggerRobotIO(
        robot,
        runtime.config.hardware,
        controller_factory=lambda r, c: PiperDaggerController(r, c, master_bus=master, bridge=bridge),
    )
    runtime.start()
    try:
        correct(runtime)
        # A lost forwarding link allows the human-held master to diverge from
        # the follower. A watchdog must not command it back without operator input.
        master.pose = _pose(0.4)
        master.pose["gripper.pos"] = 0.03
        pose_before = dict(master.pose)
        writes_before = (len(master.writes), len(follower.writes))
        events.clear()
        failed = True
        wait_for(lambda: runtime.phase == DaggerPhase.FAULT)
        with runtime._transition_lock:
            assert runtime.io.controller.mode == PiperControlMode.CORRECTING
            assert master.pose == pose_before
            assert (len(master.writes), len(follower.writes)) == writes_before
            assert "master:follower" not in events
            assert not runtime.recorder.active
            assert not dataset.frames
            assert not dataset.saved
        with pytest.raises(RuntimeError):
            runtime.resume()
        if not recover:
            runtime.stop()
            assert (len(master.writes), len(follower.writes)) == writes_before
            assert "master:follower" not in events
            assert "master:stop" in events and "follower:stop" in events
            assert not dataset.saved and dataset.finalized
            return
        failed = False
        runtime.reset()
        assert runtime.state == State.READY
        assert runtime.io.controller.mode == PiperControlMode.PAUSED
        assert not dataset.frames
    finally:
        runtime.stop()


def test_capture_error_cannot_reappear_after_discard_and_new_window():
    robot = Robot()
    reader_failed = threading.Event()
    continue_error = threading.Event()
    recorder, dataset = make_recorder(robot, lambda: _pose(0.5))
    inner_lock = recorder._sample_lock

    class PauseAfterFailedSample:
        def __enter__(self):
            return inner_lock.__enter__()

        def __exit__(self, error_type, error, traceback):
            inner_lock.__exit__(error_type, error, traceback)
            # Simulate a preemption exactly when a failed sample releases its
            # capture barrier, before the recording loop handles the exception.
            if error_type is not None and threading.current_thread().name == "PiperDaggerRecorder":
                reader_failed.set()
                assert continue_error.wait(3)

    failed_once = False

    def observation():
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise OSError("old camera failure")
        return robot.get_observation()

    recorder._sample_lock = PauseAfterFailedSample()
    recorder._observation_reader = observation
    recorder.begin("old window")
    recorder.start()
    try:
        assert reader_failed.wait(1)
        recorder.pause_capture()
        recorder.finish_window(save=False)
        recorder.acknowledge_error()
        recorder.begin("new window")
        continue_error.set()
        wait_for(lambda: recorder.last_error is not None or recorder.frames_in_window > 0)
        assert recorder.last_error is None
        assert recorder.active
        assert dataset.frames and all(frame["task"] == "new window" for frame in dataset.frames)
    finally:
        continue_error.set()
        recorder.close()
