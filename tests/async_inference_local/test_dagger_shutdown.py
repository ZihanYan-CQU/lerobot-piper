import threading

from test_dagger_runtime import correct, setup

from lerobot.async_inference_local.runtime.dagger import DaggerPhase
from lerobot.async_inference_local.runtime.deployment import State


def test_stop_closes_hardware_before_watchdog_discard_finishes(monkeypatch):
    runtime, robot, _, dataset = setup(save_partial_on_stop=True)
    entered, release = threading.Event(), threading.Event()
    clear = dataset.clear_episode_buffer
    errors = []

    def blocked_clear():
        entered.set()
        assert release.wait(5)
        clear()

    def stop():
        try:
            runtime.stop()
        except Exception as exc:
            errors.append(exc)

    correct(runtime)
    monkeypatch.setattr(dataset, "clear_episode_buffer", blocked_clear)
    runtime.io.controller.health_error = RuntimeError("CAN bridge failed")
    worker = threading.Thread(target=stop, daemon=True)
    try:
        assert entered.wait(1)
        assert runtime.phase == DaggerPhase.FAULT
        assert not runtime.recorder.active and not runtime.io.controller.enabled
        worker.start()
        assert robot.closed.wait(0.5), "Dataset discard must not postpone hardware shutdown"
        assert "master:stop" in robot.events and "follower:stop" in robot.events
        assert runtime.state == State.STOPPED
        assert worker.is_alive(), "Finalization should still wait for the pending discard"
        assert not dataset.finalized
    finally:
        release.set()
        if worker.ident is not None:
            worker.join(3)
        runtime.stop()
    assert not worker.is_alive() and not errors
    assert dataset.finalized and not dataset.saved and not dataset.frames
    assert runtime.io.controller.feedback_calls == 0


def test_stop_waits_for_active_watchdog_hardware_access(monkeypatch):
    runtime, robot, _, dataset = setup()
    entered, release = threading.Event(), threading.Event()
    errors = []

    def blocked_health_check():
        entered.set()
        assert release.wait(5)
        assert not robot.closed.is_set()

    def stop():
        try:
            runtime.stop()
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=stop, daemon=True)
    try:
        correct(runtime)
        monkeypatch.setattr(runtime.io.controller, "assert_correction_healthy", blocked_health_check)
        assert entered.wait(1)
        worker.start()
        assert runtime.stop_event.wait(1)
        assert not robot.closed.wait(0.1), "Never disconnect during an active hardware check"
    finally:
        release.set()
        if worker.ident is not None:
            worker.join(3)
        runtime.stop()
    assert robot.closed.is_set() and not worker.is_alive() and not errors
    assert dataset.finalized
