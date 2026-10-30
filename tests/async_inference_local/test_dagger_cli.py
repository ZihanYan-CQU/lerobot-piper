import threading

import pytest
from test_dagger_runtime import DaggerPhase, State, correct, setup
from test_runtime import wait_for

from lerobot.async_inference_local.scripts import dagger


def start_cli(monkeypatch, **overrides):
    runtime, robot, engine, dataset = setup(start=False, **overrides)
    commands, errors = [], []
    monkeypatch.setattr(dagger, "make_robot_from_config", lambda _: robot)
    monkeypatch.setattr(dagger, "LocalDagger", lambda *a: runtime)
    monkeypatch.setattr(dagger, "_read_commands", commands.append)

    def run():
        try:
            dagger.run(runtime.dagger_config)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    wait_for(lambda: bool(commands))
    return runtime, robot, engine, dataset, commands[0], worker, errors


def test_prompt_collects_one_window_and_exits_at_target(monkeypatch):
    runtime, robot, _, dataset, commands, worker, errors = start_cli(monkeypatch, num_episodes=1)
    try:
        commands.put("/start updated task")
        wait_for(lambda: runtime.state == State.RUNNING and not runtime._operator_pending)
        commands.put("/pause")
        wait_for(lambda: runtime.state == State.PAUSED and not runtime._operator_pending)
        commands.put("/correct")
        wait_for(lambda: runtime.recorder.frames_in_window > 0 and not runtime._operator_pending)
        commands.put("/correct")
        worker.join(2)
        assert not worker.is_alive() and not errors
        assert robot.closed.is_set() and dataset.finalized
        assert len(dataset.saved) == 1
        assert dataset.saved[0][0]["task"] == "updated task"
    finally:
        runtime.stop()
        worker.join(1)


def test_prompt_status_and_stop_remain_available_during_save(monkeypatch, capsys):
    runtime, robot, _, dataset, commands, worker, errors = start_cli(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = dataset.save_episode

    def save(**kw):
        entered.set()
        assert release.wait(4)
        original(**kw)

    monkeypatch.setattr(dataset, "save_episode", save)
    try:
        correct(runtime)
        commands.put("/correct")
        assert entered.wait(1)
        commands.put("/status")
        seen = []

        def saw_status():
            seen.append(capsys.readouterr().out)
            return '"saving": true' in "".join(seen)

        wait_for(saw_status, timeout=0.5)
        commands.put("/stop")
        assert robot.closed.wait(0.5)
        assert runtime.state == State.STOPPED
    finally:
        release.set()
        worker.join(3)
        runtime.stop()
    assert not worker.is_alive() and not errors
    assert len(dataset.saved) == 1 and dataset.finalized


def test_keyboard_events_map_to_generation_fenced_operator_requests(monkeypatch):
    runtime, _, _, _ = setup()
    captured = []
    listener = object()
    monkeypatch.setattr(dagger, "create_key_listener", lambda cb, **kw: captured.append(cb) or listener)
    try:
        cfg = runtime.dagger_config
        cfg.input_device = "keyboard"
        assert dagger._start_inputs(runtime, cfg) == [listener]
        captured[0](cfg.advance_key)
        wait_for(lambda: runtime.state == State.RUNNING and not runtime._operator_pending)
        captured[0](cfg.advance_key)
        wait_for(lambda: runtime.state == State.PAUSED and not runtime._operator_pending)
        captured[0](cfg.advance_key)
        wait_for(lambda: runtime.phase == DaggerPhase.CORRECTING and not runtime._operator_pending)
        captured[0](cfg.stop_key)
        assert runtime.stop_event.is_set()
    finally:
        runtime.stop()


def test_absent_keyboard_requires_an_operable_input_mode(monkeypatch):
    runtime, _, _, _ = setup()
    monkeypatch.setattr(dagger, "create_key_listener", lambda *a, **kw: None)
    try:
        runtime.dagger_config.input_device = "keyboard"
        with pytest.raises(RuntimeError, match="input_device=prompt"):
            dagger._start_inputs(runtime, runtime.dagger_config)
    finally:
        runtime.stop()


def test_pedal_uses_advance_reset_and_separate_keyboard_stop(monkeypatch):
    runtime, _, _, _ = setup()
    callbacks = {}

    def keyboard(cb, **kw):
        callbacks["keyboard"] = cb
        return object()

    def pedal(cb, **kw):
        callbacks["pedal"] = cb
        return object()

    monkeypatch.setattr(dagger, "create_key_listener", keyboard)
    monkeypatch.setattr(dagger, "start_pedal_listener", pedal)
    try:
        cfg = runtime.dagger_config
        cfg.input_device = "pedal"
        assert len(dagger._start_inputs(runtime, cfg)) == 2
        callbacks["pedal"](cfg.pedal_advance)
        wait_for(lambda: runtime.state == State.RUNNING and not runtime._operator_pending)
        callbacks["pedal"](cfg.pedal_advance)
        wait_for(lambda: runtime.state == State.PAUSED and not runtime._operator_pending)
        callbacks["pedal"](cfg.pedal_reset)
        wait_for(lambda: runtime.state == State.READY and not runtime._operator_pending)
        callbacks["keyboard"](cfg.stop_key)
        assert runtime.stop_event.is_set()
    finally:
        runtime.stop()


def test_video_save_failure_after_stop_is_reported_and_not_retried(monkeypatch):
    runtime, robot, _, dataset, commands, worker, errors = start_cli(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    attempts = []

    def fail_save(**kw):
        attempts.append(kw)
        entered.set()
        assert release.wait(4)
        raise OSError("encoder failed during exit")

    monkeypatch.setattr(dataset, "save_episode", fail_save)
    try:
        correct(runtime)
        commands.put("/correct")
        assert entered.wait(1)
        commands.put("/stop")
        assert robot.closed.wait(0.5)
    finally:
        release.set()
        worker.join(3)
        runtime.stop()
    assert not worker.is_alive()
    assert len(errors) == 1
    assert runtime.failure is not None
    assert not dataset.saved and dataset.finalized
    assert len(attempts) == 1
