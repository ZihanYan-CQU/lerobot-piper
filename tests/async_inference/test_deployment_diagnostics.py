# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Crash-file capture and process-level CUDA failure recovery regressions."""

import logging
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from types import SimpleNamespace

import grpc
import pytest
import torch

import lerobot.async_inference.server.policy_server_deployment as server_module
from lerobot.async_inference.server.policy_server_deployment import (
    PolicyServerDeployment,
    PolicyServerDeploymentConfig,
)
from lerobot.async_inference.shared.deployment_diagnostics import is_fatal_cuda_error
from lerobot.async_inference.shared.helpers import TimedObservation
from lerobot.async_inference.shared.rtc_remote import RemoteRTCObservation
from lerobot.transport import services_pb2
from tests.async_inference.test_policy_server_deployment import _FakePipeline, _FakePolicy, _policy_setup


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX fatal signals and resource limits")
def test_sigsegv_captures_last_log_and_all_python_threads(tmp_path):
    fault_path = tmp_path / "crash.log"
    # Only this disposable child receives the signal. Disable core files, and
    # import no GPU libraries in it; the real process and hardware stay untouched.
    script = textwrap.dedent("""\
        import logging
        import os
        from pathlib import Path
        import resource
        import signal
        import sys
        import threading
        from lerobot.async_inference.shared.deployment_diagnostics import server_fault_diagnostics

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        ready = threading.Event()
        def waiting_inference_worker():
            ready.set()
            threading.Event().wait()
        threading.Thread(target=waiting_inference_worker, daemon=True).start()
        assert ready.wait(5)
        logger = logging.getLogger("crash-test")
        logger.setLevel(logging.INFO)
        with server_fault_diagnostics(logger, "test", Path(sys.argv[1])):
            logger.info("last checkpoint before native crash")
            os.kill(os.getpid(), signal.SIGSEGV)
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(fault_path)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == -signal.SIGSEGV, result.stderr
    content = fault_path.read_text()
    assert "last checkpoint before native crash" in content
    assert "Fatal Python error: Segmentation fault" in content
    assert "waiting_inference_worker" in content
    assert "Current thread" in content
    assert "thread=MainThread" in content


@pytest.mark.parametrize(
    "message,fatal",
    [
        ("CUDA error: an illegal memory access was encountered", True),
        ("AcceleratorError: CUDA error: device-side assert triggered", True),
        ("CUDA error: unspecified launch failure", True),
        ("CUDA error: misaligned address", True),
        ("CUDA error: an illegal instruction was encountered", True),
        ("Runtime failed: cudaErrorIllegalAddress", True),
        ("CUDA out of memory. Tried to allocate 2 GiB", False),
        ("CUDA error: out of memory", False),
        ("CUDA error: invalid device ordinal", False),
        ("non-CUDA illegal memory access", False),
        ("broken observation pipeline", False),
    ],
)
def test_fatal_cuda_classification(message, fatal):
    assert is_fatal_cuda_error(RuntimeError(message)) is fatal


def test_cleanup_exception_preserves_fatal_cuda_cause_even_with_a_chain_cycle():
    original = RuntimeError("CUDA error: an illegal memory access was encountered")
    cleanup = ValueError("RNG restore failed")
    cleanup.__context__ = original
    original.__cause__ = cleanup
    assert is_fatal_cuda_error(cleanup)
    original.args = ("ordinary error",)
    assert not is_fatal_cuda_error(cleanup)


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(server_module, "get_policy_class", lambda _: _FakePolicy)
    monkeypatch.setattr(
        server_module, "make_pre_post_processors", lambda *args, **kwargs: (_FakePipeline(), _FakePipeline())
    )
    instance = PolicyServerDeployment(PolicyServerDeploymentConfig(action_wait_timeout=0))
    yield instance
    instance.stop()


@pytest.mark.parametrize(
    "enabled,device,initialized,expected",
    [
        (True, "cuda:1", True, 2),
        (False, "cuda:1", True, 0),
        (True, "cpu", True, 0),
        (True, "cuda:1", False, 0),
        (True, None, True, 0),
    ],
)
def test_cuda_sync_is_opt_in_and_brackets_phase_work(
    server, monkeypatch, enabled, device, initialized, expected
):
    server.config.debug_cuda_sync = enabled
    server.device = device
    events = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda target: events.append(target))
    with server._diagnostic_phase("test_phase"):
        events.append("work")
    assert events == ([device, "work", device] if expected else ["work"])


@pytest.mark.parametrize("verbose", [False, True])
def test_disabled_phases_skip_timing_thread_metadata_and_sync(server, monkeypatch, caplog, verbose):
    def forbidden(*args, **kwargs):
        pytest.fail("disabled phase diagnostics must not collect metadata or synchronize CUDA")

    with monkeypatch.context() as patch, caplog.at_level(logging.INFO):
        patch.setattr(server_module.time, "perf_counter", forbidden)
        patch.setattr(server_module.threading, "get_native_id", forbidden)
        patch.setattr(server, "_diagnostic_cuda_sync", forbidden)
        with server._diagnostic_phase("calibration", verbose=verbose, sequence=0):
            pass
    assert "RTC PHASE" not in caplog.text


def test_diagnostic_sync_failure_latches_without_further_gpu_calls(server, monkeypatch, caplog):
    server.config.debug_cuda_sync = True
    server.device = "cuda:0"
    events = []

    def sync(device):
        events.append(device)
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    with (
        caplog.at_level(logging.INFO),
        pytest.raises(RuntimeError, match="illegal memory access"),
        server._diagnostic_phase("failing_sync"),
    ):
        pytest.fail("work must not run after a failed entry synchronization")
    server._diagnostic_cuda_sync("cleanup", "before", server.device)
    assert events == ["cuda:0"]
    assert not server.running
    assert "RTC PHASE failed phase=failing_sync" in caplog.text
    assert "RTC PHASE end phase=failing_sync" not in caplog.text
    assert "Traceback" in caplog.text


def test_reconnect_logs_distinguish_sessions_and_reused_model(server, caplog):
    server.config.debug_rtc_diagnostics = True
    with caplog.at_level(logging.INFO):
        server.SendPolicyInstructions(_policy_setup(), None)
        model = server.policy
        worker = server._inference_thread
        server.Ready(services_pb2.Empty(), None)
        server.SendPolicyInstructions(_policy_setup(), None)
    assert server.policy is model
    assert server._inference_thread is not worker
    assert not worker.is_alive()
    assert f"session=1 cache_hit=False model={id(model):#x}" in caplog.text
    assert f"session=2 cache_hit=True model={id(model):#x}" in caplog.text
    assert "RTC worker join end session=1" in caplog.text
    assert "phase=worker_initialize session=2" in caplog.text
    assert "worker_native_tid=" in caplog.text


def test_fatal_warmup_blocks_all_reconnect_entry_points(server, monkeypatch):
    def fail_warmup():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(server, "_initialize_worker", fail_warmup)
    with pytest.raises(RuntimeError, match="restart the server process"):
        server.SendPolicyInstructions(_policy_setup(), None)
    server._reset_server()
    assert server.failed
    assert "illegal memory access" in server.failure_traceback
    assert not server.running

    aborts = []

    def abort(code, details):
        aborts.append(code)
        raise RuntimeError(details)

    context = SimpleNamespace(abort=abort)
    for rpc, request in (
        (server.Ready, services_pb2.Empty()),
        (server.SendPolicyInstructions, _policy_setup()),
        (server.GetActions, services_pb2.Empty()),
    ):
        with pytest.raises(RuntimeError, match="restart the server process"):
            rpc(request, context)
    assert aborts == [grpc.StatusCode.FAILED_PRECONDITION] * 3
    assert server._inference_thread is None
    assert server.model_loads == 1
    assert server.model_cache_hits == 0


def test_ordinary_startup_failure_can_still_reconnect(server, monkeypatch):
    def fail_warmup():
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(server, "_initialize_worker", fail_warmup)
    with pytest.raises(RuntimeError, match="worker failed"):
        server.SendPolicyInstructions(_policy_setup(), None)
    assert server._fatal_cuda_error is None
    monkeypatch.setattr(server, "_initialize_worker", lambda: None)
    server.Ready(services_pb2.Empty(), None)
    server.SendPolicyInstructions(_policy_setup(), None)
    assert server.running and not server.failed


def test_fatal_model_load_cannot_be_retried_in_same_process(server, monkeypatch, caplog):
    class BrokenPolicy(_FakePolicy):
        @classmethod
        def from_pretrained(cls, checkpoint):
            raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setattr(server_module, "get_policy_class", lambda _: BrokenPolicy)
    with pytest.raises(RuntimeError, match="device-side assert"):
        server.SendPolicyInstructions(_policy_setup(), None)
    with pytest.raises(RuntimeError, match="restart the server process"):
        server.Ready(services_pb2.Empty(), None)
    assert server._inference_thread is None
    assert not server.config.debug_rtc_diagnostics and not server.config.debug_cuda_sync
    assert "RTC PHASE failed phase=policy_load" in caplog.text


def test_fatal_inference_stops_without_snapshot_retry(server, monkeypatch):
    calls = []

    def fail_inference(request):
        calls.append(request.observation_id)
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(server, "_predict_remote_chunk", fail_inference)
    server.SendPolicyInstructions(_policy_setup(), None)
    request = RemoteRTCObservation(TimedObservation(time.time(), 0, {}), 0, -1)
    server._observation_mailbox.put(request)
    assert server._worker_failed.wait(2)
    server._inference_thread.join(timeout=2)
    assert calls == [0]
    assert server._action_mailbox.empty()
    assert not server.running


def test_cuda_failure_while_ready_joins_worker_cannot_be_cleared(server, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    joining = threading.Event()
    ready_errors = []

    def fail_after_release(request):
        entered.set()
        assert release.wait(5)
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(server, "_predict_remote_chunk", fail_after_release)
    server.SendPolicyInstructions(_policy_setup(), None)
    server._observation_mailbox.put(RemoteRTCObservation(TimedObservation(time.time(), 0, {}), 0, -1))
    assert entered.wait(2)
    worker = server._inference_thread
    original_join = worker.join

    def join(timeout=None):
        joining.set()
        return original_join(timeout)

    monkeypatch.setattr(worker, "join", join)

    def reconnect():
        try:
            server.Ready(services_pb2.Empty(), None)
        except RuntimeError as exc:
            ready_errors.append(str(exc))

    reconnect_thread = threading.Thread(target=reconnect)
    reconnect_thread.start()
    try:
        assert joining.wait(2)
        release.set()
        reconnect_thread.join(timeout=2)
        assert not reconnect_thread.is_alive()
        assert len(ready_errors) == 1 and "restart the server process" in ready_errors[0]
        assert not server.running
        assert server.failed
        assert server._inference_thread is None
    finally:
        release.set()
        reconnect_thread.join(timeout=2)
