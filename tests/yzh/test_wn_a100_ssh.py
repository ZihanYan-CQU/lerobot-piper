"""No real SSH connections, terminals, credentials, or desktop windows."""

import json
import queue
import shlex
import subprocess
import sys
import threading
from dataclasses import asdict, replace

import pytest

from lerobot.yzh.wn_a100 import remote


@pytest.fixture
def config():
    return remote.load_config()


class FakeProbe:
    def __init__(self, returncode=0, error="", timeout=False):
        self.returncode = returncode
        self.error = error
        self.timeout = timeout

    def communicate(self, timeout=None):
        if self.timeout:
            raise subprocess.TimeoutExpired("ssh", timeout)
        return None, self.error


def test_missing_master_is_offline_without_starting_ssh(config, monkeypatch):
    monkeypatch.setattr(remote.Path, "is_socket", lambda _self: False)
    monkeypatch.setattr(
        remote.subprocess, "Popen", lambda *a, **k: pytest.fail("no SSH")
    )
    event = remote.check_session(config)
    assert not event.connected and "未找到" in event.error


@pytest.mark.parametrize("returncode", [0, 255])
def test_probe_checks_authenticated_channel_without_new_auth_or_port_forwarding(
    config, monkeypatch, returncode
):
    monkeypatch.setattr(remote.Path, "is_socket", lambda _self: True)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/incompatible/lib")
    captured, ended = [], []
    probe = FakeProbe(returncode, "Connection closed")

    def create(command, **options):
        captured.append((command, options))
        return probe

    monkeypatch.setattr(remote.subprocess, "Popen", create)
    monkeypatch.setattr(remote, "_end_client", lambda child: ended.append(child))
    event = remote.check_session(config)
    assert event.connected is (returncode == 0)
    command, options = captured[0]
    assert command[-1] == "true"
    assert command[command.index("-S") + 1] == config.control_path
    assert "BatchMode=yes" in command and "ProxyCommand=false" in command
    assert not any(flag in command for flag in ("-J", "-M", "-L", "-R", "-D"))
    assert options["stdin"] == subprocess.DEVNULL
    assert options["stdout"] == subprocess.DEVNULL
    assert options["start_new_session"]
    assert "LD_LIBRARY_PATH" not in options["env"]
    assert ended == [probe]


@pytest.mark.parametrize("cancel", [False, True])
def test_probe_timeout_and_cancel_reap_only_the_probe(config, monkeypatch, cancel):
    monkeypatch.setattr(remote.Path, "is_socket", lambda _self: True)
    probe, ended = FakeProbe(timeout=True), []
    monkeypatch.setattr(remote.subprocess, "Popen", lambda *a, **k: probe)
    monkeypatch.setattr(remote, "_end_client", lambda child: ended.append(child))
    clock = iter((0, 1, 6))
    monkeypatch.setattr(remote.time, "monotonic", lambda: next(clock))
    stop = threading.Event()
    if cancel:
        stop.set()
    event = remote.check_session(config, stop)
    assert not event.connected
    assert ("取消" if cancel else "超时") in event.error
    assert ended == [probe]


def test_probe_os_error_is_not_reported_as_authenticated(config, monkeypatch):
    monkeypatch.setattr(remote.Path, "is_socket", lambda _self: True)

    def fail(*args, **kwargs):
        raise FileNotFoundError("ssh missing")

    monkeypatch.setattr(remote.subprocess, "Popen", fail)
    event = remote.check_session(config)
    assert not event.connected and "ssh missing" in event.error


def test_login_script_is_quoted_valid_bash_without_embedded_credentials(config):
    config = replace(config, control_path="/tmp/连接 ' ; $(touch MUST_NOT_RUN).sock")
    script = remote.login_script(config)
    assert remote.login_command(config) in script.splitlines()
    assert script.index("trap '' HUP") < script.index(remote.login_command(config))
    # Authentication stays interactive; only the later reuse-only probe is
    # BatchMode. A zero SSH login exit code alone is not a verified session.
    assert "BatchMode=yes" not in remote.login_command(config)
    assert "BatchMode=yes" in script and "ProxyCommand=false" in script
    assert "sshpass" not in script and "PRIVATE" not in script
    result = subprocess.run(
        ["bash", "-n"], input=script, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr  # Syntax check, never execute.


def test_login_subprocess_inherits_ignored_hup_but_can_still_be_cancelled(
    config, monkeypatch
):
    check_signals = (
        "import signal; "
        "assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN; "
        "assert signal.getsignal(signal.SIGINT) != signal.SIG_IGN; "
        "print('TEST_AUTH_OK')"
    )
    monkeypatch.setattr(
        remote,
        "login_command",
        lambda _: shlex.join([sys.executable, "-c", check_signals]),
    )
    monkeypatch.setattr(
        remote,
        "ssh_command",
        lambda *args: [sys.executable, "-c", "print('TEST_SHARED_SESSION_OK')"],
    )
    result = subprocess.run(
        ["bash", "-c", remote.login_script(config)],
        input="\n",
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "TEST_AUTH_OK\n" in result.stdout
    assert "TEST_SHARED_SESSION_OK\n" in result.stdout
    assert "共享连接验证成功" in result.stdout
    assert "\\n" not in result.stdout  # Human-readable terminal lines.


@pytest.mark.parametrize("auth_succeeds", [False, True])
def test_failed_auth_or_unusable_master_keeps_the_error_visible_and_fails(
    config, monkeypatch, auth_succeeds
):
    auth = (
        "print('TEST_AUTH_OK')"
        if auth_succeeds
        else "import sys; print('TEST_AUTH_FAILED',file=sys.stderr); sys.exit(255)"
    )
    probe = "import sys; print('TEST_MASTER_UNAVAILABLE',file=sys.stderr); sys.exit(1)"
    monkeypatch.setattr(
        remote, "login_command", lambda _: shlex.join([sys.executable, "-c", auth])
    )
    monkeypatch.setattr(
        remote, "ssh_command", lambda *args: [sys.executable, "-c", probe]
    )
    # No wall-clock backoff needed for this isolated shell regression.
    script = "sleep() { :; }\n" + remote.login_script(config)
    result = subprocess.run(
        ["bash", "-c", script], input="\n", text=True, capture_output=True, timeout=10
    )
    assert result.returncode != 0
    assert "共享连接验证成功" not in result.stdout
    assert "登录未完成" in result.stdout
    if auth_succeeds:
        assert result.stderr.count("TEST_MASTER_UNAVAILABLE") == 3
        assert "这不一定是密码错误" in result.stdout
    else:
        assert "TEST_AUTH_FAILED" in result.stderr
        assert "TEST_MASTER_UNAVAILABLE" not in result.stderr


@pytest.mark.parametrize(
    "terminal",
    ["gnome-terminal", "konsole", "xfce4-terminal", "xterm", "x-terminal-emulator"],
)
def test_terminal_launch_arguments_preserve_script_as_one_argument(
    config, monkeypatch, terminal
):
    monkeypatch.setattr(
        remote.shutil,
        "which",
        lambda name: "/usr/bin/" + name if name in ("bash", terminal) else None,
    )
    command = remote.terminal_command(config)
    assert command[0] == "/usr/bin/" + terminal
    if terminal == "xfce4-terminal":
        child = shlex.split(command[-1].split("=", 1)[1])
        assert child[-1] == remote.login_script(config)
    else:
        assert command[-1] == remote.login_script(config)
    if terminal == "gnome-terminal":
        assert "--wait" in command


@pytest.mark.parametrize("bash_exists", [False, True])
def test_no_terminal_has_clear_error_and_never_starts_shell(
    config, monkeypatch, bash_exists
):
    monkeypatch.setattr(
        remote.shutil,
        "which",
        lambda name: "/bin/bash" if bash_exists and name == "bash" else None,
    )
    with pytest.raises(RuntimeError, match="终端"):
        remote.terminal_command(config)


def test_opening_terminal_does_not_capture_passwords_or_replace_user_environment(
    config, monkeypatch
):
    calls = []
    monkeypatch.setenv("LD_LIBRARY_PATH", "/wrong/library")
    monkeypatch.setenv("WN_KEEP_ENV", "present")
    monkeypatch.setattr(
        remote, "terminal_command", lambda _config: ["gnome-terminal", "--wait"]
    )
    monkeypatch.setattr(
        remote.subprocess,
        "Popen",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "child",
    )
    assert remote.open_login_terminal(config) == "child"
    args, options = calls[0]
    assert args == (["gnome-terminal", "--wait"],)
    assert all(
        options[name] == subprocess.DEVNULL for name in ("stdin", "stdout", "stderr")
    )
    assert options.get("shell", False) is False
    assert options["start_new_session"]
    assert "LD_LIBRARY_PATH" not in options["env"]
    assert options["env"]["WN_KEEP_ENV"] == "present"


def test_session_poller_bounded_queue_and_stop_drops_inflight_result(
    config, monkeypatch
):
    events = queue.Queue(maxsize=1)
    poller = remote.SessionPoller(events, config)
    count = 0

    def check(_config, _stop):
        nonlocal count
        count += 1
        if count == 3:
            poller.stop()
        return remote.SessionEvent(count != 1, str(count))

    monkeypatch.setattr(remote, "check_session", check)
    monkeypatch.setattr(poller.refresh_event, "wait", lambda _delay: None)
    poller.run()
    assert count == 3
    assert events.get_nowait() == remote.SessionEvent(True, "2")
    assert events.empty()
    assert poller.stop_event.is_set() and poller.refresh_event.is_set()


def test_loaded_configuration_keeps_new_log_path(config):
    assert config.logs[1].path == "/home/yzh/具身智能/训练日志/Lerobot/training.log"
    assert "password" not in json.dumps(asdict(config))
