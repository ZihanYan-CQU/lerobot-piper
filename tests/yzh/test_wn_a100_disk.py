"""Disk quota parsing and bounded SSH statistics without accessing a server."""

from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import threading
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from lerobot.yzh.wn_a100 import remote

from lerobot.yzh.wn_a100 import monitor as core


@pytest.fixture
def config():
    return remote.DiskConfig(remote.load_config())


def payload(size="2.3T", home="/home/yzh"):
    return {"user": "yzh", "home": home, "output": f"{size}\t{home}\n"}


@pytest.mark.parametrize(
    "size, expected",
    [
        ("0", 0),
        ("512", 512),
        ("4.0K", 4096),
        ("500M", 500 * 1024**2),
        ("999G", 999 * 1024**3),
        ("2.3T", int(Decimal("2.3") * 1024**4)),
        ("3.0T", remote.QUOTA_BYTES),
        ("3.1T", int(Decimal("3.1") * 1024**4)),
    ],
)
def test_du_human_units_and_original_text(size, expected):
    sample = remote.parse_disk_sample(payload(size), now=100)
    assert sample.size_text == size
    assert sample.used_bytes == expected
    assert sample.over_quota is (expected > remote.QUOTA_BYTES)
    assert sample.checked_at == 100


@pytest.mark.parametrize(
    "size, over",
    [
        ("3072G", False),
        ("3073G", True),
        ("3T", False),
        ("3.0T", False),
        ("3.1T", True),
        ("4T", True),
    ],
)
def test_exceeding_not_equal_to_three_tib_is_an_alert(size, over):
    sample = remote.parse_disk_sample(payload(size))
    rows, level = remote.disk_rows(sample, 2000)
    assert sample.over_quota is over
    assert rows == (f"{size}/3T", f"{sample.used_bytes / remote.QUOTA_BYTES:.1%}")
    assert (level == "over") is over
    assert ("超额" in remote.disk_text(sample, None)) is over


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {},
        {"error": "permission denied"},
        {**payload(), "user": "root"},
        {**payload(), "home": "/"},
        {**payload(), "output": "2.3T\t/someone_else\n"},
        {**payload(), "output": "2.3T\t/home/yzh\n1G\t/other\n"},
        payload("-1T"),
        payload("nanT"),
        payload("2,3T"),
        payload("2.3TB"),
        {**payload(), "output": "x" * 9000},
    ],
)
def test_invalid_partial_or_wrong_user_data_is_not_a_normal_measurement(bad):
    with pytest.raises(remote.DiskError):
        remote.parse_disk_sample(bad)


def test_unicode_and_spaces_in_remote_home_are_paired_with_the_reported_path():
    sample = remote.parse_disk_sample(payload("2.3T", "/home/yzh/中文 路径"))
    assert sample.home == "/home/yzh/中文 路径"


def test_scan_and_errors_keep_last_value_and_warning_until_successful_recovery():
    old = remote.parse_disk_sample(payload("3.1T"), now=100)
    scanning = remote.apply_disk_event(old, remote.DiskEvent(scanning=True))
    assert scanning.scanning and scanning.over_quota
    failed = remote.apply_disk_event(scanning, remote.DiskEvent(error="SSH 断开"))
    assert failed.size_text == "3.1T" and failed.over_quota and not failed.scanning
    assert remote.disk_rows(failed, 2000) == (("3.1T/3T", "103.3%"), "over")
    assert "超额 · 上次" in remote.disk_text(failed, None)
    fresh = remote.apply_disk_event(
        failed, remote.DiskEvent(sample=remote.parse_disk_sample(payload("2.3T")))
    )
    assert not fresh.over_quota and not fresh.error and not fresh.scanning
    assert remote.disk_rows(fresh, 2000) == (
        ("2.3T/3T", "76.7%"),
        "normal",
    )


def test_unknown_and_expired_data_are_never_invented_as_zero_usage():
    unknown = remote.DiskSample()
    for sample, status in (
        (unknown, "等待统计"),
        (replace(unknown, scanning=True), "统计中"),
        (replace(unknown, error="offline"), "读取失败"),
    ):
        assert remote.disk_rows(sample, 100) == (("…/3T", "—%"), "unknown")
        assert status in remote.disk_text(sample, None)
    sample = remote.parse_disk_sample(payload(), now=100)
    assert sample.is_stale(900, now=1000)
    assert not sample.is_stale(901, now=1000)


@pytest.mark.parametrize(
    "scanning, error", [(False, ""), (True, ""), (False, "offline")]
)
def test_known_usage_stays_two_lines_even_while_refreshing_or_failed(scanning, error):
    sample = replace(
        remote.parse_disk_sample(payload()), scanning=scanning, error=error
    )
    rows, level = remote.disk_rows(sample, 2000)
    assert rows == ("2.3T/3T", "76.7%")
    assert level == ("stale" if error else "normal")
    if scanning:
        assert "刷新中" in remote.disk_text(sample, None)
    if error:
        assert "上次 · 更新失败" in remote.disk_text(sample, None)


class FakeProcess:
    pid = 123456
    returncode = 0

    def __init__(self, output):
        self.output = output

    def communicate(self, timeout=None):
        return self.output, ""


def test_ssh_reuses_master_without_forwarding_and_runs_exact_requested_du(
    config, monkeypatch
):
    calls, stopped = [], []
    process = FakeProcess(json.dumps(payload()))
    monkeypatch.setattr(
        remote.subprocess,
        "Popen",
        lambda args, **kwargs: (calls.append((args, kwargs)), process)[1],
    )
    monkeypatch.setattr(remote, "_end_client", stopped.append)
    result = remote.fetch_disk(config)
    args, kwargs = calls[0]
    assert args[:3] == ["ssh", "-4", "-T"]
    assert config.connection.control_path in args
    for option in (
        "ControlMaster=no",
        "ProxyCommand=false",
        "BatchMode=yes",
        "ClearAllForwardings=yes",
        "StrictHostKeyChecking=yes",
        "PermitLocalCommand=no",
    ):
        assert option in args
    remote_args = shlex.split(args[-1])
    assert remote_args[:2] == ["python3", "-c"]
    assert remote_args[2] == remote.REMOTE_DISK_READER
    assert "['sh', '-c', 'du -sh \"$HOME\"']" in remote_args[2]
    assert "--block-size" not in remote_args[2]
    assert "LD_LIBRARY_PATH" not in kwargs["env"]
    assert kwargs["start_new_session"]
    assert stopped == [process] and result.size_text == "2.3T"
    assert config.interval == 900 and config.remote_timeout < config.timeout


@pytest.mark.parametrize(
    "mode", ["ssh", "partial", "invalid_json", "oversized", "cancel", "timeout"]
)
def test_failed_or_cancelled_requests_clean_up_only_their_own_client(
    config, monkeypatch, mode
):
    process = FakeProcess(json.dumps(payload()))
    stop = threading.Event()
    stopped = []
    if mode == "ssh":
        process.returncode = 255
    elif mode == "partial":
        process.output = json.dumps({"error": "du 统计不完整"})
    elif mode == "invalid_json":
        process.output = "not JSON"
    elif mode == "oversized":
        process.output = "x" * 17000
    elif mode == "cancel":
        stop.set()
    else:
        config = replace(config, timeout=0)
    monkeypatch.setattr(remote.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(remote, "_end_client", stopped.append)
    with pytest.raises(remote.DiskError):
        remote.fetch_disk(config, stop)
    assert stopped == [process]


def test_wrong_user_is_rejected_before_ssh_or_du(config, monkeypatch):
    config = replace(config, connection=replace(config.connection, target="other@host"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Must not connect")

    monkeypatch.setattr(remote.subprocess, "Popen", forbidden)
    with pytest.raises(remote.DiskError, match="yzh"):
        remote.fetch_disk(config)


@pytest.mark.parametrize("error", ["partial", "timeout", "wrong_user"])
def test_remote_wrapper_rejects_partial_output_and_bounds_du_lifetime(
    monkeypatch, capsys, error
):
    process = FakeProcess("2.3T\t/fake/yzh\n")
    process.pid = 123456
    process.returncode = 1 if error == "partial" else 0
    process.poll = lambda: None if error == "timeout" else process.returncode
    calls, killed = [], []
    if error == "timeout":

        def communicate(timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("du", timeout)
            return "", ""

        process.communicate = communicate
    import pwd
    import sys
    import signal

    monkeypatch.setattr(
        pwd,
        "getpwuid",
        lambda _uid: SimpleNamespace(
            pw_name="other" if error == "wrong_user" else "yzh"
        ),
    )
    monkeypatch.setattr(os, "environ", {"HOME": "/fake/yzh"})
    monkeypatch.setattr(os, "nice", lambda value: None)
    monkeypatch.setattr(os, "killpg", lambda pid, signum: killed.append((pid, signum)))
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        subprocess, "Popen", lambda *args, **kwargs: (calls.append(args), process)[1]
    )
    monkeypatch.setattr(sys, "argv", ["remote", "1"])
    exec(remote.REMOTE_DISK_READER, {})
    row = json.loads(capsys.readouterr().out)
    assert row.get("error") and "output" not in row
    assert (
        not calls
        if error == "wrong_user"
        else calls[0][0] == ["sh", "-c", remote.DU_COMMAND]
    )
    assert bool(killed) is (error == "timeout")


def test_poller_does_not_queue_another_scan_when_refreshed_during_du(
    config, monkeypatch
):
    events = queue.Queue()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def fetch(_config, _stop):
        calls.append(True)
        entered.set()
        assert release.wait(1)
        return remote.parse_disk_sample(payload())

    monkeypatch.setattr(remote, "fetch_disk", fetch)
    poller = remote.DiskPoller(events, config)
    poller.start()
    try:
        assert entered.wait(1)
        assert events.get(timeout=1).scanning
        for _ in range(5):
            poller.refresh()
        assert not poller.refresh_event.is_set()
        release.set()
        assert events.get(timeout=1).sample.size_text == "2.3T"
        assert len(calls) == 1
    finally:
        release.set()
        poller.stop()
        poller.join(1)
    assert not poller.is_alive()


def test_demo_never_executes_du_or_ssh(config, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Demo must not access a server")

    monkeypatch.setattr(remote, "fetch_disk", forbidden)
    events = queue.Queue()
    poller = remote.DiskPoller(events, config, demo=True)
    poller.start()
    try:
        assert events.get(timeout=1).scanning
        assert events.get(timeout=1).sample.size_text == "2.3T"
    finally:
        poller.stop()
        poller.join(1)


def test_disk_once_json_reports_user_supplied_example_without_gui(capsys):
    assert core.main(["--disk-once", "--demo", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["used"] == "2.3T" and result["quota"] == "3T"
    assert not result["over_quota"]
    assert result["du_command"] == remote.DU_COMMAND


@pytest.mark.parametrize(
    "args",
    [
        ["--disk-interval", "0"],
        ["--disk-timeout", "nan"],
        ["--disk-once", "--server-id", "server1"],
    ],
)
def test_invalid_arguments_and_other_server_are_rejected(args):
    with pytest.raises(SystemExit):
        core.main(args)
