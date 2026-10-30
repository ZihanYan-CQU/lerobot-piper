"""Shared SSH connection, training logs, disk usage, and terminal login.

This module has no GUI dependency and never sees authentication credentials.
All background reads reuse the same user-established SSH ControlMaster.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# Connection configuration and training progress.

DEFAULT_CONFIG = Path(__file__).with_name("config.json")
TRAINING_COLORS = ("#5DBBFF", "#C1A0FF")
TAIL_BYTES = 32768
PROCESS_STALE_AFTER = 30
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TRAINING_BAR = re.compile(
    r"(?:^|\n)\s*Training:\s*[\d.]+%\|[^\n]*?\|\s*"
    r"(?P<step>[\d,]+)/(?P<total>[\d,]+)\s*\[(?P<timing>[^\]\n]*)\]"
)

# Only current-user processes whose stdout/stderr refer to one of our log
# inodes are considered. Other environment fields never enter the response.
REMOTE_GPU_HELPERS = r"""
def format_process_argv(argv):
    rendered = []
    for argument in argv:
        key, separator, value = argument.partition('=')
        if separator and key.startswith('--') and shlex.quote(key) == key:
            # Quote the value, not the entire --option=value token. This also
            # keeps non-ASCII paths recognizable by the one-option-per-line UI.
            rendered.append(key + '=' + shlex.quote(value))
        else:
            rendered.append(shlex.quote(argument))
    return ' '.join(rendered)


def read_log_processes(logs, proc_root='/proc'):
    masks = {path: set() for path in logs.values()}
    candidates = {}
    try:
        pids = os.listdir(proc_root)
    except OSError:
        return {path: [] for path in masks}, {path: [] for path in masks}
    for pid in pids:
        if not pid.isdigit():
            continue
        root = os.path.join(proc_root, pid)
        try:
            if os.stat(root).st_uid != os.getuid():
                continue
            matched = set()
            for fd in ('1', '2'):
                try:
                    stat = os.stat(os.path.join(root, 'fd', fd))
                    path = logs.get(f'{stat.st_dev}:{stat.st_ino}')
                    if path:
                        matched.add(path)
                except OSError:
                    pass
            if not matched:
                continue
            with open(os.path.join(root, 'environ'), 'rb') as stream:
                values = stream.read(262144).split(b'\0')
            selected = {}
            for entry in values:
                key, _, value = entry.partition(b'=')
                if key in (b'CUDA_VISIBLE_DEVICES', b'NVIDIA_VISIBLE_DEVICES'):
                    selected[key] = value.decode('ascii', 'replace').strip()
            mask = selected.get(b'CUDA_VISIBLE_DEVICES', '')
            if not mask or mask == '-1':
                continue
            # A restricted container may renumber CUDA devices. Leave such a
            # numeric mask unknown rather than mislabel a host/web GPU index.
            container_mask = selected.get(b'NVIDIA_VISIBLE_DEVICES', 'all')
            if container_mask != 'all' and not mask.startswith('GPU-'):
                mask = '?'
            for path in matched:
                masks[path].add(mask)
            try:
                with open(os.path.join(root, 'cmdline'), 'rb') as stream:
                    raw = stream.read(65537)
                if len(raw) > 65536 or not raw.endswith(b'\0'):
                    continue
                argv = [part.decode('utf-8', 'replace') for part in raw.split(b'\0') if part]
                if not argv:
                    continue
                with open(os.path.join(root, 'status')) as stream:
                    status = dict(line.split(':', 1) for line in stream if ':' in line)
                candidates[int(pid)] = dict(
                    pid=int(pid), ppid=int(status['PPid'].strip()),
                    name=status.get('Name', '').strip(), command=format_process_argv(argv),
                    paths=matched, mask=mask,
                )
            except (OSError, ValueError, KeyError):
                pass
        except OSError:
            continue
    all_masks = dict(masks)
    all_masks.update({pid: {row['mask']} for pid, row in candidates.items()})
    gpu_uuids = {}
    if any('GPU-' in mask for values in all_masks.values() for mask in values):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=3, check=True,
            )
            for line in result.stdout.splitlines():
                index, uuid = (part.strip() for part in line.split(','))
                if index.isdigit() and 0 <= int(index) < 8:
                    gpu_uuids[uuid] = int(index)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    mapped = {}
    for path, values in all_masks.items():
        indexes, valid = set(), True
        for mask in values:
            for token in mask.split(','):
                token = token.strip()
                if token.isdigit() and 0 <= int(token) < 8:
                    indexes.add(int(token))
                elif token.startswith('GPU-'):
                    matches = [index for uuid, index in gpu_uuids.items() if uuid.startswith(token)]
                    if len(matches) == 1:
                        indexes.add(matches[0])
                    else:
                        valid = False
                else:
                    valid = False
        mapped[path] = sorted(indexes) if valid else []
    processes = {path: [] for path in masks}
    for pid, row in sorted(candidates.items()):
        for path in row['paths']:
            # Forked data-loader workers inherit both CUDA masks and log fds.
            # Show the training parent once, not every inherited worker PID.
            parent, visited = row['ppid'], {pid}
            inherited = False
            while parent in candidates and parent not in visited:
                visited.add(parent)
                ancestor = candidates[parent]
                if path in ancestor['paths']:
                    inherited = True
                    break
                parent = ancestor['ppid']
            if not inherited and mapped[pid]:
                processes[path].append(dict(
                    pid=pid, name=row['name'], command=row['command'],
                    gpu_indexes=mapped[pid],
                ))
    return {path: mapped[path] for path in masks}, processes
""".strip()

# The program reads bounded tails and metadata only, never a complete log. Its
# own server clock lets us measure file age without assuming synchronized hosts.
REMOTE_TRAINING_READER = (
    """
import json, os, shlex, subprocess, sys, time
"""
    + REMOTE_GPU_HELPERS
    + """
rows = []
for path in sys.argv[1:]:
    try:
        with open(path, 'rb') as stream:
            stat = os.fstat(stream.fileno())
            start = max(0, stat.st_size - 32768)
            stream.seek(start)
            tail = stream.read(32768).decode('utf-8', 'replace')
            if start:
                # Discard the first possibly partial CR-delimited tqdm record.
                ends = [i for i in (tail.find('\\r'), tail.find('\\n')) if i >= 0]
                tail = tail[min(ends) + 1:] if ends else ''
            rows.append(dict(path=path, tail=tail, size=stat.st_size,
                             identity=f'{stat.st_dev}:{stat.st_ino}',
                             mtime=stat.st_mtime, server_now=time.time()))
    except OSError as exc:
        rows.append(dict(path=path, error=str(exc)))
associations, processes = read_log_processes({row['identity']: row['path'] for row in rows if 'identity' in row})
for row in rows:
    row['gpu_indexes'] = associations.get(row['path'], [])
    row['processes'] = processes.get(row['path'], [])
print(json.dumps(rows, ensure_ascii=False))
"""
).strip()


class RemoteError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingLog:
    label: str
    path: str
    step_offset: int = 0
    gpu_index: int | None = None

    @property
    def filename(self) -> str:
        return Path(self.path).name

    @property
    def gpu_label(self) -> str:
        return f"GPU {self.gpu_index}" if self.gpu_index is not None else "GPU ?"


@dataclass(frozen=True)
class RemoteConfig:
    target: str
    port: int
    jump: str
    control_path: str
    logs: tuple[TrainingLog, ...]
    interval: float = 10
    timeout: float = 15
    stale_after: float = 180


@dataclass(frozen=True)
class TrainingProgress:
    step: int
    total: int
    elapsed_seconds: int | None = None
    eta_seconds: int | None = None
    seconds_per_step: float | None = None

    @property
    def percent(self) -> float:
        return 100 * self.step / self.total


@dataclass(frozen=True)
class TrainingProcess:
    """An own-user PID and argv read in the SSH terminal's PID namespace."""

    pid: int
    name: str
    command: str
    gpu_indexes: tuple[int, ...]


@dataclass(frozen=True)
class TrainingSample:
    log: TrainingLog
    progress: TrainingProgress | None = None
    checked_at: float = 0  # Local monotonic clock, not a wall-clock timestamp.
    file_age: float = 0  # Server clock minus that server's file mtime.
    mtime: float | None = None
    size: int = 0
    identity: str = ""
    error: str = ""
    gpu_indexes: tuple[int, ...] = ()
    gpu_source: str = ""
    processes: tuple[TrainingProcess, ...] = ()

    @property
    def effective_gpu_indexes(self) -> tuple[int, ...]:
        # An explicit physical/web index can override a container's remapping.
        return (
            (self.log.gpu_index,)
            if self.log.gpu_index is not None
            else self.gpu_indexes
        )

    @property
    def gpu_label(self) -> str:
        indexes = self.effective_gpu_indexes
        return (
            "GPU " + "/".join(str(index) for index in indexes) if indexes else "GPU ?"
        )

    def status(self, stale_after: float, *, now: float | None = None) -> str:
        if self.error:
            return "读取失败；上次数据" if self.progress else "读取失败"
        if not self.checked_at:
            return "等待日志"
        if not self.progress:
            return "尚无训练进度"
        # Reaching the last tqdm step is not proof that final checkpoint saving
        # has finished, so deliberately do not claim the entire job succeeded.
        if self.progress.step == self.progress.total:
            return "步数已达 100%"
        age = self.file_age + max(
            0, (now if now is not None else time.monotonic()) - self.checked_at
        )
        return "日志未更新" if age >= stale_after else "日志活跃"

    def is_fresh(self, stale_after: float, *, now: float | None = None) -> bool:
        return self.status(stale_after, now=now) in ("日志活跃", "步数已达 100%")


@dataclass(frozen=True)
class TrainingEvent:
    samples: tuple[TrainingSample, ...] = ()
    error: str = ""


def _gpu_index(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value < 8:
        raise ValueError("gpu_index 必须是网页中的 GPU 编号（0–7）或 null")
    return value


def _gpu_indexes(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(
        type(index) is int and 0 <= index < 8 for index in value
    ):
        return ()
    return tuple(sorted(set(value)))


def _training_processes(value: Any) -> tuple[TrainingProcess, ...]:
    """Invalid optional process metadata must not discard valid log progress."""
    if not isinstance(value, list) or len(value) > 256:
        return ()
    processes: dict[int, TrainingProcess] = {}
    conflicts: set[int] = set()
    for row in value:
        if not isinstance(row, dict):
            continue
        pid = row.get("pid")
        name, command = row.get("name", ""), row.get("command")
        indexes = _gpu_indexes(row.get("gpu_indexes"))
        if (
            type(pid) is not int
            or not 0 < pid <= 2**31 - 1
            or not isinstance(name, str)
            or len(name) > 256
            or not isinstance(command, str)
            or not command.strip()
            or len(command) > 65536
            or re.search(r"[\x00-\x1f\x7f]", name + command)
            or not indexes
        ):
            continue
        process = TrainingProcess(pid, name, command.strip(), indexes)
        if pid in processes and processes[pid] != process:
            conflicts.add(pid)
        processes[pid] = process
    return tuple(processes[pid] for pid in sorted(processes) if pid not in conflicts)


def load_config(path: str | Path = DEFAULT_CONFIG) -> RemoteConfig:
    try:
        path = Path(path)
        # Only the exact former bundled path is a migration alias. Missing
        # custom configurations must still fail, never silently use defaults.
        legacy = Path(__file__).resolve().parent.parent / "wn_a100_training.json"
        if not path.exists() and path.resolve() == legacy:
            path = DEFAULT_CONFIG
        raw = json.loads(path.read_text(encoding="utf-8"))
        target = str(raw["target"])
        jump = str(raw["jump"])
        if not re.fullmatch(
            r"[A-Za-z0-9_][A-Za-z0-9_.-]*@[A-Za-z0-9][A-Za-z0-9_.-]*", target
        ):
            raise ValueError("target 必须是 user@host")
        if not re.fullmatch(
            r"[A-Za-z0-9_][A-Za-z0-9_.-]*@[A-Za-z0-9][A-Za-z0-9_.-]*:\d+", jump
        ):
            raise ValueError("jump 必须是 user@host:port")
        if not 1 <= int(jump.rsplit(":", 1)[1]) <= 65535:
            raise ValueError("跳板机 SSH 端口超出范围")
        port = int(raw["port"])
        if not 1 <= port <= 65535:
            raise ValueError("SSH 端口超出范围")
        runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
        control_path = str(raw["control_path"]).replace("{runtime_dir}", runtime)
        if not Path(control_path).is_absolute() or len(os.fsencode(control_path)) > 100:
            raise ValueError("control_path 必须是较短的绝对路径")
        logs = tuple(
            TrainingLog(
                str(item["label"]),
                str(item["path"]),
                int(item.get("step_offset", 0)),
                _gpu_index(item.get("gpu_index")),
            )
            for item in raw["logs"]
        )
        if len(logs) != 2 or len({log.path for log in logs}) != 2:
            raise ValueError("必须指定两个不同的训练日志")
        if any(not Path(log.path).is_absolute() or log.step_offset < 0 for log in logs):
            raise ValueError("日志需使用绝对路径，step_offset 不可为负数")
        interval, timeout, stale_after = (
            float(raw.get(key, default))
            for key, default in (
                ("interval", 10),
                ("timeout", 15),
                ("stale_after", 180),
            )
        )
        if not all(
            math.isfinite(value) and value > 0
            for value in (interval, timeout, stale_after)
        ):
            raise ValueError("刷新间隔、超时和停更阈值必须为正数")
        return RemoteConfig(
            target, port, jump, control_path, logs, interval, timeout, stale_after
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RemoteError(f"训练日志配置不可用：{exc}") from exc


def login_command(config: RemoteConfig) -> str:
    return shlex.join(
        [
            "env",
            "-u",
            "LD_LIBRARY_PATH",
            "ssh",
            "-4",
            "-M",
            "-S",
            config.control_path,
            "-o",
            "ControlPersist=8h",
            "-J",
            config.jump,
            "-p",
            str(config.port),
            "-fN",
            config.target,
        ]
    )


def ssh_command(config: RemoteConfig, remote_command: str | None = None) -> list[str]:
    if remote_command is None:
        remote_command = shlex.join(
            [
                "python3",
                "-c",
                REMOTE_TRAINING_READER,
                *(log.path for log in config.logs),
            ]
        )
    return [
        "ssh",
        "-4",
        "-T",
        "-n",
        "-S",
        config.control_path,
        "-o",
        "ControlMaster=no",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "PermitLocalCommand=no",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ProxyCommand=false",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "ConnectionAttempts=1",
        "-p",
        str(config.port),
        config.target,
        remote_command,
    ]


def _duration(raw: str) -> int | None:
    if not re.fullmatch(r"\d+(?::\d{2}){1,2}", raw.strip()):
        return None
    parts = [int(part) for part in raw.strip().split(":")]
    if any(part >= 60 for part in parts[1:]):
        return None
    return sum(value * 60**index for index, value in enumerate(reversed(parts)))


def parse_progress(
    tail: str, *, step_offset: int = 0
) -> tuple[TrainingProgress | None, str]:
    text = ANSI_ESCAPE.sub("", tail).replace("\r", "\n")
    matches = list(TRAINING_BAR.finditer(text))
    if not matches:
        return None, ""
    latest = matches[-1]
    step = int(latest["step"].replace(",", ""))
    total = int(latest["total"].replace(",", ""))
    if total <= 0 or not 0 <= step <= total:
        return None, "日志中的训练步数无效"
    timing = latest["timing"]
    elapsed_text, _, remaining = timing.partition("<")
    eta_text, _, rate_text = remaining.partition(",")
    rate = re.search(r"(\d+(?:\.\d+)?)\s*(s/step|s/it|it/s|step/s)", rate_text)
    seconds_per_step = None
    if rate:
        value = float(rate[1])
        if math.isfinite(value) and value > 0:
            seconds_per_step = value if rate[2].startswith("s/") else 1 / value
    progress = TrainingProgress(
        step + step_offset,
        total + step_offset,
        _duration(elapsed_text),
        _duration(eta_text),
        seconds_per_step,
    )
    trailing = text[latest.end() :]
    error = (
        "最新进度之后出现异常，需检查训练日志"
        if "Traceback (most recent call last):" in trailing
        else ""
    )
    return progress, error


def parse_training_samples(
    payload: Any, config: RemoteConfig, *, now: float | None = None
) -> tuple[TrainingSample, ...]:
    if not isinstance(payload, list) or len(payload) != len(config.logs):
        raise RemoteError("远端日志响应格式不正确")
    rows = {row.get("path"): row for row in payload if isinstance(row, dict)}
    if set(rows) != {log.path for log in config.logs}:
        raise RemoteError("远端返回的日志路径不匹配")
    checked_at = now if now is not None else time.monotonic()
    samples = []
    for log in config.logs:
        row = rows[log.path]
        if row.get("error"):
            samples.append(
                TrainingSample(
                    log, checked_at=checked_at, error=str(row["error"])[:1000]
                )
            )
            continue
        try:
            mtime, server_now = float(row["mtime"]), float(row["server_now"])
            if not math.isfinite(mtime) or not math.isfinite(server_now):
                raise ValueError("日志时间无效")
            progress, error = parse_progress(
                str(row["tail"]), step_offset=log.step_offset
            )
            gpu_indexes = _gpu_indexes(row.get("gpu_indexes", []))
            samples.append(
                TrainingSample(
                    log,
                    progress,
                    checked_at,
                    max(0, server_now - mtime),
                    mtime,
                    int(row["size"]),
                    str(row["identity"]),
                    error,
                    gpu_indexes,
                    "进程 CUDA_VISIBLE_DEVICES" if gpu_indexes else "",
                    _training_processes(row.get("processes")),
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            samples.append(
                TrainingSample(log, checked_at=checked_at, error=f"日志解析失败：{exc}")
            )
    return tuple(samples)


def apply_training_event(
    previous: tuple[TrainingSample, ...], event: TrainingEvent
) -> tuple[TrainingSample, ...]:
    """Only a transport/file-read failure may retain a grayed-out last value.

    A successfully read empty/new/truncated log must clear the preceding run.
    No monotonic max is taken: a new run can legitimately restart at step zero.
    """
    if event.error:
        return tuple(
            replace(sample, error=event.error, processes=()) for sample in previous
        )
    old = {sample.log.path: sample for sample in previous}
    samples = []
    for sample in event.samples:
        earlier = old.get(sample.log.path)
        if sample.error and sample.progress is None and not sample.identity and earlier:
            sample = replace(earlier, error=sample.error, processes=())
        elif (
            earlier
            and not sample.gpu_indexes
            and earlier.gpu_indexes
            and sample.identity
            and sample.identity == earlier.identity
            and sample.size == earlier.size
            and sample.mtime == earlier.mtime
        ):
            # Keep a completed/unchanged log's last verified card association.
            # A rotated, truncated or newly appended run must be matched anew.
            sample = replace(
                sample,
                gpu_indexes=earlier.gpu_indexes,
                gpu_source="上次进程匹配（日志未变化）",
            )
        samples.append(sample)
    return tuple(samples)


def _end_client(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        # Only this short-lived client process group, never the user's master.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.communicate()


def _ssh_output(
    config: RemoteConfig,
    command: str | None = None,
    *,
    stop_event: threading.Event | None = None,
    timeout: float | None = None,
    max_bytes: int = 1024 * 1024,
    discard_output: bool = False,
) -> str:
    """One reuse-only transport for logs, disk scans and health checks.

    Cancel/reap only this newly created client group, never the shared master.
    Domain parsers and independent refresh intervals stay in their callers.
    """
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    process = None
    try:
        process = subprocess.Popen(
            ssh_command(config, command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL if discard_output else subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            start_new_session=True,
        )
        deadline = time.monotonic() + (config.timeout if timeout is None else timeout)
        while True:
            if stop_event and stop_event.is_set():
                raise RemoteError("SSH 读取已取消")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RemoteError("SSH 读取超时；保留上次数据，正在重试")
            try:
                output, error = process.communicate(timeout=min(0.25, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            detail = error.strip()[-800:]
            raise RemoteError(
                "SSH 共享连接不可用，请在终端重新登录。"
                + (f"\n{detail}" if detail else "")
            )
        output = output or ""
        if len(output.encode("utf-8")) > max_bytes:
            raise RemoteError("SSH 响应过大")
        return output
    except OSError as exc:
        raise RemoteError(f"SSH 读取失败：{exc}") from exc
    finally:
        if process is not None:
            _end_client(process)


def fetch_training(
    config: RemoteConfig, stop_event: threading.Event | None = None
) -> tuple[TrainingSample, ...]:
    output = _ssh_output(config, stop_event=stop_event)
    try:
        return parse_training_samples(json.loads(output), config)
    except ValueError as exc:
        raise RemoteError(f"训练日志读取失败：{exc}") from exc


def demo_samples(config: RemoteConfig, tick: int = 0) -> tuple[TrainingSample, ...]:
    now = time.monotonic()
    return tuple(
        TrainingSample(
            log,
            TrainingProgress(min(total, step + tick), total, 5000, eta, 4.69),
            now,
            gpu_indexes=(gpu_index,),
            gpu_source="演示配对",
        )
        for log, (step, total, eta, gpu_index) in zip(
            config.logs, ((1060, 20000, 88893, 3), (5033, 20000, 68919, 6)), strict=True
        )
    )


def format_duration(seconds: int | None) -> str:
    if seconds is None:
        return "待估算"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def training_text(sample: TrainingSample, config: RemoteConfig) -> str:
    status = sample.status(config.stale_after)
    lines = [
        f"{sample.log.label} · {sample.log.filename} · {sample.gpu_label} · {status}"
    ]
    if not sample.effective_gpu_indexes:
        lines.append(
            "卡号尚未识别；自动读取写入该日志的进程 CUDA_VISIBLE_DEVICES，不猜测文件名或卡号顺序"
        )
    elif sample.log.gpu_index is not None:
        lines.append("卡号来源：逐日志配置覆盖（网页编号）")
    elif sample.gpu_source:
        lines.append(f"卡号来源：{sample.gpu_source}")
    if sample.progress:
        progress = sample.progress
        lines.append(f"{progress.percent:.2f}% · {progress.step}/{progress.total} 步")
        qualifier = (
            "最新日志估计"
            if sample.is_fresh(config.stale_after)
            else "上次日志估计，非当前倒计时"
        )
        lines.append(f"预计剩余 {format_duration(progress.eta_seconds)}（{qualifier}）")
    if sample.error:
        lines.append(sample.error)
    return "\n".join(lines)


class TrainingPoller(threading.Thread):
    def __init__(
        self,
        events: queue.Queue[TrainingEvent],
        config: RemoteConfig,
        *,
        demo: bool = False,
    ) -> None:
        super().__init__(daemon=True, name="wn-a100-training")
        self.events = events
        self.config = config
        self.demo = demo
        self.stop_event = threading.Event()
        self.refresh_event = threading.Event()

    def refresh(self) -> None:
        self.refresh_event.set()

    def stop(self) -> None:
        self.stop_event.set()
        self.refresh_event.set()

    def run(self) -> None:
        tick, failures = 0, 0
        while not self.stop_event.is_set():
            self.refresh_event.clear()
            try:
                samples = (
                    demo_samples(self.config, tick)
                    if self.demo
                    else fetch_training(self.config, self.stop_event)
                )
                event = TrainingEvent(samples=samples)
                failures = 0
            except RemoteError as exc:
                event = TrainingEvent(error=str(exc))
                failures += 1
            if self.stop_event.is_set():
                break
            try:
                self.events.put_nowait(event)
            except queue.Full:
                try:
                    self.events.get_nowait()
                except queue.Empty:
                    pass  # The UI may have drained the queue in the meantime.
                self.events.put_nowait(event)
            tick += 1
            delay = (
                self.config.interval
                if not failures
                else min(
                    max(30, self.config.interval),
                    self.config.interval * 2 ** min(failures, 4),
                )
            )
            self.refresh_event.wait(delay)


# Low-frequency HOME disk usage.

TIB = 1024**4
QUOTA_BYTES = 3 * TIB
DEFAULT_INTERVAL = 900.0
DEFAULT_TIMEOUT = 195.0
DU_COMMAND = 'du -sh "$HOME"'
SIZE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)([KMGTPEZY]?)")

# The shell expands HOME remotely, never on the desktop. No second du pass,
# apparent-size mode, symlink dereferencing, remote files or credentials.
REMOTE_DISK_READER = r"""
import json, os, pwd, signal, subprocess, sys, time
process = None
row = {'user': pwd.getpwuid(os.getuid()).pw_name, 'home': os.environ.get('HOME', '')}
def cancel(signum, frame):
    raise SystemExit(128 + signum)
for signum in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
    signal.signal(signum, cancel)
try:
    if row['user'] != 'yzh' or not os.path.isabs(row['home']) or row['home'] == '/':
        raise ValueError('磁盘统计只允许 yzh 用户自己的 HOME')
    try:
        os.nice(10)
    except OSError:
        pass
    environment = os.environ.copy()
    environment['LC_ALL'] = 'C'
    process = subprocess.Popen(
        ['sh', '-c', 'du -sh "$HOME"'],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8', errors='replace', env=environment,
        start_new_session=True,
    )
    output, error = process.communicate(timeout=float(sys.argv[1]))
    if process.returncode:
        raise RuntimeError('du 统计失败或不完整：' + error.strip()[-1000:])
    row['output'] = output
except subprocess.TimeoutExpired:
    row['error'] = 'du 扫描超时，保留上次占用；可增加 --disk-timeout'
except (OSError, ValueError, RuntimeError) as exc:
    row['error'] = str(exc)
finally:
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
row['finished_at'] = time.time()
print(json.dumps(row, ensure_ascii=False))
""".strip()


class DiskError(RemoteError):
    pass


@dataclass(frozen=True)
class DiskConfig:
    connection: RemoteConfig
    interval: float = DEFAULT_INTERVAL
    timeout: float = DEFAULT_TIMEOUT

    @property
    def remote_timeout(self) -> float:
        return max(0.25, self.timeout - 15)

    @property
    def stale_after(self) -> float:
        return 2 * self.interval + self.timeout


@dataclass(frozen=True)
class DiskSample:
    # Approximate bytes converted from du's human-readable output, not an
    # invented byte-precise measurement. Keep the original displayed size.
    size_text: str = ""
    used_bytes: int | None = None
    checked_at: float = 0.0
    received_at: float = 0.0
    home: str = ""
    error: str = ""
    scanning: bool = False

    @property
    def over_quota(self) -> bool:
        return self.used_bytes is not None and self.used_bytes > QUOTA_BYTES

    def is_stale(self, stale_after: float, *, now: float | None = None) -> bool:
        return bool(
            self.used_bytes is not None
            and (
                self.error
                or (time.monotonic() if now is None else now) - self.checked_at
                >= stale_after
            )
        )


@dataclass(frozen=True)
class DiskEvent:
    sample: DiskSample | None = None
    error: str = ""
    scanning: bool = False


def parse_disk_sample(payload: object, *, now: float | None = None) -> DiskSample:
    if not isinstance(payload, dict):
        raise DiskError("磁盘响应格式错误")
    if payload.get("error"):
        raise DiskError(str(payload["error"]))
    home = payload.get("home", "")
    if (
        payload.get("user") != "yzh"
        or not isinstance(home, str)
        or not home.startswith("/")
        or home == "/"
    ):
        raise DiskError("响应不是 yzh 的 HOME 用量")
    output = payload.get("output", "")
    if not isinstance(output, str) or len(output) > 8192:
        raise DiskError("du 响应异常")
    size_text, separator, reported_home = output.rstrip("\n").partition("\t")
    match = SIZE_PATTERN.fullmatch(size_text)
    if not separator or not match or reported_home != home:
        raise DiskError("无法解析 du -sh 的用量及 HOME")
    number, unit = match.groups()
    factor = 1024 ** (" KMGTPEZY".index(unit) if unit else 0)
    used_bytes = int(Decimal(number) * factor)
    return DiskSample(
        size_text,
        used_bytes,
        time.monotonic() if now is None else now,
        time.time(),
        home,
    )


def apply_disk_event(sample: DiskSample, event: DiskEvent) -> DiskSample:
    if event.sample is not None:
        return replace(event.sample, scanning=False, error="")
    if event.scanning:
        return replace(sample, scanning=True)
    return replace(sample, scanning=False, error=event.error)


def _sample_status(sample: DiskSample, stale_after: float) -> tuple[str, str]:
    if sample.used_bytes is None:
        return (
            "统计中" if sample.scanning else "读取失败" if sample.error else "等待统计",
            "unknown",
        )
    stale = sample.is_stale(stale_after)
    if sample.over_quota:
        status = "超额 · 上次" if stale else "超额 · yzh"
        level = "over"
    elif stale:
        status, level = "上次 · 更新失败" if sample.error else "上次 · 已过期", "stale"
    else:
        status = (
            "刷新中 · yzh"
            if sample.scanning
            else f"yzh · {sample.used_bytes / QUOTA_BYTES:.1%}"
        )
        level = "normal"
    return status, level


def disk_rows(sample: DiskSample, stale_after: float) -> tuple[tuple[str, str], str]:
    """Keep the carousel to usage/quota and percentage; details live in the tooltip."""
    _status, level = _sample_status(sample, stale_after)
    if sample.used_bytes is None:
        return (("…/3T", "—%"), level)
    return (
        (f"{sample.size_text}/3T", f"{sample.used_bytes / QUOTA_BYTES:.1%}"),
        level,
    )


def disk_text(sample: DiskSample, config: DiskConfig | None) -> str:
    stale_after = (
        config.stale_after if config else 2 * DEFAULT_INTERVAL + DEFAULT_TIMEOUT
    )
    status, _level = _sample_status(sample, stale_after)
    lines = [
        f"yzh 磁盘：{sample.size_text or '待统计'} / 3T · {status}",
        f"远端指令：{DU_COMMAND}",
    ]
    if sample.home:
        lines.append(f"统计目录：{sample.home}")
    if sample.received_at:
        when = datetime.fromtimestamp(sample.received_at, ZoneInfo("Asia/Shanghai"))
        lines.append(f"上次完成于 {when:%m-%d %H:%M:%S}")
    lines.append("K/M/G/T 按 1024 换算；du -h 是近似显示，以报告值判断是否超过 3T。")
    lines.append(
        f"每 {(config.interval if config else DEFAULT_INTERVAL) / 60:g} 分钟独立统计；F5 可刷新，扫描中不重复启动。"
    )
    if sample.error:
        lines.append(sample.error)
    return "\n".join(lines)


def fetch_disk(
    config: DiskConfig, stop_event: threading.Event | None = None
) -> DiskSample:
    if not config.connection.target.startswith("yzh@"):
        raise DiskError("只允许统计 yzh 用户的磁盘占用")
    command = shlex.join(
        ["python3", "-c", REMOTE_DISK_READER, str(config.remote_timeout)]
    )
    try:
        output = _ssh_output(
            config.connection,
            command,
            stop_event=stop_event,
            timeout=config.timeout,
            max_bytes=16384,
        )
        return parse_disk_sample(json.loads(output))
    except DiskError:
        raise
    except (RemoteError, ValueError) as exc:
        raise DiskError(f"磁盘读取失败：{exc}") from exc


class DiskPoller(threading.Thread):
    def __init__(
        self, events: queue.Queue[DiskEvent], config: DiskConfig, *, demo: bool = False
    ) -> None:
        super().__init__(daemon=True, name="wn-a100-disk")
        self.events, self.config, self.demo = events, config, demo
        self.stop_event = threading.Event()
        self.refresh_event = threading.Event()
        self.scanning_event = threading.Event()

    def refresh(self) -> None:
        if not self.scanning_event.is_set():
            self.refresh_event.set()

    def stop(self) -> None:
        self.stop_event.set()
        self.refresh_event.set()

    def _publish(self, event: DiskEvent) -> None:
        if self.stop_event.is_set():
            return
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            self.events.put_nowait(event)

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.refresh_event.clear()
            self.scanning_event.set()
            self._publish(DiskEvent(scanning=True))
            try:
                sample = (
                    parse_disk_sample(
                        {
                            "user": "yzh",
                            "home": "/home/yzh",
                            "output": "2.3T\t/home/yzh\n",
                        }
                    )
                    if self.demo
                    else fetch_disk(self.config, self.stop_event)
                )
                self._publish(DiskEvent(sample=sample))
            except DiskError as exc:
                self._publish(DiskEvent(error=str(exc)))
            finally:
                self.scanning_event.clear()
            self.refresh_event.wait(self.config.interval)


# Terminal authentication and independent connection health.


@dataclass(frozen=True)
class SessionEvent:
    connected: bool
    error: str = ""


def check_session(
    config: RemoteConfig, stop: threading.Event | None = None
) -> SessionEvent:
    # A successful local `ssh -O check` alone does not prove the remote channel
    # is usable. A no-op over the authenticated master checks both ends.
    try:
        if not Path(config.control_path).is_socket():
            return SessionEvent(False, "未找到 SSH 共享连接，请点击登录页连接。")
        _ssh_output(
            config,
            "true",
            stop_event=stop,
            timeout=min(5, config.timeout),
            discard_output=True,
        )
        return SessionEvent(True)
    except (RemoteError, OSError) as exc:
        return SessionEvent(False, "无法检查 SSH 连接：" + str(exc))


def login_script(config: RemoteConfig) -> str:
    # All configurable words are shell-quoted. Passwords/verification codes are
    # handled by SSH's terminal directly, never captured or saved by this app.
    probe = ["env", "-u", "LD_LIBRARY_PATH", *ssh_command(config, "true")]
    timeout = shutil.which("timeout")
    if timeout:
        probe = [timeout, "10s", *probe]
    return "\n".join(
        (
            "printf '%s\\n' 'WN-A100 SSH 登录' '请在此终端完成认证；监控不会读取或保存密码。'",
            # ssh -f daemonizes its master, but the ProxyJump subprocess can
            # remain in this terminal's process group. Let both inherit HUP
            # ignored before exec, or the shell/terminal exit tears down the
            # just-authenticated transport. SIGINT remains usable to cancel.
            "trap '' HUP",
            login_command(config),
            "wn_login_status=$?",
            'if [ "$wn_login_status" -eq 0 ]; then',
            "    printf '%s\\n' '认证命令已完成，正在验证共享连接…'",
            "    wn_verified=0",
            "    for wn_attempt in 1 2 3; do",
            f"        if {shlex.join(probe)}; then",
            "            wn_verified=1",
            "            break",
            "        fi",
            "        sleep 1",
            "    done",
            '    if [ "$wn_verified" -eq 1 ]; then',
            "        printf '%s\\n' '共享连接验证成功，悬浮球将自动恢复。'",
            "    else",
            "        wn_login_status=1",
            "        printf '%s\\n' '认证后未能建立可用共享连接；这不一定是密码错误，请保留上方报错。'",
            "    fi",
            "fi",
            'if [ "$wn_login_status" -ne 0 ]; then',
            "    printf '%s\\n' '登录未完成，请查看上方错误。修复后可再次点击悬浮球登录页。'",
            "    read -r -p '按 Enter 关闭此窗口…' wn_login_ack",
            "fi",
            'exit "$wn_login_status"',
        )
    )


def terminal_command(config: RemoteConfig) -> list[str]:
    script = login_script(config)
    shell = shutil.which("bash")
    if not shell:
        raise RuntimeError("未找到 bash；请在本机终端手动运行登录命令。")
    # These flags keep the launcher attached to its terminal child, allowing
    # the UI to prevent repeated clicks opening multiple password prompts.
    for name, arguments in (
        (
            "gnome-terminal",
            ["--wait", "--title=WN-A100 SSH 登录", "--", shell, "-c", script],
        ),
        ("konsole", ["--separate", "-e", shell, "-c", script]),
        (
            "xfce4-terminal",
            ["--disable-server", "--command=" + shlex.join([shell, "-c", script])],
        ),
        ("xterm", ["-T", "WN-A100 SSH 登录", "-e", shell, "-c", script]),
        ("x-terminal-emulator", ["-e", shell, "-c", script]),
    ):
        executable = shutil.which(name)
        if executable:
            return [executable, *arguments]
    raise RuntimeError("未找到可用的本机终端；请在“查看训练进度”中复制登录命令。")


def open_login_terminal(config: RemoteConfig) -> subprocess.Popen:
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    return subprocess.Popen(
        terminal_command(config),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=environment,
        start_new_session=True,
    )


class SessionPoller(threading.Thread):
    """Independent health checks, including when no yzh GPU jobs are running."""

    def __init__(self, events: queue.Queue[SessionEvent], config: RemoteConfig):
        super().__init__(daemon=True, name="wn-a100-ssh-session")
        self.events, self.config = events, config
        self.stop_event = threading.Event()
        self.refresh_event = threading.Event()

    def refresh(self) -> None:
        self.refresh_event.set()

    def stop(self) -> None:
        self.stop_event.set()
        self.refresh_event.set()

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.refresh_event.clear()
            event = check_session(self.config, self.stop_event)
            if self.stop_event.is_set():
                break
            try:
                self.events.put_nowait(event)
            except queue.Full:
                try:
                    self.events.get_nowait()
                except queue.Empty:
                    pass
                self.events.put_nowait(event)
            self.refresh_event.wait(5 if event.connected else 2)
