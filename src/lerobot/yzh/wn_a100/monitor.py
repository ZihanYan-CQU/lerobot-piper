#!/usr/bin/env python3
"""A lightweight desktop widget for the WN-A100 GPU server.

The public dashboard already exposes a small JSON endpoint.  This monitor talks
to that endpoint directly, so it does not need browser automation or a local
NVIDIA driver.  The optional Qt desktop shell handles native window operations;
Pillow renders the interface through FreeType.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import shutil
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from . import remote

DEFAULT_API_BASE = "https://gmcn.cslcqu.cn"
DEFAULT_SERVER_ID = "server7"
DEFAULT_SERVER_NAME = "WN-A100"
DEFAULT_INTERVAL = 3.0
DEFAULT_TIMEOUT = 10.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024

WINDOW_WIDTH = 970
WINDOW_HEIGHT = 524
EXPECTED_GPU_COUNT = 8
HISTORY_LENGTH = 48
RENDER_SCALE = 2
FONT_PIXEL_RATIO = 1.28

COLORS = {
    "window": "#070B12",
    "panel": "#0B111C",
    "panel_border": "#1D2A3E",
    "card": "#111A29",
    "card_hover": "#152136",
    "card_border": "#22324A",
    "track": "#202C3F",
    "text": "#F5F7FB",
    "muted": "#8E9AAF",
    "subtle": "#5F6D82",
    "blue": "#4C9BFF",
    "cyan": "#41D6E8",
    "green": "#48D597",
    "amber": "#FFB454",
    "red": "#FF6178",
    "yzh_border": "#55F7A5",
    "conflict_border": "#FFA34D",
    "button_hover": "#1B2940",
}

# Keep user colors identical to gmcn.cslcqu.cn/js/utils.js.  A deterministic
# color makes the same user immediately recognizable across cards and refreshes.
USER_COLORS = (
    "#37A2DA",
    "#67E0E3",
    "#D77162",
    "#96BFFF",
    "#FF9F7F",
    "#FB7293",
    "#E062AE",
    "#E690D1",
    "#E7BCF3",
    "#9D96F5",
    "#8378EA",
    "#32C5E9",
    "#9FE6B8",
    "#FFDB5C",
    "#73A0FA",
    "#73DEB3",
    "#F2637B",
    "#FFB980",
)
HIGHLIGHT_USERNAME = "yzh"


def _font_files() -> tuple[tuple[Path, int], tuple[Path, int]]:
    """Return regular/bold CJK font faces and their TTC indexes."""
    linux_regular = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    linux_bold = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    if linux_regular.is_file():
        return (linux_regular, 2), (
            linux_bold if linux_bold.is_file() else linux_regular,
            2,
        )

    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    windows_regular = windows / "msyh.ttc"
    windows_bold = windows / "msyhbd.ttc"
    if windows_regular.is_file():
        return (windows_regular, 0), (
            windows_bold if windows_bold.is_file() else windows_regular,
            0,
        )

    mac_regular = Path("/System/Library/Fonts/PingFang.ttc")
    if mac_regular.is_file():
        return (mac_regular, 0), (mac_regular, 0)

    fallback = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    fallback_bold = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return (fallback, 0), (fallback_bold if fallback_bold.is_file() else fallback, 0)


class MonitorError(RuntimeError):
    """Raised when the dashboard response cannot be fetched or understood."""


@dataclass(frozen=True, slots=True)
class ProcessCommand:
    """Unverified public endpoint metadata, never an actionable SSH-terminal PID."""

    pid: int
    command: str = ""


@dataclass(frozen=True, slots=True)
class UserGpuUsage:
    """Aggregated GPU-memory usage for one user on one card."""

    username: str
    memory_gib: float
    process_count: int
    pids: tuple[int, ...] = ()
    commands: tuple[ProcessCommand, ...] = ()


@dataclass(frozen=True, slots=True)
class GpuMetric:
    """One normalized GPU status record."""

    index: int
    utilization: float
    memory_percent: float
    memory_used_gib: float
    memory_total_gib: float
    temperature_c: float | None
    power_w: float | None
    power_limit_w: float | None
    process_count: int
    user_usages: tuple[UserGpuUsage, ...]

    @property
    def memory_free_gib(self) -> float:
        return max(0.0, self.memory_total_gib - self.memory_used_gib)

    @property
    def users(self) -> tuple[str, ...]:
        return tuple(usage.username for usage in self.user_usages)

    def has_user(self, username: str) -> bool:
        target = username.casefold()
        return any(usage.username.casefold() == target for usage in self.user_usages)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A complete server sample returned by the dashboard."""

    server_id: str
    server_name: str
    gpus: tuple[GpuMetric, ...]
    received_at: float
    source_timestamp_ms: int | None
    cpu_percent: float | None
    host_memory_percent: float | None
    disk_percent: float | None

    @property
    def average_utilization(self) -> float:
        return sum(gpu.utilization for gpu in self.gpus) / len(self.gpus)

    @property
    def average_memory_percent(self) -> float:
        return sum(gpu.memory_percent for gpu in self.gpus) / len(self.gpus)

    @property
    def idle_gpu_count(self) -> int:
        return sum(
            gpu.utilization < 10 and gpu.memory_percent < 10 for gpu in self.gpus
        )


@dataclass(frozen=True, slots=True)
class Announcement:
    """A currently active notice relevant to this server."""

    notice_id: str
    kind: str
    text: str
    gpu_indexes: tuple[int, ...] = ()
    is_server_rules: bool = False
    expires_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class PollEvent:
    snapshot: Snapshot | None = None
    error: str | None = None
    announcements: tuple[Announcement, ...] | None = None
    notice_error: str | None = None


class _NoticeTextParser(HTMLParser):
    """Extract announcement text without interpreting the page's HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.hidden += 1
        elif tag in {"p", "br", "li", "div"} and not self.hidden:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif tag in {"p", "li", "div"} and not self.hidden:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def _notice_plain_text(content: str) -> str:
    parser = _NoticeTextParser()
    parser.feed(content)
    return "\n".join(
        line
        for raw in "".join(parser.parts).splitlines()
        if (line := " ".join(raw.split()))
    )


def _is_default_server_rules(kind: str, text: str) -> bool:
    """Recognize the standing rules, which the API labels as ordinary operations.

    Match the rule wording rather than a mutable notice ID, ignoring HTML and
    spacing. A temporary maintenance or quota announcement must remain visible.
    """
    compact = re.sub(r"\s+", "", text)
    return kind == "operations" and all(
        phrase in compact
        for phrase in ("默认情况下", "每个用户每台服务器最多使用", "服务器使用规范")
    )


def format_notice_deadline(deadline: datetime | None) -> str:
    """Display an API deadline in local time without a timezone suffix."""
    if deadline is None:
        return "未注明"
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    return deadline.astimezone().strftime("%Y-%m-%d %H:%M")


def parse_announcements(
    payload: Any,
    *,
    server_id: str = DEFAULT_SERVER_ID,
    server_name: str = DEFAULT_SERVER_NAME,
    now: datetime | None = None,
) -> tuple[Announcement, ...]:
    """Filter and format the public /api/notifications payload like the website."""
    rows = payload.get("notifications") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise MonitorError("公告接口没有有效的 notifications 列表")
    current_time = now or datetime.now(timezone.utc)
    result: list[Announcement] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "active":
            continue
        expiration = None
        try:
            raw_expiration = row.get("expires_at")
            if raw_expiration:
                expiration = datetime.fromisoformat(
                    str(raw_expiration).replace("Z", "+00:00")
                )
                if expiration.tzinfo is None:
                    expiration = expiration.replace(tzinfo=timezone.utc)
                if expiration <= current_time:
                    continue
        except ValueError:
            pass

        kind = str(row.get("type") or "operations")
        allocations = row.get("allocations")
        if not isinstance(allocations, list) or not allocations:
            server_ids = row.get("server_ids")
            server_ids = server_ids if isinstance(server_ids, list) else []
            allocations = [
                {"server_id": key, "gpu_indexes": row.get("gpu_indexes", [])}
                for key in server_ids
            ]
        matched = [
            allocation
            for allocation in allocations
            if isinstance(allocation, dict) and allocation.get("server_id") == server_id
        ]
        gpu_indexes: tuple[int, ...] = ()
        if kind == "gpu_usage":
            if not matched:
                continue
            gpu_indexes = tuple(
                sorted(
                    {
                        index
                        for item in matched
                        for index in _device_indexes(item.get("gpu_indexes"))
                        if index >= 0
                    }
                )
            )
            gpu_label = "、".join(str(index) for index in gpu_indexes)
            allocation_text = (
                f"{server_name}的{gpu_label}号GPU"
                if gpu_label
                else f"{server_name}（卡号待补充）"
            )
            reason = _notice_plain_text(str(row.get("content") or ""))
            template = str(
                row.get("template_text")
                or "{reason_text}{user}需要占用{allocation_text}{time_text}，感谢支持！"
            )
            replacements = {
                "reason_text": f"因{reason}，" if reason else "",
                "user": str(row.get("user") or "未知用户"),
                "allocation_text": allocation_text,
                "time_text": (
                    f"，预计占用至{format_notice_deadline(expiration)}"
                    if row.get("show_time") and expiration
                    else ""
                ),
            }
            text = re.sub(
                r"\{([a-z_]+)\}",
                lambda match: replacements.get(match.group(1), match.group(0)),
                template,
            )
        else:
            if server_id not in (row.get("server_ids") or []) and not matched:
                continue
            text = _notice_plain_text(str(row.get("content") or ""))
        if text.strip():
            result.append(
                Announcement(
                    str(row.get("id") or ""),
                    kind,
                    text.strip(),
                    gpu_indexes,
                    _is_default_server_rules(kind, text),
                    expiration,
                )
            )
    # The full viewer retains the standing rules; the banner excludes them.
    result.sort(key=lambda item: item.kind != "gpu_usage")
    return tuple(result)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        result = float(value)
        return result if math.isfinite(result) else None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    result = float(match.group(0))
    return result if math.isfinite(result) else None


def _clamp_percent(value: float | None) -> float:
    return min(100.0, max(0.0, value or 0.0))


def _javascript_int32(value: int) -> int:
    """Apply JavaScript's signed 32-bit conversion used by the web color hash."""
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def user_color(username: str) -> str:
    """Return exactly the deterministic username color used by the web dashboard."""
    hash_value = 0
    encoded = username.encode("utf-16-le", errors="surrogatepass")
    for offset in range(0, len(encoded), 2):
        code_unit = int.from_bytes(encoded[offset : offset + 2], "little")
        shifted = _javascript_int32(_javascript_int32(hash_value) << 5)
        hash_value = code_unit + shifted - hash_value
    return USER_COLORS[abs(hash_value) % len(USER_COLORS)]


def parse_capacity_gib(value: Any) -> float | None:
    """Convert dashboard memory strings such as ``6560MiB`` to GiB."""
    number = _number(value)
    if number is None:
        return None
    if isinstance(value, int | float):
        return number / 1024.0

    text = str(value).strip().lower().replace(" ", "")
    if "tib" in text or text.endswith("tb"):
        return number * 1024.0
    if "gib" in text or text.endswith("gb"):
        return number
    if "mib" in text or text.endswith("mb"):
        return number / 1024.0
    if "kib" in text or text.endswith("kb"):
        return number / (1024.0 * 1024.0)
    if text.endswith("b") or "byte" in text:
        return number / (1024.0**3)
    # Bare values in this API are MiB (for example process.gpu_memory).
    return number / 1024.0


def _parse_memory_row(row: dict[str, Any]) -> tuple[float, float, float]:
    used = parse_capacity_gib(row.get("memory_used_human"))
    total = parse_capacity_gib(row.get("memory_total_human"))
    free = parse_capacity_gib(row.get("memory_free_human"))

    usage = row.get("memory_usage")
    if isinstance(usage, str) and "/" in usage:
        raw_used, raw_total = usage.split("/", 1)
        used = used if used is not None else parse_capacity_gib(raw_used)
        total = total if total is not None else parse_capacity_gib(raw_total)

    if total is None and used is not None and free is not None:
        total = used + free
    if used is None and total is not None and free is not None:
        used = max(0.0, total - free)

    percent = _number(row.get("memory_percent"))
    if percent is None and used is not None and total:
        percent = used / total * 100.0
    if total is None or total <= 0:
        raise MonitorError("GPU 数据缺少有效的显存总量")
    if used is None:
        used = total * _clamp_percent(percent) / 100.0
    return max(0.0, used), total, _clamp_percent(percent)


def _parse_power(row: dict[str, Any]) -> tuple[float | None, float | None]:
    current = _number(row.get("power_usage"))
    limit = None
    status = row.get("power_status")
    if isinstance(status, str):
        values = re.findall(r"-?\d+(?:\.\d+)?", status)
        if current is None and values:
            current = float(values[0])
        if len(values) >= 2:
            limit = float(values[1])
    return current, limit


def _device_indexes(value: Any) -> tuple[int, ...]:
    if isinstance(value, bool) or value is None:
        return ()
    if isinstance(value, int | float):
        return (int(value),)
    if isinstance(value, list | tuple | set):
        indexes: list[int] = []
        for item in value:
            indexes.extend(_device_indexes(item))
        return tuple(dict.fromkeys(indexes))
    return tuple(dict.fromkeys(int(part) for part in re.findall(r"\d+", str(value))))


def _process_pid(value: Any) -> int | None:
    """Accept complete positive integer PIDs, never rounded or guessed values."""
    if type(value) is int:
        pid = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]{1,10}", value.strip()):
        pid = int(value.strip())
    else:
        return None
    return pid if 0 < pid <= 2**31 - 1 else None


def _command_text(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 65536:
        return ""
    # The public endpoint supplies text, not NUL-delimited argv or terminal
    # escape sequences. Reject control data rather than inventing arguments.
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", value):
        return ""
    return value.strip()


def _command_tokens(text: str) -> list[str] | None:
    """Retain the source quotes/escaping, including quoted option-like values."""
    tokens, current = [], []
    quote = ""
    index = 0
    while index < len(text):
        character = text[index]
        if character == "\\" and quote != "'":
            if index + 1 == len(text):
                return None
            following = text[index + 1]
            if following == "\n":
                index += 2
                continue
            if following == "\r" and text[index + 2 : index + 3] == "\n":
                index += 3
                continue
            current.extend((character, following))
            index += 2
            continue
        if quote:
            current.append(character)
            if character == quote:
                quote = ""
        elif character in ("'", '"'):
            quote = character
            current.append(character)
        elif character.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(character)
        index += 1
    if quote:
        return None
    if current:
        tokens.append("".join(current))
    return tokens


def format_process_command(command: str) -> str:
    """Read-only formatting: one option/value group per line, no shell execution."""
    text = _command_text(command)
    if not text:
        return "网页未提供可用指令"
    tokens = _command_tokens(text)
    if tokens is None:
        return "参数引号不完整，以下按网页原文显示：\n" + text
    groups: list[list[str]] = [[]]
    options_started, trailing_shell = False, False
    for token in tokens:
        if token in (">", ">>", "<", "2>", "2>>", "2>&1", "&", "|", "&&", ";"):
            if not trailing_shell:
                groups.append([])
                trailing_shell = True
        elif not trailing_shell and (
            re.match(r"^--[A-Za-z0-9][A-Za-z0-9_.-]*(?:=|$)", token)
            or (options_started and re.match(r"^-[A-Za-z](?:$|[=A-Za-z])", token))
            or token == "--"
        ):
            groups.append([])
            options_started = True
            if token == "--":
                trailing_shell = True
        groups[-1].append(token)
    lines = [" ".join(group) for group in groups if group]
    return "\n".join(
        line if index == 0 else "    " + line for index, line in enumerate(lines)
    )


def parse_snapshot(
    payload: Any,
    *,
    server_id: str = DEFAULT_SERVER_ID,
    server_name: str = DEFAULT_SERVER_NAME,
) -> Snapshot:
    """Validate and normalize ``/api/server_detail`` JSON."""
    if not isinstance(payload, dict):
        raise MonitorError("接口返回的不是 JSON 对象")

    status_box = payload.get("status")
    rows = status_box.get("status") if isinstance(status_box, dict) else None
    if not isinstance(rows, list) or not rows:
        raise MonitorError("接口中没有 GPU 状态数据")

    process_counts: defaultdict[int, int] = defaultdict(int)
    user_process_counts: defaultdict[int, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    user_memory_gib: defaultdict[int, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    user_pids: defaultdict[int, defaultdict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    user_commands: defaultdict[int, defaultdict[str, dict[int, str | None]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    processes = payload.get("processes", [])
    if isinstance(processes, list):
        for process in processes:
            if not isinstance(process, dict):
                continue
            username = str(process.get("username") or "未知用户").strip() or "未知用户"
            process_memory = parse_capacity_gib(process.get("gpu_memory_human"))
            if process_memory is None:
                process_memory = parse_capacity_gib(process.get("gpu_memory"))
            pid = _process_pid(process.get("pid"))
            command = (
                _command_text(process.get("command"))
                if username.casefold() == HIGHLIGHT_USERNAME.casefold()
                else ""
            )
            for index in _device_indexes(process.get("device_idx")):
                process_counts[index] += 1
                user_process_counts[index][username] += 1
                if pid is not None:
                    user_pids[index][username].add(pid)
                    commands = user_commands[index][username]
                    if pid not in commands or commands[pid] == "":
                        commands[pid] = command
                    elif (
                        command
                        and commands[pid] is not None
                        and commands[pid] != command
                    ):
                        commands[pid] = (
                            None  # Conflicting duplicate records: do not guess.
                        )
                if process_memory is not None:
                    user_memory_gib[index][username] += max(0.0, process_memory)

    metrics: list[GpuMetric] = []
    seen: set[int] = set()
    for fallback_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        raw_index = _number(row.get("device_idx", row.get("index", fallback_index)))
        index = fallback_index if raw_index is None else int(raw_index)
        if index in seen:
            raise MonitorError(f"GPU {index} 在接口中重复出现")
        seen.add(index)

        memory_used, memory_total, memory_percent = _parse_memory_row(row)
        power_w, power_limit_w = _parse_power(row)
        usernames = sorted(
            user_process_counts[index],
            key=lambda username: (
                -user_memory_gib[index][username],
                username.casefold(),
            ),
        )
        user_usages = tuple(
            UserGpuUsage(
                username=username,
                memory_gib=user_memory_gib[index][username],
                process_count=user_process_counts[index][username],
                pids=tuple(sorted(user_pids[index][username])),
                commands=tuple(
                    ProcessCommand(pid, user_commands[index][username].get(pid) or "")
                    for pid in sorted(user_pids[index][username])
                ),
            )
            for username in usernames
        )
        metrics.append(
            GpuMetric(
                index=index,
                utilization=_clamp_percent(_number(row.get("gpu_utilization"))),
                memory_percent=memory_percent,
                memory_used_gib=memory_used,
                memory_total_gib=memory_total,
                temperature_c=_number(row.get("temperature")),
                power_w=power_w,
                power_limit_w=power_limit_w,
                process_count=process_counts[index],
                user_usages=user_usages,
            )
        )
    if not metrics:
        raise MonitorError("没有可用的 GPU 记录")
    metrics.sort(key=lambda gpu: gpu.index)

    raw_timestamp = (
        _number(status_box.get("timestamp")) if isinstance(status_box, dict) else None
    )
    source_timestamp_ms = int(raw_timestamp) if raw_timestamp is not None else None
    system = payload.get("system_status")
    system = system if isinstance(system, dict) else {}
    return Snapshot(
        server_id=server_id,
        server_name=server_name,
        gpus=tuple(metrics),
        received_at=time.time(),
        source_timestamp_ms=source_timestamp_ms,
        cpu_percent=_number(system.get("cpu_percent")),
        host_memory_percent=_number(system.get("memory_percent")),
        disk_percent=_number(system.get("disk_usage_percent")),
    )


def build_detail_url(api_base: str, server_id: str) -> str:
    base = api_base.rstrip("/")
    query = urllib.parse.urlencode({"server": server_id})
    return f"{base}/api/server_detail?{query}"


def _fetch_json(url: str, timeout: float) -> Any:
    """Read the public monitor directly, without unrelated desktop proxy settings."""
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "WN-A100-Desktop-Monitor/1.0",
        },
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise MonitorError(f"HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise MonitorError(f"无法连接监控服务: {reason}") from exc
    except TimeoutError as exc:
        raise MonitorError("连接监控服务超时") from exc

    if len(body) > MAX_RESPONSE_BYTES:
        raise MonitorError("接口响应过大，已拒绝解析")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MonitorError("接口返回了无效 JSON") from exc
    return payload


def fetch_snapshot(
    api_base: str,
    server_id: str,
    server_name: str,
    timeout: float,
) -> Snapshot:
    """Fetch one GPU sample using the standard-library HTTPS client."""
    payload = _fetch_json(build_detail_url(api_base, server_id), timeout)
    return parse_snapshot(payload, server_id=server_id, server_name=server_name)


def fetch_announcements(
    api_base: str,
    server_id: str,
    server_name: str,
    timeout: float,
) -> tuple[Announcement, ...]:
    payload = _fetch_json(f"{api_base.rstrip('/')}/api/notifications", timeout)
    return parse_announcements(payload, server_id=server_id, server_name=server_name)


def demo_snapshot(tick: int = 0) -> Snapshot:
    """Generate deterministic data for previewing the widget offline."""
    metrics: list[GpuMetric] = []
    for index in range(EXPECTED_GPU_COUNT):
        phase = tick / 3.0 + index * 0.7
        utilization = _clamp_percent(52 + 47 * math.sin(phase))
        memory_percent = _clamp_percent(18 + index * 9 + 5 * math.sin(phase / 2))
        total = 80.0
        used = total * memory_percent / 100
        if memory_percent < 20:
            user_usages: tuple[UserGpuUsage, ...] = ()
        elif index % 3 == 0:
            user_usages = (UserGpuUsage("yzh", used, 1, (90000 + index,)),)
        elif index == 4:
            user_usages = (
                UserGpuUsage("research", used * 0.62, 2),
                UserGpuUsage("guest", used * 0.38, 1),
            )
        else:
            user_usages = (UserGpuUsage("research", used, 1 + index % 2),)
        metrics.append(
            GpuMetric(
                index=index,
                utilization=utilization,
                memory_percent=memory_percent,
                memory_used_gib=used,
                memory_total_gib=total,
                temperature_c=36 + utilization * 0.28,
                power_w=65 + utilization * 3.1,
                power_limit_w=400,
                process_count=sum(usage.process_count for usage in user_usages),
                user_usages=user_usages,
            )
        )
    return Snapshot(
        server_id=DEFAULT_SERVER_ID,
        server_name=DEFAULT_SERVER_NAME,
        gpus=tuple(metrics),
        received_at=time.time(),
        source_timestamp_ms=int(time.time() * 1000),
        cpu_percent=17.2,
        host_memory_percent=23.8,
        disk_percent=76.1,
    )


def snapshot_as_dict(snapshot: Snapshot) -> dict[str, Any]:
    return {
        "server_id": snapshot.server_id,
        "server_name": snapshot.server_name,
        "received_at": (
            datetime.fromtimestamp(snapshot.received_at)
            .astimezone()
            .isoformat(timespec="seconds")
        ),
        "source_timestamp_ms": snapshot.source_timestamp_ms,
        "system": {
            "cpu_percent": snapshot.cpu_percent,
            "memory_percent": snapshot.host_memory_percent,
            "disk_percent": snapshot.disk_percent,
        },
        "gpus": [
            {
                "index": gpu.index,
                "utilization": gpu.utilization,
                "memory_percent": gpu.memory_percent,
                "memory_used_gib": gpu.memory_used_gib,
                "memory_total_gib": gpu.memory_total_gib,
                "temperature_c": gpu.temperature_c,
                "power_w": gpu.power_w,
                "process_count": gpu.process_count,
                "users": list(gpu.users),
                "user_usages": [
                    {
                        "username": usage.username,
                        "memory_gib": usage.memory_gib,
                        "process_count": usage.process_count,
                        **({"pids": list(usage.pids)} if usage.pids else {}),
                        **(
                            {
                                "process_commands": [
                                    {"pid": item.pid, "command": item.command}
                                    for item in usage.commands
                                ]
                            }
                            if any(item.command for item in usage.commands)
                            else {}
                        ),
                    }
                    for usage in gpu.user_usages
                ],
            }
            for gpu in snapshot.gpus
        ],
    }


def format_snapshot(snapshot: Snapshot) -> str:
    lines = [
        f"{snapshot.server_name}  {len(snapshot.gpus)} GPUs  "
        f"{datetime.fromtimestamp(snapshot.received_at).strftime('%Y-%m-%d %H:%M:%S')}",
        "GPU   利用率    显存占用                  温度    功耗    进程",
    ]
    for gpu in snapshot.gpus:
        temperature = (
            "--" if gpu.temperature_c is None else f"{gpu.temperature_c:.0f}°C"
        )
        power = "--" if gpu.power_w is None else f"{gpu.power_w:.0f}W"
        users = ", ".join(
            f"{usage.username} {usage.memory_gib:.1f}GiB" for usage in gpu.user_usages
        )
        process_summary = f"{gpu.process_count:>2}" + (f" [{users}]" if users else "")
        lines.append(
            f"{gpu.index:>3}   {gpu.utilization:>5.0f}%    "
            f"{gpu.memory_used_gib:>5.1f}/{gpu.memory_total_gib:<5.1f} GiB "
            f"({gpu.memory_percent:>5.1f}%)   {temperature:>4}   {power:>4}   {process_summary}"
        )
    return "\n".join(lines)


class Poller(threading.Thread):
    """Perform network I/O away from Tk's event loop."""

    def __init__(
        self,
        events: queue.Queue[PollEvent],
        *,
        api_base: str,
        server_id: str,
        server_name: str,
        interval: float,
        timeout: float,
        demo: bool,
    ) -> None:
        super().__init__(name="wn-a100-poller", daemon=True)
        self.events = events
        self.api_base = api_base
        self.server_id = server_id
        self.server_name = server_name
        self.interval = interval
        self.timeout = timeout
        self.demo = demo
        self.stop_event = threading.Event()
        self.refresh_event = threading.Event()
        self.failures = 0
        self.tick = 0

    def stop(self) -> None:
        self.stop_event.set()
        self.refresh_event.set()

    def refresh(self) -> None:
        self.refresh_event.set()

    def _publish(self, event: PollEvent) -> None:
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            self.events.put_nowait(event)

    def _wait(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not self.stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            if self.refresh_event.wait(min(0.25, remaining)):
                self.refresh_event.clear()
                return

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                if self.demo:
                    snapshot = demo_snapshot(self.tick)
                    self.tick += 1
                else:
                    snapshot = fetch_snapshot(
                        self.api_base,
                        self.server_id,
                        self.server_name,
                        self.timeout,
                    )
                self.failures = 0
                self._publish(PollEvent(snapshot=snapshot))
                delay = self.interval
            except (
                Exception
            ) as exc:  # Keep the desktop monitor alive after transient failures.
                self.failures += 1
                self._publish(PollEvent(error=str(exc)))
                delay = min(30.0, max(self.interval, 2.0 ** min(self.failures, 4)))
            self._wait(delay)


class AnnouncementPoller(Poller):
    """Refresh notices independently, so slow notices never stall the GPU poller."""

    def run(self) -> None:
        self.name = "wn-a100-announcements"
        while not self.stop_event.is_set():
            try:
                if self.demo:
                    announcements = (
                        Announcement(
                            "demo",
                            "gpu_usage",
                            "演示公告：research 申请 WN-A100 的 3、4 号 GPU，用于模型训练。",
                            gpu_indexes=(3, 4),
                        ),
                    )
                else:
                    announcements = fetch_announcements(
                        self.api_base, self.server_id, self.server_name, self.timeout
                    )
                self._publish(PollEvent(announcements=announcements))
            except Exception as exc:
                self._publish(PollEvent(notice_error=str(exc)))
            self._wait(self.interval)


def _level_color(value: float, *, memory: bool = False) -> str:
    if value >= 92:
        return COLORS["red"]
    if value >= (82 if memory else 88):
        return COLORS["amber"]
    return COLORS["green"] if memory else COLORS["blue"]


def _compact_memory(gib: float) -> str:
    if gib >= 1:
        value = f"{gib:.1f}".rstrip("0").rstrip(".")
        return f"{value}G"
    return f"{gib * 1024:.0f}M"


def _config_path() -> Path:
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "wn-a100-monitor" / "config.json"


def _load_config() -> dict[str, Any]:
    try:
        data = json.loads(_config_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_config(data: dict[str, Any]) -> None:
    path = _config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的数值")
    return parsed


def _opacity(value: str) -> float:
    parsed = float(value)
    if not 0.55 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("透明度必须在 0.55 到 1.0 之间")
    return parsed


def _desktop_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def autostart_path() -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "autostart" / "wn-a100-monitor.desktop"


def project_uv_command() -> list[str]:
    """Build a direct uv command with a fixed project and isolated desktop environment."""
    project_root = next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "pyproject.toml").is_file()
            and (parent / "src" / "lerobot" / "yzh" / "wn_a100").is_dir()
        ),
        None,
    )
    if project_root is None:
        raise SystemExit("无法找到项目 pyproject.toml；请在完整项目中安装启动项。")
    project_python = project_root / ".venv" / "bin" / "python"
    if not os.access(project_python, os.X_OK):
        raise SystemExit("项目环境不存在；请先执行 uv sync --locked --inexact --extra desktop。")
    uv_path = Path.home() / ".local" / "bin" / "uv"
    uv_executable = str(uv_path) if os.access(uv_path, os.X_OK) else shutil.which("uv")
    if not uv_executable:
        raise SystemExit("未找到 uv；请先安装独立 uv 并加入 PATH。")
    cleared_variables = (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONUSERBASE",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "VIRTUAL_ENV",
        "VIRTUAL_ENV_PROMPT",
        "CONDA_PREFIX",
        "CONDA_PREFIX_1",
        "CONDA_SHLVL",
        "CONDA_DEFAULT_ENV",
        "CONDA_EXE",
        "CONDA_PYTHON_EXE",
        "CONDA_PROMPT_MODIFIER",
        "_CE_CONDA",
        "_CE_M",
        "UV_PROJECT",
        "UV_WORKING_DIR",
        "UV_PROJECT_ENVIRONMENT",
        "UV_ENV_FILE",
        "UV_PYTHON",
        "UV_PYTHON_PREFERENCE",
        "UV_NO_MANAGED_PYTHON",
        "UV_ISOLATED",
        "UV_NO_PROJECT",
        "UV_CONFIG_FILE",
        "UV_NO_CONFIG",
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "QML_IMPORT_PATH",
        "QML2_IMPORT_PATH",
    )
    command = ["/usr/bin/env"]
    for variable in cleared_variables:
        command.extend(("-u", variable))
    path = ":".join(
        (
            str(project_python.parent), str(Path.home() / ".local" / "bin"),
            "/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin",
        )
    )
    command.extend(
        (
            f"PATH={path}", "PYTHONNOUSERSITE=1", f"UV_PROJECT_ENVIRONMENT={project_root / '.venv'}",
            str(Path(uv_executable).absolute()), "run", "--project", str(project_root),
            "--python", str(project_python), "--no-sync", "--no-env-file", "--",
            "python", "-m", "lerobot.yzh.wn_a100",
        )
    )
    return command


def install_autostart(args: argparse.Namespace) -> Path:
    """Create a desktop entry invoking uv directly in the existing project environment."""
    command = project_uv_command()
    destination = autostart_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    command.extend(
        (
            "--api-base",
            args.api_base,
            "--server-id",
            args.server_id,
            "--server-name",
            args.server_name,
            "--interval",
            str(args.interval),
        )
    )
    if args.no_topmost:
        command.append("--no-topmost")
    if args.no_training:
        command.append("--no-training")
    command.extend(("--training-config", args.training_config))
    content = "\n".join(
        (
            "[Desktop Entry]",
            "Type=Application",
            "Version=1.0",
            "Name=WN-A100 GPU Monitor",
            "Comment=Monitor WN-A100 GPU memory and utilization",
            "Exec=" + " ".join(_desktop_quote(part) for part in command),
            "Terminal=false",
            "X-GNOME-Autostart-enabled=true",
            "StartupNotify=false",
            "",
        )
    )
    destination.write_text(content, encoding="utf-8")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WN-A100 GPU 桌面实时监控组件")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="监控站根地址")
    parser.add_argument("--server-id", default=DEFAULT_SERVER_ID, help="服务器内部 ID")
    parser.add_argument(
        "--server-name", default=DEFAULT_SERVER_NAME, help="窗口显示名称"
    )
    parser.add_argument(
        "--interval",
        type=_positive_float,
        default=DEFAULT_INTERVAL,
        help="刷新间隔（秒）",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_TIMEOUT,
        help="网络超时（秒）",
    )
    parser.add_argument(
        "--opacity", type=_opacity, default=0.97, help="窗口透明度（0.55-1.0）"
    )
    parser.add_argument("--no-topmost", action="store_true", help="启动时不置顶")
    parser.add_argument(
        "--windowed", action="store_true", help="使用系统标题栏，便于调试"
    )
    parser.add_argument("--demo", action="store_true", help="使用模拟数据预览界面")
    parser.add_argument("--once", action="store_true", help="只请求一次并在终端输出")
    parser.add_argument(
        "--json",
        action="store_true",
        help="与 --once、--training-once 或 --disk-once 一起输出 JSON",
    )
    parser.add_argument(
        "--training-config",
        default=str(remote.DEFAULT_CONFIG),
        help="远程 SSH 连接与训练日志的配置文件",
    )
    parser.add_argument(
        "--no-training", action="store_true", help="关闭 SSH 训练进度读取"
    )
    parser.add_argument(
        "--training-once", action="store_true", help="只读一次训练日志，在终端验证进度"
    )
    parser.add_argument(
        "--no-disk", action="store_true", help="关闭 SSH 磁盘用量统计及轮播页"
    )
    parser.add_argument(
        "--disk-interval",
        type=_positive_float,
        default=900,
        help="du 统计间隔（秒，默认 15 分钟）",
    )
    parser.add_argument(
        "--disk-timeout",
        type=_positive_float,
        default=195,
        help="磁盘 SSH 统计超时（秒）",
    )
    parser.add_argument(
        "--disk-once",
        action="store_true",
        help="只运行一次远端 du -sh，在终端验证磁盘占用",
    )
    parser.add_argument(
        "--install-autostart", action="store_true", help="安装 GNOME 登录自启动项"
    )
    parser.add_argument(
        "--uninstall-autostart", action="store_true", help="移除 GNOME 登录自启动项"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.install_autostart and args.uninstall_autostart:
        raise SystemExit("--install-autostart 和 --uninstall-autostart 不能同时使用")
    if args.install_autostart:
        path = install_autostart(args)
        print(f"已安装开机自启动: {path}")
        return 0
    if args.uninstall_autostart:
        path = autostart_path()
        path.unlink(missing_ok=True)
        print(f"已移除开机自启动: {path}")
        return 0
    if args.disk_once:
        if args.training_once:
            raise SystemExit("--disk-once 和 --training-once 请分开使用")
        if args.server_id != DEFAULT_SERVER_ID:
            raise SystemExit("磁盘统计仅适用于 WN-A100")
        try:
            config = remote.DiskConfig(
                remote.load_config(args.training_config),
                args.disk_interval,
                args.disk_timeout,
            )
            sample = (
                remote.parse_disk_sample(
                    {"user": "yzh", "home": "/home/yzh", "output": "2.3T\t/home/yzh\n"}
                )
                if args.demo
                else remote.fetch_disk(config)
            )
        except remote.RemoteError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if args.json:
            print(
                json.dumps(
                    {
                        "user": "yzh",
                        "home": sample.home,
                        "du_command": remote.DU_COMMAND,
                        "used": sample.size_text,
                        "used_bytes_approx": sample.used_bytes,
                        "quota": "3T",
                        "quota_bytes": remote.QUOTA_BYTES,
                        "over_quota": sample.over_quota,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(remote.disk_text(sample, config))
        return 0
    if args.training_once:
        try:
            config = remote.load_config(args.training_config)
            samples = (
                remote.demo_samples(config)
                if args.demo
                else remote.fetch_training(config)
            )
        except remote.RemoteError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "label": sample.log.label,
                            "path": sample.log.path,
                            "gpu_indexes": list(sample.effective_gpu_indexes),
                            "step": sample.progress.step if sample.progress else None,
                            "total": sample.progress.total if sample.progress else None,
                            "percent": (
                                sample.progress.percent if sample.progress else None
                            ),
                            "eta_seconds": (
                                sample.progress.eta_seconds if sample.progress else None
                            ),
                            "status": sample.status(config.stale_after),
                            "error": sample.error,
                            "processes": [
                                {
                                    "pid": process.pid,
                                    "name": process.name,
                                    "command": process.command,
                                    "gpu_indexes": list(process.gpu_indexes),
                                    "source": "SSH /proc",
                                }
                                for process in sample.processes
                            ],
                        }
                        for sample in samples
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(
                "\n\n".join(remote.training_text(sample, config) for sample in samples)
            )
        return 1 if any(sample.error for sample in samples) else 0
    if args.once:
        try:
            snapshot = (
                demo_snapshot()
                if args.demo
                else fetch_snapshot(
                    args.api_base, args.server_id, args.server_name, args.timeout
                )
            )
        except MonitorError as exc:
            print(f"获取失败: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(snapshot_as_dict(snapshot), ensure_ascii=False, indent=2))
        else:
            print(format_snapshot(snapshot))
        return 0

    try:
        from .desktop import run_desktop
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("PyQt5"):
            print(
                "桌面界面需要 PyQt5。请在项目根目录执行：\n"
                "  uv sync --locked --inexact --extra desktop\n"
                "同步时同时保留本机其他所需 extra，再通过 uv run --no-sync 运行对应命令。\n"
                f"当前解释器：{sys.executable}",
                file=sys.stderr,
            )
            return 2
        raise
    return run_desktop(args)


if __name__ == "__main__":
    raise SystemExit(main())
