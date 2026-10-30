#!/usr/bin/env python3
"""Bind Linux camera nodes to persistent /dev/video_* aliases using udev.

This standalone helper needs only Python 3.10+ and udevadm, not LeRobot dependencies.
Run with --list to inspect cameras or --dry-run to preview the default wrist/top mapping.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

SYS_VIDEO = Path("/sys/class/video4linux")
DEV_DIR = Path("/dev")
RULES_FILE = Path("/etc/udev/rules.d/99-lerobot-cameras.rules")
RULES_DIRS = (
    Path("/usr/lib/udev/rules.d"),
    Path("/usr/local/lib/udev/rules.d"),
    Path("/run/udev/rules.d"),
    Path("/etc/udev/rules.d"),
)
MARKER = "# Managed by camera_aliases.py. Re-running replaces all mappings in this file.\n"
DEFAULT_MAPS = ["/dev/video4=/dev/video_wrist", "/dev/video10=/dev/video_top"]
# linux/videodev2.h: _IOWR('V', 2, struct v4l2_fmtdesc), whose size is 64 bytes.
VIDIOC_ENUM_FMT = 0xC0405602
COLOR_FORMATS = frozenset(
    {"RGB3", "BGR3", "RGB4", "BGR4", "XR24", "AR24", "XB24", "AB24", "RGBP", "RGBR",
     "YUYV", "UYVY", "YVYU", "VYUY", "NV12", "NV21", "YU12", "YV12", "MJPG", "JPEG"}
)


@dataclass(frozen=True)
class Camera:
    """A video node and its udev identity, including its stream index."""

    node: str
    name: str
    index: str
    properties: dict[str, str]
    interface: str = ""

    @property
    def capture(self) -> bool:
        """Return whether the node advertises video capture rather than metadata only."""
        return "capture" in self.properties.get("ID_V4L_CAPABILITIES", "").split(":")


def udevadm(*args: str) -> str:
    """Run udevadm without a shell and report actionable errors."""
    executable = shutil.which("udevadm")
    if executable is None:
        raise RuntimeError("找不到 udevadm，请先安装系统的 udev 软件包。")
    try:
        result = subprocess.run([executable, *args], capture_output=True, text=True, check=True, timeout=20)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "无详细输出").strip()
        raise RuntimeError(f"udevadm {' '.join(args)} 失败：{detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"udevadm {' '.join(args)} 超时。") from exc
    return result.stdout


def discover_cameras() -> list[Camera]:
    """Read sysfs so listing and previews also work without access to /dev nodes."""
    paths = sorted(
        (p for p in SYS_VIDEO.glob("video*") if re.fullmatch(r"video[0-9]+", p.name)),
        key=lambda p: int(p.name[5:]),
    )
    cameras = []
    for path in paths:
        output = udevadm("info", "--query=property", f"--path={path}")
        properties = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
        interface_file = path / "device" / "interface"
        interface = interface_file.read_text().strip() if interface_file.exists() else ""
        cameras.append(
            Camera(
                node=path.name,
                name=(path / "name").read_text().strip(),
                index=(path / "index").read_text().strip(),
                properties=properties,
                interface=interface,
            )
        )
    return cameras


def capture_formats(node: str) -> set[str]:
    """Enumerate capture formats without starting a stream or changing camera settings."""
    formats = set()
    fd = os.open(DEV_DIR / node, os.O_RDONLY | os.O_NONBLOCK)
    try:
        for buffer_type in (1, 9):  # VIDEO_CAPTURE and VIDEO_CAPTURE_MPLANE
            index = 0
            while True:
                buffer = bytearray(64)
                struct.pack_into("=II", buffer, 0, index, buffer_type)
                try:
                    fcntl.ioctl(fd, VIDIOC_ENUM_FMT, buffer, True)
                except OSError as exc:
                    if exc.errno == errno.EINVAL:
                        break
                    raise
                fourcc = struct.unpack_from("=I", buffer, 44)[0]
                formats.add(fourcc.to_bytes(4, "little").decode("ascii", errors="replace"))
                index += 1
    finally:
        os.close(fd)
    return formats


def filter_rgb_cameras(cameras: list[Camera]) -> tuple[list[Camera], list[str]]:
    """Keep color sensors and exclude metadata, depth, infrared, and unknown nodes.

RealSense USB interface labels distinguish RGB from Depth even though every node's
product name contains "Depth Camera". Neutral webcam labels fall back to V4L2 formats.
"""
    rgb, unavailable = [], []
    for camera in cameras:
        if not camera.capture:
            continue
        label = camera.interface or camera.name
        if re.search(r"\b(?:rgb|colou?r)\b", label, re.IGNORECASE):
            rgb.append(camera)
            continue
        if re.search(r"\b(?:depth|infrared|ir|monochrome)\b", label, re.IGNORECASE):
            continue
        try:
            if capture_formats(camera.node) & COLOR_FORMATS:
                rgb.append(camera)
        except OSError as exc:
            unavailable.append(f"/dev/{camera.node}：无法确认彩色格式（{exc.strerror or exc}）")
    return rgb, unavailable


def identity_token(camera: Camera) -> str:
    """Identify a panel selection so a reused videoN cannot silently bind another camera."""
    identity = [camera.index, camera.interface]
    identity.extend(
        camera.properties.get(key, "")
        for key in ("ID_SERIAL", "ID_SERIAL_SHORT", "ID_USB_INTERFACE_NUM", "ID_PATH", "DEVPATH")
    )
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def validate_selections(cameras: list[Camera], mappings: list[tuple[str, str]], expected: list[str]) -> None:
    """Recheck RGB eligibility and the GUI snapshot before a privileged installation."""
    by_node = {camera.node: camera for camera in cameras}
    expected_tokens = {}
    for item in expected:
        node, separator, token = item.partition("=")
        if not separator or node in expected_tokens or not re.fullmatch(r"[0-9a-f]{64}", token):
            raise ValueError("相机身份校验参数无效，请关闭面板后重新打开。")
        expected_tokens[node] = token
    selected_nodes = {node for node, _ in mappings}
    if set(expected_tokens) != selected_nodes:
        raise ValueError("相机身份校验不完整，请刷新面板后重新选择。")
    selected = []
    for node in selected_nodes:
        device = by_node.get(node)
        if device is None or identity_token(device) != expected_tokens[node]:
            raise ValueError(f"/dev/{node} 的设备已变化或已断开，请刷新面板后重新选择。")
        selected.append(device)
    rgb, _ = filter_rgb_cameras(selected)
    if {device.node for device in rgb} != selected_nodes:
        raise ValueError("所选设备不再能确认为 RGB 相机，请刷新面板后重新选择。")


def parse_mapping(value: str) -> tuple[str, str]:
    """Accept DEVICE=ALIAS with optional /dev/ prefixes or a numeric device index."""
    source, separator, alias = value.partition("=")
    if not separator:
        raise ValueError(f"映射格式应为 /dev/video4=/dev/video_wrist，收到：{value}")
    source, alias = source.removeprefix("/dev/"), alias.removeprefix("/dev/")
    if source.isascii() and source.isdecimal():
        source = f"video{source}"
    if not re.fullmatch(r"video[0-9]+", source):
        raise ValueError(f"相机设备必须为 /dev/videoN、videoN 或数字，收到：{source}")
    if not re.fullmatch(r"video_[A-Za-z0-9][A-Za-z0-9_-]*", alias):
        raise ValueError(f"别名必须以 video_ 开头，后接英文字母、数字、下划线或连字符：{alias}")
    return source, alias


def select_identity(camera: Camera, cameras: list[Camera], mode: str) -> tuple[dict[str, str], str]:
    """Choose a unique serial/interface/index identity or a physical connection path."""
    if not camera.capture:
        raise ValueError(f"/dev/{camera.node} 不是视频采集节点，可能是元数据节点；请用 --list 查看。")
    if not re.fullmatch(r"[0-9]+", camera.index):
        raise ValueError(f"/dev/{camera.node} 缺少有效的视频接口索引。")

    candidates = []
    props = camera.properties
    if mode != "path" and props.get("ID_SERIAL_SHORT") and props.get("ID_SERIAL"):
        serial = {"ID_SERIAL": props["ID_SERIAL"]}
        if props.get("ID_USB_INTERFACE_NUM"):
            serial["ID_USB_INTERFACE_NUM"] = props["ID_USB_INTERFACE_NUM"]
        candidates.append((serial, "序列号"))
    if mode != "serial" and props.get("ID_PATH"):
        candidates.append(({"ID_PATH": props["ID_PATH"]}, "连接位置（请保持 USB 插口不变）"))

    for identity, label in candidates:
        matches = [
            other
            for other in cameras
            if other.capture
            and other.index == camera.index
            and all(other.properties.get(key) == value for key, value in identity.items())
        ]
        if len(matches) == 1 and matches[0].node == camera.node:
            return identity, label
    raise ValueError(
        f"无法唯一识别 /dev/{camera.node}（识别模式：{mode}）。"
        "请用 --list 检查序列号和接口；相同或缺失序列号的相机可尝试 --match path。"
    )


def make_rules(cameras: list[Camera], mappings: list[tuple[str, str]], mode: str) -> str:
    """Build rules that survive videoN renumbering and distinguish RealSense streams."""
    by_node = {camera.node: camera for camera in cameras}
    aliases = [alias for _, alias in mappings]
    if len(aliases) != len(set(aliases)):
        raise ValueError("同一个别名只能出现一次，请检查 --map 参数。")
    lines = [MARKER.rstrip(), "# Generated from the currently connected cameras; do not match videoN numbers."]
    for node, alias in mappings:
        if node not in by_node:
            raise ValueError(f"未找到 /dev/{node}，请连接相机并运行 --list 查看当前编号。")
        camera = by_node[node]
        identity, label = select_identity(camera, cameras, mode)
        fields = ['ACTION=="add|change"', 'SUBSYSTEM=="video4linux"', 'KERNEL=="video[0-9]*"']
        fields.append('ENV{ID_V4L_CAPABILITIES}=="*:capture:*"')
        for key, value in identity.items():
            if not re.fullmatch(r"[A-Za-z0-9_.:/+@-]+", value):
                raise ValueError(f"{key} 含不能安全写入 udev 规则的字符；可尝试 --match path。")
            fields.append(f'ENV{{{key}}}=="{value}"')
        fields.extend([f'ATTR{{index}}=="{camera.index}"', f'SYMLINK+="{alias}"'])
        lines.extend([f"# /dev/{alias} <- /dev/{node}; {label}", ", ".join(fields)])
    return "\n".join(lines) + "\n"


def check_conflicts(mappings: list[tuple[str, str]]) -> None:
    """Reject occupied alias names and static alias assignments in other active rules."""
    aliases = {alias for _, alias in mappings}
    for alias in aliases:
        path = DEV_DIR / alias
        if path.exists() and not path.is_symlink():
            raise ValueError(f"{path} 已存在且不是软链接，不能用作相机别名。")
    # Same-name files in /etc override the lower-priority copies, including /dev/null masks.
    effective_rules = {}
    for directory in RULES_DIRS:
        effective_rules.update({path.name: path for path in directory.glob("*.rules")})
    for path in effective_rules.values():
        if path == RULES_FILE:
            continue
        for line in path.read_text().splitlines():
            if line.lstrip().startswith("#"):
                continue
            for assignment in re.findall(r'\bSYMLINK\s*(?:\+=|:=|=)\s*"([^"]*)"', line):
                overlap = aliases.intersection(assignment.split())
                if overlap:
                    raise ValueError(
                        f"{path} 已经定义了 {', '.join(sorted(overlap))}；"
                        "请先移除该文件中对应的旧映射，避免两个设备争用同一别名。"
                    )


def write_rules(content: str) -> None:
    """Back up this helper's previous rules and atomically install the new file."""
    if RULES_FILE.is_symlink():
        raise ValueError(f"{RULES_FILE} 是软链接，拒绝覆盖。")
    if RULES_FILE.exists():
        old_content = RULES_FILE.read_text()
        if not old_content.startswith(MARKER):
            raise ValueError(f"{RULES_FILE} 不是本程序生成的规则文件，拒绝覆盖。")
        if old_content == content:
            return
        backup = RULES_FILE.with_name(f"{RULES_FILE.name}.bak-{time.time_ns()}")
        shutil.copy2(RULES_FILE, backup)
        print(f"旧规则已备份：{backup}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=RULES_FILE.parent, prefix=".camera-aliases-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        os.replace(temporary, RULES_FILE)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def apply_rules(content: str, mappings: list[tuple[str, str]]) -> None:
    """Install, reload, trigger video events, then verify every requested symlink."""
    if os.geteuid() != 0:
        raise PermissionError("写入 /etc/udev/rules.d 需要管理员权限，请用 sudo 运行；预览可加 --dry-run。")
    for node, _ in mappings:
        if not (DEV_DIR / node).is_char_device():
            raise ValueError(f"{DEV_DIR / node} 不可见或不是字符设备，请在连接相机的主机上运行。")
    write_rules(content)
    print(f"规则已保存：{RULES_FILE}")
    try:
        udevadm("control", "--reload-rules")
        # Refresh all video nodes so an old camera releases an alias when it is reassigned.
        udevadm("trigger", "--action=change", "--subsystem-match=video4linux")
        udevadm("settle", "--timeout=10")
        for node, alias in mappings:
            target, link = DEV_DIR / node, DEV_DIR / alias
            if not link.is_symlink() or not link.exists() or not link.samefile(target):
                raise RuntimeError(f"{link} 未正确指向 {target}。请检查旧 udev 规则及相机连接。")
            print(f"已验证：{link} -> {target}")
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"规则已保存，但立即生效验证失败：{exc}\n"
            "请在主机上检查 udev 服务和冲突规则，解决后重新运行本程序。"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    """Provide a standalone Chinese command-line interface."""
    parser = argparse.ArgumentParser(
        description="给指定相机建立重启后仍有效的 /dev/video_* 别名。默认：video4→video_wrist，video10→video_top。",
        epilog="每次运行替换本程序规则文件中的全部映射；请一次传入所有需要保留的 --map。",
    )
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--list", action="store_true", help="列出相机、序列号、接口和采集能力，不修改系统")
    operation.add_argument("--list-rgb", action="store_true", help="只列出 RGB 彩色相机，不修改系统")
    operation.add_argument("--gui", action="store_true", help="打开 RGB 相机与机械臂 CAN 绑定面板")
    operation.add_argument("--dry-run", action="store_true", help="只预览生成的规则，不修改系统")
    parser.add_argument(
        "--map", action="append", dest="mappings", metavar="设备=别名", help="可重复指定；例：4=video_wrist"
    )
    parser.add_argument("--expect-camera", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument(
        "--match",
        choices=("auto", "serial", "path"),
        default="auto",
        help="auto 优先按序列号绑定，无法唯一识别时按连接位置绑定（默认：auto）",
    )
    args = parser.parse_args(argv)
    if (args.list or args.list_rgb or args.gui) and args.mappings:
        parser.error("--list、--list-rgb、--gui 不能同时指定 --map。")
    try:
        if args.gui:
            if __package__:
                from .camera_aliases_gui import run_gui
            else:
                from camera_aliases_gui import run_gui

            return run_gui()
        cameras = discover_cameras()
        if not cameras:
            raise ValueError("未检测到相机，请连接相机后重试。")
        if args.list or args.list_rgb:
            listed = cameras
            if args.list_rgb:
                listed, unavailable = filter_rgb_cameras(cameras)
                for message in unavailable:
                    print(message, file=sys.stderr)
                if not listed:
                    raise ValueError("未检测到可确认的 RGB 相机。")
            for camera in listed:
                props = camera.properties
                kind = "RGB 彩色" if args.list_rgb else "视频采集" if camera.capture else "非采集/元数据"
                print(f"/dev/{camera.node}  [{kind}]  {camera.name}")
                print(
                    f"  序列号={props.get('ID_SERIAL_SHORT', '无')}  "
                    f"接口={props.get('ID_USB_INTERFACE_NUM', '无')}  索引={camera.index}"
                )
                print(f"  连接位置={props.get('ID_PATH', '无')}")
                if camera.interface:
                    print(f"  接口名称={camera.interface}")
            return 0
        mappings = [parse_mapping(value) for value in (args.mappings or DEFAULT_MAPS)]
        if args.expect_camera:
            validate_selections(cameras, mappings, args.expect_camera)
        content = make_rules(cameras, mappings, args.match)
        check_conflicts(mappings)
        if args.dry_run:
            print(f"仅预览，未修改系统。安装目标：{RULES_FILE}\n")
            print(content, end="")
        else:
            if "ENV{ID_PATH}" in content:
                print("提示：部分映射按连接位置绑定，这些相机请保持 USB 插口和连接拓扑不变。")
            apply_rules(content, mappings)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
