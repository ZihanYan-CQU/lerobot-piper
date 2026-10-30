#!/usr/bin/env python3
"""Bind Piper master/follower CAN ports using the bundled can_activate.sh."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SYS_NET = Path("/sys/class/net")
ACTIVATE_SCRIPT = Path(__file__).resolve().with_name("can_activate.sh")
USB_PATTERN = r"[0-9]+-[0-9]+(?:\.[0-9]+)*:[0-9]+\.[0-9]+"
DEFAULT_MASTER = "1-1.2:1.0"
DEFAULT_FOLLOWER = "1-1.3:1.0"


@dataclass(frozen=True)
class CanPort:
    """The USB adapter identity and its current interface name/state."""

    name: str
    usb: str
    ifindex: int
    serial: str
    up: bool
    bitrate: int | None = None


def run(*args: str) -> str:
    """Execute one checked command without shell interpolation."""
    try:
        result = subprocess.run(list(args), capture_output=True, text=True, check=True, timeout=30)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError((exc.stderr or exc.stdout or str(exc)).strip()) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"命令执行超时：{' '.join(args[:3])}") from exc
    return result.stdout


def link_details() -> dict[str, dict]:
    """Read the kernel's configured bitrates and administrative link flags."""
    return {link["ifname"]: link for link in json.loads(run("ip", "-details", "-json", "link", "show", "type", "can"))}


def discover_ports(*, strict: bool = False) -> list[CanPort]:
    """List USB CAN adapters; never include Ethernet, virtual CAN, or unrelated devices."""
    try:
        details = link_details()
    except (OSError, RuntimeError):
        if strict:
            raise
        details = {}  # Sysfs still identifies ports when netlink access is unavailable.
    ports = []
    for path in sorted(SYS_NET.iterdir()):
        if (path / "type").read_text().strip() != "280":
            continue
        device = (path / "device").resolve()
        if not re.fullmatch(USB_PATTERN, device.name):
            continue
        serial_file = device.parent / "serial"
        serial = serial_file.read_text().strip() if serial_file.exists() else ""
        info = details.get(path.name, {})
        bitrate = info.get("linkinfo", {}).get("info_data", {}).get("bittiming", {}).get("bitrate")
        ports.append(CanPort(
            name=path.name,
            usb=device.name,
            ifindex=int((path / "ifindex").read_text()),
            serial=serial,
            up=bool(int((path / "flags").read_text(), 16) & 1),
            bitrate=bitrate,
        ))
    return ports


def identity_token(port: CanPort) -> str:
    """Detect unplugging/replacement while allowing a checked interface rename."""
    return hashlib.sha256(json.dumps([port.usb, port.ifindex, port.serial]).encode()).hexdigest()


def validate_bindings(ports: list[CanPort], master: str, follower: str, bitrate: int) -> list[tuple[CanPort, str]]:
    """Resolve both roles and validate the entire request before changing any interface."""
    if not 1 <= bitrate <= 1_000_000:
        raise ValueError("CAN 波特率必须在 1 到 1000000 之间。")
    if master == follower:
        raise ValueError("主臂和从臂必须选择不同的 USB CAN 端口。")
    bindings = []
    for usb, name in ((master, "can_master"), (follower, "can_follower")):
        if not re.fullmatch(USB_PATTERN, usb):
            raise ValueError(f"USB 端口格式无效：{usb}")
        matches = [port for port in ports if port.usb == usb]
        if len(matches) != 1:
            raise ValueError(f"无法唯一找到 USB CAN 端口 {usb}，请连接设备后刷新。")
        bindings.append((matches[0], name))
    return bindings


def validate_expected(bindings: list[tuple[CanPort, str]], expected: list[str]) -> None:
    """Reject stale GUI selections before activating either arm port."""
    tokens = {}
    for item in expected:
        usb, separator, token = item.partition("=")
        if not separator or usb in tokens or not re.fullmatch(r"[0-9a-f]{64}", token):
            raise ValueError("CAN 端口身份参数无效，请刷新面板。")
        tokens[usb] = token
    if set(tokens) != {port.usb for port, _ in bindings}:
        raise ValueError("CAN 端口身份参数不完整，请刷新面板。")
    for port, _ in bindings:
        if tokens[port.usb] != identity_token(port):
            raise ValueError(f"USB 端口 {port.usb} 的设备已变化，请刷新面板后重新选择。")


def find_current(original: CanPort) -> CanPort:
    """Find the same physical adapter even after its interface has been renamed."""
    for port in discover_ports(strict=True):
        if port.usb == original.usb and identity_token(port) == identity_token(original):
            return port
    raise RuntimeError(f"USB CAN 设备 {original.usb} 已断开或被替换。")


def temporary_name(index: int) -> str:
    """Allocate a short unused interface name for role swaps or recovery."""
    for attempt in range(100):
        name = f"cb{os.getpid():x}{index:x}{attempt:x}"[:15]
        if not (SYS_NET / name).exists():
            return name
    raise RuntimeError("无法分配临时 CAN 接口名称。")


def restore_ports(originals: list[CanPort]) -> str:
    """Attempt to restore names, configured bitrates, and administrative link states."""
    errors, staged = [], []
    for index, original in enumerate(originals):
        try:
            current = find_current(original)
            run("ip", "link", "set", "dev", current.name, "down")
            temporary = temporary_name(index)
            run("ip", "link", "set", "dev", current.name, "name", temporary)
            staged.append((original, temporary))
        except (OSError, RuntimeError) as exc:
            errors.append(f"{original.usb}：{exc}")
    for original, temporary in staged:
        try:
            run("ip", "link", "set", "dev", temporary, "name", original.name)
            if original.bitrate is not None:
                run("ip", "link", "set", "dev", original.name, "type", "can", "bitrate", str(original.bitrate))
            if original.up:
                run("ip", "link", "set", "dev", original.name, "up")
        except (OSError, RuntimeError) as exc:
            errors.append(f"{original.usb}：{exc}")
    if errors:
        return "部分接口恢复失败，请刷新检查：\n" + "\n".join(errors)
    return "已恢复原接口名和启停状态；原来未设置波特率的接口可能保留新波特率，但保持关闭。"


def apply_bindings(bindings: list[tuple[CanPort, str]], bitrate: int) -> None:
    """Call can_activate.sh for both roles and verify the actual kernel state."""
    if os.geteuid() != 0:
        raise PermissionError("激活 CAN 接口需要管理员权限，请使用面板的系统授权或 sudo。")
    if not ACTIVATE_SCRIPT.is_file():
        raise ValueError(f"找不到现有激活脚本：{ACTIVATE_SCRIPT}")
    for executable in ("bash", "ip", "ethtool", "sudo", "dpkg-query"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"缺少系统命令 {executable}，请安装后重试。")
    for package in ("ethtool", "can-utils"):
        status = run("dpkg-query", "-W", "-f=${Status}", package).strip()
        if status != "install ok installed":
            raise RuntimeError(f"请先安装系统软件包 {package}。")

    selected_names = {port.name for port, _ in bindings}
    for _, target in bindings:
        if (SYS_NET / target).exists() and target not in selected_names:
            raise ValueError(f"接口名称 {target} 被未选择的设备占用，请先处理该名称冲突。")

    # Recheck every identity before the first modification, including after an authorization delay.
    for port, _ in bindings:
        find_current(port)
    changed = []
    try:
        # Free both final names before invoking the existing script, allowing master/follower swaps.
        for index, (port, target) in enumerate(bindings):
            if port.name != target:
                current = find_current(port)
                changed.append(port)
                run("ip", "link", "set", "dev", current.name, "down")
                run("ip", "link", "set", "dev", current.name, "name", temporary_name(index))
        for original, target in bindings:
            current = find_current(original)
            if current.name == target and current.up and current.bitrate == bitrate:
                continue
            if original not in changed:
                changed.append(original)
            run("bash", str(ACTIVATE_SCRIPT), target, str(bitrate), original.usb)
            current = find_current(original)
            if current.name != target or not current.up or current.bitrate != bitrate:
                raise RuntimeError(f"{original.usb} 激活后的名称、波特率或启用状态与请求不符。")
        for original, target in bindings:
            current = find_current(original)
            if current.name != target or not current.up or current.bitrate != bitrate:
                raise RuntimeError(f"{target} 最终验证失败。")
            print(f"已验证：{original.usb} → {target}，波特率 {bitrate}，状态 UP")
    except (OSError, RuntimeError) as exc:
        recovery = restore_ports(changed) if changed else "尚未修改接口。"
        raise RuntimeError(f"CAN 绑定失败：{exc}\n{recovery}") from exc


def main(argv: list[str] | None = None) -> int:
    """Expose the same two bindings to the GUI and a standalone command line."""
    parser = argparse.ArgumentParser(description="绑定并激活 Piper 主臂/从臂 USB CAN 接口。")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--list", action="store_true", help="只列出 USB CAN 端口")
    operation.add_argument("--dry-run", action="store_true", help="只预览对应的 can_activate.sh 命令")
    parser.add_argument("--master", default=DEFAULT_MASTER)
    parser.add_argument("--follower", default=DEFAULT_FOLLOWER)
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--expect-port", action="append", default=[], help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        ports = discover_ports(strict=not (args.list or args.dry_run))
        if args.list:
            for port in ports:
                print(f"{port.usb}  {port.name}  {'UP' if port.up else 'DOWN'}  波特率={port.bitrate or '未设置/未读取'}")
            return 0
        bindings = validate_bindings(ports, args.master, args.follower, args.bitrate)
        if args.expect_port:
            validate_expected(bindings, args.expect_port)
        if args.dry_run:
            for port, name in bindings:
                print(f'bash {ACTIVATE_SCRIPT} {name} {args.bitrate} "{port.usb}"')
            return 0
        apply_bindings(bindings, args.bitrate)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
