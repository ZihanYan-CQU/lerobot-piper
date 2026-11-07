#!/usr/bin/env python3
"""Discover, bind, and activate selected USB CAN ports with checked ip commands."""

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


def validate_bitrate(bitrate: int) -> None:
    """Reject unsupported rates before any interface is changed."""
    if not 1 <= bitrate <= 1_000_000:
        raise ValueError("CAN 波特率必须在 1 到 1000000 之间。")


def resolve_binding(ports: list[CanPort], usb: str, name: str) -> tuple[CanPort, str]:
    """Resolve a USB identity to exactly one connected CAN adapter."""
    if not re.fullmatch(USB_PATTERN, usb):
        raise ValueError(f"USB 端口格式无效：{usb}")
    matches = [port for port in ports if port.usb == usb]
    if len(matches) != 1:
        raise ValueError(f"无法唯一找到 USB CAN 端口 {usb}，请连接设备后刷新。")
    return matches[0], name


def validate_bindings(
    ports: list[CanPort], master: str | None, follower: str | None, bitrate: int
) -> list[tuple[CanPort, str]]:
    """Validate the selected roles; an omitted role is never configured."""
    validate_bitrate(bitrate)
    if not master and not follower:
        raise ValueError("请至少为一只机械臂选择 USB CAN 端口；另一只可保持“本次不配置”。")
    if master and master == follower:
        raise ValueError("主臂和从臂必须选择不同的 USB CAN 端口。")
    return [
        resolve_binding(ports, usb, name)
        for usb, name in ((master, "can_master"), (follower, "can_follower"))
        if usb
    ]


def validate_expected(bindings: list[tuple[CanPort, str]], expected: list[str]) -> None:
    """Reject stale GUI selections before activating the selected arm ports."""
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
    """Configure only selected adapters, verifying kernel state and recovering on failure."""
    if os.geteuid() != 0:
        raise PermissionError("激活 CAN 接口需要管理员权限，请使用面板的系统授权或 sudo。")
    if shutil.which("ip") is None:
        raise RuntimeError("缺少系统命令 ip，请安装 iproute2 后重试。")

    selected_names = {port.name for port, _ in bindings}
    for _, target in bindings:
        if (SYS_NET / target).exists() and target not in selected_names:
            raise ValueError(f"接口名称 {target} 被未选择的设备占用，请先处理该名称冲突。")

    # Recheck every identity before the first modification, including after an authorization delay.
    for port, _ in bindings:
        find_current(port)
    changed = []
    try:
        # Free the selected names first, allowing master/follower swaps.
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
            run("ip", "link", "set", "dev", current.name, "down")
            if current.name != target:
                run("ip", "link", "set", "dev", current.name, "name", target)
            run("ip", "link", "set", "dev", target, "type", "can", "bitrate", str(bitrate))
            run("ip", "link", "set", "dev", target, "up")
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
    """Support one or both arm roles, plus the legacy shell entry point."""
    parser = argparse.ArgumentParser(description="绑定并激活 Piper 主臂/从臂 USB CAN 接口。")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--list", action="store_true", help="只列出 USB CAN 端口")
    operation.add_argument("--dry-run", action="store_true", help="只预览所选端口的目标名称与波特率")
    parser.add_argument("--master", help=f"主臂 USB 端口；两个角色均不指定时默认为 {DEFAULT_MASTER}")
    parser.add_argument("--follower", help=f"从臂 USB 端口；两个角色均不指定时默认为 {DEFAULT_FOLLOWER}")
    parser.add_argument("--interface", help="单接口模式的目标名称，供 can_activate.sh 兼容入口使用")
    parser.add_argument("--usb", help="单接口模式的 USB 端口；仅连接一个适配器时可省略")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--expect-port", action="append", default=[], help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        ports = discover_ports(strict=not (args.list or args.dry_run))
        if args.list:
            for port in ports:
                print(f"{port.usb}  {port.name}  {'UP' if port.up else 'DOWN'}  波特率={port.bitrate or '未设置/未读取'}")
            return 0
        if args.interface is not None:
            if args.master is not None or args.follower is not None:
                raise ValueError("--interface 不能与 --master 或 --follower 同时使用。")
            if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,14}", args.interface):
                raise ValueError("接口名称须为 1 到 15 位字母、数字、下划线、点或连字符，且不能以点或连字符开头。")
            validate_bitrate(args.bitrate)
            usb = args.usb or (ports[0].usb if len(ports) == 1 else None)
            if usb is None:
                raise ValueError("存在多个或没有 USB CAN 端口，请明确指定 --usb。")
            bindings = [resolve_binding(ports, usb, args.interface)]
        else:
            if args.usb is not None:
                raise ValueError("--usb 需与 --interface 一起使用。")
            if args.master is None and args.follower is None:
                args.master, args.follower = DEFAULT_MASTER, DEFAULT_FOLLOWER
            bindings = validate_bindings(ports, args.master, args.follower, args.bitrate)
        if args.expect_port:
            validate_expected(bindings, args.expect_port)
        if args.dry_run:
            for port, name in bindings:
                print(f"{port.usb} → {name}，波特率 {args.bitrate}（当前：{port.name}）")
            return 0
        apply_bindings(bindings, args.bitrate)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
