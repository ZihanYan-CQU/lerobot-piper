"""Read camera aliases and CAN state without authorization or device changes."""

from __future__ import annotations

from dataclasses import dataclass

if __package__:
    from . import camera_aliases as cameras
    from . import can_port_bindings as can
else:
    import camera_aliases as cameras
    import can_port_bindings as can


@dataclass(frozen=True)
class BindingCheck:
    """One actual binding result; None means an explicitly unselected CAN role."""

    name: str
    ok: bool | None
    detail: str


def rule_lines(content: str) -> set[tuple[str, ...]]:
    """Ignore comments and separator whitespace in this tool's generated rules."""
    return {
        tuple(field.strip() for field in line.split(","))
        for line in content.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def check_cameras(
    snapshot: list[cameras.Camera], mappings: list[tuple[str | None, str]], current: list[cameras.Camera]
) -> list[BindingCheck]:
    """Check live symlinks, selected identities, and matching persistent rules."""
    before = {camera.node: camera for camera in snapshot}
    by_node = {camera.node: camera for camera in current}
    content = cameras.RULES_FILE.read_text() if cameras.RULES_FILE.exists() else ""
    installed = rule_lines(content) if content.startswith(cameras.MARKER) else set()
    results = []
    for node, alias in mappings:
        name = f"/dev/{alias}"
        try:
            cameras.parse_mapping(f"{node or 'video0'}={alias}")
            if sum(value == alias for _, value in mappings) > 1:
                raise ValueError("设备名称重复，请为每个位置使用不同别名。")
            if node and sum(value == node for value, _ in mappings) > 1:
                raise ValueError("同一相机被重复选择，请分别选择对应设备。")
            if not node:
                raise ValueError("未选择相机或相机已断开，请连接并刷新后选择。")
            camera, original = by_node.get(node), before.get(node)
            if camera is None:
                raise ValueError(f"/dev/{node} 已断开，请连接后刷新。")
            if original is None or cameras.identity_token(camera) != cameras.identity_token(original):
                raise ValueError(f"/dev/{node} 的设备已变化，请刷新后重新选择。")
            link, target = cameras.DEV_DIR / alias, cameras.DEV_DIR / node
            if not link.is_symlink():
                reason = "已被其他文件占用" if link.exists() else "别名不存在，尚未绑定"
                raise ValueError(reason + "。")
            if not link.exists():
                raise ValueError(f"别名指向 {link.readlink()}，目标已断开或不存在。")
            if not link.samefile(target):
                raise ValueError(f"实际指向 {link.resolve()}，所选设备为 /dev/{node}，绑定不一致。")
            if not target.is_char_device():
                raise ValueError(f"/dev/{node} 不是可用的视频字符设备。")
            expected = set()
            for mode in ("auto", "serial", "path"):
                try:
                    expected.update(rule_lines(cameras.make_rules(current, [(node, alias)], mode)))
                except ValueError:
                    continue
            if not installed.intersection(expected):
                raise ValueError(f"当前链接正确（/dev/{node}），但持久化规则未匹配，请保存绑定。")
            results.append(BindingCheck(name, True, f"已绑定 → /dev/{node}；持久化规则已保存。"))
        except (OSError, ValueError) as exc:
            results.append(BindingCheck(name, False, str(exc)))
    return results


def check_can(
    snapshot: list[can.CanPort], master: str | None, follower: str | None,
    bitrate: int, current: list[can.CanPort],
) -> list[BindingCheck]:
    """Compare only selected roles to fresh kernel names, rates, and UP flags."""
    can.validate_bitrate(bitrate)
    if master and master == follower:
        raise ValueError("主臂和从臂不能选择同一个 USB CAN 端口。")
    before = {port.usb: port for port in snapshot}
    by_usb = {port.usb: port for port in current}
    results = []
    for usb, target, title in ((master, "can_master", "主臂"), (follower, "can_follower", "从臂")):
        name = f"{title} · {target}"
        if not usb:
            results.append(BindingCheck(name, None, "本次不配置，未检查。"))
            continue
        port, original = by_usb.get(usb), before.get(usb)
        if port is None:
            results.append(BindingCheck(name, False, f"USB {usb} 已断开，请连接后刷新。"))
            continue
        if original is None or can.identity_token(port) != can.identity_token(original):
            results.append(BindingCheck(name, False, f"USB {usb} 的设备已变化，请刷新后重新选择。"))
            continue
        actual = f"USB {usb} → {port.name}，{'UP' if port.up else 'DOWN'}，波特率 {port.bitrate or '未读取'}"
        differences = []
        if port.name != target:
            differences.append(f"接口名应为 {target}")
        if port.bitrate != bitrate:
            differences.append(f"波特率应为 {bitrate}")
        if not port.up:
            differences.append("接口尚未启用")
        detail = actual + ("；" + "；".join(differences) if differences else "；已绑定并启用。")
        results.append(BindingCheck(name, not differences, detail))
    return results


def default_can_selections(ports: list[can.CanPort]) -> tuple[str | None, str | None]:
    """Recognize existing role names before falling back to example USB positions."""
    by_name = {port.name: port.usb for port in ports}
    selected = [by_name.get(name) for name in ("can_master", "can_follower")]
    available = {port.usb for port in ports} - set(selected)
    for index, usb in enumerate((can.DEFAULT_MASTER, can.DEFAULT_FOLLOWER)):
        if selected[index] is None and usb in available:
            selected[index] = usb
            available.remove(usb)
    return selected[0], selected[1]
