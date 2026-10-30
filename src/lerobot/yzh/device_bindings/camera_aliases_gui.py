#!/usr/bin/env python3
"""Native GTK panel for RGB camera aliases and Piper master/follower CAN ports."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

if __package__:
    from . import can_port_bindings as can
    from . import camera_aliases as core
else:
    import can_port_bindings as can
    import camera_aliases as core


def saved_bindings(cameras: list[core.Camera]) -> list[tuple[str | None, str]]:
    """Restore all managed aliases, leaving disconnected cameras unselected."""
    aliases = []
    if core.RULES_FILE.exists():
        content = core.RULES_FILE.read_text()
        if content.startswith(core.MARKER):
            aliases = re.findall(r'^.*SYMLINK\+="(video_[A-Za-z0-9_-]+)"', content, re.MULTILINE)
    if not aliases:
        return [core.parse_mapping(value) for value in core.DEFAULT_MAPS]
    result = []
    for alias in aliases:
        matches = [
            camera.node
            for camera in cameras
            if f"/dev/{alias}" in camera.properties.get("DEVLINKS", "").split()
        ]
        result.append((matches[0] if len(matches) == 1 else None, alias))
    return result


def install_command(cameras: list[core.Camera], mappings: list[tuple[str, str]]) -> list[str]:
    """Pass an identity snapshot to the installer and use the desktop authentication agent."""
    by_node = {camera.node: camera for camera in cameras}
    command = ["/usr/bin/python3", str(Path(core.__file__).resolve())]
    for node, alias in mappings:
        command.extend(["--map", f"{node}={alias}"])
    for node in sorted({node for node, _ in mappings}):
        command.extend(["--expect-camera", f"{node}={core.identity_token(by_node[node])}"])
    if os.geteuid() != 0:
        pkexec = shutil.which("pkexec")
        if pkexec is None:
            raise RuntimeError("找不到 pkexec，无法显示系统授权窗口。请安装系统的 policykit-1 软件包。")
        command.insert(0, pkexec)
    return command


class CameraPanel:
    """A responsive desktop window with one RGB selector per editable alias."""

    def __init__(self, gtk, gdk, glib, window=None):
        self.gtk, self.gdk, self.glib = gtk, gdk, glib
        self.cameras: list[core.Camera] = []
        self.rgb: list[core.Camera] = []
        self.rows: list[dict] = []
        self.loaded = False
        self.busy = False
        self.saving = False
        self.closed = False
        self.previews: dict[str, subprocess.Popen] = {}
        owns_window = window is None
        self.window = window if window is not None else gtk.Window(title="RGB 相机绑定")
        if owns_window:
            self.window.set_default_size(940, 720)
            self.window.set_position(gtk.WindowPosition.CENTER)
            self.window.connect("delete-event", self.on_close)
            self.window.connect("destroy", lambda *_: gtk.main_quit() if gtk.main_level() else None)

        css = gtk.CssProvider()
        css.load_from_data(b"""
            window { background: #f4f6fa; }
            .title { font-size: 25px; font-weight: bold; color: #172b4d; }
            .muted { color: #52637c; }
            .count { color: #146f59; font-weight: bold; }
            .card { background: white; border: 1px solid #dbe2ed; border-radius: 10px; padding: 16px; }
            .row-title { font-weight: bold; color: #243752; }
            .primary { background: #2563eb; color: white; font-weight: bold; border-radius: 6px; }
            .status { color: #334760; }
            entry, combobox button { min-height: 30px; }
            button { padding: 7px 12px; }
        """)
        gtk.StyleContext.add_provider_for_screen(gdk.Screen.get_default(), css, gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        outer = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=16)
        outer.set_border_width(28)
        self.widget = outer
        if owns_window:
            self.window.add(outer)
        outer.pack_start(self.label("相机 RGB 绑定", "title"), False, False, 0)
        outer.pack_start(self.label("为每个位置选择彩色相机，再保存对应的设备名称。", "muted"), False, False, 0)

        toolbar = gtk.Box(spacing=10)
        self.count = self.label("正在查找 RGB 相机…", "count")
        toolbar.pack_start(self.count, True, True, 0)
        self.refresh_button = gtk.Button(label="刷新相机")
        self.refresh_button.connect("clicked", lambda *_: self.refresh())
        toolbar.pack_start(self.refresh_button, False, False, 0)
        self.add_button = gtk.Button(label="＋ 添加别名")
        self.add_button.connect("clicked", self.add_clicked)
        toolbar.pack_start(self.add_button, False, False, 0)
        outer.pack_start(toolbar, False, False, 0)

        self.scroll = gtk.ScrolledWindow()
        self.scroll.set_policy(gtk.PolicyType.NEVER, gtk.PolicyType.AUTOMATIC)
        self.rows_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=12)
        self.scroll.add(self.rows_box)
        outer.pack_start(self.scroll, True, True, 0)

        self.validation = self.label("", "muted")
        outer.pack_start(self.validation, False, False, 0)
        self.status = self.label("仅读取相机信息；点击保存后才会更新绑定。", "status")
        self.status.set_line_wrap(True)
        self.status.set_selectable(True)
        outer.pack_start(self.status, False, False, 0)
        footer = gtk.Box(spacing=16)
        note = self.label("保存时可能需要输入系统管理员密码。", "muted")
        footer.pack_start(note, True, True, 0)
        self.save_button = gtk.Button(label="保存绑定")
        self.save_button.get_style_context().add_class("primary")
        self.save_button.connect("clicked", self.save)
        footer.pack_start(self.save_button, False, False, 0)
        outer.pack_start(footer, False, False, 0)
        self.widget.show_all()
        if owns_window:
            self.window.show_all()
        self.refresh()

    def label(self, text: str, style: str):
        """Create a left-aligned, styled label without markup interpolation."""
        label = self.gtk.Label(label=text)
        label.set_xalign(0)
        label.get_style_context().add_class(style)
        return label

    def add_row(self, node: str | None, alias: str) -> None:
        """Add an editable role and a read-only combo containing RGB nodes only."""
        gtk = self.gtk
        card = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=10)
        card.get_style_context().add_class("card")
        fields = gtk.Box(spacing=12)
        name_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=6)
        name_box.pack_start(self.label("设备名称", "row-title"), False, False, 0)
        alias_box = gtk.Box(spacing=5)
        alias_box.pack_start(self.label("/dev/", "muted"), False, False, 0)
        entry = gtk.Entry(text=alias)
        entry.set_width_chars(17)
        alias_box.pack_start(entry, True, True, 0)
        name_box.pack_start(alias_box, False, False, 0)
        fields.pack_start(name_box, False, False, 0)
        selection = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=6)
        selection.pack_start(self.label("RGB 相机", "row-title"), False, False, 0)
        combo = gtk.ComboBoxText()
        combo.append("", "请选择 RGB 相机")
        for device in self.rgb:
            serial = device.properties.get("ID_SERIAL_SHORT", "无序列号")
            combo.append(device.node, f"/dev/{device.node}   ·   {serial}")
        combo.set_hexpand(True)
        combo.set_active_id(node if node in {device.node for device in self.rgb} else "")
        selection.pack_start(combo, False, False, 0)
        fields.pack_start(selection, True, True, 0)
        remove = gtk.Button(label="移除")
        remove.set_valign(gtk.Align.END)
        fields.pack_start(remove, False, False, 0)
        card.pack_start(fields, False, False, 0)
        detail_line = gtk.Box(spacing=8)
        detail = self.label("", "muted")
        detail.set_line_wrap(True)
        detail.set_hexpand(True)
        detail_line.pack_start(detail, True, True, 0)
        preview = gtk.Button(label="预览画面")
        detail_line.pack_start(preview, False, False, 0)
        card.pack_start(detail_line, False, False, 0)
        row = {"card": card, "entry": entry, "combo": combo, "detail": detail, "preview": preview}
        self.rows.append(row)
        self.rows_box.pack_start(card, False, False, 0)
        entry.connect("changed", lambda *_: self.update_validation())
        combo.connect("changed", lambda *_: self.update_validation())
        remove.connect("clicked", lambda *_: self.remove_row(row))
        preview.connect("clicked", lambda *_: self.preview(row))
        card.show_all()
        self.update_validation()

    def add_clicked(self, *_args) -> None:
        """Offer a free alias name for another camera position."""
        names = {row["entry"].get_text() for row in self.rows}
        number = 1
        while f"video_camera{number}" in names:
            number += 1
        self.add_row(None, f"video_camera{number}")

    def remove_row(self, row: dict) -> None:
        """Remove an alias from the next saved configuration."""
        self.rows.remove(row)
        row["card"].destroy()
        self.update_validation()

    def selections(self) -> list[tuple[str, str]]:
        """Validate all rows before allowing a save."""
        if not self.rows:
            raise ValueError("请至少添加一个别名。")
        mappings = []
        for row in self.rows:
            node = row["combo"].get_active_id()
            if not node:
                raise ValueError("请为每个别名选择一台 RGB 相机。")
            mappings.append(core.parse_mapping(f"{node}={row['entry'].get_text().strip()}"))
        if len({alias for _, alias in mappings}) != len(mappings):
            raise ValueError("设备名称不能重复。")
        if len({node for node, _ in mappings}) != len(mappings):
            raise ValueError("不同位置请选择不同的 RGB 相机。")
        return mappings

    def update_validation(self) -> None:
        """Keep missing or duplicate selections visible and prevent invalid saves."""
        by_node = {device.node: device for device in self.rgb}
        for row in self.rows:
            device = by_node.get(row["combo"].get_active_id())
            row["detail"].set_text(device.interface or device.name if device else "尚未选择相机")
            row["preview"].set_sensitive(device is not None and not self.busy)
        try:
            self.selections()
            self.validation.set_text("绑定会在重启后保留。移除的别名会在保存时取消。")
            valid = True
        except ValueError as exc:
            self.validation.set_text(str(exc))
            valid = False
        self.save_button.set_sensitive(valid and not self.busy)

    def set_busy(self, value: bool) -> None:
        """Disable edits while scanning or saving so a submitted selection stays intact."""
        self.busy = value
        self.refresh_button.set_sensitive(not value)
        self.add_button.set_sensitive(not value)
        self.rows_box.set_sensitive(not value)
        self.update_validation()

    def background(self, work, complete) -> None:
        """Perform hardware queries and privileged work without blocking the GTK loop."""
        self.set_busy(True)

        def worker():
            try:
                result, error = work(), None
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                result, error = None, str(exc)
            self.glib.idle_add(finish, result, error)

        def finish(result, error):
            if not self.closed:
                self.set_busy(False)
                complete(result, error)
            return False

        threading.Thread(target=worker, daemon=True).start()

    def refresh(self) -> None:
        """Rescan RGB sensors; preserve choices only if the selected identity is unchanged."""
        old_devices = {device.node: device for device in self.cameras}
        previous = [
            (row["combo"].get_active_id(), row["entry"].get_text()) for row in self.rows
        ]
        self.status.set_text("正在读取 RGB 相机信息…")

        def scan():
            cameras = core.discover_cameras()
            rgb, unavailable = core.filter_rgb_cameras(cameras)
            bindings = saved_bindings(cameras) if not self.loaded else previous
            return cameras, rgb, unavailable, bindings

        def complete(result, error):
            if error:
                self.cameras, self.rgb = [], []
                for row in self.rows:
                    row["combo"].remove_all()
                self.count.set_text("读取失败")
                self.status.set_text(error)
                self.update_validation()
                return
            cameras, rgb, unavailable, bindings = result
            current_devices = {device.node: device for device in cameras}
            for row in self.rows:
                row["card"].destroy()
            self.rows = []
            self.cameras, self.rgb = cameras, rgb
            for node, alias in bindings:
                if self.loaded and node:
                    before, after = old_devices.get(node), current_devices.get(node)
                    if before is None or after is None or core.identity_token(before) != core.identity_token(after):
                        node = None
                self.add_row(node, alias)
            self.loaded = True
            self.count.set_text(f"已找到 {len(rgb)} 个 RGB 视频接口")
            if rgb:
                message = "选择相机后点击“保存绑定”。可先预览画面确认安装位置。"
            else:
                message = "没有找到可确认的 RGB 相机，请检查连接后刷新。"
            if unavailable:
                message += "\n" + "\n".join(unavailable)
            self.status.set_text(message)
            self.update_validation()

        self.background(scan, complete)

    def save(self, *_args) -> None:
        """Install the selected mappings, authenticating only when the user clicks save."""
        try:
            mappings = self.selections()
            core.make_rules(self.cameras, mappings, "auto")
            core.check_conflicts(mappings)
            command = install_command(self.cameras, mappings)
        except (OSError, ValueError, RuntimeError) as exc:
            self.show_error(str(exc))
            return
        self.saving = True
        self.status.set_text("正在保存绑定，请完成系统授权并稍候…")

        def install():
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            if result.returncode != 0:
                detail = (result.stderr + "\n" + result.stdout).strip()
                if result.returncode == 126:
                    detail = "已取消系统授权，绑定未更新。"
                raise RuntimeError(detail or "保存失败，请确认系统授权服务可用。")
            return result.stdout

        def complete(output, error):
            self.saving = False
            if error:
                self.status.set_text("保存未完成。")
                self.show_error(error)
            else:
                details = "；".join(f"/dev/{alias} → /dev/{node}" for node, alias in mappings)
                self.status.set_text(f"绑定已保存并验证：{details}")
                if "连接位置" in output:
                    self.status.set_text(self.status.get_text() + "\n部分相机按连接位置绑定，请保持 USB 插口不变。")

        self.background(install, complete)

    def preview(self, row: dict) -> None:
        """Open a selected color stream in ffplay without altering its alias."""
        node = row["combo"].get_active_id()
        executable = shutil.which("ffplay")
        if executable is None:
            self.show_error("预览需要 ffplay，请安装系统的 ffmpeg 软件包。")
            return
        if node in self.previews and self.previews[node].poll() is None:
            self.status.set_text(f"/dev/{node} 的预览窗口已打开。")
            return
        try:
            process = subprocess.Popen(
                [executable, "-hide_banner", "-loglevel", "error", "-f", "video4linux2",
                 "-window_title", f"RGB 预览 · /dev/{node}", "-i", f"/dev/{node}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
        except OSError as exc:
            self.show_error(str(exc))
            return
        self.previews[node] = process

        def watch():
            _, error = process.communicate()
            if process.returncode and not self.closed:
                self.glib.idle_add(self.show_error, f"/dev/{node} 预览失败：\n{error.strip()}")

        threading.Thread(target=watch, daemon=True).start()

    def show_error(self, message: str) -> bool:
        """Show actionable failures while leaving the current choices intact."""
        if self.closed:
            return False
        dialog = self.gtk.MessageDialog(
            transient_for=self.window, modal=True, message_type=self.gtk.MessageType.ERROR,
            buttons=self.gtk.ButtonsType.CLOSE, text="操作未完成",
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()
        return False

    def on_close(self, *_args) -> bool:
        """Keep the installer visible until it finishes, and close this panel's previews."""
        if self.saving:
            self.status.set_text("绑定正在保存，请等待系统授权或保存结果后再关闭。")
            return True
        self.closed = True
        for process in self.previews.values():
            if process.poll() is None:
                process.terminate()
        return False


def can_install_command(ports: list[can.CanPort], master: str, follower: str, bitrate: int) -> list[str]:
    """Authenticate once for both arm roles and preserve their selected USB identities."""
    bindings = can.validate_bindings(ports, master, follower, bitrate)
    command = [
        "/usr/bin/python3", str(Path(can.__file__).resolve()),
        "--master", master, "--follower", follower, "--bitrate", str(bitrate),
    ]
    for port, _ in bindings:
        command.extend(["--expect-port", f"{port.usb}={can.identity_token(port)}"])
    if os.geteuid() != 0:
        pkexec = shutil.which("pkexec")
        if pkexec is None:
            raise RuntimeError("找不到 pkexec，无法显示系统授权窗口。")
        command.insert(0, pkexec)
    return command


class CanPanel:
    """Select the USB adapters assigned to can_master and can_follower."""

    # The two pages share GTK labels, asynchronous job handling, and error presentation.
    label = CameraPanel.label
    background = CameraPanel.background
    show_error = CameraPanel.show_error
    on_close = CameraPanel.on_close

    def __init__(self, gtk, gdk, glib, window):
        self.gtk, self.gdk, self.glib, self.window = gtk, gdk, glib, window
        self.busy = self.saving = self.closed = self.loaded = False
        self.previews = {}
        self.ports: list[can.CanPort] = []
        self.rows = []
        outer = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=16)
        outer.set_border_width(28)
        self.widget = outer
        outer.pack_start(self.label("机械臂 CAN 绑定", "title"), False, False, 0)
        outer.pack_start(self.label("为主臂和从臂选择 USB CAN 端口，并设置通信波特率。", "muted"), False, False, 0)
        toolbar = gtk.Box(spacing=12)
        self.count = self.label("正在查找 USB CAN 端口…", "count")
        toolbar.pack_start(self.count, True, True, 0)
        self.refresh_button = gtk.Button(label="刷新端口")
        self.refresh_button.connect("clicked", lambda *_: self.refresh())
        toolbar.pack_start(self.refresh_button, False, False, 0)
        outer.pack_start(toolbar, False, False, 0)
        self.fields = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=14)
        for role, title in (("can_master", "主臂"), ("can_follower", "从臂")):
            card = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=10)
            card.get_style_context().add_class("card")
            card.pack_start(self.label(f"{title}  ·  {role}", "row-title"), False, False, 0)
            combo = gtk.ComboBoxText()
            card.pack_start(combo, False, False, 0)
            detail = self.label("", "muted")
            card.pack_start(detail, False, False, 0)
            self.rows.append({"role": role, "combo": combo, "detail": detail})
            self.fields.pack_start(card, False, False, 0)
            combo.connect("changed", lambda *_: self.update_validation())
        bitrate_line = gtk.Box(spacing=14)
        bitrate_line.pack_start(self.label("CAN 波特率（bit/s）", "row-title"), False, False, 0)
        self.bitrate = gtk.ComboBoxText.new_with_entry()
        for rate in (125000, 250000, 500000, 1000000):
            self.bitrate.append_text(str(rate))
        self.bitrate.get_child().set_text("1000000")
        bitrate_line.pack_start(self.bitrate, False, False, 0)
        self.fields.pack_start(bitrate_line, False, False, 0)
        outer.pack_start(self.fields, False, False, 0)
        self.validation = self.label("", "muted")
        outer.pack_start(self.validation, False, False, 0)
        self.status = self.label("", "status")
        self.status.set_line_wrap(True)
        self.status.set_selectable(True)
        outer.pack_start(self.status, True, True, 0)
        footer = gtk.Box(spacing=12)
        note = self.label("CAN 端口激活在重启或重新插拔后需要再次执行。", "muted")
        note.set_line_wrap(True)
        footer.pack_start(note, True, True, 0)
        self.save_button = gtk.Button(label="绑定并激活主从臂")
        self.save_button.get_style_context().add_class("primary")
        self.save_button.connect("clicked", self.save)
        footer.pack_start(self.save_button, False, False, 0)
        outer.pack_start(footer, False, False, 0)
        self.bitrate.get_child().connect("changed", lambda *_: self.update_validation())
        self.widget.show_all()
        self.refresh()

    def selections(self) -> tuple[str, str, int]:
        """Require two distinct connected CAN adapters and a valid bitrate."""
        master, follower = (row["combo"].get_active_id() for row in self.rows)
        if not master or not follower:
            raise ValueError("请分别为主臂和从臂选择 USB CAN 端口。")
        try:
            bitrate = int(self.bitrate.get_child().get_text().strip())
        except ValueError as exc:
            raise ValueError("波特率必须为整数，例如 1000000。") from exc
        can.validate_bindings(self.ports, master, follower, bitrate)
        return master, follower, bitrate

    def update_validation(self) -> None:
        """Show current interface state and reject selecting the same adapter twice."""
        if not hasattr(self, "save_button"):
            return
        by_usb = {port.usb: port for port in self.ports}
        for row in self.rows:
            port = by_usb.get(row["combo"].get_active_id())
            text = "请选择端口"
            if port:
                text = f"当前名称：{port.name}   ·   {'已启用' if port.up else '未启用'}   ·   波特率：{port.bitrate or '未设置/未读取'}"
            row["detail"].set_text(text)
        try:
            self.selections()
            self.validation.set_text("执行时会重新配置所选 CAN 接口；请先停止占用这些接口的机械臂程序。")
            valid = True
        except ValueError as exc:
            self.validation.set_text(str(exc))
            valid = False
        self.save_button.set_sensitive(valid and not self.busy)

    def set_busy(self, value: bool) -> None:
        """Prevent edits while port discovery or activation is in progress."""
        self.busy = value
        self.fields.set_sensitive(not value)
        self.refresh_button.set_sensitive(not value)
        self.update_validation()

    def refresh(self) -> None:
        """Read USB CAN ports, retaining explicit selections when adapters are unchanged."""
        previous = [row["combo"].get_active_id() for row in self.rows]
        old_ports = {port.usb: port for port in self.ports}
        self.status.set_text("正在读取端口信息…")

        def complete(ports, error):
            if error:
                self.ports = []
                self.status.set_text(error)
            else:
                self.ports = ports
                self.status.set_text("选择端口后点击“绑定并激活主从臂”，随后完成系统授权。")
            by_usb = {port.usb: port for port in self.ports}
            defaults = (can.DEFAULT_MASTER, can.DEFAULT_FOLLOWER)
            for index, row in enumerate(self.rows):
                combo = row["combo"]
                selected = previous[index] if self.loaded else defaults[index]
                if self.loaded and selected:
                    before, after = old_ports.get(selected), by_usb.get(selected)
                    if before is None or after is None or can.identity_token(before) != can.identity_token(after):
                        selected = None
                combo.remove_all()
                combo.append("", "请选择 USB CAN 端口")
                for port in self.ports:
                    combo.append(port.usb, f"{port.usb}   ·   {port.name}   ·   {port.serial or '无序列号'}")
                combo.set_active_id(selected if selected in by_usb else "")
            self.loaded = True
            self.count.set_text(f"已找到 {len(self.ports)} 个 USB CAN 端口")
            if not self.ports and not error:
                self.status.set_text("未发现 USB CAN 适配器，请检查连接后刷新。")
            self.update_validation()

        self.background(can.discover_ports, complete)

    def save(self, *_args) -> None:
        """Run the existing activation script through the checked privileged helper."""
        try:
            master, follower, bitrate = self.selections()
            command = can_install_command(self.ports, master, follower, bitrate)
        except (OSError, ValueError, RuntimeError) as exc:
            self.show_error(str(exc))
            return
        self.saving = True
        self.status.set_text("正在绑定并激活 CAN 接口，请完成系统授权…")

        def activate():
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                detail = (result.stderr + "\n" + result.stdout).strip()
                if result.returncode == 126:
                    detail = "已取消系统授权，CAN 端口未修改。"
                raise RuntimeError(detail or "CAN 绑定失败。")
            return result.stdout, can.discover_ports()

        def complete(result, error):
            self.saving = False
            if error:
                self.status.set_text("CAN 绑定未完成，请查看错误信息后刷新端口状态。")
                self.show_error(error)
            else:
                output, self.ports = result
                self.status.set_text(output.strip())
                # Refresh visible names without replacing the user's selected USB paths.
                for row in self.rows:
                    selected = row["combo"].get_active_id()
                    row["combo"].remove_all()
                    row["combo"].append("", "请选择 USB CAN 端口")
                    for port in self.ports:
                        row["combo"].append(port.usb, f"{port.usb}   ·   {port.name}   ·   {port.serial or '无序列号'}")
                    row["combo"].set_active_id(selected)
                self.update_validation()

        self.background(activate, complete)


def create_device_window(gtk, gdk, glib):
    """Put camera and arm configuration in separate tabs of one native window."""
    window = gtk.Window(title="Piper 设备绑定 · 相机与机械臂")
    window.set_default_size(980, 770)
    window.set_position(gtk.WindowPosition.CENTER)
    notebook = gtk.Notebook()
    window.add(notebook)
    cameras = CameraPanel(gtk, gdk, glib, window=window)
    arms = CanPanel(gtk, gdk, glib, window)
    notebook.append_page(cameras.widget, gtk.Label(label="RGB 相机"))
    notebook.append_page(arms.widget, gtk.Label(label="机械臂 CAN"))

    def close(*_args):
        for panel in (cameras, arms):
            if panel.saving:
                panel.status.set_text("正在执行绑定，请完成系统授权或等待结果后再关闭。")
                return True
        cameras.on_close()
        arms.on_close()
        return False

    window.connect("delete-event", close)
    window.connect("destroy", lambda *_: gtk.main_quit() if gtk.main_level() else None)
    window.show_all()
    return window, notebook, cameras, arms


def run_gui() -> int:
    """Start the GTK panel using the system Python's desktop bindings."""
    try:
        import gi

        gi.require_version("Gtk", "3.0")
        gi.require_version("Gdk", "3.0")
        from gi.repository import Gdk, GLib, Gtk
    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            "图形面板需要 GTK 3 和 PyGObject。请使用 /usr/bin/python3 运行；"
            "若仍缺少组件，请安装 python3-gi 和 gir1.2-gtk-3.0。"
        ) from exc
    initialized, _ = Gtk.init_check()
    if not initialized:
        raise RuntimeError("无法连接桌面显示，请在 Linux 图形桌面的终端中运行面板。")
    create_device_window(Gtk, Gdk, GLib)
    Gtk.main()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run_gui())
    except (OSError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)
