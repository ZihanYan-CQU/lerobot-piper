#!/usr/bin/env python3
"""Native GTK panel for RGB camera aliases and Piper master/follower CAN ports."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime

if __package__:
    from . import binding_checks as checks
    from . import can_port_bindings as can
    from . import camera_aliases as core
    from .privileged_session import PrivilegedSession
else:
    import binding_checks as checks
    import can_port_bindings as can
    import camera_aliases as core
    from privileged_session import PrivilegedSession


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


def camera_request(cameras: list[core.Camera], mappings: list[tuple[str, str]]) -> dict:
    """Send explicit mappings and their device identities to the shared helper."""
    by_node = {camera.node: camera for camera in cameras}
    return {
        "operation": "camera",
        "mappings": [f"{node}={alias}" for node, alias in mappings],
        "expected": [f"{node}={core.identity_token(by_node[node])}" for node in sorted({n for n, _ in mappings})],
    }


class DevicePanel:
    """Shared GTK job, authorization, error, and window-lifecycle behavior."""

    def __init__(self, gtk, gdk, glib, window, session=None):
        self.gtk, self.gdk, self.glib, self.window = gtk, gdk, glib, window
        self.session = session if session is not None else PrivilegedSession()
        self.loaded = self.busy = self.saving = self.closed = False
        self.previews: dict[str, subprocess.Popen] = {}

    def label(self, text: str, style: str):
        """Create a left-aligned, styled label without markup interpolation."""
        label = self.gtk.Label(label=text)
        label.set_xalign(0)
        label.get_style_context().add_class(style)
        return label

    def add_check_section(self, outer) -> None:
        """Keep a visible, selectable report separate from save-progress messages."""
        gtk = self.gtk
        frame = gtk.Frame(label="绑定检查 · 仅读取，无需密码")
        box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=6)
        box.set_border_width(10)
        self.check_summary = self.label("尚未检查", "muted")
        box.pack_start(self.check_summary, False, False, 0)
        self.check_details = self.label("点击“检查绑定”可重新读取当前设备状态。", "status")
        self.check_details.set_line_wrap(True)
        self.check_details.set_selectable(True)
        scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.PolicyType.NEVER, gtk.PolicyType.AUTOMATIC)
        scroll.set_min_content_height(65)
        scroll.set_max_content_height(130)
        scroll.set_propagate_natural_height(True)
        scroll.add(self.check_details)
        box.pack_start(scroll, True, True, 0)
        frame.add(box)
        outer.pack_start(frame, False, False, 0)

    def check_message(self, summary: str, detail: str, style: str = "muted") -> None:
        """Replace any previous pass state, including after edits or read errors."""
        self.check_summary.set_text(summary)
        context = self.check_summary.get_style_context()
        for name in ("muted", "check-ok", "check-issue"):
            context.remove_class(name)
        context.add_class(style)
        self.check_details.set_text(detail)

    def configuration_changed(self, *_args) -> None:
        """Invalidate results when they no longer describe the visible selection."""
        self.update_validation()
        if hasattr(self, "check_summary"):
            self.check_message("配置已更改，请重新检查", "点击“检查绑定”，确认当前选择是否已在系统中生效。")

    def show_checks(self, results: list[checks.BindingCheck]) -> None:
        """Summarize only checked items; skipped roles never count as a pass."""
        selected = [item for item in results if item.ok is not None]
        passed = sum(item.ok for item in selected)
        if not selected:
            summary, style = "尚未选择要检查的设备", "muted"
        elif passed == len(selected):
            summary, style = f"全部已绑定 · {passed}/{len(selected)} 项通过", "check-ok"
        else:
            summary, style = f"存在未就绪项 · {passed}/{len(selected)} 项通过", "check-issue"
        summary += f" · 检查时间 {datetime.now().astimezone():%H:%M:%S}"
        lines = []
        for item in results:
            state = "未检查" if item.ok is None else "通过" if item.ok else "未就绪"
            lines.append(f"[{state}] {item.name}：{item.detail}")
        self.check_message(summary, "\n".join(lines), style)

    def run_check(self, work) -> None:
        """Run live inspection without starting the privileged helper."""
        self.check_message("正在检查…", "正在读取当前设备和绑定状态。")

        def complete(result, error):
            if error:
                self.check_message("检查失败", error, "check-issue")
            else:
                self.show_checks(result)

        self.background(work, complete)

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
        self.session.close()
        return False


class CameraPanel(DevicePanel):
    """A responsive desktop window with one RGB selector per editable alias."""

    def __init__(self, gtk, gdk, glib, window=None, session=None):
        owns_window = window is None
        window = window if window is not None else gtk.Window(title="RGB 相机绑定")
        super().__init__(gtk, gdk, glib, window, session)
        self.cameras: list[core.Camera] = []
        self.rgb: list[core.Camera] = []
        self.rows: list[dict] = []
        if owns_window:
            self.window.set_default_size(980, 870)
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
            .check-ok { color: #14733c; font-weight: bold; }
            .check-issue { color: #b54708; font-weight: bold; }
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
        self.check_button = gtk.Button(label="检查绑定")
        self.check_button.connect("clicked", self.check_bindings)
        toolbar.pack_start(self.check_button, False, False, 0)
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
        self.add_check_section(outer)
        footer = gtk.Box(spacing=16)
        note = self.label("首次保存时授权；本窗口内相机与 CAN 共用授权，关闭后结束。", "muted")
        note.set_line_wrap(True)
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
        entry.connect("changed", self.configuration_changed)
        combo.connect("changed", self.configuration_changed)
        remove.connect("clicked", lambda *_: self.remove_row(row))
        preview.connect("clicked", lambda *_: self.preview(row))
        card.show_all()
        self.configuration_changed()

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
        self.configuration_changed()

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
        self.check_button.set_sensitive(not value)
        self.add_button.set_sensitive(not value)
        self.rows_box.set_sensitive(not value)
        self.update_validation()

    def check_mappings(self) -> list[tuple[str | None, str]]:
        """Include incomplete rows so offline or unselected cameras remain visible."""
        return [(row["combo"].get_active_id() or None, row["entry"].get_text().strip()) for row in self.rows]

    def check_bindings(self, *_args) -> None:
        """Inspect aliases against a fresh camera inventory without saving anything."""
        snapshot, mappings = list(self.cameras), self.check_mappings()
        self.run_check(lambda: checks.check_cameras(snapshot, mappings, core.discover_cameras()))

    def show_current_checks(self) -> None:
        """Inspect the inventory just read by refresh without another hardware scan."""
        try:
            self.show_checks(checks.check_cameras(self.cameras, self.check_mappings(), self.cameras))
        except (OSError, ValueError, RuntimeError) as exc:
            self.check_message("检查失败", str(exc), "check-issue")

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
                self.check_message("检查失败", error, "check-issue")
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
            self.show_current_checks()

        self.background(scan, complete)

    def save(self, *_args) -> None:
        """Install the selected mappings, authenticating only when the user clicks save."""
        try:
            mappings = self.selections()
            core.make_rules(self.cameras, mappings, "auto")
            core.check_conflicts(mappings)
            request = camera_request(self.cameras, mappings)
        except (OSError, ValueError, RuntimeError) as exc:
            self.show_error(str(exc))
            return
        self.saving = True
        self.status.set_text("正在保存绑定，请完成系统授权并稍候…")

        def install():
            output = self.session.execute(request)
            results = checks.check_cameras(self.cameras, mappings, core.discover_cameras())
            return output, results

        def complete(result, error):
            self.saving = False
            if error:
                self.status.set_text("保存未完成。")
                self.check_message("保存未完成，请重新检查", error, "check-issue")
                self.show_error(error)
            else:
                output, results = result
                details = "；".join(f"/dev/{alias} → /dev/{node}" for node, alias in mappings)
                self.status.set_text(f"绑定已保存并验证：{details}")
                if "连接位置" in output:
                    self.status.set_text(self.status.get_text() + "\n部分相机按连接位置绑定，请保持 USB 插口不变。")
                self.show_checks(results)

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


def can_request(ports: list[can.CanPort], master: str | None, follower: str | None, bitrate: int) -> dict:
    """Preserve only the selected roles and their USB identity snapshots."""
    bindings = can.validate_bindings(ports, master, follower, bitrate)
    return {
        "operation": "can", "master": master, "follower": follower, "bitrate": bitrate,
        "expected": [f"{port.usb}={can.identity_token(port)}" for port, _ in bindings],
    }


class CanPanel(DevicePanel):
    """Select the USB adapters assigned to can_master and can_follower."""

    def __init__(self, gtk, gdk, glib, window, session=None):
        super().__init__(gtk, gdk, glib, window, session)
        self.ports: list[can.CanPort] = []
        self.rows = []
        outer = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=16)
        outer.set_border_width(28)
        self.widget = outer
        outer.pack_start(self.label("机械臂 CAN 绑定", "title"), False, False, 0)
        outer.pack_start(self.label("可只配置主臂、只配置从臂，或同时配置；不需要的角色选择“本次不配置”。", "muted"), False, False, 0)
        toolbar = gtk.Box(spacing=12)
        self.count = self.label("正在查找 USB CAN 端口…", "count")
        toolbar.pack_start(self.count, True, True, 0)
        self.refresh_button = gtk.Button(label="刷新端口")
        self.refresh_button.connect("clicked", lambda *_: self.refresh())
        toolbar.pack_start(self.refresh_button, False, False, 0)
        self.check_button = gtk.Button(label="检查绑定")
        self.check_button.connect("clicked", self.check_bindings)
        toolbar.pack_start(self.check_button, False, False, 0)
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
            combo.connect("changed", self.configuration_changed)
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
        self.add_check_section(outer)
        footer = gtk.Box(spacing=12)
        note = self.label("CAN 端口激活在重启或重新插拔后需要再次执行。", "muted")
        note.set_line_wrap(True)
        footer.pack_start(note, True, True, 0)
        self.save_button = gtk.Button(label="绑定并激活所选机械臂")
        self.save_button.get_style_context().add_class("primary")
        self.save_button.connect("clicked", self.save)
        footer.pack_start(self.save_button, False, False, 0)
        outer.pack_start(footer, False, False, 0)
        self.bitrate.get_child().connect("changed", self.configuration_changed)
        self.widget.show_all()
        self.refresh()

    def check_parameters(self) -> tuple[str | None, str | None, int]:
        """Snapshot GTK fields on the main thread, allowing explicitly skipped roles."""
        master, follower = (row["combo"].get_active_id() for row in self.rows)
        master, follower = master or None, follower or None
        try:
            bitrate = int(self.bitrate.get_child().get_text().strip())
        except ValueError as exc:
            raise ValueError("波特率必须为整数，例如 1000000。") from exc
        return master, follower, bitrate

    def selections(self) -> tuple[str | None, str | None, int]:
        """Require at least one role, distinct adapters, and a valid bitrate."""
        master, follower, bitrate = self.check_parameters()
        can.validate_bindings(self.ports, master, follower, bitrate)
        return master, follower, bitrate

    def update_validation(self) -> None:
        """Show current interface state and reject selecting the same adapter twice."""
        if not hasattr(self, "save_button"):
            return
        by_usb = {port.usb: port for port in self.ports}
        for row in self.rows:
            port = by_usb.get(row["combo"].get_active_id())
            text = "本次不配置此机械臂"
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
        self.check_button.set_sensitive(not value)
        self.update_validation()

    def current_checks(self, ports: list[can.CanPort]) -> list[checks.BindingCheck]:
        """Describe the visible roles, including explicitly skipped ones."""
        master, follower, bitrate = self.check_parameters()
        return checks.check_can(self.ports, master, follower, bitrate, ports)

    def check_bindings(self, *_args) -> None:
        """Read current kernel state; checking never activates a port or requests a password."""
        try:
            master, follower, bitrate = self.check_parameters()
        except ValueError as exc:
            self.check_message("检查失败", str(exc), "check-issue")
            return
        snapshot = list(self.ports)
        self.run_check(lambda: checks.check_can(snapshot, master, follower, bitrate, can.discover_ports(strict=True)))

    def show_current_checks(self) -> None:
        """Reuse a just-read inventory after refresh or activation."""
        try:
            self.show_checks(self.current_checks(self.ports))
        except (OSError, ValueError, RuntimeError) as exc:
            self.check_message("检查失败", str(exc), "check-issue")

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
                self.status.set_text("选择需要配置的端口后点击“绑定并激活所选机械臂”；本窗口仅首次执行需要授权。")
            by_usb = {port.usb: port for port in self.ports}
            defaults = checks.default_can_selections(self.ports)
            for index, row in enumerate(self.rows):
                combo = row["combo"]
                selected = previous[index] if self.loaded else defaults[index]
                if self.loaded and selected:
                    before, after = old_ports.get(selected), by_usb.get(selected)
                    if before is None or after is None or can.identity_token(before) != can.identity_token(after):
                        selected = None
                combo.remove_all()
                combo.append("", "本次不配置")
                for port in self.ports:
                    combo.append(port.usb, f"{port.usb}   ·   {port.name}   ·   {port.serial or '无序列号'}")
                combo.set_active_id(selected if selected in by_usb else "")
            self.loaded = True
            self.count.set_text(f"已找到 {len(self.ports)} 个 USB CAN 端口")
            if not self.ports and not error:
                self.status.set_text("未发现 USB CAN 适配器，请检查连接后刷新。")
            self.update_validation()
            if error:
                self.check_message("检查失败", error, "check-issue")
            else:
                self.show_current_checks()

        self.background(can.discover_ports, complete)

    def save(self, *_args) -> None:
        """Activate the selected roles through the shared privileged helper."""
        try:
            master, follower, bitrate = self.selections()
            request = can_request(self.ports, master, follower, bitrate)
        except (OSError, ValueError, RuntimeError) as exc:
            self.show_error(str(exc))
            return
        self.saving = True
        self.status.set_text("正在绑定并激活 CAN 接口，请完成系统授权…")

        def activate():
            return self.session.execute(request), can.discover_ports()

        def complete(result, error):
            self.saving = False
            if error:
                self.status.set_text("CAN 绑定未完成，请查看错误信息后刷新端口状态。")
                self.check_message("绑定未完成，请重新检查", error, "check-issue")
                self.show_error(error)
            else:
                output, self.ports = result
                self.status.set_text(output.strip())
                # Refresh visible names without replacing the user's selected USB paths.
                for row in self.rows:
                    selected = row["combo"].get_active_id()
                    row["combo"].remove_all()
                    row["combo"].append("", "本次不配置")
                    for port in self.ports:
                        row["combo"].append(port.usb, f"{port.usb}   ·   {port.name}   ·   {port.serial or '无序列号'}")
                    row["combo"].set_active_id(selected)
                self.update_validation()
                self.show_current_checks()

        self.background(activate, complete)


def create_device_window(gtk, gdk, glib):
    """Put camera and arm configuration in separate tabs of one native window."""
    window = gtk.Window(title="Piper 设备绑定 · 相机与机械臂")
    window.set_default_size(1000, 880)
    window.set_position(gtk.WindowPosition.CENTER)
    notebook = gtk.Notebook()
    window.add(notebook)
    session = PrivilegedSession()
    cameras = CameraPanel(gtk, gdk, glib, window=window, session=session)
    arms = CanPanel(gtk, gdk, glib, window, session=session)
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
