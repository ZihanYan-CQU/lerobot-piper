"""Test GUI selection and privilege handoff without a desktop or camera writes."""

import importlib.util
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1]
for name in ("camera_aliases", "can_port_bindings", "privileged_session", "binding_checks", "camera_aliases_gui"):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

core = sys.modules["camera_aliases"]
gui = sys.modules["camera_aliases_gui"]
can = sys.modules["can_port_bindings"]


def camera(node, alias, serial):
    return core.Camera(
        node, "RealSense", "0",
        {"ID_SERIAL": serial, "ID_SERIAL_SHORT": serial, "ID_USB_INTERFACE_NUM": "03",
         "ID_V4L_CAPABILITIES": ":capture:", "DEVLINKS": f"/dev/{alias}"},
        interface="RealSense RGB",
    )


class CameraPanelTest(unittest.TestCase):
    def setUp(self):
        self.devices = [camera("video4", "video_wrist", "123"), camera("video10", "video_top", "456")]
        self.mappings = [("video4", "video_wrist"), ("video10", "video_top")]

    def test_saved_custom_aliases_are_preserved_and_disconnected_nodes_stay_unselected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.rules"
            path.write_text(core.MARKER + '\n'.join(f'SYMLINK+="{alias}"' for alias in ("video_wrist", "video_top", "video_extra")))
            with patch.object(core, "RULES_FILE", path):
                bindings = gui.saved_bindings(self.devices)
        self.assertEqual(bindings, self.mappings + [(None, "video_extra")])

    def test_camera_request_supplies_all_expected_identities(self):
        request = gui.camera_request(self.devices, self.mappings)
        self.assertEqual(request["operation"], "camera")
        self.assertEqual(request["mappings"], [f"{node}={alias}" for node, alias in self.mappings])
        core.validate_selections(self.devices, self.mappings, request["expected"])

    def test_rows_reject_missing_and_duplicate_camera_selections(self):
        panel = gui.CameraPanel.__new__(gui.CameraPanel)
        for mappings in ([(None, "video_wrist")], [("video4", "video_wrist"), ("video4", "video_top")], self.mappings):
            panel.rows = []
            for node, alias in mappings:
                combo, entry = Mock(), Mock()
                combo.get_active_id.return_value = node
                entry.get_text.return_value = alias
                panel.rows.append({"combo": combo, "entry": entry})
            if mappings == self.mappings:
                self.assertEqual(panel.selections(), self.mappings)
            else:
                with self.assertRaises(ValueError):
                    panel.selections()

    def test_can_request_carries_selected_roles_bitrate_and_device_identities(self):
        ports = [
            can.CanPort("can0", "1-1.2:1.0", 7, "master", False),
            can.CanPort("can1", "1-1.3:1.0", 8, "follower", False),
        ]
        request = gui.can_request(ports, ports[0].usb, ports[1].usb, 1000000)
        self.assertEqual((request["master"], request["follower"], request["bitrate"]), (ports[0].usb, ports[1].usb, 1000000))
        can.validate_expected([(ports[0], "can_master"), (ports[1], "can_follower")], request["expected"])
        request = gui.can_request(ports, None, ports[1].usb, 500000)
        self.assertIsNone(request["master"])
        can.validate_expected([(ports[1], "can_follower")], request["expected"])

    def test_can_panel_allows_either_single_role_but_rejects_empty_or_duplicate_roles(self):
        panel = gui.CanPanel.__new__(gui.CanPanel)
        panel.ports = [can.CanPort("can0", "1-1.2:1.0", 7, "master", False)]
        panel.bitrate = Mock()
        panel.bitrate.get_child().get_text.return_value = "1000000"
        for values in (("1-1.2:1.0", ""), ("", "1-1.2:1.0"), ("", ""), ("1-1.2:1.0", "1-1.2:1.0")):
            with self.subTest(values=values):
                panel.rows = [{"combo": Mock()} for _ in values]
                for row, value in zip(panel.rows, values, strict=True):
                    row["combo"].get_active_id.return_value = value
                if values.count("") == 1:
                    self.assertEqual(panel.selections(), tuple(value or None for value in values) + (1000000,))
                else:
                    with self.assertRaises(ValueError):
                        panel.selections()

    def test_result_summary_excludes_skipped_roles_and_is_invalidated_after_an_edit(self):
        panel = gui.DevicePanel.__new__(gui.DevicePanel)
        panel.check_summary, panel.check_details = Mock(), Mock()
        panel.update_validation = Mock()
        panel.show_checks([
            gui.checks.BindingCheck("主臂", None, "本次不配置"),
            gui.checks.BindingCheck("从臂", True, "已绑定并启用"),
        ])
        self.assertIn("全部已绑定 · 1/1", panel.check_summary.set_text.call_args.args[0])
        self.assertIn("[未检查] 主臂", panel.check_details.set_text.call_args.args[0])
        panel.configuration_changed()
        self.assertEqual(panel.check_summary.set_text.call_args.args[0], "配置已更改，请重新检查")
        panel.show_checks([gui.checks.BindingCheck("从臂", False, "接口未启用")])
        self.assertIn("存在未就绪项", panel.check_summary.set_text.call_args.args[0])
        panel.show_checks([gui.checks.BindingCheck("从臂", None, "本次不配置")])
        self.assertIn("尚未选择", panel.check_summary.set_text.call_args.args[0])

    def test_can_check_snapshots_widgets_before_worker_and_never_requests_authorization(self):
        panel = gui.CanPanel.__new__(gui.CanPanel)
        port = can.CanPort("can_follower", "1-9.3:1.0", 7, "follower", True, 1000000)
        panel.ports = [port]
        panel.session = Mock()
        panel.rows = [{"combo": Mock()}, {"combo": Mock()}]
        panel.bitrate = Mock()
        main_thread = threading.get_ident()

        def widget(value):
            self.assertEqual(threading.get_ident(), main_thread, "GTK fields must be read on the main thread")
            return value

        panel.rows[0]["combo"].get_active_id.side_effect = lambda: widget("")
        panel.rows[1]["combo"].get_active_id.side_effect = lambda: widget(port.usb)
        panel.bitrate.get_child().get_text.side_effect = lambda: widget("1000000")
        panel.run_check = Mock()
        panel.check_bindings()
        work = panel.run_check.call_args.args[0]
        results, errors = [], []

        def worker():
            try:
                results.extend(work())
            except Exception as exc:
                errors.append(exc)

        with patch.object(can, "discover_ports", return_value=[port]) as discover:
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(timeout=3)
        self.assertFalse(errors)
        self.assertTrue(results[1].ok)
        discover.assert_called_once_with(strict=True)
        panel.session.execute.assert_not_called()


@unittest.skipUnless(os.environ.get("CAMERA_GUI_TEST") == "1", "Set CAMERA_GUI_TEST=1 inside a desktop session")
class NativePanelTest(unittest.TestCase):
    """Exercise the real GTK widgets with deterministic RGB/depth device fixtures."""

    def test_panel_choices_and_validation(self):
        import time

        import gi

        gi.require_version("Gtk", "3.0")
        gi.require_version("Gdk", "3.0")
        from gi.repository import Gdk, GLib, Gtk

        initialized, _ = Gtk.init_check()
        self.assertTrue(initialized, "A desktop display is required")
        devices = [camera("video4", "video_wrist", "123"), camera("video10", "video_top", "456")]
        devices.append(core.Camera("video0", "RealSense", "0", {"ID_V4L_CAPABILITIES": ":capture:"}, "RealSense Depth"))
        devices.append(core.Camera("video5", "RealSense", "1", {"ID_V4L_CAPABILITIES": ":"}, "RealSense RGB"))
        with (
            patch.object(core, "discover_cameras", return_value=devices),
            patch.object(gui, "saved_bindings", return_value=[("video4", "video_wrist"), ("video10", "video_top")]),
        ):
            panel = gui.CameraPanel(Gtk, Gdk, GLib)
            try:
                deadline = time.monotonic() + 5
                while not panel.loaded and time.monotonic() < deadline:
                    while Gtk.events_pending():
                        Gtk.main_iteration_do(False)
                    time.sleep(0.01)
                self.assertTrue(panel.loaded)
                for row in panel.rows:
                    self.assertEqual({item[1] for item in row["combo"].get_model()}, {"", "video4", "video10"})
                self.assertTrue(panel.save_button.get_sensitive())
                panel.rows[1]["combo"].set_active_id("video4")
                self.assertFalse(panel.save_button.get_sensitive())
                panel.rows[1]["combo"].set_active_id("video10")
                panel.rows[1]["entry"].set_text("video_wrist")
                self.assertFalse(panel.save_button.get_sensitive())
                panel.rows[1]["entry"].set_text("video_top")
                self.assertTrue(panel.save_button.get_sensitive())
                panel.add_clicked()
                self.assertFalse(panel.save_button.get_sensitive())
                panel.remove_row(panel.rows[-1])
                self.assertTrue(panel.save_button.get_sensitive())
                with (
                    patch.object(core, "check_conflicts"),
                    patch.object(panel.session, "execute", return_value="已验证") as install,
                ):
                    panel.save()
                    self.assertFalse(panel.save_button.get_sensitive())
                    deadline = time.monotonic() + 5
                    while panel.saving and time.monotonic() < deadline:
                        while Gtk.events_pending():
                            Gtk.main_iteration_do(False)
                        time.sleep(0.01)
                    self.assertFalse(panel.saving)
                    self.assertIn("绑定已保存并验证", panel.status.get_text())
                    self.assertTrue(panel.save_button.get_sensitive())
                    install.assert_called_once()
                while Gtk.events_pending():
                    Gtk.main_iteration_do(False)
                last_card = panel.rows[-1]["card"].get_allocation()
                self.assertLessEqual(last_card.y + last_card.height, panel.scroll.get_allocated_height())
                snapshot = os.environ.get("CAMERA_GUI_SNAPSHOT")
                if snapshot:
                    while Gtk.events_pending():
                        Gtk.main_iteration_do(False)
                    width, height = panel.window.get_size()
                    pixbuf = Gdk.pixbuf_get_from_window(panel.window.get_window(), 0, 0, width, height)
                    self.assertIsNotNone(pixbuf)
                    pixbuf.savev(snapshot, "png", [], [])
            finally:
                panel.on_close()
                panel.window.destroy()

    def test_combined_window_has_rgb_and_can_pages(self):
        import time

        import gi

        gi.require_version("Gtk", "3.0")
        gi.require_version("Gdk", "3.0")
        from gi.repository import Gdk, GLib, Gtk

        self.assertTrue(Gtk.init_check()[0])
        devices = [camera("video4", "video_wrist", "123"), camera("video10", "video_top", "456")]
        ports = [
            can.CanPort("can0", "1-1.2:1.0", 7, "master", False),
            can.CanPort("can_follower", "1-1.3:1.0", 8, "follower", True, 1000000),
        ]

        def pump_until(predicate):
            deadline = time.monotonic() + 5
            while not predicate() and time.monotonic() < deadline:
                while Gtk.events_pending():
                    Gtk.main_iteration_do(False)
                time.sleep(0.01)
            self.assertTrue(predicate())

        with (
            patch.object(core, "discover_cameras", return_value=devices),
            patch.object(gui, "saved_bindings", return_value=[("video4", "video_wrist"), ("video10", "video_top")]),
            patch.object(can, "discover_ports", return_value=ports),
        ):
            window, notebook, cameras, arms = gui.create_device_window(Gtk, Gdk, GLib)
            try:
                pump_until(lambda: cameras.loaded and arms.loaded)
                self.assertEqual(notebook.get_n_pages(), 2)
                self.assertIs(cameras.session, arms.session)
                notebook.set_current_page(1)
                self.assertEqual(arms.selections(), ("1-1.2:1.0", "1-1.3:1.0", 1000000))
                self.assertTrue(arms.save_button.get_sensitive())
                arms.rows[1]["combo"].set_active_id("1-1.2:1.0")
                self.assertFalse(arms.save_button.get_sensitive())
                arms.rows[1]["combo"].set_active_id("1-1.3:1.0")
                arms.rows[0]["combo"].set_active_id("")
                self.assertEqual(arms.selections(), (None, "1-1.3:1.0", 1000000))
                self.assertTrue(arms.save_button.get_sensitive())
                arms.rows[1]["combo"].set_active_id("")
                self.assertFalse(arms.save_button.get_sensitive())
                arms.rows[0]["combo"].set_active_id("1-1.2:1.0")
                self.assertEqual(arms.selections(), ("1-1.2:1.0", None, 1000000))
                self.assertTrue(arms.save_button.get_sensitive())
                arms.rows[1]["combo"].set_active_id("1-1.3:1.0")
                arms.bitrate.get_child().set_text("0")
                self.assertFalse(arms.save_button.get_sensitive())
                arms.bitrate.get_child().set_text("1000000")
                self.assertTrue(arms.save_button.get_sensitive())
                with (
                    patch.object(arms.session, "execute", return_value="已验证：主从臂 CAN") as activate,
                ):
                    arms.save()
                    self.assertFalse(arms.save_button.get_sensitive())
                    pump_until(lambda: not arms.saving)
                    activate.assert_called_once()
                    self.assertIn("已验证", arms.status.get_text())
                snapshot = os.environ.get("CAN_GUI_SNAPSHOT")
                if snapshot:
                    # Wait for the frame clock to paint labels updated by the job callback.
                    until = time.monotonic() + 0.15
                    while time.monotonic() < until:
                        while Gtk.events_pending():
                            Gtk.main_iteration_do(False)
                        time.sleep(0.01)
                    width, height = window.get_size()
                    pixbuf = Gdk.pixbuf_get_from_window(window.get_window(), 0, 0, width, height)
                    self.assertIsNotNone(pixbuf)
                    pixbuf.savev(snapshot, "png", [], [])
            finally:
                cameras.on_close()
                arms.on_close()
                window.destroy()


if __name__ == "__main__":
    unittest.main()
