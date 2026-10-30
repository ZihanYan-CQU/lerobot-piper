"""Test GUI selection and privilege handoff without a desktop or camera writes."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1]
for name in ("camera_aliases", "can_port_bindings", "camera_aliases_gui"):
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

    def test_install_command_authenticates_and_supplies_all_expected_identities(self):
        with patch.object(gui.os, "geteuid", return_value=1000), patch.object(gui.shutil, "which", return_value="/usr/bin/pkexec"):
            command = gui.install_command(self.devices, self.mappings)
        self.assertEqual(command[:2], ["/usr/bin/pkexec", "/usr/bin/python3"])
        self.assertEqual(command.count("--map"), 2)
        expected = [command[index + 1] for index, value in enumerate(command) if value == "--expect-camera"]
        core.validate_selections(self.devices, self.mappings, expected)

    def test_already_elevated_installer_does_not_prompt_again(self):
        with patch.object(gui.os, "geteuid", return_value=0):
            self.assertEqual(gui.install_command(self.devices, self.mappings)[0], "/usr/bin/python3")

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

    def test_can_command_carries_both_roles_bitrate_and_device_identities(self):
        ports = [
            can.CanPort("can0", "1-1.2:1.0", 7, "master", False),
            can.CanPort("can1", "1-1.3:1.0", 8, "follower", False),
        ]
        with patch.object(gui.os, "geteuid", return_value=0):
            command = gui.can_install_command(ports, ports[0].usb, ports[1].usb, 1000000)
        self.assertIn("--master", command)
        self.assertIn("--follower", command)
        self.assertEqual(command[command.index("--bitrate") + 1], "1000000")
        expected = [command[index + 1] for index, value in enumerate(command) if value == "--expect-port"]
        can.validate_expected([(ports[0], "can_master"), (ports[1], "can_follower")], expected)


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
                    patch.object(gui, "install_command", return_value=["mock-installer"]),
                    patch.object(gui.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "已验证", "")) as install,
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
                notebook.set_current_page(1)
                self.assertEqual(arms.selections(), ("1-1.2:1.0", "1-1.3:1.0", 1000000))
                self.assertTrue(arms.save_button.get_sensitive())
                arms.rows[1]["combo"].set_active_id("1-1.2:1.0")
                self.assertFalse(arms.save_button.get_sensitive())
                arms.rows[1]["combo"].set_active_id("1-1.3:1.0")
                arms.bitrate.get_child().set_text("0")
                self.assertFalse(arms.save_button.get_sensitive())
                arms.bitrate.get_child().set_text("1000000")
                self.assertTrue(arms.save_button.get_sensitive())
                with (
                    patch.object(gui, "can_install_command", return_value=["mock-can-installer"]),
                    patch.object(gui.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "已验证：主从臂 CAN", "")) as activate,
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
