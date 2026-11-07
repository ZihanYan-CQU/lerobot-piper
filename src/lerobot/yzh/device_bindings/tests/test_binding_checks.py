"""Exercise live binding reports using temporary symlinks and CAN snapshots."""

import importlib.util
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
for name in ("camera_aliases", "can_port_bindings", "binding_checks"):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

checks = sys.modules["binding_checks"]
cameras, can = checks.cameras, checks.can


def camera(node, serial):
    return cameras.Camera(node, "RGB", "0", {
        "ID_SERIAL": serial, "ID_SERIAL_SHORT": serial, "ID_USB_INTERFACE_NUM": "03",
        "ID_V4L_CAPABILITIES": ":capture:", "ID_PATH": f"pci-test-{serial}",
    }, "RGB")


class CameraChecksTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.rules = self.root / "cameras.rules"
        self.devices = [camera("video4", "123"), camera("video10", "456")]
        self.mappings = [("video4", "video_wrist"), ("video10", "video_top")]
        (self.root / "video4").symlink_to("/dev/null")
        (self.root / "video10").symlink_to("/dev/zero")
        for node, alias in self.mappings:
            (self.root / alias).symlink_to(node)
        self.rules.write_text(cameras.make_rules(self.devices, self.mappings, "auto"))
        for name, value in (("DEV_DIR", self.root), ("RULES_FILE", self.rules)):
            patcher = patch.object(cameras, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def inspect(self):
        return checks.check_cameras(self.devices, self.mappings, self.devices)

    def test_correct_live_links_and_saved_rules_pass_without_modification(self):
        original = self.rules.read_text()
        with patch.object(cameras, "apply_rules") as install:
            results = self.inspect()
        self.assertEqual([item.ok for item in results], [True, True])
        self.assertIn("持久化规则已保存", results[0].detail)
        self.assertEqual(self.rules.read_text(), original)
        install.assert_not_called()

    def test_missing_wrong_and_dangling_aliases_are_reported_separately(self):
        link = self.root / "video_wrist"
        link.unlink()
        self.assertIn("尚未绑定", self.inspect()[0].detail)
        link.symlink_to("video10")
        result = self.inspect()[0]
        self.assertFalse(result.ok)
        self.assertIn("不一致", result.detail)
        link.unlink()
        link.symlink_to("video99")
        self.assertIn("目标已断开", self.inspect()[0].detail)

    def test_live_alias_without_persistent_rules_is_not_a_complete_binding(self):
        self.rules.unlink()
        result = self.inspect()[0]
        self.assertFalse(result.ok)
        self.assertIn("当前链接正确", result.detail)
        self.assertIn("持久化规则未匹配", result.detail)

    def test_offline_or_replaced_camera_does_not_reuse_an_old_pass(self):
        result = checks.check_cameras(self.devices, self.mappings, self.devices[1:])
        self.assertFalse(result[0].ok)
        self.assertIn("已断开", result[0].detail)
        replacement = camera("video4", "999")
        result = checks.check_cameras(self.devices, self.mappings, [replacement, self.devices[1]])
        self.assertIn("设备已变化", result[0].detail)

    def test_unselected_and_duplicate_rows_cannot_report_all_bound(self):
        result = checks.check_cameras(self.devices, [(None, "video_wrist")], self.devices)
        self.assertFalse(result[0].ok)
        for mappings in (
            [("video4", "video_wrist"), ("video4", "video_wrist")],
            [("video4", "video_wrist"), ("video4", "video_top")],
        ):
            with self.subTest(mappings=mappings):
                self.assertTrue(all(not item.ok for item in checks.check_cameras(self.devices, mappings, self.devices)))

    def test_path_rules_and_video_renumbering_remain_valid(self):
        self.rules.write_text(cameras.make_rules(self.devices, self.mappings, "path"))
        self.assertTrue(self.inspect()[0].ok)
        renamed = replace(self.devices[0], node="video8")
        (self.root / "video8").symlink_to("/dev/null")
        (self.root / "video_wrist").unlink()
        (self.root / "video_wrist").symlink_to("video8")
        result = checks.check_cameras([renamed], [("video8", "video_wrist")], [renamed])
        self.assertTrue(result[0].ok)


class CanChecksTest(unittest.TestCase):
    def setUp(self):
        self.port = can.CanPort("can_follower", "1-9.3:1.0", 7, "123", True, 1000000)

    def test_single_arm_passes_and_unselected_role_is_not_counted_as_checked(self):
        with patch.object(can, "apply_bindings") as activate, patch.object(can, "run") as command:
            results = checks.check_can([self.port], None, self.port.usb, 1000000, [self.port])
        self.assertIsNone(results[0].ok)
        self.assertTrue(results[1].ok)
        self.assertIn("UP", results[1].detail)
        activate.assert_not_called()
        command.assert_not_called()

    def test_name_bitrate_and_down_state_mismatches_are_all_visible(self):
        current = replace(self.port, name="can0", up=False, bitrate=500000)
        result = checks.check_can([self.port], None, self.port.usb, 1000000, [current])[1]
        self.assertFalse(result.ok)
        for text in ("接口名应为 can_follower", "波特率应为 1000000", "尚未启用", "DOWN"):
            self.assertIn(text, result.detail)

    def test_disconnected_replaced_and_unreadable_ports_are_not_ready(self):
        for current, detail in (
            ([], "已断开"),
            ([replace(self.port, ifindex=9)], "设备已变化"),
            ([replace(self.port, bitrate=None)], "未读取"),
        ):
            with self.subTest(current=current):
                result = checks.check_can([self.port], None, self.port.usb, 1000000, current)[1]
                self.assertFalse(result.ok)
                self.assertIn(detail, result.detail)

    def test_reopening_recognizes_existing_role_names_on_different_usb_ports(self):
        self.assertEqual(checks.default_can_selections([self.port]), (None, self.port.usb))
        swapped = replace(self.port, usb=can.DEFAULT_MASTER)
        self.assertEqual(checks.default_can_selections([swapped]), (None, can.DEFAULT_MASTER))


if __name__ == "__main__":
    unittest.main()
