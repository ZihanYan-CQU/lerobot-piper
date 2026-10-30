"""Hardware-independent regression tests for the standalone camera alias helper.

Run directly with Python; neither pytest nor LeRobot's ML dependencies are needed.
"""

import errno
import importlib.util
import io
import re
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from fnmatch import fnmatchcase
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "camera_aliases.py"
SPEC = importlib.util.spec_from_file_location("camera_aliases", SCRIPT)
aliases = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = aliases
SPEC.loader.exec_module(aliases)


def camera(node="video4", serial="wrist123", interface="03", index="0", capture=True, port="2.2"):
    return aliases.Camera(
        node=node,
        name="Test camera",
        index=index,
        properties={
            "ID_SERIAL": f"TestCamera_{serial}",
            "ID_SERIAL_SHORT": serial,
            "ID_USB_INTERFACE_NUM": interface,
            "ID_PATH": f"pci-0000:24:00.3-usb-0:{port}:1.{int(interface)}",
            "ID_V4L_CAPABILITIES": ":capture:" if capture else ":",
        },
    )


def matches_event(rule, device):
    """Evaluate the emitted equality predicates independently of identity selection."""
    values = {
        "ACTION": "add",
        "SUBSYSTEM": "video4linux",
        "KERNEL": device.node,
        "ATTR{index}": device.index,
    }
    values.update({f"ENV{{{key}}}": value for key, value in device.properties.items()})
    for key, pattern in re.findall(r'([A-Z]+(?:\{[^}]+\})?)=="([^"]+)"', rule):
        if not any(fnmatchcase(values.get(key, ""), option) for option in pattern.split("|")):
            return False
    return True


class CameraAliasesTest(unittest.TestCase):
    def setUp(self):
        self.wrist = camera()
        self.top = camera("video10", "top456", port="2.3")
        self.depth = camera("video0", interface="00")
        self.metadata = camera("video5", index="1", capture=False)
        self.cameras = [self.wrist, self.top, self.depth, self.metadata]
        self.mappings = [("video4", "video_wrist"), ("video10", "video_top")]

    def test_rules_follow_serial_after_renumbering_and_distinguish_streams(self):
        content = aliases.make_rules(self.cameras, self.mappings, "auto")
        rules = [line for line in content.splitlines() if not line.startswith("#")]
        wrist_moved = camera("video12", port="8.1")
        top_moved = camera("video2", "top456", port="8.2")
        self.assertTrue(matches_event(rules[0], wrist_moved))
        self.assertTrue(matches_event(rules[1], top_moved))
        for wrong_device in (self.depth, self.metadata, top_moved):
            self.assertFalse(matches_event(rules[0], wrong_device))
        self.assertFalse(matches_event(rules[1], wrist_moved))
        self.assertIn('SYMLINK+="video_wrist"', rules[0])
        self.assertIn('SYMLINK+="video_top"', rules[1])

    def test_duplicate_serials_fall_back_to_unique_ports(self):
        duplicate = camera("video10", port="2.3")
        devices = [self.wrist, duplicate]
        content = aliases.make_rules(devices, self.mappings, "auto")
        rules = [line for line in content.splitlines() if not line.startswith("#")]
        self.assertTrue(matches_event(rules[0], self.wrist))
        self.assertFalse(matches_event(rules[0], duplicate))
        self.assertNotIn("ENV{ID_SERIAL}", content)
        with self.assertRaisesRegex(ValueError, "无法唯一识别"):
            aliases.make_rules(devices, self.mappings, "serial")

    def test_missing_serial_falls_back_to_port(self):
        props = dict(self.wrist.properties)
        del props["ID_SERIAL_SHORT"]
        device = replace(self.wrist, properties=props)
        content = aliases.make_rules([device], self.mappings[:1], "auto")
        self.assertIn("ENV{ID_PATH}", content)

    def test_indistinguishable_devices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "无法唯一识别"):
            aliases.make_rules([self.wrist, replace(self.wrist, node="video10")], self.mappings, "auto")

    def test_metadata_missing_device_and_duplicate_alias_are_rejected(self):
        for mapping in (
            [("video5", "video_meta")],
            [("video99", "video_unknown")],
            [("video4", "video_wrist"), ("video10", "video_wrist")],
        ):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                aliases.make_rules(self.cameras, mapping, "auto")

    def test_mapping_formats_and_unsafe_names(self):
        for value in ("4=video_wrist", "video4=video_wrist", "/dev/video4=/dev/video_wrist"):
            self.assertEqual(aliases.parse_mapping(value), ("video4", "video_wrist"))
        for value in (
            "4", "4=video10", "4=../video_wrist", "4=/tmp/video_wrist", "4=video_foo/bar", "../4=video_x"
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                aliases.parse_mapping(value)

    def test_rule_injection_or_wildcards_in_identity_are_rejected(self):
        for serial in ('bad"value', "*", "foo|bar", "foo\nbar"):
            device = camera(serial=serial)
            with self.subTest(serial=serial), self.assertRaises(ValueError):
                aliases.make_rules([device], self.mappings[:1], "serial")

    def test_dry_run_and_list_never_apply_changes(self):
        with (
            patch.object(aliases, "discover_cameras", return_value=self.cameras),
            patch.object(aliases, "check_conflicts"),
            patch.object(aliases, "apply_rules") as apply,
            redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(aliases.main(["--dry-run"]), 0)
            self.assertEqual(aliases.main(["--list"]), 0)
            apply.assert_not_called()
            self.assertIn('SYMLINK+="video_top"', output.getvalue())
            self.assertIn("非采集/元数据", output.getvalue())

    def test_invalid_mapping_does_not_apply_changes(self):
        with (
            patch.object(aliases, "discover_cameras", return_value=self.cameras),
            patch.object(aliases, "apply_rules") as apply,
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(aliases.main(["--map", "99=video_front"]), 1)
            apply.assert_not_called()

    def test_rule_backups_and_unmanaged_file_protection(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / "99-lerobot-cameras.rules"
            with patch.object(aliases, "RULES_FILE", path):
                original = aliases.MARKER + "# first mapping\n"
                updated = aliases.MARKER + "# second mapping\n"
                aliases.write_rules(original)
                aliases.write_rules(updated)
                aliases.write_rules(updated)
                backups = list(path.parent.glob("*.bak-*"))
                self.assertEqual(len(backups), 1)
                self.assertEqual(backups[0].read_text(), original)
                self.assertEqual(path.read_text(), updated)
                self.assertEqual(path.stat().st_mode & 0o777, 0o644)
                path.write_text("# manually maintained\n")
                with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
                    aliases.write_rules(updated)
                self.assertEqual(path.read_text(), "# manually maintained\n")

    def test_failed_atomic_replace_preserves_previous_rules(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / "99-lerobot-cameras.rules"
            original = aliases.MARKER + "# old\n"
            path.write_text(original)
            with (
                patch.object(aliases, "RULES_FILE", path),
                patch.object(aliases.os, "replace", side_effect=OSError("full")),
            ):
                with self.assertRaises(OSError):
                    aliases.write_rules(aliases.MARKER + "# new\n")
            self.assertEqual(path.read_text(), original)
            self.assertEqual(list(path.parent.glob(".camera-aliases-*")), [])

    def test_conflicting_rule_and_non_symlink_alias_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rule = root / "99-other.rules"
            with patch.object(aliases, "DEV_DIR", root), patch.object(aliases, "RULES_DIRS", (root,)):
                rule.write_text('# SYMLINK+="video_wrist"\n')
                aliases.check_conflicts(self.mappings)
                rule.write_text('SUBSYSTEM=="video4linux", SYMLINK+="video_wrist video_front"\n')
                with self.assertRaisesRegex(ValueError, "已经定义"):
                    aliases.check_conflicts(self.mappings)
                rule.write_text("# empty\n")
                (root / "video_wrist").write_text("do not overwrite")
                with self.assertRaisesRegex(ValueError, "不是软链接"):
                    aliases.check_conflicts(self.mappings)

    def test_apply_requires_root_before_writing(self):
        with (
            patch.object(aliases.os, "geteuid", return_value=1000),
            patch.object(aliases, "write_rules") as write,
        ):
            with self.assertRaises(PermissionError):
                aliases.apply_rules("rules", self.mappings)
            write.assert_not_called()

    def test_apply_reloads_and_verifies_links_and_reports_activation_failure(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            # Real harmless character devices stand in for video nodes; all writes stay in this tempdir.
            (root / "video4").symlink_to("/dev/null")
            rule_path = root / "99-lerobot-cameras.rules"
            content = aliases.make_rules(self.cameras, self.mappings[:1], "auto")

            def event(*args):
                if args[0] == "trigger":
                    (root / "video_wrist").symlink_to(root / "video4")
                return ""

            with (
                patch.object(aliases, "DEV_DIR", root),
                patch.object(aliases, "RULES_FILE", rule_path),
                patch.object(aliases.os, "geteuid", return_value=0),
                patch.object(aliases, "udevadm", side_effect=event) as udev,
            ):
                aliases.apply_rules(content, self.mappings[:1])
                self.assertEqual([call.args[0] for call in udev.call_args_list], ["control", "trigger", "settle"])
                self.assertEqual(rule_path.read_text(), content)
                (root / "video_wrist").unlink()
                udev.side_effect = None
                with self.assertRaisesRegex(RuntimeError, "规则已保存，但立即生效验证失败"):
                    aliases.apply_rules(content, self.mappings[:1])

    def test_rgb_filter_uses_sensor_interface_and_excludes_metadata(self):
        rgb = replace(self.wrist, interface="Intel RealSense Depth Camera 435i RGB")
        depth = replace(self.depth, interface="Intel RealSense Depth Camera 435i Depth")
        metadata = replace(self.metadata, interface=rgb.interface)
        infrared = replace(self.top, interface="Infrared Camera")
        with patch.object(aliases, "capture_formats") as probe:
            devices, errors = aliases.filter_rgb_cameras([rgb, depth, metadata, infrared])
        self.assertEqual([device.node for device in devices], ["video4"])
        self.assertEqual(errors, [])
        probe.assert_not_called()

    def test_generic_camera_requires_color_formats_and_unreadable_nodes_stay_hidden(self):
        with patch.object(aliases, "capture_formats", return_value={"MJPG", "YUYV"}):
            devices, _ = aliases.filter_rgb_cameras([self.wrist])
            self.assertEqual(devices, [self.wrist])
        with patch.object(aliases, "capture_formats", return_value={"Z16 ", "GREY", "Y16 "}):
            devices, _ = aliases.filter_rgb_cameras([self.wrist])
            self.assertEqual(devices, [])
        with patch.object(aliases, "capture_formats", side_effect=PermissionError("permission denied")):
            devices, errors = aliases.filter_rgb_cameras([self.wrist])
            self.assertEqual(devices, [])
            self.assertIn("无法确认彩色格式", errors[0])

    def test_v4l2_format_query_supports_multiplane_and_closes_descriptor(self):
        def enum_format(fd, request, buffer, mutate):
            self.assertEqual(fd, 123)
            self.assertEqual(request, 0xC0405602)
            self.assertTrue(mutate)
            self.assertEqual(len(buffer), 64)
            index, buffer_type = struct.unpack_from("=II", buffer)
            if index != 0:
                raise OSError(errno.EINVAL, "end of formats")
            fourcc = b"YUYV" if buffer_type == 1 else b"NV12"
            struct.pack_into("=I", buffer, 44, int.from_bytes(fourcc, "little"))

        with (
            patch.object(aliases.os, "open", return_value=123),
            patch.object(aliases.os, "close") as close,
            patch.object(aliases.fcntl, "ioctl", side_effect=enum_format),
        ):
            self.assertEqual(aliases.capture_formats("video4"), {"YUYV", "NV12"})
            close.assert_called_once_with(123)

    def test_selection_identity_cannot_silently_follow_a_reused_video_number(self):
        device = replace(self.wrist, interface="RGB")
        expected = [f"video4={aliases.identity_token(device)}"]
        aliases.validate_selections([device], self.mappings[:1], expected)
        other_camera = replace(self.top, node="video4", interface="RGB")
        with self.assertRaisesRegex(ValueError, "设备已变化"):
            aliases.validate_selections([other_camera], self.mappings[:1], expected)
        with self.assertRaisesRegex(ValueError, "不完整"):
            aliases.validate_selections([device], self.mappings[:1], [])
        with self.assertRaisesRegex(ValueError, "参数无效"):
            aliases.validate_selections([device], self.mappings[:1], expected + expected)

    def test_selection_cannot_save_non_rgb_device_even_with_matching_identity(self):
        depth = replace(self.depth, interface="Depth")
        with self.assertRaisesRegex(ValueError, "RGB"):
            aliases.validate_selections(
                [depth], [("video0", "video_wrist")], [f"video0={aliases.identity_token(depth)}"]
            )

    def test_list_rgb_never_displays_depth_or_metadata_nodes(self):
        devices = [
            replace(self.wrist, interface="RGB"),
            replace(self.top, interface="RGB"),
            replace(self.depth, interface="Depth"),
            replace(self.metadata, interface="RGB"),
        ]
        with patch.object(aliases, "discover_cameras", return_value=devices), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(aliases.main(["--list-rgb"]), 0)
        self.assertIn("/dev/video4", output.getvalue())
        self.assertIn("/dev/video10", output.getvalue())
        self.assertNotIn("/dev/video0 ", output.getvalue())
        self.assertNotIn("/dev/video5 ", output.getvalue())


if __name__ == "__main__":
    unittest.main()
