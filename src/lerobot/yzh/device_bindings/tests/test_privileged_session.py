"""Test authorization reuse and device validation without privileges or hardware."""

import importlib.util
import io
import json
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
for name in ("camera_aliases", "can_port_bindings", "privileged_session"):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

helper = sys.modules["privileged_session"]
can = helper.can
cameras = helper.cameras

ECHO_HELPER = '''
import json
import os
import sys
for line in sys.stdin:
    request = json.loads(line)
    if request.get("fail"):
        reply = {"ok": False, "error": "invalid selection"}
    else:
        reply = {"ok": True, "output": f'{os.getpid()}:{request["operation"]}'}
    print(json.dumps(reply), flush=True)
'''


class PrivilegedSessionTest(unittest.TestCase):
    def test_both_tabs_and_repeated_saves_share_one_lazily_authorized_process(self):
        native_popen = subprocess.Popen
        with (
            patch.object(helper, "helper_command", return_value=["/usr/bin/python3", "-u", "-c", ECHO_HELPER]) as authorize,
            patch.object(helper.subprocess, "Popen", wraps=native_popen) as start,
        ):
            session = helper.PrivilegedSession(timeout=5)
            start.assert_not_called()
            try:
                camera = session.execute({"operation": "camera"})
                arm = session.execute({"operation": "can"})
                again = session.execute({"operation": "camera"})
                self.assertEqual(camera.split(":")[0], arm.split(":")[0])
                self.assertEqual(camera, again)
                authorize.assert_called_once()
                start.assert_called_once()
                process = session.process
            finally:
                session.close()
            self.assertEqual(process.poll(), 0)
            self.assertIsNone(session.process)
            with self.assertRaisesRegex(RuntimeError, "关闭"):
                session.execute({"operation": "camera"})

    def test_validation_failure_does_not_discard_a_valid_authorization(self):
        with patch.object(helper, "helper_command", return_value=["/usr/bin/python3", "-u", "-c", ECHO_HELPER]) as authorize:
            session = helper.PrivilegedSession(timeout=5)
            try:
                before = session.execute({"operation": "can"})
                with self.assertRaisesRegex(RuntimeError, "invalid selection"):
                    session.execute({"fail": True})
                self.assertEqual(session.execute({"operation": "can"}), before)
                authorize.assert_called_once()
            finally:
                session.close()

    def test_cancelled_authorization_is_reported_without_retrying_the_operation(self):
        with patch.object(helper, "helper_command", return_value=["/usr/bin/python3", "-c", "raise SystemExit(126)"]) as authorize:
            session = helper.PrivilegedSession(timeout=5)
            try:
                with self.assertRaisesRegex(RuntimeError, "取消系统授权"):
                    session.execute({"operation": "can"})
                authorize.assert_called_once()
            finally:
                session.close()

    def test_stalled_helper_times_out_without_replaying_a_request(self):
        with patch.object(helper, "helper_command", return_value=["/usr/bin/python3", "-c", "import time; time.sleep(0.2)"]) as authorize:
            session = helper.PrivilegedSession(timeout=0.01)
            try:
                with self.assertRaisesRegex(RuntimeError, "结果未确认"):
                    session.execute({"operation": "can"})
                authorize.assert_called_once()
            finally:
                session.close()

    def test_only_the_helper_is_elevated_and_root_needs_no_second_authorization(self):
        with (
            patch.object(helper.os, "geteuid", return_value=1000),
            patch.object(helper.shutil, "which", return_value="/usr/bin/pkexec"),
        ):
            self.assertEqual(helper.helper_command()[:2], ["/usr/bin/pkexec", "/usr/bin/python3"])
        with patch.object(helper.os, "geteuid", return_value=0):
            self.assertEqual(helper.helper_command()[0], "/usr/bin/python3")

    def test_pipe_errors_do_not_prevent_later_requests_and_eof_ends_the_server(self):
        source = io.StringIO('invalid-json\n{"operation": "camera"}\n')
        output = io.StringIO()
        with patch.object(helper.os, "geteuid", return_value=0), patch.object(helper, "dispatch", return_value="verified") as dispatch:
            helper.serve(source, output)
        replies = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertFalse(replies[0]["ok"])
        self.assertEqual(replies[1], {"ok": True, "output": "verified"})
        dispatch.assert_called_once_with({"operation": "camera"})


class DispatchTest(unittest.TestCase):
    def setUp(self):
        self.port = can.CanPort("can0", "1-1.3:1.0", 8, "follower", False)
        self.request = {
            "operation": "can", "master": None, "follower": self.port.usb, "bitrate": 1000000,
            "expected": [f"{self.port.usb}={can.identity_token(self.port)}"],
        }

    def test_single_arm_request_rechecks_identity_and_does_not_add_default_master(self):
        with (
            patch.object(can, "discover_ports", return_value=[self.port]),
            patch.object(can, "apply_bindings") as activate,
        ):
            helper.dispatch(self.request)
            activate.assert_called_once_with([(self.port, "can_follower")], 1000000)

    def test_device_replacement_after_authorization_is_rejected_before_writes(self):
        with (
            patch.object(can, "discover_ports", return_value=[replace(self.port, ifindex=9)]),
            patch.object(can, "apply_bindings") as activate,
        ):
            with self.assertRaisesRegex(RuntimeError, "设备已变化"):
                helper.dispatch(self.request)
            activate.assert_not_called()

    def test_camera_request_uses_the_existing_rgb_and_rule_validators(self):
        camera = cameras.Camera("video4", "RGB", "0", {
            "ID_SERIAL": "123", "ID_SERIAL_SHORT": "123", "ID_USB_INTERFACE_NUM": "03",
            "ID_V4L_CAPABILITIES": ":capture:",
        }, "RGB")
        request = {
            "operation": "camera", "mappings": ["video4=video_wrist"],
            "expected": [f"video4={cameras.identity_token(camera)}"],
        }
        with (
            patch.object(cameras, "discover_cameras", return_value=[camera]),
            patch.object(cameras, "check_conflicts"),
            patch.object(cameras, "apply_rules") as install,
        ):
            helper.dispatch(request)
            self.assertIn('SYMLINK+="video_wrist"', install.call_args.args[0])
            self.assertEqual(install.call_args.args[1], [("video4", "video_wrist")])
            request["expected"] = ["video4=" + "0" * 64]
            install.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "设备已变化"):
                helper.dispatch(request)
            install.assert_not_called()

    def test_missing_identities_empty_roles_and_general_commands_are_rejected(self):
        for request in (
            {**self.request, "expected": []},
            {**self.request, "follower": None},
            {**self.request, "bitrate": True},
            {**self.request, "operation": "shell", "command": "whoami"},
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                helper.dispatch(request)


if __name__ == "__main__":
    unittest.main()
