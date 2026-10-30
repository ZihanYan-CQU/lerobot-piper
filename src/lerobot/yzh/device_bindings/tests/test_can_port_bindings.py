"""Exercise CAN binding, role swaps, and recovery without touching network interfaces."""

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "can_port_bindings.py"
SPEC = importlib.util.spec_from_file_location("can_port_bindings", SCRIPT)
can = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = can
SPEC.loader.exec_module(can)


class CanBindingsTest(unittest.TestCase):
    def setUp(self):
        self.master = can.CanPort("can0", "1-1.2:1.0", 7, "master-serial", False, 500000)
        self.follower = can.CanPort("can_follower", "1-1.3:1.0", 8, "follower-serial", True, 1000000)
        self.ports = [self.master, self.follower]

    def test_default_bindings_and_invalid_selection(self):
        self.assertEqual(
            can.validate_bindings(self.ports, can.DEFAULT_MASTER, can.DEFAULT_FOLLOWER, 1000000),
            [(self.master, "can_master"), (self.follower, "can_follower")],
        )
        for master, follower, bitrate in (
            (self.master.usb, self.master.usb, 1000000),
            ("1-1.9:1.0", self.follower.usb, 1000000),
            (self.master.usb, self.follower.usb, 0),
            (self.master.usb, self.follower.usb, 2000000),
        ):
            with self.subTest(master=master, follower=follower, bitrate=bitrate), self.assertRaises(ValueError):
                can.validate_bindings(self.ports, master, follower, bitrate)

    def test_reconnected_adapter_is_rejected_even_at_the_same_usb_port(self):
        bindings = [(self.master, "can_master"), (self.follower, "can_follower")]
        tokens = [f"{port.usb}={can.identity_token(port)}" for port, _ in bindings]
        can.validate_expected(bindings, tokens)
        with self.assertRaisesRegex(ValueError, "设备已变化"):
            can.validate_expected([(replace(self.master, ifindex=12), "can_master"), bindings[1]], tokens)
        with self.assertRaisesRegex(ValueError, "不完整"):
            can.validate_expected(bindings, tokens[:1])

    def test_discovery_only_lists_usb_can_interfaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            net = root / "net"
            net.mkdir()
            device = root / "usb" / "1-1.2" / "1-1.2:1.0"
            device.mkdir(parents=True)
            (device.parent / "serial").write_text("master-serial")
            for name, kind in (("can0", "280"), ("ether0", "1"), ("vcan0", "280")):
                path = net / name
                path.mkdir()
                (path / "type").write_text(kind)
                (path / "ifindex").write_text("7")
                (path / "flags").write_text("0x1")
                if name != "vcan0":
                    (path / "device").symlink_to(device)
            details = {"can0": {"linkinfo": {"info_data": {"bittiming": {"bitrate": 1000000}}}}}
            with patch.object(can, "SYS_NET", net), patch.object(can, "link_details", return_value=details):
                ports = can.discover_ports()
            self.assertEqual(len(ports), 1)
            self.assertEqual(ports[0].usb, "1-1.2:1.0")
            self.assertEqual(ports[0].bitrate, 1000000)
            self.assertTrue(ports[0].up)

    def exercise_activation(self, ports, *, fail_second=False, false_success=False):
        state = {port.usb: port for port in ports}
        calls = []
        scripts = 0

        def command(*args):
            nonlocal scripts
            calls.append(args)
            if args[0] == "dpkg-query":
                return "install ok installed"
            if args[0] == "bash":
                scripts += 1
                if fail_second and scripts == 2:
                    raise RuntimeError("activation failed")
                if not false_success:
                    _, _, target, bitrate, usb = args
                    state[usb] = replace(state[usb], name=target, up=True, bitrate=int(bitrate))
                return "script claims success"
            self.assertEqual(args[:4], ("ip", "link", "set", "dev"))
            port = next(port for port in state.values() if port.name == args[4])
            action = args[5]
            if action == "name":
                self.assertFalse(any(other.name == args[6] and other.usb != port.usb for other in state.values()))
                port = replace(port, name=args[6])
            elif action == "type":
                port = replace(port, bitrate=int(args[-1]))
            else:
                self.assertIn(action, ("up", "down"))
                port = replace(port, up=action == "up")
            state[port.usb] = port
            return ""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "can_activate.sh"
            script.write_text("# fixture")
            with (
                patch.object(can, "SYS_NET", root),
                patch.object(can, "ACTIVATE_SCRIPT", script),
                patch.object(can.os, "geteuid", return_value=0),
                patch.object(can.shutil, "which", return_value="/usr/bin/tool"),
                patch.object(can, "find_current", side_effect=lambda original: state[original.usb]),
                patch.object(can, "run", side_effect=command),
                redirect_stdout(io.StringIO()),
            ):
                bindings = [(ports[0], "can_master"), (ports[1], "can_follower")]
                if fail_second or false_success:
                    with self.assertRaisesRegex(RuntimeError, "CAN 绑定失败"):
                        can.apply_bindings(bindings, 1000000)
                else:
                    can.apply_bindings(bindings, 1000000)
        return state, calls

    def test_activation_uses_the_requested_script_arguments_and_skips_unchanged_port(self):
        state, calls = self.exercise_activation(self.ports)
        script_calls = [call for call in calls if call[0] == "bash"]
        self.assertEqual(len(script_calls), 1)
        self.assertEqual(script_calls[0][-3:], ("can_master", "1000000", "1-1.2:1.0"))
        self.assertEqual(state[self.master.usb].name, "can_master")
        self.assertTrue(state[self.master.usb].up)

    def test_swapping_master_and_follower_frees_names_first(self):
        ports = [replace(self.master, name="can_follower", up=True), replace(self.follower, name="can_master")]
        state, _ = self.exercise_activation(ports)
        self.assertEqual(state[ports[0].usb].name, "can_master")
        self.assertEqual(state[ports[1].usb].name, "can_follower")

    def test_second_activation_failure_restores_original_names_rates_and_states(self):
        ports = [self.master, replace(self.follower, name="can1", bitrate=500000)]
        state, _ = self.exercise_activation(ports, fail_second=True)
        self.assertEqual(list(state.values()), ports)

    def test_zero_exit_without_correct_kernel_state_is_not_success(self):
        state, _ = self.exercise_activation(self.ports, false_success=True)
        self.assertEqual(list(state.values()), self.ports)

    def test_dry_run_never_activates_ports(self):
        with (
            patch.object(can, "discover_ports", return_value=self.ports),
            patch.object(can, "apply_bindings") as activate,
            redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(can.main(["--dry-run"]), 0)
            activate.assert_not_called()
            self.assertIn('can_master 1000000 "1-1.2:1.0"', output.getvalue())
            self.assertIn('can_follower 1000000 "1-1.3:1.0"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
