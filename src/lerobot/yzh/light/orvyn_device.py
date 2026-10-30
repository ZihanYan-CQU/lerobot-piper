#!/usr/bin/env python3
import argparse
import json
import pathlib
import struct
import sys
import time
import zlib

USB_VID = 0x36B7
USB_PID = 0x2030
REPORT_SIZE = 64
REQUEST_REPORT_ID = 0x11
RESPONSE_REPORT_ID = 0x12
CHUNK_DATA_OFFSET = 7
CHUNK_MAX_LENGTH = 53

CMD_GET_RUNTIME_INFO = 0x01
CMD_LIST_APPS = 0x02
CMD_GET_APP_INFO = 0x03
CMD_GET_BOARD_INFO = 0x04
CMD_GET_BOARD_RESOURCE = 0x05
CMD_BEGIN_APP_INSTALL = 0x10
CMD_WRITE_APP_CHUNK = 0x11
CMD_END_APP_INSTALL = 0x12
CMD_REMOVE_APP = 0x13
CMD_ENABLE_APP = 0x14
CMD_DISABLE_APP = 0x15
CMD_START_APP = 0x16
CMD_STOP_APP = 0x17
CMD_GET_APP_CRASH_INFO = 0x18
CMD_GET_INSTALL_STATUS = 0x19
CMD_ABORT_APP_INSTALL = 0x1A
CMD_GET_APP_LOG = 0x1B
CMD_REBOOT = 0x21
CMD_APP_CHANNEL_WRITE = 0x30
CMD_APP_CHANNEL_READ = 0x31
CMD_PING = 0x7F

STATUS_NAMES = {
    0: "ok",
    1: "invalid command",
    2: "invalid argument",
    3: "not found",
    4: "busy",
    5: "unsupported",
    6: "CRC error",
    7: "no space",
    8: "bad state",
    9: "internal error",
}
APP_STATE_NAMES = {
    0: "empty",
    1: "installed",
    2: "ready",
    3: "running",
    4: "stopped",
    5: "crashed",
    6: "disabled",
}
BOARD_RESOURCE_TYPES = {
    0: "unknown",
    1: "rgb",
    2: "button",
    3: "buzzer",
    4: "radar",
    5: "usb",
    6: "sensor",
    7: "display",
    16: "expansion_port",
    17: "i2c_bus",
    18: "spi_bus",
    19: "uart_port",
    20: "gpio_bank",
    21: "power_rail",
}
BOARD_RESOURCE_FLAGS = {
    1 << 0: "onboard",
    1 << 1: "expansion",
    1 << 2: "shared",
    1 << 3: "optional",
    1 << 4: "debug",
}

APP_MAGIC = 0x4F525641
APP_HEADER_VERSION = 1
APP_HEADER_FORMAT = "<IHH17I16s"
APP_HEADER_SIZE = struct.calcsize(APP_HEADER_FORMAT)
APP_FLAG_POSITION_INDEPENDENT = 1 << 0
APP_CAPABILITY_NAMES = {
    1 << 0: "rgb",
    1 << 1: "buzzer",
    1 << 2: "radar",
    1 << 4: "usb_channel",
    1 << 5: "display",
    1 << 6: "input",
    1 << 7: "sensor",
    1 << 9: "i2c_bus",
    1 << 10: "power",
}


class RuntimeToolError(RuntimeError):
    pass


class RuntimeTransportError(RuntimeToolError):
    pass


class RuntimeStatusError(RuntimeToolError):
    def __init__(self, command, status):
        name = STATUS_NAMES.get(status, f"unknown status {status}")
        super().__init__(f"command 0x{command:02X} failed: {name}")
        self.command = command
        self.status = status


def u16(data, offset=0):
    return struct.unpack_from("<H", data, offset)[0]


def u32(data, offset=0):
    return struct.unpack_from("<I", data, offset)[0]


def c_string(data):
    raw = bytes(data)
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace")


def put_u16(buffer, offset, value):
    struct.pack_into("<H", buffer, offset, value)


def put_u32(buffer, offset, value):
    struct.pack_into("<I", buffer, offset, value)


def parse_number(value):
    try:
        return int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from exc


def parse_hex_bytes(value):
    compact = value.replace(" ", "").replace(":", "")
    if len(compact) % 2 != 0:
        raise argparse.ArgumentTypeError("hex data must contain complete bytes")
    try:
        return bytes.fromhex(compact)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid hex data") from exc


def display_path(path):
    if isinstance(path, bytes):
        return path.decode("utf-8", errors="backslashreplace")
    return str(path)


def import_hid():
    try:
        import hid
    except ImportError as exc:
        raise RuntimeToolError(
            "USB access requires a Python hidapi binding providing the 'hid' module"
        ) from exc
    return hid


def enumerate_devices():
    hid = import_hid()
    devices = []
    for item in hid.enumerate(USB_VID, USB_PID):
        devices.append({
            "path": display_path(item.get("path", "")),
            "serial_number": item.get("serial_number"),
            "manufacturer": item.get("manufacturer_string"),
            "product": item.get("product_string"),
            "interface_number": item.get("interface_number"),
        })
    return devices


class HidTransport:
    def __init__(self, path=None, timeout_ms=1500):
        self.path = path
        self.timeout_ms = timeout_ms
        self.hid = import_hid()
        self.device = None
        self._open()

    def _open(self):
        last_error = None
        for _ in range(5):
            device = self.hid.device()
            try:
                if self.path is None:
                    device.open(USB_VID, USB_PID)
                else:
                    raw_path = self.path.encode() if isinstance(self.path, str) else self.path
                    device.open_path(raw_path)
                self.device = device
                break
            except OSError as exc:
                last_error = exc
                try:
                    device.close()
                except OSError:
                    pass
                time.sleep(0.1)
        if self.device is None:
            raise RuntimeToolError(f"failed to open Orvyn USB HID device: {last_error}")
        self.device.set_nonblocking(0)

    def reopen(self):
        try:
            self.close()
        except OSError:
            pass
        self.device = None
        last_error = None
        for _ in range(20):
            try:
                self._open()
                return
            except RuntimeToolError as exc:
                last_error = exc
                time.sleep(0.2)
        raise RuntimeToolError(f"failed to reopen Orvyn USB HID device: {last_error}")

    def close(self):
        if self.device is not None:
            self.device.close()

    def exchange(self, request):
        try:
            written = self.device.write(request)
        except OSError as exc:
            raise RuntimeTransportError(f"HID write failed: {exc}") from exc
        if written <= 0:
            raise RuntimeToolError("failed to write HID request")
        deadline = time.monotonic() + (self.timeout_ms / 1000.0)
        while time.monotonic() < deadline:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            try:
                response = bytes(self.device.read(REPORT_SIZE, remaining_ms))
            except OSError as exc:
                raise RuntimeTransportError(f"HID read failed: {exc}") from exc
            if not response:
                continue
            if len(response) == REPORT_SIZE - 1:
                response = bytes([RESPONSE_REPORT_ID]) + response
            if len(response) != REPORT_SIZE:
                continue
            if response[0] != RESPONSE_REPORT_ID:
                continue
            if response[1] == request[1] and response[2] == request[2]:
                return response
        raise RuntimeTransportError(f"timed out waiting for command 0x{request[1]:02X}")


class RuntimeClient:
    def __init__(self, transport):
        self.transport = transport
        self.sequence = 0

    def command(self, command, payload=b"", flags=0):
        if len(payload) > 60:
            raise RuntimeToolError("command payload exceeds 60 bytes")
        request = bytearray(REPORT_SIZE)
        request[0] = REQUEST_REPORT_ID
        request[1] = command
        request[2] = self.sequence
        request[3] = flags
        request[4:4 + len(payload)] = payload
        response = self.transport.exchange(bytes(request))
        self.sequence = (self.sequence + 1) & 0xFF
        if response[3] != 0:
            raise RuntimeStatusError(command, response[3])
        return response[4:]

    def reconnect(self):
        reopen = getattr(self.transport, "reopen", None)
        if reopen is None:
            return False
        reopen()
        return True

    def runtime_info(self):
        payload = self.command(CMD_GET_RUNTIME_INFO)
        git_tag_bytes = payload[37:]
        git_tag = git_tag_bytes.split(b'\x00', 1)[0].decode('utf-8', errors='ignore').strip()
        return {
            "runtime_version": u32(payload, 0),
            "arch": u32(payload, 4),
            "machine": u32(payload, 8),
            "abi": u32(payload, 12),
            "features": u32(payload, 16),
            "flash_page_size": u32(payload, 20),
            "app_storage_size": u32(payload, 24),
            "app_ram_size": u32(payload, 28),
            "max_app_count": u32(payload, 32),
            "app_running": bool(payload[36]),
            "git_tag": git_tag if git_tag else "unknown",
        }

    def board_info(self):
        payload = self.command(CMD_GET_BOARD_INFO)
        return {
            "schema_version": payload[0],
            "board_revision_major": payload[1],
            "board_revision_minor": payload[2],
            "resource_count": payload[3],
            "board_id": u32(payload, 4),
            "capability_mask": u32(payload, 8),
            "runtime_version": u32(payload, 12),
            "logical_name": c_string(payload[16:32]),
            "display_name": c_string(payload[32:60]),
        }

    def board_resource(self, index):
        payload = self.command(CMD_GET_BOARD_RESOURCE, bytes([index]))
        flags = payload[3]
        return {
            "schema_version": payload[0],
            "index": payload[1],
            "type": payload[2],
            "type_name": BOARD_RESOURCE_TYPES.get(payload[2], f"unknown ({payload[2]})"),
            "flags": flags,
            "flag_names": [
                name for bit, name in BOARD_RESOURCE_FLAGS.items() if flags & bit
            ],
            "api_id": u32(payload, 4),
            "instance": u32(payload, 8),
            "param0": u32(payload, 12),
            "param1": u32(payload, 16),
            "param2": u32(payload, 20),
            "logical_name": c_string(payload[24:40]),
            "display_name": c_string(payload[40:60]),
        }

    def hardware_info(self):
        info = self.board_info()
        resources = [
            self.board_resource(index)
            for index in range(info["resource_count"])
        ]
        return {
            "board": info,
            "resources": resources,
        }

    def list_apps(self):
        payload = self.command(CMD_LIST_APPS)
        apps = []
        for index in range(payload[56]):
            offset = index * 8
            apps.append({
                "app_id": u32(payload, offset),
                "state": payload[offset + 4],
                "enabled": bool(payload[offset + 5]),
            })
        return apps

    def app_info(self, app_id):
        payload = self.command(CMD_GET_APP_INFO, struct.pack("<I", app_id))
        values = struct.unpack_from("<11I4B", payload)
        return {
            "app_id": values[0],
            "app_version": values[1],
            "image_address": values[2],
            "image_size": values[3],
            "stack_size": values[4],
            "heap_size": values[5],
            "data_size": values[6],
            "bss_size": values[7],
            "launch_count": values[8],
            "crash_count": values[9],
            "last_exit_code": struct.unpack("<i", struct.pack("<I", values[10]))[0],
            "state": values[11],
            "enabled": bool(values[12]),
            "auto_start": bool(values[13]),
            "slot": values[14],
        }

    def crash_info(self, app_id):
        payload = self.command(CMD_GET_APP_CRASH_INFO, struct.pack("<I", app_id))
        return {
            "crash_count": u32(payload, 0),
            "last_exit_code": struct.unpack_from("<i", payload, 4)[0],
            "launch_count": u32(payload, 8),
            "fault_type": u32(payload, 12),
            "msp": u32(payload, 16),
            "psp": u32(payload, 20),
            "lr": u32(payload, 24),
            "pc": u32(payload, 28),
            "xpsr": u32(payload, 32),
        }

    def install_status(self):
        payload = self.command(CMD_GET_INSTALL_STATUS)
        return {
            "active": bool(payload[0]),
            "slot": payload[1],
            "next_sequence": u16(payload, 2),
            "total_size": u32(payload, 4),
            "received_size": u32(payload, 8),
            "expected_crc32": u32(payload, 12),
        }

    def abort_install(self):
        self.command(CMD_ABORT_APP_INSTALL)

    def app_action(self, command, app_id):
        self.command(command, struct.pack("<I", app_id))

    def stop(self):
        self.command(CMD_STOP_APP)

    def reboot(self):
        self.command(CMD_REBOOT)

    def ping(self):
        return bytes(self.command(CMD_PING)[:4]).decode("ascii", errors="replace")

    def get_app_log(self):
        payload = self.command(CMD_GET_APP_LOG)
        length = payload[0]
        if length > 59:
            raise RuntimeToolError("device returned an invalid App log length")
        return bytes(payload[1:1 + length])

    def app_channel_write(self, command, data=b""):
        if command < 0 or command > 0xFF:
            raise RuntimeToolError("App command must fit in one byte")
        if len(data) > 56:
            raise RuntimeToolError("App command data exceeds 56 bytes")
        self.command(CMD_APP_CHANNEL_WRITE, bytes([command, len(data)]) + data)

    def app_channel_read(self):
        payload = self.command(CMD_APP_CHANNEL_READ)
        length = payload[1]
        if length > 56:
            raise RuntimeToolError("device returned an invalid App response length")
        return {
            "command": payload[0],
            "data": bytes(payload[2:2 + length]),
        }


def parse_app_package(package):
    if len(package) < APP_HEADER_SIZE:
        raise RuntimeToolError("App package is shorter than its header")
    fields = struct.unpack_from(APP_HEADER_FORMAT, package)
    names = (
        "magic", "header_version", "header_size", "app_id", "app_version",
        "flags", "required_runtime_version", "required_arch",
        "required_machine", "required_abi", "required_features", "image_size",
        "entry_offset", "data_load_offset", "data_size", "bss_size",
        "stack_size", "heap_size", "image_crc32", "required_capabilities",
        "reserved",
    )
    header = dict(zip(names, fields))
    if header["magic"] != APP_MAGIC:
        raise RuntimeToolError("invalid App package magic")
    if header["header_version"] != APP_HEADER_VERSION:
        raise RuntimeToolError("unsupported App package header version")
    if header["header_size"] != APP_HEADER_SIZE:
        raise RuntimeToolError("unsupported App package header size")
    if header["flags"] & APP_FLAG_POSITION_INDEPENDENT == 0:
        raise RuntimeToolError("App package is not position independent")
    if header["image_size"] != len(package) - APP_HEADER_SIZE:
        raise RuntimeToolError("App package image size is inconsistent")
    if header["entry_offset"] & 1 == 0:
        raise RuntimeToolError("App entry is not a Thumb address")
    if (header["entry_offset"] & ~1) >= header["image_size"]:
        raise RuntimeToolError("App entry is outside the image")
    if header["data_load_offset"] > header["image_size"]:
        raise RuntimeToolError("App data load offset is outside the image")
    if header["data_size"] > header["image_size"] - header["data_load_offset"]:
        raise RuntimeToolError("App data load range is outside the image")
    image = package[APP_HEADER_SIZE:]
    if zlib.crc32(image) & 0xFFFFFFFF != header["image_crc32"]:
        raise RuntimeToolError("App package image CRC32 mismatch")
    return header


def describe_capability_mask(mask):
    names = [
        name for bit, name in APP_CAPABILITY_NAMES.items()
        if (mask & bit) != 0
    ]
    unknown = mask & ~sum(APP_CAPABILITY_NAMES.keys())
    if unknown:
        names.append(f"unknown 0x{unknown:08X}")
    return ", ".join(names) if names else "none"


def validate_target(header, package, target, board):
    mismatches = []
    if header["required_runtime_version"] > target["runtime_version"]:
        mismatches.append("Runtime API version is too old")
    if header["required_arch"] != target["arch"]:
        mismatches.append("architecture differs")
    if header["required_machine"] != target["machine"]:
        mismatches.append("machine differs")
    if header["required_abi"] != target["abi"]:
        mismatches.append("ABI differs")
    if header["required_features"] & ~target["features"]:
        mismatches.append("required CPU features are missing")
    ram_required = (
        header["data_size"] + header["bss_size"]
        + header["stack_size"] + header["heap_size"]
    )
    if ram_required > target["app_ram_size"]:
        mismatches.append("App RAM requirement exceeds the target")
    if len(package) > target["app_storage_size"]:
        mismatches.append("package exceeds total App storage")
    missing_capabilities = header["required_capabilities"] & ~board["capability_mask"]
    if missing_capabilities:
        mismatches.append(
            "required board capabilities are missing: "
            f"{describe_capability_mask(missing_capabilities)} "
            f"(mask 0x{missing_capabilities:08X})"
        )
    if mismatches:
        raise RuntimeToolError("target is incompatible: " + "; ".join(mismatches))


def install_package(client, package, progress=None):
    header = parse_app_package(package)
    target = client.runtime_info()
    board = client.board_info()
    validate_target(header, package, target, board)
    package_crc = zlib.crc32(package) & 0xFFFFFFFF
    status = client.install_status()

    if status["active"]:
        matching = (
            status["total_size"] == len(package)
            and status["expected_crc32"] == package_crc
            and status["received_size"] <= len(package)
        )
        if not matching:
            client.abort_install()
            status = {"active": False}

    if not status["active"]:
        payload = struct.pack("<III", len(package), package_crc, header["app_id"])
        begin = client.command(CMD_BEGIN_APP_INSTALL, payload)
        slot_size = u32(begin, 4)
        if len(package) > slot_size:
            client.abort_install()
            raise RuntimeToolError(
                f"package is {len(package)} bytes but the target slot is {slot_size}"
            )
        offset = 0
        chunk_sequence = 0
    else:
        offset = status["received_size"]
        chunk_sequence = status["next_sequence"]

    while offset < len(package):
        chunk = package[offset:offset + CHUNK_MAX_LENGTH]
        payload = bytearray(CHUNK_DATA_OFFSET + len(chunk))
        put_u32(payload, 0, offset)
        put_u16(payload, 4, chunk_sequence)
        payload[6] = len(chunk)
        payload[7:] = chunk
        try:
            response = client.command(CMD_WRITE_APP_CHUNK, payload)
        except RuntimeToolError:
            recovered = client.install_status()
            expected_offset = offset + len(chunk)
            expected_sequence = (chunk_sequence + 1) & 0xFFFF
            if (
                recovered["active"]
                and recovered["total_size"] == len(package)
                and recovered["expected_crc32"] == package_crc
                and recovered["received_size"] == expected_offset
                and recovered["next_sequence"] == expected_sequence
            ):
                response = None
            else:
                raise
        expected_offset = offset + len(chunk)
        expected_sequence = (chunk_sequence + 1) & 0xFFFF
        if response is not None:
            if u32(response, 0) != expected_offset or u16(response, 4) != expected_sequence:
                raise RuntimeToolError("device acknowledged an unexpected install offset")
        offset = expected_offset
        chunk_sequence = expected_sequence
        if progress is not None:
            progress(offset, len(package))

    try:
        result = client.command(CMD_END_APP_INSTALL)
    except RuntimeTransportError:
        client.reconnect()
        info = client.app_info(header["app_id"])
        if info["app_version"] != header["app_version"]:
            raise
        installed = {
            "app_id": info["app_id"],
            "app_version": info["app_version"],
            "slot": info["slot"],
            "package_crc32": package_crc,
            "package_size": len(package),
            "commit_response": "recovered",
        }
        return installed
    installed = {
        "app_id": u32(result, 0),
        "app_version": u32(result, 4),
        "slot": result[8],
        "package_crc32": package_crc,
        "package_size": len(package),
    }
    if installed["app_id"] != header["app_id"]:
        raise RuntimeToolError("device committed a different App ID")
    return installed


KNOWN_APP_NAMES = {
    0x4F525601: "rgb_event",
    0x4F520001: "default_app",
}


def resolve_app_name(app_id):
    if app_id in KNOWN_APP_NAMES:
        return KNOWN_APP_NAMES[app_id]
    try:
        script_dir = pathlib.Path(__file__).parent
        search_roots = [script_dir.parent.parent.parent, script_dir.parent.parent, script_dir.parent]
        for root in search_roots:
            if not root.exists():
                continue
            for path in root.glob("**/manifest.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    man_id = data.get("app_id")
                    if man_id:
                        parsed_id = int(man_id, 16) if isinstance(man_id, str) and man_id.startswith("0x") else int(man_id)
                        if parsed_id == app_id:
                            if "name" in data:
                                return data["name"]
                except Exception:
                    pass
            for path in root.glob("**/*.orvyn-app.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    man_id = data.get("app_id")
                    if man_id:
                        parsed_id = int(man_id, 16) if isinstance(man_id, str) and man_id.startswith("0x") else int(man_id)
                        if parsed_id == app_id:
                            if "app_name" in data:
                                return data["app_name"]
                except Exception:
                    pass
    except Exception:
        pass
    return "Unknown"


def json_ready(value):
    if isinstance(value, dict):
        result = {}
        app_id_val = value.get("app_id")
        if app_id_val is not None:
            try:
                val_int = int(app_id_val, 16) if isinstance(app_id_val, str) and app_id_val.startswith("0x") else int(app_id_val)
                result["app_name"] = resolve_app_name(val_int)
            except Exception:
                pass
        for key, item in value.items():
            if key in (
                "app_id", "image_address", "msp", "psp", "lr", "pc", "xpsr",
                "board_id", "capability_mask",
            ):
                result[key] = f"0x{item:08X}"
            elif key == "state":
                result[key] = APP_STATE_NAMES.get(item, f"unknown ({item})")
            else:
                result[key] = json_ready(item)
        return result
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def print_json(value):
    print(json.dumps(json_ready(value), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manage Orvyn Runtime Apps over USB HID")
    parser.add_argument("--path", help="exact HID path from the devices command")
    parser.add_argument("--timeout", type=int, default=1500, help="HID timeout in milliseconds")
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("devices")
    subparsers.add_parser("info")
    subparsers.add_parser("hardware-info")
    subparsers.add_parser("list")
    subparsers.add_parser("status")
    subparsers.add_parser("abort")
    subparsers.add_parser("stop")
    subparsers.add_parser("reboot")
    subparsers.add_parser("ping")
    subparsers.add_parser("logs")
    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("package")
    app_command_parser = subparsers.add_parser("app-command")
    app_command_parser.add_argument("command", type=parse_number)
    app_command_parser.add_argument("data", nargs="?", default=b"", type=parse_hex_bytes)
    app_command_parser.add_argument(
        "--response-timeout", type=int, default=1000,
        help="milliseconds to wait for the App response",
    )
    for action in ("app-info", "crash", "start", "enable", "disable", "remove"):
        action_parser = subparsers.add_parser(action)
        action_parser.add_argument("app_id", type=parse_number)
    args = parser.parse_args(argv)

    try:
        if args.action == "devices":
            print_json(enumerate_devices())
            return 0

        transport = HidTransport(path=args.path, timeout_ms=args.timeout)
        client = RuntimeClient(transport)
        try:
            if args.action == "info":
                print_json(client.runtime_info())
            elif args.action == "hardware-info":
                print_json(client.hardware_info())
            elif args.action == "list":
                print_json(client.list_apps())
            elif args.action == "status":
                print_json(client.install_status())
            elif args.action == "abort":
                client.abort_install()
                print_json({"aborted": True})
            elif args.action == "install":
                package = pathlib.Path(args.package).read_bytes()

                def progress(done, total):
                    print(f"\rInstalling: {done}/{total} bytes", end="", file=sys.stderr)

                result = install_package(client, package, progress)
                print(file=sys.stderr)
                print_json(result)
            elif args.action == "app-info":
                print_json(client.app_info(args.app_id))
            elif args.action == "crash":
                print_json(client.crash_info(args.app_id))
            elif args.action == "start":
                client.app_action(CMD_START_APP, args.app_id)
                print_json({"started": f"0x{args.app_id:08X}"})
            elif args.action == "stop":
                client.stop()
                print_json({"stopped": True})
            elif args.action == "reboot":
                client.reboot()
                print_json({"rebooting": True})
            elif args.action == "enable":
                client.app_action(CMD_ENABLE_APP, args.app_id)
                print_json({"enabled": f"0x{args.app_id:08X}"})
            elif args.action == "disable":
                client.app_action(CMD_DISABLE_APP, args.app_id)
                print_json({"disabled": f"0x{args.app_id:08X}"})
            elif args.action == "remove":
                client.app_action(CMD_REMOVE_APP, args.app_id)
                print_json({"removed": f"0x{args.app_id:08X}"})
            elif args.action == "ping":
                print_json({"reply": client.ping()})
            elif args.action == "logs":
                print("Streaming device logs... Press Ctrl+C to stop.", file=sys.stderr)
                try:
                    while True:
                        log_data = client.get_app_log()
                        if log_data:
                            sys.stdout.write(log_data.decode("utf-8", errors="replace"))
                            sys.stdout.flush()
                        else:
                            time.sleep(0.05)
                except KeyboardInterrupt:
                    pass
            elif args.action == "app-command":
                client.app_channel_write(args.command, args.data)
                deadline = time.monotonic() + (args.response_timeout / 1000.0)
                while True:
                    try:
                        message = client.app_channel_read()
                        print_json({
                            "command": f"0x{message['command']:02X}",
                            "data_hex": message["data"].hex(),
                        })
                        break
                    except RuntimeStatusError as exc:
                        if exc.status != 3 or time.monotonic() >= deadline:
                            raise
                        time.sleep(0.01)
        finally:
            transport.close()
        return 0
    except (OSError, RuntimeToolError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
