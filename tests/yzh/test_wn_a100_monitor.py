from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from lerobot.yzh.wn_a100.monitor import (
    MonitorError,
    build_detail_url,
    format_notice_deadline,
    format_process_command,
    parse_announcements,
    parse_capacity_gib,
    parse_snapshot,
    snapshot_as_dict,
    user_color,
)


def sample_payload() -> dict:
    return {
        "processes": [
            {"device_idx": 0, "username": "yzh", "gpu_memory": 1024},
            {"device_idx": 0, "username": "yzh", "gpu_memory": 512},
            {
                "device_idx": "0, 1",
                "username": "bob",
                "gpu_memory_human": "2GiB",
            },
        ],
        "status": {
            "timestamp": 1_790_250_216_000,
            "status": [
                {
                    "device_idx": 0,
                    "gpu_utilization": 97,
                    "memory_percent": 89.7,
                    "memory_used_human": "71.75GiB",
                    "memory_total_human": "80.00GiB",
                    "temperature": 58,
                    "power_usage": 280.558,
                    "power_status": "281W / 400W",
                },
                {
                    "device_idx": 1,
                    "gpu_utilization": 0,
                    "memory_free_human": "6560MiB",
                    "memory_usage": "72.84GiB / 80.00GiB",
                    "temperature": 38,
                    "power_status": "70W / 400W",
                },
            ],
        },
        "system_status": {
            "cpu_percent": 7.1,
            "memory_percent": 11.0,
            "disk_usage_percent": 76.19,
        },
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("80.00GiB", 80.0),
        ("6560MiB", 6.40625),
        (81920, 80.0),
        ("1048576KiB", 1.0),
    ],
)
def test_parse_capacity_gib(raw: object, expected: float) -> None:
    assert parse_capacity_gib(raw) == pytest.approx(expected)


def test_parse_snapshot_normalizes_gpu_and_process_data() -> None:
    snapshot = parse_snapshot(sample_payload())

    assert [gpu.index for gpu in snapshot.gpus] == [0, 1]
    assert snapshot.gpus[0].utilization == 97
    assert snapshot.gpus[0].memory_used_gib == pytest.approx(71.75)
    assert snapshot.gpus[0].power_limit_w == 400
    assert snapshot.gpus[0].process_count == 3
    assert snapshot.gpus[0].users == ("bob", "yzh")
    assert snapshot.gpus[0].user_usages[0].memory_gib == 2
    assert snapshot.gpus[0].user_usages[1].memory_gib == 1.5
    assert snapshot.gpus[0].user_usages[1].process_count == 2
    assert snapshot.gpus[0].has_user("YZH")
    assert snapshot.gpus[1].memory_total_gib == 80
    assert snapshot.gpus[1].process_count == 1
    assert snapshot.cpu_percent == 7.1


def test_snapshot_json_shape_is_stable() -> None:
    data = snapshot_as_dict(parse_snapshot(sample_payload()))

    assert data["server_id"] == "server7"
    assert data["system"]["memory_percent"] == 11.0
    assert data["gpus"][0]["users"] == ["bob", "yzh"]
    assert data["gpus"][0]["user_usages"][1] == {
        "username": "yzh",
        "memory_gib": 1.5,
        "process_count": 2,
    }


def test_pids_are_sorted_and_paired_by_gpu_and_user_without_guessing():
    payload = sample_payload()
    payload["processes"][0].update(pid=3146357, device_idx=[0, 1])
    payload["processes"][1]["pid"] = "2233948"
    payload["processes"][2]["pid"] = 99999
    snapshot = parse_snapshot(payload)
    own = [
        next(usage for usage in gpu.user_usages if usage.username == "yzh")
        for gpu in snapshot.gpus
    ]
    assert own[0].pids == (2233948, 3146357)
    assert own[1].pids == (3146357,)
    assert 99999 not in own[0].pids
    assert own[0].memory_gib == 1.5 and own[0].process_count == 2
    json_usages = snapshot_as_dict(snapshot)["gpus"][0]["user_usages"]
    assert next(usage for usage in json_usages if usage["username"] == "yzh")[
        "pids"
    ] == [2233948, 3146357]


@pytest.mark.parametrize(
    "raw, expected",
    [
        (1, (1,)),
        (3146357, (3146357,)),
        (" 2233948 ", (2233948,)),
        ("000123", (123,)),
        (None, ()),
        (False, ()),
        (True, ()),
        (0, ()),
        (-1, ()),
        (1.0, ()),
        (1.9, ()),
        ("1.0", ()),
        ("1e3", ()),
        ("PID 123", ()),
        ("123 / 456", ()),
        ("", ()),
        ("unknown", ()),
        ("１２３", ()),
        ("9" * 100, ()),
        (2**31, ()),
        ("9999999999", ()),
    ],
)
def test_pid_validation_never_turns_an_invalid_value_into_a_killable_id(raw, expected):
    payload = sample_payload()
    payload["processes"][0]["pid"] = raw
    own = next(
        usage
        for usage in parse_snapshot(payload).gpus[0].user_usages
        if usage.username == "yzh"
    )
    assert own.pids == expected


def test_duplicate_pid_records_do_not_repeat_ids_and_a_new_snapshot_clears_old_ids():
    payload = sample_payload()
    for row in payload["processes"][:2]:
        row["pid"] = 3146357
    own = next(
        usage
        for usage in parse_snapshot(payload).gpus[0].user_usages
        if usage.username == "yzh"
    )
    assert own.pids == (3146357,)
    for row in payload["processes"][:2]:
        row.pop("pid")
    own = next(
        usage
        for usage in parse_snapshot(payload).gpus[0].user_usages
        if usage.username == "yzh"
    )
    assert own.pids == ()


def test_command_records_stay_paired_by_pid_gpu_and_user():
    payload = sample_payload()
    payload["processes"][0].update(
        pid=3146357,
        device_idx=[0, 1],
        command="lerobot-train --batch_size=32 --steps=20000",
    )
    payload["processes"][1].update(
        pid=2233948, command="lerobot-train --batch_size=16 --steps=10000"
    )
    payload["processes"][2].update(
        pid=99999, command="private-other-user --do-not-copy=this"
    )
    snapshot = parse_snapshot(payload)
    own = [
        next(usage for usage in gpu.user_usages if usage.username == "yzh")
        for gpu in snapshot.gpus
    ]
    assert [(item.pid, item.command) for item in own[0].commands] == [
        (2233948, "lerobot-train --batch_size=16 --steps=10000"),
        (3146357, "lerobot-train --batch_size=32 --steps=20000"),
    ]
    assert own[1].commands == (own[0].commands[1],)
    assert all(
        not item.command
        for usage in snapshot.gpus[0].user_usages
        if usage.username != "yzh"
        for item in usage.commands
    )
    serialized = next(
        usage
        for usage in snapshot_as_dict(snapshot)["gpus"][0]["user_usages"]
        if usage["username"] == "yzh"
    )
    assert serialized["process_commands"][0] == {
        "pid": 2233948,
        "command": "lerobot-train --batch_size=16 --steps=10000",
    }


@pytest.mark.parametrize(
    "command",
    [None, {}, [], 123, "", " \t\n ", "train\x00--x=1", "train\x1b[31m", "x" * 65537],
)
def test_invalid_command_never_breaks_pid_parsing_or_becomes_a_fake_command(command):
    payload = sample_payload()
    payload["processes"][0].update(pid=3146357, command=command)
    own = next(
        usage
        for usage in parse_snapshot(payload).gpus[0].user_usages
        if usage.username == "yzh"
    )
    assert own.pids == (3146357,)
    assert own.commands[0].command == ""
    assert format_process_command(own.commands[0].command) == "网页未提供可用指令"


@pytest.mark.parametrize(
    "other, expected",
    [
        ("", "train --steps=1"),
        ("train --steps=1", "train --steps=1"),
        ("train --steps=2", ""),
    ],
)
def test_duplicate_pid_command_conflicts_do_not_pick_an_arbitrary_training_job(
    other, expected
):
    payload = sample_payload()
    payload["processes"][0].update(pid=3146357, command="train --steps=1")
    payload["processes"][1].update(pid=3146357, command=other)
    own = next(
        usage
        for usage in parse_snapshot(payload).gpus[0].user_usages
        if usage.username == "yzh"
    )
    assert [(item.pid, item.command) for item in own.commands] == [(3146357, expected)]


@pytest.mark.parametrize(
    "command, expected",
    [
        (
            "/env/bin/python -u /env/bin/lerobot-train --policy.type=pi05 --batch_size=32 --steps=20000",
            [
                "/env/bin/python -u /env/bin/lerobot-train",
                "    --policy.type=pi05",
                "    --batch_size=32",
                "    --steps=20000",
            ],
        ),
        (
            "lerobot-train --dataset.root=\"/home/yzh/具身智能/训练 数据\" --policy.pretrained_path='/home/yzh/（Lerobot）模型' --steps 20000",
            [
                "lerobot-train",
                '    --dataset.root="/home/yzh/具身智能/训练 数据"',
                "    --policy.pretrained_path='/home/yzh/（Lerobot）模型'",
                "    --steps 20000",
            ],
        ),
        (
            "train --description \"--this is text\" --rate -0.5 --message='value > output' --bool",
            [
                "train",
                '    --description "--this is text"',
                "    --rate -0.5",
                "    --message='value > output'",
                "    --bool",
            ],
        ),
        (
            'train --name="a \\"quoted\\" value" --path /tmp/has\\ space --tag x --tag y',
            [
                "train",
                '    --name="a \\"quoted\\" value"',
                "    --path /tmp/has\\ space",
                "    --tag x",
                "    --tag y",
            ],
        ),
        (
            "CUDA_VISIBLE_DEVICES=4 nohup lerobot-train \\\n --steps=20000 \\\n --batch_size=32 > /tmp/training_2.log 2>&1 &",
            [
                "CUDA_VISIBLE_DEVICES=4 nohup lerobot-train",
                "    --steps=20000",
                "    --batch_size=32",
                "    > /tmp/training_2.log 2>&1 &",
            ],
        ),
        (
            "train --steps=100 -- --literal-option value",
            ["train", "    --steps=100", "    -- --literal-option value"],
        ),
    ],
)
def test_command_parameter_lines_keep_values_quotes_and_order(command, expected):
    assert format_process_command(command).splitlines() == expected


@pytest.mark.parametrize(
    "command", ["lerobot-train --steps=20000", "python -m train --batch_size=32"]
)
def test_formatter_never_invents_environment_nohup_or_log_redirection(command):
    formatted = format_process_command(command)
    assert "CUDA_VISIBLE_DEVICES" not in formatted
    assert "nohup" not in formatted and ">" not in formatted and "&" not in formatted


def test_formatter_preserves_unparseable_original_text_and_never_evaluates_shell():
    command = 'train --path="missing quote --steps=20000'
    assert (
        format_process_command(command)
        == "参数引号不完整，以下按网页原文显示：\n" + command
    )
    inert = 'train --text="$(touch /tmp/do-not-execute)" --payload="<script>not HTML</script>"'
    assert "$(touch /tmp/do-not-execute)" in format_process_command(inert)
    assert "<script>not HTML</script>" in format_process_command(inert)


def test_user_color_matches_web_dashboard_hash_palette() -> None:
    assert user_color("yzh") == "#FFB980"
    assert user_color("gxy") == "#E7BCF3"


def test_parse_snapshot_rejects_missing_gpu_status() -> None:
    with pytest.raises(MonitorError, match="GPU 状态"):
        parse_snapshot({"status": {"status": []}})


def test_build_detail_url_encodes_server_id() -> None:
    assert build_detail_url("https://example.test/", "server 7") == (
        "https://example.test/api/server_detail?server=server+7"
    )


def test_announcements_filter_by_server_and_format_allocations_and_html() -> None:
    payload = {
        "notifications": [
            {
                "id": "rules",
                "status": "active",
                "type": "operations",
                "server_ids": ["server7"],
                "content": "<p>最多&nbsp;<b>2</b>张卡</p><script>hidden()</script>",
            },
            {
                "id": "request",
                "status": "active",
                "type": "gpu_usage",
                "user": "wx",
                "content": "模型训练",
                "show_time": True,
                "expires_at": "2026-10-07T12:00:00Z",
                "allocations": [
                    {"server_id": "server7", "gpu_indexes": [6, 3, 4, 5]},
                    {"server_id": "server1", "gpu_indexes": [0]},
                ],
            },
            {
                "id": "other",
                "status": "active",
                "type": "operations",
                "server_ids": ["server1"],
                "content": "其他服务器",
            },
            {
                "id": "expired",
                "status": "active",
                "type": "operations",
                "server_ids": ["server7"],
                "expires_at": "2026-10-01T00:00:00Z",
                "content": "已过期",
            },
            {
                "id": "inactive",
                "status": "inactive",
                "server_ids": ["server7"],
                "content": "不显示",
            },
        ]
    }
    notices = parse_announcements(
        payload, now=datetime(2026, 10, 3, tzinfo=timezone.utc)
    )
    assert [notice.notice_id for notice in notices] == ["request", "rules"]
    assert "wx需要占用WN-A100的3、4、5、6号GPU" in notices[0].text
    assert notices[0].gpu_indexes == (3, 4, 5, 6)
    assert notices[0].expires_at == datetime(2026, 10, 7, 12, tzinfo=timezone.utc)
    assert "2026-10-07" in notices[0].text
    assert notices[1].text == "最多 2张卡"
    assert notices[1].gpu_indexes == ()
    assert not notices[1].is_server_rules


@pytest.mark.skipif(
    not hasattr(time, "tzset"), reason="Local timezone override needs Unix tzset"
)
@pytest.mark.parametrize(
    ("local_zone", "expected"),
    [
        ("Asia/Shanghai", "2026-10-07 20:00"),
        ("UTC", "2026-10-07 12:00"),
        ("America/New_York", "2026-10-07 08:00"),
    ],
)
def test_deadline_uses_precise_local_time_without_timezone_suffix(
    monkeypatch, local_zone, expected
) -> None:
    try:
        with monkeypatch.context() as tz_patch:
            tz_patch.setenv("TZ", local_zone)
            time.tzset()
            assert (
                format_notice_deadline(datetime(2026, 10, 7, 12, tzinfo=timezone.utc))
                == expected
            )
            assert format_notice_deadline(datetime(2026, 10, 7, 12)) == expected
            assert format_notice_deadline(None) == "未注明"
    finally:
        time.tzset()


def test_deadline_is_retained_even_when_the_web_template_hides_it() -> None:
    (notice,) = parse_announcements(
        {
            "notifications": [
                {
                    "id": "hidden-deadline",
                    "status": "active",
                    "type": "gpu_usage",
                    "server_ids": ["server7"],
                    "gpu_indexes": [3, 4],
                    "show_time": False,
                    "expires_at": "2026-10-07T12:00:00Z",
                }
            ]
        },
        now=datetime(2026, 10, 3, tzinfo=timezone.utc),
    )
    assert notice.expires_at == datetime(2026, 10, 7, 12, tzinfo=timezone.utc)
    assert "2026-10-07" not in notice.text


def test_multiple_requests_keep_local_gpu_groups_and_deadlines_paired() -> None:
    rows = []
    for notice_id, indexes, deadline in (
        ("first", [4, 1], "2026-10-07T12:00:00Z"),
        ("second", [7, 3], "2026-10-10T02:00:00+08:00"),
        ("third", [6, 0, 2], None),
    ):
        rows.append(
            {
                "id": notice_id,
                "status": "active",
                "type": "gpu_usage",
                "template_text": "所有申请共享相同正文",
                "show_time": False,
                "expires_at": deadline,
                "allocations": [
                    {"server_id": "server1", "gpu_indexes": [5]},
                    {"server_id": "server7", "gpu_indexes": indexes},
                ],
            }
        )
    notices = parse_announcements(
        {"notifications": rows}, now=datetime(2026, 10, 3, tzinfo=timezone.utc)
    )
    assert [item.text for item in notices] == ["所有申请共享相同正文"] * 3
    assert [
        (item.notice_id, item.gpu_indexes, item.expires_at) for item in notices
    ] == [
        ("first", (1, 4), datetime(2026, 10, 7, 12, tzinfo=timezone.utc)),
        ("second", (3, 7), datetime(2026, 10, 9, 18, tzinfo=timezone.utc)),
        ("third", (0, 2, 6), None),
    ]


def test_invalid_deadlines_are_unknown_not_guessed_from_notice_dates() -> None:
    (notice,) = parse_announcements(
        {
            "notifications": [
                {
                    "id": "unknown-deadline",
                    "status": "active",
                    "type": "gpu_usage",
                    "server_ids": ["server7"],
                    "gpu_indexes": [2],
                    "expires_at": "invalid",
                    "content": "2026-10-09开始训练",
                    "show_time": True,
                }
            ]
        }
    )
    assert notice.expires_at is None
    assert format_notice_deadline(notice.expires_at) == "未注明"


def test_default_rules_are_identified_but_temporary_operations_are_retained() -> None:
    notices = parse_announcements(
        {
            "notifications": [
                {
                    "id": "standing-rules-new-id",
                    "status": "active",
                    "type": "operations",
                    "server_ids": ["server1", "server7"],
                    "content": (
                        "<p>1. 默认情况下，<b>每个用户每台服务器最多使用&nbsp;2&nbsp;张显卡</b></p>"
                        "<p>更多规定参阅《服务器使用规范》</p>"
                    ),
                },
                {
                    "id": "maintenance",
                    "status": "active",
                    "type": "operations",
                    "server_ids": ["server7"],
                    "content": "10月3日服务器维护，最多使用2张显卡。",
                },
            ]
        }
    )
    assert notices[0].is_server_rules
    assert not notices[1].is_server_rules
    assert notices[1].gpu_indexes == ()  # Dates and quota text are not GPU requests.


def test_requested_card_indexes_only_come_from_this_servers_active_allocations() -> (
    None
):
    notices = parse_announcements(
        {
            "notifications": [
                {
                    "id": "multi-server",
                    "status": "active",
                    "type": "gpu_usage",
                    "user": "alice",
                    "allocations": [
                        {"server_id": "server1", "gpu_indexes": [0, 1, 7]},
                        {"server_id": "server7", "gpu_indexes": [4, 2, 4, -1]},
                        {"server_id": "server7", "gpu_indexes": [6]},
                    ],
                },
                {
                    "id": "other-server",
                    "status": "active",
                    "type": "gpu_usage",
                    "allocations": [{"server_id": "server1", "gpu_indexes": [0]}],
                },
                {
                    "id": "unknown-cards",
                    "status": "active",
                    "type": "gpu_usage",
                    "server_ids": ["server7"],
                    "content": "预计10月8日申请2张卡，未指定卡号",
                },
                {
                    "id": "legacy",
                    "status": "active",
                    "type": "gpu_usage",
                    "server_ids": ["server7"],
                    "gpu_indexes": [3],
                },
                {
                    "id": "expired",
                    "status": "active",
                    "type": "gpu_usage",
                    "expires_at": "2026-10-01T00:00:00Z",
                    "allocations": [{"server_id": "server7", "gpu_indexes": [7]}],
                },
                {
                    "id": "revoked",
                    "status": "inactive",
                    "type": "gpu_usage",
                    "allocations": [{"server_id": "server7", "gpu_indexes": [0]}],
                },
            ]
        },
        now=datetime(2026, 10, 3, tzinfo=timezone.utc),
    )
    assert [item.notice_id for item in notices] == [
        "multi-server",
        "unknown-cards",
        "legacy",
    ]
    assert [item.gpu_indexes for item in notices] == [(2, 4, 6), (), (3,)]


def test_empty_notice_response_clears_announcements_and_invalid_response_is_error() -> (
    None
):
    assert parse_announcements({"notifications": []}) == ()
    with pytest.raises(MonitorError, match="公告接口"):
        parse_announcements({})
