"""SSH/log parser checks without a server, credentials, Qt, or GPU hardware."""

from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import sys
import threading
from dataclasses import replace

import pytest

from lerobot.yzh.wn_a100 import remote


FIRST = "Training:   5%|▌         | 1060/20000 [1:23:27<24:41:33,  4.69s/step]"
SECOND = "Training:  25%|██▌       | 5033/20000 [6:28:34<19:08:39,  4.60s/step]"


@pytest.fixture
def config():
    return remote.load_config()


def test_both_default_training_logs_use_relocated_paths_in_remote_command(config):
    paths = [
        "/home/yzh/具身智能/训练日志/Lerobot/training_2.log",
        "/home/yzh/具身智能/训练日志/Lerobot/training.log",
    ]
    assert [log.path for log in config.logs] == paths
    command_parts = shlex.split(remote.ssh_command(config)[-1])
    assert command_parts[3:] == paths
    assert not any("（Lerobot）训练日志/" in part for part in command_parts[3:])


def payload(config, first=FIRST, second=SECOND):
    return [
        dict(
            path=log.path,
            tail=tail,
            mtime=900,
            server_now=905,
            size=1000,
            identity=str(index),
        )
        for index, (log, tail) in enumerate(
            zip(config.logs, (first, second), strict=True)
        )
    ]


def test_parse_user_example_uses_exact_steps_not_rounded_percent():
    progress, error = remote.parse_progress(FIRST)
    assert not error
    assert progress.step == 1060 and progress.total == 20000
    assert progress.percent == pytest.approx(5.3)
    assert progress.elapsed_seconds == 5007
    assert progress.eta_seconds == 88893
    assert progress.seconds_per_step == 4.69


def test_carriage_returns_ansi_unicode_and_latest_not_highest_counter():
    progress, error = remote.parse_progress(
        f"\x1b[32m{SECOND}\x1b[0m\r{FIRST}\rDownloading: 100%|████| 999/999 [00:00<00:00, 1s/it]"
    )
    assert progress.step == 1060 and not error
    # A complete record remains usable while the next tqdm write is partial.
    assert remote.parse_progress(FIRST + "\rTraining:  5%|▌ | 1061/")[0].step == 1060


@pytest.mark.parametrize(
    ("line", "step", "total", "eta", "rate"),
    [
        (
            "Training: 100%|████| 20,000/20,000 [25:01:00<00:00, 4.5s/it]",
            20000,
            20000,
            0,
            4.5,
        ),
        ("Training:   0%| | 0/20000 [00:00<?, ?it/s]", 0, 20000, None, None),
        ("Training:  10%|█ | 10/100 [00:05<00:45, 2.00it/s]", 10, 100, 45, 0.5),
    ],
)
def test_various_valid_tqdm_records(line, step, total, eta, rate):
    progress, error = remote.parse_progress(line)
    assert not error
    assert (
        progress.step,
        progress.total,
        progress.eta_seconds,
        progress.seconds_per_step,
    ) == (step, total, eta, rate)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Loading: 100%|████| 1/1 [00:00<00:00, 1s/it]",
        "Training: 10%|█ | 11/10 [00:01<00:00, 1s/it]",
        "Training: 0%| | 0/0 [00:00<?, ?it/s]",
    ],
)
def test_no_valid_progress_is_not_invented(text):
    assert remote.parse_progress(text)[0] is None


def test_traceback_after_progress_is_gray_but_old_traceback_before_new_run_is_not(
    config,
):
    error_text = "Traceback (most recent call last):\nRuntimeError: CUDA out of memory"
    progress, error = remote.parse_progress(FIRST + "\n" + error_text)
    assert progress.step == 1060 and error
    samples = remote.parse_training_samples(
        payload(config, FIRST + "\n" + error_text), config, now=100
    )
    assert not samples[0].is_fresh(180, now=100)
    assert samples[1].is_fresh(180, now=100)
    assert remote.parse_progress(error_text + "\n" + FIRST)[1] == ""


def test_explicit_resume_offset_uses_original_total():
    progress, _ = remote.parse_progress(
        "Training: 25%|██▌| 5/20 [00:05<00:15, 1s/it]", step_offset=80
    )
    assert (progress.step, progress.total, progress.percent) == (85, 100, 85)


def test_samples_are_paired_by_full_path_not_response_order(config):
    samples = remote.parse_training_samples(
        list(reversed(payload(config))), config, now=100
    )
    assert (
        samples[0].log.filename == "training_2.log" and samples[0].progress.step == 1060
    )
    assert (
        samples[1].log.filename == "training.log" and samples[1].progress.step == 5033
    )
    assert samples[0].progress.eta_seconds == 88893
    assert samples[1].progress.eta_seconds == 68919
    assert samples[0].file_age == 5


def test_gpu_zero_and_one_eta_and_progress_are_paired_by_full_log_path(config):
    rows = payload(config)
    rows[0]["gpu_indexes"] = [0]
    rows[1]["gpu_indexes"] = [1]
    samples = remote.parse_training_samples(list(reversed(rows)), config, now=100)
    assert [
        (sample.gpu_label, sample.progress.step, sample.progress.eta_seconds)
        for sample in samples
    ] == [
        ("GPU 0", 1060, 88893),
        ("GPU 1", 5033, 68919),
    ]
    assert all(sample.gpu_source == "进程 CUDA_VISIBLE_DEVICES" for sample in samples)


@pytest.mark.parametrize("bad", [None, "0", [-1], [8], [True], [0, "1"]])
def test_unreliable_gpu_metadata_never_discards_valid_training_progress(config, bad):
    rows = payload(config)
    rows[0]["gpu_indexes"] = bad
    sample = remote.parse_training_samples(rows, config, now=100)[0]
    assert sample.progress.step == 1060 and sample.progress.eta_seconds == 88893
    assert sample.gpu_label == "GPU ?"
    assert not sample.error


def test_manual_physical_gpu_override_is_optional_and_zero_is_valid(config):
    log = replace(config.logs[0], gpu_index=0)
    sample = remote.TrainingSample(log, gpu_indexes=(7,))
    assert sample.gpu_label == "GPU 0"
    assert "配置覆盖" in remote.training_text(sample, config)
    assert remote.TrainingSample(config.logs[0]).gpu_label == "GPU ?"


def test_unchanged_finished_log_retains_card_but_new_run_clears_association(config):
    rows = payload(config)
    rows[0]["gpu_indexes"] = [0]
    rows[1]["gpu_indexes"] = [1]
    old = remote.parse_training_samples(rows, config, now=100)
    fresh = remote.parse_training_samples(payload(config), config, now=101)
    unchanged = remote.apply_training_event(old, remote.TrainingEvent(samples=fresh))
    assert [sample.gpu_label for sample in unchanged] == ["GPU 0", "GPU 1"]
    assert "上次进程匹配" in unchanged[0].gpu_source
    for changes in (
        {"identity": "rotated"},
        {"size": 10},
        {"mtime": 901},
        {"size": 1001},
    ):
        changed_rows = payload(config)
        changed_rows[0].update(changes)
        changed = remote.apply_training_event(
            old,
            remote.TrainingEvent(
                samples=remote.parse_training_samples(changed_rows, config, now=102)
            ),
        )
        assert changed[0].gpu_label == "GPU ?"
    failed = remote.apply_training_event(
        old, remote.TrainingEvent(error="SSH disconnected")
    )
    assert failed[0].gpu_label == "GPU 0" and not failed[0].is_fresh(180, now=102)


@pytest.fixture
def remote_gpu_reader():
    namespace = {"os": os, "shlex": shlex, "subprocess": subprocess}
    exec(remote.REMOTE_GPU_HELPERS, namespace)
    return lambda logs, proc_root: namespace["read_log_processes"](logs, proc_root)[0]


def fake_log_process(proc_root, pid, path, mask, extra=b""):
    root = proc_root / str(pid)
    (root / "fd").mkdir(parents=True)
    for fd in ("1", "2"):
        (root / "fd" / fd).symlink_to(path)
    (root / "environ").write_bytes(
        b"CUDA_VISIBLE_DEVICES="
        + mask.encode()
        + b"\0PRIVATE_SECRET=never_output\0"
        + extra
    )
    stat = path.stat()
    return f"{stat.st_dev}:{stat.st_ino}"


def fake_process_identity(proc, pid, ppid=1):
    argv = ["/env/bin/python", "/env/bin/lerobot-train", "--job_name=SSH实际任务"]
    (proc / str(pid) / "cmdline").write_bytes(
        b"\0".join(argument.encode() for argument in argv) + b"\0"
    )
    (proc / str(pid) / "status").write_text(
        f"Name:\tlerobot-train\nPid:\t{pid}\nPPid:\t{ppid}\nNSpid:\t{pid}\n"
    )
    return "/env/bin/python /env/bin/lerobot-train --job_name='SSH实际任务'"


def test_ssh_argv_quotes_values_without_merging_chinese_path_options():
    from lerobot.yzh.wn_a100 import monitor as core

    namespace = {"os": os, "shlex": shlex, "subprocess": subprocess}
    exec(remote.REMOTE_GPU_HELPERS, namespace)
    argv = [
        "/env/bin/python",
        "/env/bin/lerobot-train",
        "--dataset.root=/home/yzh/具身智能/训练 数据",
        "--output_dir=/home/yzh/微调模型/test'quoted",
        "--job_name=pi05",
        "--steps=20000",
        "--description",
        "--this is a value",
    ]
    command = namespace["format_process_argv"](argv)
    assert shlex.split(command) == argv
    lines = core.format_process_command(command).splitlines()
    assert len(lines) == 6
    assert lines[1].startswith("    --dataset.root=")
    assert lines[2].startswith("    --output_dir=")
    assert lines[3] == "    --job_name=pi05"
    assert lines[4] == "    --steps=20000"
    assert lines[5] == "    --description '--this is a value'"


def test_ssh_process_details_use_terminal_pids_and_exclude_inherited_workers(tmp_path):
    first, second, unrelated = [tmp_path / name for name in ("a.log", "b.log", "c.log")]
    for path in (first, second, unrelated):
        path.touch()
    proc = tmp_path / "proc"
    first_id = fake_log_process(proc, 992732, first, "4")
    fake_log_process(proc, 992733, first, "4")
    second_id = fake_log_process(proc, 835616, second, "1")
    fake_log_process(proc, 999, unrelated, "7")
    first_command = fake_process_identity(proc, 992732)
    fake_process_identity(proc, 992733, ppid=992732)
    second_command = fake_process_identity(proc, 835616)
    fake_process_identity(proc, 999)
    namespace = {"os": os, "shlex": shlex, "subprocess": subprocess}
    exec(remote.REMOTE_GPU_HELPERS, namespace)
    associations, processes = namespace["read_log_processes"](
        {first_id: str(first), second_id: str(second)}, str(proc)
    )
    assert associations == {str(first): [4], str(second): [1]}
    assert processes == {
        str(first): [
            {
                "pid": 992732,
                "name": "lerobot-train",
                "command": first_command,
                "gpu_indexes": [4],
            }
        ],
        str(second): [
            {
                "pid": 835616,
                "name": "lerobot-train",
                "command": second_command,
                "gpu_indexes": [1],
            }
        ],
    }
    assert "PRIVATE_SECRET" not in json.dumps(processes)


def process_payload(config):
    rows = payload(config)
    for index, row in enumerate(rows):
        row["processes"] = [
            {
                "pid": (992732, 835616)[index],
                "name": "lerobot-train",
                "command": f"lerobot-train --job_name=task_{index}",
                "gpu_indexes": [(4, 1)[index]],
            }
        ]
    return rows


def test_progress_and_ssh_pid_command_pairs_follow_log_paths_not_response_order(config):
    samples = remote.parse_training_samples(
        list(reversed(process_payload(config))), config, now=100
    )
    assert [
        (
            sample.processes[0].pid,
            sample.processes[0].gpu_indexes,
            sample.processes[0].command,
        )
        for sample in samples
    ] == [
        (992732, (4,), "lerobot-train --job_name=task_0"),
        (835616, (1,), "lerobot-train --job_name=task_1"),
    ]


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "wrong",
        [None],
        [{"pid": True}],
        [{"pid": -1}],
        [{"pid": "992732"}],
        [{"pid": 2**31}],
    ],
)
def test_invalid_ssh_process_metadata_does_not_hide_valid_training_progress(
    config, bad
):
    rows = process_payload(config)
    rows[0]["processes"] = bad
    samples = remote.parse_training_samples(rows, config)
    assert samples[0].processes == () and samples[0].progress.step == 1060
    assert samples[1].processes[0].pid == 835616


def test_conflicting_ssh_records_for_same_pid_are_not_arbitrarily_chosen(config):
    rows = process_payload(config)
    rows[0]["processes"].append(
        {**rows[0]["processes"][0], "command": "other --job_name=wrong"}
    )
    assert remote.parse_training_samples(rows, config)[0].processes == ()


@pytest.mark.parametrize("failure", ["transport", "file", "exited"])
def test_old_progress_can_survive_but_old_ssh_pids_never_survive_failed_reads(
    config, failure
):
    old = remote.parse_training_samples(process_payload(config), config, now=100)
    rows = payload(config)
    if failure == "file":
        rows[0] = {"path": config.logs[0].path, "error": "permission denied"}
    event = (
        remote.TrainingEvent(error="SSH disconnected")
        if failure == "transport"
        else remote.TrainingEvent(
            samples=remote.parse_training_samples(rows, config, now=101)
        )
    )
    new = remote.apply_training_event(old, event)
    assert [sample.progress.step for sample in new] == [1060, 5033]
    assert all(sample.processes == () for sample in new)


def test_gpu_env_pairs_by_inode_and_deduplicates_workers_without_leaking_fields(
    tmp_path, remote_gpu_reader
):
    first, second, unrelated = [
        tmp_path / name for name in ("training_2.log", "training.log", "other.log")
    ]
    for path in (first, second, unrelated):
        path.write_text("sample log")
    proc = tmp_path / "proc"
    first_identity = fake_log_process(proc, 10, first, "0")
    fake_log_process(proc, 11, first, "0")
    second_identity = fake_log_process(proc, 20, second, "1")
    fake_log_process(proc, 99, unrelated, "7")
    result = remote_gpu_reader(
        {first_identity: str(first), second_identity: str(second)}, str(proc)
    )
    assert result == {str(first): [0], str(second): [1]}
    assert "PRIVATE_SECRET" not in json.dumps(result)


def test_relocated_log_keeps_gpu_association_by_inode(tmp_path, remote_gpu_reader):
    old = tmp_path / "（Lerobot）训练日志" / "training.log"
    old.parent.mkdir()
    old.write_text(FIRST, encoding="utf-8")
    proc = tmp_path / "proc"
    identity = fake_log_process(proc, 10, old, "4")
    moved = tmp_path / "训练日志" / "Lerobot" / "training.log"
    moved.parent.mkdir(parents=True)
    old.rename(moved)
    # A running writer retains its open descriptor after a rename. Fake /proc
    # fd links reflect the kernel's updated path for that same open inode.
    for number in ("1", "2"):
        fd = proc / "10" / "fd" / number
        fd.unlink()
        fd.symlink_to(moved)
    assert remote_gpu_reader({identity: str(moved)}, str(proc)) == {str(moved): [4]}


@pytest.mark.parametrize(
    "mask, expected",
    [
        ("7", [7]),
        ("1,0,1", [0, 1]),
        ("", []),
        ("-1", []),
        ("8", []),
        ("unknown", []),
        ("0,broken", []),
    ],
)
def test_gpu_env_validates_masks_and_does_not_guess(
    tmp_path, remote_gpu_reader, mask, expected
):
    path = tmp_path / "training.log"
    path.touch()
    proc = tmp_path / "proc"
    identity = fake_log_process(proc, 10, path, mask)
    assert remote_gpu_reader({identity: str(path)}, str(proc)) == {str(path): expected}


def test_gpu_env_skips_other_users_and_ambiguous_container_indices(
    tmp_path, remote_gpu_reader, monkeypatch
):
    path = tmp_path / "training.log"
    path.touch()
    proc = tmp_path / "proc"
    identity = fake_log_process(
        proc, 10, path, "0", b"NVIDIA_VISIBLE_DEVICES=GPU-abcdef\0"
    )
    assert remote_gpu_reader({identity: str(path)}, str(proc)) == {str(path): []}
    (proc / "10" / "environ").write_bytes(b"CUDA_VISIBLE_DEVICES=0\0")
    current_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: current_uid + 1)
    assert remote_gpu_reader({identity: str(path)}, str(proc)) == {str(path): []}


def test_gpu_env_uuid_masks_require_an_unambiguous_physical_index(
    tmp_path, remote_gpu_reader, monkeypatch
):
    path = tmp_path / "training.log"
    path.touch()
    proc = tmp_path / "proc"
    identity = fake_log_process(proc, 10, path, "GPU-aaa")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, "3, GPU-aaa001\n6, GPU-bbb002\n"
        ),
    )
    assert remote_gpu_reader({identity: str(path)}, str(proc)) == {str(path): [3]}
    (proc / "10" / "environ").write_bytes(b"CUDA_VISIBLE_DEVICES=GPU-\0")
    assert remote_gpu_reader({identity: str(path)}, str(proc)) == {str(path): []}


@pytest.mark.parametrize("bad_index", [-1, 8, True, "0", 0.0])
def test_config_gpu_index_must_be_a_physical_integer_or_null(tmp_path, bad_index):
    raw = json.loads(remote.DEFAULT_CONFIG.read_text())
    raw["logs"][0]["gpu_index"] = bad_index
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(remote.RemoteError, match="gpu_index"):
        remote.load_config(path)


def test_one_missing_log_does_not_hide_the_other(config):
    rows = payload(config)
    rows[0] = dict(path=config.logs[0].path, error="File not found")
    samples = remote.parse_training_samples(rows, config, now=100)
    assert samples[0].error and samples[0].progress is None
    assert samples[1].progress.step == 5033


@pytest.mark.parametrize("bad", [None, [], [{"path": "wrong"}, {"path": "wrong"}]])
def test_response_paths_must_match_configuration(config, bad):
    with pytest.raises(remote.RemoteError):
        remote.parse_training_samples(bad, config)


def test_stale_uses_server_age_plus_local_monotonic_elapsed_not_client_clock(config):
    sample = remote.parse_training_samples(payload(config), config, now=100)[0]
    assert sample.is_fresh(180, now=274)
    assert sample.status(180, now=275) == "日志未更新"
    complete = replace(sample, progress=remote.TrainingProgress(20000, 20000))
    assert complete.status(180, now=999999) == "步数已达 100%"
    assert not replace(complete, error="SSH disconnected").is_fresh(180, now=100)


def test_transport_failure_retains_gray_values_then_recovers_and_new_log_resets(config):
    old = remote.parse_training_samples(payload(config), config, now=100)
    failed = remote.apply_training_event(
        old, remote.TrainingEvent(error="SSH disconnected")
    )
    assert [sample.progress.step for sample in failed] == [1060, 5033]
    assert all(not sample.is_fresh(180, now=100) for sample in failed)
    recovered = remote.apply_training_event(failed, remote.TrainingEvent(samples=old))
    assert not any(sample.error for sample in recovered)
    new_rows = payload(config, "", "Training: 0%| | 0/100 [00:00<?, ?it/s]")
    new_rows[0].update(size=0, identity="new file")
    new = remote.apply_training_event(
        old,
        remote.TrainingEvent(
            samples=remote.parse_training_samples(new_rows, config, now=101)
        ),
    )
    assert new[0].progress is None
    assert new[1].progress.step == 0 and new[1].progress.total == 100


def test_per_file_failure_retains_only_that_last_progress(config):
    old = remote.parse_training_samples(payload(config), config, now=100)
    rows = payload(config, FIRST, SECOND.replace("5033/", "5034/"))
    rows[0] = dict(path=config.logs[0].path, error="permission denied")
    samples = remote.apply_training_event(
        old,
        remote.TrainingEvent(
            samples=remote.parse_training_samples(rows, config, now=101)
        ),
    )
    assert samples[0].progress.step == 1060 and samples[0].error
    assert samples[1].progress.step == 5034 and not samples[1].error


def test_ssh_command_requires_existing_master_and_never_prompts_or_forwards(config):
    command = remote.ssh_command(config)
    assert command[0] == "ssh"
    assert command[command.index("-S") + 1] == config.control_path
    assert all(
        option in command
        for option in (
            "ControlMaster=no",
            "ClearAllForwardings=yes",
            "PermitLocalCommand=no",
            "BatchMode=yes",
            "StrictHostKeyChecking=yes",
            "ProxyCommand=false",
        )
    )
    assert all(
        flag not in command for flag in ("-L", "-R", "-D", "-f", "-N", "-J", "-M", "-i")
    )
    command_parts = shlex.split(command[-1])
    assert command_parts[:2] == ["python3", "-c"]
    assert command_parts[2] == remote.REMOTE_TRAINING_READER
    assert command_parts[3:] == [log.path for log in config.logs]
    login = shlex.split(remote.login_command(config))
    assert login[login.index("-J") + 1] == "jumper@cps.cslcqu.cn:6666"
    assert "ControlPersist=8h" in login and "-L" not in login


def test_remote_reader_bounded_cr_tails_unicode_paths_and_independent_missing_file(
    tmp_path,
):
    path = tmp_path / "（Lerobot）训练日志.log"
    path.write_bytes(b"private head\n" + (FIRST + "\r").encode() * 600)
    missing = tmp_path / "missing.log"
    result = subprocess.run(
        [sys.executable, "-c", remote.REMOTE_TRAINING_READER, str(path), str(missing)],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = json.loads(result.stdout)
    assert rows[0]["path"] == str(path)
    assert len(rows[0]["tail"].encode()) <= remote.TAIL_BYTES
    assert "private head" not in rows[0]["tail"]
    assert rows[0]["tail"].startswith("Training:")
    assert rows[0]["size"] == path.stat().st_size
    assert rows[1]["error"]


@pytest.mark.parametrize(
    "update",
    [
        {"port": 0},
        {"target": "-o ProxyCommand=evil"},
        {"control_path": "relative.sock"},
        {"interval": 0},
        {"timeout": "nan"},
        {"logs": []},
    ],
)
def test_invalid_config_cannot_become_ssh_options(tmp_path, update):
    raw = json.loads(remote.DEFAULT_CONFIG.read_text())
    raw.update(update)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(remote.RemoteError):
        remote.load_config(path)


class FakeClient:
    def __init__(self, output="", error="", returncode=0, timeout=False):
        self.output, self.error = output, error
        self.returncode = returncode
        self.timeout = timeout
        self.pid = 99999999

    def communicate(self, timeout=None):
        if self.timeout and timeout is not None:
            raise subprocess.TimeoutExpired("ssh", timeout)
        return self.output, self.error

    def poll(self):
        return self.returncode


def test_fetch_removes_ld_library_path_and_cleans_only_its_client(config, monkeypatch):
    captured, ended = [], []
    monkeypatch.setenv("LD_LIBRARY_PATH", "/some/incompatible/cuda/lib")
    client = FakeClient(json.dumps(payload(config)))

    def launch(command, **kwargs):
        captured.append((command, kwargs))
        return client

    monkeypatch.setattr(remote.subprocess, "Popen", launch)
    monkeypatch.setattr(remote, "_end_client", ended.append)
    samples = remote.fetch_training(config)
    assert samples[1].progress.step == 5033
    assert "LD_LIBRARY_PATH" not in captured[0][1]["env"]
    assert captured[0][1]["start_new_session"] is True
    assert ended == [client]


@pytest.mark.parametrize(
    ("client", "message"),
    [
        (FakeClient(error="Connection refused", returncode=255), "共享连接不可用"),
        (FakeClient(output="not JSON"), "读取失败"),
        (FakeClient(timeout=True), "读取超时"),
    ],
)
def test_failed_ssh_or_json_and_timeouts_are_reported_without_auth_prompts(
    config, monkeypatch, client, message
):
    ended = []
    monkeypatch.setattr(remote.subprocess, "Popen", lambda *a, **k: client)
    monkeypatch.setattr(remote, "_end_client", ended.append)
    with pytest.raises(remote.RemoteError, match=message):
        remote.fetch_training(replace(config, timeout=0.01))
    assert ended == [client]


def test_cancelled_read_and_stopping_poller_never_close_user_master(
    config, monkeypatch
):
    client = FakeClient()
    ended = []
    monkeypatch.setattr(remote.subprocess, "Popen", lambda *a, **k: client)
    monkeypatch.setattr(remote, "_end_client", ended.append)
    stop = threading.Event()
    stop.set()
    with pytest.raises(remote.RemoteError, match="取消"):
        remote.fetch_training(config, stop)
    assert ended == [client]
    events = queue.Queue(maxsize=8)
    poller = remote.TrainingPoller(events, config, demo=True)
    poller.start()
    assert events.get(timeout=1).samples[0].progress.step == 1060
    poller.refresh()
    assert events.get(timeout=1).samples[1].progress.step == 5034
    poller.stop()
    poller.join(timeout=1)
    assert not poller.is_alive()


def test_formatted_eta_and_stale_status_do_not_claim_a_live_countdown(config):
    sample = remote.parse_training_samples(payload(config), config)[0]
    assert "24:41:33" in remote.training_text(sample, config)
    assert "上次日志估计" in remote.training_text(
        replace(sample, error="SSH disconnected"), config
    )
    assert remote.format_duration(None) == "待估算"


def test_cli_training_demo_json_and_existing_gpu_once_do_not_connect(
    monkeypatch, capsys
):
    from lerobot.yzh.wn_a100 import monitor as core

    def forbidden(*args, **kwargs):
        raise AssertionError("demo / GPU --once must not open SSH")

    monkeypatch.setattr(remote, "fetch_training", forbidden)
    assert core.main(["--training-once", "--json", "--demo"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["step"] == 1060 and rows[0]["percent"] == 5.3
    assert rows[1]["step"] == 5033 and rows[1]["eta_seconds"] == 68919
    assert core.main(["--once", "--json", "--demo"]) == 0
    assert "gpus" in json.loads(capsys.readouterr().out)


def test_cli_training_failure_returns_nonzero_without_loading_qt(monkeypatch, capsys):
    from lerobot.yzh.wn_a100 import monitor as core

    def fail(*args, **kwargs):
        raise remote.RemoteError("共享连接不可用")

    monkeypatch.setattr(remote, "fetch_training", fail)
    assert core.main(["--training-once"]) == 1
    assert "共享连接不可用" in capsys.readouterr().err
