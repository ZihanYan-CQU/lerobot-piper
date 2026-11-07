"""Run the actual latent-replay learner entrypoint, including resume."""

import pytest

from examples.tutorial.pi05.train_rlt_stage2 import main
from lerobot.rl.rlt.checkpoint import load_checkpoint
from lerobot.rl.rlt.replay import FixedWindowReplayBuffer
from tests.rl.test_rlt_replay import config, transition


def test_stage2_training_entrypoint_saves_and_resumes(tmp_path, monkeypatch):
    replay = FixedWindowReplayBuffer(config(), capacity=4)
    replay.add(transition())
    source = tmp_path / "replay.pt"
    replay.save(source)
    first = tmp_path / "first"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_rlt_stage2.py",
            "--replay",
            str(source),
            "--output-dir",
            str(first),
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--hidden-dims",
            "16",
            "16",
            "--device",
            "cpu",
        ],
    )
    main()
    algorithm, restored = load_checkpoint(first)
    assert algorithm.optimization_step == 2 and len(restored) == 1
    second = tmp_path / "second"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_rlt_stage2.py",
            "--resume",
            str(first),
            "--output-dir",
            str(second),
            "--steps",
            "1",
            "--batch-size",
            "2",
        ],
    )
    main()
    resumed, _ = load_checkpoint(second)
    assert resumed.optimization_step == 3


def test_stage2_entrypoint_requires_executed_replay(tmp_path, monkeypatch):
    replay = FixedWindowReplayBuffer(config(), capacity=4)
    source = tmp_path / "empty.pt"
    replay.save(source)
    destination = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        ["train_rlt_stage2.py", "--replay", str(source), "--output-dir", str(destination), "--steps", "1"],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and not destination.exists()
