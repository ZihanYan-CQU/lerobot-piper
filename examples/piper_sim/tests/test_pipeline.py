import json
import sys

import numpy as np


def test_expert_uses_contact_physics_and_reaches_goal(robot):
    from lerobot_robot_piper_mujoco.expert import PickPlaceExpert

    sim = robot.simulation
    expert = PickPlaceExpert(sim)
    initial_z = sim.cube_position[2]
    highest_z = initial_z
    while not expert.done:
        robot.send_action(expert.action())
        highest_z = max(highest_z, sim.cube_position[2])
    assert highest_z > initial_z + 0.035
    assert sim.success
    np.testing.assert_allclose(sim.cube_position[:2], sim.goal_position[:2], atol=0.035)


def test_collect_reload_and_replay(tmp_path, model_path, monkeypatch):
    from lerobot_robot_piper_mujoco.cli import main

    from lerobot.datasets import LeRobotDataset

    root = tmp_path / "dataset"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "piper-sim",
            "record",
            "--root",
            str(root),
            "--model-path",
            model_path,
            "--width",
            "80",
            "--height",
            "64",
            "--no-video",
            "--success-only",
            "--no-randomize",
        ],
    )
    main()
    manifest = json.loads((root / "simulation.json").read_text())
    assert len(manifest["episodes"]) == 1
    assert manifest["episodes"][0]["source"] == "scripted_oracle"
    assert manifest["episodes"][0]["success"]
    dataset = LeRobotDataset("local/piper_sim_pick_place", root=root, video_backend="pyav")
    assert dataset.num_episodes == 1 and dataset.num_frames > 100
    assert dataset[0]["observation.images.top"].shape == (3, 64, 80)
    monkeypatch.setattr(sys, "argv", ["piper-sim", "replay", "--root", str(root)])
    main()
