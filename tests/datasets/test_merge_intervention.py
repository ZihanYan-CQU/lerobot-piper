#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import hashlib
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.aggregate import validate_all_metadata
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.io_utils import EPISODE_SAVE_MARKER
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def make_sources(tmp_path, with_images=False):
    sources = []
    for name, flags, offset in (
        ("demonstrations", None, 100),
        ("corrections", [[True, False], [True, True, False]], 200),
    ):
        features = {
            key: {"dtype": "float32", "shape": (2,), "names": ["joint_1.pos", "gripper.pos"]}
            for key in ("action", "observation.state")
        }
        if flags is not None:
            features["intervention"] = {"dtype": "bool", "shape": (1,), "names": None}
        if with_images:
            features["observation.images.top"] = {
                "dtype": "image",
                "shape": (8, 8, 3),
                "names": ["height", "width", "channels"],
            }
        ds = LeRobotDataset.create(f"test/{name}", fps=10, features=features, root=tmp_path / name)
        for ep_index, length in enumerate((2, 3)):
            for frame_index in range(length):
                value = np.array([offset + ep_index, frame_index], dtype=np.float32)
                frame = {"action": value, "observation.state": value + 1, "task": "place block"}
                if flags is not None:
                    frame["intervention"] = np.array([flags[ep_index][frame_index]], dtype=bool)
                if with_images:
                    frame["observation.images.top"] = np.full((8, 8, 3), frame_index, dtype=np.uint8)
                ds.add_frame(frame)
            ds.save_episode()
        ds.finalize()
        sources.append(ds)
    return sources


def source_contents(sources):
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for ds in sources
        for folder in ("meta", "data")
        for path in sorted((ds.root / folder).rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize("corrections_first", [False, True])
@pytest.mark.parametrize("concatenate_data", [False, True])
@pytest.mark.parametrize("with_images", [False, True])
def test_merge_fills_only_missing_intervention(tmp_path, corrections_first, concatenate_data, with_images):
    sources = make_sources(tmp_path, with_images)
    if corrections_first:
        sources.reverse()
    before = source_contents(sources)
    output = tmp_path / "merged"
    merged = merge_datasets(
        sources, "test/merged", output, concatenate_data=concatenate_data, fill_missing_intervention=True
    )

    assert merged.num_frames == 10
    assert merged.num_episodes == 4
    assert merged.features["intervention"] == {"dtype": "bool", "shape": (1,), "names": None}
    expected_flags = [True, False, True, True, False] if corrections_first else [False] * 5
    expected_flags += [False] * 5 if corrections_first else [True, False, True, True, False]
    data = pd.concat([pd.read_parquet(path) for path in sorted((output / "data").rglob("*.parquet"))])
    assert data["intervention"].dtype == bool
    assert data["intervention"].tolist() == expected_flags
    for i, flag in enumerate(expected_flags):
        item = merged[i]
        assert item["intervention"].item() is flag
        source = sources[i // 5][i % 5]
        np.testing.assert_array_equal(item["action"].numpy(), source["action"].numpy())
        np.testing.assert_array_equal(item["observation.state"].numpy(), source["observation.state"].numpy())
        if with_images:
            np.testing.assert_array_equal(
                item["observation.images.top"].numpy(), source["observation.images.top"].numpy()
            )

    stats = merged.meta.stats["intervention"]
    np.testing.assert_array_equal(stats["count"], [10])
    np.testing.assert_allclose(stats["mean"], [0.3])
    np.testing.assert_allclose(stats["std"], [np.sqrt(0.21)])
    np.testing.assert_array_equal(stats["min"], [0])
    np.testing.assert_array_equal(stats["max"], [1])
    expected_stats = aggregate_stats([ds.meta.stats for ds in sources])
    for feature in ("action", "observation.state"):
        for stat, value in expected_stats[feature].items():
            np.testing.assert_array_equal(merged.meta.stats[feature][stat], value)

    episodes = pd.concat(
        [pd.read_parquet(path) for path in sorted((output / "meta" / "episodes").rglob("*.parquet"))]
    )
    for _, row in episodes.iterrows():
        assert int(row["stats/intervention/count"][0]) == row["length"]
        flags = data.loc[data["episode_index"] == row["episode_index"], "intervention"].to_numpy()
        for stat, value in (
            ("mean", flags.mean()),
            ("std", flags.std()),
            ("min", flags.min()),
            ("max", flags.max()),
        ):
            np.testing.assert_allclose(row[f"stats/intervention/{stat}"], [value])
    assert source_contents(sources) == before
    assert "intervention" not in next(ds for ds in sources if ds.repo_id.endswith("demonstrations")).features


def test_merge_requires_opt_in_for_missing_intervention(tmp_path):
    sources = make_sources(tmp_path)
    output = tmp_path / "merged"
    with pytest.raises(ValueError, match="Same features is expected"):
        merge_datasets(sources, "test/merged", output)
    assert not output.exists()


@pytest.mark.parametrize(
    "mismatch", ["action_shape", "joint_order", "extra_feature", "flag_dtype", "flag_shape", "fps"]
)
def test_intervention_filling_keeps_other_validation(mismatch):
    features = {"action": {"dtype": "float32", "shape": (2,), "names": ["joint_1.pos", "gripper.pos"]}}
    first = SimpleNamespace(features=features, fps=10, robot_type="piper")
    second = SimpleNamespace(features=copy.deepcopy(features), fps=10, robot_type="piper")
    second.features["intervention"] = {"dtype": "bool", "shape": (1,), "names": None}
    if mismatch == "action_shape":
        second.features["action"]["shape"] = (3,)
    elif mismatch == "joint_order":
        second.features["action"]["names"].reverse()
    elif mismatch == "extra_feature":
        second.features["reward"] = {"dtype": "float32", "shape": (1,), "names": None}
    elif mismatch == "flag_dtype":
        second.features["intervention"]["dtype"] = "float32"
    elif mismatch == "flag_shape":
        second.features["intervention"]["shape"] = (2,)
    else:
        second.fps = 30
    with pytest.raises(ValueError):
        validate_all_metadata([first, second], fill_missing_intervention=True)


def test_gui_merge_command_handles_intervention(tmp_path):
    pytest.importorskip("tkinter", reason="the dataset picker requires tkinter")
    from lerobot.yzh.merge_datasets_gui import PROJECT_ROOT, build_merge_command

    sources = make_sources(tmp_path)
    command, output = build_merge_command([ds.root for ds in sources], "test/merged", tmp_path)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    merged = LeRobotDataset("test/merged", root=output)
    assert [merged[i]["intervention"].item() for i in range(10)] == [False] * 5 + [
        True,
        False,
        True,
        True,
        False,
    ]


@pytest.mark.parametrize("corruption", ["tail", "index", "episode_index", "frame_index", "marker"])
def test_merge_rejects_incomplete_sources_before_creating_output(tmp_path, corruption):
    sources = make_sources(tmp_path)
    correction_root = sources[1].root
    path = next(correction_root.glob("data/*/*.parquet"))
    if corruption == "marker":
        (correction_root / EPISODE_SAVE_MARKER).write_text("{}")
        expected_error = "interrupted episode save"
    else:
        df = pd.read_parquet(path)
        if corruption == "tail":
            tail = df.iloc[-1:].copy()
            tail["index"] = len(df)
            tail["episode_index"] = 2
            tail["frame_index"] = 0
            df = pd.concat([df, tail], ignore_index=True)
        else:
            df.loc[3, corruption] = df.loc[2, corruption] if corruption != "episode_index" else 99
        df.to_parquet(path, index=False)
        expected_error = "Inconsistent episode data"
    before = source_contents(sources)
    output = tmp_path / "merged"
    with pytest.raises(ValueError, match=expected_error):
        merge_datasets(sources, "test/merged", output, fill_missing_intervention=True)
    assert not output.exists()
    assert source_contents(sources) == before
