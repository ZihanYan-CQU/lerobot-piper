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

from pathlib import Path
from types import SimpleNamespace

import draccus
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.processor import (  # noqa: E402
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.scripts.lerobot_train import _make_processor_kwargs  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def make_cfg(pretrained_path=None, **overrides):
    policy = PI05Config(
        device="cpu",
        pretrained_path=pretrained_path,
        push_to_hub=False,
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(1,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(1,))},
    )
    return TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"), policy=policy, **overrides)


def test_use_pretrained_stats_cli_and_config_round_trip(tmp_path):
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=lerobot/dummy",
            "--policy.type=pi05",
            f"--policy.pretrained_path={tmp_path}",
            "--policy.push_to_hub=false",
            "--use_pretrained_stats=true",
        ],
    )
    cfg.validate()
    assert cfg.use_pretrained_stats
    assert not cfg.resume
    encoded = draccus.encode(cfg)
    assert draccus.decode(TrainPipelineConfig, encoded).use_pretrained_stats
    del encoded["use_pretrained_stats"]
    assert not draccus.decode(TrainPipelineConfig, encoded).use_pretrained_stats


def test_use_pretrained_stats_requires_a_checkpoint():
    cfg = make_cfg(use_pretrained_stats=True)
    with pytest.raises(ValueError, match="requires a pretrained policy checkpoint"):
        cfg.validate()


@pytest.mark.parametrize(
    "use_pretrained_stats,resume,expected_normalized,expected_physical",
    [
        (False, False, -0.5, 10.0),
        (True, False, 0.0, 5.0),
        (False, True, 0.0, 5.0),
        (True, True, 0.0, 5.0),
    ],
)
def test_training_stats_selection_changes_both_processor_outputs(
    tmp_path, use_pretrained_stats, resume, expected_normalized, expected_physical
):
    cfg = make_cfg(tmp_path, use_pretrained_stats=use_pretrained_stats, resume=resume)
    features = {**cfg.policy.input_features, **cfg.policy.output_features}
    norm_map = {
        FeatureType.STATE: NormalizationMode.QUANTILES,
        FeatureType.ACTION: NormalizationMode.QUANTILES,
    }
    saved_stats = {key: {"q01": [0.0], "q99": [10.0]} for key in features}
    dataset_stats = {key: {"q01": [0.0], "q99": [20.0]} for key in features}
    saved_preprocessor = PolicyProcessorPipeline(
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            NormalizerProcessorStep(features=features, norm_map=norm_map, stats=saved_stats),
            DeviceProcessorStep(device="cpu"),
        ],
    )
    saved_postprocessor = PolicyProcessorPipeline(
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        steps=[
            UnnormalizerProcessorStep(
                features=cfg.policy.output_features, norm_map=norm_map, stats=saved_stats
            )
        ],
    )
    saved_preprocessor.save_pretrained(tmp_path)
    saved_postprocessor.save_pretrained(tmp_path)

    kwargs = _make_processor_kwargs(
        cfg,
        SimpleNamespace(config=cfg.policy),
        SimpleNamespace(stats=dataset_stats),
        torch.device("cpu"),
        cfg.policy.pretrained_path,
    )
    if use_pretrained_stats or resume:
        assert "dataset_stats" not in kwargs
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=str(tmp_path), **kwargs
    )
    normalized = preprocessor({OBS_STATE: torch.tensor([[5.0]]), ACTION: torch.tensor([[[5.0]]])})
    torch.testing.assert_close(normalized[OBS_STATE], torch.tensor([[expected_normalized]]))
    torch.testing.assert_close(normalized[ACTION], torch.tensor([[[expected_normalized]]]))
    physical = postprocessor(torch.zeros(1, 1, 1))
    torch.testing.assert_close(physical, torch.tensor([[[expected_physical]]]))

    # Newly saved processors must retain the selected statistics for inference.
    export_path = Path(tmp_path) / "export"
    preprocessor.save_pretrained(export_path)
    postprocessor.save_pretrained(export_path)
    restored_preprocessor, restored_postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=str(export_path)
    )
    restored = restored_preprocessor({OBS_STATE: torch.tensor([[5.0]]), ACTION: torch.tensor([[[5.0]]])})
    torch.testing.assert_close(restored[OBS_STATE], normalized[OBS_STATE])
    torch.testing.assert_close(restored[ACTION], normalized[ACTION])
    torch.testing.assert_close(restored_postprocessor(torch.zeros(1, 1, 1)), physical)


def test_rebuilt_processors_cannot_silently_skip_pretrained_stats(tmp_path):
    """Recipe fine-tuning clears the processor path even with a policy checkpoint."""
    cfg = make_cfg(tmp_path, use_pretrained_stats=True)
    with pytest.raises(ValueError, match="requires loading saved processors"):
        _make_processor_kwargs(
            cfg, SimpleNamespace(config=cfg.policy), SimpleNamespace(stats={}), torch.device("cpu"), None
        )


def test_rebuilt_processors_keep_using_dataset_stats_by_default(tmp_path):
    cfg = make_cfg(tmp_path)
    stats = {OBS_STATE: {"q01": [0.0], "q99": [20.0]}}
    kwargs = _make_processor_kwargs(
        cfg, SimpleNamespace(config=cfg.policy), SimpleNamespace(stats=stats), torch.device("cpu"), None
    )
    assert kwargs["dataset_stats"] == stats
    assert "preprocessor_overrides" not in kwargs
    assert "postprocessor_overrides" not in kwargs
