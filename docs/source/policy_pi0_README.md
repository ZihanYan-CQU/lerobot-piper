# π₀ (pi0)

LeRobot's port of **π₀**, Physical Intelligence's Vision-Language-Action model for general robot control, is adapted from [OpenPI](https://github.com/Physical-Intelligence/openpi).

See the [π₀ guide](https://huggingface.co/docs/lerobot/main/en/pi0) for installation, training, and checkpoints.

## Model Overview

π₀ uses a continuous state projection, concatenates time with actions, and has a 48-token tokenizer limit. See the [model overview](https://huggingface.co/docs/lerobot/main/en/pi0#model-overview) for architecture details.

## Relative Actions

<a id="how-it-works"></a>
<a id="configuration"></a>
<a id="training-example"></a>
<a id="recomputing-stats-for-an-existing-dataset"></a>

Relative actions are optional (`use_relative_actions=false` by default). Before enabling them, recompute dataset statistics in relative space with the policy's chunk size. Follow the [relative-action setup and training commands](https://huggingface.co/docs/lerobot/main/en/pi0#relative-actions); see [Action Representations](https://huggingface.co/docs/lerobot/main/en/action_representations#using-relative-actions-in-lerobot) for the shared processing flow.

## Citation

If you use this work, please cite both **OpenPI** and the π₀ paper:

```bibtex
@misc{openpi2024,
  author       = {Physical Intelligence Lab},
  title        = {OpenPI: PyTorch Implementation of π0 and π0.5 Policies},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Physical-Intelligence/openpi}},
  license      = {Apache-2.0}
}

@misc{black2024pi0visionlanguageactionflowmodel,
  title        = {π₀: A Vision-Language-Action Flow Model for General Robot Control},
  author       = {Kevin Black and Noah Brown and Danny Driess and Adnan Esmail and Michael Equi and Chelsea Finn and Niccolo Fusai and Lachy Groom and Karol Hausman and Brian Ichter and Szymon Jakubczak and Tim Jones and Liyiming Ke and Sergey Levine and Adrian Li-Bell and Mohith Mothukuri and Suraj Nair and Karl Pertsch and Lucy Xiaoyang Shi and James Tanner and Quan Vuong and Anna Walling and Haohuan Wang and Ury Zhilinsky},
  year         = {2024},
  eprint       = {2410.24164},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2410.24164},
}
```

---

## License

This port follows the **Apache 2.0 License**, consistent with the original [OpenPI repository](https://github.com/Physical-Intelligence/openpi).
