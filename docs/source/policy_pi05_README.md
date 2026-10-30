# π₀.₅ (pi05)

LeRobot's port of **π₀.₅**, Physical Intelligence's Vision-Language-Action model with open-world generalization, is adapted from [OpenPI](https://github.com/Physical-Intelligence/openpi).

See the [π₀.₅ guide](https://huggingface.co/docs/lerobot/main/en/pi05) for installation, training, and checkpoints.

## Model Overview

π₀.₅ uses discretized state input, AdaRMS time conditioning in the action expert, and a 200-token tokenizer limit. See the [model overview](https://huggingface.co/docs/lerobot/main/en/pi05#model-overview) for architecture details.

## Relative Actions

<a id="how-it-works"></a>
<a id="configuration"></a>
<a id="training-example"></a>

Relative actions are optional (`use_relative_actions=false` by default). Before enabling them, recompute dataset statistics in relative space with the policy's chunk size. Follow the [relative-action setup and training commands](https://huggingface.co/docs/lerobot/main/en/pi05#relative-actions); see [Action Representations](https://huggingface.co/docs/lerobot/main/en/action_representations#using-relative-actions-in-lerobot) for the shared processing flow.

## EMA of the policy weights

See [EMA of the policy weights](https://huggingface.co/docs/lerobot/main/en/pi05#ema-of-the-policy-weights) in the training guide for configuration, checkpoint loading, and PEFT limitations. In a local checkout, this section is in `docs/source/pi05.mdx`.

## Citation

If you use this work, please cite both **OpenPI** and the π₀.₅ paper:

```bibtex
@misc{openpi2024,
  author       = {Physical Intelligence Lab},
  title        = {OpenPI: PyTorch Implementation of π0 and π0.5 Policies},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Physical-Intelligence/openpi}},
  license      = {Apache-2.0}
}

@misc{intelligence2025pi05visionlanguageactionmodelopenworld,
  title        = {π₀.₅: a Vision-Language-Action Model with Open-World Generalization},
  author       = {Physical Intelligence and Kevin Black and Noah Brown and James Darpinian and Karan Dhabalia and Danny Driess and Adnan Esmail and Michael Equi and Chelsea Finn and Niccolo Fusai and Manuel Y. Galliker and Dibya Ghosh and Lachy Groom and Karol Hausman and Brian Ichter and Szymon Jakubczak and Tim Jones and Liyiming Ke and Devin LeBlanc and Sergey Levine and Adrian Li-Bell and Mohith Mothukuri and Suraj Nair and Karl Pertsch and Allen Z. Ren and Lucy Xiaoyang Shi and Laura Smith and Jost Tobias Springenberg and Kyle Stachowicz and James Tanner and Quan Vuong and Homer Walke and Anna Walling and Haohuan Wang and Lili Yu and Ury Zhilinsky},
  year         = {2025},
  eprint       = {2504.16054},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2504.16054},
}
```

---

## License

This port follows the **Apache 2.0 License**, consistent with the original [OpenPI repository](https://github.com/Physical-Intelligence/openpi).
