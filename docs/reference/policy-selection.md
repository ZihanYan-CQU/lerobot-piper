# 策略选择与原有性能快照

[返回参考目录](README.md)

以下内容迁自原 `AGENT_GUIDE.md` 第 6 节，保留原有测量条件。这是既有参考快照，本次整理没有重新测量；不用于保证当前机器的显存占用或训练耗时。

Match the policy to the user's **GPU memory** and **time budget**. Numbers below come from an internal profiling run (one training update per policy). They are **indicative only** — see caveats.

## Profiling snapshot (indicative)

Training duration depends on the task and dataset; see [training budgets](training-budget.md).

> **Human-facing version:** the [Compute Hardware Guide](../source/hardware_guide.mdx) reuses the table below and adds a cloud-GPU tier guide and a Hugging Face Jobs pointer.

| Policy      | Batch | Update (ms) | Peak GPU mem (GB) | Best for                                                                                         |
| ----------- | ----: | ----------: | ----------------: | ------------------------------------------------------------------------------------------------ |
| `act`       |     4 |    **83.9** |          **0.94** | First-time users, laptops, single-task. Fast and reliable.                                       |
| `diffusion` |     4 |       168.6 |              4.94 | Multi-modal action distributions; needs mid-range GPU.                                           |
| `smolvla`   |     1 |       357.8 |              3.93 | Language-conditioned, multi-task, small VLA. **Unfreeze vision encoder for big gains** (see [training budgets](training-budget.md)). |
| `xvla`      |     1 |       731.6 |             15.52 | Large VLA, multi-task.                                                                           |
| `wall_x`    |     1 |       716.5 |             15.95 | Large VLA with world-model objective.                                                            |
| `pi0`       |     1 |       940.3 |             15.50 | Strong large VLA baseline (Physical Intelligence).                                               |
| `pi05`      |     1 |      1055.8 |             16.35 | Newer π policy; similar footprint to `pi0`.                                                      |

**Critical caveats:**

- **Optimizer:** measured with **SGD**. LeRobot's default is **AdamW**, which keeps extra optimizer state → **peak memory will be noticeably higher** with the default, especially for `pi0`, `pi05`, `wall_x`, `xvla`.
- **Batch size:** the large policies were profiled at batch 1. In practice use a **larger batch** for stable training (see [batch size guidance](training-budget.md#batch-size-guidance)). Memory scales roughly linearly with batch.

## Decision rules

- **< 8 GB VRAM (laptop, 3060, M-series Mac):** → `act`. Maybe `diffusion` if you have ~6–8 GB free.
- **12–16 GB VRAM (4070/4080, A4000):** → `smolvla` with defaults, or `act`/`diffusion` with larger batch. `pi0`/`pi05`/`wall_x`/`xvla` feasible only with small batch + gradient accumulation.
- **24+ GB VRAM (3090/4090/A5000):** → any policy. Prefer `smolvla` (unfrozen) for multi-task; `act` for single-task grasp-and-place (still often the best ROI). Could experiment with `pi0` or `pi05` or `xvla`
- **80 GB (A100/H100):** → any, with healthy batch. `pi05`, `xvla`, `wall_x` become comfortable.
- **CPU only:** → don't train here. Use Google Colab (see [`docs/source/notebooks.mdx`](../source/notebooks.mdx)) or a rented GPU.
