# 训练预算参考

[返回参考目录](README.md) · [完整训练指令](../workflows/training/README.md)

以下由原 `AGENT_GUIDE.md` 第 7 节拆出，经验范围并非所有策略通用的收敛保证。

Use epochs and sample exposure to interpret the step budget. The ranges below are the original guide’s starting heuristics; convergence and useful training duration must be checked for the actual task.

## Rule of thumb

- **Typical total: 5–10 epochs.** Start at 5, eval, then decide if more helps.
- Very small datasets (< 30 episodes) may want slightly more epochs — but first, **collect more data**.
- VLAs with a pretrained vision backbone typically need **fewer** epochs than training from scratch.

## Steps ↔ epochs conversion

```text
samples_per_loop_step = per_device_batch_size * data_parallel_world_size
total_frames         = sum of usable frames over all episodes
loop_steps_per_epoch ≈ ceil(total_frames / samples_per_loop_step)
total_loop_steps     ≈ epochs * loop_steps_per_epoch

effective_batch_size = samples_per_loop_step * gradient_accumulation_steps
optimizer_updates    ≈ total_loop_steps / gradient_accumulation_steps
```

本地 `lerobot-train --steps` 计数的是读取 micro-batch 的训练循环步数，填写 `total_loop_steps`，不能把梯度累积后的优化器更新次数直接代入。普通采样下上式是近似值，尾批、丢弃帧和分布式补齐会影响精确步数；干预比例采样的两组曝光另见 [训练架构](../architecture/training.md)。

Single-GPU examples for `--batch_size=8` and no gradient accumulation:

| Dataset size            |  Frames | Steps / epoch | 5 epochs | 10 epochs |
| ----------------------- | ------: | ------------: | -------: | --------: |
| 50 eps × 30 s @ 30 fps  |  45,000 |        ~5,625 |      28k |       56k |
| 100 eps × 30 s @ 30 fps |  90,000 |       ~11,250 |      56k |      113k |
| 300 eps × 30 s @ 30 fps | 270,000 |       ~33,750 |     169k |      338k |

Pass the resulting total with `--steps=<N>`; eval at intermediate checkpoints (`outputs/train/.../checkpoints/`).

## Per-policy starting points (single-task, ~50 episodes)

| Policy         | Batch | Steps (first run) | Notes                                                             |
| -------------- | ----: | ----------------: | ----------------------------------------------------------------- |
| `act`          |  8–16 |           30k–80k | Usually converges under 50k for single-task.                      |
| `diffusion`    |  8–16 |          80k–150k | Benefits from longer training than ACT.                           |
| `smolvla`      |   4–8 |           30k–80k | Pretrained VLM → converges fast.                                  |
| `pi0` / `pi05` |   1–4 |           30k–80k | Memory-bound; use gradient accumulation for effective batch ≥ 16! |

## Batch size guidance

- **Bigger batch is preferable** for stable gradients on teleop data.
- If GPU memory is the bottleneck, use **gradient accumulation** to raise _effective_ batch without raising peak memory.
- Scale **learning rate** gently with batch; most LeRobot defaults work fine for a 2–4× batch change.

## Scale LR schedule & checkpoints with `--steps`

当前 `CosineDecayWithWarmupSchedulerConfig` 在训练总步数小于配置的 decay 步数时，会自动按比例缩短 warmup 和 decay；SmolVLA 使用这一调度器。缩短训练时先核对 `Auto-scaling LR scheduler` 日志和最终配置，不能假定未手动改 decay 就不会衰减。不同策略可能使用不同调度器，实现见 [schedulers.py](../../src/lerobot/optim/schedulers.py)。

需要明确控制实验设置时，可以显式指定 decay 和保存间隔。例如下面的片段用于支持这些字段的策略：

```bash
uv run --locked lerobot-train ... \
  --steps=5000 \
  --policy.scheduler_decay_steps=5000 \
  --save_freq=5000
```

同时核对 warmup 在短训练中所占比例。`save_freq` 按需要评估的间隔设置（例如每 1k–5k 个循环 step 保存）；保存频率不会因学习率调度器自动缩放而自动调整。

For checkpoint statistics, intervention sampling and resume semantics, see [training architecture](../architecture/training.md). Complete Piper recipes are maintained in [training workflows](../workflows/training/README.md).

## SmolVLA: unfreeze the vision encoder for real gains

SmolVLA ships with `freeze_vision_encoder=True`. Unfreezing usually **improves performance substantially** on specialized tasks, at the cost of more VRAM and slower steps. Enable with:

```bash
uv run --locked lerobot-train ... --policy.type=smolvla \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false
```

## Signals to stop / keep going

- Compare intermediate checkpoints on the task; a flat training loss alone is not a sufficient stopping rule.
- If loss keeps falling, check whether task success also improves before extending the budget.

上述带 `...` 的命令只是参数片段，需要合并到完整训练命令。干预比例采样下，一个虚拟 epoch 不等于每组样本刚好遍历一次；两组的曝光预算见 [训练架构](../architecture/training.md) 与 [历史实验](../research/training-experiments.md)。
