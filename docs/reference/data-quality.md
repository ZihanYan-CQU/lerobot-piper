# 数采质量参考

[返回参考目录](README.md) · [Piper 数采命令](../workflows/recording/README.md) · [审核与合并](../workflows/datasets/README.md)

原指南中的数采经验集中于此。下面的数量和时长是初次试验参考，实际需要由任务覆盖度和评估决定。

Good data beats clever models. Adopt these defaults and deviate only with evidence.

## Setup & ergonomics

- **Fix the rig and cameras** before touching the software. If the rig vibrates or the operator gets frustrated, fix that first — more bad data won't help.
- **Lighting matters more than resolution.** Diffuse, consistent light. Avoid moving shadows.
- **"Can you do the task from the camera view alone?"** If no, your cameras are wrong. Fix before recording.
- Enable **action interpolation** for rollouts when available for smoother trajectories.

## Practice before you record

- Do 5–10 demos without recording. Build a deliberate, repeatable strategy.
- Hesitant or inconsistent demos teach the model hesitation.

## Quality over speed

Deliberate, high-quality execution beats fast sloppy runs. Optimize for speed only **after** strategy is dialed in — never trade quality for it.

## Consistency within and across episodes

Same grasp, approach vector, and timing. Coherent strategies are much easier to learn than wildly varying movements.

## Start small, then extend (the golden rule)

- **First 50 episodes = constrained version** of the task: one object, fixed position, fixed camera setup, one operator.
- Train a quick ACT model. See what fails.
- **Then add diversity** along one axis at a time: more positions → more lighting → more objects → more operators.
- Don't try to collect the "perfect dataset" on day one. Iterate.

## Policy choice for beginners

See [policy selection](policy-selection.md) and [training budgets](training-budget.md).

## Recommended defaults for your first task

| Setting          | Value                                                                                                                                                 |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Episodes         | **50** to start, scale to 100–300 after first training                                                                                                |
| Episode length   | 20–45 s (shorter is fine for grasp/place)                                                                                                             |
| Reset time       | 10 s                                                                                                                                                  |
| FPS              | 30                                                                                                                                                    |
| Cameras          | **2 cameras recommended**: 1 fixed front + 1 wrist. Multi-view often outperforms single-view. A single fixed camera also works to keep things simple. |
| Task description | Short, specific, action-phrased sentence                                                                                                              |

## Troubleshooting signal

- Policy fails at one specific stage → record 10–20 more episodes **targeting that stage**.
- Policy flaps / oscillates → check demonstration consistency and training progress; re-record worst episodes (use **←** to redo).
- Policy ignores the object → check camera framing and lighting as well as model and input configuration.
