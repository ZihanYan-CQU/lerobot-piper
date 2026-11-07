# 中断后继续同一轮训练

[返回训练目录](README.md) · [续训语义](../../architecture/training.md#续训与采样位置) · [训练排错](../../troubleshooting/training.md)

本页用于恢复被中断的训练，包括优化器、调度器、步数和采样状态。用新增数据开始下一轮微调时，使用 [guided](guided.md) 或 [trained](trained.md) 配方，不要套用这里的 `resume=true`。

在训练服务器的仓库根目录运行。先把路径改成要恢复的 checkpoint；该目录必须保留完整的 `pretrained_model` 和同级 `training_state`，只有模型权重不能恢复训练状态。

```bash
TRAIN_CONFIG='/path/to/dagger_round_2/checkpoints/001000/pretrained_model/train_config.json'
```

## 单卡

下面沿用单卡配方的 GPU 4，按实际机器调整可见设备：

```bash
CUDA_VISIBLE_DEVICES=4 uv run --no-sync lerobot-train \
  --config_path="$TRAIN_CONFIG" \
  --resume=true
```

## 多卡

保留原训练的数据并行规模。下面对应原双卡配方，不要直接用它恢复其他规模的任务：

```bash
CUDA_VISIBLE_DEVICES=4,5 uv run --no-sync torchrun \
  --standalone \
  --nproc-per-node=2 \
  --module lerobot.scripts.lerobot_train \
  --config_path="$TRAIN_CONFIG" \
  --resume=true
```

`steps` 从保存的训练配置恢复，表示目标总循环步数，不是追加步数。例如 checkpoint 在 1000 步、目标为 2715 步，则继续剩余 1715 步。如需修改目标总步数，显式添加 `--steps=目标总步数`，同时核对学习率调度计划。

复现同一采样序列需要保持代码 / sampler 版本、数据及顺序 / 标记、筛选条件、seed、ratio、每卡 batch size 和数据并行进程数一致。旧版本 sampler 的 checkpoint 不能承诺恢复旧版本原本会取到的后续序列。详细边界见 [训练架构](../../architecture/training.md#续训与采样位置)。
