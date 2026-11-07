# 动作前缀微调 / trained RTC

[训练目录](README.md) · [部署指令](../deployment/README.md)

训练最大前缀长度 `D=25`，配套远端部署预算 `B=25`、触发阈值 `Q=25`、动作块 `C=50`。必须实际完成本训练后再使用 trained 推理。

所有命令都在训练服务器的仓库根目录执行。先完成[环境准备](../setup/README.md)，按需[下载模型与数据](downloads.md)。本页变量在本页定义，无需从其他页面复制变量。

## 路径配置

```bash
DATASET_NAMESPACE='sukasu403'
DATASET_BASE_DIR='/home/yzh/具身智能/训练数据'
MODEL_ROOT='/home/yzh/具身智能/微调模型/Lerobot'
BASE_MODEL_ROOT='/home/yzh/具身智能/基座模型/Lerobot'
TRAIN_LOG_DIR='/home/yzh/具身智能/训练日志/Lerobot'

mkdir -p "$TRAIN_LOG_DIR"
```

`DATASET_NAMESPACE` 是数据集的用户名或组织名。每次执行时，直接修改本页命令中 `--dataset.repo_id` 的数据集名称、`--dataset.root` 的子目录、`--policy.pretrained_path` 的模型目录和 checkpoint，以及 `--policy.text_tokenizer_name` 的 tokenizer 子目录；命令里的具体名称都是可替换的示例。`MODEL_ROOT` 同时用于读取起始模型和保存本轮模型，`RUN_NAME` 决定本轮输出子目录和训练日志文件名。

`use_pretrained_stats=true` 需要起始模型目录包含保存的 processor 配置和统计量。`intervention_sampling_ratio` 需要数据集中存在可用于分组的干预标记及两组有效样本。

## 单卡（T3）

```bash
RUN_NAME='pi05_shape_matching_trained_delay25_single_b32_s10000_ratio30_lr5e-6'

CUDA_VISIBLE_DEVICES=4 nohup uv run --no-sync lerobot-train \
  --dataset.repo_id="${DATASET_NAMESPACE}/dagger_merged" \
  --dataset.root="${DATASET_BASE_DIR}/shape-matching_89_with_dagger_30" \
  --policy.type=pi05 \
  --policy.pretrained_path="${MODEL_ROOT}/pi0.5_shapeMatching_b32_s20000/checkpoints/020000/pretrained_model" \
  --policy.text_tokenizer_name="${BASE_MODEL_ROOT}/paligemma-3b-pt-224" \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.rtc_training_max_delay=25 \
  --policy.optimizer_lr=5e-6 \
  --policy.scheduler_decay_lr=5e-7 \
  --policy.scheduler_decay_steps=10000 \
  --steps=10000 \
  --output_dir="${MODEL_ROOT}/${RUN_NAME}" \
  --job_name="$RUN_NAME" \
  --batch_size=32 \
  --accelerator.gradient_accumulation.steps=1 \
  --num_workers=8 \
  --save_freq=1000 \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --seed=1000 \
  --use_pretrained_stats=true \
  --intervention_sampling_ratio=0.3 \
  > "$TRAIN_LOG_DIR/$RUN_NAME.log" 2>&1 &

TRAIN_PID=$!
echo "训练进程 PID：$TRAIN_PID"
```

## 双卡（T4）

```bash
RUN_NAME='pi05_shape_matching_trained_delay25_multi_b32_s20000_ratio50'

CUDA_VISIBLE_DEVICES=4,5 nohup uv run --no-sync torchrun \
  --standalone \
  --nproc-per-node=2 \
  --module lerobot.scripts.lerobot_train \
  --dataset.repo_id="${DATASET_NAMESPACE}/dagger_merged" \
  --dataset.root="${DATASET_BASE_DIR}/shape-matching_89_with_dagger_30" \
  --policy.type=pi05 \
  --policy.pretrained_path="${MODEL_ROOT}/pi0.5_shapeMatching_b32_s20000/checkpoints/020000/pretrained_model" \
  --policy.text_tokenizer_name="${BASE_MODEL_ROOT}/paligemma-3b-pt-224" \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.rtc_training_max_delay=25 \
  --steps=20000 \
  --output_dir="${MODEL_ROOT}/${RUN_NAME}" \
  --job_name="$RUN_NAME" \
  --batch_size=16 \
  --accelerator.gradient_accumulation.steps=1 \
  --num_workers=8 \
  --save_freq=1000 \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --seed=1000 \
  --use_pretrained_stats=true \
  --intervention_sampling_ratio=0.5 \
  > "$TRAIN_LOG_DIR/$RUN_NAME.log" 2>&1 &

TRAIN_PID=$!
echo "训练进程 PID：$TRAIN_PID"
```

双卡配方每卡 batch 为 16，有效总 batch 为 32。改变卡数时，同时修改 `CUDA_VISIBLE_DEVICES`、`--nproc-per-node` 和每卡 `--batch_size`。有效总 batch = 每卡 batch × 卡数 × 梯度累积步数。

## 查看进度和取出模型

启动后在同一终端执行：

```bash
tail -f "$TRAIN_LOG_DIR/$RUN_NAME.log"
```

`Ctrl+C` 只退出日志查看，后台训练继续运行。每轮新训练使用新的 `RUN_NAME`，避免覆盖日志或复用已有输出目录。单卡示例输出到 `${MODEL_ROOT}/${RUN_NAME}/checkpoints/010000/pretrained_model`，双卡示例对应 `checkpoints/020000/pretrained_model`。把所选完整 checkpoint 路径填入部署客户端的 `--pretrained_name_or_path`；该路径由推理服务器读取。
