<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# LeRobot / Piper 操作手册

本机仓库：`/home/cpy/桌面/yzh/piper/lerobot`。日常操作统一从本页开始；以下命令在对应机器的仓库根目录执行。

| 要做什么 | 入口 |
| --- | --- |
| 用现有 JSON 启动普通部署 | [Deployment 快速启动](#deployment-quickstart) |
| 安装或迁移运行环境 | [环境与路径](#环境与路径) |
| 下载数据、guided / trained 单卡或多卡训练 | [训练](#训练) |
| 启动 deployment / DAgger 推理服务 | [服务端推理](#服务端推理) |
| 建立 SSH 隧道，启动机器人或接管采集 | [客户端推理](#客户端推理) |
| 审核合并数据、绑定设备、查看 GPU | [本地工具](#local-tools) |
| 排查环境、进程和存储问题 | [常用检查](#troubleshooting) |
| 查训练原理、DAgger、RECAP 与部署实现 | [专题参考](#references) |
| 查看通用教程、构建文档 | [文档维护](#documentation) |

<a id="deployment-quickstart"></a>

## Deployment 快速启动

先按[环境与路径](#环境与路径)安装各端依赖。配置保存在仓库中，按实际设备和模型修改后即可复用：

| 配置 | 使用前检查 |
| --- | --- |
| [服务端 JSON](../src/lerobot/async_inference/config/policy_server_deployment.json) | `host`、`port`、`cuda_visible_devices`、`fault_log_path` |
| [客户端 JSON](../src/lerobot/async_inference/config/robot_client_deployment.json) | `server_address`、`pretrained_name_or_path`、`robot.cameras`、`task` |

模型路径由服务端读取，相机设备路径属于客户端。JSON 直接填写实际值，不展开 `$变量`；命令行参数可覆盖同名配置。服务端 `rtc.mode` 与客户端 `rtc_mode` 必须一致。

先在 GPU 服务器启动：

```bash
uv run --locked --extra pi05-server python -m lerobot.async_inference.server.policy_server_deployment   --config_path=src/lerobot/async_inference/config/policy_server_deployment.json
```

按[客户端推理](#客户端推理)建立 SSH 隧道，再在机器人电脑启动：

```bash
uv run --locked --extra piper-client python -m lerobot.async_inference.client.robot_client_deployment   --config_path=src/lerobot/async_inference/config/robot_client_deployment.json
```

当前 JSON 使用 guided RTC，客户端图像传输设为 `openpi_uint8`、服务端精度档位设为 `lerobot`；下方完整命令配方使用自己的显式参数，两者不应视为完全相同的配置。JSON 参数覆盖、GPU 选择及模块迁移说明见[异步推理目录](../src/lerobot/async_inference/README.md#deployment-json-配置)。

## 环境与路径

**运行环境与迁移：** 以下使用 Bash，并在各自机器的 LeRobot 仓库根目录执行。Python 版本由 `.python-version` 固定为 3.12.14，项目配置要求使用 uv 管理的 Python，避免借用 Conda 解释器及其动态库。依赖由 `pyproject.toml` 声明、`uv.lock` 锁定。迁移时携带完整代码及这三个文件，在新机器用 uv 重建 `.venv`，不要复制旧机器的 `.venv` 或 Conda 环境。

新机器先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 和 Git，再按用途选择一条同步命令：

| 机器用途 | 安装命令 | 包含的依赖 |
| --- | --- | --- |
| Pi0.5 训练，单卡 / 多卡、guided / trained | `uv sync --locked --extra pi05-training` | 训练、数据集、Pi0.5 |
| 服务端推理，deployment / dagger | `uv sync --locked --extra pi05-server` | gRPC、Pi0.5、推理诊断 |
| Piper 客户端，deployment / dagger | `uv sync --locked --extra piper-client` | gRPC、pyAgxArm、CAN、RealSense、键盘控制、DAgger 数据集 |

`piper-client` 已包含 `intelrealsense`，在 Linux 上会安装 `pyrealsense2`；安装和运行客户端时无需额外添加 `--extra intelrealsense`。

同一机器承担多种用途时，一次列出全部所需 extra。例如：

```bash
uv sync --locked \
  --extra pi05-training \
  --extra pi05-server \
  --extra piper-client
```

`uv sync` 默认会移除未选中的依赖，因此更新环境时也应保留这台机器需要的全部 extra。下面每条主命令使用 `uv run --locked --extra ...`，明确选择运行所需依赖；它不会自动升级锁定版本。只有环境已同步并验证后，才使用 `uv run --no-sync` 跳过同步。

默认 Python 包源已在项目中设为清华源；Linux 的 `torch`、`torchvision` 仍使用项目指定的 PyTorch CUDA 12.8 源。`pyAgxArm` 从官方 GitHub 仓库 `agilexrobotics/pyAgxArm` 安装，固定到当前使用的提交 `4b0f06585db3324222616999afeda9037df2e8bd`，不依赖某台机器上的本地源码目录。清华源不会代理 GitHub 或 PyTorch 专用源。

uv 管理 Python 依赖；新机器仍需配置 NVIDIA 驱动、FFmpeg 共享库，以及客户端的 CAN 接口、相机设备和访问权限。DAgger 的全局键盘接管还需要可用的桌面会话。OpenPI 使用自己的仓库和环境，本文件的 Python 版本及 extra 仅用于 LeRobot。

无需激活 Conda，也不要默认把 `$CONDA_PREFIX/lib` 加到 `LD_LIBRARY_PATH`。若终端仍显示 `(lerobot)`，`uv run` 仍使用项目 `.venv`；没有指定 `--python` 的 `uv pip install` 可能选中 Conda。新增依赖应写入项目并重新锁定，例如客户端依赖用 `uv add --optional piper-client 包名`，随后连同更新的 `pyproject.toml` 和 `uv.lock` 一起迁移。若 `--locked` 提示锁文件不一致，先检查是否复制了配套文件或设置了不同的 `UV_DEFAULT_INDEX`，由维护端确认依赖变更后执行 `uv lock`。

**路径约定：** 数据集、基座模型和微调模型的公共配置只保存基础目录或命名空间；具体数据集名称、模型目录及 checkpoint 在每条执行命令中指定。`/home/yzh/具身智能/...` 是服务器路径，客户端传入的模型路径也由服务器读取。请按实际机器修改基础目录，并在执行时选择 GPU、数据集和模型。服务端与客户端应使用同一版本的代码，并选择相同的用途和 RTC 模式。

**目录规则：** 统一使用“用途目录/框架或来源”的结构。以下路径均相对于 `/home/yzh/具身智能`：

| 用途 | LeRobot | OpenPI |
| --- | --- | --- |
| 基座模型 | `基座模型/Lerobot` | `基座模型/OpenPI` |
| 微调模型 | `微调模型/Lerobot` | `微调模型/OpenPI` |
| 训练日志 | `训练日志/Lerobot` | `训练日志/OpenPI` |

本手册使用 LeRobot 目录。原来带 CodeX 前缀的调试日志也按同一规则改为 `DEBUG日志/CodeX`；训练数据基础目录为 `训练数据`。

## 训练

以下保留 4 条训练、4 条服务端、4 条客户端完整配方及 3 条下载命令。先执行所在章节的公共配置，再展开选择一条方案；不要整篇一次执行。

在训练服务器执行。guided-RTC 对应普通 Pi0.5 微调，推理时再启用 guided 引导；trained-RTC 则需要在训练时实际学习动作前缀条件，不能只修改已有模型的配置。

| 设置 | 单卡 | 多卡 |
| --- | --- | --- |
| GPU | `4` | `4,5`，共 2 张 |
| 每卡 batch size | `32` | `16` |
| 梯度累积 | `1` | `1` |
| 有效总 batch size | `32` | `32` |
| 训练步数 | `10000` | `20000` |
| 干预数据目标采样比例 | `0.3` | `0.5` |
| 学习率 / 最终学习率 | `5e-6` / `5e-7` | 保留原文件的省略设置，使用当前 Pi0.5 默认值 |

单卡、多卡保留了原文件中不同的训练设置，因此两者不仅是 GPU 数量不同。在当前 `policy.type=pi05` 配置下，多卡默认学习率为 `2.5e-5`、最终学习率为 `2.5e-6`。如需做相同配置的单卡与多卡对照，应同时对齐步数、采样比例、学习率和调度参数。

两种 RTC 模式的训练区别是 `policy.rtc_training_max_delay`：guided 使用 `0`，trained 使用 `25`。以下均显式设置 `chunk_size=50`。

**训练公共配置——每个新终端先执行一次：**

```bash
DATASET_NAMESPACE='sukasu403'
DATASET_BASE_DIR='/home/yzh/具身智能/训练数据'
MODEL_ROOT='/home/yzh/具身智能/微调模型/Lerobot'
BASE_MODEL_ROOT='/home/yzh/具身智能/基座模型/Lerobot'
TRAIN_LOG_DIR='/home/yzh/具身智能/训练日志/Lerobot'

mkdir -p "$TRAIN_LOG_DIR"
```

`DATASET_NAMESPACE` 是数据集的用户名或组织名。每次执行时，直接修改下面命令中 `--dataset.repo_id` 的数据集名称、`--dataset.root` 的子目录、`--policy.pretrained_path` 的模型目录和 checkpoint，以及 `--policy.text_tokenizer_name` 的 tokenizer 子目录；命令里的具体名称都是可替换的示例。`MODEL_ROOT` 同时用于读取起始模型和保存本轮模型，`RUN_NAME` 决定本轮输出子目录和训练日志文件名。

`use_pretrained_stats=true` 需要起始模型目录包含保存的 processor 配置和统计量。`intervention_sampling_ratio` 需要数据集中存在可用于分组的干预标记及两组有效样本。

**训练前下载——先执行上面的训练公共配置，再按需选择：**

**D1：下载 PaliGemma 资源，排除 `.safetensors` 权重文件。**

```bash
uv run --locked hf download google/paligemma-3b-pt-224 \
  --local-dir "${BASE_MODEL_ROOT}/paligemma-3b-pt-224" \
  --exclude "*.safetensors"
```

下载目录与下文训练命令的 `--policy.text_tokenizer_name` 一致。`"*.safetensors"` 保留引号，交给下载工具匹配。

**D2：下载训练数据集的 `v1.0` 版本。**

```bash
uv run --locked hf download "${DATASET_NAMESPACE}/shape-matching_89_with_dagger_30" \
  --local-dir "${DATASET_BASE_DIR}/shape-matching_89_with_dagger_30" \
  --repo-type dataset \
  --revision v1.0
```

训练这份下载的数据时，将具体训练命令中的 `--dataset.repo_id` 改为 `"${DATASET_NAMESPACE}/shape-matching_89_with_dagger_30"`，替换原示例中的 `dagger_merged`；`--dataset.root` 已与上述下载目录一致。

**D3：下载 Pi0.5 基座模型。**

```bash
uv run --locked hf download lerobot/pi05_base \
  --local-dir "${BASE_MODEL_ROOT}/pi05_base"
```

基座模型保存在 `基座模型/Lerobot` 下；具体训练选择哪个起始模型，仍由执行命令中的 `--policy.pretrained_path` 指定。

### guided-RTC

#### 单卡

<details>
<summary>T1：普通微调，单卡 32 batch，训练 10000 步。</summary>

```bash
RUN_NAME='pi05_shape_matching_guided_single_b32_s10000_ratio30_lr5e-6'

CUDA_VISIBLE_DEVICES=4 nohup uv run --locked --extra pi05-training lerobot-train \
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
  --policy.rtc_training_max_delay=0 \
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

</details>

#### 多卡

<details>
<summary>T2：普通微调，双卡各 16 batch，训练 20000 步。</summary>

```bash
RUN_NAME='pi05_shape_matching_guided_multi_b32_s20000_ratio50'

CUDA_VISIBLE_DEVICES=4,5 nohup uv run --locked --extra pi05-training torchrun \
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
  --policy.rtc_training_max_delay=0 \
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

</details>

多卡数量变化时，同时修改 `CUDA_VISIBLE_DEVICES`、`--nproc-per-node` 和每卡 `--batch_size`。有效总 batch size = 每卡 batch size × 卡数 × 梯度累积步数。

### trained-RTC

训练最大前缀长度设为 `25`，对应后文服务端 `trained.delay_steps=25`、客户端 `trained.queue_threshold=25`、动作块长度 `50`。

#### 单卡

<details>
<summary>T3：开启 trained-RTC 微调，单卡 32 batch，训练 10000 步。</summary>

```bash
RUN_NAME='pi05_shape_matching_trained_delay25_single_b32_s10000_ratio30_lr5e-6'

CUDA_VISIBLE_DEVICES=4 nohup uv run --locked --extra pi05-training lerobot-train \
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

</details>

#### 多卡

<details>
<summary>T4：开启 trained-RTC 微调，双卡各 16 batch，训练 20000 步。</summary>

```bash
RUN_NAME='pi05_shape_matching_trained_delay25_multi_b32_s20000_ratio50'

CUDA_VISIBLE_DEVICES=4,5 nohup uv run --locked --extra pi05-training torchrun \
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

</details>

**查看日志：** 启动后在同一终端执行 `tail -f "$TRAIN_LOG_DIR/$RUN_NAME.log"`；按 `Ctrl+C` 只退出日志查看，后台训练继续运行。

四种方案使用独立的输出目录和日志。再次发起一轮新训练时修改 `RUN_NAME`，避免复用已有目录或覆盖日志。单卡完成后的模型位于 `${MODEL_ROOT}/${RUN_NAME}/checkpoints/010000/pretrained_model`；多卡对应 `checkpoints/020000/pretrained_model`。推理时，将本轮模型子目录和所选 checkpoint 填入具体客户端命令的 `--pretrained_name_or_path`。

## 服务端推理

在 GPU 服务器执行，选择与客户端对应的一条命令。这里保留 `127.0.0.1:8080`，供客户端通过 SSH 隧道连接；同一端口同时只运行一个服务端。

服务端启动命令不指定模型路径，模型由客户端的 `pretrained_name_or_path` 指定。

**服务端公共配置——每个新终端先执行一次：**

```bash
SERVER_HOST='127.0.0.1'
SERVER_PORT=8080
FAULT_LOG_DIR='/home/yzh/具身智能/DEBUG日志/'

mkdir -p "$FAULT_LOG_DIR"
export PYTHONFAULTHANDLER=1
unset CUDA_LAUNCH_BLOCKING TORCH_SHOW_CPP_STACKTRACES
```

原文件中的 `CUDA_LAUNCH_BLOCKING=1` 用于故障定位，会改变 CUDA 执行时序和延迟，因此常规推理配置取消该设置；需要排查 CUDA 故障时再在启动前开启。

### deployment

#### guided-RTC

<details>
<summary>S1：GPU 7，guided 引导窗口 20，保留原文件中的 OpenPI 编译配置。</summary>

```bash
CUDA_VISIBLE_DEVICES=7 uv run --locked --extra pi05-server python -m lerobot.async_inference.server.policy_server_deployment \
  --host="$SERVER_HOST" \
  --port="$SERVER_PORT" \
  --fps=30 \
  --inference_latency=0 \
  --rtc.mode=guided \
  --rtc.execution_horizon=20 \
  --rtc.max_guidance_weight=10 \
  --rtc.prefix_attention_schedule=LINEAR \
  --reuse_last_policy=true \
  --pi05_inference.compile_mode=max-autotune \
  --pi05_inference.profile=openpi \
  --pi05_inference.warmup_timeout_s=1800 \
  --pi05_inference.compile_model=true \
  --debug_rtc_diagnostics=false \
  --debug_cuda_sync=false \
  --rtc.debug=false \
  --fault_log_path="$FAULT_LOG_DIR/deployment_guided_crash.log"
```

</details>

#### trained-RTC

<details>
<summary>S2：GPU 7，固定前缀预算 25 步。</summary>

```bash
CUDA_VISIBLE_DEVICES=7 uv run --locked --extra pi05-server python -m lerobot.async_inference.server.policy_server_deployment \
  --host="$SERVER_HOST" \
  --port="$SERVER_PORT" \
  --fps=30 \
  --inference_latency=0 \
  --rtc.mode=trained \
  --trained.delay_steps=25 \
  --reuse_last_policy=true \
  --pi05_inference.warmup_timeout_s=1800 \
  --debug_rtc_diagnostics=false \
  --debug_cuda_sync=false \
  --rtc.debug=false \
  --fault_log_path="$FAULT_LOG_DIR/deployment_trained_crash.log"
```

</details>

### dagger

#### guided-RTC

<details>
<summary>S3：GPU 2，guided 引导窗口 25，配合人工接管采集客户端。</summary>

```bash
CUDA_VISIBLE_DEVICES=2 uv run --locked --extra pi05-server python -m lerobot.async_inference.server.policy_server_dagger \
  --host="$SERVER_HOST" \
  --port="$SERVER_PORT" \
  --fps=30 \
  --inference_latency=0 \
  --rtc.mode=guided \
  --rtc.execution_horizon=25 \
  --reuse_last_policy=true \
  --pi05_inference.warmup_timeout_s=1800 \
  --debug_rtc_diagnostics=false \
  --debug_cuda_sync=false \
  --rtc.debug=false \
  --fault_log_path="$FAULT_LOG_DIR/dagger_guided_crash.log"
```

</details>

#### trained-RTC

<details>
<summary>S4：GPU 2，固定前缀预算 25 步，配合 trained-RTC 人工接管采集客户端。</summary>

```bash
CUDA_VISIBLE_DEVICES=2 uv run --locked --extra pi05-server python -m lerobot.async_inference.server.policy_server_dagger \
  --host="$SERVER_HOST" \
  --port="$SERVER_PORT" \
  --fps=30 \
  --inference_latency=0 \
  --rtc.mode=trained \
  --trained.delay_steps=25 \
  --reuse_last_policy=true \
  --pi05_inference.warmup_timeout_s=1800 \
  --debug_rtc_diagnostics=false \
  --debug_cuda_sync=false \
  --rtc.debug=false \
  --fault_log_path="$FAULT_LOG_DIR/dagger_trained_crash.log"
```

</details>

**模型复用：** 上述命令保留原文件的 `reuse_last_policy=true`。DAgger 每轮训练若覆盖同一个模型目录，应将对应服务端命令改为 `--reuse_last_policy=false` 后重启，避免重连时仍复用旧权重；当前 DAgger 代码的默认值也是 `false`。一次连接内的多次接管不会因此反复加载模型。

**参数清理：** 已移除原文件的旧模块命令、关闭 RTC 的方案和已删除的 `rtc_full_vjp` 参数。S1 保留显式编译设置；S2、S3、S4 的编译与精度配置沿用各自 checkpoint。

## 客户端推理

在连接 Piper 和相机的机器人电脑执行。顺序为：**启动服务端 → 建立 SSH 隧道 → 执行客户端公共配置 → 启动对应客户端**。客户端保持前台运行，以便输入控制命令或使用键盘接管。

**SSH 隧道——跨机器连接时执行一次：**

```bash
env -u LD_LIBRARY_PATH ssh \
  -4 -fN \
  -o ExitOnForwardFailure=yes \
  -J jumper@cps.cslcqu.cn:6666 \
  -L 127.0.0.1:8080:127.0.0.1:8080 \
  -p 1143 \
  yzh@192.168.1.50
```

已建立隧道时不必重复执行。同机运行服务端和客户端时可以跳过。这里仅对 SSH 进程取消 `LD_LIBRARY_PATH`，当前终端的 Python 动态库设置继续保留。

**客户端公共配置——每个新终端先执行一次：**

```bash
SERVER_ADDRESS='127.0.0.1:8080'
TASK='Place the block on the shape-matching board.'

# 服务器上的模型基础目录；具体模型和 checkpoint 在执行命令中填写。
MODEL_ROOT='/home/yzh/具身智能/微调模型/Lerobot'

# DAgger 数据集命名空间；每轮采集的具体名称在执行命令中填写。
DAGGER_DATASET_NAMESPACE='block'

CAMERAS='{
  "top": {
    "type": "opencv",
    "index_or_path": "/dev/video_top",
    "width": 640,
    "height": 480,
    "fps": 30,
    "rotation": 0
  },
  "wrist": {
    "type": "opencv",
    "index_or_path": "/dev/video_wrist",
    "width": 640,
    "height": 480,
    "fps": 30,
    "rotation": 180
  }
}'
```

四条客户端命令统一使用 `MODEL_ROOT` 指定的基础目录。每次执行时，修改 `--pretrained_name_or_path` 中的模型子目录和 checkpoint；DAgger 同时在 `--dagger_dataset_repo_id` 中填写本轮采集的数据集名称。trained-RTC 命令中的示例模型应替换为实际完成 trained-RTC 训练的 checkpoint。

未显式设置时，客户端默认使用 `lerobot_float32`，在本地完成模型尺寸缩放后以 float32 无损压缩上传。下方 C1 已显式设置 `--image_transport_mode=openpi_uint8`，C2–C4 沿用默认值。要在其他命令中对比 OpenPI 的 PIL 缩放/补边与 RGB uint8 上传，增加 `--image_transport_mode=openpi_uint8`，保留 `--resize_images_on_client=true`。两端需同步更新代码并重启，服务端无需新增参数；`--resize_images_on_client=false` 则上传原图。两种模式的像素数值并不等价，详见 [Deployment 第 15 节](../src/lerobot/yzh/README_DEPLOYMENT.md#15-π05-图像上传float32-无损压缩--openpi-uint8-可选)。

相机设备路径属于客户端。以下命令均只在当前客户端进程中清除代理变量，供 gRPC 连接使用。

### deployment

#### guided-RTC

<details>
<summary>C1：配套 S1，50 步动作块，EMA 系数均为 0.25。</summary>

```bash
env \
  -u http_proxy \
  -u https_proxy \
  -u HTTP_PROXY \
  -u HTTPS_PROXY \
  -u all_proxy \
  -u ALL_PROXY \
  -u grpc_proxy \
  -u GRPC_PROXY \
uv run --locked --extra piper-client python -m lerobot.async_inference.client.robot_client_deployment \
  --server_address="$SERVER_ADDRESS" \
  --robot.type=piper_follower \
  --robot.cameras="$CAMERAS" \
  --policy_type=pi05 \
  --pretrained_name_or_path="${MODEL_ROOT}/pi0.5_shapeMatching_b32_s20000/checkpoints/020000/pretrained_model" \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=1.0 \
  --fps=30 \
  --interpolation_multiplier=2 \
  --adaptive_rtt_trigger=true \
  --rtt_trigger_margin_steps=5 \
  --robot.teleop_gripper_alpha=0.25 \
  --robot.teleop_joint_alpha=0.25 \
  --reset_to_initial_position=true \
  --reset_duration_s=3.0 \
  --reset_fps=50 \
  --rtc_mode=guided \
  --resize_images_on_client=true \
  --image_transport_mode=openpi_uint8 \
  --continuous_observation=true \
  --observation_max_age_s=0.1 \
  --startup_calibration_runs=3 \
  --startup_calibration_timeout_s=1800 \
  --latency_history_ttl_s=300 \
  --observation_capture_fps=30 \
  --debug_rtc_diagnostics=false \
  --log_rtc_timing=false \
  --debug_visualize_queue_size=false \
  --suppress_can_debug_logs=true \
  --task="$TASK"
```

</details>

#### trained-RTC

<details>
<summary>C2：配套 S2，50 步动作块，固定触发阈值 25，EMA 系数均为 1.0。</summary>

```bash
env \
  -u http_proxy \
  -u https_proxy \
  -u HTTP_PROXY \
  -u HTTPS_PROXY \
  -u all_proxy \
  -u ALL_PROXY \
  -u grpc_proxy \
  -u GRPC_PROXY \
uv run --locked --extra piper-client python -m lerobot.async_inference.client.robot_client_deployment \
  --server_address="$SERVER_ADDRESS" \
  --robot.type=piper_follower \
  --robot.cameras="$CAMERAS" \
  --policy_type=pi05 \
  --pretrained_name_or_path="${MODEL_ROOT}/pi0.5_shapeMatching_b32_s20000_trainedRTC_delay25/checkpoints/020000/pretrained_model" \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --fps=30 \
  --interpolation_multiplier=2 \
  --robot.teleop_gripper_alpha=1.0 \
  --robot.teleop_joint_alpha=1.0 \
  --reset_to_initial_position=true \
  --reset_duration_s=3.0 \
  --reset_fps=50 \
  --rtc_mode=trained \
  --trained.queue_threshold=25 \
  --trained.replan_on_result=true \
  --resize_images_on_client=true \
  --continuous_observation=true \
  --observation_max_age_s=0.1 \
  --startup_calibration_runs=3 \
  --startup_calibration_timeout_s=1800 \
  --latency_history_ttl_s=300 \
  --observation_capture_fps=30 \
  --debug_rtc_diagnostics=false \
  --log_rtc_timing=false \
  --debug_visualize_queue_size=false \
  --suppress_can_debug_logs=true \
  --task="$TASK"
```

</details>

**Deployment 启动后的交互：**

| 输入 | 作用 |
| --- | --- |
| `/start` | 使用配置中的任务开始一轮运行 |
| `/task 新任务描述` | 修改任务描述 |
| `/reset` | 停止当前轮次、清空动作并按配置复位 |
| `/status` | 查看状态 |
| `/stop` | 断开并退出客户端 |

### dagger

#### guided-RTC

<details>
<summary>C3：配套 S3，键盘接管，采集 10 个纠正 episode。</summary>

```bash
env \
  -u http_proxy \
  -u https_proxy \
  -u HTTP_PROXY \
  -u HTTPS_PROXY \
  -u all_proxy \
  -u ALL_PROXY \
  -u grpc_proxy \
  -u GRPC_PROXY \
uv run --locked --extra piper-client python -m lerobot.async_inference.client.robot_client_dagger \
  --server_address="$SERVER_ADDRESS" \
  --robot.type=piper_follower \
  --robot.cameras="$CAMERAS" \
  --policy_type=pi05 \
  --pretrained_name_or_path="${MODEL_ROOT}/pi0.5_shapeMatching_b32_s20000_dagger_b32_s20000/checkpoints/005000/pretrained_model" \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --fps=30 \
  --interpolation_multiplier=2 \
  --adaptive_rtt_trigger=true \
  --rtt_trigger_margin_steps=5 \
  --robot.teleop_gripper_alpha=1.0 \
  --robot.teleop_joint_alpha=0.25 \
  --reset_to_initial_position=true \
  --reset_duration_s=3.0 \
  --reset_fps=50 \
  --rtc_mode=guided \
  --resize_images_on_client=true \
  --continuous_observation=true \
  --observation_max_age_s=0.1 \
  --startup_calibration_runs=3 \
  --startup_calibration_timeout_s=1800 \
  --latency_history_ttl_s=300 \
  --observation_capture_fps=30 \
  --debug_rtc_diagnostics=false \
  --log_rtc_timing=false \
  --debug_visualize_queue_size=false \
  --suppress_can_debug_logs=true \
  --task="$TASK" \
  --dagger_dataset_repo_id="${DAGGER_DATASET_NAMESPACE}/dagger_0-10" \
  --dagger_dataset_resume=false \
  --dagger_dataset_push_to_hub=false \
  --dagger_num_episodes=10 \
  --dagger_input_device=keyboard
```

</details>

#### trained-RTC

<details>
<summary>C4：配套 S4，沿用 trained deployment 的 EMA 设置，键盘接管，采集 10 个纠正 episode。</summary>

```bash
env \
  -u http_proxy \
  -u https_proxy \
  -u HTTP_PROXY \
  -u HTTPS_PROXY \
  -u all_proxy \
  -u ALL_PROXY \
  -u grpc_proxy \
  -u GRPC_PROXY \
uv run --locked --extra piper-client python -m lerobot.async_inference.client.robot_client_dagger \
  --server_address="$SERVER_ADDRESS" \
  --robot.type=piper_follower \
  --robot.cameras="$CAMERAS" \
  --policy_type=pi05 \
  --pretrained_name_or_path="${MODEL_ROOT}/pi0.5_shapeMatching_b32_s20000_trainedRTC_delay25/checkpoints/020000/pretrained_model" \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --fps=30 \
  --interpolation_multiplier=2 \
  --robot.teleop_gripper_alpha=1.0 \
  --robot.teleop_joint_alpha=1.0 \
  --reset_to_initial_position=true \
  --reset_duration_s=3.0 \
  --reset_fps=50 \
  --rtc_mode=trained \
  --trained.queue_threshold=25 \
  --trained.replan_on_result=true \
  --resize_images_on_client=true \
  --continuous_observation=true \
  --observation_max_age_s=0.1 \
  --startup_calibration_runs=3 \
  --startup_calibration_timeout_s=1800 \
  --latency_history_ttl_s=300 \
  --observation_capture_fps=30 \
  --debug_rtc_diagnostics=false \
  --log_rtc_timing=false \
  --debug_visualize_queue_size=false \
  --suppress_can_debug_logs=true \
  --task="$TASK" \
  --dagger_dataset_repo_id="${DAGGER_DATASET_NAMESPACE}/dagger_trained_0-10" \
  --dagger_dataset_resume=false \
  --dagger_dataset_push_to_hub=false \
  --dagger_num_episodes=10 \
  --dagger_input_device=keyboard
```

</details>

**DAgger 操作：** 默认右方向键依次推进「准备 → 自主运行 → 暂停 → 人工纠正 → 保存并复位」，左方向键放弃未保存的纠正并复位，`Esc` 结束。`dagger_num_episodes=10` 统计保存的纠正 episode。默认主臂和从臂 CAN 接口分别为 `can_master`、`can_follower`，应与机器人电脑上的接口一致。

`dagger_dataset_resume=false` 用于新建采集；如果该数据集目录已存在，应换一个新名称，或在确实要续录同一数据集时改为 `true`。两条命令均显式关闭数据上传。

**trained-RTC 配套关系：**

| 参数 | 本手册取值 | 要求 |
| --- | --- | --- |
| 训练 `policy.rtc_training_max_delay` | `25` | 模型必须实际在此设置下训练并保存 |
| 服务端 `trained.delay_steps` | `25` | 大于 0，且不超过训练最大前缀长度 |
| 客户端 `actions_per_chunk` | `50` | 至少是服务端 delay 的 2 倍，且不超过模型 chunk size |
| 客户端 `trained.queue_threshold` | `25` | 不小于服务端 delay，且小于动作块长度 |
| 客户端 `trained.replan_on_result` | `true` | 接纳结果后允许尽快发起下一次请求 |

trained 的触发预算独立于 guided，因此 trained 命令省略了原文件中不参与该模式调度的 `chunk_size_threshold`、`adaptive_rtt_trigger` 和 `rtt_trigger_margin_steps`。两类客户端均保留 `fps=30`、`interpolation_multiplier=2`，即策略动作频率 30 Hz、插值命令目标频率 60 Hz。

原文件中已删除的 `observation_similarity_atol` 和 `active_splice_bridge_steps` 不再出现在命令中。完整参数说明可参阅仓库内的 [Deployment 文档](../src/lerobot/yzh/README_DEPLOYMENT.md)、[DAgger 文档](../src/lerobot/yzh/README_DAGGER.md)和[训练文档](../src/lerobot/yzh/README_TRAINING.md)。

<a id="local-tools"></a>

## 工具与常用检查

以下命令在仓库根目录运行。桌面工具需要可用的 Linux 图形会话。

### 数据集与设备

```bash
# 多相机审查：标记待删除 episode，另存过滤副本
uv run --locked --extra dataset --with customtkinter python src/lerobot/yzh/lerobot_review_dataset.py

# 合并本地演示与 DAgger 数据
uv run --locked --extra dataset python src/lerobot/yzh/merge_datasets_gui.py

# 相机别名与主从臂 CAN 绑定面板
bash src/lerobot/yzh/device_bindings/start.sh
```

审查工具选择包含 `meta/info.json` 和 `data/` 的数据集根目录，并需要支持 Xft 的 Tk。合并工具优先使用 Zenity，设 `LEROBOT_GUI_BACKEND=tk` 可切换界面；原示范缺少的 `intervention` 会补为 `False`。RECAP 的 sidecar 尚未接入这些编辑/合并工具，不直接用它们处理 RECAP 数据。

设备绑定是独立系统工具，启动脚本使用系统 Python/GTK；系统依赖、相机别名和 CAN 激活见[设备绑定说明](../src/lerobot/yzh/device_bindings/README.md)。

### WN-A100 监控

```bash
# 桌面面板；Qt 是此工具额外需要的依赖
uv run --locked --with 'PyQt5>=5.15.10,<6' python -m lerobot.yzh.wn_a100

# 只读取一次 JSON，无需图形桌面或 Qt
uv run --locked python -m lerobot.yzh.wn_a100 --once --json
```

界面预览用 `--demo`，全部选项用 `--help`。监控、训练日志、磁盘和 SSH 登录配置见 [WN-A100 说明](../src/lerobot/yzh/wn_a100/README.md)；旧编号脚本仅作兼容入口。

<a id="troubleshooting"></a>

### 登录、日志与排错

```bash
uv run --locked hf auth login
uv run --locked --extra pi05-training wandb login

# 按实际训练日志位置替换
tail -f /path/to/train.log
watch -n 1 nvidia-smi
ps -u "$USER" -f

# 检查数据所在文件系统、临时目录及缓存
df -h "$HOME" /tmp
du -sh ~/.cache/huggingface
```

W&B 需同时在训练命令设置 `--wandb.enable=true`；`--wandb.disable_artifact=true` 关闭模型 Artifact 上传，`WANDB_MODE=offline` 用于离线记录。`tail -f` 中按 `Ctrl+C` 只退出日志查看。停止自己的训练进程用 `kill <PID>`。

| 现象 | 处理 |
| --- | --- |
| SSH 动态库冲突 | 仅对 SSH 清理变量：`env -u LD_LIBRARY_PATH /usr/bin/ssh user@server` |
| HF SSL / 镜像错误 | 检查 `HF_ENDPOINT`；错误镜像可用 `unset HF_ENDPOINT` 恢复官方源 |
| 磁盘不足 | 对实际数据目录执行 `df -h /path/to/data`、`du -sh /path/to/data/*`，同时检查 `/tmp` |
| 桌面工具无法开窗 | 在图形桌面终端运行并检查 `DISPLAY`；GPU 监控可改用 `--once` |
| 锁文件或缺依赖 | 确认代码、`pyproject.toml`、`uv.lock`、`.python-version` 配套，再按用途同步 extra |

认证信息通过交互登录或本机配置保存；不写入共享命令或文档。

<a id="references"></a>

## 专题参考

完整运行配方只在本页维护。以下文档保留各主题独有的原理、数据约定和排障细节：

| 主题 | 文档 |
| --- | --- |
| 训练统计、纠正采样、loss 与步数 | [Training](../src/lerobot/yzh/README_TRAINING.md) |
| RTC 调度、图像上传、编译与故障定位 | [Deployment](../src/lerobot/yzh/README_DEPLOYMENT.md) |
| 自主运行、人工接管与纠正数据 | [DAgger](../src/lerobot/yzh/README_DAGGER.md) |
| 整轮采集、成功失败标注与模型追溯 | [RECAP](../src/lerobot/yzh/README_RECAP.md) |
| 入口模块、JSON 配置和双端迁移 | [异步推理目录](../src/lerobot/async_inference/README.md) |

DAgger 记录接管纠正窗口；RECAP 记录完整任务，必须明确标注成功或失败才保存，两套数据流程分别运行。RECAP 当前采集说明不代表完整强化学习训练已经实现。

通用教程：[安装](source/installation.mdx)、[采集与训练](source/il_robots.mdx)、[π0.5](source/pi05.mdx)、[多 GPU](source/multi_gpu_training.mdx)、[动作表示](source/action_representations.mdx)、[RTC](source/rtc.mdx)、[硬件选型](source/hardware_guide.mdx)。完整分类见 [`source/_toctree.yml`](source/_toctree.yml)。

<a id="documentation"></a>
<a id="doc-maintenance"></a>
<a id="generating-the-documentation"></a>

<details>
<summary>文档维护：目录约定、构建与预览</summary>

## 文档放在哪里

- `docs/README.md`：文档导航、实验操作、本地工具和文档维护方法。
- `docs/source/*.mdx`：通用教程与 API 文档，由 `_toctree.yml` 组织到文档网站。
- `docs/source/policy_*_README.md`：策略简介、论文和引用；多个策略源码目录的 `README.md` 是指向这些文件的符号链接，保留原路径。使用步骤放在对应 `.mdx` 教程中。

同一操作只在一处维护完整步骤，其他入口通过链接引用；新增内容优先补充已有主题。

<a id="building-the-documentation"></a>
<a id="previewing-the-documentation"></a>

## 本地构建与预览

在仓库根目录执行，先安装 Node.js，再准备 Python 环境：

```bash
uv sync --locked --extra dataset
uv pip install -r docs-requirements.txt

# 生成文档到临时目录
uv run --no-sync doc-builder build lerobot docs/source/ --build_dir /tmp/lerobot-docs

# 在浏览器中预览
uv run --no-sync doc-builder preview lerobot docs/source/
```

预览地址：[http://localhost:3000](http://localhost:3000)。`docs-requirements.txt` 已包含 `watchdog`，无需单独安装。构建产物不提交到 Git。

<a id="adding-a-new-element-to-the-navigation-bar"></a>
<a id="adding-a-new-tutorial"></a>

## 新增或调整教程

1. 在 `source/` 中编辑已有 `.mdx`，或为新主题创建文件。
2. 新页面加入 [`source/_toctree.yml`](source/_toctree.yml)，`local` 填不带扩展名的相对路径。
3. 新增页面后重启 `doc-builder preview`，检查导航、代码块和链接。

<a id="renaming-section-headers-and-moving-sections"></a>

移动内容或修改标题时，在原位置保留旧锚点并链接到新位置，例如：

```html
<a id="old-section"></a>
<a href="./new-page#new-section">本节已迁移至新页面</a>
```

<a id="writing-source-documentation"></a>
<a id="writing-a-multi-line-code-block"></a>
<a id="adding-an-image"></a>

参数名、路径和代码值使用反引号；命令使用标注语言的代码块。图片和视频优先托管于 [documentation-images](https://huggingface.co/datasets/huggingface/documentation-images)，通过 URL 引用。

</details>
