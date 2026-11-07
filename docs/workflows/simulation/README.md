# Piper MuJoCo 仿真指令

[返回文档首页](../../README.md) · [环境准备](../setup/README.md) · [仿真实现](../../architecture/simulation.md) · [仿真排错与边界](../../troubleshooting/simulation.md)

| 要做的事 | 入口 |
| --- | --- |
| 安装环境、准备模型、检查虚拟相机 | [安装与首次检查](#安装与首次检查) |
| 主臂遥操作、数采、回放 | [遥操作](#真机主臂遥操作仿真从臂) · [数采](#数据采集) · [回放](#回放) |
| ACT 训练、闭环评估 | [训练](#act-训练) · [推理和评估](#模型推理和评估) |
| 逐条验证数采到推理 | [五步训练短流程](smoke.md) |
| 测试、环境检查与问题定位 | [测试](#测试) · [仿真排错](../../troubleshooting/simulation.md) |

仿真插件的类型为 `piper_mujoco`，标准命令是 `piper-sim`。从 **LeRobot 仓库根目录**
执行下列命令，`uv --project examples/piper_sim` 选择仿真自己的 `.venv`；真机主项目
继续使用仓库根 `.venv`。只有显式选择 `--controller leader` 才会连接真实主臂。

## 安装与首次检查

`examples/piper_sim` 有独立的 `pyproject.toml`、`.python-version` 和 `uv.lock`，固定
使用 uv 管理的 Python **3.12.14**。先按[环境准备](../setup/README.md)
把独立 `uv` 加入 PATH，退出已激活的 Conda 环境。以下清理与渲染设置在当前终端执行一次；
每个新终端都需要重新执行，设置不会自动传递给其他终端：

```bash
unset UV_PROJECT_ENVIRONMENT UV_PROJECT UV_WORKING_DIR UV_PYTHON UV_CONFIG_FILE UV_NO_CONFIG
unset UV_PYTHON_PREFERENCE UV_NO_MANAGED_PYTHON UV_ENV_FILE UV_ISOLATED UV_NO_PROJECT
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
unset VIRTUAL_ENV
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

uv sync --project examples/piper_sim --locked --extra test

# 已有 outputs/piper_sim/assets/piper.xml 时可直接执行 doctor。
# 首次准备可复用本机 Menagerie 资源；复制到 outputs，不修改来源。
uv run --project examples/piper_sim --no-sync piper-sim prepare \
  --source "$HOME/桌面/openrua-project/sim/piper/vendor/agilex_piper"

uv run --project examples/piper_sim --no-sync piper-sim doctor
```

安装位置固定为 `examples/piper_sim/.venv`。它与根环境共用 uv 默认下载缓存，已安装包
分别管理。`--no-sync` 表示日常运行不重新安装包；首次使用及依赖变更后，先执行上面的
`uv sync`。本项目将当前仓库 LeRobot 与仿真插件都以 editable 方式安装。

| 依赖 | 用途 |
| --- | --- |
| `lerobot[training]` | ACT 训练、数据集读写、视频解码；ACT 无独立 extra |
| `lerobot[core_scripts,evaluation]` | 标准 `lerobot-record/replay/teleoperate/rollout/eval` 命令 |
| `lerobot[piper]` | `--controller leader` 的真实主臂 CAN/SDK 依赖 |
| `mujoco`、`numpy`、`gymnasium` | 仿真物理、数组和环境接口 |
| 本项目 `test` extra | 下文测试使用的 pytest |

只在主动修改依赖或根项目元数据后更新仿真锁文件：

```bash
uv lock --project examples/piper_sim --python 3.12.14 --managed-python
uv sync --project examples/piper_sim --locked --extra test
```

没有本地资源时，`uv run --project examples/piper_sim --no-sync piper-sim prepare` 会从
固定的 Menagerie commit 下载 XML、网格和 MIT 许可证，并记录 SHA-256。
默认模型目录由插件源码位置推导到当前仓库的 `outputs/piper_sim/assets`，也支持
`PIPER_SIM_ASSETS` 或 `--model-path` 显式覆盖。

数采与回放的 `--root` 必须明确指定。直接 CLI 使用 Hugging Face/PyTorch 的标准缓存
位置；如果希望复用此前脚本的仿真专用缓存，可在当前终端额外设置：

```bash
export HF_LEROBOT_HOME="$PWD/outputs/piper_sim/datasets"
export TORCH_HOME="$PWD/outputs/piper_sim/torch"
```

这些缓存变量不改变 Python 环境，也不改变下文显式传入的 `--root`。`MUJOCO_GL=egl`
用于虚拟相机的离屏渲染；使用 `--viewer` 时还需要可用的桌面显示。

## 真机主臂遥操作仿真从臂

主臂继续使用现有 `can_master` 配置。先按原有方式完成 CAN 设置，并清空主臂运动空间。
只有显式选择 `--controller leader` 才导入并连接真机主臂。
连接与退出涉及主臂驱动的初始化、回零/夹爪及断开；操作前阅读[主臂生命周期与使用边界](../../troubleshooting/simulation.md#真机主臂与验证边界)。

```bash
uv run --project examples/piper_sim --no-sync piper-sim teleoperate \
  --controller leader --viewer --realtime --duration 120
```

接入主臂时，先逐关节低幅度确认方向、零位与模型限位。需要调整时，用 `--joint-signs`
指定六个 `1/-1`，用 `--joint-offsets` 指定六个 rad 偏置。两者也会被存入数据集的
仿真说明文件。关节与夹爪的单位、映射关系见[动作与时序约定](../../architecture/simulation.md#动作与时序约定)。

从臂目标每帧默认最多变化 0.10 rad。录制前会让虚拟从臂逐渐对齐主臂；无法在八秒内
对齐时退出并提示检查姿态/限位。此过程不向主臂发送对齐指令。

查看器中 Enter 提前结束当前回合；录制时 Backspace 丢弃并重录；关闭查看器停止。

## 数据采集

```bash
# 人操作真实主臂，虚拟相机和虚拟从臂提供观测。
uv run --project examples/piper_sim --no-sync piper-sim record \
  --controller leader --viewer --episodes 10 --duration 20 \
  --root outputs/piper_sim/datasets/human_v1 --repo-id local/piper_human

# 程序专家：无需 CAN，适合验证流程和生成抓取放置示范。
uv run --project examples/piper_sim --no-sync piper-sim record \
  --controller expert --episodes 20 --success-only \
  --width 160 --height 120 \
  --root outputs/piper_sim/datasets/expert_v1 --repo-id local/piper_sim_pick_place

# 在同一数据集继续追加；保持原相机、单位和 FPS 配置一致。
uv run --project examples/piper_sim --no-sync piper-sim record \
  --controller expert --episodes 10 --success-only --resume \
  --width 160 --height 120 \
  --root outputs/piper_sim/datasets/expert_v1 --repo-id local/piper_sim_pick_place
```

场景提供红色方块、绿色目标区域和物体位置随机化；`--no-randomize` 固定初始位置。
程序专家与策略输入的区别见[场景、程序专家与成功判定](../../architecture/simulation.md#场景程序专家与成功判定)。

也可使用 `--controller keyboard --viewer`：`1..6` 增大对应关节，`QWERTY` 减小对应
关节，空格切换夹爪。当前键盘控制为关节模式；真机主臂是主要的人工示教入口。

输出为标准 `LeRobotDataset`，默认 H.264 视频，支持 `--no-video` 保存图像。
默认相机为 `top,wrist`，
可用 `--cameras top,wrist,down` 增加第三路。数据不会自动上传 Hub。

失败回合默认仍可保存；`--success-only` 则只保留成功回合。
标准数据与 `simulation.json` 的字段、来源标签及回放依赖见[数据与回放语义](../../architecture/simulation.md#数据与回放语义)。

## 回放

```bash
uv run --project examples/piper_sim --no-sync piper-sim replay \
  --root outputs/piper_sim/datasets/expert_v1 --repo-id local/piper_sim_pick_place \
  --episode 0 --video outputs/piper_sim/replay_0.mp4

# 加 --viewer --realtime 可以在查看器里按原速度播放。
```

回放会对比每帧关节状态；[误差超限或缺少 `simulation.json`](../../troubleshooting/simulation.md#回放误差或缺少仿真说明文件)时按排错页检查。

## ACT 训练

首次运行可先执行[五步训练短流程](smoke.md)，逐条确认数据读写、checkpoint 和推理。
该流程只验证工程链路；任务学习应使用更多演示和训练步数。
本机 8 GB 显存建议先用双相机、160×120、较小 batch 实测，再增加分辨率和模型规模。

```bash
uv run --project examples/piper_sim --no-sync lerobot-train \
  --dataset.repo_id=local/piper_sim_pick_place \
  --dataset.root=outputs/piper_sim/datasets/expert_v1 \
  --dataset.video_backend=pyav \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=30 --policy.n_action_steps=30 \
  --batch_size=4 --steps=20000 --save_freq=5000 --num_workers=2 \
  --env_eval_freq=0 --wandb.enable=false \
  --output_dir=outputs/piper_sim/train/act_pick_place
```

默认 ACT 会使用预训练视觉骨干；若需要完全离线运行，增加
`--policy.pretrained_backbone_weights=null`（视觉骨干从零训练）。

## 模型推理和评估

```bash
uv run --project examples/piper_sim --no-sync piper-sim evaluate \
  --policy outputs/piper_sim/train/act_pick_place/checkpoints/last/pretrained_model \
  --device cuda --width 160 --height 120 \
  --episodes 20 --seed 10000 --duration 20 \
  --report outputs/piper_sim/evaluation/metrics.json \
  --video-dir outputs/piper_sim/evaluation/videos
```

相机名称和尺寸必须与训练一致。评估用独立种子，输出每回合视频、成功率、实际耗时
和仿真时间。[成功判定](../../architecture/simulation.md#场景程序专家与成功判定)与[验证范围](../../troubleshooting/simulation.md#真机主臂与验证边界)分别维护。

插件也注册 `RobotConfig` 与 `EnvConfig`，可从原有入口显式选择新类型：

```bash
uv run --project examples/piper_sim --no-sync lerobot-rollout \
  --robot.type=piper_mujoco --robot.width=160 --robot.height=120 \
  --policy.path=outputs/piper_sim/train/act_pick_place/checkpoints/last/pretrained_model \
  --strategy.type=base --duration=20 --fps=30

uv run --project examples/piper_sim --no-sync lerobot-eval \
  --policy.path=outputs/piper_sim/train/act_pick_place/checkpoints/last/pretrained_model \
  --env.type=piper_mujoco --env.observation_width=160 --env.observation_height=120 \
  --eval.n_episodes=10 --eval.batch_size=1 \
  --output_dir=outputs/piper_sim/evaluation/native_eval
```

自动复位和严格动作记录请优先使用本插件的 `record`/`replay`/`evaluate` 会话入口。
原 `lerobot-record` 的主臂回零和人工复位行为保持原样。

## 测试

```bash
uv run --project examples/piper_sim --no-sync pytest examples/piper_sim/tests -q
```

测试中的真实主臂由替身代替，不连接 CAN。[测试覆盖](../../architecture/simulation.md#测试覆盖)
与[现场验证边界](../../troubleshooting/simulation.md#真机主臂与验证边界)见对应专题。
可选的 [smoke.sh](../../../examples/piper_sim/smoke.sh) 批处理调用相同的直接 uv CLI，
使用标准 `examples/piper_sim/.venv`；其完整逐条命令在[短流程](smoke.md)维护。
