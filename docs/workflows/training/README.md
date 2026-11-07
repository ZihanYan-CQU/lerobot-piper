# Pi0.5 训练指令

[返回文档首页](../../README.md) · [环境准备](../setup/README.md) · [训练原理与实现](../../architecture/training.md) · [训练排错](../../troubleshooting/training.md)

首次准备依赖：`uv sync --locked --inexact --extra pi05-training`。下方 `uv run --no-sync` 只运行已准备好的 `.venv`，不在启动时安装依赖。

| 要做什么 | 指令页 |
| --- | --- |
| 下载 tokenizer、数据集、Pi0.5 基座模型 | [下载资源](downloads.md) |
| 普通 Pi0.5 微调，推理时使用 guided RTC | [guided：单卡 / 多卡完整命令](guided.md) |
| 训练动作前缀条件，推理时使用 trained RTC | [trained：单卡 / 多卡完整命令](trained.md) |
| 从完整 checkpoint 恢复中断的同一轮训练 | [中断续训](resume.md) |
| 使用 RLinf 后端、RLT Stage 1 / Stage 2、导出模型 | [RLT 流程](../rlt/README.md) |
| 在 MuJoCo 中采数据并训练 ACT | [仿真训练](../simulation/README.md) |
| 先审查、合并演示与纠正数据 | [数据集处理](../datasets/README.md) |
| 使用训练结果上机 | [真机部署](../deployment/README.md) |

每个命令页都包含自己的路径配置。选择一条配方运行，不要把整页或所有方案一次执行。

guided RTC 对应普通 Pi0.5 微调；trained RTC 必须在训练时真正学习动作前缀条件，不能只修改已有 checkpoint 的配置。

## 现有四条配方的差别

| 设置 | 单卡 | 多卡 |
| --- | --- | --- |
| GPU | `4` | `4,5`，共 2 张 |
| 每卡 batch size | `32` | `16` |
| 梯度累积 | `1` | `1` |
| 有效总 batch size | `32` | `32` |
| 训练步数 | `10000` | `20000` |
| 干预数据目标采样比例 | `0.3` | `0.5` |
| 学习率 / 最终学习率 | `5e-6` / `5e-7` | 使用当前 Pi0.5 默认值 |

单卡、多卡使用不同的训练设置，因此两者不仅是 GPU 数量不同。在当前 `policy.type=pi05` 配置下，多卡默认学习率为 `2.5e-5`、最终学习率为 `2.5e-6`。如需做相同配置的单卡与多卡对照，应同时对齐步数、采样比例、学习率和调度参数。

两种 RTC 模式的训练区别是 `policy.rtc_training_max_delay`：guided 使用 `0`，trained 使用 `25`。四条配方均显式设置 `chunk_size=50`。

训练集的 `repo_id` 与本地 `root` 应指向同一份数据。现有命令保留 `dagger_merged` 的 repo_id 与 `shape-matching_89_with_dagger_30` 的本地目录示例；如果使用下载页的同名 Hub 数据集，应同时核对并替换 repo_id。

`intervention_sampling_ratio` 需要有效的 True / False 两组样本；只有纠正片段的全 True 数据不能直接启用混合采样。`use_pretrained_stats=true` 要求起始 checkpoint 带完整 processor 配置和统计量。原理见[训练架构](../../architecture/training.md)。
