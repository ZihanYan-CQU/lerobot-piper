# LeRobot 用户帮助导航

日常操作从 [文档首页](docs/README.md) 开始。开发约定保留在 [AGENTS.md](AGENTS.md)；本文件帮助用户和 AI 助手选择文档入口。

## 按当前任务查找

| 当前任务 | 文档 |
| --- | --- |
| 准备 Piper 环境 | [安装与迁移](docs/workflows/setup/README.md) |
| 遥操作、录制 Piper 数据 | [数采](docs/workflows/recording/README.md) |
| 看懂指示灯颜色、闪烁与对应阶段 | [指示灯](docs/workflows/tools/status-indicator.md) |
| Piper 仿真数采、训练与评估 | [仿真](docs/workflows/simulation/README.md) |
| 检查、删除坏 episode、合并纠正数据 | [数据处理](docs/workflows/datasets/README.md) |
| 训练 Pi0.5，选择单卡 / 多卡和 RTC 模式 | [训练](docs/workflows/training/README.md) |
| RLinf 后端、RLT 训练和导出 | [RLT](docs/workflows/rlt/README.md) |
| 在真机执行模型 | [远程与单机部署](docs/workflows/deployment/README.md) |
| 自主执行时接管并录制纠正 | [DAgger](docs/workflows/dagger/README.md) |
| 保存完整任务与成功失败结果 | [RECAP](docs/workflows/recap/README.md) |
| Piper 真机评估 | [评估指令](docs/workflows/evaluation/README.md) |
| 选择策略、数据规模与训练步数 | [参数与经验参考](docs/reference/README.md) |
| 了解代码和数据流 | [代码架构](docs/architecture/README.md) |
| 遇到异常或查过去的问题 | [问题排查与记录](docs/troubleshooting/README.md) |

## 协助用户时先确认什么

根据已经提供的上下文判断目标、机器人 / 遥操作器 / 相机、运行机器和 GPU、已有数据集，以及当前要完成的步骤。仅对完成任务必需且尚不明确的信息提问；用户已说明的设备和目标不用重复确认。

流程页中的设备、目录和模型路径属于示例。给出命令前结合当前代码、配置与用户机器核对，本项目使用 Piper，按其 CAN 接口、相机和主从臂约定核对。

## 原指南内容的新位置

本项目操作以 Piper 为主；其他硬件的上游教程保留在 `docs/source/`。数据质量、性能快照和训练预算在参考目录。checkpoint 统计量和干预采样语义统一维护在 [训练架构](docs/architecture/training.md)，审核与合并命令统一维护在 [数据处理](docs/workflows/datasets/README.md)。

<a id="L406"></a>

原第 406 行引用的 [PiperX 35,999 步模型卡](https://huggingface.co/Elvinky/pi05-piperx-full558-intervention-value-1to1-step35999) 是历史参考，其中第二个采样池混合人工干预与 value 筛选数据，不能直接当成纯 50% 人工干预训练配方。背景与证据见 [Evo-RL 两轮 SFT 对照](docs/research/evo_rl_two_stage_sft_2026-10-16.md)。

通用教程：[安装](docs/source/installation.mdx)、[模仿学习](docs/source/il_robots.mdx)、[硬件选型](docs/source/hardware_guide.mdx)；更多页面见 [网站目录](docs/source/_toctree.yml)。
