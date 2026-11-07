# DAgger 人工纠正采集指令

[返回文档首页](../../README.md) · [环境准备](../setup/README.md) · [DAgger 实现](../../architecture/dagger.md) · [DAgger 排错](../../troubleshooting/dagger.md)

首次准备依赖：服务端执行 `uv sync --locked --inexact --extra pi05-server`；机器人电脑执行 `uv sync --locked --inexact --extra piper-client`；同机运行时同时选择这两个 extra。下方 `uv run --no-sync` 只运行已准备好的 `.venv`。

DAgger 只保存人工纠正窗口，一次纠正对应一个 episode。需要完整任务的自主段、接管段及成功/失败标签时，使用 [RECAP](../recap/README.md)。

| 运行方式 | 完整命令 |
| --- | --- |
| GPU 服务器 + 机器人电脑，guided RTC | [远端 guided 双端命令](guided.md) |
| GPU 服务器 + 机器人电脑，trained RTC | [远端 trained 双端命令](trained.md) |
| GPU 与机器人在同一机器 | [本地 DAgger](local.md) |

远端页包含各自的路径、相机和服务端/客户端变量，可从新终端逐段执行。所有命令从仓库根目录运行。启动顺序是服务端 → [SSH 隧道](../deployment/connection.md) → 客户端。主从 CAN 默认为 `can_master` / `can_follower`，必须先完成[设备绑定](../setup/README.md)。

## 操作流程

默认右方向键每次推进一步：**准备 → 自主运行 → 暂停 → 人工纠正 → 保存并复位**。复位完成后恢复物块/夹具，再按右键开始下一轮。

| 输入 | 行为 |
| --- | --- |
| 右方向键 | 开始、暂停、进入纠正、保存当前纠正，按当前阶段执行一步 |
| 左方向键 | 放弃未保存窗口并复位；自主运行时先暂停 |
| Esc | 结束，默认丢弃尚未完成纠正窗口 |

保存/复位期间重复输入会被忽略，完成后需重新按键；不会自动开始下一轮。接管对齐未通过时保持暂停，重新对齐后重试；复位失败时先恢复并成功复位。完整状态边界见[DAgger 架构](../../architecture/dagger.md)。

远端也可用 `--dagger_input_device=prompt`，支持 `/start [task]`、`/pause`、`/correct`（开始/结束纠正）、`/reset`、`/task`、`/status`、`/stop`。远端 prompt 保存是同步的，不能在同一输入路径保存时再读 `/stop`；键盘 Esc 可在保存期间带外触发退出。本地输入方式见[本地页](local.md)。

## 指示灯

DAgger 的绿闪表示自主运行，绿常亮表示人工纠正；同一个灯态在普通数采中可能含义不同。远端、本地 DAgger 的准备与复位提示也有差异，统一见[指示灯作用与配置](../tools/status-indicator.md)。

## 数据存储与续录

远端使用 `dagger_dataset_*` 参数，本地使用 `dataset.*`。不要混用两套参数名。

- 远端 `dagger_dataset_repo_id` 未设置时只运行接管、不记录；本地对应 `dataset.repo_id`。
- 设置 `dagger_dataset_root`（本地 `dataset.root`）可以明确选择机器人电脑上的保存目录。
- 新建时 `dagger_dataset_resume=false`；续录需原 repo_id、显式 root 和 `dagger_dataset_resume=true`。已有目录不能作为新的采集目录重复创建。
- `dagger_num_episodes`（本地 `num_episodes`）统计本次运行成功保存的纠正数，不是数据集总数；0 表示不限。
- `dagger_dataset_fps=0` 继承 policy fps，不乘插值倍数；远端配方显式关闭 Hub 上传。
- 观测来自从臂，`action` 是主臂的完整绝对控制目标，`intervention=True`；不把自主策略输出当作专家动作。

## 采集后训练

1. [审查并合并数据](../datasets/README.md)：纠正数据保留 `intervention=True`，原始演示缺标记时补 False。
2. 选择[训练配方](../training/README.md)，设置干预采样比例，并按需保留起始 checkpoint 的 processor 统计量。
3. 使用新 checkpoint 开启下一次部署连接。服务端不会在当前会话内自动监视磁盘更新。

远端完整配方保留 `reuse_last_policy=true`；如果训练会覆盖相同模型路径，改为 false 后重启服务。代码默认 false 可避免下一次连接复用旧权重。采集或保存报错时见[集中排错](../../troubleshooting/dagger.md)，先检查数据完整性再续录。
