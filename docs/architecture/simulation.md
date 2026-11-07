# Piper MuJoCo 仿真实现与数据语义

[架构首页](README.md) · [仿真操作](../workflows/simulation/README.md) · [仿真排错与边界](../troubleshooting/simulation.md)

仿真插件位于 [examples/piper_sim](../../examples/piper_sim/README.md)，注册类型为
`piper_mujoco`，同时提供 `RobotConfig` 与 `EnvConfig`。日常入口为 `piper-sim`，
标准 LeRobot 命令也可显式选择该类型；完整参数配方在[操作流程](../workflows/simulation/README.md)维护。

| 源码 | 对应职责 |
| --- | --- |
| [config.py](../../examples/piper_sim/lerobot_robot_piper_mujoco/config.py) | 类型注册、配置与默认资源位置 |
| [controllers.py](../../examples/piper_sim/lerobot_robot_piper_mujoco/controllers.py) | 主臂、程序专家、键盘与策略动作来源 |
| [simulation.py](../../examples/piper_sim/lerobot_robot_piper_mujoco/simulation.py) | 关节与夹爪映射、物理推进、状态恢复和成功判定 |
| [scene.py](../../examples/piper_sim/lerobot_robot_piper_mujoco/scene.py) | 从模型资源构造任务场景与碰撞体修正 |
| [cli.py](../../examples/piper_sim/lerobot_robot_piper_mujoco/cli.py) | 数采、回放、推理与评估会话 |
| [assets.py](../../examples/piper_sim/lerobot_robot_piper_mujoco/assets.py) | 固定版本的模型资源准备与 SHA-256 记录 |

## 动作与时序约定

默认六关节的 SDK rad 数值直接映射到 MJCF。`--joint-signs` 与 `--joint-offsets` 的
操作方法见[主臂遥操作](../workflows/simulation/README.md#真机主臂遥操作仿真从臂)。
夹爪始终使用 SDK 的总开口宽度（米），内部换算为 MuJoCo 单指位移，最大总开口 0.07 m。

主臂模式使用真实时间节拍，仿真以固定子步推进；相机读取不会推进物理时间。
关节每帧变化上限与录制前对齐的操作说明在[遥操作流程](../workflows/simulation/README.md#真机主臂遥操作仿真从臂)。

## 场景、程序专家与成功判定

场景中的红色方块和绿色目标区域支持物体位置随机化。程序专家使用仿真物体位姿求解
任务轨迹，通过普通位置执行器和碰撞接触完成抓取，没有将物体焊接到夹爪或直接移动到目标。
模型的输入仍然只有相机和七维关节/夹爪状态，不包含专家使用的物体位姿。

成功需要方块先被提升至少 3.5 cm，再放入目标区、松爪并稳定至少 0.3 秒。
评估结果的适用范围见[真机主臂与验证边界](../troubleshooting/simulation.md#真机主臂与验证边界)。

## 数据与回放语义

输出使用标准 `LeRobotDataset`。每帧记录动作前的观测，以及实际下发给仿真的限位后动作。
程序专家和人工示范使用相同字段，但明确标记数据来源。

额外的 `simulation.json` 保存相机/控制配置、完整回合初始状态、随机种子、动作来源及
成功标记。`--joint-signs`、`--joint-offsets` 的设置也保存在仿真说明文件中。
保存视频/图像、选择相机、继续追加和筛选成功回合的参数在[数采流程](../workflows/simulation/README.md#数据采集)维护。

回放恢复录制时的完整仿真初始状态，而非只回零机械臂，并对比每帧关节状态。
默认误差容差与缺少 `simulation.json` 时的处理见[回放排错](../troubleshooting/simulation.md#回放误差或缺少仿真说明文件)。

## 策略与测试覆盖

训练器、归一化、模型实现、checkpoint 格式全部使用现有 LeRobot 代码。
推理读取 checkpoint 自带的前后处理器并执行闭环动作。

### 测试覆盖

[仿真测试](../../examples/piper_sim/tests)覆盖插件导入无硬件副作用、动作映射/限位、
控制时间、虚拟相机、随机复位、主臂适配器生命周期、物理抓取、标准数据集读写与确定性回放。
测试中的真实主臂由替身代替，不连接 CAN；实际主臂联动和 GUI 显示需在设备旁验收。

测试通过仿真项目的 uv 环境直接运行，命令见[仿真测试流程](../workflows/simulation/README.md#测试)。

## 模型来源与许可证

资源来自 [MuJoCo Menagerie AgileX Piper](https://github.com/google-deepmind/mujoco_menagerie/tree/main/agilex_piper)，
默认下载固定 commit 为 `71f066ad0be9cd271f7ed58c030243ef157af9f4`，遵循资源自带 MIT 许可证。
资源准备会在 `provenance.json` 记录文件 SHA-256；使用 `--source` 复制本地资源时，版本标记为
`local-copy`，不能据此认定与默认下载版本相同。来源目录若含 `SOURCE.json`，还会复制为
`original-source.json` 供追溯。下载或复制命令见[安装与首次检查](../workflows/simulation/README.md#安装与首次检查)。

该版本手指 box 碰撞体的局部 Y 偏移与可见手指朝向相反，`scene.py` 在构造任务场景时
镜像该偏移，使碰撞体覆盖实际指尖；缓存中的 vendor XML 和网格保持原样。
