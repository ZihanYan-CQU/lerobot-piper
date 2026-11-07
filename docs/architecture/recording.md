# 人工示教数采结构

[返回架构导航](README.md) · [数采指令](../workflows/recording/README.md) · [现场操作](../workflows/recording/single-arm.md) · [数采排错](../troubleshooting/recording.md)

`lerobot-record` 负责人工主臂带动从臂的演示采集。状态来自从臂反馈，示教目标来自主臂；相机观察、状态、动作和任务文字一起写入 LeRobot 数据集。模型部署与 DAgger 使用各自入口，不在这个录制配置中加载策略。

## 维护位置

| 内容 | 源码 |
| --- | --- |
| 录制、复位、重录、保存顺序与退出清理 | [lerobot_record.py](../../src/lerobot/scripts/lerobot_record.py) |
| 全局键盘 / TTY 输入与按键到事件的映射 | [keyboard_input.py](../../src/lerobot/utils/keyboard_input.py) |
| 默认条数、时长、命名、视频参数 | [configs/dataset.py](../../src/lerobot/configs/dataset.py) |
| 三相机默认配置与从臂控制选项 | [config_piper_follower.py](../../src/lerobot/robots/piper_follower/config_piper_follower.py) |
| 主臂连接、动作读取、回零 | [piper_leader_PyAgxArm.py](../../src/lerobot/teleoperators/piper_leader/piper_leader_PyAgxArm.py) |
| 从臂观察、动作写入、相机连接与退出 | [piper_follower_PyAgxArm.py](../../src/lerobot/robots/piper_follower/piper_follower_PyAgxArm.py) |
| CAN、SDK、温度监测、回零和失能 | [piper_PyAgxArm.py](../../src/lerobot/motors/piper/piper_PyAgxArm.py) |
| RGB 别名与 CAN 系统配置 | [设备绑定结构](device-bindings.md) |
| 录制 / 复位阶段到 USB 灯态 | [指示灯结构与调用范围](status-indicator.md) |
| 采后多相机审查、过滤副本与数据合并 | [数据工具结构](dataset-tools.md) |

## 数据与控制边界

当前 Piper 主从臂实现使用固定 CAN 接口 `can_master` / `can_follower`，配置没有串口式的 `port` 字段。默认相机是 `top`、`wrist`、`down`；完整的命令行 `robot.cameras` 字典可以替换这组配置。摄像头的 `width` / `height` 表示旋转后的输出尺寸，90° / 270° 时 OpenCV 会交换实际采集宽高。

录制阶段积累当前 episode 缓冲。正常一轮结束后执行回零和不写盘的复位遥操作，再调用 `save_episode()`；最后一轮或停止时跳过复位。重录标记使程序清空当前缓冲而不增加成功保存计数，之后仍使用原 episode 编号。此前已保存的数据不受重录影响。

`resume=true` 读取已有数据集并检查机器人、FPS 和特征兼容性；`num_episodes` 是本次额外保存的数量。新建默认给 repo_id 加日期时间戳，`no_stamp=true` 适用于调用者已生成唯一名称或自行维护批次名；续采不会再次追加时间戳。

当前默认采集流程按本机 Piper 实现说明；策略部署与复位的动作下发区别见 [部署架构](deployment.md)。文档整理不代表新的真机验证。
