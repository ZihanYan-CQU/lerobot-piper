# Piper 指令核查记录：2026-11-07

[返回文档维护](README.md) · [流程指令](../workflows/README.md) · [环境准备](../workflows/setup/README.md)

## 结论与范围

本次在实际仓库 `/home/cpy/桌面/yzh/piper/lerobot` 核查本地 Piper 指南，使用当前已安装的项目环境。未发现已核查配方的入口或参数与当前代码不一致，但**当前机器尚不具备直接执行所有流程的条件**。参数解析通过、程序能显示帮助，都不代表真机任务或模型训练已运行成功。

范围包括 `docs/workflows/` 的环境、数采、数据处理、训练、RLT、仿真、部署、DAgger、RECAP、评估及辅助工具，并补查文档网站构建入口。上游 `docs/source/` 全部教程和研究页中的历史实验命令不在逐项实跑范围内。完整操作配方仍在各流程页维护，本页只保存本次证据、缺项和验证边界。

## 各流程结果

| 流程 | 本次验证 | 结果与边界 |
| --- | --- | --- |
| [环境](../workflows/setup/README.md) | 主项目与仿真环境的解释器；根环境依赖一致性 | 两者均使用 uv 管理的 Python 3.12.14；根环境 116 个已安装包通过 `uv pip check`。这不表示所有可选 extra 已装齐 |
| [数采](../workflows/recording/README.md)、[部署](../workflows/deployment/README.md)、[DAgger](../workflows/dagger/README.md)、[RECAP](../workflows/recap/README.md)、[指示灯](../workflows/tools/status-indicator.md) | 17 次帮助检查；47 组真实配置解析，包括 22 条完整配方及 JSON 覆盖、灯参数、trained RTC、续采、键盘/踏板变体 | 全部通过入口/参数检查；未构造机器人或策略、未连接控制服务、未验证灯效 |
| [数据处理](../workflows/datasets/README.md) | 实际运行 info、删除 episode 到新目录、合并两份本地数据 | 全部通过。使用临时无视频合成数据，未处理用户数据；真实视频数据的大规模合并、图形回放未实测 |
| [训练](../workflows/training/README.md)、[RLT](../workflows/rlt/README.md) | 8 次帮助检查；14 项实际配置/参数检查，覆盖单卡/双卡、guided/trained、恢复训练、导出、LoRA 合并及 RLT | 正向检查全部通过；3 种 RLT 非法组合按预期返回错误。未加载权重或执行训练；当前根环境缺少部分训练依赖 |
| [仿真](../workflows/simulation/README.md) | 18 项运行检查、16 条完整命令解析；doctor 两种分辨率；无窗口 expert 执行 3 步 | 全部通过，MuJoCo 3.8.1，30 Hz，两路虚拟相机。3 步仅为启动验证，任务未完成；未据此评估成功率，也未运行完整采集/训练链路 |
| [WN-A100](../workflows/tools/wn-a100.md) | 帮助与 10 组参数解析、训练/磁盘 demo；真实 `--once` HTTP 查询 | 均通过，指南 19 个选项存在于当前 CLI，真实接口返回 8 卡状态。SSH 日志/磁盘、桌面显示和自启动安装未验证 |
| [桌面数据工具](../architecture/dataset-tools.md)、[GitHub 工具](../workflows/tools/github-upload.md) | 审查模块导入；合并/上传工具帮助；相关现有测试 | 数据审查 27 项、合并窗口 10 项、上传窗口 9 项测试通过。Qt 使用 offscreen；未验证真实桌面交互、GitHub 登录或真实上传 |
| [设备绑定](../workflows/tools/device-bindings.md) | 系统 GTK 导入；相机、CAN、桌面启动器帮助与参数；只读设备状态 | 入口检查通过；未执行绑定、激活、规则保存或桌面安装，设备现状见下表 |
| [评估](../workflows/evaluation/README.md) | 通用 eval 帮助及其引用的部署配方解析 | 入口通过；未进行真机任务评估或加载策略测量成功率 |
| [文档网站](README.md#构建和预览上游文档网站) | 检查构建前置依赖及当前入口 | 本机缺 doc-builder、watchdog，PATH 中无 Node/npm，构建与预览未运行，相关参数尚未通过已安装程序实证 |

数据处理实跑使用两份各含 2 个 episode、6 帧的临时数据：过滤后得到 1 个 episode、3 帧；合并后得到 4 个 episode、12 帧。无 `intervention` 的演示帧补为 `false`，纠正数据的原有标记保留；合并前后输入文件哈希一致。该检查覆盖指南中的显式输入目录和 `fill_missing_intervention=true` 用法。

## 当前机器的缺项与设备状态

以下是核查时的本机状态，不代表训练服务器或以后插拔设备后的状态。

| 项目 | 实际结果 | 对使用者的影响与入口 |
| --- | --- | --- |
| 普通 Pi0.5 训练依赖 | 根 `.venv` 缺 `accelerate`、`wandb`、`transformers` | 训练前按[训练环境](../workflows/training/README.md)补齐 `pi05-training`；当前不能直接跑完整训练，`wandb login` 也缺入口。Pi0.5 本地推理同样需要模型依赖 |
| PEFT | 缺 `peft` | 使用相关 LoRA/合并流程前按[RLT 指南](../workflows/rlt/README.md)准备对应 extra；不影响所有非 PEFT 流程 |
| USB 指示灯 | `import hid` 报 `ModuleNotFoundError` | 按[灯依赖说明](../workflows/tools/status-indicator.md)补齐 `gamepad`。默认可选灯允许降级无灯；要求必有灯或独立查询时仍需依赖及设备 |
| GPU | PyTorch 2.11.0+cu128 的 `cuda.is_available()` 为 true，1 张 RTX 4060，驱动报告显存 8188 MiB | 已确认 CUDA 可见；未确认该显存能承载指定 Pi0.5 配方。服务器示例的 GPU 4、4/5 不能原样套用本机 |
| 相机别名 | `/dev/video_top`、`/dev/video_wrist` 存在；`/dev/video_down` 不存在 | 三相机[现场采集配方](../workflows/recording/single-arm.md)当前不能原样使用。只用两路时查[双相机配方](../workflows/recording/two-camera.md)，并核对实际画面；本次未打开相机 |
| CAN | `can_follower` 为 UP，bitrate 1000000；没有 `can_master` | 需要主臂的遥操作/数采还缺设备条件，先按[设备绑定](../workflows/tools/device-bindings.md)处理。只读接口状态不证明机械臂反馈、固件或运动正常 |
| 模型、数据和路径 | 服务器 `/home/yzh/...` 及 `/absolute/path/...` 等示例不是本机可用资源 | 在实际运行端替换 checkpoint、数据、GPU、服务器地址及会话标识；解析检查未证明模型文件或 Hub revision 可用 |
| 网站构建依赖 | 尚未准备 | 仅影响文档网站构建/预览；安装步骤见[文档维护](README.md#构建和预览上游文档网站)，不影响直接阅读 Markdown |

以上缺项在现有流程页已有准备说明，本次没有为本机缺包改写正确的启动命令，也没有同步或安装依赖。RLT Stage2 的独立学习入口不加载完整 VLA，不能把普通 Pi0.5 训练缺包的结论直接套用到所有 RLT 流程。

## 验证方法与尚未执行的操作

- Python 检查使用现有 uv 环境及 `--no-sync`；仿真显式指定 `--project examples/piper_sim`。审计缓存、合成数据和日志放在 `/tmp`，关闭字节码写入。
- 配置检查调用实际 draccus/argparse 和适用的配置验证逻辑，在业务执行前截断；仅为解析 checkpoint 配置使用临时 mock，不加载权重。硬件组配置解析加入 socket 及 Python 设备文件打开拦截，且不构造机器人或 runtime；帮助检查先确认入口会在业务执行前退出。
- GPU、相机路径、CAN 状态和监控 HTTP 查询在正常权限下复核；受限沙箱最初产生的 CUDA、设备可见性、DNS 和字体缓存错误不作为本机故障结论。
- 少数首次检查超时，串行复核均通过；数据合并重试耗时约 4.2 秒。保留初次结果，未把超时归因于指令错误。
- 未实际执行机械臂运动、复位、录制，未验证相机画面、实体灯效、远端推理连接、完整训练、checkpoint 恢复、Hub 下载/上传或真实 GitHub 推送。未安装自启动、修改设备规则或改动旧 Conda 环境。

本次相关运行日志和检查脚本暂存在 `/tmp/lerobot-command-audit-20261107/`：`main/` 记录工具与数据实跑，`hardware/` 记录硬件流程参数，`training/` 记录训练/RLT，`simulation/` 记录仿真和构建前置条件。该临时目录可能被系统清理，本页保留可长期查阅的结论。

后续复核应针对发生变化的代码、环境和设备重新执行相应层级的检查，不能把这份日期记录视为未来所有环境的运行保证。
