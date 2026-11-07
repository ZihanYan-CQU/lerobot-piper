# 数据集审查与合并工具结构

[架构导航](README.md) · [数据处理指令](../workflows/datasets/README.md) · [数据工具排错](../troubleshooting/datasets.md)

采后数据工具读取本地 LeRobot 数据集，审查工具生成过滤副本，合并工具生成新数据集。两者复用 `datasets` 的编辑实现；图形窗口负责交互和任务状态，不参与机器人控制。

## 模块边界

| 入口 / 模块 | 职责 |
| --- | --- |
| [merge_datasets_gui.py](../../src/lerobot/yzh/merge_datasets_gui.py) | 合并启动入口、后端选择、目录发现、自动命名、输出路径校验与命令构造；保留 Tk / Zenity 兼容界面 |
| [merge_datasets_window.py](../../src/lerobot/yzh/merge_datasets_window.py) | PyQt5 的 `MergeWindow`、轻量元数据 `DatasetEntry` 和 `MergeWorker`；负责选择列表、配置预览、日志与进程生命周期 |
| [lerobot_edit_dataset.py](../../src/lerobot/scripts/lerobot_edit_dataset.py) | 解析合并 CLI 配置；按显式输入 roots 加载数据，调用合并 API |
| [dataset_tools.py](../../src/lerobot/datasets/dataset_tools.py) / [aggregate.py](../../src/lerobot/datasets/aggregate.py) | `delete_episodes`、`merge_datasets` 与聚合实现，负责数据、视频、episode 索引和元数据的输出 |
| [lerobot_review_dataset.py](../../src/lerobot/yzh/lerobot_review_dataset.py) | 独立的 CustomTkinter 审查面板；`LocalDataset` 管元数据，`EpisodeVideoDecoder` 管分段视频，`ReviewPanel` 管播放、删除标记和另存 |

合并入口默认优先 Qt。`auto` 仅在缺少 PyQt5 导入时回退到可用的 Zenity、再到 Tk；显式 Qt 或使用 Qt 专属的 `source` / `demo` 功能时，需要 Qt。Qt 初始化把插件搜索路径指向当前 PyQt5 包，并启用高 DPI。界面样式与 GitHub 上传工具保持一致，但两者是独立窗口与任务实现。

## Qt 选择与预览

目录发现识别包含 `meta/info.json` 和 `data/` 的根目录，扫描所选目录及下一层 namespace，按自然顺序排列。`DatasetEntry` 只读取 `info.json` 的 episode 数、帧数和 FPS；读取失败的条目仍显示错误，勾选它会阻止启动。预览阶段不加载整套数据或解码视频。

勾选决定输入集合，列表顺序决定传给后端的合并顺序。搜索只隐藏条目，已勾选但被隐藏的输入仍参与合并；刷新保留手动添加的目录、勾选和既有顺序。自动名称由 namespace 与共同名称前缀生成，用户编辑后停止自动覆盖。

输出固定为配置的 LeRobot home 下的 `namespace/dataset`，不随扫描目录改变。`build_merge_command` 检查至少两个不同输入、输出 Repo ID、目录存在性及输入 / 输出包含关系；命令预览与执行使用同一构造函数。预览通过只说明这些前置条件成立，完整数据和特征兼容性由实际合并后端校验。`demo` 使用内置条目展示交互，不扫描或合并真实数据。

## 合并工作线程与后端

开始合并时窗口保存当前参数并锁定配置；`MergeWorker` 在 QThread 中启动独立进程组，以当前 Python 解释器执行编辑 CLI。参数以列表传入，不经 shell；子进程工作目录为仓库根目录，环境清理外部 Python / 动态库路径，再把本仓库 `src` 设为 Python 搜索路径。

工作线程持续读取合并后的 stdout / stderr，通过 Qt signals 更新日志和结果；主窗口的 QTimer 更新用时。进度条在运行中显示忙碌状态，不估算完成百分比。只有进程退出码为 0 且未请求取消才显示成功；停止先向进程组发 SIGTERM，4 秒后仍未结束再发 SIGKILL。任务尚未退出时窗口拒绝关闭，输出可能保留部分文件；处理方式见[数据工具排错](../troubleshooting/datasets.md)。

GUI 明确传入每份输入的 repo ID 与 root、输出目录，并关闭 Hub 上传。它开启 `fill_missing_intervention`：当输入中已有兼容的标量布尔 `intervention` 特征时，为缺少该列的来源补 `False`，已有值保留；所有输入均没有该列时，不凭空增加它。

后端调用链为 `handle_merge` → `merge_datasets` → `aggregate_datasets`。聚合前核对 FPS、robot type、features 和实际记录的完整性，再创建输出元数据、整理表格 / 视频及索引。Qt / Tk / Zenity 共用这条数据处理路径，界面切换不会改变合并规则。RECAP sidecar 的索引关系不由这些通用工具维护，数据边界见[RECAP 架构](recap.md)。

## 审查播放与过滤副本

审查窗口保留 CustomTkinter 实现。元数据加载与过滤保存分别运行在后台 Python 线程，通过 `root.after` 返回 UI 状态；播放、逐相机解码和画面缩放在 Tk 定时回调中执行，不使用合并工具的 QThread 或 CLI 子进程。

视频读取按 episode 元数据中的每路 `from_timestamp` / `to_timestamp` 定位，因此一个 MP4 可包含多个 episode。各路 `EpisodeVideoDecoder` 使用同一 episode 相对时间，播放时长取各相机片段的最短值；PyAV 按时间 seek / decode，并缓存已转换图像。切集会关闭原解码器、重置时间线，再开始新一集。

episode 编号页与当前播放 episode 分开保存：翻页只更新编号按钮，明确选集才切换视频。播放由 `playing`、`busy`、拖动状态和播放模式共同控制；到末尾后按模式暂停、单集循环或切下一集，解码失败会停止播放。

删除决定先保存在内存集合中，至少保留一个 episode。保存时复制删除集合，锁定界面并在输出父目录创建临时目录，调用 `delete_episodes` 生成过滤副本；成功后再次核对目标路径，再重命名到最终目录。源数据不被修改，输出不得已存在或与源目录互相包含。加载 / 保存期间禁止关闭，未保存删除标记由窗口单独跟踪。

## 维护验证

界面状态与数据输出分别验证：[Qt 窗口测试](../../tests/yzh/test_merge_datasets_window.py)使用测试子进程检查交互和任务生命周期；[审查测试](../../tests/scripts/test_review_dataset.py)检查播放状态、分段解码与过滤副本；[数据工具测试](../../tests/datasets/test_dataset_tools.py)及[干预标记合并测试](../../tests/datasets/test_merge_intervention.py)覆盖底层编辑 / 聚合约定。

这些入口用于定位软件行为；实际数据处理命令与验收步骤集中在[数据处理流程](../workflows/datasets/README.md)。
