# 数据集检查、过滤、合并与传输

[返回文档导航](../../README.md) · 上一步：[演示数采](../recording/README.md) / [DAgger](../dagger/README.md) · 下一步：[训练](../training/README.md) · [数据工具实现](../../architecture/dataset-tools.md) · [数据工具排错](../../troubleshooting/datasets.md)

首次准备依赖：`uv sync --locked --inexact --extra dataset_viz --extra desktop`。下方 `uv run --no-sync` 只运行已准备好的 `.venv`，不在启动时安装依赖。

命令在仓库根目录执行。每节都有自己的示例变量，按实际数据修改后单独执行；输入目录必须是完整的数据集根目录，至少包含 `meta/info.json` 和 `data/`；使用视频特征时还需保留对应视频文件。

| 需要做什么 | 入口 |
| --- | --- |
| 播放一个 episode、看动作 | [命令行可视化](#1-查看数据集与单个-episode) |
| 多相机人工审查并另存过滤副本 | [审查面板](#2-人工审查与过滤) |
| 按已知 episode 编号删除 | [命令行过滤](#命令行过滤已知-episode) |
| 合并分批演示或演示 + DAgger | [合并](#3-合并数据集) |
| 上传 / 下载 Hugging Face 数据 | [传输](#4-hugging-face-上传与下载) |

本页的编辑、合并工具尚未接入 RECAP 的 sidecar。RECAP 数据按 [RECAP 流程](../recap/README.md)处理，避免把附加标注与 episode 索引分离。

## 1. 查看数据集与单个 episode

```bash
DATASET_REPO_ID='block/shape-matching_0-10'
DATASET_ROOT="$HOME/.cache/huggingface/lerobot/${DATASET_REPO_ID}"

uv run --no-sync lerobot-edit-dataset \
  --repo_id="$DATASET_REPO_ID" \
  --root="$DATASET_ROOT" \
  --operation.type=info \
  --operation.show_features=true

uv run --no-sync lerobot-dataset-viz \
  --repo-id "$DATASET_REPO_ID" \
  --root "$DATASET_ROOT" \
  --episode-index 0
```

`episode-index` 从 `0` 开始。可视化命令使用 `--repo-id` / `--episode-index`（连字符），编辑命令使用 `--repo_id`（下划线），不要混用。桌面可视化需要可用图形会话。

显式 `--root` 能避免因 HF 缓存或自定义存储目录不同而找错数据。省略时按当前 `HF_LEROBOT_HOME` 约定查找，必要时可能从 Hub 下载。

## 2. 人工审查与过滤

```bash
uv run --no-sync \
  python src/lerobot/yzh/lerobot_review_dataset.py
```

此审查面板要求数据集包含视频相机。在面板中选择数据集根目录，查看多路相机，标记待删除的 episode，选择输出目录，再另存过滤副本。工具默认建议未占用的 `*_filtered` 名称；输出目录不能已存在，也不能与源目录互相包含。原始数据保留。

Episode 使用编号按钮选择，每页显示 10 个；翻页只切换编号页，不打断当前播放。也可输入从 `0` 开始的编号后按回车或点击“跳转”，并用“上一集 / 下一集”逐集切换。当前集高亮，待删除集显示红色。

面板默认 2 倍速、单集循环，可调回 1 倍查看细节。通过“播完暂停 / 单集循环 / 连续播放”按钮切换播放方式；连续播放会自动进入下一集，最后一集结束后回到第 `0` 集，暂停按钮可随时停止播放。用“标记删除当前 Episode” / “保留当前 Episode”调整清单，再点击“生成过滤数据集”；至少保留一条。没有删除项时保留原数据集即可。

界面需要带 Xft 支持的 Tk。已同步环境的直接入口是 `uv run --no-sync python -m lerobot.yzh.lerobot_review_dataset`，使用前需安装 `desktop` 和 `dataset` extra；普通采集能运行不代表审核依赖齐全。依赖和开窗问题统一见[数据工具排错](../../troubleshooting/datasets.md)。

### 命令行过滤已知 episode

旧笔记中“从 70–80 批次删除第 4 号 episode”的完整配方：

```bash
SOURCE_REPO_ID='block/shape-matching_70-80'
SOURCE_ROOT="$HOME/.cache/huggingface/lerobot/${SOURCE_REPO_ID}"
FILTERED_REPO_ID='block/shape-matching_70-80_filtered'
FILTERED_ROOT="$HOME/.cache/huggingface/lerobot/${FILTERED_REPO_ID}"

uv run --no-sync lerobot-edit-dataset \
  --repo_id="$SOURCE_REPO_ID" \
  --root="$SOURCE_ROOT" \
  --new_repo_id="$FILTERED_REPO_ID" \
  --new_root="$FILTERED_ROOT" \
  --operation.type=delete_episodes \
  --operation.episode_indices='[4]' \
  --push_to_hub=false
```

`[4]` 指数据集内部索引为 `4` 的 episode，不是目录名中的采集编号。多项写为 `'[0, 4, 7]'`。使用不同的 `new_repo_id`、`new_root` 输出副本；不要省略它们后误做原地编辑。输出目录应尚不存在。

## 3. 合并数据集

### 图形面板：适合目录不在同一位置

```bash
uv run --no-sync python src/lerobot/yzh/merge_datasets_gui.py
```

选择输入数据集、确认输出位置与标识，再执行。面板向合并 CLI 显式传入每份输入的根目录，且不上传 Hub。混合已有标量布尔 `intervention` 的数据与缺少该列的演示时，GUI 为缺列来源补 `False`，保留已有标记；所有输入都没有该列时不会新增。

默认使用深色 Qt 桌面面板，与 GitHub 代码上传工具保持一致。左侧扫描源目录及下一级命名空间目录，也可用“＋ 添加目录”加入其他位置的数据集；列表显示 episode / 帧数 / FPS。勾选决定是否参与合并，选中行后用 ↑ / ↓ 调整顺序，Ctrl / Shift 可选中多行一起移动。

搜索只隐藏不匹配的行，隐藏行原有的勾选仍参与合并；“全选”只勾选当前可见行，“清空勾选”清除全部选择。刷新保留仍可找到的已选数据集、手动添加目录和排序。右侧自动生成 Repo ID，手动修改名称后不会被选择或刷新覆盖，可以点击“恢复自动”重新使用建议名称。

输出位置是当前 LeRobot 数据存储根目录加 `namespace/dataset` 格式的 Repo ID，右侧会显示完整路径。根目录由 `HF_LEROBOT_HOME` 决定，未设置时按 `HF_HOME` / `XDG_CACHE_HOME` 推导，通常是 `~/.cache/huggingface/lerobot`。更换输入源目录或使用 `--source` 不会改变输出根目录；需要另存到其他位置时，在启动前将 `HF_LEROBOT_HOME` 设为一个已存在的目录，或使用下方可指定 `--new_root` 的 CLI。

至少勾选两个数据集、元信息可读、Repo ID 合法且输出目录未被占用后，“开始合并”才可点击；输入的完整性和特征兼容性还会在实际合并时检查。右上角“展开日志”可查看“运行日志”和“执行命令”，开始合并时自动展开。运行时配置锁定并显示用时，完成后可以打开输出目录。点击“停止合并”会结束合并进程，可能留下不完整的输出目录，检查后再使用。

指定启动时扫描的源目录，或仅预览示例界面（不读取、合并真实数据）：

```bash
uv run --no-sync python src/lerobot/yzh/merge_datasets_gui.py --source /path/to/datasets
uv run --no-sync python src/lerobot/yzh/merge_datasets_gui.py --demo
```

Qt 需要 `desktop` extra；未安装且未传入 `--source` / `--demo` 时，`auto` 模式优先回退到可用的 Zenity，否则使用 Tk。`--source`、`--demo` 只支持 Qt；指定 `--backend qt` 或这两个参数时，缺少 Qt 会直接报错。旧界面可显式选择 `--backend zenity` 或 `--backend tk`；`--backend` 优先于环境变量，原环境变量仍然有效：

```bash
LEROBOT_GUI_BACKEND=tk uv run --no-sync \
  python src/lerobot/yzh/merge_datasets_gui.py
```

### 命令行：合并已有缓存中的分批演示

下面保留旧笔记中的九批合并方案。输入均放在默认 LeRobot 缓存对应的 `block/...` 目录时可直接使用；若实际在其他目录，改用下一节的 `operation.roots`。

```bash
MERGED_REPO_ID='block/shape-matching_89'
MERGED_ROOT="$HOME/.cache/huggingface/lerobot/${MERGED_REPO_ID}"

uv run --no-sync lerobot-edit-dataset \
  --new_repo_id="$MERGED_REPO_ID" \
  --new_root="$MERGED_ROOT" \
  --operation.type=merge \
  --operation.repo_ids='["block/shape-matching_0-10", "block/shape-matching_10-20", "block/shape-matching_20-30", "block/shape-matching_30-40", "block/shape-matching_40-50", "block/shape-matching_50-60", "block/shape-matching_60-70", "block/shape-matching_70-80_filtered", "block/shape-matching_80-90"]' \
  --push_to_hub=false
```

`shape-matching_89` 只是原数据集名称，实际轮数以合并后元信息为准。输入的相机、FPS、特征和任务编码等需要兼容，输出目录应是新的目录。

### 命令行：显式目录合并演示 + DAgger

```bash
DATASET_BASE_DIR='/home/yzh/具身智能/训练数据'
DEMO_ROOT="${DATASET_BASE_DIR}/shape-matching_89"
DAGGER_ROOT="${DATASET_BASE_DIR}/dagger_raw"
MERGED_REPO_ID='sukasu403/shape-matching_89_with_dagger_30'
MERGED_ROOT="${DATASET_BASE_DIR}/shape-matching_89_with_dagger_30"

uv run --no-sync lerobot-edit-dataset \
  --new_repo_id="$MERGED_REPO_ID" \
  --new_root="$MERGED_ROOT" \
  --operation.type=merge \
  --operation.repo_ids='["block/shape-matching_89", "sukasu403/dagger_raw"]' \
  --operation.roots="[\"${DEMO_ROOT}\", \"${DAGGER_ROOT}\"]" \
  --operation.fill_missing_intervention=true \
  --push_to_hub=false
```

按实际数据修改两个输入标识及目录；`operation.repo_ids` 与 `operation.roots` 必须一一对应且长度相同。CLI 默认不补 `intervention`，混合无标记的演示与 DAgger 时应像上面显式打开。原本已有的干预标记保留，不把整份 DAgger 的所有帧都改成干预帧。更多 DAgger 审核规则见 [DAgger 流程](../dagger/README.md)。

## 4. Hugging Face 上传与下载

需登录时先执行 `uv run --no-sync hf auth login`。本地标识和 Hub 仓库名可以不同；上传目标要使用自己有写权限的用户名 / 组织名。

### 上传已审核的数据

```bash
HUB_REPO_ID='sukasu403/shape-matching_89_with_dagger_30'
LOCAL_DATASET_ROOT="$HOME/.cache/huggingface/lerobot/block/shape-matching_89_with_dagger_30"

uv run --no-sync hf upload "$HUB_REPO_ID" "$LOCAL_DATASET_ROOT" \
  --repo-type dataset
```

这会写入远端数据集仓库，运行前核对目标仓库和本地目录。命令默认上传到 `main`；旧笔记使用 `--revision v1.0`，需要版本分支时确认该分支已存在且可写，再添加该参数。下载的 revision 可以是已有分支、标签或提交。

### 下载到训练服务器

```bash
HUB_REPO_ID='sukasu403/shape-matching_89_with_dagger_30'
DATASET_ROOT='/home/yzh/具身智能/训练数据/shape-matching_89_with_dagger_30'
DATASET_REVISION='v1.0'

uv run --no-sync hf download "$HUB_REPO_ID" \
  --local-dir "$DATASET_ROOT" \
  --repo-type dataset \
  --revision "$DATASET_REVISION"
```

该仓库没有 `v1.0` 时改为实际版本，例如 `main`；下载 `dagger_merged` 时同时修改 `HUB_REPO_ID` 和 `DATASET_ROOT`。随后在训练命令中用一致的 `dataset.repo_id` 与 `dataset.root`。基座模型和 tokenizer 的下载配方放在[训练流程](../training/README.md)。

参数依据：[数据编辑入口](../../../src/lerobot/scripts/lerobot_edit_dataset.py)、[可视化入口](../../../src/lerobot/scripts/lerobot_dataset_viz.py)、[审查工具](../../../src/lerobot/yzh/lerobot_review_dataset.py)、[合并工具](../../../src/lerobot/yzh/merge_datasets_gui.py)。

## 采后验收与交付

以下是三相机现场指南生成的数据结构，实际文件以采集配置为准：

```text
piper_demo_年月日_时分秒/
├── meta/
│   ├── info.json                 基本信息和已保存条数
│   ├── stats.json                数据统计
│   ├── tasks.parquet             任务说明
│   └── episodes/                 每条演示索引和视频位置
├── data/                         状态、动作、时间戳等
└── videos/
    ├── observation.images.top/
    ├── observation.images.wrist/
    └── observation.images.down/
```

视频和表格按 `chunk-000/file-000.*` 分块，一个 MP4 可能包含多条 episode。不能用 MP4 数量当采集条数，也不能单删某个 MP4 来删除演示。

把 `DATASET_ROOT` 改成本次实际数据集目录，完全退出采集后做只读检查：

```bash
DATASET_ROOT='/home/cpy/桌面/yzh/piper/lerobot/outputs/datasets/piper_demo_实际时间戳'

uv run --no-sync python - "$DATASET_ROOT" <<'PY_CHECK'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())
print("数据目录：", root)
print("已保存条数：", info["total_episodes"])
print("总帧数：", info["total_frames"])
print("记录帧率：", info["fps"])
print("机械臂类型：", info["robot_type"])
print("相机：", [key for key in info["features"] if key.startswith("observation.images.")])
PY_CHECK
```

以下入口使用已经同步的项目环境。现场三相机配方应看到 `30` FPS、`piper_follower`、`top` / `wrist` / `down`；两相机配方则没有 `down`。提前结束导致条数少于计划数量是正常的，重录丢弃的片段不计数。

计数正确不代表演示质量合格。检查夹爪与目标是否可见，是否黑屏、模糊、相机位移或长时间停顿，再按本页审查流程过滤坏片段。审核工具暂不可用时可先用播放器检查各相机视频，但按 episode 筛选仍须使用数据工具。

交付完整数据集根目录，保持 `meta/`、`data/`、`videos/` 相对位置；只交付视频不足以训练。同时注明任务描述、保留的 episode 数、相机安装位置与画面方向、物体摆放方式，以及交付的是原始数据还是过滤副本、是否仍有待处理片段。
