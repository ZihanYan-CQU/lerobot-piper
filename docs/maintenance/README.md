# 文档维护

[返回文档首页](../README.md)

最近一次[指令核查记录（2026-11-07）](command-audit-2026-11-07.md)区分实际运行、参数检查和仍缺少的前置条件；日常完整命令继续在各流程页维护。

## 各类文档的职责

| 位置 | 放什么 | 不在这里重复维护什么 |
| --- | --- | --- |
| 根目录 `README.md` | LeRobot 项目介绍、本地文档入口 | 本机完整实验命令 |
| `docs/README.md` | 按任务查找文档的总导航 | 完整流程和问题历史 |
| `docs/workflows/<流程>/README.md` | 准备条件、执行命令、操作顺序、输出位置 | 大段代码原理、历次排错记录 |
| `docs/architecture/` | 模块职责、数据流、状态机、参数语义、数据约定 | 重复的完整启动配方 |
| `docs/troubleshooting/` | 症状、检查步骤、原因、处理办法、日期和验证范围 | 常规操作流程 |
| `docs/research/` | 论文对照、实验预算、历史实验和未实现方案 | 未验证为现状的承诺 |
| `docs/reference/` | 数据质量、策略选择、训练预算等跨流程参考 | 设备专用启动命令 |
| `docs/source/` | 上游通用教程和 API 文档，供文档网站构建 | Piper 本机实验日志 |
| 源码目录的 README | 目录职责和指向上述文档的入口 | 各维护一套的操作手册 |
| 资源目录的 README（如 `utils/assets/`） | 资源来源、原文件名、版本 / 下载日期、许可证或商标归属 | 操作命令与工具使用手册；从对应架构页链接进入 |
| `AGENTS.md` / `AGENT_GUIDE.md` | 开发约定 / 用户帮助导航 | 混合的实验笔记 |

一个操作只保留一个完整配方。不同机器或模式确实需要不同配方时，在所属流程内拆成子页，并从该流程的 README 链接过去。每页写清命令执行目录、变量定义和实际需要修改的值，避免依赖读者在另一页执行过的 shell 变量。

`docs/source/policy_*_README.md` 是策略简介、论文和引用；多个策略源码目录的 `README.md` 是指向这些文件的符号链接。保留这些上游组织方式和符号链接，使用步骤仍放在对应 `.mdx` 教程。

## 构建和预览上游文档网站

在仓库根目录执行，先安装 Node.js，再准备 Python 环境：

```bash
uv sync --locked --inexact --extra dataset
uv pip install --python .venv/bin/python -r docs-requirements.txt

# 生成文档到临时目录
uv run --no-sync doc-builder build lerobot docs/source/ --build_dir /tmp/lerobot-docs

# 在浏览器中预览
uv run --no-sync doc-builder preview lerobot docs/source/
```

文档工具临时安装到明确指定的项目环境，不写入 `uv.lock`；后续完整同步可能移除它们，需要构建时重新安装。

预览地址：[http://localhost:3000](http://localhost:3000)。`docs-requirements.txt` 已包含 `watchdog`，无需单独安装。构建产物不提交到 Git。此构建面向 `docs/source/`；本地流程手册直接通过 Markdown 阅读。

## 新增或移动内容

1. 本地流程优先补充已有主题；新流程在 `docs/workflows/` 建目录和 README，并加入总导航。
2. 网站教程在 `docs/source/` 编辑，新增页面加入 [`_toctree.yml`](../source/_toctree.yml)，`local` 填不带扩展名的相对路径。修改后重启预览并检查导航。
3. 移动正文时修正相对链接，在旧位置保留简短跳转；已有章节引用应保留锚点或更新引用方。
4. 问题记录写明日期、现象、已验证的结论和仍未确定的事项；后续结果另行补充，避免把旧猜测写成当前事实。
5. 检查 Markdown 链接、代码围栏和命令语法。纯文档整理无需启动机械臂或训练任务。

旧锚点迁移示例：

```html
<a id="old-section"></a>
<a href="./new-page.md#new-section">本节已迁移至新页面</a>
```

参数名、路径和代码值使用反引号；命令使用标注语言的代码块。网站图片和视频优先托管于 [documentation-images](https://huggingface.co/datasets/huggingface/documentation-images)，通过 URL 引用。

## 新增 README 的接入约定

新增内容先按用途归类，再同时补总导航和对应分类索引。源码或示例目录的 README 保留目录职责与跳转；
用户查找指令不应依赖知道代码放在哪个包。

| 主题 | 日常操作 | 实现与数据约定 | 问题处理 |
| --- | --- | --- | --- |
| USB 指示灯 | [灯态与配置](../workflows/tools/status-indicator.md) | [状态接口与后端](../architecture/status-indicator.md) | [连接与灯态排查](../troubleshooting/status-indicator.md) |
| Piper 仿真 | [仿真流程](../workflows/simulation/README.md) · [短流程](../workflows/simulation/smoke.md) | [插件与模型来源](../architecture/simulation.md) | [资源、回放与环境检查](../troubleshooting/simulation.md) |
| RLinf / RLT | [两阶段操作入口](../workflows/rlt/README.md) | [模型、特征与 replay](../architecture/rlt.md) | [checkpoint 与数据边界](../troubleshooting/rlt.md) |
| 数据审查与合并 | [数据处理](../workflows/datasets/README.md) | [窗口与编辑后端](../architecture/dataset-tools.md) | [播放、合并与输出](../troubleshooting/datasets.md) |
| 设备绑定 | [配置流程](../workflows/tools/device-bindings.md) | [模块与独立测试](../architecture/device-bindings.md) | [工具排错](../troubleshooting/tools.md) |
| GitHub 上传 | [上传流程](../workflows/tools/github-upload.md) | [模块与开发验证](../architecture/github-upload.md) | [工具排错](../troubleshooting/tools.md) |

本机流程直接使用 `uv run --no-sync`；仿真显式选择 `--project examples/piper_sim`。
内部桌面启动器按工具的实际实现说明，已退役入口集中记在环境排错页，不再作为日常命令的包装前缀。
上游网站教程、策略介绍符号链接和第三方来源 / 许可证说明保留原职责，不迁成 Piper 实验记录。
