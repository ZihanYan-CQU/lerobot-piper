# GitHub 上传工具结构

[架构索引](README.md) · [操作流程](../workflows/tools/github-upload.md) · [工具排错](../troubleshooting/tools.md)

| 文件 | 职责 |
| --- | --- |
| [github_upload_gui.py](../../utils/github_upload_gui.py) | CLI、Git 状态、代码快照、目标校验与提交推送后端 |
| [github_upload_window.py](../../utils/github_upload_window.py) | PyQt5 窗口、参数文件编辑器、预览确认、文件大小与上传进度 |
| [github_upload_config.py](../../utils/github_upload_config.py) | 项目外的本机参数解析与保存、桌面与应用菜单图标 |
| [github_upload_gui.sh](../../utils/github_upload_gui.sh) | 桌面图标的内部启动器，使用已有项目环境并检查 PyQt5；不自动安装依赖 |
| [github_upload_config.example.toml](../../utils/github_upload_config.example.toml) | 可编辑参数格式示例，不存放实际凭据 |
| [assets/github-mark-white.svg](../../utils/assets/github-mark-white.svg) / [来源说明](../../utils/assets/README.md) | 桌面、启动栏和 Qt 窗口共用的 GitHub 官方白色标志；原始来源和商标说明随资源保留 |

日常终端从 Python CLI 进入，完整命令放在[操作页](../workflows/tools/github-upload.md)。
双击图标经内部启动器进入同一个 CLI。启动器通过 uv 固定根 `.venv`，清理外部 Python / 动态库 / Conda 环境；
旧 `GITHUB_UPLOAD_PYTHON` 变量会被清除。项目环境或依赖缺失时提示 uv 同步命令。

普通文件夹上传、代码快照与完整历史模式共享预览和目标确认；界面不另行实现一套 Git 推送逻辑。
Git 操作后端使用标准库，窗口使用项目 `desktop` extra 的 PyQt5。
参数默认放在 `~/.config/lerobot-github-upload/config.toml`；可用 `--config` 指定项目外文件。
参数内容与操作步骤在流程页维护。

## 参数编辑与图标

内置 TOML 编辑器复用 `parse_preferences()` 与 `save_preferences_text()`：先校验字段、类型和参数文件位于项目外，再检查打开后文件是否被外部修改。保存时保留编辑的注释，使用私有文件权限写入；成功后重载窗口，失败时保留原文件。主表单的“保存配置”则按当前表单值序列化，两条路径均由配置模块负责写盘。

`application_icon_path()` 定位随工具提供的白色 SVG。桌面文件的 `Icon` 使用资源绝对路径；Qt 窗口也加载同一文件，并以 `lerobot-github-upload` 作为应用名和 desktop file name，与快捷方式的 `StartupWMClass` 配套。迁移时保留 `utils/assets/`，再按[操作流程](../workflows/tools/github-upload.md#启动)重新安装快捷方式。

资源目录 README 只维护来源与商标归属，不另设启动命令或使用手册。

## 开发验证

在仓库根目录运行；测试使用临时本地裸仓库，不连接 GitHub：

```bash
uv run --no-sync python -m unittest discover -s tests/utils -p 'test_github_upload*.py' -v
```
