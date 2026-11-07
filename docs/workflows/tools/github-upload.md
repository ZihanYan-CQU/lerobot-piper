# GitHub 代码上传助手

[返回工具导航](README.md) · [问题记录](../../troubleshooting/tools.md)

独立的中文桌面工具，使用 PyQt5，配色参考 `src/lerobot/yzh/wn_a100/monitor.py`。
界面使用平滑中文字体与 Qt 缩放，不依赖 Tk 的位图字体支持。
支持双击启动、常用参数保存、普通文件夹首次上传、已有 Git 仓库继续推送、改动预览和提交者身份。

默认勾选“仅上传当前代码”：从当前文件生成独立版本，跳过原项目提交历史和 Git LFS
资源（权重、测试数据、视频、模型网格等）。原项目的文件、提交历史和暂存区保留在本地。
后续上传会接着个人仓库的最新版本更新，支持新增、修改、删除；代码相同时不重复生成提交。

## 启动

已创建桌面图标时，双击 **GitHub 代码上传** 即可打开窗口，不需要先激活环境或进入项目目录。
也可以在应用菜单搜索“GitHub 代码上传”。移动整个项目后，需要重新安装启动图标。
可以将该应用添加到 Ubuntu 左侧栏的收藏，点击侧栏图标即可启动或切回窗口。

在项目根目录运行：

```bash
uv run --no-sync python utils/github_upload_gui.py
```

首次使用先执行 `uv sync --locked --inexact --extra desktop`；之后直接运行上述 Python 命令，
不在启动时下载依赖。桌面图标由内部启动器使用项目 `.venv`，无需手动输入脚本前缀。
环境缺失时按上述同步命令重建；解释器与动态库排查见[环境整理记录](../../troubleshooting/uv-conda.md)。

只查看示例界面（不会读写 Git 仓库、登录或上传）：

```bash
uv run --no-sync python utils/github_upload_gui.py --demo
```

也可以指定本地项目：

```bash
uv run --no-sync python utils/github_upload_gui.py /path/to/project
```

在其他 Linux 电脑上安装桌面图标：

```bash
uv run --no-sync python utils/github_upload_gui.py --install-launcher
```

首次创建参数文件；重复安装会保留已有参数。若桌面提示不允许启动，右键图标选择“允许启动”。

## 固定参数

默认参数文件是 `~/.config/lerobot-github-upload/config.toml`；设置了 `XDG_CONFIG_HOME` 时跟随该目录。
可以直接编辑，也可以在窗口左侧向下滚动到“常用参数”，点击 **保存配置**。
窗口顶部的 **参数文件** 可一键打开内置编辑器，直接修改实际 TOML 文件。
点击 **保存并应用** 后立即更新窗口，不需要重启；格式错误时保留原文件并显示提示。
点击“取消”则丢弃本次编辑。左侧“常用参数”中也有 **编辑参数** 入口。
若编辑期间文件已被其他程序修改，保存会被拒绝；关闭编辑器后重新打开，再合并本次修改。
保存本地目录、仓库地址、目标分支、默认提交说明、GitHub 用户名、提交者姓名和邮箱、Token 及上传模式。
主窗口中修改的值默认仅用于本次上传；点击 **保存配置** 才会写入文件。
内置 TOML 编辑器则通过 **保存并应用** 写入并立即重载，两者都属于显式保存。

参数示例见 [`utils/github_upload_config.example.toml`](../../../utils/github_upload_config.example.toml)。
常用字段如下：

```toml
directory = "/path/to/project"
remote = "https://github.com/你的用户名/仓库.git"
branch = "main"
username = "你的GitHub用户名"
name = "Git提交者姓名"
email = "your-email@example.com"
token = ""
code_only = true
```

在 `token` 的引号内填入自己的有效 Token 一次，后续启动和连续上传会自动使用。
GitHub 用户名用于 HTTPS 鉴权，`name` 和 `email` 用于提交记录；姓名和邮箱留空时从项目 Git 配置读取。
使用已有 Git 凭据或 SSH 时将 `token` 留空。用外部编辑器修改文件后重新启动窗口，或点击 **重新读取**；内置编辑器保存后会自动应用。

也可以显式指定另一个参数文件（必须放在待上传项目之外）：

```bash
uv run --no-sync python utils/github_upload_gui.py --config /path/outside/project/config.toml
```

参数文件以明文保存在项目之外，权限为 `600`，仅当前系统用户可读写，不会随项目上传。
工具不会把 Token 放入桌面启动命令、仓库地址或 Git 配置，也不会将其写进操作日志。

## 上传

1. 选择本地项目根目录，然后预览改动。普通文件夹会初始化为 Git 仓库。
2. 填写 GitHub 仓库地址和目标分支。新仓库可点击窗口中的创建入口，在 GitHub 创建空仓库。
3. 填写提交说明。姓名和邮箱会读取已有 Git 配置，也可以在窗口中填写。
4. 已配置 Git 凭据或 SSH 时，Token 留空；使用 HTTPS 时也可填写有目标仓库写权限的 Token。
5. 核对文件列表及确认弹窗中的目标，执行提交并上传。

手动输入且未保存的 Token 只用于本次操作，操作结束后恢复已保存的 Token（没有保存时清空）。权限配置可参考
[GitHub 的 Token 说明](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)。

在默认代码模式下，预览列出本次快照包含的全部文件，并显示跳过的 LFS 资源数量。
列表显示每个文件的大小，上方汇总全部选中文件的未压缩总大小，确认弹窗也会显示该大小。
忽略的环境目录和跳过的 LFS 资源不计入代码模式的大小；删除的文件标记为“已删除”。
这些资源不会出现在个人仓库中，需要使用时可从原项目获取。已有暂存文件采用当前工作区的内容，原暂存区不变。
代码上传不使用强制推送；如果多人同时更新目标分支导致推送被拒绝，重新预览并上传即可读取最新远端版本。

如果取消勾选“仅上传当前代码”，工具会提交全部新增、修改、删除和已有暂存内容，
并推送当前本地分支的完整历史到指定分支；这时历史中引用的 LFS 数据也必须完整可用。
该模式汇总的是当前改动文件的大小，不包含历史对象大小。
新文件遵循 `.gitignore`；已经被 Git 跟踪的文件仍会提交。部分暂存的文件将以当前工作区内容提交。
基础忽略规则仅在缺少 `.gitignore` 且开启对应选项时创建，请自行补充数据集、大模型和私密文件规则。
子模块和嵌套仓库的内容需在其各自仓库中上传。

工具不会覆盖已有 `origin`、切换本地分支或执行强制推送。完整历史模式遇到远端历史分歧时会提示处理，已生成的本地提交保留。
预览后文件或分支发生变化会要求重新确认；上传期间请避免同时修改代码或运行其他 Git 操作。

上传时底部显示准备文件、统计对象、压缩打包、发送代码等阶段的进度。
百分比来自文件准备或 Git 当前阶段的实际进度；连接及等待服务器确认时显示等待动画。
Git 报告传输量时，同时显示已发送大小及速度。网络传输会压缩并复用远端已有内容，
因此实际发送大小通常与未压缩文件总大小不同，进度条不把两者作为同一个总量。
只有远端确认推送成功后才显示“上传成功”和最终 100%。

工具的模块职责和开发验证入口见[架构说明](../../architecture/github-upload.md)。
