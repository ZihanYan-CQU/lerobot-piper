# GitHub 代码上传助手

独立的中文桌面工具，使用 CustomTkinter，配色参考 `src/lerobot/yzh/wn_a100_monitor.py`。
支持普通文件夹首次上传、已有 Git 仓库继续推送、改动预览、提交者身份和临时 Token。

默认勾选“仅上传当前代码”：从当前文件生成独立版本，跳过原项目提交历史和 Git LFS
资源（权重、测试数据、视频、模型网格等）。原项目的文件、提交历史和暂存区保留在本地。
后续上传会接着个人仓库的最新版本更新，支持新增、修改、删除；代码相同时不重复生成提交。

## 启动

在项目根目录运行：

```bash
bash utils/github_upload_gui.sh
```

启动器优先复用已有图形环境（包括项目环境或名为 `lerobot` 的 Conda 环境）。缺少界面依赖时，
通过 uv 或 pip 安装 CustomTkinter 和 Pillow，新增依赖放在项目的 `.cache/` 下，不需要安装训练依赖。
如果系统 Python 缺少 Tkinter，Ubuntu / Debian 可安装 `python3-tk`。
也可设置 `GITHUB_UPLOAD_PYTHON` 指定已有的 Python 解释器。

只查看示例界面（不会读写 Git 仓库、登录或上传）：

```bash
bash utils/github_upload_gui.sh --demo
```

也可以指定本地项目：

```bash
bash utils/github_upload_gui.sh /path/to/project
```

## 上传

1. 选择本地项目根目录，然后预览改动。普通文件夹会初始化为 Git 仓库。
2. 填写 GitHub 仓库地址和目标分支。新仓库可点击窗口中的创建入口，在 GitHub 创建空仓库。
3. 填写提交说明。姓名和邮箱会读取已有 Git 配置，也可以在窗口中填写。
4. 已配置 Git 凭据或 SSH 时，Token 留空；使用 HTTPS 时也可填写有目标仓库写权限的 Token。
5. 核对文件列表及确认弹窗中的目标，执行提交并上传。

Token 不保存到磁盘或 Git 配置，操作结束后清空。权限配置可参考
[GitHub 的 Token 说明](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)。

在默认代码模式下，预览列出本次快照包含的全部文件，并显示跳过的 LFS 资源数量。
这些资源不会出现在个人仓库中，需要使用时可从原项目获取。已有暂存文件采用当前工作区的内容，原暂存区不变。
代码上传不使用强制推送；如果多人同时更新目标分支导致推送被拒绝，重新预览并上传即可读取最新远端版本。

如果取消勾选“仅上传当前代码”，工具会提交全部新增、修改、删除和已有暂存内容，
并推送当前本地分支的完整历史到指定分支；这时历史中引用的 LFS 数据也必须完整可用。
新文件遵循 `.gitignore`；已经被 Git 跟踪的文件仍会提交。部分暂存的文件将以当前工作区内容提交。
基础忽略规则仅在缺少 `.gitignore` 且开启对应选项时创建，请自行补充数据集、大模型和私密文件规则。
子模块和嵌套仓库的内容需在其各自仓库中上传。

工具不会覆盖已有 `origin`、切换本地分支或执行强制推送。远端历史有分歧时会提示处理，已生成的本地提交保留。
预览后文件或分支发生变化会要求重新确认；上传期间请避免同时修改代码或运行其他 Git 操作。

## 验证

后端测试使用临时本地裸仓库，不连接 GitHub：

```bash
python3 -m unittest discover -s tests/utils -p test_github_upload_gui.py -v
```
