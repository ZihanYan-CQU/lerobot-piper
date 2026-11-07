# 相机与机械臂端口绑定工具

此目录可单独复制到 Linux 电脑，使用系统 Python / GTK 配置 RGB 相机别名和 Piper 主从 CAN。
相机别名持久化；CAN 配置在重启或重新插拔后需要再次激活。

- [系统依赖、启动、设备选择与绑定检查](../../../../docs/workflows/tools/device-bindings.md)
- [模块职责、授权进程与独立测试](../../../../docs/architecture/device-bindings.md)
- [设备问题与排错](../../../../docs/troubleshooting/tools.md)

独立复制此目录时，请同时保留上述操作说明。目录内的 `start.sh` 与 `.desktop` 文件是本工具入口；
迁移后按操作页重新生成图标。
