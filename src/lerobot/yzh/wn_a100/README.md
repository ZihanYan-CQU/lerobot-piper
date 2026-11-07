# WN-A100 监控模块

此目录实现 GPU / 公告面板、训练进度、磁盘统计和共享 SSH 连接管理。
`monitor.py` 负责数据与 CLI，`desktop.py` 负责界面，`remote.py` 负责 SSH；
服务器与训练日志配置只维护本目录的 `config.json`。

- [启动、悬浮球、登录与监控操作](../../../../docs/workflows/tools/wn-a100.md)
- [模块职责、入口关系与历史文件迁移](../../../../docs/architecture/wn-a100.md)
- [工具问题与排错](../../../../docs/troubleshooting/tools.md)

日常命令统一在操作页维护；不要直接运行 `monitor.py` 或 `desktop.py`。
