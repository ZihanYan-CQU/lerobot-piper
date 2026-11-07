# 环境、认证与运行日志

[排错索引](README.md) · [环境准备](../workflows/setup/README.md) · [训练流程](../workflows/training/README.md)

本机 uv 环境清单、直接运行命令和外部环境混用排查见 [环境整理记录](uv-conda.md)。

这里集中通用环境诊断。认证和日志查看命令在仓库根目录执行，依赖先按对应流程准备。

```bash
uv run --no-sync hf auth login
uv run --no-sync wandb login

# 按实际训练日志位置替换
tail -f /path/to/train.log
watch -n 1 nvidia-smi
ps -u "$USER" -f

# 检查数据所在文件系统、临时目录及缓存
df -h "$HOME" /tmp
du -sh ~/.cache/huggingface
```

W&B 需同时在训练命令设置 `--wandb.enable=true`；`--wandb.disable_artifact=true` 关闭模型 Artifact 上传，`WANDB_MODE=offline` 用于离线记录。`tail -f` 中按 `Ctrl+C` 只退出日志查看。停止自己的训练进程用 `kill <PID>`。

| 现象 | 处理 |
| --- | --- |
| SSH 动态库冲突 | 仅对 SSH 清理变量：`env -u LD_LIBRARY_PATH /usr/bin/ssh user@server` |
| HF SSL / 镜像错误 | 检查 `HF_ENDPOINT`；错误镜像可用 `unset HF_ENDPOINT` 恢复官方源 |
| 磁盘不足 | 对实际数据目录执行 `df -h /path/to/data`、`du -sh /path/to/data/*`，同时检查 `/tmp` |
| 桌面工具无法开窗 | 在图形桌面终端运行并检查 `DISPLAY`；GPU 监控可改用 `--once` |
| 锁文件或缺依赖 | 确认代码、`pyproject.toml`、`uv.lock`、`.python-version` 配套，再按用途同步 extra |

认证信息通过交互登录或本机配置保存；不写入共享命令或文档。
