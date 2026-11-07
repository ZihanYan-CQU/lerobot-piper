# 远端连接与代码迁移

[部署目录](README.md) · [DAgger](../dagger/README.md) · [RECAP](../recap/README.md) · [环境准备](../setup/README.md)

所有命令从对应机器的仓库根目录执行。先启动服务器，再在机器人电脑建立隧道，最后启动客户端。远端示例监听 `127.0.0.1:8080`，客户端经本地 `127.0.0.1:8080` 连接。

## SSH 隧道

**SSH 隧道——跨机器连接时执行一次：**

```bash
env -u LD_LIBRARY_PATH ssh \
  -4 -fN \
  -o ExitOnForwardFailure=yes \
  -J jumper@cps.cslcqu.cn:6666 \
  -L 127.0.0.1:8080:127.0.0.1:8080 \
  -p 1143 \
  yzh@192.168.1.50
```

已建立隧道时不必重复执行。同机运行服务端和客户端时可以跳过。这里仅对 SSH 进程取消 `LD_LIBRARY_PATH`，当前终端的 Python 动态库设置继续保留。

跳板机、端口、用户名和目标主机是现有环境示例；迁移到其他机器时修改对应字段。SSH 的转发端口应与服务端 `port` 和客户端 `server_address` 一致。

## 迁移服务端

两台机器需先具备相同版本的 LeRobot 基础代码和各自所需依赖。这里划分的是 `async_inference` 子包，并非独立安装包；`policies/`、`processor/`、`transport/`、`utils/` 等目录仍来自 LeRobot。

首次迁移应携带完整仓库，包括 `pyproject.toml`、`uv.lock` 和 `.python-version`，然后在目标机器运行对应的 `uv sync --locked --extra ...`。不要搬运旧机器的 `.venv`。下面的增量复制仅适用于目标机器已有配套基础代码和环境的情况；依赖配置有变化时也需同步上述文件。

在仓库根目录运行下列命令，将主机名和目标仓库路径替换为实际值：

```bash
rsync -av --exclude='__pycache__/' \
  src/lerobot/async_inference/__init__.py \
  src/lerobot/async_inference/shared \
  src/lerobot/async_inference/server \
  user@server:/path/to/lerobot/src/lerobot/async_inference/
```

迁移客户端时，同样携带 `__init__.py` 和 `shared/`，将 `server/` 换为 `client/`。

使用上述 JSON 启动方式时，也需将 `config/` 中对应的配置文件复制到目标机器的 `src/lerobot/async_inference/config/`。

**此次目录调整需要两端同步更新并重启。** 通信字段和协议版本没有变化，但 pickle 会记录对象的完整 Python 模块路径；通信类型现在位于 `lerobot.async_inference.shared.*`。旧平铺目录进程与新目录进程不能混用。外部启动脚本、服务配置中的模块路径也应更新为上表路径。

## 旧 RTC 命令迁移

旧 RTC 迭代链的八个实现文件及六个专用测试文件已删除。当前 deployment / dagger / recap 不依赖它们；公共协议和仍适用的传输测试继续保留，并通过当前 deployment 入口验证。

| 旧入口 | 当前入口 |
| --- | --- |
| `policy_server_rtc`、`policy_server_rtc_resettable` | `server.policy_server_deployment` |
| `robot_client_rtc` 及 interpolated / filtered / resettable 变体 | `client.robot_client_deployment` |
| `policy_server_rtc_resettable_dagger` | `server.policy_server_dagger` |
| `robot_client_rtc_interpolated_filtered_resettable_dagger` | `client.robot_client_dagger` |

表中的模块路径均位于 `lerobot.async_inference.` 下。两端应成对切换；旧配置中的 `observation_similarity_atol`、`cooldown_margin_steps`、`aggregate_fn_name`（客户端）和 `obs_queue_timeout`（服务端）不适用于当前 deployment 入口，应移除。[部署流程](README.md) 中的命令已同步迁移，其他外部脚本请按当前入口的 `--help` 检查参数。
