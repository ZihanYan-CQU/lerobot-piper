# 异步推理目录

[架构索引](README.md) · [远程部署架构](deployment.md) · [迁移与连接指令](../workflows/deployment/connection.md)

代码按部署位置分为三类，主要入口的文件名和算法实现保持不变：

| 目录 | 职责 | 迁移到哪端 |
| --- | --- | --- |
| `shared/` | 双端共用的配置、通信对象、图像传输、诊断、RTC 与 RECAP 辅助逻辑 | 两端 |
| `server/` | 策略加载、推理、PI05 预热和 guided RTC 处理器 | 服务端 |
| `client/` | 机器人控制、插值、人工接管、DAgger / RECAP 数据记录 | 客户端 |

`shared/configs.py` 和 `shared/rtc_trained.py` 仍完整保留两端代码。`shared/` 不导入本目录下的服务端或客户端入口；服务端与客户端也不互相导入。子包的 `__init__.py` 不主动加载入口。

## 当前使用的入口

| 用途 | 服务端模块 | 客户端模块 |
| --- | --- | --- |
| Deployment | `lerobot.async_inference.server.policy_server_deployment` | `lerobot.async_inference.client.robot_client_deployment` |
| DAgger | `lerobot.async_inference.server.policy_server_dagger` | `lerobot.async_inference.client.robot_client_dagger` |
| RECAP | `lerobot.async_inference.server.policy_server_recap` | `lerobot.async_inference.client.robot_client_recap` |

## 文件维护

- 会话代际编号常量及三个编解码函数已整体合入 `shared/rtc_remote.py`，原 `shared/rtc_session.py` 已删除。通信对象和会话编号规则现在统一维护，编号格式与函数行为不变。
- `server/policy_server_dagger.py` 虽然较短，但定义了独立入口和 DAgger 模型缓存默认值，RECAP 也复用它；保留可以直接比较三种运行方式。
- `client/piper_hil_dagger.py` / `piper_hil_recap.py` 分别承载硬件与记录逻辑，不并入已经较大的客户端入口。
- `shared/metadata_recap.py` 同时被服务端、客户端和记录器使用，保留它可以避免三者相互导入。

基础异步入口 `server/policy_server.py`、`client/robot_client.py` 仍由 `examples/tutorial/async-inf/`、异步推理和 Unitree 文档使用，也有独立端到端测试，因此本轮保留。
