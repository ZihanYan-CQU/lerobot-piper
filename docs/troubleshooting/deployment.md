# 远程部署排错：请求、复位、重连与诊断日志

[排错索引](README.md) · [部署指令](../workflows/deployment/README.md) · [部署架构](../architecture/deployment.md)

先按现象定位，再只开启需要的诊断。下文的重连/CUDA 记录保留 2026-10-16 的证据与未验证范围，不表示已经确认根因或解决所有重连问题。延迟与编译测速另见 [PI05 性能排查](pi05-performance.md)。

<a id="13-从现象定位代码"></a>

## 从现象定位代码

| 现象 / 问题 | 先看哪里 |
| --- | --- |
| 为什么暂时不发新请求 | `rollout_state` → 请求槽 → `_request_observation_if_ready()` 的队列阈值 |
| 图像采集比预期慢 | `RTC OBS SNAPSHOT` 的 sensor/camera/cache age，以及 `RTC RPC PHASES` 的 observation wait/capture 时间 |
| 收到结果却没执行 | generation/retired ID → `_handle_action_chunk()` → guided 迟到检查或 trained `accept()` |
| 队列经常空 | RTT、`actions_per_chunk`、触发阈值、被丢弃原因；`RTC STARVATION` 是诊断，不会补造动作 |
| chunk 衔接跳变 | `RTC ALIGN`（服务端生成对齐）、`RTC MERGE LIVE` / splice、`RTC COMMAND ANOMALY` 三层分开看 |
| 原始动作连续但实体落后 | 插值目标、`send_action()` 返回值、实际关节反馈、EMA 分别比较 |
| 复位后不能开始 | `/status`、`reset_error`、位置反馈和 settle 条件 |
| 修改 checkpoint 文件却仍用旧模型 | deployment 的 `reuse_last_policy` 以引用/路径匹配；关闭复用并重连或使用新 checkpoint 路径 |

代码阅读可配合现有回归测试：

- [客户端生命周期与调度](../../tests/async_inference/test_robot_client_deployment.py)
- [服务器缓存、前缀及重试](../../tests/async_inference/test_policy_server_deployment.py)
- [trained 隔离、硬前缀和拒绝规则](../../tests/async_inference/test_trained_rtc_deployment.py)
- [请求生命周期、历史绘图、废弃配置](../../tests/async_inference/test_deployment_request_lifecycle.py)
- [复位、异常和传输回归](../../tests/async_inference/test_deployment_p2_regressions.py)

这些测试帮助理解实现的行为边界，不能替代在目标模型、网络和机器人上的延迟及轨迹验证。

<a id="19-重连时的段错误与-cuda-非法内存访问"></a>

## 重连时的段错误与 CUDA 非法内存访问

服务端默认将连接、模型加载/复用、线程退出/启动、PI05 预热进度、校准结果和错误写入 `logs/policy_server_deployment_fault_<PID>.log`，路径相对于服务端启动目录，启动时会打印绝对路径。DAgger 和 RECAP 入口使用各自的文件名前缀。日志同时记录 PyTorch/CUDA/gRPC 版本、编译开关和相关环境变量。

详细的 `RTC PHASE begin/end` 跟踪仅在 `--debug_rtc_diagnostics=true` 或 `--debug_cuda_sync=true` 时开启，包含 session 编号、进程/线程 ID、模型对象 ID 和设备；`begin` 后没有对应 `end` 的阶段可帮助缩小崩溃范围。只需要阶段日志时使用 `debug_rtc_diagnostics`；需要额外 CUDA 同步时再使用 `debug_cuda_sync`。两者均关闭时跳过这些阶段的计时和元数据采集，实际失败仍会记录并检查是否需要重启服务端。

该文件还接收 `faulthandler` 在 SIGSEGV、SIGABRT 等致命信号发生时输出的所有 Python 线程堆栈，普通阶段日志逐条刷新。Python 3.12 的这些堆栈用于定位调用位置，不是完整的 C/CUDA 原生堆栈；必要时还需结合 core dump。参见 [Python faulthandler](https://docs.python.org/3.12/library/faulthandler.html)。需要固定文件名时，在原服务端命令后追加 `--fault_log_path=logs/deployment_crash.log`（追加写入；并行运行的服务应使用不同文件）。

排查复现时，在**启动服务端的同一终端**先设置：

```bash
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
```

然后在原服务端启动命令后追加：

```bash
--debug_cuda_sync=true
```

这会在加载、预热、校准、推理等阶段的边界记录并同步 CUDA，额外打印 guided 预处理、预测和后处理的细分阶段。`CUDA_LAUNCH_BLOCKING=1` 让异步错误更接近触发它的操作暴露；`TORCH_SHOW_CPP_STACKTRACES=1` 用于 PyTorch C++ 异常，不能代替段错误堆栈。诊断会改变执行时序并增加延迟，不要用这次运行测稳态 RTT；完成后关闭 `debug_cuda_sync` 并取消这些环境变量。参见 [PyTorch CUDA 异步执行说明](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)和 [C++ 异常堆栈开关](https://docs.pytorch.org/docs/2.14/debugging_environment_variables.html)。

2026-10-16 提供的旧日志在 guided RTC 预热的编译路径中报告 `CUDA error: an illegal memory access was encountered`，随后 RNG 状态恢复又报告同一错误。CUDA 异步执行意味着 `set_rng_state` 可能只是发现错误的位置，不能据此断言 RNG 恢复是根因，也不能仅凭 `Segmentation fault` 认定两次故障同源。

非法内存访问、device-side assert 等致命 CUDA 错误会让当前进程的 CUDA 状态不可继续使用。服务端现在保留首次错误，停止推理重试，并让新连接收到 `FAILED_PRECONDITION` 和 `restart the server process`；`Ready`、重新加载权重或清空缓存都不会清除该状态。必须重启**服务端进程**后再连接客户端。普通输入错误及 OOM 不被自动归入这类不可恢复错误。参见 [NVIDIA CUDA 错误定义](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html)。

如果需要缩小编译/重连相关范围，每组实验都从全新的服务端进程开始，一次只改变一项，并比较多次客户端连接：

| 实验 | 修改原服务端参数 | 用途 |
| --- | --- | --- |
| 禁用 CUDA Graphs，保留编译 | `--pi05_inference.compile_model=true --pi05_inference.compile_mode=max-autotune-no-cudagraphs` | 比较 CUDA Graphs 相关路径 |
| 禁用编译 | `--pi05_inference.compile_model=false` | 比较 eager 路径 |
| 禁用重连时的权重复用 | `--reuse_last_policy=false` | 比较保留模型运行时是否与故障相关 |

这些是定位手段，不代表已确认根因或修复段错误。下一次复现时保留 fault 日志、终端 stderr（编译器/原生库输出可能只在这里）、完整启动命令，以及第几次客户端连接触发崩溃。

<a id="20-正常推理时关闭诊断开销"></a>

## 正常推理时关闭诊断开销

以下适用于 canonical deployment 以及继承它的 DAgger / RECAP 入口。正常运行默认关闭逐请求/逐控制周期的可选诊断；旧启动命令若显式传了 `true`，需删除或改为 `false`。

| 开关 | 位置 / 默认值 | 开启的工作；关闭后的行为 |
| --- | --- | --- |
| `debug_rtc_diagnostics` | 两端分别设置 / `false` | 客户端动作差分、反向/时序告警、观测跟踪、缺动作统计、合并对比、控制频率统计；服务端逐块对齐、启动/推理阶段跟踪和传输日志。关闭后跳过这些计算及诊断张量复制；服务端阶段跟踪还需同时关闭 `debug_cuda_sync` |
| `log_rtc_timing` | 客户端 / `false` | 可单独收集/打印 `RTC RPC PHASES`、观测年龄和图像上传统计；关闭它及 `debug_rtc_diagnostics` 后不采集可选子阶段时间，也不逐块打印这些日志 |
| `debug_visualize_queue_size` | 客户端 / `false` | 逐 tick 保存队列长度，退出时画图；关闭后不收集历史 |
| `debug_cuda_sync` | 服务端 / `false` | GPU 阶段边界的诊断同步及日志；关闭后不增加这些同步点 |
| `rtc.debug` | 服务端 / `false` | 模型 RTC 去噪轨迹跟踪，会保存中间张量；平时保持关闭 |
| `enable_fault_handler` | 服务端 / `true` | 保存额外 fault 日志并注册致命信号堆栈处理；不轮询、不逐步复制张量、不调用 CUDA 同步。可用 `false` 关闭这份日志/处理器，不会取消致命 CUDA 错误后的重启要求 |
| `suppress_can_debug_logs` | 客户端 / `true` | 保持开启可抑制 python-can 的原始 CAN 帧 DEBUG 日志 |

两端启动入口默认将进程日志级别设为 INFO，`debug_rtc_diagnostics=true` 才启用 DEBUG 文件记录。相机读帧计时只在相机日志器允许 DEBUG 时计算，DAgger 双臂命令耗时跟随客户端 `debug_rtc_diagnostics`。启停、重连和实际错误的日志仍可用。`PI05 warmup ordinary/rtc ... starting`、每轮预热耗时以及简洁的 `RTC calibration warmup/measure/ready` 结果默认保留；`RTC PHASE begin/end` 和每次接收的 `Starting receiver` 在服务端两个诊断开关均关闭时不再输出，也不采集对应的可选阶段计时、线程元数据和接收分块统计。

明确指定正常配置时，在各自原命令后使用：

```bash
# 服务端
--debug_rtc_diagnostics=false --debug_cuda_sync=false --rtc.debug=false

# 客户端
--debug_rtc_diagnostics=false --log_rtc_timing=false --debug_visualize_queue_size=false --suppress_can_debug_logs=true
```

如果此前按[重连时的段错误与 CUDA 非法内存访问](deployment.md#19-重连时的段错误与-cuda-非法内存访问)开启过环境变量，在**服务端启动终端**取消后再启动新进程：

```bash
unset CUDA_LAUNCH_BLOCKING TORCH_SHOW_CPP_STACKTRACES TORCH_LOGS TORCH_COMPILE_DEBUG
```

`debug_cuda_sync=false` 无法抵消进程外设置的 `CUDA_LAUNCH_BLOCKING=1`。若还希望关闭全部 Python 致命信号处理，除 `--enable_fault_handler=false` 外，也要取消外部设置的 `PYTHONFAULTHANDLER`；通常没有必要为稳态推理速度关闭信号处理。

仍保留请求总耗时、快照单调时钟和服务端推理耗时，因为它们参与 RTC 延迟估计与请求超时；观测时间戳参与过期拒收。动作有限值/形状检查、前缀缓存、输出搬到 CPU 及插值所需复制也属于执行逻辑。RECAP 还需记录实际下发动作、来源和时间，其必要快照会保留，但 debug 关闭时不计算差分、反向或异常指标。

预热与启动校准只在连接阶段运行，其中的同步用于在允许正式执行前确认结果完成和有效，不会插入每个稳态请求。不要为了关闭诊断而取消必要预热；需要调整启动成本时，单独使用已有 `pi05_inference.warmup_runs` / `startup_calibration_runs` 设置。
