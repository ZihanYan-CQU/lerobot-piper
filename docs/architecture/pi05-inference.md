# PI05 推理架构：图像输入、精度与预热

[架构索引](README.md) · [部署指令](../workflows/deployment/README.md) · [性能排查](../troubleshooting/pi05-performance.md)

本文解释远端 deployment、DAgger、RECAP 共用的 PI05 输入/精度配置。完整启动配方在流程文档维护。部署档位含 2026-10-20 调整与 2026-10-30 视觉推理更新；历史测量、未完成验证和优化建议见排错记录。单机输入路径见 [本地推理架构](local-inference.md)。

<a id="15-π05-图像上传float32-无损压缩--openpi-uint8-可选"></a>

## π0.5 图像上传：float32 无损压缩 / OpenPI uint8 可选

OpenPI 的[远程推理文档](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)明确建议客户端提前缩放图像以减少带宽和延迟，示例使用 224×224；其 [PyTorch 预处理](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/preprocessing_pytorch.py)与 LeRobot 的 [π0.5 配置](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi05/configuration_pi05.py)也默认采用 224×224。这是模型已有的输入尺寸，不是将模型输入继续降分辨率。

Deployment、DAgger、RECAP 客户端共用该选项，guided、trained 和启动延迟校准都适用。默认保持原有 float32 流程，在客户端命令后添加以下参数即可显式选择：

```bash
--resize_images_on_client=true --image_transport_mode=lerobot_float32
```

选择 OpenPI 的图像处理和 uint8 上传：

```bash
--resize_images_on_client=true --image_transport_mode=openpi_uint8
```

| 模式 | 客户端处理 | 上传图像格式 | 服务端处理 |
| --- | --- | --- | --- |
| `lerobot_float32`（默认） | 复现原有两阶段缩放：feature 尺寸插值，再执行 π0.5 等比例缩放和居中补边 | CHW float32 `[0,1]`，字节重排 + zlib 无损压缩；不划算时原图上传 | 解压恢复 float32，跳过已完成的缩放 |
| `openpi_uint8` | 从原始 RGB 图像直接使用 PIL 双线性等比例缩放和居中黑色补边，匹配 `openpi_client.image_tools.resize_with_pad` | HWC RGB uint8 原始字节，不做 zlib / JPEG 编码 | 转为 CHW float32 `[0,1]`，跳过已完成的缩放 |

两种模式都在本地完成图像缩放。客户端不访问服务器上的 checkpoint 文件，也不硬编码 224；服务器加载 checkpoint 后协商实际 `image_features` 和 `image_resolution`。224 模型使用 224×224，384 模型使用 384×384。两张 224×224 RGB uint8 图像的像素载荷固定为 301,056 B，完整请求还包含状态、RTC 前缀和序列化元数据。

`openpi_uint8` 会改变缩放算法及像素数值，**不与原 float32 预处理逐位等价**；模型输入最终仍为浮点张量，不是 INT8 模型推理。float32 模式的“无损”指保留既有缩放后的 float32 像素。相机采集、录制和 DAgger / RECAP 使用的原始图像保持原尺寸；只有后台上传用的字典副本改变。

两端需同步更新 `shared/` 和各自的 `client/`、`server/` 代码并重启；服务端无需新增命令行参数。启动日志应出现 `Client image resize enabled (lossless float32)` 或 `Client image resize enabled (OpenPI uint8)`。开启 `--log_rtc_timing=true` 或 `--debug_rtc_diagnostics=true` 后，每次发送输出：

```text
RTC IMAGE UPLOAD obs=... raw_image_bytes=... image_bytes=... payload_bytes=... prepare_ms=...
```

`payload_bytes` 是完整序列化请求大小；`image_bytes` 是图像载荷合计，不含对象元数据；`prepare_ms` 包括客户端图像处理、所选模式的编码与请求序列化。可结合 `RTC RPC PHASES` 对照两种模式的本地耗时和上传时间。

默认 float32 模式遇到旧服务器未返回能力信息、非 π0.5 策略、视觉 memory、自定义图像处理、视觉归一化或提前低精度转换等无法确认等价的处理链时，保持原图上传。显式选择 `openpi_uint8` 时，若策略或处理链不支持，或服务端未确认该模式，则启动报错，避免参数被静默忽略。uint8 模式始终发送处理后的像素，即使输入小于模型尺寸。

需要与原流程对照时，仅在客户端关闭：

```bash
--resize_images_on_client=false
```

`resize_images_on_client=false` 优先于 `image_transport_mode`，此时不启用任何一种本地处理。

<a id="18-lerobot--openpi-bf16-与编译部署选项"></a>

## LeRobot / OpenPI BF16 与编译部署选项

`--pi05_inference.profile` 控制模型参数的精度方案，`--pi05_inference.compile_model` 独立控制是否使用 `torch.compile` 加速。下面的参数均加在**服务端启动命令**后。

| `profile` | 模型浮点参数的精度 | 不额外指定时的编译行为 |
| --- | --- | --- |
| `checkpoint`（默认） | 按 `config.json` 的 `dtype` 统一：`bfloat16` → 全部 BF16；`float32` → 全部 FP32 | 沿用 checkpoint 的 `compile_model` / `compile_mode` |
| `lerobot` | 强制使用 LeRobot BF16 方案：视觉塔、视觉投影、Gemma 归一化和动作/时间投影等保留 FP32 | 关闭；可用 `compile_model=true` 开启 |
| `openpi` | 强制使用 OpenPI BF16 方案：视觉编码器和视觉投影也用 BF16，视觉嵌入、Gemma 归一化和动作/时间投影等保留 FP32 | 开启，模式为 `max-autotune` |

**2026-10-30 同步官方 LeRobot 视觉推理：** `backend=lerobot` 下，`lerobot` 档位保留 FP32 视觉参数，在 CUDA 单帧推理时使用 BF16 autocast，并合并同尺寸多路相机进行编码。输出恢复图像输入的 dtype，相机顺序与掩码保持对应。训练、CPU、FP32 配置和 MEM 视频不额外启用这项推理 autocast；训练仍逐相机处理。已有服务端只需更新 [modeling_pi05.py](../../src/lerobot/policies/pi05/modeling_pi05.py) 并重启；开启编译仍需显式传入 `--pi05_inference.compile_model=true`。

**2026-10-20 起，`checkpoint` 的含义调整为“统一采用保存的 dtype”**，会覆盖运行时的 `precision_profile`，不再保留旧 checkpoint 的模块混合精度划分。它不会改动磁盘上的配置或权重；训练入口的默认精度划分仍为 `lerobot`。例如，你的 checkpoint 保存了 `"dtype": "bfloat16"`，使用默认部署选项时，视觉嵌入、归一化参数和动作/时间投影也会转成 BF16。若要保持原来的 LeRobot BF16 混合方案，请显式选择 `lerobot`；两者现在不再等效。

这里的“全部 BF16”指所有模型浮点**参数**。归一化、softmax、位置编码等内部数值计算仍可使用 FP32，Euler/RTC 采样累计和返回给后处理器的动作也保留 FP32。BF16 参数转为 FP32 无法恢复保存时丢失的精度，但后续激活计算和数值稳定性仍会受精度影响；官方保留部分 FP32 并非没有作用。`config.json` 写 BF16 也不代表权重文件中的每个张量原本都是 BF16。

OpenPI 档位还设置 `high` FP32 matmul；两种官方精度方案开启编译时默认使用 `max-autotune`。对齐所用的固定官方版本见 [精度与编译对齐记录](../troubleshooting/pi05_official_alignment_2026-10-15.md)；普通 PI0.5 Gemma 注意力保持 eager，本选项不强制切换 FlashAttention。

对照时可以分别使用：

```bash
--pi05_inference.profile=checkpoint --pi05_inference.compile_model=false
--pi05_inference.profile=checkpoint --pi05_inference.compile_model=true --pi05_inference.compile_mode=max-autotune
--pi05_inference.profile=lerobot --pi05_inference.compile_model=false
--pi05_inference.profile=lerobot --pi05_inference.compile_model=true
--pi05_inference.profile=openpi --pi05_inference.compile_model=false
--pi05_inference.profile=openpi --pi05_inference.compile_model=true
```

每行是独立一组服务器启动参数。`compile_model=false` 表示关闭编译，仍可在 GPU 上进行 BF16 推理；`true` 会增加首次预热时间。比较精度方案时统一编译模式、观测输入和 RTC 参数，并比较预热后的延迟及任务效果。离线分析脚本可用 `--precision-profile checkpoint` 选择统一参数精度，用 `--matmul-precision high` 统一 FP32 matmul 设置。

尚未同步 2026-10-20 部署档位代码的服务端，需要同时更新以下四个文件，然后重启服务端：

- [configuration_pi05.py](../../src/lerobot/policies/pi05/configuration_pi05.py)
- [modeling_pi05.py](../../src/lerobot/policies/pi05/modeling_pi05.py)
- [pi05_inference.py](../../src/lerobot/async_inference/server/pi05_inference.py)
- [policy_server_deployment.py](../../src/lerobot/async_inference/server/policy_server_deployment.py)

使用离线分析工具时，再同步 [lerobot_profile_pi05.py](../../src/lerobot/scripts/lerobot_profile_pi05.py)。

eager 和编译模型都在正式推理线程预热，普通/RTC 代表形状默认各三次，包括预处理、后处理和 CPU 输出；握手等待其成功。随后 Pi05 客户端在 `STARTING` 状态完成两次真实往返预热（普通/RTC 各一次）和默认三次 RTC 测量，清理临时状态并确认最后一次 RTT 已反馈后进入 `READY`。校准结果经过独立协议包装，不进入动作队列或录制数据。

配置项可直接附加到现有启动命令：

| 位置 | 参数 | 默认行为 |
| --- | --- | --- |
| 服务端 | `--pi05_inference.warmup_runs=3` | 普通和 RTC 合成预热各三次；0 关闭 |
| 服务端 | `--pi05_inference.warmup_timeout_s=600` | 等待工作线程预热的上限 |
| 客户端 | `--startup_calibration_runs=3` | 两次真实预热之后测量三次往返；0 关闭整个往返校准 |
| 客户端 | `--startup_calibration_timeout_s=600` | 单次校准等待上限 |
| 两端 | `--latency_history_ttl_s=300` | 跨 rollout 保留统计，连续空闲 300 秒后失效 |

日志依次出现 `PI05 warmup`、`RTC calibration warmup`、`RTC calibration measure`、`RTC calibration ready`。每轮合成预热开始前会打印路径、轮次、prefix 和 delay；等待期间每 30 秒报告经过的时间。首次编译可能耗时较长；失败或超时阻止启动，需要解决错误后重新连接。非法内存访问等致命 CUDA 错误必须先重启服务端进程，见[重连时的段错误与 CUDA 非法内存访问](../troubleshooting/deployment.md#19-重连时的段错误与-cuda-非法内存访问)。已开始的单次 GPU 工作只能等待返回，服务器在旧线程退出前不会启动第二个推理线程。

同一客户端重连时，校准独占动作响应通道，后台接收线程在校准完成后恢复。所有 PI05 档位都在构造模型前应用部署设备，避免先按训练时保存的设备分配显存。权重文件缺失、损坏或严格加载失败会使启动报错；不会继续使用未正确加载的模型。

校准覆盖真实图像编码/传输、预/后处理和结果反序列化；自主控制期间的负载变化仍靠实际请求持续更新预算。guided 的动态 delay/prefix 已移出编译区域；实际模型输入形状变化仍可能产生新图，真实慢请求不会被当作预热丢弃。轻量引导跳过不需要的模型梯度图。trained 的固定 delay 和拒收条件不变。`--pi05_inference.compile_model=false` 可单独关闭编译，`--pi05_inference.compile_mode=reduce-overhead` 可比较其他编译模式。

官方固定版本、精度转换边界、完整命令和验证范围见 [PI0.5 官方对齐说明](../troubleshooting/pi05_official_alignment_2026-10-15.md)。

## 轻量 guided 的编译边界

普通无动作前缀、trained 和默认轻量 guided 路径编译完整采样函数。轻量 guided 在图外准备固定形状的动作前缀和权重，在图内执行图像/语言预填充、去噪器、引导修正与 Euler 更新；每个请求重新计算当前观测的 KV，不复用上一观测的数据。

`rtc_fast_guidance=true` 使用轻量 identity-VJP 的直接计算，`false` 是原轻量参考路径。两者均不对去噪器求导；不能将其等同论文完整 VJP。参考/诊断路径保留分段编译。算法差异、数值与编译验证及重连限制见 [2026-10-16 开销复查](../troubleshooting/pi05-performance.md#21-bf16--openpi--guided-rtc-的部署开销复查)。
