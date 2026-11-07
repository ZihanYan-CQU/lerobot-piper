# PI0.5 推理精度、编译与官方实现对齐

[排错索引](README.md) · [PI05 性能排查](pi05-performance.md) · [当前推理架构](../architecture/pi05-inference.md)

这是按原日期保留的诊断与修订记录；“当前”“本轮”等表述指相应记录当时，后续修订以文内日期为准。本次整理未重新执行历史测试，也未把待验证方向改为已完成结果。

2026-10-15 调研，2026-10-16 完成代码与本地验证。此次实现针对现有 LeRobot checkpoint 和 deployment/DAgger 入口，不迁移 checkpoint 格式，也不替换系统安装的 Transformers。

2026-10-20：按部署需求调整 `profile=checkpoint`，统一采用 checkpoint 配置中的 `dtype`；不再原样继承模块精度划分。`lerobot` / `openpi` 保持下列固定版本的精度方案，训练默认值不变。

2026-10-30：`lerobot` 档位的视觉推理同步到 [LeRobot `9069415`](https://github.com/huggingface/lerobot/blob/90694152570656b88b01683ff3af199647f56bde/src/lerobot/policies/pi05/modeling_pi05.py)。视觉塔和视觉投影的**参数仍为 FP32**；当模型为 BF16 配置、处于 `eval()` 且输入在 CUDA 上时，普通单帧视觉路径使用 BF16 autocast，输出恢复输入 dtype。同尺寸的多路相机合并为一次视觉编码，再按原相机顺序拆回；图像掩码仍按各相机处理。

CPU、FP32 配置和 MEM 视频路径不启用这项推理 autocast；训练继续逐相机处理，并保留调用方已有的 AMP 上下文。这次同步限于 `backend=lerobot`、`precision_profile=lerobot`，其他档位和默认部署选项不变。已有部署档位代码只需同步本次修改的 `modeling_pi05.py` 并重启服务端即可获得这两项更新。

本轮定向回归共 120 项通过，覆盖视觉计算/参数 dtype、训练反向传播、相机顺序与掩码、MEM、CPU 图捕获、CUDA Inductor 普通采样、guided RTC 图复用、TrainedRTC 前缀保持，以及部署档位加载。GPU 使用 RTX 4060。服务端预热测试保留双路 224×224 图像、200 个 token、50 步动作与两次工作线程会话，缩小模型宽度/层数；独立复测完整预热与重连约 45 秒通过。首次子进程测试曾超过 240 秒时限，因此测试预算改为与服务端默认值一致的 600 秒，并在超时时输出已捕获的进度日志；最后该项测试通过。尚未验证服务器完整 checkpoint 的 A100 性能或机器人任务成功率。

首次启用这些部署档位时，需一起同步 `src/lerobot/policies/pi05/configuration_pi05.py`、`src/lerobot/policies/pi05/modeling_pi05.py`、`src/lerobot/async_inference/server/pi05_inference.py` 和 `src/lerobot/async_inference/server/policy_server_deployment.py`，然后重启服务端。离线精度对照另需同步 `src/lerobot/scripts/lerobot_profile_pi05.py`。

## 官方做法并不完全相同

最初固定核对版本：LeRobot `8975047fbb76e7fef3f482988ceccc17e2102a90`；OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`。LeRobot 视觉推理在 2026-10-30 按上方版本更新。

| 项目 | LeRobot 普通 PI0.5 路径 | OpenPI PyTorch 推理路径 |
| --- | --- | --- |
| 精度 | 配置默认 FP32，支持 BF16；BF16 下视觉参数保留 FP32，CUDA 单帧推理使用 BF16 autocast | 推理加载转换为混合 BF16；视觉 patch/position embedding 和 Gemma 归一化等保留 FP32 |
| 编译 | `compile_model=false` 为默认，可选 `max-autotune` | `pytorch_compile_mode="max-autotune"` 为默认，编译动作采样 |
| FP32 matmul | 开启编译时设置 `high` | 初始化时设置 `high`，独立于是否编译 |
| 语言前缀/动作专家注意力 | 普通采样强制 eager | 普通采样同样强制 eager |
| SigLIP 注意力 | 由 Transformers 支持的后端决定 | 同样通过 Transformers 后端选择；不是强制整模型 FlashAttention |

精度依据：[LeRobot 模型](https://github.com/huggingface/lerobot/blob/8975047fbb76e7fef3f482988ceccc17e2102a90/src/lerobot/policies/pi05/modeling_pi05.py)、[OpenPI 混合精度规则](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/gemma_pytorch.py)、[OpenPI 推理加载](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/policy_config.py)。

编译与 attention 依据：[LeRobot 配置](https://github.com/huggingface/lerobot/blob/8975047fbb76e7fef3f482988ceccc17e2102a90/src/lerobot/policies/pi05/configuration_pi05.py)、[OpenPI 配置](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py)、[OpenPI sampler](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/pi0_pytorch.py)。

2026-10-15 核对时，上一次排查中的全视觉 FP32/eager 是性能候选，不应称为偏离当时 LeRobot 官方实现的错误。“两家官方都默认 BF16 + compile + FlashAttention”也不准确。这里对齐 OpenPI 的推理精度和编译默认，同时保留两家普通 PI0.5 使用的 eager Gemma 注意力。

## 已实现的选项

PI05 模型配置新增 `precision_profile`：

- `checkpoint`：所有模型浮点参数统一采用 `dtype`，包括视觉嵌入、归一化参数和动作/时间/历史状态投影。部署入口的同名 profile 会在加载前选择此方案。
- `lerobot`：模型配置默认；训练或直接加载旧 checkpoint 未包含该字段时继续使用原有参数精度划分。CUDA BF16 单帧推理采用上述 autocast 和相机合并。部署默认 `checkpoint` 会显式覆盖这一字段。
- `openpi`：BF16 时，视觉 Transformer 和投影层采用 BF16，仅视觉嵌入及 Gemma 归一化等保留 FP32。FP32 模型仍保持 FP32。

OpenPI 自带的 [SigLIP 补丁](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py)在视觉 embedding 后转换激活类型。这里在模型内部实现同样的边界，兼容本项目的 Transformers 5，避免只改权重导致 FP32 输入进入 BF16 Linear 的错误。前向过程不修改参数 dtype；MEM 视觉路径也在相同位置转换激活。

使用用户配置进行 meta-device 检查：OpenPI 档位下视觉塔有 411,468,912 个 BF16 参数、973,440 个 FP32 参数，视觉投影的 2,361,344 个参数全部为 BF16。原 LeRobot 档位的这两部分全部为 FP32。这证明实际 dtype 已改变，不代表已经测得相应的速度提升。

`lerobot` / `openpi` 中保留 FP32 的 AdaRMSNorm 线性映射与动作/时间投影维持原有规则。`checkpoint` 则将这些参数也转换到配置的 dtype，并在图像、状态、动作和时间投影边界匹配输入类型。归一化、softmax 等内部数值计算和 Euler/RTC 采样累计仍保留必要的 FP32；后处理接收的动作始终为 FP32。因此“全 BF16”指参数统一，不表示每个中间张量或 CUDA 累加器都使用 BF16。

BF16 转 FP32 无法恢复权重原有的细节，但 FP32 仍会影响后续激活计算和误差累积。另外，BF16 配置下保存的权重可能本来就包含官方保留的 FP32 模块，不能凭 `config.dtype` 推断文件内每个张量的精度。全 BF16 模式需要独立比较推理效果和延迟。

补齐了上游已有的 `text_tokenizer_name` 配置，支持用户 checkpoint 中的本地 tokenizer 路径。

服务器新增 `pi05_inference` 参数组，参数会在创建/加载模型前覆盖 checkpoint 的运行时配置；不改磁盘上的 checkpoint。参数参与模型缓存 key，切换精度/编译选项后会重新加载，避免复用旧运行配置。

2026-10-20 验证：160 项相关测试通过，包含统一 BF16/FP32 参数、BF16/FP32 权重文件交叉加载、普通采样、guided RTC 编译图复用、历史图像/状态前向与反向，以及部署缓存、DAgger/RECAP 和 trained RTC 回归。CUDA 验证使用本机 RTX 4060；新增全 BF16 预热测试保留双路 224×224 图像、200 个 token、50 步动作和重连流程，但缩小模型宽度/层数。它验证类型衔接和编译执行，不证明完整 checkpoint 在 A100 上更快或实机任务效果不变。

## A100 上如何启用

服务端现在提供三个显式选项；保留原有 host、port 和 RTC 参数，在启动命令末尾追加相应参数：

| `pi05_inference.profile` | 加载精度 | 默认编译行为 |
| --- | --- | --- |
| `checkpoint` | 按保存的 `dtype` 统一所有模型浮点参数：全 BF16 或全 FP32 | 沿用 checkpoint |
| `lerobot` | 强制 BF16；视觉塔和视觉投影参数保持 FP32，CUDA 单帧推理使用 BF16 autocast | 关闭，可显式开启 |
| `openpi` | 强制 BF16；视觉嵌入和指定 Gemma 归一化等保持 FP32 | 开启 `max-autotune` |

`lerobot` 是显式的 **LeRobot BF16 对照项**，不依赖 checkpoint 原先是 FP32 还是 BF16，也不继承其中的 OpenPI 精度设置。三个档位均支持独立覆盖 `compile_model` / `compile_mode`；`checkpoint` 不再与 BF16 下的 `lerobot` 等效。

```bash
# 统一采用 checkpoint 的 dtype，不编译。
--pi05_inference.profile=checkpoint --pi05_inference.compile_model=false

# 统一采用 checkpoint 的 dtype，显式指定与官方档位相同的编译模式。
--pi05_inference.profile=checkpoint --pi05_inference.compile_model=true --pi05_inference.compile_mode=max-autotune

# LeRobot BF16，不编译。
--pi05_inference.profile=lerobot --pi05_inference.compile_model=false

# LeRobot BF16，开启编译。
--pi05_inference.profile=lerobot --pi05_inference.compile_model=true

# OpenPI BF16，不编译。
--pi05_inference.profile=openpi --pi05_inference.compile_model=false

# OpenPI BF16，开启编译。
--pi05_inference.profile=openpi --pi05_inference.compile_model=true
```

保持默认编译模式时，两组开启编译的实验都会使用 `max-autotune` 和 `high` FP32 matmul；优先用这一对比较视觉精度划分。每组重新启动服务，使用相同 checkpoint、相机、观测与 RTC 设置，并剔除预热。

只写下面一个选项仍可启用 OpenPI 组合：

```bash
--pi05_inference.profile=openpi
```

这一个选项等价于运行时设置：

```text
dtype=bfloat16
precision_profile=openpi
compile_model=true
compile_mode=max-autotune
float32_matmul_precision=high
```

`policy_server_deployment`、继承它的 `policy_server_dagger`/`policy_server_recap` 使用相同参数。默认仍为 `profile=checkpoint`，统一参数 dtype 并继承保存的编译设置；默认选项不影响其他策略类型，非 PI05 策略不接受显式的 PI05 编译或官方 profile 覆盖参数。离线分析脚本的 `--precision-profile checkpoint` 选择同一参数精度路径。

编译模式可独立覆盖，例如：

```bash
--pi05_inference.profile=openpi \
--pi05_inference.compile_mode=reduce-overhead
```

若先单测 OpenPI 精度而暂不开编译：

```bash
--pi05_inference.profile=openpi \
--pi05_inference.compile_model=false
```

若显式选择 LeRobot BF16 精度划分并开启编译：

```bash
--pi05_inference.profile=lerobot \
--pi05_inference.compile_model=true
```

无编译时，OpenPI 会设置 `high`，LeRobot 沿用进程当前 FP32 matmul 设置；因此这两组的默认结果不是仅精度划分一个变量的对照。该设置是进程级的，不要在同一进程切换档位来测量它的默认值。下面离线实验用 `--matmul-precision high` 固定这个变量。

训练/普通策略入口仍可以直接设置 `policy.dtype`、`policy.precision_profile`、`policy.compile_model` 等模型配置；本次部署选项不改变已有训练默认值。

## 预热与 RTC

无论 eager 还是编译模型，服务器都在**正式推理工作线程**完成预热，握手等待预热成功后才返回。默认 `warmup_runs=3`，普通无前缀与代表性 RTC 前缀各三次；覆盖原始观测转换、预处理、模型、后处理及 CPU 输出。形状来自已加载 checkpoint 和客户端观测特征；trained RTC 使用已协商的固定 delay，guided 使用执行窗口内的代表 delay。

服务器预热使用合成传感器数据，不填充正式动作缓存，结束或失败时清理策略、processors 和调试跟踪，并恢复 RNG。随后 Pi05 客户端默认使用真实相机/状态、任务指令和已协商的图像传输方式，进行一次普通、一次 RTC 往返预热，再做三次 RTC 往返测量。只有后三次更新延迟估计；最后一次 RTT 通过完成消息反馈，确认临时状态清理完毕后，客户端从 `STARTING` 进入 `READY`。

校准结果由独立协议对象包裹，虽包含正常大小的动作数据以测量返回/反序列化成本，但不进入动作队列，也不写入 DAgger/RECAP 采集。RTT 起点在传感器读取后、图像编码/序列化前，终点在返回解码后；客户端补充预算还包含该次传感器读取时间。它是启动估计，不包含自主控制时的全部 CPU 竞争或邮箱等待，后续实际请求继续更新统计。

guided 的 P90 和 1 步 margin 保留，另加与客户端同样的快升慢降保护：真实慢样本立即抬高预算，下降按 0.2 系数平滑。服务器保护值还限制在当前 RTT 窗口最大值以内，旧尖峰退出窗口后不再持续抬高预算。普通 rollout reset 清动作/请求历史，保留仍有效的性能统计；默认连续 300 秒未更新则过期。重连或重新加载配置会清空旧统计。trained 的固定 delay 和迟到拒收规则不变。

普通无前缀采样仍整段编译；guided RTC 有前缀时，仅编译接收固定形状张量的模型 `denoise_step`，前缀构造、Python 时间/延迟控制和 VJP 留在编译区域之外。前缀长度和 delay 的变化不再进入去噪器的编译守卫，但相机/文本等实际张量形状变化仍可能产生新图。用 `TORCH_LOGS=recompiles,graph_breaks` 检查真实运行；必要时使用 `--pi05_inference.compile_model=false` 做对照。保留现有 KV clone，因为当前 Transformers cache 会更新，直接删除会改变去噪语义；保留 `no_grad` / 局部 `enable_grad`，不使用会阻断 guided RTC 的全局 `inference_mode`。

服务器 `--pi05_inference.warmup_runs=0` 关闭合成预热；客户端 `--startup_calibration_runs=0` 关闭真实往返预热/校准。两项都关闭时，首次实际请求承担冷启动成本。服务器 `--pi05_inference.warmup_timeout_s=600` 限制线程预热等待，客户端 `--startup_calibration_timeout_s=600` 限制每次校准等待；两端 `--latency_history_ttl_s=300` 控制各自统计的有效期。失败或超时阻止启动，不交付动作；单次已提交的 GPU 工作不能被 Python 强行取消。

部署协议已升级到 **5**，两端需一起更新并重启。启动日志区分 `PI05 warmup`、`RTC calibration warmup`、`RTC calibration measure` 和 `RTC calibration ready`。这些改动转移冷启动成本、改善初始预算，不代表降低了稳态模型 FLOPs；A100 实际收益仍应按以下四组对照测量。

## 2026-10-16 代码复查

本次复查定位并修复了以下问题：

- **滑动窗口之外的保护拖尾**：10 个 400 ms 样本之后出现一次 20 s 停顿，再收到 10 个 400 ms 样本，旧实现仍估计约 2.50 s；现在旧尖峰退出窗口后回到 400 ms。窗口内真实尖峰仍立即生效。
- **重连的接收竞争**：已有后台接收线程可能取走启动校准响应。校准现在独占接收通道，等待旧轮询完成后才重新握手；等待期间不持有 rollout 生命周期锁，停止仍可取消启动。
- **预热冗余分配**：提供完整预处理回调时，不再先创建一组马上被覆盖的 GPU 合成输入，也不再依赖未使用的合成 state schema。
- **加载失败被吞掉**：原 PI05 加载器可能在文件错误或权重不匹配时返回未正确加载的模型，且忽略具名的 revision/token/local_files_only 等参数。现在正确传递参数，加载失败直接报错。诊断脚本复用这条严格加载路径，删除重复的 key 修复/重映射实现；显式 `strict=False` 的部分加载语义保留。
- **设备覆盖过晚**：默认 checkpoint 档位也在模型构造前应用部署设备，避免先分配到训练配置保存的设备再搬运。

服务器合成预热用于正式线程的算子/编译与处理器初始化，客户端真实往返预热/测量覆盖实际输入和传输，两阶段仍分别保留。KV 缓存复制和 guided RTC 的局部梯度路径也保留：它们承担数据所有权与算法语义，不能仅按重复调用删除。

## 2026-10-16 guided RTC 编译回退修复

本节记录当天较早的修复阶段。后续已补齐前缀编译，并将默认轻量引导改为完整采样编译；当前实现与重连验证限制见 [PI05 部署开销复查](pi05-performance.md#21-bf16--openpi--guided-rtc-的部署开销复查)。

A100 日志显示普通预热第三次为 58.9 ms，但首次 RTC 为 307911.6 ms，后两次仍为 4718.4 / 4495.0 ms。真实观测 RTC 校准为 6511.4 / 4680.5 / 4598.4 ms；这不是只有第一次编译慢。日志还明确报告 `RTCProcessor.denoise_step` 的时间值守卫触发 `recompile_limit (8)`。PyTorch 达到该上限后会对相应帧回退 eager，见 [torch.compile 官方说明](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html)。

根因是把原本针对普通推理的整段采样编译扩展到了含动态 Python 控制和局部 autograd 的 guided RTC。小模型 CPU 对照复现了时间标量守卫及 8 次上限，累计编译图从普通路径的 1 个增加到首个 RTC 后的 14 个，改变前缀后增加到 16 个；同时出现 `requires_grad_` / `autograd.grad` 图中断。

修复内容：

- PI05 在公开采样入口选择编译边界：普通/训练式路径维持原有整段编译；有前缀的 guided 路径使用 eager 采样循环和单独编译的模型去噪器。视觉/语言前缀预填充在该 guided 路径仍为 eager；这是避开不稳定图的实现取舍，不承诺 RTC 达到普通推理的 59 ms。
- 此阶段部署默认轻量引导仍使用原始 identity-VJP 校正、前缀权重和引导强度，只给模型速度预测包上 `no_grad`，避免在不使用速度 Jacobian 时记录模型参数梯度图。完整 VJP 当时作为可选扩展保留；2026-10-16 后续按部署需求移除了该实现及开关。当前编译覆盖、引导路径和验证限制以 [PI05 部署开销复查](pi05-performance.md#21-bf16--openpi--guided-rtc-的部署开销复查)为准。
- 每轮合成预热开始前打印路径、轮次、前缀长度和 delay；握手等待期间每 30 秒报告等待时间。日志仅表示仍在等待，不假定编译器已完成多少。

2026-10-16 编译修复需要同步至服务端的运行文件为 `policies/pi05/modeling_pi05.py`、`async_inference/server/policy_server_deployment.py`、`async_inference/server/pi05_inference.py`（均位于 `src/lerobot/` 下）；2026-10-20 参数精度更新还需要文首列出的配置文件。在此前协议 5 两端代码已一致的前提下，该编译修复不修改客户端协议。更新后重新启动服务端并重新连接客户端，清除旧编译对象和旧的 6.6 s 校准预算。尚未更新时，可用 `--pi05_inference.profile=openpi --pi05_inference.compile_model=false` 保留 BF16 并关闭编译作对照；延长超时不能修复稳态 RTC 开销。

### 后续 A100 CUDA 非法内存访问：尚未确定首个失败算子

15:11 的后续日志显示，RTC 预热调用单独编译的去噪器时出现 `CUDA error: an illegal memory access was encountered`。Dynamo 的 RNG 恢复和外层 `fork_rng` 的退出均报错。CUDA 异步错误可能在后续 API 才被报告，因此这些恢复位置不足以证明随机数状态是根因。**前面的重编译修复不等于已经解决这次 A100 内核崩溃。**

用户随后反馈重新运行部署后错误消失。目前记录为一次未稳定复现的 CUDA 异常，保留能正常运行的配置，继续核对 RTC 稳态耗时；不据此认定根因已消除，也不强制改变编译模式。

如果再次出现非法地址访问，先完整退出并重启服务端进程，不能在受影响的进程内继续尝试模型推理。恢复运行可保留 OpenPI BF16 并关闭编译；要定位编译问题，在新进程中先比较 `compile_mode=default`，再比较 `max-autotune-no-cudagraphs`。后者保留自动调优、关闭推理图回放；这些是区分原因的实验，不是已证实有效的修复。定位时可在启动前设置 `CUDA_LAUNCH_BLOCKING=1`，正式测速需移除这个调试设置。依据：[CUDA 错误状态](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__TYPES.html)、[PyTorch 异步执行](https://docs.pytorch.org/docs/2.11/notes/cuda.html#asynchronous-execution)。

本地增加了两类覆盖：使用后台工作线程、模型内部随机噪声、双 224px 图像、200 个文本 token、50 个动作的实际预热流程；以及在临时实验中保留 2048/1024 宽度、缩减到一层并强制启用 Triton GEMM 候选。两类对照暂未复现非法访问。RTX 4060 默认由于 SM 数量不足会跳过部分 GEMM 模板自动调优，测试环境不能等同于 A100；强制启用候选也不能模拟 A100 的硬件资源与内核选择。

多模式测试在同一进程切换工作线程时，还遇到 PyTorch 2.11 CUDAGraph Trees 的线程局部状态断言。这与用户此次非法访问堆栈不同，不能混为同一根因。模式对照按独立服务进程运行；后续若同进程重连触发该线程状态断言，应单独排查或对照关闭 CUDA Graph。

`pi05_inference.py` 现在在预热开始时记录 PyTorch/CUDA/Triton 版本、GPU 型号、计算能力和 SM 数量，并在 RNG 清理之前记录原始异常及所在预热阶段。该改动改善诊断信息，不改变模型算法或编译模式。

本轮预热配置/异常日志测试 11 项通过；三个独立进程中的 CUDA 后台预热回归（`default`、`max-autotune`、`max-autotune-no-cudagraphs`）均通过，包含 RNG 恢复、随机采样推进和有限输出检查。首个异常日志在模拟 RNG 恢复再次失败时仍可保留。

## 离线比较

在 A100 上使用服务端相同的环境，下面的 checkpoint 应是实际微调部署目录：

```bash
PI05_CHECKPOINT='/实际部署的checkpoint/pretrained_model'

# 四组统一 high，只改变视觉精度划分与是否编译。
# LeRobot BF16，无编译。
uv run --no-sync python -m lerobot.scripts.lerobot_profile_pi05 "$PI05_CHECKPOINT" \
  --dtype bfloat16 --precision-profile lerobot --compile-mode eager \
  --matmul-precision high \
  --warmup 10 --runs 50 --output /tmp/pi05_lerobot.json

# LeRobot BF16，编译。
TORCH_LOGS=recompiles,graph_breaks uv run --no-sync python -m lerobot.scripts.lerobot_profile_pi05 \
  "$PI05_CHECKPOINT" --dtype bfloat16 --precision-profile lerobot --compile-mode max-autotune \
  --matmul-precision high \
  --warmup 10 --runs 50 --output /tmp/pi05_lerobot_compile.json

# OpenPI BF16，无编译。
uv run --no-sync python -m lerobot.scripts.lerobot_profile_pi05 "$PI05_CHECKPOINT" \
  --dtype bfloat16 --precision-profile openpi --compile-mode eager \
  --matmul-precision high \
  --warmup 10 --runs 50 --output /tmp/pi05_openpi_eager.json

# OpenPI BF16，编译。
TORCH_LOGS=recompiles,graph_breaks uv run --no-sync python -m lerobot.scripts.lerobot_profile_pi05 \
  "$PI05_CHECKPOINT" --dtype bfloat16 --precision-profile openpi --compile-mode max-autotune \
  --matmul-precision high \
  --warmup 10 --runs 50 --output /tmp/pi05_openpi_compile.json
```

Conda 环境无 uv 时，在原服务端环境用 `python` 替换 `uv run --no-sync python`。按顺序运行四个实验，不要同时争抢 GPU。若还要测原来的 `highest` 设置，单独将第一组的 `--matmul-precision high` 改为 `highest`。查看报告的 `effective_policy`、实际参数 dtype、P50/P95、`fixed_input_actions` 和源码路径；可用相同固定输入的动作输出比较数值差异，脚本本身不自动计算跨报告的误差。合成输入比较计算成本；真实精度变化仍需用已有观测/任务验证。

## 本地验证结果与边界

- 2026-10-16 guided RTC 编译修复后：部署、权重加载及 padding loss 回归 **449 项通过**；精度与编译回归另 **14 项通过**，包含 RTX 4060 上普通采样、轻量 guided RTC、完整 VJP 的 CUDA/Inductor `max-autotune`。10 个去噪时间值和 7 次请求的前缀/延迟切换总共保持两个 Dynamo 图，固定噪声输出与 eager 在测试容差内一致；轻量校正的 4 种 schedule 与原实现逐值一致。此前的 CUDA 编译验证仅覆盖普通无前缀路径，不能证明 guided RTC 已兼容。
- 2026-10-16 代码复查修复后：完整异步部署套件及 PI05 权重加载、精度、记忆、训练时 RTC 回归 **462 项通过**；包含真实 gRPC、后台接收线程重连，以及缩小模型的 CUDA/Inductor 编译。8 个本轮修改的 Python 文件通过 Ruff 检查与格式检查。新回归在修复前复现了加载失败被吞掉、具名参数丢失、保护拖尾、重连竞争、重复预热输入依赖和设备覆盖遗漏。
- 部署与相关策略回归：439 项通过。覆盖异步请求、模型复用、RTC、DAgger/RECAP、视觉记忆和本次新增配置逻辑。
- 上一轮精度/预热定向测试：17 项通过，包含 CPU BF16 前向/反向、参数类型稳定性、完整缩小版采样器的 Dynamo 图捕获，以及 RTX 4060 上 `max-autotune` / Inductor 编译后重复采样与 eager 输出一致性。这组与前一组存在重叠，不应直接相加。
- 本轮新增显式 LeRobot BF16 选项和删除原始图像重复复制后：部署/配置 30 项、图像/观测处理 33 项通过；1 项 CUDA 图像测试因沙箱内不可见 GPU 跳过。覆盖两档位的配置覆盖、编译选项优先级、切换档位后的缓存失效与 checkpoint 不被改写；四组 CLI 参数也已验证。
- 完整预热/校准改动的部署套件：411 项通过。随后补充真实 gRPC（guided/trained、图像协商一致性）与缩小模型 CUDA 编译验证，46 项通过；最后增加停止与启动并发的回归后，启动/服务端/客户端/RECAP 定向套件 88 项通过。各次测试存在重叠，不能将数量直接相加。覆盖预热失败/超时、同线程执行、校准不下发动作、重连清统计、复位保留统计、旧消息拒绝及 TTL 过期。
- 全尺寸的原有编译测试在加载默认 FP32 模型时因 RTX 4060 的 8 GB 显存不足而退出，尚未进入编译；没有修改该测试来绕过限制。
- 服务端嵌套 CLI 参数、旧 checkpoint 默认值、运行时覆盖优先级、严格加载诊断脚本与代码格式检查已验证。

初次排查时沙箱内无法访问 CUDA；后续在沙箱外确认本机为 RTX 4060 并完成缩小模型的 GPU 测试。尚未连接用户的 A100，也没有完整微调 checkpoint。**A100 上完整模型的延迟、编译稳定性和真实任务效果仍需运行上述对照测试确认。**

日志中持续升高的延迟、预处理、冷启动和 RTC 预算的进一步分析见 [延迟排查第 8 节](pi05_inference_latency_2026-10-15.md#8-2026-10-16bf16-选项之外日志揭示的优化顺序)。
