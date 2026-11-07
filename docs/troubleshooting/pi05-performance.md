# PI05 性能排查：延迟试验与优化记录

[排错索引](README.md) · [PI05 推理架构](../architecture/pi05-inference.md) · [部署指令](../workflows/deployment/README.md)

本页集中保存原 Deployment README 中的云端延迟试验、优化候选与 2026-10-15/16 的本地验证记录。日期、设备、缩小模型和未验证边界均保留；没有 A100 实测收益的建议仍是待验证方向。本次文档整理没有重跑这些实验。

更早的日志分析见 [2026-10-15 延迟排查](pi05_inference_latency_2026-10-15.md)，固定官方版本与精度修订见 [官方对齐记录](pi05_official_alignment_2026-10-15.md)，崩溃复现见 [部署排错](deployment.md)。

<a id="14-云端延迟抖动的试验设置"></a>

## 云端延迟抖动的试验设置

保持 `actions_per_chunk=50`、`fps=30` 及原有 `rtc.execution_horizon`，先比较本次调度和延迟统计改动。新版本默认立即允许下一次采集、使用 P90 + 1 步的 guided 估计；请求时序按需通过 `--log_rtc_timing=true` 开启。合格结果仍在收到时合并，不等待某个固定接管步；没有为掩盖抖动扩大 chunk 或延长旧轨迹执行。

按现有启动命令追加或显式核对：

```bash
# guided 服务器
--rtc_delay_buffer_size=10 --rtc_delay_quantile=0.9 --rtc_delay_margin_steps=1

# guided 客户端
--adaptive_rtt_trigger=true --log_rtc_timing=true

# trained 客户端（保持已经验证过的 trained.delay_steps 预算）
--trained.replan_on_result=true --log_rtc_timing=true
```

需要试验 guided 迟到接纳时，再配置独立的逐特征容差。下面是**Piper 小容差起点示例**，不是通用安全阈值；应按动作尺度、速度和实际衔接误差核对。键名必须与客户端启动时打印的 `RTC diagnostic action order` 一致：

```bash
--guided_late_action_tolerances='{"joint_1.pos":0.02,"joint_2.pos":0.02,"joint_3.pos":0.02,"joint_4.pos":0.02,"joint_5.pos":0.02,"joint_6.pos":0.02,"gripper.pos":0.001}'
```

对照时同时比较服务器耗时、完整请求 RTT 的 P50/P90/P99、迟到块接纳/拒收、`RTC STARVATION` 时长，以及相同任务的成功率和回退。新日志用于确定优化下一步应放在模型推理、排队还是传输。这里的修改改善重规划调度与抖动处理，并未降低 π0.5 本身的计算成本；单请求模式的新观测闭环速度仍受实际端到端延迟限制。trained 的固定预算不足仍会拒收，不能用 guided 容差绕过。

<a id="图像传输基准2026-10-15"></a>

## 图像传输基准（2026-10-15）

2026-10-15 原有 float32 模式的本地测量：构造两路 640×480 RGB 请求，2 次预热后测量 10 次，CPU 使用 16 个 PyTorch 线程。下表是测试数据，不是实际云端链路的 RTT，也不代表新增 uint8 模式：

| 图像来源 | 原完整请求 | 新完整请求 | 减少 | 客户端处理/序列化中位数 | 解码及观测准备中位数 |
| --- | --- | --- | --- | --- | --- |
| 仓库相机示例图调整到 640×480 | 1,843,809 B | 508,972 B | 72.4% | 8.9 ms | 3.2 ms |
| 随机噪声图 | 1,843,809 B | 738,482 B | 60.0% | 12.4 ms | 3.4 ms |

float32 回归直接调用生产代码 `PI05Policy._preprocess_images()`，检查原有 CPU 视觉输入逐位一致，并保留 CUDA 绝对误差 `2e-6` 的可选测试。uint8 回归使用 OpenPI 实现生成的像素校验值，覆盖横/竖图、奇数边长、已是目标尺寸、224/384 和非方形模型尺寸；两种模式均覆盖 guided/trained 请求及实际 gRPC 启动校准。实际 A100 云端 RTT 和机械臂任务效果仍需部署后比较。

<a id="17-π05-推理加速与延迟稳定后续优化建议"></a>

## π0.5 推理加速与延迟稳定：后续优化建议

以下是后续优化的验证方向，尚未在 A100 上测量收益。Pi05 配置已有 `compile_model`（默认 `false`）和 `compile_mode`（默认 `max-autotune`）；普通采样和默认轻量 guided RTC 整段编译，训练 `forward` 仍可编译。现在已增加部署专用的 `--pi05_inference.profile=openpi`、编译覆盖参数和连接阶段预热，详见[LeRobot / OpenPI BF16 与编译部署选项](../architecture/pi05-inference.md#18-lerobot--openpi-bf16-与编译部署选项)；服务器参数不是裸的 `--compile_model=true`。

`torch.compile` 可以减少 Python 调度和算子开销，但首次编译、输入变化触发的重编译可能造成延迟尖峰。2026-10-16 较早版本的 A100 日志中，guided RTC 的 Python 时间值变化曾触发 8 次重编译上限，预热后每次 RTC 仍耗时约 4.5 秒。当前默认轻量路径先在编译区外准备固定形状的动作前缀及延迟权重，再编译完整采样循环；时间序列在图内展开，不把逐步变化的 Python 时间作为编译函数入参。参考和调试路径保留独立编译的前缀预填充与去噪器。普通与有前缀路径仍分别预热。用相同观测序列检查 `TORCH_LOGS=recompiles,graph_breaks`，不要通过提高重编译上限或延长超时处理稳态延迟。PyTorch 对 [编译守卫与回退](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html)有明确说明。

建议按以下顺序验证：

| 优先方向 | 对当前项目的具体做法 | 需要验证的边界 |
| --- | --- | --- |
| 编译与启动前预热 | 对比 eager、`reduce-overhead`、`max-autotune`；预热真实图像/文本形状、无前缀及有前缀路径，完成后才允许 rollout | 预热结果不发给机械臂、不写入正式动作缓存；报告首次耗时与稳态 P50/P95/P99、最大值、重编译次数 |
| 稳定输入形状 | 固定 batch=1、相机数量、模型原生图像尺寸、文本 padding 与张量布局；RTC 前缀使用固定容量缓冲及有效 mask | 当前 delay 数值仍随观测/执行进度变化，不为减少编译把条件延迟或执行区间拉长 |
| A100 数值精度 | 检查实际 checkpoint dtype；本仓库默认是 float32，支持 bfloat16，可评估 BF16/TF32 | 精度变化需要比较动作误差和任务效果；不能仅凭默认配置判断当前服务器是否已使用 BF16 |
| Attention 内核 | 当前 Pi05 有显式 eager attention 调用，可评估 SDPA 等融合实现 | 不是简单改一个配置就生效；需核对 mask、AdaRMS、缓存，以及 guided RTC 所需梯度 |
| 减少系统干扰 | 推理 GPU 避免同时训练/跑其他模型；检查 CPU 线程竞争、GPU 时钟/温度/功率限制，关闭高频 debug/可视化 | 用 [NVIDIA-SMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html)记录事件和资源数据，确认尖峰来源，不能假设所有云端波动都来自模型 |
| 最后评估减少采样步数 | 保持 50 步预测长度，单独实验降低 `num_inference_steps` | 采样步数与预测/执行长度不同；会影响动作质量，需要任务回归，不直接作为默认改动 |

`reduce-overhead` 会尝试使用 CUDA Graphs 降低 Python/内核调度开销，适合小 batch，但要求计算路径兼容；`max-autotune` 会进行算子选择并在支持时使用 CUDA Graphs，首次编译成本更高。二者需在实际模型上比较，不能把模式名当作性能保证。参见 [PyTorch compile 参数](https://docs.pytorch.org/docs/stable/generated/torch.compile)与[性能调优指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)。轻量参考路径仍使用 autograd 求 identity 校正，不能把整条共享推理路径直接套上 `torch.inference_mode()`；默认轻量引导的等价优化见[BF16 / OpenPI / guided RTC 的部署开销复查](pi05-performance.md#21-bf16--openpi--guided-rtc-的部署开销复查)。

基准同时记录客户端完整 RTT 和服务端处理耗时；分析 GPU 子阶段时使用 CUDA Event 或明确的同步边界，普通 CPU 计时可能只统计到 kernel 提交。现有服务端处理时间包含后处理与输出交付前的 CPU 拷贝，不能把其中的 `predict_ms` 直接等同于纯 GPU 耗时。参见 [CUDA 异步执行与计时](https://docs.pytorch.org/docs/2.7/notes/cuda.html#asynchronous-execution)。这些优化以缩短一次计算、尽早使用新观测为目标，保持长预测、短执行的调度原则。

<a id="21-bf16--openpi--guided-rtc-的部署开销复查"></a>

## BF16 / OpenPI / guided RTC 的部署开销复查

`PI05 warmup ... ms` 从预处理前计时到预测、后处理及 CUDA 完成同步之后。`RTC PHASE` 位于外层，warmup 的开始/结束 INFO 打印也不在这段计时内。因此关日志通常不会让这个数值明显下降。首次 ordinary / RTC 的几十秒或几秒可能包含编译和自动调优；后续约百毫秒主要是模型和数据处理，不能把首轮与稳态平均比较。

2026-10-16 的优化范围集中在本地 deployment / guided RTC 扩展引入的开销。已撤回上一轮对 `flow_matching.py` 的时间张量生成修改，以及 `processor_pi05.py` 的状态复制修改；两者恢复到该轮优化前的版本，保留此前的其他功能改动。

当前保留的优化不改变去噪步数、精度及所选引导公式：

| 项目 | 当前处理 | 影响范围 |
| --- | --- | --- |
| guided 完整采样编译 | 默认轻量路径将 `_prefill_prefix`、全部去噪步及引导修正放进同一编译入口；参考、调试路径分别编译前缀和去噪器 | 补齐前缀覆盖，并减少每个去噪步往返 Python 的调度及小算子启动开销；每个动作块仍处理当前观测 |
| 轻量 RTC 的 identity VJP | 直接计算修正量，省掉每个去噪步的 autograd 遍历；保留输入梯度 dtype 的舍入语义 | 有前缀的轻量 guided 路径，包括 RTC warmup |
| 前缀权重和引导系数 | 复用不变的 mask 和标量系数；mask 缓存只有一个条目，系数最多 128 个 CPU 标量。前缀动作、噪声和模型输出不缓存 | 避免每步重复构造和传输常量 |
| 本地轻量引导中的前缀补齐 | 默认编译路径每个请求在目标设备补齐一次，并把 delay / horizon 转换成固定形状权重张量 | 省掉去噪循环内重复补齐；延迟数值和有效前缀长度变化不再成为完整采样图的 Python 编译守卫 |
| 最短请求耗时 | deployment / DAgger / RECAP 的 `inference_latency` 默认改为 0；显式正值仍可模拟延迟 | 取消继承自旧服务端的约 33 ms 人为下限；原本超过下限的请求不受影响 |

编译直接跟随已有 `compile_model` / `compile_mode`，不增加独立参数。`--pi05_inference.profile=openpi` 默认开启编译；显式设置 `--pi05_inference.compile_model=false` 则整条推理路径使用 eager。普通无动作前缀、trained 及默认轻量 guided 路径都编译整个采样函数。默认 guided 在图外准备动作前缀和权重，保留原生 CPU 标量系数的精度与 BF16 舍入语义；图内覆盖图像/语言预填充、去噪器、引导修正及 Euler 更新。参考、调试路径继续使用独立编译的前缀和去噪器。前缀编译复用的是计算图，不是上一观测的 KV 数据；每个动作块重新计算当前图像、文本及状态对应的前缀。首次 RTC warmup 会包含完整 guided 图的编译/调优成本，可能更慢，稳态延迟需单独比较。

服务端开关 `--rtc_fast_guidance=true` 默认开启，`false` 可切回之前的轻量参考实现作对照。这里的等价只针对既有轻量路径，不表示与 RTC 论文的完整 VJP 等价，也不是官方推荐模式的名称。两种实现都不对去噪器求导；本地完整 VJP 实现和参数已删除。普通无前缀和 trained 路径不会使用轻量引导修正。原有 PI05 warmup 的轮次、开始及耗时日志保留。

完整 VJP 不是为加速设计的。它计算去噪器对当前噪声动作的导数，用于前缀约束的引导修正，每个去噪步因此增加反向计算及中间激活开销。[RTC 作者参考实现](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py)仍使用 `jax.vjp`；没有证据说明作者因“对推理无用”而舍弃它。当前 [LeRobot RTC](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/rtc/modeling_rtc.py) 在计算速度后才启用输入梯度，从代码求导可知这条路径未包含去噪器 Jacobian；[OpenPI 主线 PyTorch 采样器](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py)未包含这套 guided RTC。此次删除是用户选择精简部署算法，不代表论文方法无效。若原先已使用默认关闭配置，删除这段未执行代码本身不会进一步缩短推理耗时。

同步到服务端时，本次完整 guided 编译涉及 `src/lerobot/policies/pi05/modeling_pi05.py` 与 `src/lerobot/async_inference/server/rtc_guidance.py`。若尚未同步上一轮回退，还需同步恢复后的 `src/lerobot/policies/common/flow_matching.py` 和 `src/lerobot/policies/pi05/processor_pi05.py`，然后重启服务端。

之前报告的 **2.22 ms → 0.65 ms** 来自简单速度函数的 RTC 辅助计算测试，并包含现已撤回的时间张量修改，不作为当前版本的测速结果。它也没有测量完整 PI05、A100、网络或机械臂。当前版本的真实收益应以相同权重、输入、采样步数及编译模式下的稳态 P50/P95 为准。

本轮 **139 项 CPU 回归、40 项 CUDA 数值/编译检查通过**，Ruff 检查通过。CUDA 验证使用本机 RTX 4060 与缩小版 BF16/OpenPI PI05：普通推理、默认轻量引导和参考引导与 eager 输出对照；变更图像、文本、padding、delay 及动作前缀长度后，默认轻量路径复用普通和 guided 两个完整图，参考路径仍复用普通采样、独立前缀和去噪器三个图；KV cache 不被去噪污染。

稳态编译范围对照使用宽度 32、单层的小模型，输入为双 224px 相机、200 tokens、50 步动作及 10 步去噪，保持 BF16/OpenPI、`max-autotune`、相同权重/噪声/轻量引导；每组 9 次预热、90 次测量，包含输入处理和结果复制到 CPU。原分段范围的 P50/P95 为 **2.46/3.26 ms**；完整采样图为 **1.46/1.89 ms**，切回完整图复测为 **1.54/1.88 ms**。这说明缩小模型上的调度开销减少；不是完整 PI05 或 A100 测速，不能把该比例套用于用户百毫秒级的推理。

**重连稳定性检查尚未全部通过。** 两个独立工作线程依次复用同一模型时，`default`、`max-autotune-no-cudagraphs` 的单次检查通过；`max-autotune` 单次可以通过，但重复对照出现线程退出时的 weakref/锁异常、段错误或退出超时。缩回分段编译范围也复现段错误，完整采样范围也有异常；不能据此确定用户此前的 CUDA 非法访问与它同源，也不能宣称本次优化修复了重连问题。离线复现脚本及原始日志保存在本地 `logs/pi05_compile_alignment_2026-10-16/`（该目录被 Git 忽略）；当前结果不足以确认 `max-autotune` 跨工作线程复用的部署稳定性。[重连时的段错误与 CUDA 非法内存访问](deployment.md#19-重连时的段错误与-cuda-非法内存访问)已有关闭 CUDA Graphs 的对照参数，本次没有静默更改默认编译模式。

已核对 [OpenPI 的 PyTorch 采样实现](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py)和 [LeRobot 的 PI05 实现](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi05/modeling_pi05.py)：两者都在完整 `sample_actions` 入口调用 `torch.compile`，覆盖前缀及去噪循环；这不代表其任意运行配置都没有 graph break。当前默认轻量 guided 对齐这一编译范围，并验证本地常见输入变化不会重编译。`openpi` 档位的 `max-autotune`、FP32 matmul `high` 已对齐，无需新增开关。注意力后端 `eager` 与 PyTorch 是否编译是不同概念；两家默认 Gemma attention 仍使用 eager 实现，不因开启编译而改成 FlashAttention。LeRobot 主线同样保留每步 KV cache 复制，本地 Transformers 的 Gemma attention 在传入 cache 时会调用 `update`，直接删掉 clone 会污染下一步前缀。

减少 `warmup_runs` 只减少连接阶段的预热次数，不会让每次模型调用更快；设为 0 还会把首次编译带到校准或正式请求中。减少 `num_inference_steps`、图像分辨率或文本长度会改变推理行为，需要任务效果验证，本次没有修改。`torch.compile` 模式的首次成本与稳态延迟应分别比较，见 [PyTorch 编译参数](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)。

离线脚本原来只测 ordinary，现在支持与 deployment 相同的 guided 处理器。请在服务端环境运行，checkpoint、编译模式、prefix、delay、horizon 和引导参数应匹配实际配置；下面的 20/10/20 是示例：

```bash
PI05_CHECKPOINT='/实际部署的checkpoint/pretrained_model'

# 优化后的轻量 guided RTC；输出逐轮 warmup 和稳态 P50/P95。
uv run --no-sync python -m lerobot.scripts.lerobot_profile_pi05 "$PI05_CHECKPOINT" \
  --dtype bfloat16 --precision-profile openpi --compile-mode max-autotune \
  --rtc-mode guided --rtc-prefix-length 20 --rtc-delay 10 --rtc-execution-horizon 20 \
  --warmup 5 --runs 50 --phase-runs 0 --output /tmp/pi05_guided_fast.json

# 原轻量引导作对照：前缀/去噪器分段编译，包含 autograd 参考路径的开销。
uv run --no-sync python -m lerobot.scripts.lerobot_profile_pi05 "$PI05_CHECKPOINT" \
  --dtype bfloat16 --precision-profile openpi --compile-mode max-autotune \
  --rtc-mode guided --rtc-prefix-length 20 --rtc-delay 10 --rtc-execution-horizon 20 \
  --no-rtc-fast-guidance \
  --warmup 5 --runs 50 --phase-runs 0 --output /tmp/pi05_guided_reference.json
```

顺序运行，勿同时争抢 GPU。报告中的输入、固定噪声和 `fixed_input_actions` 可用于数值对照；这些是设备上的合成输入，计时不包含 tokenizer、外部处理器或 RPC。`--trace=/tmp/pi05_guided_trace.json` 单独采集一次详细 trace，只有显式运行离线脚本才生效。若需 eager 阶段分解，另用 `--compile-mode=eager --phase-runs=5`；不要将插桩轮次当作正式延迟。
