# PI0.5 / A100 普通推理延迟排查

[排错索引](README.md) · [PI05 性能排查](pi05-performance.md) · [当前推理架构](../architecture/pi05-inference.md)

这是按原日期保留的诊断与修订记录；“当前”“本轮”等表述指相应记录当时，后续修订以文内日期为准。本次整理未重新执行历史测试，也未把待验证方向改为已完成结果。

后续更新：进一步核对固定版本后确认，全视觉 FP32 和 eager Gemma 注意力也是 LeRobot 官方普通 PI0.5 路径的设计，不能将其称作本地偏离官方的错误。目前已实现可选 OpenPI 混合精度、服务端编译覆盖与连接阶段预热，并补齐 `text_tokenizer_name`；见 [官方对齐说明](pi05_official_alignment_2026-10-15.md)。下文保留首次排查时的基线与日志分析。

日期：2026-10-15。依据：用户提供的 checkpoint 配置、引用对话中完整的两组服务端日志、当前工作区源码，以及本地 meta-device 参数检查。首次排查时沙箱内无法访问 CUDA；后续在沙箱外使用 RTX 4060 验证了缩小模型的编译路径，见上方更新。未连接用户的 A100，因此本文没有声称测得任何 A100 优化收益。

结论：当前基础推理存在明确的优化候选，优先级是 **FP32 视觉路径及其矩阵乘精度、未编译的 10 轮动作去噪、强制 eager attention**。现有证据不足以把 300 ms 中的具体毫秒数分别归给这些因素，也不能把它当成 PI0.5 或 A100 的性能下限。

## 1. 从日志确定瓶颈范围

每组排除首次冷启动，使用引用对话中读到的全部后续样本。原日志是 `perf_counter` 主机分段计时，表中不是严格的 GPU kernel 时间。

| 模式 | 后续样本数 | 服务端总时长均值 | predict 均值 | preprocess 均值 |
| --- | ---: | ---: | ---: | ---: |
| 无历史动作前缀 | 27 | 337.59 ms | 312.46 ms | 24.00 ms |
| 带历史动作前缀 | 23 | 448.01 ms | 422.89 ms | 22.52 ms |

无前缀预测范围 258.33–365.03 ms，中位数 334.44 ms。预测区间占服务端日志总时间约 92.6%。即使假设其他服务端阶段全部归零，按这组日志也只减少约 25 ms；这不包含尚未提供的网络和客户端时间，**不能推导本地化只能改善 7.4% 的端到端时间**。

两组不是同时、同观测、同硬件负载的配对 A/B，均值差约 110 ms 不能全部当作 RTC 的准确额外成本。无前缀的前 11 个热样本平均预测 270.80 ms，后 16 个平均 341.11 ms，增加约 26%。需另外排查 GPU 共享、MIG 配额、功耗/时钟、CPU 调度和后台负载。

首次无前缀预测约 769 ms，属于另一类冷启动问题。该 checkpoint `compile_model=false`，不能把这次冷启动直接叫作 `torch.compile` 编译。

## 2. BF16 配置没有覆盖整个模型

检查位置：[`to_bfloat16_for_selected_params`](../../src/lerobot/policies/pi05/modeling_pi05.py)。代码先转换 BF16，随后将所有名称包含以下片段的参数改回 FP32：

```python
params_to_keep_float32 = [
    "vision_tower",
    "multi_modal_projector",
    "input_layernorm",
    "post_attention_layernorm",
    "model.norm",
]
```

用本地 PyTorch 2.11.0 / Transformers 5.5.4，在 meta device 创建与用户相同的 `gemma_2b + gemma_300m`、BF16、224×224 配置，得到：

| 组件 | BF16 参数数 | FP32 参数数 |
| --- | ---: | ---: |
| SigLIP 视觉编码器 | 0 | 412,442,352 |
| 视觉投影层 | 0 | 2,361,344 |
| 语言模型（含 embedding） | 2,508,455,936 | 75,776 |
| 动作专家主体 | 311,427,072 | 116,505,600 |

参数数量不是计算占比。动作专家的 FP32 部分含 AdaRMSNorm 内部的线性映射，不能理解为只有几个小的归一化标量。外部动作投影、时间 MLP 也保留 FP32。

`embed_image()` 显式将输入转换为 FP32；部署预测没有外层 autocast。视觉塔为 27 层，`embed_prefix()` 按相机顺序各跑一次。因此双相机实际执行两次 FP32 视觉编码。输入特征中写的 640×480 会先调整到 224×224，并非以 640×480 运行视觉 Transformer。

这与当前 [OpenPI 的精度选择](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/gemma_pytorch.py)不同：其 FP32 视觉保留列表只列 patch embedding 的权重/偏置与 position embedding，而不是整个视觉塔。当前仓库的注释说明全视觉 FP32 与训练时 dtype 稳定性有关，不能直接删除该逻辑并认为训练也等价。

优先实验两项：

1. 记录 A100 进程的 `torch.get_float32_matmul_precision()` / TF32 状态，再单独比较 `highest` 与 `high`。当前代码仅在 `compile_model=true` 的分支显式设置 `high`；`compile_model=false` 本身不保证 TF32 开启。本地检查值为 `highest/False`，**不是远端进程的实测值**。
2. 单独比较仅视觉路径的 BF16 autocast。诊断脚本提供该实验，不修改训练代码或 checkpoint；它改变计算精度，需要进一步比较真实观测上的动作差异和任务表现。

`high` 允许内部使用更快的 FP32 矩阵乘策略，不会把参数存储类型改成 BF16，也不单独控制卷积精度。参见 [PyTorch 2.11 精度说明](https://docs.pytorch.org/docs/2.11/generated/torch.set_float32_matmul_precision.html)。

## 3. 无动作前缀仍需完成完整模型采样

一次普通推理近似包含：

```text
两路图像预处理
  → 两次视觉编码
  → 图像/文本观测前缀的语言模型 prefill（一次）
  → 10 × [时间/动作 embedding + KV 克隆 + 18 层动作专家 + Euler 更新]
  → 输出 50 × 7 个动作值
```

224×224、patch=14 对应每路 256 个图像 token，两路加 200 个文本 padding 槽，观测前缀总长为 712。去掉 RTC 历史动作前缀并不会去掉这个**观测前缀**。

`sample_actions()` 每次请求只 prefill 一次，并在去噪轮次之间复用该 KV；不是把 2B 语言模型完整运行十遍。但 18 层动作专家会运行十遍，eager 模式下有大量小算子、类型转换、Python dispatch 和 CUDA kernel launch。

`sample_actions()` 和 `denoise_step()` 都将相应模型的 `_attn_implementation` 写成 `"eager"`。仅在加载后设置一次 `"sdpa"` 会被后续调用覆盖。后续若增加可配置 SDPA，需要验证非因果 mask、KV cache、BF16 输出误差及 guided RTC 反向路径；不能承诺切换后整模型加速倍数。

`denoise_step()` 每次调用 [`clone_past_key_values`](../../src/lerobot/policies/common/vla_utils.py)，复制每层 KV。标准配置下是 10 轮 × 18 层 × K/V，即 360 次 tensor clone 调用，另有 cache 更新/拼接。它既有复制也有分配/调度成本。但当前 Transformers cache 会更新；直接去掉 clone 可能让同一个 cache 混入先前去噪轮次的 action token。优化需保留只读的观测前缀语义。

`compile_model=false` 是确定事实。`max-autotune`/`reduce-overhead` 的收益需要预热后测量，且首轮无前缀与 guided RTC 的动态条件可能产生不同图。本次脚本只测固定形状的普通推理。参见 [torch.compile 参数](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html)。

## 4. 可以排除或需要纠正的解释

| 项目 | 当前判断 |
| --- | --- |
| 忘记设置 `dtype=bfloat16` | 已设置；问题是实际模型混合精度边界 |
| `use_amp=false` 所以全模型 FP32 | 不成立；参数转换与训练 AMP 开关是两件事 |
| `gradient_checkpointing=true` 导致推理重计算 | 当前路径先 `eval()`，checkpoint 分支要求 `self.training`，不会触发 |
| 每个请求重新加载 checkpoint | 当前 deployment 缓存模型，加载也不在该 predict 计时区间 |
| 预测 50 步就逐步推理 50 次 | 50 是动作序列长度；10 才是去噪轮数 |
| 减少 `n_action_steps` / 客户端 `actions_per_chunk` 就降低模型成本 | `predict_action_chunk()` 仍生成 `chunk_size`，服务端随后裁切，通常不能降低核心采样成本 |
| 7 维动作意味着小模型 | 模型内部仍用 32 维 padding、50 个 token、大型视觉/语言/专家网络 |
| 本地化自动消除数百毫秒预测时间 | 本地化主要改变传输和调度；同实现同算力的模型采样成本仍在 |

10→5 去噪步只能减少与轮数有关的部分，不能把视觉和 prefill 时间减半；会影响策略质量，排在实现优化和测量之后。

## 5. 现有阶段日志的计时限制

[`policy_server_deployment.py`](../../src/lerobot/async_inference/server/policy_server_deployment.py) 目前只在阶段边界读取 `perf_counter()`，没有 CUDA 同步。CUDA 调度是异步的，某阶段提交的 GPU 工作可能在后续 CPU 拷贝时才等待完成；因此 `predict_ms` 不严格等于模型 GPU 时间，`preprocess_ms` 也可能包含同步等待。参见 [PyTorch CUDA 计时说明](https://docs.pytorch.org/docs/2.11/notes/cuda.html#asynchronous-execution)。

完整返回路径最终需要 CPU 动作，端到端“慢”的现象是真实的，但应使用同步的完整预测时长与 trace 做归因。不应在正式控制循环每个小阶段都插入 `synchronize()`，因为那会改变性能。

## 6. 已添加的离线诊断脚本

文件：[`lerobot_profile_pi05.py`](../../src/lerobot/scripts/lerobot_profile_pi05.py)。在 A100 上使用实际部署的同一 Python 环境和源码运行。路径应指向本次部署的微调 checkpoint 的 `pretrained_model` 目录，不能直接照抄 config 中可能指向基座的 `pretrained_path`。

先在该环境、非实机运行时段，测原始基线：

```bash
PI05_CHECKPOINT='/实际部署的checkpoint/pretrained_model'
uv run --no-sync python -m lerobot.scripts.lerobot_profile_pi05 "$PI05_CHECKPOINT" \
  --warmup 10 --runs 50 --phase-runs 5 \
  --trace /tmp/pi05_baseline_trace.json \
  --output /tmp/pi05_baseline.json
```

如果服务端使用 Conda 而没有 uv，激活同一 Conda 环境后，将 `uv run --no-sync python` 换为 `python`，避免为了测试重新解析/升级部署依赖。该脚本不启动客户端、服务器或机械臂。

随后每次只变一个因素，运行完一条再运行下一条，避免多个 benchmark 争抢同一张卡：

```bash
# 只放宽 FP32 matmul 内部精度。
uv run --no-sync python -m lerobot.scripts.lerobot_profile_pi05 "$PI05_CHECKPOINT" \
  --matmul-precision high --warmup 10 --runs 50 --output /tmp/pi05_high.json

# 与上一条比较：相同 high 设置下增加编译，首次编译单列在 warmup_ms。
TORCH_LOGS=recompiles,graph_breaks uv run --no-sync python -m lerobot.scripts.lerobot_profile_pi05 \
  "$PI05_CHECKPOINT" --matmul-precision high --compile-mode max-autotune \
  --warmup 10 --runs 50 --output /tmp/pi05_high_compile.json

# 与第一条原始基线比较：只对视觉路径增加 BF16 autocast。
uv run --no-sync python -m lerobot.scripts.lerobot_profile_pi05 "$PI05_CHECKPOINT" \
  --vision-autocast --warmup 10 --runs 50 --output /tmp/pi05_vision_amp.json
```

还可用 `--image-size checkpoint` 测 480×640 输入在 `predict_action_chunk()` 内缩放的成本，或用 `--num-steps 5` 量化轮数敏感性。默认输入是模型原生 224×224，匹配客户端已提前缩放的场景。

报告包含：

- 实际导入的模型源码路径和 SHA256、软件版本、TF32 状态、CPU 线程数、GPU 起止快照和进程列表。
- 冷启动/预热逐次时长，以及独立、无阶段插桩的同步预测 P50/P95 与全部顺序样本。
- 单独的 eager 阶段记录：图像处理、两路视觉编码、语言 prefill、10 轮去噪、动作专家、KV clone。
- 固定合成图像/token/noise 得到的模型空间动作，便于不同精度实验比较数值差异。

阶段时间有嵌套：`vision_per_camera` 已对两路相机求和，`action_expert`/`kv_clone` 已对全部去噪轮次求和；不要再与包含它们的 `embed_prefix_inclusive`/`denoise_inclusive` 相加。CUDA Event 度量的是流上区间，含 CPU 提交不足时的空闲，不等于纯 kernel 时间；需要 trace 来辨认两者。编译模式跳过 Python 阶段包装，避免插桩改变编译图。

该脚本使用合成观测，衡量固定形状的基础开销，不包含外部 processor、tokenizer、摄像头、网络和后处理，不能替代真实观测端到端测试或任务成功率评估。仅支持本地单文件 `model.safetensors`，严格加载失败即退出，防止误测随机模型。

首次排查时，用户配置中的 `text_tokenizer_name` 尚未在当前工作区定义；后续已补齐该字段。引用对话的无前缀修改仍不完全存在于当前 deployment 文件，因此远端源码可能有差异。诊断脚本不加载 tokenizer，未知模型字段会报错，模型源码 SHA256 用于确认测量的是哪份代码。

持续变慢时，在同一测量窗口另开终端记录：

```bash
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,power.draw,power.limit,clocks.sm,temperature.gpu --format=csv -l 1
nvidia-smi -L
```

`-L` 用于确认完整型号/MIG。单个起止快照不能排除中间的降频或其他进程瞬时占用。

## 7. 首轮诊断的验证边界

已完成：当前模型的 meta 参数类型检查、诊断 CLI/配置解析/合成输入/计时包装的 CPU smoke 检查和 Ruff 检查。工作区未安装 uv，检查使用现有 `lerobot` Conda 环境。未修改生产模型、训练逻辑或部署默认配置。

尚待 A100 结果：各阶段实际毫秒占比、TF32/编译/视觉 autocast 的收益、长时间运行的 P95、真实动作误差与任务表现。不能据现有日志承诺降低到某个毫秒数。

## 8. 2026-10-16：BF16 选项之外，日志揭示的优化顺序

### 8.1 优先解释持续升高的延迟，而不只看平均值

无前缀日志在第 12 个热请求（`obs=4294967308`，19:03:01）出现台阶：前 11 个热请求的 `predict_ms` 平均 270.80，后 16 个平均 341.11，增加约 26%。相应预处理从平均 21.74 ms 增至 25.55 ms。该配置未开启编译，不能直接解释为重编译；日志也一直是 `prefix=0`。

这提示应检查运行条件或输入路径是否变化，不能仅凭日志断定降频。先保持同一观测、噪声、形状，记录完整顺序样本，同时观察 GPU 其他进程、MIG 配置、SM 时钟、功耗/温度和 CPU 负载。现有诊断脚本只抓 GPU 起止快照，无法排除运行中间的资源变化；可使用上节连续采样命令及 [NVIDIA 时钟限制原因](https://docs.nvidia.com/deploy/nvidia-smi/index.html#clocks-event-reasons)定位。若固定输入仍出现同样台阶，优先检查运行环境；若只发生于实机请求，再查输入与 CPU pipeline。

在同一配置下，另做 CPU 线程数 1/2/4 与当前值的独立对照，观察预处理和完整延迟，而非预设线程越多越快。线程与亲和性会影响调度、缓存和内存访问成本，参见 [PyTorch CPU 调优](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#utilize-openmp)。这里没有承诺能把后段恢复到 271 ms。

### 8.2 预处理的 24 ms 是第二个有量级的目标

无前缀平均 `preprocess_ms=24.00`。当前计时范围包括观测整理和 processor pipeline，不包含前面的完整网络接收；可能涵盖图像复制/缩放、归一化、tokenizer 和设备搬运，需要分别测量才能分配成本。

本次在 `helpers.prepare_raw_observation()` 找到并删除了一处确定的冗余：先为所有相机创建一组 tensor，随后又创建第二组并直接覆盖第一组。保留原来的最终转换/缩放路径，避免改动像素与输入语义。该修正只影响原始图像支路；已成功协商客户端预缩放的压缩图像会绕过它，所以不能将 24 ms 都归因于这次复制。

进一步检查 `resize_images_on_client` 是否实际协商成功，而不只看开关值。当前实现已支持客户端提前生成模型原生 224×224 输入；这减少传输和服务端整理工作，但仍需把客户端开销计入完整 RTT。先做逐阶段测量，再考虑减少 CPU/GPU 往返或调整 processor 顺序。

### 8.3 冷启动与稳定运行分开处理

首次无前缀预测 768.81 ms，后处理区间 22.34 ms；对应稳态平均分别为 312.46 ms 和 0.34 ms。首次带前缀实验同样在约 779 ms / 23 ms 附近。两次首次请求实际都没有历史前缀，说明初始化成本值得单独处理，但不能据这些 CPU 区间精确定位到某个 CUDA 算子。

已将连接阶段预热扩展到 eager 与编译模型，并移到正式推理线程，覆盖完整预/后处理和 CPU 输出。随后用真实观测完成两次不计入统计的往返预热及默认三次 RTC 测量，成功清理临时状态后才允许开始 rollout。服务器合成预热与客户端真实校准分别可关闭，具体参数见[官方对齐说明](pi05_official_alignment_2026-10-15.md#预热与-rtc)。离线四组实验仍分别 warmup 后统计 P50/P95。

### 8.4 RTC 的条件预算可能影响响应新观测的速度

RTC 日志中，前 9 个有前缀请求均为 `conditioned_delay=30`，而 `measured_delay` 多为 14，之后条件预算降至 18–19。按 30 Hz，30 步是 1 秒，14 步约 467 ms。较长的旧动作条件区间可能减弱新观测对近期动作的影响；这属于闭环响应问题，不等同于少算若干去噪步。

当前服务器优先使用客户端 RTT 滑动窗口分位数，缺少反馈时才参考服务端时长，并加 margin。冷启动样本或真实网络/排队开销都可能造成这类差距；现有日志没有客户端 RTT，无法区分。应配对查看 RTT、sensor age、条件预算和缺动作/迟到块数据，再判断窗口或 margin 是否过于保守。不能仅因服务端测得 14 步就把完整预算硬改为 14，也不能将 30 步解释成服务器额外睡眠 1 秒。

实现已改为仅剔除明确标记的启动预热，正常运行的慢样本全部保留；P90 外增加快升慢降保护。普通复位保留同连接、同模型的有效统计，动作及过期反馈仍按 generation 隔离；统计默认空闲 300 秒过期，重连清空。nearest-rank P90 在样本数 1–9 时确实取最大值，第 10 个样本才允许排除一个尾部值，因此早期冷请求可能解释日志中“前九次 30，之后 18–19”的形状；缺少配对 RTT 时仍不能据此断定原因。

### 8.5 暂时不要优先改动亚毫秒阶段

无前缀的 prefix 平均 0.78 ms、postprocess 平均 0.34 ms；RTC 模式也分别约 0.75 ms、0.76 ms。日志中的 `Starting receiver` 来自每个上传 RPC 的接收函数，不代表反复加载模型或重建 CUDA 上下文；它位于上述模型计时之外。调整日志级别可以减少输出，但不能据此解释 300 ms 的预测开销。

模型内部下一步则取决于离线 phase/trace：视觉占主导时可评估将两路同尺寸相机合批编码；去噪占主导时先看编译后小算子启动与 KV cache 复制。两者目前都只是候选，不是这组日志已经证明的瓶颈。减少去噪步数会改变策略输出，放在实现开销优化与任务回归之后。

上述分段均是主机计时，不是纯 GPU kernel 时间。现有同步离线基准和独立 trace 用于避免误归因；参见 [PyTorch CUDA 计时说明](https://docs.pytorch.org/docs/2.11/notes/cuda.html#asynchronous-execution)。
