# 模型训练 README：checkpoint 统计量与 DAgger 采样

> 日常操作统一从 [docs/README.md](../../../docs/README.md) 进入。本文保留统计量、干预采样、padding loss、曝光预算和续训语义的技术说明；实验配方用于解释参数，不替代主入口的环境与启动配置。

本文更新于 2026-10-16，解释统计量保留、干预比例采样，以及新接入 Pi0.5 的时间 padding loss mask。第 7 节提供第二轮 SFT 命令，第 10–13 节记录官方 loss 核对、本地数据预算和多次接管建议。命令中的模型路径、数据路径和输出目录需要替换为服务器上的实际值。

## 1. 先分清四件事

| 要决定的事情 | 参数 | 含义 |
| --- | --- | --- |
| 从哪个模型开始 | `--policy.path=.../pretrained_model` | 加载模型配置和权重；用于从已有模型开始新一轮微调 |
| 用哪套归一化统计量 | `--use_pretrained_stats=true/false` | 新一轮微调时，保留模型保存的统计量，还是使用当前数据集的统计量 |
| 两组数据各取多少 | `--intervention_sampling_ratio=0.3` | 干预组的目标占比为 30%；不设置时按数据集原有帧分布采样 |
| 是新训练还是接着中断处训练 | `--resume=false/true` | 新微调重新开始训练状态；resume 恢复原训练的优化器、调度器、步数等状态 |

这四件事相互关联，但不能相互替代。例如，`use_pretrained_stats=true` 不会冻结模型；`resume=false` 也不表示随机初始化模型，只要指定了 `policy.path`，仍会加载已有权重。

当前比例采样器的组内行为已经统一为“不放回遍历，耗尽后重洗，跨 epoch 接着遍历”。没有额外的 `replacement` 或 `sampling_mode` 命令行开关；fixed / probabilistic 是根据 batch size 和比例自动选择的。

## 2. `use_pretrained_stats`：决定模型使用哪把数值尺度

### 2.1 统计量是什么

机器人观测和动作包含关节、夹爪等数值。归一化器先将这些数值变换成模型使用的数值范围；输出端的反归一化器再将预测还原为动作数值。

不同策略使用的统计量不同。本地 Pi0.5 配置默认对 STATE 和 ACTION 使用 `QUANTILES`，读取 `q01`、`q99`。并非所有策略都使用均值和标准差；其他模式也可能使用 `mean/std`、`min/max` 等。实际采用哪种方式以 checkpoint 的配置及显式覆盖项为准。

用一个动作维度的示意数字解释：

| 项目 | 旧 checkpoint 的统计量 | 新数据集的统计量 |
| --- | --- | --- |
| `q01` / `q99` | 0 / 100 | 20 / 60 |
| 原始动作 50 归一化后 | 0 | 0.5 |
| 模型输出 0 反归一化后 | 50 | 40 |

这里的公式是 `2 × (x - q01) / (q99 - q01) - 1`，数字仅作演示。换一套统计量，同一个原始动作会变成不同的训练数值，同一个模型输出也会被解释成不同的动作。

### 2.2 设为 true / false 分别做什么

| 训练情形 | 统计量来源 |
| --- | --- |
| `resume=false`，`use_pretrained_stats=true` | 加载 checkpoint 保存的 processor 状态，不用当前数据集统计量覆盖归一化器和反归一化器 |
| `resume=false`，`use_pretrained_stats=false`，默认情况 | 使用当前训练数据集元数据中的统计量；即使加载了模型权重，也会覆盖 processor 中相应的统计量 |
| `resume=true`，正常恢复完整 checkpoint | 保留原训练保存的 processor 统计量；不需要靠该开关实现统计量恢复 |

当你沿用同一台机器人、相同关节顺序、单位和动作定义，从已经能完成任务的模型开始 DAgger 纠错微调时，`true` 可以保持原来的数值尺度。这里保留的是模型实际保存的统计量，不保证它们一定等同于某份原始数据集现在的 `stats.json`。

如果希望新一轮训练明确改用新数据集的尺度，可以选择 `false`。硬件、单位、关节顺序或绝对/相对动作定义发生变化时，需要先核对数据语义；这个开关只选择统计量来源，不负责转换这些定义。

### 2.3 checkpoint 要包含哪些内容

模型路径应指向完整的 `pretrained_model` 目录。普通 safetensors checkpoint 的相关内容包括：

```text
checkpoints/020000/
├── pretrained_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── train_config.json
│   ├── policy_preprocessor.json
│   ├── policy_preprocessor_step_*.safetensors
│   ├── policy_postprocessor.json
│   └── policy_postprocessor_step_*.safetensors
└── training_state/
    └── 优化器、调度器、随机数状态和步数等
```

processor 的 JSON 描述流程，对应的状态文件保存有状态步骤的数据。只有模型权重文件，并不足以保证能恢复所需的统计量。

这个开关用于 policy 训练，要求存在 pretrained 路径。如果当前训练流程通过语言 recipe 重建 processors 而不加载旧 processors，代码会拒绝 `use_pretrained_stats=true`。

### 2.4 它和采样比例相互独立

`use_pretrained_stats=true` 不会忽略纠错数据。新数据的观测和动作仍参与训练，只是使用旧尺度处理。

`intervention_sampling_ratio=0.3` 也不会把统计量重新计算成“70% 原示范 + 30% 干预”的加权统计量。采样器只决定取哪些起点帧；`false` 使用的是数据集元数据中的统计量，`true` 使用的是 checkpoint 保存的统计量。

## 3. `intervention_sampling_ratio`：先决定组，再决定组内哪一帧

### 3.1 A 组和 B 组如何划分

- A 组：`intervention=False`，即非干预帧。
- B 组：`intervention=True`，即人工干预帧。

在“原示范 + 仅人工纠错片段”的合并数据中，A 对应原示范，B 对应纠错数据。如果还加入自主 rollout，其中标为 `False` 的帧也会进入 A 组。分组依据是帧的布尔标记，不是文件名、数据集名字或是否来自某个 episode。

设置 `ratio=0.3` 表示干预组占目标抽样量的 30%，不表示每个 episode 取 30%，也不表示只保留纠错数据中的 30%。这是起点帧级别的采样。

### 3.2 三种实际设置

| 设置 | 组选择方式 | 组内取样方式 |
| --- | --- | --- |
| 不传该参数，或设为 `null` | 不做 A/B 比例平衡，按有效帧的自然分布采样 | 普通 sampler 打乱全部有效帧；单卡完整一轮中每帧一次 |
| 设置比例，且每卡 `batch_size × ratio` 为整数 | fixed：每批分配固定数量给 A 和 B | 各组独立打乱、不放回遍历，耗尽才重洗 |
| 设置比例，但每卡 `batch_size × ratio` 不是整数 | probabilistic：每个位置按概率选 A 或 B | 同样从选中组的打乱序列取下一帧，不是组内有放回抽样 |

默认普通采样并不会自动给原示范和纠错各 50%。哪组帧多，整体上就会取到更多该组样本。普通采样在多卡末尾对齐时可能有补样，不能将单卡的无重复保证直接套到对齐后的所有卡。

### 3.3 为什么整数决定模式

一批里只能放整数个样本：

| 每卡 batch size | ratio | 相乘结果 | 实际行为 |
| --- | --- | --- | --- |
| 8 | 0.5 | 4 | 每批固定 4 个 A + 4 个 B |
| 10 | 0.3 | 3 | 每批固定 7 个 A + 3 个 B |
| 8 | 0.3 | 2.4 | 无法每批放 2.4 个 B，因此按概率选择组 |

对于 `batch_size=8, ratio=0.3`，一批可能有 0、1、2、3……甚至 8 个 B；不是强制在“2 个”和“3 个”之间轮流切换。每批 B 数量的期望是 2.4，抽样越多，整体比例通常越接近 30%，但有限段不强制恰好等于 30%。

判断依据始终是每张卡的 micro-batch。两张卡的总 batch，或者梯度累积后的有效 batch，乘比例得到整数，也不会让代码改成 fixed。例如每卡 batch size=5、ratio=0.3，即使两卡合计对应 3 个 B，仍然使用概率模式。

## 4. “不放回”到底保证什么

### 4.1 每组有自己的取样队列

假设 A 有 12 帧、B 有 4 帧。打乱后可能得到下面两条队列，编号代表不同起点帧：

```text
A 的第 1 轮：A7 A2 A10 A4 A1 A9 A12 A3 A8 A6 A11 A5
B 的第 1 轮：B3 B1 B4 B2
B 的第 2 轮：B2 B4 B1 B3
```

选到 A，就取 A 队列的下一项；选到 B，就取 B 队列的下一项。B 的第 1 轮取完，就开始 B 的第 2 轮，A 的进度不会重置。

保证的是：**在同一个组的同一轮中，每个有效起点帧只出现一次；取完该组所有帧后，才能进入下一轮。** 小组可以比大组更早进入第 2、3、4 轮，以满足指定的组比例。

### 4.2 为什么同一个 batch 里仍可能重复

上例 B 的第 1 轮最后一个是 B2，第 2 轮第一个也恰好是 B2。如果同一批跨过这个边界，可能出现：

```text
某一批：[B2（第 1 轮末尾）, A6, B2（第 2 轮开头）, A11]
```

每一轮内部都没有重复，但两轮交界允许重复。当前实现没有额外禁止“跨轮相邻重复”。若 B 只有一帧，而一批需要两个 B，那么重复也是不可避免的。

### 4.3 起点不重复，不等于 action chunk 不重叠

假设 `chunk_size=50`：以第 100 帧为起点，训练动作可能覆盖 100～149；以第 101 帧为起点，可能覆盖 101～150。两个起点不同，但动作窗口重叠了 49 步。

采样器控制起点帧，不负责让时间窗口互不重叠，也没有实现按动作段去重、episode 均衡或人工接管片段均衡。

## 5. epoch、组内一轮和训练 step 是不同的计数

| 名称 | 在当前代码中的含义 |
| --- | --- |
| episode | 录制出来的一条轨迹；通常包含许多帧 |
| batch | 一张卡一次读取的起点样本集合 |
| step | 训练循环读取一批数据的一步；本地实现计数的是 micro-batch，开启梯度累积后不等于每步都更新一次优化器 |
| sampler 的虚拟 epoch | sampler 预先规定的一段抽样长度；读完后 epoch 编号加一 |
| 某组的一轮遍历 | 该组的一次完整随机排列，从第一项取到最后一项 |

组内一轮由该组帧数决定。虚拟 epoch 由采样器的长度公式决定。它们可以在不同位置结束，不能互相替代。

### 5.1 概率模式：达到规定的抽样次数就结束 epoch

继续使用 A=12 帧、B=4 帧，令 batch size=4、ratio=0.3，单卡。总共有 16 个有效帧，因此一个虚拟 epoch 抽 16 次，也就是 4 批。下面是一种可能的结果，使用上一节的队列；仅用于解释机制，不表示某个 seed 必定生成这个顺序。

| 本 epoch 批次 | 一种可能的 batch | A 数量 | B 数量 |
| --- | --- | --- | --- |
| 第 1 批 | A7、B3、A2、B1 | 2 | 2 |
| 第 2 批 | A10、A4、B4、A1 | 3 | 1 |
| 第 3 批 | A9、A12、A3、A8 | 4 | 0 |
| 第 4 批 | B2、A6、B2、A11 | 2 | 2 |
| 合计 | 抽了 16 次 | 11 | 5 |

这时程序结束第 1 个虚拟 epoch，但 A5 还没有被抽到。B 已经完整遍历一轮，并且从第 2 轮取了一个 B2。实际 B 比例是 5/16=31.25%，不需要强制等于 30%。

**第 2 个虚拟 epoch 第一次选到 A 时，会先取剩下的 A5。只有取完 A5，A 才重洗。** B 则接着第 2 轮取 B4。epoch 换号不会清空两组进度。

这就是“一个虚拟 epoch 不保证全覆盖，但组内完整一轮不漏帧”的含义。样本不是永远漏掉，而是可能在下一个虚拟 epoch 才轮到。

### 5.2 固定模式：按配额计算足够长的抽样段

同样 A=12、B=4，改为 batch size=4、ratio=0.5。每批必须有 2 个 A、2 个 B。

A 第一次完整遍历需要 12/2=6 批，B 第一次完整遍历需要 4/2=2 批。因此采样器将第一个完整虚拟 epoch 设为 6 批，共抽 24 次：12 个 A、12 个 B。A 走完 1 轮，B 走完 3 轮。

所以虚拟 epoch 不一定只有“数据集帧数那么多次抽样”。固定模式为了保持配额，可能需要增加抽样次数。

### 5.3 对旧说明的更正：固定模式也不保证每个后续 epoch 全覆盖

此前更新包中“固定模式的 epoch 长度保证覆盖两组的全部有效帧”这一表述过于宽泛。准确含义是：

- 从两组第 1 轮起点开始的第一个完整虚拟 epoch，固定模式保证两组均至少完整遍历一次。
- 后续虚拟 epoch 仍保证每组抽取次数不少于该组帧数，但**不能据此保证该段内包含所有不同帧**。
- 因为进度跨 epoch 保留，某个 epoch 可能由“上一轮的后半段 + 下一轮的前半段”组成；两部分可能包含相同帧，也可能都没有某一帧。

已用实际 sampler 验证一个小反例：A=5 帧、B=2 帧、batch size=4、ratio=0.5、seed=7。

| 虚拟 epoch | A 抽取次数 | A 中不同帧数量 | B 抽取次数 | B 中不同帧数量 |
| --- | --- | --- | --- | --- |
| 第 1 个 | 6 | 5 | 6 | 2 |
| 第 2 个 | 6 | 4 | 6 | 2 |

第 2 个 epoch 虽然取了 6 个 A，却只有 4 个不同的 A。这与组内不放回并不矛盾：截取的区间跨越了组内轮次边界。两种模式严格保证的都是“每组每个完整轮次全覆盖”，不是“任何一段等长抽样都全覆盖”。

### 5.4 长度怎么算；多卡为何要补齐

令 A、B 有效帧数分别为 `N_A`、`N_B`，每卡 batch size 为 `b`，数据并行进程数为 `W`。

概率模式先计算全局批次数：

```text
K0 = ceil((N_A + N_B) / b)
```

固定模式令每批 B 配额为 `q_B = round(b × ratio)`，A 配额为 `q_A = b - q_B`：

```text
K0 = max(ceil(N_A / q_A), ceil(N_B / q_B))
```

两种模式随后都计算：

```text
全局批次数 K = ceil(K0 / W) × W
虚拟 epoch 总抽样次数 L = K × b
每卡每个虚拟 epoch 的 step 数 = K / W
```

`ceil` 表示向上取整。例如总有效帧数 17、batch size=8、两卡，概率模式先得到 3 批，再补成能平分给两卡的 4 批，共抽 32 次，每卡 2 步。比例采样器把这些次数直接纳入组内序列生成，不是简单复制开头的几帧。

各卡获得全局序列中不同的整批。组内不放回的保证针对全局逻辑序列，而不是让每张卡分别把全量数据遍历一遍。跨组内轮次后，不同卡仍可能拿到相同起点帧。

### 5.5 日志里的 `epch` 还有另一层含义

当前 `MetricsTracker` 计算：

```text
日志 epch = 累计抽样次数 / dataset.num_frames
累计抽样次数 = step × 每卡 batch_size × 数据并行进程数
```

它是按抽样总量折算的进度，不是 sampler 的 epoch 编号，也不是全体不同帧实际完整覆盖的次数。比例采样、尾帧过滤、多卡补齐等都会使这些概念出现差异。

因此，比较不同采样设置时，最好同时记录 step、每卡 batch size、数据并行规模和 ratio；不要只凭日志显示的 `epch` 判断“模型已经把所有数据看过几遍”。

## 6. 用当前合并数据理解比例和曝光次数

2026-10-04 本地数据集元数据为：

| 数据 | episodes | 帧数 |
| --- | --- | --- |
| `block/shape-matching_89` | 89 | 117,485 |
| `block/dagger_merged` | 30 | 17,375，全部标为干预 |
| `block/shape-matching_89_with_dagger_30` | 119 | 134,860 |

以下假设没有划出验证集、没有额外过滤帧，且沿用这两组标记；实际训练以启动日志里的两组有效帧数为准。

不设置比例时，干预帧的自然占比约为 `17375 / 134860 = 12.88%`。设置 30% 后，相同抽样预算下，干预组预期曝光次数约为自然采样时的 `30% / 12.88% ≈ 2.33` 倍。

单个数据并行进程时，三个配置的虚拟 epoch 对比如下：

| 每卡 batch / ratio | 模式 | 总抽样次数 | A 抽取次数 | B 抽取次数 |
| --- | --- | --- | --- | --- |
| 8 / 0.3 | probabilistic | 134,864 | 期望约 94,405 | 期望约 40,459 |
| 10 / 0.3 | fixed | 167,840 | 117,488 | 50,352 |
| 8 / 0.5 | fixed | 234,976 | 117,488 | 117,488 |

概率模式第一种设置中，A 的预期抽取次数少于 117,485，所以一个虚拟 epoch 通常还走不完 A 的第一轮；B 则已经平均走了约 2.33 轮。下一 epoch 会继续剩下的队列。

第二种设置虽然仍是 30%，但因为每批固定 7A+3B，并按固定模式计算长度，一个虚拟 epoch 的总抽样次数更多。不同配置的“一次 epoch”不能当作等量训练。相同 step 下若 batch size 不同，抽样总量和有效 batch 也不同。

30% 是已有示例配置；对照 Evo-RL 的 SFT + Int. 时，建议先用 50%。采样比例本身不保证解决犹豫、夹爪行为或成功率问题。比较 checkpoint 时，仍应使用一致的任务评估条件。

## 7. 常用命令

在项目根目录执行，按主入口准备 `pi05-training` 环境。以下以普通、完整的 policy checkpoint 为例；步数按干预曝光预算计算，不代表已验证的最佳停止点。

### 7.1 从旧模型开始一轮纠错微调：对齐 Evo-RL 的 SFT + Int. loss

以下按已核对的 17,375 个干预帧、单卡 batch=32 计算，目标为 2.5 个干预起点曝光 epochs。从采集这批干预时使用的 SFT1 checkpoint 开始。

```bash
uv run --locked --extra pi05-training lerobot-train \
  --policy.path=/path/to/base_run/checkpoints/020000/pretrained_model \
  --dataset.repo_id=block/shape-matching_89_with_dagger_30 \
  --dataset.root=/path/to/shape-matching_89_with_dagger_30 \
  --use_pretrained_stats=true \
  --intervention_sampling_ratio=0.5 \
  --sample_weighting=null \
  --policy.mask_action_padding_loss=true \
  --policy.rtc_training_max_delay=0 \
  --policy.optimizer_lr=2.5e-6 \
  --policy.scheduler_decay_lr=2.5e-7 \
  --policy.scheduler_warmup_steps=200 \
  --policy.scheduler_decay_steps=2715 \
  --batch_size=32 \
  --accelerator.gradient_accumulation.steps=1 \
  --seed=1000 \
  --resume=false \
  --steps=2715 \
  --save_freq=500 \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --output_dir=outputs/train/dagger_round_2
```

新一轮 `resume=false` 需要使用新的输出目录。模型从旧权重开始，优化器、调度器和训练步数重新开始。

200 步 warmup 沿用公开 D3 干预实验；本地总长度较短，其占比更高。这里对齐的是 loss、混合比例和干预曝光预算，不是完整数值复现：公开 D3 使用 8 卡、全局 batch=64、FP32 且关闭 AMP；上面命令单卡 batch=32，dtype 沿用原 checkpoint（首轮命令记录为 BF16）。直接对照 Evo-RL 时关闭训练 RTC；若要继续训练 RTC，需要单独标记该实验。

### 7.2 在上面的命令中切换采样设置

| 想要的行为 | 替换参数 |
| --- | --- |
| 每批严格 7A+3B | `--batch_size=10 --intervention_sampling_ratio=0.3` |
| 每批严格 4A+4B | `--batch_size=8 --intervention_sampling_ratio=0.5` |
| 数据自然比例，保留旧统计量 | `--intervention_sampling_ratio=null`，保留 `--use_pretrained_stats=true` |
| 改用当前数据集统计量 | `--use_pretrained_stats=false`，保留采样参数 |

每项是替换原参数，不是在同一条命令中保留两个相互冲突的值。从保存的配置开始时，显式传 `null` 可清除其中已有的采样比例。

改变 batch、干预比例或数据量后，必须按第 11 节重新计算 `steps`，并同步修改 `policy.scheduler_decay_steps`；不能继续照抄 2715。

### 7.3 中断后继续同一轮训练

```bash
uv run --locked --extra pi05-training lerobot-train \
  --config_path=/path/to/dagger_round_2/checkpoints/001000/pretrained_model/train_config.json \
  --resume=true
```

这时从 checkpoint 的训练配置恢复，不必再把新微调的所有参数重新写一遍。`steps` 是这轮训练的目标总步数，不是额外追加步数；例如从 step=1000 恢复到目标 steps=2715，是再训练 1715 个循环 step。

要恢复相同采样序列，需要保持 sampler 版本、数据及其顺序/标记、筛选条件、seed、ratio、每卡 batch size 和数据并行进程数相同。代码在 checkpoint 有对应记录时会拒绝改变 batch size 或数据并行规模的比例采样 resume；它不会自动检查上述所有条件，也不承诺 GPU 数值结果逐位相同。

新版组内连续采样改变了旧版本的样本顺序。继续旧 checkpoint 可以使用新采样规则，但不能称为复现旧版本本来会取到的后续样本。从旧模型开始新一轮 DAgger 微调，使用 `policy.path` 加 `resume=false`。

## 8. 数据要求和启动日志

比例采样要求：非 streaming 数据集；`intervention` 已声明并保存为 `dtype=bool, shape=(1,)`；经过训练/验证划分、episode 筛选和支持的帧过滤后，两组都非空；`0 < ratio < 1`，batch size 为正数。

合并原示范时，如果缺少标记，可以用数据合并工具的 `fill_missing_intervention=true` 填充 `False`。这是合并步骤的设置，不是 `lerobot-train` 参数。只有干预组的数据无法使用这个双组采样器；如果本来只打算训练单一组，关闭比例采样即可。

启动时应能看到：

```text
Loading normalization statistics from pretrained policy: ...
Intervention sampling (probabilistic): ... non-intervention frames, ... intervention frames.
Within each group: shuffle without replacement, reshuffle only on exhaustion; unfinished group cycles continue across epochs.
Per-sample probabilities: 0.7 non-intervention, 0.3 intervention ...
```

固定配额模式则会显示每批的两组样本数。`use_pretrained_stats=false` 不会打印上面的旧统计量提示。评估 dataloader 不使用这个比例采样器，保留评估数据本来的分布。

## 9. 对应代码

- [训练配置和参数校验](../configs/train.py)：`TrainPipelineConfig`。
- [训练入口](../scripts/lerobot_train.py)：`_make_processor_kwargs` 选择统计量来源，`make_dataloaders` 创建 sampler 和恢复采样位置。
- [采样器](../datasets/sampler.py)：`EpisodeAwareSampler`、`InterventionBalancedSampler`、`compute_sampler_state`。
- [归一化公式](../processor/normalize_processor.py)：各类统计量的实际变换。
- [训练日志进度](../utils/logging_utils.py)：`MetricsTracker` 中 `epch` 的计算。
- [项目通用训练指南](../../../AGENT_GUIDE.md)：通用策略选择、训练和评估说明。

## 10. Loss 来源与 padding mask

`sample_weighting` / RA-BC 是 LeRobot 官方的可选训练接口。本地 `utils/sample_weighting.py` 与 Git HEAD 一致，可追溯到上游提交 `8a3d64033fc0d11b0ad90fffa133700d859ebc9c`（Reward models refactor，#3142）；训练入口的加权分支也已存在于上游历史中。本次没有新增或启用样本权重。[LeRobot 官方接口](https://github.com/huggingface/lerobot/blob/main/src/lerobot/utils/sample_weighting.py)、[官方训练入口](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_train.py)

Evo-RL 发布的 D3 首轮、D3 干预轮、full558 干预轮 `train_config.json` 均为 `sample_weighting: null`。SFT + Int. 通过改变抽样比例增加干预曝光，loss 仍为 flow-matching MSE；公开 D3 额外接入时间 padding mask。[D3 首轮配置](https://huggingface.co/Elvinky/pi05-piperx-3h-seed1000-fp32/blob/2ba7d1689a2ecce55477c9bf58b7af701144849d/train_config.json)、[D3 干预配置](https://huggingface.co/Elvinky/pi05-piperx-3h-intervention-1to1-mask-8884s-seed1000-fp32/blob/c1898a9ff5b3350aefa7fd0a7cfcc0528498dae0/train_config.json)、[full558 干预配置](https://huggingface.co/Elvinky/pi05-piperx-full558-intervention-1to1-mask-10822s-seed1000-fp32/blob/eb7a43340ed2b17aedccba76a4f5c3194fa52e4f/train_config.json)

本次新增的 `--policy.mask_action_padding_loss=true` 使用下式，分子和分母都排除 padding，且仅计算真实机器人动作维度：

```text
valid[b,t] = NOT action_is_pad[b,t]
loss = sum(valid[b,t] * mse[b,t,d]) / (sum(valid[b,t]) * real_action_dim)
```

这是本卡 batch 内有效动作元素的平均，与 [Evo-RL 公开补丁](https://huggingface.co/Elvinky/pi05-piperx-3h-intervention-1to1-mask-8884s-seed1000-fp32/blob/c1898a9ff5b3350aefa7fd0a7cfcc0528498dae0/reproducibility/intervention_training.patch) 的默认 `mean` 一致。保留的 `reduction="none"` 仅供 LeRobot 显式启用样本加权时使用；普通 SFT 不调用此路径。

**使用 `sample_weighting=null`，不要用 `type=uniform` 代替。** 后者仍会先在每个样本内平均，再平均这些样本；不同样本有效长度不等时，与上面的有效元素均值不同。例如两条样本的有效时间步分别为 1 和 2，各自元素均值为 2.5 和 21.5，则默认 loss 为 `(2.5 + 2×21.5)/3 = 15.1667`，逐样本等权平均为 12。也不要因为已经按 1:1 采样，再额外乘一次干预权重。

本地 mask 开关默认 `false`，保证旧配置不悄悄改变 loss；第二轮命令显式开启。启用后缺失 `action_is_pad` 会报错；全被 mask 的项返回零损失、零梯度。`loss_per_dim` 使用相同有效区域。若同时启用训练 RTC，则有效区域为 `非 padding AND 非 RTC 前缀`；直接对照 Evo-RL 的这轮 SFT 使用 `rtc_training_max_delay=0`。

调研时 LeRobot 上游 Pi0.5 仍没有这项时间 padding 开关，因此这里采用的是“LeRobot flow-matching + Evo-RL 发布补丁的 padding reduction”。两者不能笼统说成完全相同的默认 loss。源码和训练命令记录未发现本项目主动开启 `sample_weighting`；本机没有找到已保存的实际训练 `train_config.json`，所以这不等于确认远端每一次历史运行都未开启。

验证覆盖有效元素分母、padding 梯度、真实动作维度、RTC 联合 mask、配置保存/加载、预处理字段保留，以及普通训练入口实际参数更新。相关测试：[Pi0.5 padding loss](../../../tests/policies/pi0_pi05/test_pi05_padding_loss.py)。

## 11. 本地数据量与 2.5 epochs 的步数

2026-10-16 直接读取本机 `/home/cpy/.cache/huggingface/lerobot/block` 下的 metadata 和 Parquet 核对：

| 数据 | episodes / 连续段 | 帧数 | 30 FPS 时长 |
| --- | ---: | ---: | ---: |
| `shape-matching_89` 原示范 | 89 | 117,485 | 65 分 16 秒 |
| `dagger_merged` 纯干预 | 30 | 17,375 | 9 分 39 秒 |
| `shape-matching_89_with_dagger_30` 合并集 | 119 | 134,860 | 74 分 55 秒 |

30 个干预段全部为 `intervention=True`，没有段内交回策略的转折；长度最短 395、中位数 542、最长 971 帧，没有不足 50 帧的段。这与“接管持续到任务结束”的当前采集方式一致。实际视频内容和任务成败没有通过本次计数审计确认。

即便整段长于 chunk，靠近尾部采样仍会 padding。在 H=50、所有起点均匀取样时，干预池约 4.23% 的动作时间位置为 padding，原示范约 1.86%，1:1 混合约 3.04%。因此 mask 是应补齐的实现差异，但现有数据不支持“极短片段的大量 padding 是退化主因”这一结论。

本项目 `steps` 计 micro-batch 循环次数。设 `B_loop = 每卡batch × 数据并行进程数`，`p` 为干预比例，`N_int` 为实际可采样的干预起点数：

```text
干预曝光 epochs = steps × B_loop × p / N_int
steps = ceil(2.5 × N_int / (B_loop × p))
```

这里是起点采样曝光，不是每个动作作为 chunk 内目标出现的次数；也不是混合数据集日志中的 `epch`。有筛选或验证集时，先从 N_int 中扣除对应帧。有梯度累积时不要再给 B_loop 乘累积次数；若需要完整更新结束后停止，把 steps 向上对齐到累积次数的倍数。

对当前 N_int=17,375：

| B_loop | p=0.5 的 steps | p=0.3 的 steps（期望曝光） |
| ---: | ---: | ---: |
| 8 | 10,860 | 18,099 |
| 16 | 5,430 | 9,050 |
| 32 | **2,715** | **4,525** |
| 64 | 1,358 | 2,263 |

batch=32、p=0.5 每步严格 16 个干预，2715 步抽取 43,440 个干预起点，对应 2.50014 epochs；同样抽取 43,440 个原示范起点，只额外回放约 0.370 个示范 epochs。

首轮命令记录为单卡、batch=32、无梯度累积、20,000 steps。按全部 117,485 帧都参与训练计算：`20000 × 32 / 117485 = 5.4475`，即约 **5.45 个示范曝光 epochs**。这高于论文基线的 2.5，但单凭 epoch 数不能判断过拟合；实际 run 若有筛选或不同 batch，应以其保存配置重算。

建议先用当前数据完成一次上述对照，再以有效人工控制量 **30 分钟**为第一采集目标；若主要失败类型仍缺少多样纠正，再扩到 **60 分钟**。这是本项目的分阶段工程建议，不是论文证明的最低数据量：

| 干预池目标 | 总帧数（30 FPS） | 比当前新增 | 单卡 batch=32、p=0.5、2.5 epochs 的 steps |
| --- | ---: | --- | ---: |
| 当前 | 17,375 | — | 2,715 |
| 30 分钟 | 54,000 | 36,625 帧，约 20 分 21 秒 | 8,438 |
| 60 分钟 | 108,000 | 90,625 帧，约 50 分 21 秒 | 16,875 |

优先补不同初始位置、物体姿态、失败阶段及恢复方式，避免仅靠人工一直完成简单尾段来凑时长。Evo-RL D3 的约 63 分钟分为 748 个短人工段，本地是 30 个平均约 19 秒的长段；两者 episode 数不能直接比较，独立失败恢复的覆盖也需单独记录。新增数据应来自实际待改进策略，并保留来源 checkpoint 与结果标签。

## 12. 当前采样器与 Evo-RL 的差异

| 项目 | Evo-RL 公开 D3 补丁 | 本项目 |
| --- | --- | --- |
| 两个数据池 | 原示范 A + 单独提取的纯人工片段 B | 按帧的 `intervention=False/True` 分组 |
| 组间比例 | A/B 交替；偶数 batch 下严格 1:1 | batch×ratio 为整数时固定配额，否则按概率选择组 |
| 组内抽样 | shuffle 后不放回，耗尽重洗 | 相同，额外保留跨虚拟 epoch 尚未取完的组内顺序 |
| 比例与恢复 | 该补丁固定 1:1 | 任意合法比例，并支持确定的采样位置恢复 |

对当前合并集，False 池恰好是原示范，True 池恰好是纯人工段；在 batch=32、ratio=0.5 时，两者主要训练语义相同。本项目实现更适合继续维护，建议保留，先用 p=0.5 对齐参考配置，再固定干预曝光量比较 p=0.3。没有依据认为换成它的交替索引顺序就会提升成功率。固定配额能保证每批来源比例，但不保证比概率混合获得更高最终成功率。

若以后合并完整自主 rollout，False 池会含策略动作，便不再等于专家示范；人工起点也可能产生跨回交边界的 chunk。应先导出纯人工连续段作为训练 episode，保持原示范池独立。采样比例只决定起点来源，不会自动做来源过滤或 loss 加权。[采样与 loss 复现补丁](https://huggingface.co/Elvinky/pi05-piperx-3h-intervention-1to1-mask-8884s-seed1000-fp32/blob/c1898a9ff5b3350aefa7fd0a7cfcc0528498dae0/reproducibility/intervention_training.patch)

## 13. 是否支持一次 rollout 内多次接管

当前“接管直到任务结束”是有效的采集方式。由于每次人工窗口单独保存为 episode，只需记录人工起点和 episode 结束，不必先实现多次切换才能训练；末尾 padding mask 仍然需要。

为了减少重复复位和人工完成容易尾段，建议后续支持“策略 → 人工纠正 → 同一现场交回策略 → 必要时再次接管 → 最终结果”。当机器人已回到策略能够继续的状态，就交回策略；任务结束或无法继续时仍正常结束 rollout，不必刻意凑多次接管。

目前 `robot_client_dagger.py` 的 `stop_correction()` 保存后调用 `reset_rollout()`，并没有同一 rollout 的恢复自治入口。本次仅完成 loss 集成，未改变硬件接管状态机。后续实现时，应保持物理 rollout 身份，给每个人工段单独记录边界；恢复自治时用现场新观测发起推理并清空过期动作/历史缓存。DAgger 可分别保存人工段，RECAP 保留完整轨迹和最终结果，再从成功轨迹导出人工训练片段。不能仅去掉一次 reset 调用，就让旧动作队列继续执行。
