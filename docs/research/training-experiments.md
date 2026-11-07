# 训练实验记录：本地数据预算与 Evo-RL 对照

[研究索引](README.md) · [训练指令](../workflows/training/README.md) · [训练机制](../architecture/training.md)

本页是 2026-10-04/16 的本地数据审计、历史实验配方和后续工程建议。数据集名称、17,375 个干预帧、2,715 steps、学习率和单卡 batch 都属于该次对照条件，不能作为新的数据或硬件配置的默认值。原文未声称这些参数已取得最佳任务效果；本次整理也未重新读取远端训练配置、复验数据或执行训练。

公开实验的固定来源和论文/模型卡区别见 [Evo-RL 前两轮 SFT 调研](evo_rl_two_stage_sft_2026-10-16.md)。日常完整命令统一维护在训练流程。

<a id="6-用当前合并数据理解比例和曝光次数"></a>

## 用当前合并数据理解比例和曝光次数

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

<a id="7-常用命令"></a>

<a id="7-历史-sft2-对照配方2026-10-16"></a>

## 历史 SFT2 对照配方（2026-10-16）

以下保留 2026-10-16 的实验命令，以便核对参数。日常启动和断点续训请使用 [训练流程](../workflows/training/README.md)。以下以普通、完整的 policy checkpoint 为例；步数按干预曝光预算计算，不代表已验证的最佳停止点。

<a id="71-从旧模型开始一轮纠错微调对齐-evo-rl-的-sft--int-loss"></a>

### 从旧模型开始一轮纠错微调：对齐 Evo-RL 的 SFT + Int. loss

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

<a id="72-在上面的命令中切换采样设置"></a>

### 在上面的命令中切换采样设置

| 想要的行为 | 替换参数 |
| --- | --- |
| 每批严格 7A+3B | `--batch_size=10 --intervention_sampling_ratio=0.3` |
| 每批严格 4A+4B | `--batch_size=8 --intervention_sampling_ratio=0.5` |
| 数据自然比例，保留旧统计量 | `--intervention_sampling_ratio=null`，保留 `--use_pretrained_stats=true` |
| 改用当前数据集统计量 | `--use_pretrained_stats=false`，保留采样参数 |

每项是替换原参数，不是在同一条命令中保留两个相互冲突的值。从保存的配置开始时，显式传 `null` 可清除其中已有的采样比例。

改变 batch、干预比例或数据量后，必须按[本地数据量与 2.5 epochs 的步数](training-experiments.md#11-本地数据量与-25-epochs-的步数)重新计算 `steps`，并同步修改 `policy.scheduler_decay_steps`；不能继续照抄 2715。

<a id="11-本地数据量与-25-epochs-的步数"></a>

## 本地数据量与 2.5 epochs 的步数

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

<a id="12-当前采样器与-evo-rl-的差异"></a>

## 当前采样器与 Evo-RL 的差异

| 项目 | Evo-RL 公开 D3 补丁 | 本项目 |
| --- | --- | --- |
| 两个数据池 | 原示范 A + 单独提取的纯人工片段 B | 按帧的 `intervention=False/True` 分组 |
| 组间比例 | A/B 交替；偶数 batch 下严格 1:1 | batch×ratio 为整数时固定配额，否则按概率选择组 |
| 组内抽样 | shuffle 后不放回，耗尽重洗 | 相同，额外保留跨虚拟 epoch 尚未取完的组内顺序 |
| 比例与恢复 | 该补丁固定 1:1 | 任意合法比例，并支持确定的采样位置恢复 |

对当前合并集，False 池恰好是原示范，True 池恰好是纯人工段；在 batch=32、ratio=0.5 时，两者主要训练语义相同。本项目实现更适合继续维护，建议保留，先用 p=0.5 对齐参考配置，再固定干预曝光量比较 p=0.3。没有依据认为换成它的交替索引顺序就会提升成功率。固定配额能保证每批来源比例，但不保证比概率混合获得更高最终成功率。

若以后合并完整自主 rollout，False 池会含策略动作，便不再等于专家示范；人工起点也可能产生跨回交边界的 chunk。应先导出纯人工连续段作为训练 episode，保持原示范池独立。采样比例只决定起点来源，不会自动做来源过滤或 loss 加权。[采样与 loss 复现补丁](https://huggingface.co/Elvinky/pi05-piperx-3h-intervention-1to1-mask-8884s-seed1000-fp32/blob/c1898a9ff5b3350aefa7fd0a7cfcc0528498dae0/reproducibility/intervention_training.patch)

<a id="13-是否支持一次-rollout-内多次接管"></a>

## 是否支持一次 rollout 内多次接管

当前“接管直到任务结束”是有效的采集方式。由于每次人工窗口单独保存为 episode，只需记录人工起点和 episode 结束，不必先实现多次切换才能训练；末尾 padding mask 仍然需要。

为了减少重复复位和人工完成容易尾段，建议后续支持“策略 → 人工纠正 → 同一现场交回策略 → 必要时再次接管 → 最终结果”。当机器人已回到策略能够继续的状态，就交回策略；任务结束或无法继续时仍正常结束 rollout，不必刻意凑多次接管。

目前 `robot_client_dagger.py` 的 `stop_correction()` 保存后调用 `reset_rollout()`，并没有同一 rollout 的恢复自治入口。本次仅完成 loss 集成，未改变硬件接管状态机。后续实现时，应保持物理 rollout 身份，给每个人工段单独记录边界；恢复自治时用现场新观测发起推理并清空过期动作/历史缓存。DAgger 可分别保存人工段，RECAP 保留完整轨迹和最终结果，再从成功轨迹导出人工训练片段。不能仅去掉一次 reset 调用，就让旧动作队列继续执行。
