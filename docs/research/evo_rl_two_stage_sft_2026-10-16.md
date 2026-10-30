# Evo-RL 前两轮 SFT：数据、训练参数及本项目差异

调研日期：2026-10-16。范围：首次专家示范 SFT，以及从该模型继续进行的「专家示范 + 人工干预」SFT。依据为论文、项目数据页关联的 Elvinky 公开模型、随模型发布的训练配置/复现补丁，以及本项目当前工作区源码。未进行实机效果验证。

最值得本项目先对齐的，是**连续干预片段边界、时间 padding 的 loss mask、第二轮降低学习率，以及按干预池曝光量控制训练长度**。本项目已具备 1:1 起点采样；采样比例只是配方的一部分。

同日实现更新：已接入 `policy.mask_action_padding_loss`，默认关闭、第二轮显式开启；默认标量 loss 对齐下述公开补丁的有效元素均值。LeRobot 原有的可选样本加权接口保留，但本轮配方明确使用 `sample_weighting=null`，没有新增 RA-BC 或干预 loss 权重。本地实际数据审计、训练命令、曝光步数和多次接管建议见 [训练 README 第 7、10–13 节](../../src/lerobot/yzh/README_TRAINING.md)。

## 1. 先区分三类证据

- **论文协议**：[Evo-RL v1，2026-10-09](https://arxiv.org/html/2610.13345v1)。描述成功轨迹过滤、人工连续片段和迭代方式，但不提供完整的前两轮优化器配置表。
- **公开实验记录**：主数据页关联了 [D3 干预模型](https://huggingface.co/Elvinky/pi05-piperx-3h-intervention-1to1-mask-8884s-seed1000-fp32)。其 `train_config.json`、复现补丁和数据清单能补足很多训练细节。以下精确超参数主要取自这些记录，不能宣称每个公开 checkpoint 都已与论文表格中的某一行一一对应。
- **GitHub 通用示例**：调研时 [main 的固定快照](https://github.com/MINT-SJTU/Evo-RL/blob/25a23467d2c80092889d9408abb15cb3d605fef6/README.md) 仍展示 ACP 文本标签、`steps=30000`、BF16 等模板。它与下面发布模型使用的定制 LeRobot 0.6.1 配方有差异，不能直接作为论文的精确复现参数。

## 2. 论文对第二轮 SFT 的要求

论文 §III-B 定义：保留最终成功轨迹中由人控制的帧，并保持连续片段边界；训练动作 chunk 限于同一人工控制片段。原专家示范继续参与训练，人工接管前后的策略动作不直接进入纠错模仿池。

§V-D 的消融实验从同一个训练了 2.5 epochs 的 SFT 模型开始，加入约一小时人工干预，干预采样概率为 0.5。PiperX 的 attempt success rate 从 SFT 的 22.73% 提高到 SFT + Int. 的 78.72%。这是尝试级指标，不是完整五次插入任务的成功率；也不能作为本项目效果保证。[论文 §III-B、§V-B、§V-D](https://arxiv.org/html/2610.13345v1)

这里的第二轮仍使用监督式动作学习目标；后续基于 value/advantage 的数据筛选是另一个阶段。先复现 SFT + Int.，无需先完成 value 模型。

## 3. 人工干预究竟有多少

下表来自公开模型卡，时长按帧数 / 30 FPS 计算。D1/D3/D7 表示第一轮的专家示范预算。

| 公开实验 | 初始专家示范 | 人工控制帧 | 人工控制时长 | 连续干预段 | 首轮训练步数 | 第二轮新增步数 |
|---|---|---:|---|---:|---:|---:|
| D1 | 54 episodes，107,322 帧，约 0.994 h | 103,237 | 57 分 21 秒 | 557 | 4,193 | 8,066 |
| D3 | 149 episodes，323,507 帧，约 2.995 h | 113,713 | 1 时 3 分 10 秒 | 748 | 12,637 | 8,884 |
| D7 | 336 episodes，754,251 帧，约 6.984 h | 115,937 | 1 时 4 分 25 秒 | 831 | 29,463 | 9,058 |
| full558 系列 | 558 episodes，1,255,729 帧，约 11.627 h | 138,514 | 1 时 16 分 57 秒 | 616 | 50,000 | 10,822 |

来源：[D1 模型卡](https://huggingface.co/Elvinky/pi05-piperx-1h-intervention-1to1-mask-8066s-seed1000-fp32)、[D3 模型卡](https://huggingface.co/Elvinky/pi05-piperx-3h-intervention-1to1-mask-8884s-seed1000-fp32/blob/c1898a9ff5b3350aefa7fd0a7cfcc0528498dae0/README.md)、[D7 模型卡](https://huggingface.co/Elvinky/pi05-piperx-7h-intervention-1to1-mask-9058s-seed1000-fp32)、[full558 干预模型卡](https://huggingface.co/Elvinky/pi05-piperx-full558-intervention-1to1-mask-10822s-seed1000-fp32/blob/eb7a43340ed2b17aedccba76a4f5c3194fa52e4f/README.md)。full558 原始帧数另见 [run card](https://huggingface.co/Elvinky/pi05-piperx-full558-demo-value-1to1-step1511-seed1000/blob/064a19c3bc9d926f3c24bf7a39b2184347119554/RUN_CARD.md)。

**这些是实际人工控制时间，不是打开机器人一小时，也不是一小时完整 rollout。** 例如 D3 的原始采集包含 82 条完整轨迹、463,814 帧，约 4.295 小时；人工控制占其中 24.5169%，提取后才得到约 1.053 小时的 748 个训练片段。未保存的采集、复位和等待不计入该时长。[D3 原始数据卡](https://huggingface.co/datasets/Elvinky/pi05-piperx-3h-demo-dagger-full-episodes-20260911)

D1/D3/D7 分别用各自的初始策略采集纠错，不是三种示范预算共享同一份干预。这让纠错状态对应具体策略的失败分布。约一小时是这些实验的预算，不是已经证明的最低有效数据量。

## 4. 两轮训练的精确参数：以记录最完整的 D3 为例

配置来源：[第一轮 train_config.json](https://huggingface.co/Elvinky/pi05-piperx-3h-seed1000-fp32/blob/2ba7d1689a2ecce55477c9bf58b7af701144849d/train_config.json)、[第二轮 train_config.json](https://huggingface.co/Elvinky/pi05-piperx-3h-intervention-1to1-mask-8884s-seed1000-fp32/blob/c1898a9ff5b3350aefa7fd0a7cfcc0528498dae0/train_config.json)、[第二轮实际启动参数](https://huggingface.co/Elvinky/pi05-piperx-3h-intervention-1to1-mask-8884s-seed1000-fp32/blob/c1898a9ff5b3350aefa7fd0a7cfcc0528498dae0/reproducibility/training_command.json)。

| 参数 | 第一次 SFT：专家示范 | 第二次 SFT：示范 + 干预 |
|---|---|---|
| 初始化 | `pi05_base` | 第一轮 D3 的 step 12,637 checkpoint |
| 训练数据 | D3 的 323,507 帧 | 同一 D3 示范 + 113,713 帧纯人工片段 |
| 采样 | 示范起点帧 | 每卡 4 个示范 + 4 个干预；全局 32 + 32 |
| 全局 batch | 64 | 64 |
| 每卡 batch / 卡数 | 8 / 8 | 8 / 8 |
| 训练长度 | 12,637 步，目标 2.5 个示范 epochs | 新增 8,884 步，目标 2.5 个干预曝光 epochs |
| 峰值学习率 | **2.5e-5** | **2.5e-6** |
| 最终学习率 | 2.5e-6 | 2.5e-7 |
| 调度 | cosine decay with warmup | cosine decay with warmup |
| warmup | 422 步 | 200 步 |
| decay steps | 12,637 | 8,884 |
| 优化器 | AdamW | 新建 AdamW，`resume=false` |
| betas / eps | `(0.9, 0.95)` / `1e-8` | 相同 |
| weight decay / clip norm | `0.01` / `1.0` | 相同 |
| 更新参数 | 全参数；视觉编码器不冻结；非 expert-only；非 LoRA | 相同 |
| 数值精度 | FP32，AMP 关闭 | FP32，AMP 关闭 |
| gradient checkpointing / compile | 开启 / 关闭 | 相同 |
| 随机种子 | 1000 | 1000 |
| 图像增强 | 关闭 | 关闭 |
| action chunk | 50 步；实际动作 14 维 | 相同 |
| 动作形式 | `use_relative_actions=false` | 相同 |
| 归一化 | STATE/ACTION 使用 QUANTILES | 保留第一轮 D3 的数值尺度 |
| 时间 padding loss mask | 配置未声明这一训练专用开关 | `mask_action_padding_loss=true` |
| 额外按样本加权 | `sample_weighting=null` | `sample_weighting=null`；1:1 通过 sampler 实现 |

第一轮的每卡 batch 来自训练配置，全局 batch 与 2.5 epochs 来自[第一轮模型卡](https://huggingface.co/Elvinky/pi05-piperx-3h-seed1000-fp32/blob/2ba7d1689a2ecce55477c9bf58b7af701144849d/README.md)。第二轮环境记录为 LeRobot 0.6.1、PyTorch 2.11.0+cu128、Transformers 5.5.4、Accelerate 1.14.0，8 张 RTX PRO 6000；这是该公开运行的环境，不应代替论文其他实验的硬件描述。

full558 是另一条初始化链：其 [SFT 配置](https://huggingface.co/Elvinky/pi05-piperx-full558-50k-seed1000-fp32/blob/1b6ae5f2625f2acd5fb9e5f652df5683960e9a49/config.json) 记录首轮 LR 2.5e-5 → 2.5e-6、warmup 1200、decay 50,000；[干预轮配置](https://huggingface.co/Elvinky/pi05-piperx-full558-intervention-1to1-mask-10822s-seed1000-fp32/blob/eb7a43340ed2b17aedccba76a4f5c3194fa52e4f/train_config.json) 则为 LR 2.5e-6 → 2.5e-7、warmup 200、decay 10,822。该首轮发布包未附 `train_config.json`，不能把其他实验的训练元信息无条件补到它上面。

## 5. 最容易误解的训练长度

第二轮的「2.5 epochs」以**干预池的起点采样次数**为分母。设干预帧数为 `N_int`，每次优化器更新消耗 `B` 个样本，干预比例为 `p`，目标曝光为 `E_int`：

```text
optimizer_updates = ceil(E_int * N_int / (B * p))
```

D3 的实际数字为：

```text
SFT1: ceil(2.5 * 323507 / 64) = 12637
SFT2: ceil(2.5 * 113713 / (64 * 0.5)) = 8884
```

第二轮中，专家示范的额外曝光仅为 `8884 * 32 / 323507 ≈ 0.879` 遍。full558 的干预轮相应只额外回放约 0.276 遍原示范。这里不是把整个合并数据再训 2.5 遍。[公开计划文件](https://huggingface.co/Elvinky/pi05-piperx-3h-intervention-1to1-mask-8884s-seed1000-fp32/blob/c1898a9ff5b3350aefa7fd0a7cfcc0528498dae0/reproducibility/plan.json)

**本项目的计数差异：** 当前 `lerobot_train.py` 中 `steps` 统计每卡读取的 micro-batch，而非总是统计优化器更新。无梯度累积时二者相同；有累积时，要区分 `B_update = batch_size × DP数量 × accumulation` 和每个循环 step 的样本量 `B_loop = batch_size × DP数量`。本地曝光估算为：

```text
E_int ≈ loop_steps * B_loop * p / N_int
```

如需整次累积结束后停止，令 `loop_steps` 向上对齐到 accumulation 的倍数。不能把原运行的 8,884 次更新直接等同于所有本地配置的 `--steps=8884`。同时核对实际日志中的 LR 调度长度。[本地训练计数](../../src/lerobot/scripts/lerobot_train.py#L690)

## 6. 片段边界和 loss mask 比比例采样更值得先核查

公开 D3 构建脚本逐条轨迹提取 `intervention=True` 的最大连续区间，分别建立训练 episode，保留来源 episode 和帧范围。11 个不足 50 帧的片段也全部保留，最短 16 帧。[构建脚本](https://huggingface.co/Elvinky/pi05-piperx-3h-intervention-1to1-mask-8884s-seed1000-fp32/blob/c1898a9ff5b3350aefa7fd0a7cfcc0528498dae0/reproducibility/build.py)

因此需要同时满足：

1. chunk 不跨越人工段结束边界，也不把两次不连续接管拼成连续动作。
2. 不足 50 步时可以补齐，但补齐位置不参与 loss 的分子或分母。
3. 仅对真实机器人动作维度计算损失。

公开补丁的默认标量 loss 是本卡 batch 内的有效动作元素均值：

```text
valid[b,t] = not action_is_pad[b,t]
loss = sum(loss_element[b,t,d] * valid[b,t])
       / (sum(valid[b,t]) * real_action_dim)
```

它不等于先把 padding loss 置零再对完整 `B×H×D` 求平均；也不等于每条长短不同的片段先单独平均后再等权求 batch 均值。**1:1 是起点样本比例，不保证两类数据贡献完全相等的有效动作元素数。** [实际 loss 补丁](https://huggingface.co/Elvinky/pi05-piperx-3h-intervention-1to1-mask-8884s-seed1000-fp32/blob/c1898a9ff5b3350aefa7fd0a7cfcc0528498dae0/reproducibility/intervention_training.patch)

本项目读取器把越过 episode 末尾的动作索引截到最后一帧，并生成 `action_is_pad`。调研开始时 Pi0.5 未使用这个字段；同日已在 `forward` 和 `_reduce_training_rtc_loss` 中接入，通过 `policy.mask_action_padding_loss=true` 启用。[读取器](../../src/lerobot/datasets/dataset_reader.py#L301)、[Pi0.5 loss](../../src/lerobot/policies/pi05/modeling_pi05.py#L161)

举例：20 帧纠错片段，从首帧取 50 步 chunk，就会包含 30 个末动作补齐位置。不开 mask 时这些位置仍参与监督。实际核对的本地 30 个干预段最短为 395 帧，没有这种不足 50 帧的短段；H=50 时干预池尾部 padding 占比约 4.23%。是否造成当前实机退化，需要消融验证，不能仅凭源码认定。

新增 mask 已与现有 RTC 训练合并为 `有效时间位置 = 非padding AND 非RTC已知前缀`，同步处理默认标量 loss、上游可选的 `reduction='none'` 和 `loss_per_dim`。直接对照公开 SFT + Int. 使用 `rtc_training_max_delay=0`、`sample_weighting=null`；保留接口不等于启用样本加权。

## 7. 与当前工程的对应关系

| 环节 | 本项目已有能力 | 集成时需要处理的差异 |
|---|---|---|
| 1:1 起点采样 | `InterventionBalancedSampler` 在每卡 batch=8、ratio=0.5 时固定 4+4 | 可复用；没有证据表明更换采样器本身就能解决效果问题 |
| 干预片段 | DAgger 每个纠错窗口单独保存，帧标记为 true | 已有基本片段边界；保存窗口不等于确认最终任务成功 |
| 完整轨迹与结果 | RECAP 采集已有 success/failure sidecar、segment_id、action_valid | 可据此导出成功轨迹中的有效人工连续段；原始完整轨迹另供价值学习 |
| 时间 padding | DatasetReader 已生成 mask，Pi0.5 已接入可选 loss mask | 第二轮显式开启 `policy.mask_action_padding_loss=true` |
| 归一化 | 已实现 `use_pretrained_stats=true` | 第二轮显式开启，保持第一轮保存的处理器尺度 |
| 学习率 | Pi0.5 默认峰值仍为 2.5e-5 | 复现公开干预配方需显式改为 2.5e-6，不能只改采样比例 |
| 训练预算 | 已支持 steps 和 sampler 状态恢复 | 按干预曝光计算，区分 sampler 虚拟 epoch、循环 step 和优化器更新 |

对应源码：[sampler](../../src/lerobot/datasets/sampler.py#L179)、[DAgger 保存](../../src/lerobot/async_inference/client/piper_hil_dagger.py#L1291)、[RECAP 结果保存](../../src/lerobot/async_inference/client/piper_hil_recap.py#L287)、[处理器统计量选择](../../src/lerobot/scripts/lerobot_train.py#L109)、[默认学习率](../../src/lerobot/policies/pi05/configuration_pi05.py#L111)。

如果以后直接合并完整 RECAP rollout，现有 `intervention=False` 池会同时包含专家示范和自主动作；仅设置 ratio 不会把二者区分开。同时，在完整 episode 内按人工起点采样，也不会自动限制后续 chunk 始终为人工动作。需要先导出训练视图，或加入来源/片段约束。

## 8. 公开材料中需要保留的区别

**论文成功过滤与公开 D3 构建存在证据缺口。** 论文要求成功轨迹；但 D3 的 `build.py` 抽取源数据里全部人工帧，没有显式 success 条件，源数据卡也明确结构验证不保证每条轨迹成功。因此不能声称该公开模型严格验证了论文的成功过滤。建议本项目依靠已有 RECAP outcome 做明确过滤，并记录过滤前后的数据规模。[D3 数据说明](https://huggingface.co/datasets/Elvinky/pi05-piperx-3h-demo-dagger-full-episodes-20260911)、[构建脚本](https://huggingface.co/Elvinky/pi05-piperx-3h-intervention-1to1-mask-8884s-seed1000-fp32/blob/c1898a9ff5b3350aefa7fd0a7cfcc0528498dae0/reproducibility/build.py)

**项目指南原先引用的 35,999 步实验，第二个池并非纯干预。** 它混合 138,514 个人工帧和 322,261 个 value 筛选帧；全局 32 个原示范 + 32 个混合池样本。按帧均匀采样时，人工帧占全 batch 的期望约为 `0.5 × 138514 / 460775 = 15.03%`。该训练预算不能直接当作纯 50% 人工干预的 SFT2 配方。[35,999 步模型卡](https://huggingface.co/Elvinky/pi05-piperx-full558-intervention-value-1to1-step35999)、[本地引用位置](../../AGENT_GUIDE.md#L406)

**1,511 步也不是这里的第二轮人工 SFT。** 同名公开 checkpoint 明确只使用 full558 原示范及其 value 筛选子集，没有人工干预；其预算是 0.15 个筛选池曝光 epoch。论文中的训练耗时描述与此公开记录的模型身份不能直接合并解释。[1,511 步模型记录](https://huggingface.co/Elvinky/pi05-piperx-full558-demo-value-1to1-step1511-seed1000/blob/064a19c3bc9d926f3c24bf7a39b2184347119554/README.md)

**公开模型的完整加载检查不等于实机成功率验证。** D3 发布记录还注明最终保存阶段发生分布式等待超时，之后独立验证了导出权重完整性；不要把这些包视为已认证的优化器断点续训包。[D3 发布记录](https://huggingface.co/Elvinky/pi05-piperx-3h-intervention-1to1-mask-8884s-seed1000-fp32/blob/c1898a9ff5b3350aefa7fd0a7cfcc0528498dae0/README.md)

## 9. 建议本项目的第一个对照实验

以下是基于上述证据的工程建议，不是声称已经找到当前效果差的唯一原因：

1. 已核对当前数据为 17,375 干预帧、30 个连续段、约 9 分 39 秒；H=50 的尾部 padding 占比约 4.23%。后续仍需按已有结果标签分 success/failure 统计，不能从 `intervention=True` 推断成功。
2. 使用部署时实际采集干预的那个 SFT1 checkpoint；原专家数据回放，人工段独立成 episode。已有 DAgger 数据没有最终 outcome 的部分单独标识，不能补造成功标签。
3. 启用已接入的时间 padding mask，使用已有比例采样能力，先固定 `p=0.5`、`sample_weighting=null`。直接对照公开配方时 `rtc_training_max_delay=0`；训练 RTC 作为独立实验比较。
4. 对照公开配方：新建优化器，LR 2.5e-6 → 2.5e-7，目标 2.5 个干预曝光 epochs，保留旧归一化。全局 batch 64 是公开实验设置；换 batch 后重算曝光和调度，不机械复制 8,884 步。200 步 warmup 也只适合相近长度的运行。
5. 为定位原因，在同一训练预算和固定实机初始条件下比较现有方案与补齐 mask 后的方案；记录任务成功、需要接管的次数/时长和完成耗时。检查 demo 保持能力，而非只看混合训练 loss。
6. 在 SFT + Int. 基线可靠后，再引入 value/advantage 筛选，单独评估它带来的增益。其数据来源和权重需与纯干预配方分别记录。

Pi0.5 的 loss reduction 已完成集成；后续可继续完善从 `meta/recap` 导出成功人工连续片段的离线训练视图。现有采样器和统计量保留功能可以继续复用。
