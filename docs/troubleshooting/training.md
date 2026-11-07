# 训练排错：数据标记、统计量与断点恢复

[排错索引](README.md) · [训练指令](../workflows/training/README.md) · [训练架构](../architecture/training.md)

本页用于核查训练是否按预期读取数据与配置；实验预算见 [训练实验记录](../research/training-experiments.md)。

<a id="8-数据要求和启动日志"></a>

## 数据要求和启动日志

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

## 常见配置误解

| 现象或疑问 | 核查方法 |
| --- | --- |
| 模型权重加载成功但动作尺度变了 | 检查完整 processor 配置/状态，以及 `use_pretrained_stats`；加载权重本身不等于保留统计量 |
| 干预组为空或无法创建比例采样器 | 检查布尔标记、筛选与验证集划分；只有全 True 的纠正集不能直接做两组混采 |
| 每个 batch 的干预数不同 | 查看每卡 `batch_size × ratio` 是否为整数；概率模式只保证期望比例 |
| 日志 `epch` 大于 1 但某些帧尚未出现 | 日志按总曝光数折算；虚拟 epoch 和组内完整遍历是不同计数 |
| 开启 padding mask 后报缺少 `action_is_pad` | 检查数据/预处理是否保留该字段；不要将 mask 静默关掉来掩盖缺失 |
| resume 后步数理解不一致 | `steps` 是整轮目标总步数；不是从当前 checkpoint 额外训练的步数 |
| resume 的采样顺序变化 | 保持 sampler 版本、数据及顺序/标记、筛选、seed、ratio、每卡 batch 和并行规模一致 |

`sample_weighting=null` 与 uniform 权重并不等价：样本有效长度不同时，有效动作元素均值和逐样本均值不同。比例采样也不自动加 loss 权重。公式、反例及回归测试见 [训练 loss 说明](../architecture/training.md#10-loss-来源与-padding-mask)。
