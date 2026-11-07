# DAgger 排错：接管、记录与训练衔接

[排错索引](README.md) · [DAgger 指令](../workflows/dagger/README.md) · [状态机与数据语义](../architecture/dagger.md)

这里集中现象与代码定位。共用请求/复位问题见 [部署排错](deployment.md)，合并和数据工具问题见 [工具排错](tools.md)，采样和归一化问题见 [训练排错](training.md)。

<a id="14-调试与代码阅读路线"></a>

## 调试与代码阅读路线

| 现象 | 对应函数 / 状态 |
| --- | --- |
| 右键不启动 | `start_rollout()`：必须 READY、非 FAULT、无保存/复位/停止信号 |
| 右键“丢了” | `submit_operator_event()` 的 busy 拒绝，或 `_handle_operator_event()` 的 generation/状态失配 |
| 暂停后旧动作仍返回 | 返回允许发生，检查 `pause_autonomy()` 的 generation 失效和结果接纳阶段检查 |
| 接管夹爪不对齐 | `enter_correction()`、`wait_for_master_alignment()`，区分可恢复拒绝与恢复失败 |
| 人在动主臂但从臂不跟 | CAN 桥 enabled/health、物理 `is_rx`、控制帧过滤及当前 `PiperControlMode` |
| 纠正动作有了但没数据 | `dagger_dataset_repo_id`、recorder active、`last_error`、窗口是否实际保存 |
| 模型似乎没学到纠正 | 合并标记、干预比例、实际训练步数、保存统计量、加载的 checkpoint 会话 |
| trained 连续拒绝 | `rtc_trained.py` 的预算和实际消耗，不调 guided 自适应参数 |
| 保存时很久没反应 | 视频编码/数据 finalize；先区分硬件是否已停，不等于 GPU 推理卡住 |
| 复位失败后无法开始 | `read_reset_feedback()` 双臂物理反馈、reset 容差和 settle 条件 |

## 合并前先处理数据完整性

合并会先检查所有源数据的完整性，再创建输出目录。遇到 `Parquet has ... rows but metadata declares ... frames` 或 `interrupted episode save`，先定位异常源数据；调整采样比例或补 `intervention` 不能修复索引。原示范与纠正数据的其他特征、FPS 和机器人类型仍须兼容。

确认每次人工窗口实际保存后，再检查合并后的 `intervention` 标记、训练有效起点数、实际训练配置与部署加载的 checkpoint。只有 `dagger_dataset_repo_id` 已设置且记录器成功保存，接管动作才会成为训练数据。
