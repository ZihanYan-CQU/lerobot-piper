# RECAP 异常与恢复

[排错目录](README.md) · [RECAP 采集](../workflows/recap/README.md) · [数据约定](../architecture/recap.md)

## 异常与恢复

退出会先停止 CAN 和机械臂，再清理未确认的缓冲区及临时图像；采集故障会停止策略，并在退出时丢弃本轮。已经按 S/F 提交结果的一轮继续完成保存，Esc 不会把它改为丢弃。明确标注的任务失败仍是有效数据。

sidecar 经过 `recording → saving → committed`，原子替换 JSON。这三个字段值是内部写盘阶段，不是任务结果或训练标签。正常丢弃会删除该轮临时 sidecar。保存失败不自动重试、不覆盖下一 episode，保留日志供检查，避免误删已写入的数据。追加时先校验 sidecar 状态、episode 对应关系和 schema；有未完成写盘记录就拒绝追加。不要只删元数据或手工把状态改成 committed，应先核对数据/视频/episode 元数据并修复或隔离损坏的数据集。

这不是断电恢复系统。尚未保存的数值帧仍在内存，`kill -9`、掉电或底层硬件阻塞不能保证保全；日志用于发现不完整记录。正常 `committed` 数据也应在程序完成 finalize 后再用于训练。`recap_dataset_push_to_hub` 默认 false，开启时按原 dataset 上传逻辑携带 `meta/recap`。

原有 merge/edit 工具尚未适配这些 sidecar。不要直接用它们合并、删除或重排 RECAP episode 后继续训练；下一阶段导出器需同步重写 episode 映射、过滤无效动作，并按确认的任务边界和成功/失败结果构造回报。
