# RLT 运行时与 replay

本模块负责冻结 Pi0.5 特征、固定执行窗口、执行回执、replay 与完整 checkpoint；算法更新在 [`rl/algorithms/rlt_ac`](../algorithms/rlt_ac/) 中，远端服务适配在 `async_inference` 中。

- [训练、导出、离线学习与远端采集指令](../../../../docs/workflows/rlt/README.md)
- [模块边界、数据坐标与窗口约定](../../../../docs/architecture/rlt.md)
- [模型身份、checkpoint、时延与兼容限制](../../../../docs/troubleshooting/rlt.md)

`features.py` 管理冻结模型与坐标转换，`replay.py` 管理执行窗口数据，`rollout.py` 提供同步采集接口，`rtc.py` 管理远端承诺前缀与 journal，`checkpoint.py` 保存 / 恢复 learner、replay 和 RNG。同步采集的硬件连接、观测准备和奖励回调由调用者提供。
