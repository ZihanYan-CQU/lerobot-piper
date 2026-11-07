# Piper 真机评估

[返回流程目录](../README.md) · [真机部署](../deployment/README.md) · [完整任务记录](../recap/README.md)

本页用于比较 Piper 上不同 checkpoint 的任务表现。启动命令统一使用部署或 RECAP 流程中的配方。

## 选择评估方式

| 需要什么 | 使用流程 | 记录方式 |
| --- | --- | --- |
| 运行模型，人工观察和统计完成情况 | [远程 / 单机部署](../deployment/README.md) | 操作者记录每次任务结果、接管和异常 |
| 保存完整尝试、成功失败标签和模型身份 | [RECAP](../recap/README.md) | 按 S/F 确认结果，保存完整 episode 与 sidecar |
| 在失败阶段接管，采集再训练用的纠正片段 | [DAgger](../dagger/README.md) | 仅记录人工纠正窗口，不能单独用其 episode 数当总评估次数 |

## 按相同条件比较 checkpoint

1. 先确认 [设备绑定](../tools/device-bindings.md) 与训练时的相机名字、视角、动作频率一致；两相机旧配方和当前三相机配置不能混用。
2. 在部署配方中指定待评估的 checkpoint，记录其完整路径、RTC 模式和本次配置。远程模式的模型路径由服务器读取。
3. 使用同一任务描述、物体布置规则和判定标准，按对应流程启动、复位并开始下一次尝试。
4. 分别记录自主成功、人工辅助成功、失败及因设备异常而中断的次数，注明总尝试数。普通部署不会自动判断任务成功。
5. 更换 checkpoint 后重复相同条件，再比较成功率和常见失败阶段。需要可追溯的数据时使用 RECAP 的模型身份记录。

RECAP 的 `success` 表示整轮任务成功；人工辅助后成功也可以标为 `success`，自主与人工辅助必须结合 `intervention` 区分。未确认结果的中断任务按 [RECAP 规则](../recap/README.md)处理，不能据保存的 episode 数推断全部尝试次数。

## 评估之后

- 补充示范或纠正片段后，先 [检查数据](../datasets/README.md)，再进入 [训练](../training/README.md)。
- RECAP 的 sidecar 尚未接入普通合并 / 编辑 / 训练流程，按其专用文档处理。
- 时延、接管、复位或保存异常见 [问题排查](../../troubleshooting/README.md)。
