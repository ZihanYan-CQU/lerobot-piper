# 文档已归类

本页内容已移到 [PI0.5 / A100 普通推理延迟排查](../troubleshooting/pi05_inference_latency_2026-10-15.md)。原始日期、诊断证据与验证边界保留在新位置。

旧章节索引：

<a id="1-从日志确定瓶颈范围"></a>

- [1. 从日志确定瓶颈范围](../troubleshooting/pi05_inference_latency_2026-10-15.md#1-从日志确定瓶颈范围)

<a id="2-bf16-配置没有覆盖整个模型"></a>

- [2. BF16 配置没有覆盖整个模型](../troubleshooting/pi05_inference_latency_2026-10-15.md#2-bf16-配置没有覆盖整个模型)

<a id="3-无动作前缀仍需完成完整模型采样"></a>

- [3. 无动作前缀仍需完成完整模型采样](../troubleshooting/pi05_inference_latency_2026-10-15.md#3-无动作前缀仍需完成完整模型采样)

<a id="4-可以排除或需要纠正的解释"></a>

- [4. 可以排除或需要纠正的解释](../troubleshooting/pi05_inference_latency_2026-10-15.md#4-可以排除或需要纠正的解释)

<a id="5-现有阶段日志的计时限制"></a>

- [5. 现有阶段日志的计时限制](../troubleshooting/pi05_inference_latency_2026-10-15.md#5-现有阶段日志的计时限制)

<a id="6-已添加的离线诊断脚本"></a>

- [6. 已添加的离线诊断脚本](../troubleshooting/pi05_inference_latency_2026-10-15.md#6-已添加的离线诊断脚本)

<a id="7-首轮诊断的验证边界"></a>

- [7. 首轮诊断的验证边界](../troubleshooting/pi05_inference_latency_2026-10-15.md#7-首轮诊断的验证边界)

<a id="8-2026-10-16bf16-选项之外日志揭示的优化顺序"></a>

- [8. 2026-10-16：BF16 选项之外，日志揭示的优化顺序](../troubleshooting/pi05_inference_latency_2026-10-15.md#8-2026-10-16bf16-选项之外日志揭示的优化顺序)

<a id="81-优先解释持续升高的延迟而不只看平均值"></a>

- [8.1 优先解释持续升高的延迟，而不只看平均值](../troubleshooting/pi05_inference_latency_2026-10-15.md#81-优先解释持续升高的延迟而不只看平均值)

<a id="82-预处理的-24-ms-是第二个有量级的目标"></a>

- [8.2 预处理的 24 ms 是第二个有量级的目标](../troubleshooting/pi05_inference_latency_2026-10-15.md#82-预处理的-24-ms-是第二个有量级的目标)

<a id="83-冷启动与稳定运行分开处理"></a>

- [8.3 冷启动与稳定运行分开处理](../troubleshooting/pi05_inference_latency_2026-10-15.md#83-冷启动与稳定运行分开处理)

<a id="84-rtc-的条件预算可能影响响应新观测的速度"></a>

- [8.4 RTC 的条件预算可能影响响应新观测的速度](../troubleshooting/pi05_inference_latency_2026-10-15.md#84-rtc-的条件预算可能影响响应新观测的速度)

<a id="85-暂时不要优先改动亚毫秒阶段"></a>

- [8.5 暂时不要优先改动亚毫秒阶段](../troubleshooting/pi05_inference_latency_2026-10-15.md#85-暂时不要优先改动亚毫秒阶段)
