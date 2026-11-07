# 真机部署文档导航

原混合文档已按职责拆分。日常操作请直接查看 [真机部署流程指令](../../../docs/workflows/deployment/README.md)。

- [实现与代码架构](../../../docs/architecture/deployment.md)
- [问题定位与诊断](../../../docs/troubleshooting/deployment.md)
- [PI05 图像、精度与预热](../../../docs/architecture/pi05-inference.md)
- [性能实验与历史记录](../../../docs/troubleshooting/pi05-performance.md)
- [全部文档](../../../docs/README.md)

## 原章节去向

<a id="1-系统解决什么问题"></a>

- [1. 系统解决什么问题](../../../docs/architecture/deployment.md#1-系统解决什么问题)

<a id="2-从哪些文件读起"></a>

- [2. 从哪些文件读起](../../../docs/architecture/deployment.md#2-从哪些文件读起)

<a id="3-线程队列与一次正常请求"></a>

- [3. 线程、队列与一次正常请求](../../../docs/architecture/deployment.md#3-线程队列与一次正常请求)

<a id="4-动作步插值占用与请求-id"></a>

- [4. 动作步、插值占用与请求 ID](../../../docs/architecture/deployment.md#4-动作步插值占用与请求-id)

<a id="5-guided-rtc什么时候请求如何生成如何合并"></a>

- [5. Guided RTC：什么时候请求、如何生成、如何合并](../../../docs/architecture/deployment.md#5-guided-rtc什么时候请求如何生成如何合并)

<a id="6-线性插值与-piper-ema"></a>

- [6. 线性插值与 Piper EMA](../../../docs/architecture/deployment.md#6-线性插值与-piper-ema)

<a id="7-trained-rtc训练端到底改变了什么"></a>

- [7. Trained RTC：训练端到底改变了什么](../../../docs/architecture/deployment.md#7-trained-rtc训练端到底改变了什么)

<a id="8-trained-rtc远程部署的独立路径"></a>

- [8. Trained RTC：远程部署的独立路径](../../../docs/architecture/deployment.md#8-trained-rtc远程部署的独立路径)

<a id="9-模型与传输层的性能处理"></a>

- [9. 模型与传输层的性能处理](../../../docs/architecture/deployment.md#9-模型与传输层的性能处理)

<a id="10-重试复位和故障"></a>

- [10. 重试、复位和故障](../../../docs/architecture/deployment.md#10-重试复位和故障)

<a id="11-参数应放在哪一端"></a>

- [11. 参数应放在哪一端](../../../docs/architecture/deployment.md#11-参数应放在哪一端)

<a id="12-用命令片段对应代码路径"></a>

- [12. 用命令片段对应代码路径](../../../docs/architecture/deployment.md#12-用命令片段对应代码路径)

<a id="13-从现象定位代码"></a>

- [13. 从现象定位代码](../../../docs/troubleshooting/deployment.md#13-从现象定位代码)

<a id="14-云端延迟抖动的试验设置"></a>

- [14. 云端延迟抖动的试验设置](../../../docs/troubleshooting/pi05-performance.md#14-云端延迟抖动的试验设置)

<a id="15-π05-图像上传float32-无损压缩--openpi-uint8-可选"></a>

- [15. π0.5 图像上传：float32 无损压缩 / OpenPI uint8 可选](../../../docs/architecture/pi05-inference.md#15-π05-图像上传float32-无损压缩--openpi-uint8-可选)

<a id="16-持续采集发请求时选择最新快照"></a>

- [16. 持续采集，发请求时选择最新快照](../../../docs/architecture/deployment.md#16-持续采集发请求时选择最新快照)

<a id="17-π05-推理加速与延迟稳定后续优化建议"></a>

- [17. π0.5 推理加速与延迟稳定：后续优化建议](../../../docs/troubleshooting/pi05-performance.md#17-π05-推理加速与延迟稳定后续优化建议)

<a id="18-lerobot--openpi-bf16-与编译部署选项"></a>

- [18. LeRobot / OpenPI BF16 与编译部署选项](../../../docs/architecture/pi05-inference.md#18-lerobot--openpi-bf16-与编译部署选项)

<a id="19-重连时的段错误与-cuda-非法内存访问"></a>

- [19. 重连时的段错误与 CUDA 非法内存访问](../../../docs/troubleshooting/deployment.md#19-重连时的段错误与-cuda-非法内存访问)

<a id="20-正常推理时关闭诊断开销"></a>

- [20. 正常推理时关闭诊断开销](../../../docs/troubleshooting/deployment.md#20-正常推理时关闭诊断开销)

<a id="21-bf16--openpi--guided-rtc-的部署开销复查"></a>

- [21. BF16 / OpenPI / guided RTC 的部署开销复查](../../../docs/troubleshooting/pi05-performance.md#21-bf16--openpi--guided-rtc-的部署开销复查)
