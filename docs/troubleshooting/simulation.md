# Piper MuJoCo 仿真排错与边界

[排错首页](README.md) · [仿真操作](../workflows/simulation/README.md) · [仿真实现](../architecture/simulation.md) · [uv / Conda 记录](uv-conda.md)

正常安装、数采、训练和推理命令统一在[仿真流程](../workflows/simulation/README.md)维护。
本页集中说明资源与显示要求、回放限制、真实主臂边界，以及 uv 环境检查。

## 模型资源与显示

默认模型目录是当前仓库的 `outputs/piper_sim/assets`，由插件源码位置推导。
使用其他资源目录时，检查 `PIPER_SIM_ASSETS` 或 `--model-path` 是否指向预期模型；
资源准备入口见[安装与首次检查](../workflows/simulation/README.md#安装与首次检查)。

`MUJOCO_GL=egl` 用于虚拟相机的离屏渲染；每个新终端都需要重新执行流程页的环境清理
和渲染设置。使用 `--viewer` 还需要可用的桌面显示。自动测试中的虚拟相机检查不等于
已经完成实际 GUI 显示验收。

## 回放误差或缺少仿真说明文件

回放默认最大允许误差为 0.002（关节 rad / 夹爪 m），超出会返回错误。
它依赖录制时的完整仿真初始状态；字段说明见[数据与回放语义](../architecture/simulation.md#数据与回放语义)。

若丢失 `simulation.json`，标准数据集仍可训练，但不能使用此确定性场景回放入口。

## 真机主臂与验证边界

只有显式选择 `--controller leader` 才导入并连接真机主臂。连接和退出会执行原主臂驱动
的初始化、回零/夹爪及断开流程；仿真回合复位不额外调用主臂的 `calibrate()`。
本插件没有力反馈。

录制前让虚拟从臂逐渐对齐主臂；无法在八秒内对齐时退出并提示检查姿态/限位，
此过程不向主臂发送对齐指令。关节方向、零位与映射参数的检查见[主臂遥操作流程](../workflows/simulation/README.md#真机主臂遥操作仿真从臂)。

测试中的真实主臂由替身代替，不连接 CAN。实际主臂联动和 GUI 显示需在设备旁验收。
这是简化的位置控制仿真，不用于验证真机 CAN 时序、MIT 力矩参数或电机温度。
异步 RTC/DAgger、仿真接触力回传、真机迁移的成功率属于后续独立验证范围。

## uv 环境检查

仿真统一使用 `uv run --project examples/piper_sim --no-sync ...`，解释器位于
`examples/piper_sim/.venv`。在仓库根目录核对：

```bash
uv run --project examples/piper_sim --no-sync python -c 'import sys; print(sys.executable); print(sys.base_prefix)'
cat examples/piper_sim/.venv/pyvenv.cfg
```

基础解释器应来自 uv 管理的 Python，`include-system-site-packages` 应为 `false`。
环境缺失时按[安装流程](../workflows/simulation/README.md#安装与首次检查)重新同步；
终端变量与旧入口退役记录见[环境整理](uv-conda.md)。

可选的 `smoke.sh` 批处理调用直接 uv CLI，使用标准 `.venv`；
与它等价、便于逐条执行的命令见[五步训练短流程](../workflows/simulation/smoke.md)。
