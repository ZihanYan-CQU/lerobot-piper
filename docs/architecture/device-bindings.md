# 相机与 CAN 绑定工具结构

[架构索引](README.md) · [操作流程](../workflows/tools/device-bindings.md) · [工具排错](../troubleshooting/tools.md)

实现集中在 [device_bindings](../../src/lerobot/yzh/device_bindings/)，可单独复制到 Linux 电脑。
使用系统 Python / GTK 3，不依赖 LeRobot、PyTorch 或 OpenCV。

`camera_aliases.py` 按设备身份生成持久化 udev 别名；`can_port_bindings.py` 配置并验证当前 CAN 接口。
两者的生命周期不同：相机规则持久化，CAN 重启或重新插拔后需要再次激活。
`binding_checks.py` 只读核对设备身份、规则、接口名、波特率与 UP 状态，不请求系统授权。
界面在选择变化后使旧检查结果失效，并保留检查时间。

`privileged_session.py` 为同一窗口共用授权辅助进程，首次保存或激活时授权；
关闭窗口或辅助进程退出后需要重新授权。实际绑定、回滚、端口选择和提示色均见操作页。

## 文件与开发验证

文件夹内容：

```text
device_bindings/
├── start.sh                # 图形面板启动入口
├── Piper设备配置.desktop   # 当前安装位置的双击入口
├── create_launcher.py      # 生成目录内 / 桌面快捷方式
├── camera_aliases.py       # 相机别名与 RGB 筛选
├── camera_aliases_gui.py   # 相机 / CAN 两个页签
├── can_port_bindings.py    # CAN 端口绑定、校验与恢复
├── can_activate.sh         # 旧命令兼容入口，转发至同一 CAN 后端
├── privileged_session.py  # 窗口内共用的授权辅助进程
├── binding_checks.py       # 相机 / CAN 绑定状态的只读检查
├── tests/                 # 可以独立运行的回归测试
├── README.md
└── LICENSE
```

在仓库根目录先进入工具目录，再执行独立测试：

```bash
cd src/lerobot/yzh/device_bindings
/usr/bin/python3 -m unittest discover -s tests -p 'test_*.py'
```

测试不需要硬件或管理员权限，桌面交互测试默认跳过。
