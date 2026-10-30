# 相机与机械臂端口绑定

这是可以单独迁移的 Linux 设备绑定工具，包含 RGB 相机别名和主从臂 CAN 配置面板。不需要安装 LeRobot、PyTorch 或 OpenCV；Python 版本要求为 3.10+。

## 迁移与启动

将整个 `device_bindings` 文件夹复制到另一台 Linux 电脑的任意目录，然后在该文件夹中运行：

```bash
bash start.sh
```

也可以从任意工作目录执行 `bash /实际路径/device_bindings/start.sh`。启动脚本按自身位置查找程序，CAN 后端也使用同目录附带的激活脚本，不依赖原电脑用户名或 LeRobot 项目路径。打开面板无需先加 `sudo`，点击保存或激活时再进行系统授权。

文件夹内容：

```text
device_bindings/
├── start.sh                # 图形面板启动入口
├── camera_aliases.py       # 相机别名与 RGB 筛选
├── camera_aliases_gui.py   # 相机 / CAN 两个页签
├── can_port_bindings.py    # CAN 端口绑定、校验与恢复
├── can_activate.sh        # 随工具附带的 CAN 激活脚本
├── tests/                 # 可以独立运行的回归测试
├── README.md
└── LICENSE
```

新电脑需要安装系统依赖。以 Ubuntu 22.04 为例：

```bash
sudo apt update
sudo apt install python3 python3-gi gir1.2-gtk-3.0 udev \
  iproute2 ethtool can-utils sudo policykit-1 ffmpeg
```

请在 Linux 图形桌面的终端启动面板，并使用系统 Python。新电脑的相机编号和 USB 端口可能不同，迁移后先刷新，在两个页签重新选择对应设备并执行绑定。

下面的命令均在 `device_bindings` 文件夹内执行。

默认对应关系：

| 当前设备 | 固定名称 |
| --- | --- |
| `/dev/video4` | `/dev/video_wrist` |
| `/dev/video10` | `/dev/video_top` |

## 使用

图形面板包含 **RGB 相机** 和 **机械臂 CAN** 两个页签，共用下方启动命令。

### RGB 图形面板

在图形桌面的终端中运行（无需先加 `sudo`）：

```bash
/usr/bin/python3 camera_aliases.py --gui
```

也可以直接运行 `camera_aliases_gui.py`。面板会读取已有的相机别名，并为每个别名提供 RGB 相机下拉选择。默认包含 `video_wrist` 和 `video_top`，可修改名称、添加或移除其他别名；`预览画面` 可以帮助确认相机的位置。选好后点击 **保存绑定**，通过系统授权窗口输入管理员密码，即可写入规则并验证链接。

列表优先使用相机的 USB 接口描述识别 RGB；普通摄像头则使用 [V4L2 支持的彩色图像格式](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/vidioc-enum-fmt.html)辅助判断。深度、红外和元数据节点不会列入候选项，无法确认的节点也不会作为 RGB 相机供选择。打开面板后若插拔相机，请刷新再选择；保存时程序会再次核对所选设备的身份。

面板使用系统 GTK 3（`python3-gi`、`gir1.2-gtk-3.0`）；保存时使用 `pkexec`（`policykit-1`），预览使用 `ffplay`（`ffmpeg`）。请使用 `/usr/bin/python3` 启动，避免 Conda 环境缺少系统 GTK 库。

只在终端查看 RGB 列表：

```bash
/usr/bin/python3 camera_aliases.py --list-rgb
```

### 机械臂 CAN 面板

切换到 **机械臂 CAN**，分别为主臂和从臂选择 USB CAN 端口。默认值：

| 角色 | 接口名称 | USB 端口 | 波特率 |
| --- | --- | --- | --- |
| 主臂 | `can_master` | `1-1.2:1.0` | `1000000` |
| 从臂 | `can_follower` | `1-1.3:1.0` | `1000000` |

点击 **绑定并激活主从臂** 并完成系统授权，程序会调用同目录附带的 `can_activate.sh`，达到以下命令的效果：

```bash
bash can_activate.sh can_master 1000000 "1-1.2:1.0"
bash can_activate.sh can_follower 1000000 "1-1.3:1.0"
```

面板只列出 USB CAN 适配器，并显示当前接口名、启停状态和波特率。主从臂不能选择同一个端口。后端会核对设备身份，处理主从名称互换，并检查最终名称、波特率及 UP 状态；失败时尝试恢复原接口名称、已有波特率和启停状态，无法完整恢复时报告错误。

执行前请停止正在占用所选 CAN 接口的机械臂程序。**CAN 功能与原脚本一样只配置当前运行状态，重启或重新插拔后需要再次激活；不会自动设置开机启动。** 相机页的 udev 别名仍是持久化的。

只读检查与命令预览：

```bash
/usr/bin/python3 can_port_bindings.py --list
/usr/bin/python3 can_port_bindings.py --dry-run
```

CAN 部分沿用原脚本的系统依赖：`iproute2`、`ethtool`、`can-utils` 和 `sudo`。所有配置仅在点击相应页签的执行按钮时进行。

### 命令行绑定

在本文件夹中运行：

```bash
# 查看相机及视频节点；非采集/元数据节点不能用于此绑定
python3 camera_aliases.py --list

# 预览默认映射和规则，不修改系统
python3 camera_aliases.py --dry-run

# 安装默认映射，自动重新加载规则并验证链接
sudo python3 camera_aliases.py
```

指定其他设备或名称时，重复使用 `--map 设备=别名`：

```bash
sudo python3 camera_aliases.py \
  --map /dev/video4=/dev/video_wrist \
  --map /dev/video10=/dev/video_top
```

也可以写成 `--map 4=video_wrist --map 10=video_top`。别名以 `video_` 开头，后面支持英文字母、数字、下划线和连字符。原来的 `/dev/videoN` 仍然保留。

**每次安装会替换本程序管理的全部映射，请在同一次命令中传入所有需要保留的别名。** 不传 `--map` 时使用上面的默认两组。再次绑定前应先检查当前编号；相机重启后不必重新运行程序。

## 如何保持绑定

程序把规则写入 `/etc/udev/rules.d/99-lerobot-cameras.rules`，优先按相机序列号、USB 接口号和视频索引识别相机。相机编号变化后，别名仍指向同一台相机的同一个视频接口。RealSense 的彩色、深度和元数据节点通过接口号及索引区分。

没有唯一序列号时，默认自动改用连接位置（`ID_PATH`）；这种情况下需要保持 USB 插口和连接拓扑不变。可以明确指定：

```bash
# 必须使用唯一序列号，否则报错
python3 camera_aliases.py --dry-run --match serial

# 将角色绑定到 USB 连接位置
sudo python3 camera_aliases.py --match path \
  --map 4=video_wrist --map 10=video_top
```

程序会在写入前检查设备是否存在、是否支持视频采集、别名是否重复，以及其他规则是否直接声明了同名链接。已有的本程序规则在变更前会备份到同目录的 `.rules.bak-时间戳` 文件；不会覆盖手工维护的同名规则文件。

若旧的 `/etc/udev/rules.d/99-cameras.rules` 已定义 `video_wrist` 或 `video_top`，需要先移除其中对应的旧映射，再运行程序，避免重复定义。

规则机制参见 [udev 官方文档](https://www.freedesktop.org/software/systemd/man/latest/udev.html)。程序创建的是软链接，不会修改设备访问权限。

## 检查结果

```bash
ls -l /dev/video_wrist /dev/video_top
```

LeRobot 的 `index_or_path` 继续填写 `/dev/video_wrist` 或 `/dev/video_top` 即可。`--list` 和 `--dry-run` 只读取信息；实际安装需要在连接相机的 Linux 主机上运行并使用 `sudo`。

不需要硬件或管理员权限的回归测试（桌面交互测试默认跳过）：

```bash
/usr/bin/python3 -m unittest discover -s tests -p 'test_*.py'
```
