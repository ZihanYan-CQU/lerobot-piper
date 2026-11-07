# 相机与机械臂端口绑定

[返回工具导航](README.md) · 下一步：[遥操作与数采](../recording/README.md) / [真机部署](../deployment/README.md)

这是可以单独迁移的 Linux 设备绑定工具，包含 RGB 相机别名和机械臂 CAN 配置面板，可单独配置主臂、单独配置从臂或同时配置两者。不需要安装 LeRobot、PyTorch 或 OpenCV；Python 版本要求为 3.10+。

## 迁移与启动

在仓库根目录先运行：

```bash
cd src/lerobot/yzh/device_bindings
```

**除明确注明其他工作目录的段落外，本页命令均在此目录执行。** 也可将整个 `device_bindings` 文件夹复制到其他 Linux 电脑，再进入该文件夹运行：

```bash
bash start.sh
```

日常使用可双击桌面的 **Piper 设备配置** 打开。目录移动或复制到新电脑后，在 `device_bindings` 内运行以下命令，快捷方式会记录新的实际位置，无需 sudo：

```bash
/usr/bin/python3 create_launcher.py
/usr/bin/python3 create_launcher.py --desktop
```

第一条生成目录内的 `Piper设备配置.desktop`，第二条创建当前用户的桌面快捷方式。如果桌面提示未信任，请右键图标并选择 **允许启动**。

也可以从任意工作目录执行 `bash /实际路径/device_bindings/start.sh`。启动脚本按自身位置查找程序，不依赖原电脑用户名或 LeRobot 项目路径。打开面板无需先加 `sudo`，首次点击保存或激活时再进行系统授权；同一窗口的相机与 CAN 页签共用授权，后续修改无需重复输入密码。关闭窗口会结束辅助进程和这次授权；若辅助进程异常退出，下次执行需要重新授权。

新电脑需要安装系统依赖。以 Ubuntu 22.04 为例：

```bash
sudo apt update
sudo apt install python3 python3-gi gir1.2-gtk-3.0 udev \
  iproute2 sudo policykit-1 ffmpeg
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

### 检查是否已经绑定

两个页签都提供 **检查绑定** 按钮和 **绑定检查** 结果区。打开、刷新或保存后会自动读取当前状态，也可以随时点击按钮重查；无需授权或输入密码。

| 检查对象 | 通过条件 |
| --- | --- |
| 相机别名 | 所选相机身份未变化，别名是有效软链接并指向该设备，且本工具的持久化规则与设备匹配 |
| 所选机械臂 CAN | USB 适配器身份未变化，接口名称和波特率与当前选择一致，且接口处于 UP 状态 |

**全部已绑定** 显示为绿色；**存在未就绪项** 显示为橙色，并逐项列出实际状态和不一致原因，例如别名不存在、设备断开、接口仍为 `can0`、波特率不符或状态为 DOWN。结果旁显示检查时间。角色选为 **本次不配置** 时显示 **未检查**，不计入通过项目。

修改相机、别名、CAN 端口或波特率后，原检查结果会清除并提示重新检查。相机页的 **预览画面** 可继续用于确认画面及安装位置。

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

切换到 **机械臂 CAN**，为本次需要配置的角色选择 USB CAN 端口，不需要的角色选择 **本次不配置**。至少选择一个角色；只连接一只机械臂也可以配置。默认值：

| 角色 | 接口名称 | USB 端口 | 波特率 |
| --- | --- | --- | --- |
| 主臂 | `can_master` | `1-1.2:1.0` | `1000000` |
| 从臂 | `can_follower` | `1-1.3:1.0` | `1000000` |

点击 **绑定并激活所选机械臂**，首次执行时完成系统授权。后端统一通过 `ip` 配置和验证所选接口，同目录的 `can_activate.sh` 保留为转发到同一后端的兼容入口。命令行也可以分别配置：

```bash
# 只配置主臂
sudo /usr/bin/python3 can_port_bindings.py --master "1-1.2:1.0"
# 只配置从臂
sudo /usr/bin/python3 can_port_bindings.py --follower "1-1.3:1.0"
# 同时配置主从臂
sudo /usr/bin/python3 can_port_bindings.py \
  --master "1-1.2:1.0" --follower "1-1.3:1.0" --bitrate 1000000
```

面板只列出 USB CAN 适配器，并显示当前接口名、启停状态和波特率。主从臂不能选择同一个端口。未选角色不会被重新配置；如果目标名称被未选择的设备占用，会提示名称冲突。后端会核对设备身份，处理主从名称互换，并检查最终名称、波特率及 UP 状态；失败时尝试恢复原接口名称、已有波特率和启停状态，无法完整恢复时报告错误。

执行前请停止正在占用所选 CAN 接口的机械臂程序。**CAN 功能与原脚本一样只配置当前运行状态，重启或重新插拔后需要再次激活；不会自动设置开机启动。** 相机页的 udev 别名仍是持久化的。

只读检查与命令预览：

```bash
/usr/bin/python3 can_port_bindings.py --list
/usr/bin/python3 can_port_bindings.py --dry-run
/usr/bin/python3 can_port_bindings.py --dry-run --follower "1-1.3:1.0"
```

命令行省略 `--master` 和 `--follower` 时仍使用表中的两组默认端口；只传其中一个参数时仅配置该角色。CAN 后端仅依赖 `iproute2`，不再重复调用 `ethtool`、`dpkg-query` 或嵌套 `sudo`。图形授权需要 `policykit-1`，终端提权使用 `sudo`。所有配置仅在点击相应页签的执行按钮时进行。

### 命令行绑定

在本文件夹中运行：

```bash
# 查看相机及视频节点；非采集/元数据节点不能用于此绑定
/usr/bin/python3 camera_aliases.py --list

# 预览默认映射和规则，不修改系统
/usr/bin/python3 camera_aliases.py --dry-run

# 安装默认映射，自动重新加载规则并验证链接
sudo /usr/bin/python3 camera_aliases.py
```

指定其他设备或名称时，重复使用 `--map 设备=别名`：

```bash
sudo /usr/bin/python3 camera_aliases.py \
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
/usr/bin/python3 camera_aliases.py --dry-run --match serial

# 将角色绑定到 USB 连接位置
sudo /usr/bin/python3 camera_aliases.py --match path \
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

## 旧笔记中的手动检查

从仓库根目录使用下列命令查找实际 USB CAN 端口；设备号会随插口变化，主从角色必须现场确认。此处仓库根目录的旧脚本另需 `ethtool` 和 `can-utils`，可使用 `sudo apt install ethtool can-utils` 安装；上面的新面板无需这两个依赖：

```bash
bash find_all_can_port.sh
```

得到真实 USB 端口后，按本页 CAN 面板操作，或在仓库根目录执行（先改成现场查到的端口）：

```bash
MASTER_USB_PORT='1-10.2:1.0'
FOLLOWER_USB_PORT='1-10.3:1.0'
bash can_activate.sh can_master 1000000 "$MASTER_USB_PORT"
bash can_activate.sh can_follower 1000000 "$FOLLOWER_USB_PORT"
```

原笔记中“右边主臂、左边从臂”只是当时的摆放方式，不能由设备号数值大小推断。

相机只读检查及预览（任意工作目录）：

```bash
ls -l /dev/video*
udevadm info --query=path --name=/dev/video4
ffplay /dev/video_wrist
```

`/dev/video4` 是示例节点，检查前替换为实际编号。安装了 guvcview 时，也可用 `guvcview --device=/dev/video_top` 或 `guvcview --device=/dev/video_wrist` 预览。每次只开一个预览程序，确认后关闭再采集。

旧手工规则位于 `/etc/udev/rules.d/99-cameras.rules`，工具管理的是 `99-lerobot-cameras.rules`；应只保留一处同名别名定义。继续手工维护旧规则时，按实际 USB 路径调整 `KERNELS`，然后在设备主机重新加载：

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

使用本页绑定工具时它会自动重新加载，无需再做这一步。问题处理集中在[问题记录](../../troubleshooting/README.md)，模块文件职责见[架构说明](../../architecture/device-bindings.md)。
