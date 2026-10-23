# SCARA 仿真平台（MuJoCo + ROS2 + 多轴配置入口）

双击 `menu.bat` 进入统一控制台，无需命令行。
也可以双击 `SCARA平台菜单.lnk`；桌面根目录也放了一份同名快捷方式。

## 统一菜单

`menu.bat` 由机器人配置文件驱动，当前提供：

| 功能 | 菜单动作 ID |
| --- | --- |
| 最新全栈控制台（Gazebo / Octo / Qwen） | `full_console` |
| MuJoCo 自动抓取演示 | `mujoco_auto` |
| MuJoCo 交互控制器 | `mujoco_console` |
| 5 组随机工况批量评测 | `batch_eval` |
| 双相机协同演示 | `dual_camera` |
| BC / DAgger 策略演示 | `bc_demo` |
| 场景截图 | `render_scene` |
| 行为克隆训练 | `train_bc` |
| DAgger 纠偏训练 | `train_dagger` |
| ROS2 同步与构建 | `ros2_build` |
| Simulink CTC 对比 | `simulink` |
| 两本 PDF 手册 | `open_*_manual` |

菜单内输入 `I` 查看当前机器人配置，`V` 校验全部动作入口，`P` 切换机器人配置。

```powershell
menu.bat
menu.bat --list
menu.bat --validate
menu.bat --run mujoco_auto
```

## 多轴封装接口

通用菜单代码位于 `platform\menu.py`，不包含任何 SCARA 专用尺寸。
每个机器人只增加一个配置文件：

```text
platform\profiles\<profile_id>.json
```

配置文件负责声明：

- 轴数 `dof`
- 关节名称与关节类型
- 状态维度和动作维度
- 末端执行器与控制器
- 该机器人可使用的动作及启动命令

已提供：

- `scara_teach_4rprr.json`：当前 4 轴 SCARA
- `template_6d.json`：6 轴扩展模板

六轴开发时复制模板，新增自己的适配代码和模型资源，然后把动作命令写入新 profile；
`menu.bat`、配置加载、校验、窗口启动和通用人工操作流程均无需修改。

```powershell
menu.bat --profile template_6d
menu.bat --all-profiles
```

## 兼容入口

原有 BAT 已迁移到 `_legacy_launchers`，没有删除。
它们现在只调用 `menu.bat --run <action_id>`，避免出现两套逻辑。

## 安全边界

- 菜单只启动子进程，不修改 MuJoCo、ROS2、Python 或 MATLAB 源码。
- 新建窗口运行，不占用或覆盖已有仿真进程。
- 配置校验失败时不会启动动作。
- `full_console` 调用当前正式项目 `C:\SCARA`，不会改写本项目。

## 具身智能（行为克隆 → DAgger）
- 专家数据：`python\scripts\gen_demos.py`（域随机化：物体/放置点/质量/摩擦/初始抖动）
- 策略：MLP 19→256→256→4，输入=关节+末端+物体+夹爪+放置点，输出=t+0.1 s 末端目标+夹爪
- 闭环：策略 50 Hz → `servo_track` 关节伺服（速度前馈）→ CTC 1 kHz → MuJoCo
- DAgger：策略跑自己的轨迹，把状态投影回专家计划取未来参考作纠偏标签

## 机器人
- SCARA-teach-4RPRR：a1=a2=0.225 m，J1 高 0.30 m，J3 行程 200 mm
- 两指平行夹爪（28 mm 闭合间距夹 30 mm 物体，2 mm 过盈摩擦抓取）
- 控制器 = 闭式 IK + 五次多项式 + 计算力矩控制(CTC)；策略伺服 = 速度前馈 + 位置修正
- 场景：J1 底座 + 四腿工作台 + 放置区 + 相机（cam_front / cam_top / cam_wrist）

## 想改造
- 参数：`armlab/spec.py` 的 `scara_teach_4rprr()`
- 场景（灯光/材质/台面/相机）：`armlab/scara_scene.py`
- 动作（高度/速度/夹爪行程）：`armlab/scara_task.py`、`armlab/robot_controller.py`
