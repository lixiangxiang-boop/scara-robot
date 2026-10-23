# 多轴机器人配置接口

本目录把“通用平台”和“某一台机器人”的差异隔离开。

## 目录

```text
platform/
  menu.py                 通用菜单，不包含某一轴数的尺寸
  robot_profile.py        Profile 数据模型与校验器
  active_profile.txt      当前选中的 profile_id
  profiles/
    scara_teach_4rprr.json
    template_6d.json
  adapters/               后续各轴专用程序与适配代码
```

## 通用部分

以下内容不随轴数变化：

- 菜单显示、输入、窗口启动；
- Profile 加载与动作入口校验；
- 新控制台启动方式；
- 工作目录与环境变量传递；
- 功能分组、错误提示与兼容入口；
- 与 ROS2、MATLAB、MuJoCo 的进程隔离方式。

## 轴专用部分

以下内容必须由各轴 profile 或 adapter 提供：

- 关节数量、关节名称、关节类型；
- DH/几何参数、限位、质量与惯量；
- 正逆运动学、动力学与控制器；
- 状态向量、动作向量和执行器；
- URDF/MJCF/SDF 与碰撞模型；
- 动作命令、专用任务脚本；
- 真机驱动、标定与安全门限。

## 新增一台机器人

1. 复制 `profiles/template_6d.json`。
2. 修改 `profile_id`、`display_name`、`dof`、关节信息、状态/动作维度。
3. 在 `adapters/<profile_id>/` 编写该轴专用算法和场景代码。
4. 在 profile 的 `actions` 中添加动作命令。
5. 运行校验：

```powershell
python platform\menu.py --profile <profile_id> --validate
```

6. 运行菜单并切换配置：

```powershell
menu.bat
```

## Action 格式

```json
{
  "id": "unique_action_id",
  "label": "菜单显示名称",
  "description": "一句话说明",
  "mode": "new_console",
  "command": ["{python}", "{root}\\adapters\\demo.py"]
}
```

支持两种模式：

- `new_console`：在独立命令窗口启动程序；
- `open_file`：打开 PDF、图片或文档。

占位符：

- `{root}`：平台根目录；
- `{python}`：当前 Python 解释器。

## 不变性约束

新增 6 轴机器人时不要修改 `menu.py` 或 `robot_profile.py`。
如果必须修改通用代码，说明当前抽象还缺少通用协议，应先把差异抽成 profile 字段或 adapter 接口，而不是在菜单里写 `if dof == 6`。
