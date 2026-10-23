"""armlab —— 机械臂全栈开发计算内核（N 自由度通用串链，2 轴为当前实例）。

模块导航
--------
spec        参数定义（唯一数据源）与 JSON 交换格式
kinematics  通用 FK/雅可比、2 轴解析 FK/IK、通用数值 IK
dynamics    通用动力学（雅可比法）与 2 轴解析动力学，统一 Model 接口
trajectory  关节空间/笛卡尔空间轨迹生成（任意自由度）
controllers CTC / PD+重力补偿控制器（依赖 Model 接口）
mjcf        ArmSpec → MuJoCo XML 生成与 MuJoCo 交叉验证工具
"""

from . import controllers, dynamics, kinematics, mjcf, spec, trajectory  # noqa: F401

__version__ = "0.1.0"

__all__ = ["spec", "kinematics", "dynamics", "trajectory", "controllers", "mjcf"]
