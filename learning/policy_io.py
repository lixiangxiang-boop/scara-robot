"""策略的输入/输出定义与加载（训练与执行共用，保证接口一致）。

本阶段策略是**相位分类器**（分层控制的"任务级"）：
    观测 obs(20) = [q(4), dq(4), tcp(3), tcp_yaw(1), obj_pos(3), obj_yaw(1),
                    grip(1), place(2), phase_progress(1)]
        phase_progress = 当前相位内相对进度 [0,1]（运动原语的"做到哪儿了"，
        任务存在部分可观测性：仅凭末端位姿无法区分"刚到位要夹紧"和"夹完要抬起"，
        因此把原语进度作为补充状态；数据侧由 PhaseSequencer 提供同样口径的进度）
    （也试过把位置换成"末端−物块 / 末端−放置区"的相对量：实测闭环反而更差
      （0/20），因为丢掉了绝对工作区信息，决策面在边界更容易抖。）
    输出 phase(8) = 8 个相位（approach/descend/grasp/lift/move/lower/release/retract）
                   的 logits → argmax 得相位；"怎么走"交给运动原语
                    （`PickPlaceTask.phase_segment`：MoveL + 五次多项式）。

为什么不是"连续末端目标回归"：抓手要在 30 mm 物体两侧 3~4 mm 内闭合，要求网络做
毫米级连续回归；闭环里小的回归误差会滚雪球（实测 4 种标签/伺服组合成功率都 < 10%）。
改成"相位分类 + 原语执行"后，末端精度由原语（离线精确坐标 + 专家同一套五次多项式）保证，
网络只需判断"现在该做哪一段"。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OBS_DIM = 20
N_PHASE = 8
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "learning" / "checkpoints" / "bc_policy.pt"


def build_obs(q, dq, p_tcp, yaw_tcp, p_obj, yaw_obj, grip, place_xy,
              progress: float = 0.0) -> np.ndarray:
    return np.concatenate([np.asarray(q, float).reshape(4),
                           np.asarray(dq, float).reshape(4),
                           np.asarray(p_tcp, float).reshape(3),
                           [float(yaw_tcp)],
                           np.asarray(p_obj, float).reshape(3),
                           [float(yaw_obj)],
                           [float(grip)],
                           np.asarray(place_xy, float).reshape(2),
                           [float(progress)]]).astype(np.float32)


class MLP:
    """与训练脚本结构一致的 MLP（ReLU；输出 8 个相位 logits）。"""

    def __init__(self, w1, b1, w2, b2, w3, b3, obs_mean, obs_std):
        self.w1, self.b1 = w1, b1
        self.w2, self.b2 = w2, b2
        self.w3, self.b3 = w3, b3
        self.obs_mean, self.obs_std = obs_mean, obs_std

    def logits(self, obs: np.ndarray) -> np.ndarray:
        x = (np.asarray(obs, dtype=np.float32) - self.obs_mean) / self.obs_std
        h1 = np.maximum(x @ self.w1 + self.b1, 0.0)
        h2 = np.maximum(h1 @ self.w2 + self.b2, 0.0)
        return (h2 @ self.w3 + self.b3).astype(np.float64)

    def phase(self, obs: np.ndarray) -> int:
        return int(np.argmax(self.logits(obs)))

    def probs(self, obs: np.ndarray) -> np.ndarray:
        z = self.logits(obs)
        z = z - np.max(z)
        e = np.exp(z)
        return e / np.sum(e)


class Policy:
    """加载训练好的相位策略（若安装了 torch 用 torch 推理，否则用等价 numpy 实现）。"""

    def __init__(self, path: Path | str = MODEL_PATH):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"未找到策略权重：{path}（先运行 learning/train_bc.py）")
        self.meta = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
        ckpt = _load_ckpt(path)
        self._mlp = MLP(
            np.asarray(ckpt["w1"], dtype=np.float32), np.asarray(ckpt["b1"], dtype=np.float32),
            np.asarray(ckpt["w2"], dtype=np.float32), np.asarray(ckpt["b2"], dtype=np.float32),
            np.asarray(ckpt["w3"], dtype=np.float32), np.asarray(ckpt["b3"], dtype=np.float32),
            np.asarray(self.meta["obs_mean"], dtype=np.float32),
            np.asarray(self.meta["obs_std"], dtype=np.float32))

    def phase(self, obs: np.ndarray) -> int:
        return self._mlp.phase(obs)

    def probs(self, obs: np.ndarray) -> np.ndarray:
        return self._mlp.probs(obs)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """兼容旧接口名：返回 8 维 logits。"""
        return self._mlp.logits(obs)


def _load_ckpt(path: Path) -> dict:
    """读取权重：优先 torch 保存的 state_dict（nn.Sequential），否则 npz。

    torch 的 Linear 权重是 (out, in)，而 numpy 实现用 x @ w（需要 (in, out)），
    这里统一转置并映射成 w1/b1… 的名字。
    """
    try:
        import torch
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        mapping = {"0.weight": "w1", "0.bias": "b1", "2.weight": "w2", "2.bias": "b2",
                   "4.weight": "w3", "4.bias": "b3"}
        out = {}
        for k, v in sd.items():
            name = mapping.get(k, k)
            arr = v.numpy()
            if name.startswith("w"):
                arr = arr.T
            out[name] = arr
        return out
    except Exception:
        npz = np.load(str(path).replace(".pt", ".npz"))
        return {k: npz[k] for k in npz.files}
