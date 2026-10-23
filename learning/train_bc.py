"""行为克隆（BC）训练：从专家数据学"当前相位"（8 分类）。

网络：MLP 20→256→256→8（ReLU + 交叉熵），输入标准化，Adam，按 episode 划分训练/验证。
运行：python learning/train_bc.py [--epochs 120] [--obs-noise 0.002]
输出：learning/checkpoints/bc_policy.pt(+.json)、outputs/figures/bc_training.png

`train_core()` 同时被 DAgger（learning/dagger.py）复用，保证两边训练配置一致。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
DATA = ROOT / "outputs" / "data"
CKPT = ROOT / "learning" / "checkpoints"

OBS_DIM, N_PHASE = 20, 8


def train_core(obs, act, ep, *, epochs=120, batch=256, lr=1e-3, hidden=256,
               val_frac=0.2, seed=0, out_pt: Path | str = CKPT / "bc_policy.pt",
               obs_noise: float | np.ndarray = 0.0, log=print) -> dict:
    """训练相位分类器：数据进（act = one-hot 相位），权重与指标出（写到 out_pt）。"""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y = np.argmax(act, axis=1).astype(np.int64)
    episodes = np.unique(ep)
    rng.shuffle(episodes)
    n_val = max(int(round(len(episodes) * val_frac)), 1)
    val_eps = set(episodes[:n_val].tolist())
    tr = np.array([i for i in range(len(obs)) if ep[i] not in val_eps])
    va = np.array([i for i in range(len(obs)) if ep[i] in val_eps])

    o_mean, o_std = obs[tr].mean(0), obs[tr].std(0) + 1e-6
    obs_tr = obs.astype(np.float32).copy()
    noise_arr = np.asarray(obs_noise, dtype=np.float64)
    if bool(np.any(noise_arr > 0)):          # 观测噪声增广：对状态偏差更鲁棒
        gen = np.random.default_rng(seed + 1)
        if noise_arr.ndim == 0:
            noise_arr = np.full(obs.shape[1], float(noise_arr))
        obs_tr[tr] += gen.normal(0.0, 1.0, size=(len(tr), obs.shape[1])).astype(np.float32) \
            * noise_arr.astype(np.float32)
    X = torch.tensor((obs_tr - o_mean) / o_std, dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.long)

    net = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, N_PHASE))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    tr_t = torch.tensor(tr, dtype=torch.long)
    va_t = torch.tensor(va, dtype=torch.long)
    hist = {"train": [], "val": [], "acc": []}
    best, best_state = float("inf"), None
    log(f"===== 相位分类 BC 训练：{len(tr)} 训练样本 / {len(va)} 验证样本"
        f"（{len(episodes) - n_val}/{n_val} 条轨迹，"
        f"观测噪声增广 {'开' if bool(np.any(noise_arr > 0)) else '关'}）=====")
    for ep_i in range(epochs):
        net.train()
        perm = torch.randperm(len(tr_t))
        tot = 0.0
        for k in range(0, len(perm), batch):
            idx = tr_t[perm[k:k + batch]]
            opt.zero_grad()
            loss = lossf(net(X[idx]), Y[idx])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        net.eval()
        with torch.no_grad():
            logits = net(X[va_t])
            v = float(lossf(logits, Y[va_t]))
            acc = float((logits.argmax(1) == Y[va_t]).float().mean())
        hist["train"].append(tot / len(tr_t))
        hist["val"].append(v)
        hist["acc"].append(acc)
        if v < best:
            best = v
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        if (ep_i + 1) % 20 == 0 or ep_i == 0:
            log(f"  epoch {ep_i + 1:4d}: train {hist['train'][-1]:.5f}  val {v:.5f}  "
                f"验证相位准确率 {acc * 100:.2f}%  (best {best:.5f})")
    net.load_state_dict(best_state)

    out_pt = Path(out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_pt)
    try:
        dmeta = json.loads((DATA / "demos_meta.json").read_text(encoding="utf-8"))
    except Exception:
        dmeta = {}
    with torch.no_grad():
        logits = net(X[va_t])
        acc = float((logits.argmax(1) == Y[va_t]).float().mean())
        conf = torch.softmax(logits, 1).max(1).values.mean().item()
    meta = {"obs_mean": o_mean.tolist(), "obs_std": o_std.tolist(),
            "hidden": hidden, "obs_dim": OBS_DIM, "n_phase": N_PHASE,
            "phase_names": dmeta.get("phase_names"),
            "train_samples": int(len(tr)), "val_samples": int(len(va)),
            "best_val_loss": best, "val_phase_accuracy": acc,
            "val_mean_confidence": conf, "epochs": epochs,
            "obs_noise": (float(noise_arr) if noise_arr.ndim == 0 else
                          f"per-dim(max={float(np.max(noise_arr)):.4f})"),
            "label_mode": dmeta.get("label_mode"),
            "data_episodes": dmeta.get("episodes")}
    out_pt.with_suffix(out_pt.suffix + ".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"验证集：相位准确率 {acc * 100:.2f}%，平均置信度 {conf * 100:.1f}%")
    return {"meta": meta, "hist": hist, "net": net, "y_val": y[va],
            "val_logits": logits.numpy()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obs-noise", type=float, default=0.0,
                    help=">0 时用统一噪声；0 = 关闭（DAgger 里用按维度的噪声）")
    args = ap.parse_args()

    t0 = time.time()
    data = np.load(DATA / "demos_pick_place.npz")
    res = train_core(data["obs"], data["act"], data["episode"], epochs=args.epochs,
                     batch=args.batch, lr=args.lr, hidden=args.hidden,
                     val_frac=args.val_frac, seed=args.seed,
                     obs_noise=args.obs_noise, out_pt=CKPT / "bc_policy.pt")
    hist = res["hist"]
    res["meta"]["train_time_s"] = round(time.time() - t0, 1)
    (CKPT / "bc_policy.pt.json").write_text(
        json.dumps(res["meta"], ensure_ascii=False, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(hist["train"], label="训练")
    axes[0].plot(hist["val"], label="验证")
    axes[0].set_yscale("log"); axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("交叉熵"); axes[0].set_title("分类损失")
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(np.array(hist["acc"]) * 100)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("准确率 [%]")
    axes[1].set_title(f"验证相位准确率（最终 {hist['acc'][-1] * 100:.2f}%）")
    axes[1].grid(alpha=0.3)
    # 混淆矩阵
    pred = res["val_logits"].argmax(1)
    yv = res["y_val"]
    K = N_PHASE
    cm = np.zeros((K, K))
    for a_, b_ in zip(yv, pred):
        cm[a_, b_] += 1
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = axes[2].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_xlabel("预测相位"); axes[2].set_ylabel("真实相位")
    axes[2].set_title("验证集混淆矩阵")
    names = (res["meta"].get("phase_names") or [str(i) for i in range(K)])
    axes[2].set_xticks(range(K)); axes[2].set_xticklabels(names, rotation=45, ha="right")
    axes[2].set_yticks(range(K)); axes[2].set_yticklabels(names)
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle("相位分类策略（BC / DAgger）：19 维观测 → 8 相位", y=1.02)
    figDir = ROOT / "outputs" / "figures"
    figDir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figDir / "bc_training.png", dpi=130, bbox_inches="tight")
    print(f"\n[ok] 权重：{CKPT / 'bc_policy.pt'}，图：{figDir / 'bc_training.png'}"
          f"，用时 {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
