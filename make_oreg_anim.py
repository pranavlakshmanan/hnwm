"""Render an Oregonator prediction-evolution animation.

4-panel layout:
  Input (u channel) | Surrogate prediction (u channel) | Trust signal ê | True error
animated across a sweep of horizons / time steps.
"""
from __future__ import annotations
import sys
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = Path("/home/pralak/Shortcut_actions")
OREG = ROOT / "oregonator_pipeline"
sys.path.insert(0, str(OREG / "experiments"))
sys.path.insert(0, str(OREG / "models"))

from eval_utils import load_model, step_doubling_estimator, true_error

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DT_BASE = 0.05
CKPT = OREG / "checkpoints" / "shortcut_oregonator_v3" / "seed0" / "best.pt"
DATA = OREG / "data" / "oregonator" / "oregonator_test.h5"

OUT = Path(__file__).parent / "assets" / "oregonator_pred.mp4"

TRAJ_IDX = 28
T0 = 20
HORIZONS = [2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]


def main():
    model = load_model(str(CKPT), device=DEVICE)
    with h5py.File(DATA, "r") as f:
        s_init = torch.from_numpy(np.array(f["states"][TRAJ_IDX, T0])).unsqueeze(0).to(DEVICE)
        traj = np.array(f["states"][TRAJ_IDX, T0:T0 + max(HORIZONS) + 1])

    frames = []
    print(f"computing {len(HORIZONS)} prediction frames...")
    for h in HORIZONS:
        s_true = torch.from_numpy(traj[h]).unsqueeze(0).to(DEVICE)
        target_dt = h * DT_BASE
        with torch.no_grad():
            e_map, pred_full = step_doubling_estimator(model, s_init, target_dt)
            te_map = true_error(pred_full, s_true)
        frames.append({
            "h": h,
            "inp": s_init[0, 0].cpu().numpy(),
            "pred": pred_full[0, 0].cpu().numpy(),
            "ehat": e_map[0].cpu().numpy(),
            "terr": te_map[0].cpu().numpy(),
        })
        print(f"  h={h}")

    # consistent scales
    state_vmin = float(min(f["inp"].min() for f in frames) +
                      min(f["pred"].min() for f in frames)) / 2
    state_vmax = float(max(f["inp"].max() for f in frames) +
                      max(f["pred"].max() for f in frames)) / 2
    err_vmax = float(max(max(f["ehat"].max(), f["terr"].max()) for f in frames))

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), dpi=120)
    fig.patch.set_facecolor("#fbf8f1")

    def render(idx):
        fr = frames[idx]
        for ax in axes:
            ax.clear()
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
        axes[0].imshow(fr["inp"], cmap="viridis", aspect="equal",
                          vmin=state_vmin, vmax=state_vmax)
        axes[0].set_title(r"Input  $u(t_0)$", fontsize=14, fontweight="bold", pad=8)

        axes[1].imshow(fr["pred"], cmap="viridis", aspect="equal",
                          vmin=state_vmin, vmax=state_vmax)
        axes[1].set_title(rf"Prediction  $\hat{{u}}(t_0 + {fr['h']}\Delta t)$",
                              fontsize=14, fontweight="bold", pad=8,
                              color="#a13b2b")

        axes[2].imshow(fr["ehat"], cmap="inferno", aspect="equal",
                          vmin=0, vmax=err_vmax)
        axes[2].set_title(r"Trust signal  $\hat{e}$  (label-free)",
                              fontsize=14, fontweight="bold", pad=8,
                              color="#cc6600")

        axes[3].imshow(fr["terr"], cmap="inferno", aspect="equal",
                          vmin=0, vmax=err_vmax)
        axes[3].set_title(r"True error  $|\hat{u} - u|$",
                              fontsize=14, fontweight="bold", pad=8,
                              color="#666")
        fig.suptitle(f"Horizon  h = {fr['h']}  ({fr['h']} simulator steps)",
                          fontsize=15, y=0.02, color="#4a4438")
        return []

    anim = FuncAnimation(fig, render, frames=len(frames),
                              interval=600, blit=False, repeat=True)
    writer = FFMpegWriter(fps=2, codec="libx264",
                              extra_args=["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                                              "-pix_fmt", "yuv420p",
                                              "-movflags", "+faststart",
                                              "-crf", "20"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {OUT}")
    anim.save(str(OUT), writer=writer)

    # poster: pick last frame (h=64) which is the marquee result
    render(len(frames) - 1)
    fig.savefig(str(OUT.with_suffix(".jpg").with_name("oregonator_pred_poster.jpg")),
                  dpi=140, bbox_inches="tight",
                  facecolor=fig.get_facecolor())
    print("done")


if __name__ == "__main__":
    main()
