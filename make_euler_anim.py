"""Render a clean Euler 2D prediction-evolution animation.

4-panel layout matching the Oregonator version: input density, surrogate
prediction, step-doubling trust signal ê, and true error.
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
NEURIPS = ROOT / "neurips"
sys.path.insert(0, str(NEURIPS / "experiments"))

from eval_utils_euler import load_model, step_doubling_estimator, true_error

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DT_BASE = 0.01
CKPT = NEURIPS / "training" / "checkpoints" / "shortcut_euler2d_v2_dagger" / "seed0" / "best.pt"
DATA = NEURIPS / "data" / "euler2d_v2" / "euler2d_v2_test.h5"

OUT = Path(__file__).parent / "assets" / "euler_pred.mp4"

TRAJ_IDX = 8
T0 = 5
HORIZONS = [2, 4, 8, 12, 16, 24, 32, 48, 64]


def reshape_state(flat):
    return flat.reshape(128, 128, 4).transpose(2, 0, 1)


def main():
    model = load_model(str(CKPT), device=DEVICE)
    with h5py.File(DATA, "r") as f:
        # data is (N, T, nx*ny*C) = (100, 100, 65536) for 128x128x4
        traj_flat = np.array(f["states"][TRAJ_IDX])  # (T, 65536)
        traj = np.stack([reshape_state(traj_flat[t]) for t in range(traj_flat.shape[0])])

    s_init = torch.from_numpy(traj[T0]).unsqueeze(0).to(DEVICE).float()
    print(f"input state shape: {s_init.shape}")

    frames = []
    print(f"computing {len(HORIZONS)} frames...")
    for h in HORIZONS:
        s_true = torch.from_numpy(traj[T0 + h]).unsqueeze(0).to(DEVICE).float()
        target_dt = h * DT_BASE
        with torch.no_grad():
            e_map, pred_full = step_doubling_estimator(model, s_init, target_dt)
            te_map = true_error(pred_full, s_true)
        frames.append({
            "h": h,
            "inp": s_init[0, 0].cpu().numpy(),     # density channel
            "pred": pred_full[0, 0].cpu().numpy(),
            "ehat": e_map[0].cpu().numpy(),
            "terr": te_map[0].cpu().numpy(),
        })
        print(f"  h={h}")

    state_vmin = min(f["inp"].min() for f in frames)
    state_vmax = max(f["pred"].max() for f in frames)
    err_vmax = float(max(max(f["ehat"].max(), f["terr"].max()) for f in frames))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), dpi=120)
    fig.patch.set_facecolor("#fbf8f1")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.78, bottom=0.10, wspace=0.12)

    def render(idx):
        fr = frames[idx]
        for ax in axes:
            ax.clear()
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)

        axes[0].imshow(fr["inp"], cmap="magma", aspect="equal",
                          vmin=state_vmin, vmax=state_vmax)
        axes[0].set_title("Input  (density)", fontsize=16, fontweight="bold",
                              pad=10, color="#1a1814")

        axes[1].imshow(fr["pred"], cmap="magma", aspect="equal",
                          vmin=state_vmin, vmax=state_vmax)
        axes[1].set_title("Prediction", fontsize=16, fontweight="bold",
                              pad=10, color="#a13b2b")

        axes[2].imshow(fr["ehat"], cmap="inferno", aspect="equal",
                          vmin=0, vmax=err_vmax)
        axes[2].set_title("Trust signal  ê", fontsize=16, fontweight="bold",
                              pad=10, color="#cc6600")

        axes[3].imshow(fr["terr"], cmap="inferno", aspect="equal",
                          vmin=0, vmax=err_vmax)
        axes[3].set_title("True error", fontsize=16, fontweight="bold",
                              pad=10, color="#444")

        fig.suptitle(f"Euler 2D   |   horizon  h = {fr['h']}  simulator steps",
                          fontsize=17, fontweight="bold", y=0.94,
                          color="#1a1814")
        return []

    anim = FuncAnimation(fig, render, frames=len(frames),
                              interval=600, blit=False, repeat=True)
    writer = FFMpegWriter(fps=2, codec="libx264",
                              extra_args=["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                                              "-profile:v", "main",
                                              "-level", "3.1",
                                              "-pix_fmt", "yuv420p",
                                              "-movflags", "+faststart",
                                              "-crf", "22"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {OUT}")
    anim.save(str(OUT), writer=writer)

    render(len(frames) - 1)
    fig.savefig(str(OUT.with_suffix("").with_name("euler_pred_poster.jpg")),
                  dpi=140, bbox_inches="tight",
                  facecolor=fig.get_facecolor())
    print("done")


if __name__ == "__main__":
    main()
