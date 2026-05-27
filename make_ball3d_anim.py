"""Render a multi-ball animation matching the paper aesthetic.

Composites 6 independent Ball3D trajectories into a single isometric scene,
animates them frame-by-frame, saves as MP4.

Pure ground-truth visualization; no model inference needed.
"""
from __future__ import annotations
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

ROOT = Path("/home/pralak/Shortcut_actions")
DATA = ROOT / "mujoco_pipeline" / "data" / "ball3d_test.h5"
OUT = Path(__file__).parent / "assets" / "ball3d.mp4"
POSTER = Path(__file__).parent / "assets" / "ball3d_poster.jpg"

BALL_IDX = [51, 137, 41, 166, 30, 106]
T0 = 0
T_END = 100

X_LIM = (-0.45, 0.45)
Y_LIM = (-0.45, 0.45)
Z_LIM = (0.05, 0.85)

IDENTITY = plt.cm.tab10(np.linspace(0, 0.6, len(BALL_IDX)))

KX = 0.55
KZ = 0.32
BALL_R = 0.045
SHADOW_RX = 0.055
SHADOW_RY = 0.022


def project(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    sx = x + KX * y
    sy = z + KZ * y
    return np.stack([sx, sy], axis=-1)


def screen_lims():
    corners = np.array([[X_LIM[i], Y_LIM[j], Z_LIM[k]]
                          for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    s = project(corners)
    pad = 0.05
    return (s[:, 0].min() - pad, s[:, 0].max() + pad,
              s[:, 1].min() - pad, s[:, 1].max() + pad)


SCREEN_LIMS = screen_lims()


def draw_box(ax):
    xL, xH = X_LIM; yL, yH = Y_LIM; zL, zH = Z_LIM
    floor = np.array([[xL, yL, zL], [xH, yL, zL], [xH, yH, zL], [xL, yH, zL]])
    s = project(floor)
    ax.fill(s[:, 0], s[:, 1], color="#f4f1ea",
              edgecolor="#cfc8b8", linewidth=0.7, zorder=0)

    back = np.array([[xL, yH, zL], [xH, yH, zL], [xH, yH, zH], [xL, yH, zH]])
    s = project(back)
    ax.fill(s[:, 0], s[:, 1], color="#fafafa",
              edgecolor="#dfdcd2", linewidth=0.6, zorder=0)

    side = np.array([[xH, yL, zL], [xH, yH, zL], [xH, yH, zH], [xH, yL, zH]])
    s = project(side)
    ax.fill(s[:, 0], s[:, 1], color="#f0ede6",
              edgecolor="#dfdcd2", linewidth=0.6, zorder=0)

    edges = [
        ((xL, yL, zL), (xH, yL, zL)),
        ((xL, yL, zL), (xL, yH, zL)),
        ((xH, yL, zL), (xH, yH, zL)),
        ((xL, yL, zL), (xL, yL, zH)),
        ((xH, yL, zL), (xH, yL, zH)),
        ((xH, yH, zL), (xH, yH, zH)),
        ((xL, yL, zH), (xH, yL, zH)),
        ((xL, yL, zH), (xL, yH, zH)),
        ((xH, yL, zH), (xH, yH, zH)),
        ((xL, yH, zH), (xH, yH, zH)),
    ]
    for a, b in edges:
        p = project(np.array([a, b]))
        ax.plot(p[:, 0], p[:, 1], color="#777", linewidth=0.7,
                  alpha=0.8, zorder=1)


def render_frame(ax, positions, trails):
    ax.clear()
    sx_lo, sx_hi, sy_lo, sy_hi = SCREEN_LIMS
    ax.set_xlim(sx_lo, sx_hi); ax.set_ylim(sy_lo, sy_hi)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    draw_box(ax)

    # depth-sorted ball draw
    order = np.argsort(-positions[:, 1])
    for k in order:
        wp = positions[k]
        col = IDENTITY[k]

        # trail
        tr = trails[k]
        if len(tr) > 1:
            s = project(np.array(tr))
            ax.plot(s[:, 0], s[:, 1], color=col, alpha=0.45,
                      linewidth=1.4, zorder=1.8)

        sx, sy = project(wp)
        fsx, fsy = project(np.array([wp[0], wp[1], Z_LIM[0]]))

        h = max(0.0, wp[2] - Z_LIM[0])
        shadow_alpha = float(np.clip(0.32 - 0.12 * h, 0.06, 0.32))
        shadow_scale = float(np.clip(1.0 - 0.35 * h, 0.45, 1.0))
        sh = mpatches.Ellipse((fsx, fsy),
                                  width=2 * SHADOW_RX * shadow_scale,
                                  height=2 * SHADOW_RY * shadow_scale,
                                  facecolor="#000", alpha=shadow_alpha,
                                  edgecolor="none", zorder=1.5)
        ax.add_patch(sh)

        z = 10.0 - wp[1]
        rim_color = (np.array(col[:3]) * 0.55).tolist() + [1.0]
        ax.add_patch(mpatches.Circle((sx, sy), BALL_R * 1.04,
                                          facecolor="none",
                                          edgecolor=rim_color,
                                          linewidth=1.9,
                                          zorder=z + 0.05))
        ax.add_patch(mpatches.Circle((sx, sy), BALL_R, facecolor=col,
                                          edgecolor="none", zorder=z + 0.1))
        ax.add_patch(mpatches.Circle((sx - BALL_R * 0.28, sy + BALL_R * 0.30),
                                          BALL_R * 0.30, facecolor="#fff",
                                          alpha=0.55, edgecolor="none",
                                          zorder=z + 0.2))


def main():
    with h5py.File(DATA, "r") as f:
        states = np.array(f["states"])

    trajs = states[BALL_IDX, T0:T_END, :3]
    n_balls, n_frames, _ = trajs.shape
    print(f"loaded {n_balls} trajectories, {n_frames} frames each")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    fig.patch.set_facecolor("white")

    trails = [[] for _ in range(n_balls)]
    TRAIL_LEN = 14

    def update(frame_i):
        positions = trajs[:, frame_i]
        for k in range(n_balls):
            trails[k].append(positions[k])
            if len(trails[k]) > TRAIL_LEN:
                trails[k].pop(0)
        render_frame(ax, positions, trails)
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=33,
                              blit=False, repeat=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=30, codec="libx264",
                              extra_args=["-pix_fmt", "yuv420p",
                                              "-movflags", "+faststart",
                                              "-crf", "22"])
    print(f"writing {OUT}")
    anim.save(str(OUT), writer=writer)
    print("done")

    # Render a poster from the midpoint frame
    update(n_frames // 2)
    fig.savefig(str(POSTER), dpi=150, bbox_inches="tight")
    print(f"saved poster {POSTER}")


if __name__ == "__main__":
    main()
