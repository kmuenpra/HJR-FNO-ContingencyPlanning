"""Standalone rollout-animation worker for scenario_optimization_reach.py.

Run as:   python _viz_worker.py <payload.npz> [frame_pause]

Plays a SIMPLE matplotlib animation (ax.clear() redraw loop + plt.pause), then
blocks on plt.show() until the window is closed. Lives in its own process and
imports ONLY numpy + matplotlib (never torch / odp), so it does not share the
parent's CUDA context — which is what causes the plt SIGSEGV.

The parent (visualize_constraint_result) precomputes every frame as plain numpy
and hands them over via the .npz payload.
"""
import sys
import numpy as np


def main(npz_path: str, frame_pause: float = 0.15) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d = np.load(npz_path, allow_pickle=True)
    Xg, Yg   = d["Xg"], d["Yg"]
    frames_V = d["frames_V"]               # (F, Nx, Ny) — V slice shown each frame
    V_bg     = d["V_bg"]                   # (Nx, Ny)   — fully-grown BRT at start heading (for delta contour)
    obstacle = d["obstacle"]; target = d["target"]
    traj     = d["traj"]                   # (F, 3)
    delta    = float(d["delta_hat"]); dt = float(d["dt"])
    x_lo, x_hi = d["x_lo"], d["x_hi"]
    title    = str(d["title"])

    F = len(traj); n_steps = F - 1
    vmin = float(np.percentile(frames_V[0], 2))
    vmax = float(np.percentile(frames_V[0], 98))

    legend = [
        Line2D([0], [0], color="magenta", lw=2,            label="V(x,t)=0 (current BRT)"),
        Line2D([0], [0], color="blue",    lw=2,            label="V(x,0)=0 (full BRT)"),
        Line2D([0], [0], color="lime",    lw=2.5,          label="V(x,0)=delta_hat (S_hat)"),
        Line2D([0], [0], color="red",     lw=1.5, ls="--", label="obstacle"),
        Line2D([0], [0], color="green",   lw=1.5, ls="--", label="target"),
        Line2D([0], [0], color="k",       lw=1.6,          label="rollout"),
    ]

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 6))
    for k in range(F):
        ax.clear()
        ax.pcolormesh(Xg, Yg, frames_V[k], cmap="Blues_r", shading="auto",
                      vmin=vmin, vmax=vmax)
        # MOVING: current value-function 0-sublevel set V(x,t)=0 — shrinks toward
        # the target as the rollout's time-to-go decreases (magenta).
        ax.contour(Xg, Yg, frames_V[k], levels=[0.0], colors="magenta", linewidths=2)
        # FIXED reference sets from the fully-grown V(x,0):
        ax.contour(Xg, Yg, V_bg, levels=[0.0], colors="blue", linewidths=2)            # V(x,0)=0
        if np.isfinite(delta):
            ax.contour(Xg, Yg, V_bg, levels=[delta], colors="lime", linewidths=2.5)    # V(x,0)=delta (S_hat)
        ax.contour(Xg, Yg, obstacle, levels=[0.0], colors="red",   linewidths=1.5, linestyles="--")
        ax.contour(Xg, Yg, target,   levels=[0.0], colors="green", linewidths=1.5, linestyles="--")

        cur = traj[:k + 1]
        ax.plot(cur[:, 0], cur[:, 1], "k-", lw=1.6)
        ax.plot(cur[0, 0], cur[0, 1], "ks", ms=8, mfc="white")
        ax.plot(cur[-1, 0], cur[-1, 1], "ko", ms=8)

        ax.set_xlim(float(x_lo[0]), float(x_hi[0]))
        ax.set_ylim(float(x_lo[1]), float(x_hi[1]))
        ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
        th  = (traj[k, 2] + np.pi) % (2 * np.pi) - np.pi
        ttg = max((n_steps - k) * dt, 0.0)
        ax.set_title(f"{title}step {k}/{n_steps}   theta={th:+.2f}   "
                     f"ttg={ttg:.2f}s   delta_hat={delta:.4g}")
        ax.legend(handles=legend, loc="upper right", fontsize=8)
        plt.pause(frame_pause)

    plt.ioff()
    print("  [viz] animation finished — close the window to continue...")
    plt.show()   # block until the user closes the window


if __name__ == "__main__":
    npz = sys.argv[1]
    fp  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    main(npz, fp)
