import sys
import os
import math
repo_root = os.path.abspath(os.getcwd())
sys.path.insert(0, os.path.join(repo_root, "optimized_dp"))

import numpy as np
import scipy.io as sio
import argparse

from odp.Grid     import Grid
from odp.Shapes   import CylinderShape, Union
from odp.solver   import HJSolver
# 2-input Dubins car (controls: speed, yaw-rate).  Same dynamics class used by
# optimized_dp/examples/dubins_3d_reach_avoid.py.
from odp.dynamics.DubinsCar2 import DubinsCar2


# ── 1. Grid (module-level, also reusable by importers) ────────────────────────
# Exactly the grid from dubins_3d_reach_avoid.py:
#   state = [x, y, theta], theta (index 2) is periodic.
g = Grid(
    minBounds    = np.array([-10.0, -10.0, -math.pi]),
    maxBounds    = np.array([ 10.0,  10.0,  math.pi]),
    dims         = 3,
    pts_each_dim = np.array([50, 50, 36]),
    periodicDims = [2],
)

# ── 2. Dynamics (module-level) ────────────────────────────────────────────────
# uMode="min" / dMode="max" — same as the example.
car = DubinsCar2(
    uMin  = [0.0, -1.0],       # [speedMin, wMin]
    uMax  = [1.0,  1.0],       # [speedMax, wMax]
    dMax  = [0.1, 0.1, 0.1],   # wind disturbance (x, y, theta)
    uMode = "min",
    dMode = "max",
)

# ── 3. Target set (fixed) ─────────────────────────────────────────────────────
# Cylinder at the origin, infinite in theta (ignore_dims=[2]).
# Same target across every sample.
# quadratic=False → TRUE (linear) signed distance, so values scale ~linearly
# with distance instead of distance² (keeps the SDF / value-function range small).
target_set = CylinderShape(
    g,
    ignore_dims = [2],
    center      = [0.0, 0.0, 0.0],
    radius      = 2,
    quadratic   = False,
)

# ── 4. Time horizon ───────────────────────────────────────────────────────────
# Same lookback/step as the example.
lookback_length = 8
t_step          = 0.5
tau = np.arange(0, lookback_length + 1e-5, t_step)

# ── Random-obstacle sampling parameters ───────────────────────────────────────
OBS_RADIUS_MIN       = 0.5   # min cylinder radius
OBS_RADIUS_MAX       = 2.0   # max cylinder radius
OBS_N_MIN            = 1     # min number of obstacles per sample
OBS_N_MAX            = 5     # max number of obstacles per sample
OBS_ORIGIN_CLEARANCE = 2   # obstacle centre must be >= this * radius from origin


# ── 5. Random obstacle set (importable) ───────────────────────────────────────
def random_obstacle_set(grid: Grid, rng: np.random.Generator) -> np.ndarray:
    """Random obstacle SDF on the 3D Dubins grid.

    Generates between OBS_N_MIN and OBS_N_MAX axis-aligned cylinders (infinite
    in theta), unions them, and returns the raw implicit-surface array
    (negative inside the obstacle) — the same convention as the example's
    obstacle_set, ready to pass directly to HJSolver.

    Each cylinder:
      * centre sampled uniformly from the x-y grid bounds ([-10,10]^2),
      * radius ~ U[OBS_RADIUS_MIN, OBS_RADIUS_MAX],
      * centre kept at least OBS_ORIGIN_CLEARANCE * radius from the origin so
        the target region near the origin stays clear of obstacles.
    """
    n_obs = int(rng.integers(OBS_N_MIN, OBS_N_MAX + 1))   # 1..5 inclusive

    obstacle = None
    for j in range(n_obs):
        radius = rng.uniform(OBS_RADIUS_MIN, OBS_RADIUS_MAX)

        # Rejection-sample an x-y centre at least 1.5 from the origin.
        while True:
            cx = rng.uniform(grid.min[0], grid.max[0])
            cy = rng.uniform(grid.min[1], grid.max[1])
            if np.hypot(cx, cy) >= OBS_ORIGIN_CLEARANCE:
                break

        # quadratic=False → linear signed distance (small, well-scaled values).
        cyl = CylinderShape(grid, ignore_dims=[2], center=[cx, cy, 0.0], radius=radius,
                            quadratic=False)
        obstacle = cyl if obstacle is None else Union(obstacle, cyl)
        print(f"  [obstacle {j+1}/{n_obs}] Cylinder | center={[round(cx,2), round(cy,2)]} | radius={radius:.2f}")

    return obstacle


# ── 6. Main: data generation loop (only runs when invoked as a script) ────────
def main():
    '''
    Sample command:

    python3 data_gen/dubins3D_data_gen.py --M 100 --seed 42 --plot
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=0,  help="Random seed")
    parser.add_argument("--out", type=str,
                        default=os.path.join(repo_root,
                            "data_gen/HJB_training_mat/DubinsCar2_50x50x36_reach_avoid.mat"),
                        help="Output .mat file")
    parser.add_argument("--plot", action="store_true", default=False, help="Plot each BRT after solving")
    args = parser.parse_args()

    M    = args.M
    rng  = np.random.default_rng(args.seed)
    out  = args.out
    plot = args.plot

    os.makedirs(os.path.dirname(out), exist_ok=True)

    # Preallocate:
    #   constraints : (M, nx, ny, nth)     — one obstacle SDF per sample
    #   results     : (M, nx, ny, nth, T)  — full time history per sample
    nx, ny, nth = g.pts_each_dim
    T           = len(tau)

    constraints = np.zeros((M, nx, ny, nth), dtype=np.float32)
    results     = np.zeros((M, nx, ny, nth, T), dtype=np.float32)

    # ------ Reach-Avoid solve settings (identical to dubins_3d_reach_avoid.py) --
    compMethods = {
        "TargetSetMode":   "minVWithV0",        # target is absorbing
        "ObstacleSetMode": "maxVWithObstacle",  # obstacle is always excluded
    }

    for i in range(M):
        print(f"\n[{i+1}/{M}] Generating sample...")

        obstacle_set = random_obstacle_set(g, rng)

        # HJSolver call set up exactly as in the example: pass the obstacle SDF
        # directly (NOT negated); HJSolver initialises V_0 = max(target, -obstacle)
        # internally and applies both modes at every time step.
        result = HJSolver(
            dynamics_obj     = car,
            grid             = g,
            multiple_value   = [target_set, obstacle_set],
            tau              = tau,
            compMethod       = compMethods,
            saveAllTimeSteps = True,
            accuracy         = "medium",
            verbose          = False,
        )

        constraints[i] = obstacle_set.astype(np.float32)
        results[i]     = result.astype(np.float32)

        print(f"  result shape: {result.shape}")

        if plot:
            import plotly.graph_objects as go
            from odp.Plots.plotting_utilities import slider_define

            # Meshgrid — copied from plot_isosurface in plotting_utilities.py
            complex_x = complex(0, g.pts_each_dim[0])
            complex_y = complex(0, g.pts_each_dim[1])
            complex_z = complex(0, g.pts_each_dim[2])
            mg_X, mg_Y, mg_Z = np.mgrid[
                g.min[0]:g.max[0]:complex_x,
                g.min[1]:g.max[1]:complex_y,
                g.min[2]:g.max[2]:complex_z,
            ]

            N = result.shape[3]

            target_trace = go.Isosurface(
                x=mg_X.flatten(), y=mg_Y.flatten(), z=mg_Z.flatten(),
                value=target_set.flatten(),
                caps=dict(x_show=True, y_show=True),
                isomin=-0.1, isomax=0.1, surface_count=1,
                colorscale=[[0, "green"], [1, "green"]],
                opacity=0.5, showscale=False, name="Target",
            )
            obs_trace = go.Isosurface(
                x=mg_X.flatten(), y=mg_Y.flatten(), z=mg_Z.flatten(),
                value=obstacle_set.flatten(),
                caps=dict(x_show=True, y_show=True),
                isomin=-0.1, isomax=0.1, surface_count=1,
                colorscale=[[0, "red"], [1, "red"]],
                opacity=0.5, showscale=False, name="Obstacle",
            )
            frames = [go.Frame(
                data=[go.Isosurface(
                    x=mg_X.flatten(), y=mg_Y.flatten(), z=mg_Z.flatten(),
                    value=result[:, :, :, N-k-1].flatten(),
                    caps=dict(x_show=True, y_show=True),
                    isomin=-0.1, isomax=0.1, surface_count=1,
                    colorscale=[[0, "blue"], [1, "blue"]],
                    opacity=0.3, showscale=False, name="BRT",
                )],
                traces=[0], name=str(k),
            ) for k in range(N)]
            brt_init = go.Isosurface(
                x=mg_X.flatten(), y=mg_Y.flatten(), z=mg_Z.flatten(),
                value=result[:, :, :, N-1].flatten(),
                caps=dict(x_show=True, y_show=True),
                isomin=-0.1, isomax=0.1, surface_count=1,
                colorscale=[[0, "blue"], [1, "blue"]],
                opacity=0.3, showscale=False, name="BRT",
            )
            fig = go.Figure(data=[brt_init, target_trace, obs_trace], frames=frames)
            fig.update_layout(
                title=f"BRT Sample {i+1}/{M}",
                scene=dict(
                    xaxis={"nticks": 20},
                    zaxis={"nticks": 20},
                    camera_eye={"x": 0, "y": -1, "z": 0.5},
                    aspectratio={"x": 1, "y": 1, "z": 0.6},
                ),
            )
            fig = slider_define(fig)
            fig.show()

    # ── 7. Save to .mat ───────────────────────────────────────────────────────
    sio.savemat(out, {
        "constraints": constraints,                      # (M, nx, ny, nth)
        "results":     results,                          # (M, nx, ny, nth, T)
        "target_set":  target_set.astype(np.float32),    # (nx, ny, nth) — shared
        "tau":         tau,                              # (T,)
        "M":           M,
        "nx":          nx,
        "ny":          ny,
        "nth":         nth,
        "T":           T,
        "x_axis":      g.grid_points[0],                 # (nx,)
        "y_axis":      g.grid_points[1],                 # (ny,)
        "theta_axis":  g.grid_points[2],                 # (nth,)
    })

    print(f"\nSaved {M} samples → {out}")
    print(f"  constraints : {constraints.shape}")
    print(f"  results     : {results.shape}")


if __name__ == "__main__":
    main()
