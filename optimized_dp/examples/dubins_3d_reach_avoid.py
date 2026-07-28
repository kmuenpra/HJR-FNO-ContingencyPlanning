"""
3D Dubins Car (constant speed unicycle) — Reach-Avoid BRT
==========================================================
State space:  [x, y, theta]
  x      ∈ [-3, 3]      (m)
  y      ∈ [-1, 4]      (m)
  theta  ∈ [-pi, pi]    (rad)   — periodic dimension

Target set  : cylinder centred at (0, 2) in x-y, radius 0.8
              (the car must REACH this region)
Obstacle set: cylinder centred at (1, 1) in x-y, radius 0.5
              (the car must AVOID this at all times)

Reach-Avoid BRT semantics:
  V_0 = max(target_sdf, -obstacle_sdf)
  At each time step:
    V ← min(V, V_0)           [TargetSetMode:   minVWithV0]
    V ← max(V, -obstacle)     [ObstacleSetMode: maxVWithObstacle]
  BRT = {x : V(x, 0) ≤ 0}
"""

import numpy as np
import math
import plotly.graph_objects as go
from odp.Plots.plotting_utilities import pre_plot
from odp.Plots.plotting_utilities import slider_define

from odp.Grid   import Grid
from odp.Shapes import CylinderShape
from odp.Plots  import PlotOptions, visualize_plots
from odp.solver import HJSolver, computeSpatDerivArray

# Use the 2-input DubinsCar2 class (controls: speed, yaw rate)
from odp.dynamics.DubinsCar2 import DubinsCar2

# ── 1. Grid ───────────────────────────────────────────────────────────────────
# 3 dimensions: x, y, theta
# Periodic dimension: index 2 (theta)
g = Grid(
    minBounds    = np.array([-3.0, -1.0, -math.pi]),
    maxBounds    = np.array([ 3.0,  4.0,  math.pi]),
    dims         = 3,
    pts_each_dim = np.array([60, 60, 36]),
    periodicDims = [2],
)

# ── 2. Dynamics ───────────────────────────────────────────────────────────────
# uMode="max": controller maximises value (tries to reach target, stay safe)
# dMode="min": disturbance minimises value (adversarial)
car = DubinsCar2(
    uMin  = [0.0, -1.0],       # [speedMin, wMin]
    uMax  = [1.0,  1.0],       # [speedMax, wMax]
    dMax  = [0.1, 0.1, 0.1],   # set nonzero for wind disturbance
    uMode = "min",
    dMode = "max",
)

# ── 3. Target set (reach) ─────────────────────────────────────────────────────
# Cylinder at (x=0, y=2), radius=0.8, infinite in theta (ignoreDims=[2])
target_set = CylinderShape(
    g,
    ignore_dims = [2],
    center      = [0.0, 2.0, 0.0],
    radius      = 0.8,
)

# ── 4. Obstacle set (avoid) ───────────────────────────────────────────────────
# Cylinder at (x=1, y=1), radius=0.5, infinite in theta (ignoreDims=[2])
obstacle_set = CylinderShape(
    g,
    ignore_dims = [2],
    center      = [1.0, 1.0, 0.0],
    radius      = 0.5,
)

# ── 5. Time horizon ───────────────────────────────────────────────────────────
lookback_length = 5
t_step          = 0.1
tau = np.arange(0, lookback_length + 1e-5, t_step)

# ── 6. Solver ─────────────────────────────────────────────────────────────────
# Passing [target_set, obstacle_set] activates reach-avoid mode in HJSolver.
# HJSolver internally initialises:
#   V_0 = max(target_sdf, -obstacle_sdf)
# and applies both TargetSetMode and ObstacleSetMode at every time step.
compMethods = {
    "TargetSetMode":   "minVWithV0",       # target is absorbing
    "ObstacleSetMode": "maxVWithObstacle", # obstacle is always excluded
}

result = HJSolver(
    dynamics_obj     = car,
    grid             = g,
    multiple_value   = [target_set, obstacle_set],
    tau              = tau,
    compMethod       = compMethods,
    saveAllTimeSteps = True,
    accuracy         = "medium",
)

# ── 7. Extract final BRT ──────────────────────────────────────────────────────
# result shape: (60, 60, 36, len(tau))
# Index 0 on the last axis = t=0 (full lookback applied)
brt = result[..., 0]

# ── 8. Spatial derivatives and optimal control at a sample state ──────────────
x_deriv = computeSpatDerivArray(g, brt, deriv_dim=1, accuracy="medium")
y_deriv = computeSpatDerivArray(g, brt, deriv_dim=2, accuracy="medium")
t_deriv = computeSpatDerivArray(g, brt, deriv_dim=3, accuracy="medium")

# Sample grid index
ix, iy, ith = 10, 20, 15
state_sample = (
    g.grid_points[0][ix],
    g.grid_points[1][iy],
    g.grid_points[2][ith],
)
spat_deriv_sample = (
    x_deriv[ix, iy, ith],
    y_deriv[ix, iy, ith],
    t_deriv[ix, iy, ith],
)
opt_speed, opt_w = car.optCtrl_inPython(state_sample, spat_deriv_sample)
print(f"Sample state indices : ({ix}, {iy}, {ith})")
print(f"Sample state         : {state_sample}")
print(f"Spatial derivatives  : {spat_deriv_sample}")
print(f"Optimal speed        : {opt_speed:.4f} m/s")
print(f"Optimal yaw rate     : {opt_w:.4f} rad/s")

# ── 9. Visualise: BRT (animated) + target + obstacle in one figure ────────────

# Meshgrid — copied directly from plot_isosurface in plotting_utilities.py
complex_x = complex(0, g.pts_each_dim[0])
complex_y = complex(0, g.pts_each_dim[1])
complex_z = complex(0, g.pts_each_dim[2])
mg_X, mg_Y, mg_Z = np.mgrid[
    g.min[0]:g.max[0]:complex_x,
    g.min[1]:g.max[1]:complex_y,
    g.min[2]:g.max[2]:complex_z,
]

N = result.shape[3]

# Static target trace — green, same in every frame
target_trace = go.Isosurface(
    x=mg_X.flatten(), y=mg_Y.flatten(), z=mg_Z.flatten(),
    value=target_set.flatten(),
    caps=dict(x_show=True, y_show=True),
    isomin=-0.1, isomax=0.1, surface_count=1,
    colorscale=[[0, "green"], [1, "green"]],
    opacity=0.5, showscale=False, name="Target",
)

# Static obstacle trace — red, same in every frame
obs_trace = go.Isosurface(
    x=mg_X.flatten(), y=mg_Y.flatten(), z=mg_Z.flatten(),
    value=obstacle_set.flatten(),
    caps=dict(x_show=True, y_show=True),
    isomin=-0.1, isomax=0.1, surface_count=1,
    colorscale=[[0, "red"], [1, "red"]],
    opacity=0.5, showscale=False, name="Obstacle",
)

# Animated BRT frames — blue, indexed exactly as plot_isosurface 4D branch does
frames = [go.Frame(
    data=[
        go.Isosurface(
            x=mg_X.flatten(), y=mg_Y.flatten(), z=mg_Z.flatten(),
            value=result[:, :, :, N-k-1].flatten(),
            caps=dict(x_show=True, y_show=True),
            isomin=-0.1, isomax=0.1, surface_count=1,
            colorscale=[[0, "blue"], [1, "blue"]],
            opacity=0.3, showscale=False, name="BRT",
        ),
    ],
    traces=[0],   # only update trace index 0 (BRT); indices 1,2 (target, obs) stay static
    name=str(k),
) for k in range(N)]

# Initial display before animation — matches plot_isosurface add_trace(my_V[:,:,:,N-1])
brt_init = go.Isosurface(
    x=mg_X.flatten(), y=mg_Y.flatten(), z=mg_Z.flatten(),
    value=result[:, :, :, N-1].flatten(),
    caps=dict(x_show=True, y_show=True),
    isomin=-0.1, isomax=0.1, surface_count=1,
    colorscale=[[0, "blue"], [1, "blue"]],
    opacity=0.3, showscale=False, name="BRT",
)

fig = go.Figure(data=[brt_init, target_trace, obs_trace], frames=frames)

# Layout — copied from plot_isosurface 4D branch in plotting_utilities.py
fig.update_layout(
    title="3D Set",
    scene=dict(
        xaxis={"nticks": 20},
        zaxis={"nticks": 20},
        camera_eye={"x": 0, "y": -1, "z": 0.5},
        aspectratio={"x": 1, "y": 1, "z": 0.6},
    ),
)

# slider_define is used as-is from plotting_utilities.py
fig = slider_define(fig)

fig.show()