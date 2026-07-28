"""
2D Plane (double integrator with bounded velocity + disturbance)
— Reach-Avoid BRT with a TIME-VARYING target set
=================================================================
State space:  [x, y]
  x, y ∈ [-10, 10]    (m)

Target set  : disk that translates linearly across the domain over [0, tau_end].
              Built as an SDF stack of shape (nx, ny, len(tau)) so the patched
              HJSolver slices target[..., i] at each outer time step.
Constraint  : a single randomly drawn shape (cylinder / rectangle / ellipsoid)
              that the system must STAY INSIDE for all time
              (passed as -constraint to HJSolver, same convention as
              training_HJFNO/Plane2D_data_gen.py).

This script exercises the time-varying-target patch in odp/solver.py and
visualises the resulting BRS as an animated 2D plotly figure (heatmap fill +
zero-level contour), structurally following dubins_3d_reach_avoid.py.
"""

import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

import numpy as np
import plotly.graph_objects as go

from odp.Grid     import Grid
from odp.Shapes   import CylinderShape, ShapeRectangle, ShapeEllipsoid
from odp.solver   import HJSolver
from odp.dynamics import Plane2D

# ── 1. Grid ───────────────────────────────────────────────────────────────────
g = Grid(
    minBounds    = np.array([-10.0, -10.0]),
    maxBounds    = np.array([ 10.0,  10.0]),
    dims         = 2,
    pts_each_dim = np.array([50, 50]),
    periodicDims = [],
)

# ── 2. Dynamics (matches Plane2D_data_gen.py) ─────────────────────────────────
plane = Plane2D(
    x     = [0.5, 0.8],
    vxMin = -1.0,
    vxMax =  1.0,
    vyMin = -1.0,
    vyMax =  1.0,
    dMin  = -0.1,
    dMax  =  0.1,
    uMode = "min",
    dMode = "max",
)

# ── 3. Time horizon ───────────────────────────────────────────────────────────
lookback_length = 12.0
t_step          = 0.25
tau = np.arange(0, lookback_length + 1e-5, t_step)
T   = len(tau)

# ── 4. Time-varying target set: translating disk ──────────────────────────────
# Disk of radius 1.0 that linearly translates from c_start to c_end across tau.
c_start = np.array([-4.0, -4.0])
c_end   = np.array([ 4.0,  4.0])
radius  = 1.0

nx, ny = g.pts_each_dim
target_stack = np.zeros((nx, ny, T), dtype=np.float32)
for k in range(T):
    alpha   = tau[k] / tau[-1] if tau[-1] > 0 else 0.0
    center  = (1.0 - alpha) * c_start + alpha * c_end
    sdf_k   = CylinderShape(g, ignore_dims=[], center=center.tolist(), radius=radius)
    target_stack[..., k] = sdf_k.astype(np.float32)
print(f"Target stack shape (nx, ny, T): {target_stack.shape}")

# ── 5. Random constraint set (single, time-invariant) ─────────────────────────
rng = np.random.default_rng(seed=7)

def random_constraint_set(grid: Grid, rng: np.random.Generator) -> np.ndarray:
    shape_type = rng.integers(0, 3)
    offset     = rng.uniform(-3.0, 3.0, size=grid.dims)
    if shape_type == 0:
        r = rng.uniform(6.0, 9.0)
        print(f"  [constraint] CylinderShape  | center={np.round(offset,2).tolist()} | radius={r:.2f}")
        return CylinderShape(grid, ignore_dims=[], center=offset.tolist(), radius=r)
    elif shape_type == 1:
        hw = rng.uniform(6.0, 9.0, size=grid.dims)
        print(f"  [constraint] ShapeRectangle | center={np.round(offset,2).tolist()} | half_widths={np.round(hw,2).tolist()}")
        return ShapeRectangle(grid, target_min=offset - hw, target_max=offset + hw)
    else:
        sa = rng.uniform(6.0, 9.0, size=grid.dims)
        print(f"  [constraint] ShapeEllipsoid | center={np.round(offset,2).tolist()} | semi_axes={np.round(sa,2).tolist()}")
        return ShapeEllipsoid(grid, center=offset.tolist(), semiAxLen=sa.tolist())

# Rejection-sample so the constraint set fully contains the target tube.
# Containment check: at every grid point that lies in ANY target slice
# (target_min <= 0), the constraint SDF must also be <= 0.
target_tube = target_stack.min(axis=-1)
max_attempts = 200
for attempt in range(max_attempts):
    constraint_set = random_constraint_set(g, rng)
    inside_tube = target_tube <= 0
    if np.all(constraint_set[inside_tube] <= 0):
        print(f"  [constraint] accepted on attempt {attempt + 1}")
        break
else:
    raise RuntimeError(
        f"Could not sample a constraint set containing the target tube "
        f"after {max_attempts} attempts. Loosen the sampler or shrink the target."
    )

# ── 6. Solver call (time-varying target, time-invariant constraint) ───────────
compMethods = {
    "TargetSetMode":   "minVWithV0",
    "ObstacleSetMode": "maxVWithObstacle",
}

result = HJSolver(
    dynamics_obj     = plane,
    grid             = g,
    multiple_value   = [target_stack, -constraint_set],
    tau              = tau,
    compMethod       = compMethods,
    saveAllTimeSteps = True,
    accuracy         = "medium",
    verbose          = True,
)
print(f"Result shape: {result.shape}")

# ── 7. Plot: animated BRS over time (mirrors dubins_3d_reach_avoid.py) ────────
brs_color = "blue"
x_axis    = np.linspace(g.min[0], g.max[0], nx)
y_axis    = np.linspace(g.min[1], g.max[1], ny)

N = result.shape[2]

def _fill(z):
    z_m = np.where(z <= 0, z, np.nan)
    return go.Heatmap(
        x=x_axis, y=y_axis, z=z_m.T,
        colorscale=[[0, brs_color], [1, brs_color]],
        zmin=-0.1, zmax=0.0,
        opacity=0.3, showscale=False,
        name="BRS (fill)", hoverinfo="skip",
    )

def _boundary(z, color=brs_color, name="BRS", dash="solid"):
    return go.Contour(
        x=x_axis, y=y_axis, z=z.T,
        contours=dict(start=0, end=0, size=1),
        contours_coloring="none",
        line=dict(color=color, width=2, dash=dash),
        showscale=False, name=name,
    )

# Static constraint trace — red, same every frame
constraint_trace = _boundary(-constraint_set, color="red", name="Constraint", dash="dash")

# Animated BRS frames + animated target boundary (target is time-varying!)
# Slice indexing convention: result[..., k] corresponds to tau[N-1-k] under
# saveAllTimeSteps semantics in HJSolver (newest at -1, oldest at 0).
# We display in forward-time order: frame k → tau[k] → result slice index N-1-k.
frames = [
    go.Frame(
        data=[
            _fill(result[:, :, N - 1 - k]),
            _boundary(result[:, :, N - 1 - k]),
            _boundary(target_stack[:, :, k], color="green", name="Target", dash="dash"),
        ],
        traces=[0, 1, 2],
        name=str(k),
    )
    for k in reversed(range(N))
]

slider_steps = [
    dict(
        args=[[str(k)], dict(frame=dict(duration=0, redraw=True),
                             mode="immediate", transition=dict(duration=0))],
        label=f"t={tau[k]:.2f}", method="animate",
    )
    for k in reversed(range(N))
]

fig = go.Figure(
    data=[
        _fill(result[:, :, N - 1]),                                    # trace 0 (BRS fill)
        _boundary(result[:, :, N - 1]),                                # trace 1 (BRS edge)
        _boundary(target_stack[:, :, 0], color="green",
                  name="Target", dash="dash"),                          # trace 2 (target — animated)
        constraint_trace,                                               # trace 3 (constraint — static)
    ],
    frames=frames,
    layout=go.Layout(
        title="Plane2D Reach-Avoid BRS — Time-Varying Target",
        xaxis=dict(title="x", range=[g.min[0], g.max[0]], constrain="domain"),
        yaxis=dict(title="y", range=[g.min[1], g.max[1]], scaleanchor="x", scaleratio=1),
        legend=dict(x=1.02, y=1.0),
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.08, x=0.5, xanchor="center",
            buttons=[
                dict(label="Play",  method="animate",
                     args=[None, dict(frame=dict(duration=100, redraw=True),
                                      fromcurrent=True, transition=dict(duration=0))]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate", transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            steps=slider_steps,
            currentvalue=dict(prefix="Time: ", visible=True, xanchor="center"),
            pad=dict(t=50),
        )],
    ),
)

fig.show()
