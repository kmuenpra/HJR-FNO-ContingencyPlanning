"""
Standalone demo for the topological PRM.

Scene = the SAME environment as mppi_src/navigation2d.py (Navigation2DEnv): a
20x20 m map with 10 random circular obstacles (seed=42, kept out of the safe
regions) and start (-9,-9) -> goal (8,8), reproduced with the same seed (the real
env clears its map to hide obstacles as lidar-"unknown"; here we keep them).

The topological paths can be constrained to lie inside a REACHABLE SET, from one
of two backends (REACHABLE_SOURCE):
  - "fno"  : HJR-FNO predicted reachable set  -> run in the `rrtx` conda env
  - "odp"  : optimized_dp ground-truth HJ reach-avoid BRT (HJSolver)
             -> run in the `odp` conda env (has HeteroCL; no torch needed here)
  - "none" : collision-only (no reachable-set constraint)

Run:
    # fno backend (rrtx env has torch + the HJR-FNO/rrtx chain):
    python topo_prm/demo.py
    # odp backend (odp env has HeteroCL; set REACHABLE_SOURCE = "odp" below):
    python topo_prm/demo.py
Output figure: topo_prm/demo_output.png
"""

import os
import sys
import time
import types
from pathlib import Path

# make the sibling modules importable whether run as a script or a module
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO_ROOT = Path(_HERE).resolve().parents[0]

import matplotlib

matplotlib.use("Agg")  # headless-safe; writes a PNG
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# torch is only needed by the "fno" backend (and it's present in the rrtx env);
# keep it optional so the "odp"/"none" backends run in the torch-less odp env.
try:
    import torch
except Exception:
    torch = None

from obstacle_map_2d import ObstacleMap, generate_random_obstacles
from topo_prm import TopoPRM, GraphNode

# ======================================================================
# Configuration
# ======================================================================
REACHABLE_SOURCE = "odp"   # "none" | "fno" | "odp"
TF_REACH = 8.0             # reach horizon [s] (fno) / lookback length (odp)

# odp HJSolver settings (used only when REACHABLE_SOURCE == "odp")
ODP_PTS = (60, 60, 36)     # grid: (Nx, Ny, Ntheta) over [-10,10]x[-10,10]x[-pi,pi]
ODP_TSTEP = 0.2            # time step for tau

# --- Navigation2DEnv scene parameters (mirrors mppi_src/envs/navigation_2d.py) ---
ENV_SEED = 42
SAFE_RADIUS = 2.0
SAFE_REGION_CENTERS = [(-5.0, -7.5), (0.0, 0.0), (-2.5, 5.0), (5.0, 6.0)]
SAFE_REGIONS = [(cx, cy, SAFE_RADIUS) for cx, cy in SAFE_REGION_CENTERS]
START = np.array([-9.0, -9.0])
GOAL = np.array([8.0, 8.0])


# ======================================================================
# Scene
# ======================================================================
def build_scene() -> ObstacleMap:
    """Reproduce the Navigation2DEnv obstacle map (all obstacles kept, not cleared)."""
    omap = ObstacleMap(map_size=(20, 20), cell_size=0.1)
    generate_random_obstacles(
        obstacle_map=omap,
        random_x_range=(-7.5, 7.5),
        random_y_range=(-7.5, 7.5),
        num_circle_obs=10,
        radius_range=(0.5, 1.5),
        num_rectangle_obs=0,
        width_range=(2, 2),
        height_range=(2, 2),
        max_iteration=1000,
        seed=ENV_SEED,
        keepout_circles=SAFE_REGIONS,
    )
    return omap


# ======================================================================
# Reachable-set backend: HJR-FNO (predicted)
# ======================================================================
def build_reachability_fno(obstacles):
    """HJR-FNO predicted reachable set over SAFE_REGIONS (obstacle-aware).
    Returns the oracle or None if HJR-FNO deps aren't available (e.g. wrong env)."""
    try:
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from HJR_FNO.HJR_FNO3d import HJR_FNO

        dummy_env = types.SimpleNamespace(
            obs_circle=[], obs_rectangle=[], obs_boundary=[], unknown_obs_circle=[]
        )
        hjr = HJR_FNO(env=dummy_env, safe_regions=SAFE_REGIONS, Tf_reach=TF_REACH)
        hjr.scenario_enable = False
        hjr.scenario_parallel = False
        hjr.feasibility_source = "feasible_region"  # heading-independent (geometric PRM)
        if obstacles:
            hjr.update_obs(obstacles)
        return hjr
    except Exception as e:  # noqa: BLE001
        print(f"[fno] disabled (could not initialize; run in the rrtx env): {e}")
        return None


def fno_feasible_fn(hjr):
    return lambda pts: hjr.points_feasible(np.asarray(pts, float), thetas=None)


def draw_reachable_fno(ax, hjr) -> None:
    for i in range(hjr.num_safe_regions):
        fr = hjr.feasible_region[i]
        if torch is not None and torch.is_tensor(fr):
            fr = fr.cpu().numpy()
        fr = np.asarray(fr)
        X, Y = (hjr.X, hjr.Y) if hjr.obs_list[i] else (hjr.X_fine, hjr.Y_fine)
        if hjr.safe_margin[i] > fr.min():
            ax.contourf(
                X + hjr.safe_regions[i][0], Y + hjr.safe_regions[i][1], fr,
                levels=[fr.min(), hjr.safe_margin[i]],
                colors="#ADD8E6", alpha=0.35, zorder=0,
            )


# ======================================================================
# Reachable-set backend: optimized_dp (ground-truth HJ reach-avoid BRT)
# ======================================================================
def build_reachability_odp(obstacles):
    """Ground-truth reach-avoid BRT via optimized_dp HJSolver, per safe region,
    then unioned. Returns dict(gx, gy, U) where U(x,y) = min_i (max_theta V_i) at
    t=0; feasible <=> U <= 0. Cached to an .npz (keyed by scene/grid) since the
    HJ PDE solve is slow. MUST be run in the `odp` conda env (needs HeteroCL)."""
    Nx, Ny, Nth = ODP_PTS
    cache = Path(_HERE) / f"odp_reachset_seed{ENV_SEED}_{Nx}x{Ny}x{Nth}_Tf{TF_REACH:g}.npz"
    if cache.exists():
        d = np.load(cache)
        print(f"[odp] loaded cached BRT -> {cache.name}")
        return dict(gx=d["gx"], gy=d["gy"], U=d["U"])

    try:
        import math

        odp_root = str(_REPO_ROOT / "optimized_dp")
        if odp_root not in sys.path:
            sys.path.insert(0, odp_root)
        from odp.Grid import Grid
        from odp.Shapes import CylinderShape
        from odp.solver import HJSolver
        from odp.dynamics.DubinsCar2 import DubinsCar2
    except Exception as e:  # noqa: BLE001
        print(f"[odp] disabled (could not import odp/HeteroCL; run in the odp env): {e}")
        return None

    g = Grid(
        minBounds=np.array([-10.0, -10.0, -math.pi]),
        maxBounds=np.array([10.0, 10.0, math.pi]),
        dims=3,
        pts_each_dim=np.array([Nx, Ny, Nth]),
        periodicDims=[2],
    )
    # DubinsCar2 matching the nav env / HJR-FNO: v in [0,1], w in [-1,1], no dstb.
    car = DubinsCar2(uMin=[0.0, -1.0], uMax=[1.0, 1.0], dMax=[0.0, 0.0, 0.0],
                     uMode="min", dMode="max")

    # obstacle set = union of all map obstacles (SDF <0 inside -> np.minimum)
    obs_sdf = None
    for (ox, oy, orad) in obstacles:
        c = CylinderShape(g, ignore_dims=[2], center=[ox, oy, 0.0], radius=orad)
        obs_sdf = c if obs_sdf is None else np.minimum(obs_sdf, c)
    if obs_sdf is None:  # no obstacles -> never-inside sentinel
        obs_sdf = np.full(tuple(g.pts_each_dim), 1e6)

    tau = np.arange(0.0, TF_REACH + 1e-5, ODP_TSTEP)
    comp = {"TargetSetMode": "minVWithV0", "ObstacleSetMode": "maxVWithObstacle"}

    U = None
    for k, (cx, cy, r) in enumerate(SAFE_REGIONS):
        target = CylinderShape(g, ignore_dims=[2], center=[cx, cy, 0.0], radius=r)
        print(f"[odp] solving reach-avoid BRT for region {k} at ({cx},{cy}) ...")
        res = HJSolver(dynamics_obj=car, grid=g, multiple_value=[target, obs_sdf],
                       tau=tau, compMethod=comp, saveAllTimeSteps=True, accuracy="medium", verbose=False)
        brt = res[..., 0] if res.ndim == 4 else res       # V(x,y,theta) at t=0
        Vmax = np.max(brt, axis=2)                         # max over theta (all-heading)
        U = Vmax if U is None else np.minimum(U, Vmax)     # union over regions

    gx = np.asarray(g.grid_points[0])
    gy = np.asarray(g.grid_points[1])
    np.savez(cache, gx=gx, gy=gy, U=U)
    print(f"[odp] saved BRT cache -> {cache.name}")
    return dict(gx=gx, gy=gy, U=U)


def odp_feasible_fn(reach):
    interp = RegularGridInterpolator(
        (reach["gx"], reach["gy"]), reach["U"], bounds_error=False, fill_value=1e6
    )
    return lambda pts: interp(np.atleast_2d(np.asarray(pts, float))[:, :2]) <= 0.0


def draw_reachable_odp(ax, reach) -> None:
    gx, gy, U = reach["gx"], reach["gy"], reach["U"]
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    if U.min() < 0.0:
        ax.contourf(X, Y, U, levels=[U.min(), 0.0],
                    colors="#ADD8E6", alpha=0.35, zorder=0)


# ======================================================================
# Plot helpers
# ======================================================================
def draw_safe_regions(ax) -> None:
    for k, (cx, cy, r) in enumerate(SAFE_REGIONS):
        ax.add_patch(plt.Circle((cx, cy), r, color="green", alpha=0.25, zorder=1,
                                label="safe region" if k == 0 else None))


def draw_roadmap(ax, planner: TopoPRM) -> None:
    for node in planner.graph:
        for nb in node.neighbors:
            if node.id < nb.id:
                ax.plot([node.pos[0], nb.pos[0]], [node.pos[1], nb.pos[1]],
                        color="0.8", lw=0.5, zorder=2)
    guards = np.array([n.pos for n in planner.graph if n.type == GraphNode.GUARD])
    conns = np.array([n.pos for n in planner.graph if n.type == GraphNode.CONNECTOR])
    if len(guards):
        ax.scatter(guards[:, 0], guards[:, 1], s=10, c="tab:blue", zorder=3, label="guards")
    if len(conns):
        ax.scatter(conns[:, 0], conns[:, 1], s=10, c="tab:orange", zorder=3, label="connectors")


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    omap = build_scene()
    obstacles = [
        (float(o.center[0]), float(o.center[1]), float(o.radius))
        for o in omap.circle_obs_list
    ]

    # select reachable-set backend
    feasible_fn = None
    reach_draw = None
    if REACHABLE_SOURCE == "fno":
        hjr = build_reachability_fno(obstacles)
        if hjr is not None:
            feasible_fn = fno_feasible_fn(hjr)
            reach_draw = lambda ax: draw_reachable_fno(ax, hjr)
    elif REACHABLE_SOURCE == "odp":
        reach = build_reachability_odp(obstacles)
        if reach is not None:
            feasible_fn = odp_feasible_fn(reach)
            reach_draw = lambda ax: draw_reachable_odp(ax, reach)
    elif REACHABLE_SOURCE != "none":
        raise ValueError(f"unknown REACHABLE_SOURCE={REACHABLE_SOURCE!r}")

    planner = TopoPRM(
        omap,
        clearance=0.4,
        resolution=0.2,
        sample_inflate=(2.0, 7.0),
        max_sample_num=800,
        max_sample_time=0.5,
        max_raw_path=20,
        ratio_to_short=3.0,
        reserve_num=6,
        seed=1,
    )

    # Bake the reachable set into a boolean grid ONCE (fast O(1) lookups after).
    # Re-call rasterize_feasible(...) + find_topo_paths() whenever the set updates.
    if feasible_fn is not None:
        t0 = time.time()
        planner.rasterize_feasible(feasible_fn)
        print(f"rasterize reachable set: {1e3 * (time.time() - t0):.1f} ms")

    t0 = time.time()
    paths = planner.find_topo_paths(START, GOAL)
    print(f"find_topo_paths: {1e3 * (time.time() - t0):.1f} ms")

    # ---- stats ----
    n_guard = sum(1 for n in planner.graph if n.type == GraphNode.GUARD)
    n_conn = sum(1 for n in planner.graph if n.type == GraphNode.CONNECTOR)
    print(f"backend={REACHABLE_SOURCE} | graph: {n_guard} guards, {n_conn} connectors")
    print(f"raw paths: {len(planner.raw_paths)}  ->  distinct topo paths: {len(paths)}")
    for k, p in enumerate(paths):
        print(f"  path {k}: length = {planner.path_length(p):.2f} m, {len(p)} waypoints")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7, 7), layout="tight")
    if reach_draw is not None:
        reach_draw(ax)
    omap.render(ax, zorder=1)
    draw_safe_regions(ax)
    draw_roadmap(ax, planner)

    for p in planner.raw_paths:
        ax.plot(p[:, 0], p[:, 1], color="0.85", lw=0.8, zorder=2)

    cmap = plt.get_cmap("tab10")
    for k, p in enumerate(paths):
        ax.plot(p[:, 0], p[:, 1], color=cmap(k % 10), lw=2.5, zorder=5,
                label=f"topo {k} ({planner.path_length(p):.1f} m)")

    ax.scatter(*START, marker="o", s=80, color="green", zorder=6, label="start")
    ax.scatter(*GOAL, marker="*", s=140, color="red", zorder=6, label="goal")
    label = {"fno": "inside FNO reachable set",
             "odp": "inside odp reach-avoid BRT",
             "none": "collision-only"}[REACHABLE_SOURCE if feasible_fn is not None else "none"]
    ax.set_title(f"Topological PRM — {len(paths)} distinct path(s) [{label}]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="lower right", fontsize=8)

    out = os.path.join(_HERE, "demo_output.png")
    fig.savefig(out, dpi=130)
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
