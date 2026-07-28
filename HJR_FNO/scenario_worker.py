# =====================================================================
# Torch-free numerics + scenario-optimization logic
#
# This module deliberately imports ONLY numpy / scipy / odp (no torch). That
# makes it safe to run inside ProcessPoolExecutor worker processes (spawn):
# each worker re-imports this module without dragging in torch / CUDA, so the
# per-obstacle scenario optimization can fan out across the M changed safe
# regions on the CPU while the GPU FNO query stays in the parent process.
#
# HJR_FNO3d.py imports everything back from here, so the original module-level
# names (computeGradients, eval_u, _ReachValueCache, _scenario_required_N, ...)
# keep resolving exactly as before. There is ONE implementation of each.
# =====================================================================
import os
import math
import sys as _sys
from pathlib import Path
from typing import Tuple, List

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import binom


# =====================================================================
# odp bootstrap (path-injected, not pip-installed) + heterocl stub.
# Mirrors the bootstrap that used to live in HJR_FNO3d.py so this module is
# self-contained and importable from a fresh (spawned) worker interpreter.
# =====================================================================
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ODP_ROOT = str(_REPO_ROOT / "optimized_dp")
if _ODP_ROOT not in _sys.path:
    _sys.path.insert(0, _ODP_ROOT)

# DubinsCar2.py imports `heterocl` at module top (only its hcl-based GPU-solver
# methods need it). We use only the pure-Python methods, so if heterocl is
# absent (e.g. the rrtx / HJR-FNO conda envs) install a harmless stub.
try:
    import heterocl  # noqa: F401
except Exception:
    import types as _types
    _sys.modules["heterocl"] = _types.ModuleType("heterocl")

from odp.Grid import Grid                          # noqa: E402  (re-exported)
from odp.dynamics.DubinsCar2 import DubinsCar2     # noqa: E402  (re-exported)


# =====================================================================
# Derivative functions (HelperOC-style), adapted to the odp Grid API
#   odp Grid attributes:  .dims  .dx  .grid_points[i] (1-D)  .pDim  .min  .max
# =====================================================================
def upwindFirstENO2(grid, data: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Second order ENO approximation of first derivative."""
    dxInv = 1.0 / grid.dx[dim]
    stencil = 2
    gdata = add_ghost_cells(data, dim, stencil)

    D1 = dxInv * np.diff(gdata, axis=dim)
    D2 = 0.5 * dxInv * np.diff(D1, axis=dim)

    D1 = strip_dim(D1, dim, 1, 1)

    derivL = strip_dim(D1, dim, 0, 1)
    derivR = strip_dim(D1, dim, 1, 0)

    D2_left  = strip_dim(D2, dim, 0, 2)
    D2_right = strip_dim(D2, dim, 1, 1)

    derivL = derivL + grid.dx[dim] * D2_left
    derivR = derivR - grid.dx[dim] * D2_right

    return derivL, derivR


def upwindFirstWENO5(grid, data: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fifth order WENO approximation - simplified implementation (falls back to ENO2)."""
    dxInv = 1.0 / grid.dx[dim]
    stencil = 3
    gdata = add_ghost_cells(data, dim, stencil)
    D1 = dxInv * np.diff(gdata, axis=dim)
    derivL, derivR = upwindFirstENO2(grid, data, dim)
    return derivL, derivR


def add_ghost_cells(data: np.ndarray, dim: int, stencil: int) -> np.ndarray:
    """Add ghost cells by extrapolation."""
    shape = list(data.shape)
    shape[dim] += 2 * stencil
    gdata = np.zeros(shape)

    slices = [slice(None)] * data.ndim
    slices[dim] = slice(stencil, -stencil)
    gdata[tuple(slices)] = data

    for i in range(stencil):
        slices_src = [slice(None)] * data.ndim
        slices_src[dim] = stencil
        slices_dst = [slice(None)] * data.ndim
        slices_dst[dim] = stencil - i - 1
        gdata[tuple(slices_dst)] = gdata[tuple(slices_src)]

        slices_src[dim] = -stencil - 1
        slices_dst[dim] = -stencil + i
        gdata[tuple(slices_dst)] = gdata[tuple(slices_src)]

    return gdata


def strip_dim(data: np.ndarray, dim: int, left: int, right: int) -> np.ndarray:
    """Strip entries from left and right along dimension."""
    slices = [slice(None)] * data.ndim
    slices[dim] = slice(left, -right if right > 0 else None)
    return data[tuple(slices)]


def computeGradients(grid, data: np.ndarray) -> List[np.ndarray]:
    """Compute central-difference gradients using the upwind scheme."""
    derivC = []
    for dim in range(grid.dims):
        derivL, derivR = upwindFirstWENO5(grid, data, dim)
        deriv = 0.5 * (derivL + derivR)
        derivC.append(deriv)
    return derivC


def eval_u(grid, gradients: List[np.ndarray], x: np.ndarray) -> np.ndarray:
    """Evaluate gradient at point x using interpolation (odp Grid)."""
    deriv = np.zeros(grid.dims)

    for dim in range(grid.dims):
        x_eval = np.array(x, dtype=float)
        if dim in grid.pDim:
            period = grid.max[dim] - grid.min[dim]
            while x_eval[dim] > grid.max[dim]:
                x_eval[dim] -= period
            while x_eval[dim] < grid.min[dim]:
                x_eval[dim] += period

        interp = RegularGridInterpolator(
            grid.grid_points, gradients[dim],
            bounds_error=False, fill_value=None
        )
        deriv[dim] = interp(x_eval)

    if np.any(np.isnan(deriv)):
        for dim in range(grid.dims):
            idx = np.argmin(np.abs(grid.grid_points[dim] - x[dim]))
            if dim == 0:
                deriv[dim] = gradients[dim][idx, :, :].mean()
            elif dim == 1:
                deriv[dim] = gradients[dim][:, idx, :].mean()
            else:
                deriv[dim] = gradients[dim][:, :, idx].mean()

    return deriv


# =====================================================================
# Scenario optimization for the REACH-AVOID safe margin (delta_hat)
#
# Heterocl-free port of HJR_FNO/verfication/scenario_optimization_reach.py
# ("per_c" mode, one obstacle config at a time). The original computes 3-D
# spatial derivatives with odp.solver.computeSpatDerivArray, which requires
# heterocl (only the `odp` conda env has it); here we reuse the pure-numpy
# `computeGradients` above so this runs in the planning env (rrtx / HJR-FNO).
#
# Goal: given the FNO value tube V(x,y,theta,t) for an obstacle config, find a
# scalar threshold delta_hat such that the recovered set {V(.,fully-grown) < delta_hat}
# is, with high probability, contained in the true reach-avoid set (states from
# which the FNO-induced bang-bang controller reaches the target without hitting
# the obstacle under worst-case wind).
#
# Time convention (verified empirically against this model): V[...,0] = fully-grown
# BRT (loosest sublevel set), V[...,-1] = terminal/target slice.
#
# Cost:  J = min_k max( ell(s_k), max_{j<=k} G(s_j) ),  failure iff J >= 0,
#   where ell = target SDF (<0 inside target), G = -obstacle SDF (>0 inside obstacle).
# =====================================================================
def _scenario_required_N(eps: float, beta: float) -> int:
    """Scenario-theorem sample count for (eps, beta). In the robust (k-outlier)
    scheme this is the SAMPLE BUDGET N; whatever margin it has over the
    feasibility floor ln(beta)/ln(1-eps) is what buys the failure budget k_max."""
    return int(math.ceil((2.0 / eps) * (math.log(1.0 / beta) + 1.0)))


def _scenario_max_outliers(N: int, eps: float, beta: float) -> int:
    """k_max(N, eps, beta): largest failure count k whose binomial tail still
    certifies the target (eps, beta), i.e.

        B(eps; N, k) = sum_{i=0}^k C(N,i) eps^i (1-eps)^(N-i)
                     = binom.cdf(k, N, eps)  <=  beta.

    B is increasing in k, so we scan up and stop at the first k that busts the
    budget. Returns -1 when even k=0 fails -> N is below the feasibility floor
    ln(beta)/ln(1-eps) and cannot certify eps at any delta."""
    best = -1
    for k in range(N + 1):
        if binom.cdf(k, N, eps) <= beta:
            best = k
        else:
            break                       # cdf increasing in k -> no later k works
    return best


def _scenario_delta_grid(delta_start: float, delta_floor: float, step: float):
    """Descending grid [delta_start, delta_start-step, ..., >= delta_floor].
    The sweep walks this top-down; the first delta whose empirical failure count
    k <= k_max is the largest (biggest recovered set) certifiable delta."""
    step = abs(float(step))
    n = int(math.floor((delta_start - delta_floor) / step)) + 1
    return [delta_start - j * step for j in range(max(n, 1))]


def _scenario_epsilon_from_Nk(N: int, k: int, beta: float) -> float:
    """Tightest eps with binom.cdf(k, N, eps) <= beta (bisection root of the
    decreasing-in-eps tail). Reporting only: guaranteed <= target eps when
    k <= k_max. Returns 1.0 for the degenerate k >= N."""
    if k >= N:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if binom.cdf(k, N, mid) > beta:   # violation prob too high -> raise eps
            lo = mid
        else:
            hi = mid
    return hi


def _scenario_wrap_pi(a):
    """Wrap angle(s) into [-pi, pi) — the FNO grid's theta convention."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class _ReachValueCache:
    """Per-obstacle reach-avoid value cache (heterocl-free port of
    scenario_optimization_reach.CachedConstraint).

    Holds V(x,y,theta,t) plus interpolators for each V time slice, the 3-D
    spatial gradient (dVdx, dVdy, dVdtheta), and the 2-D target (ell) /
    avoidance (G) fields. Gradients use the numpy WENO/ENO `computeGradients`
    (no odp/heterocl). All fields are in the LOCAL frame of a safe region
    (target at the origin), matching how obstacle SDFs / reachable sets are
    stored in HJR_FNO.
    """

    def __init__(self, obstacle_sdf_2d, V_full, x_axis, y_axis, theta_axis,
                 target_sdf_2d, odp_grid):
        self.V_grid = np.asarray(V_full, dtype=np.float32)          # (Nx,Ny,Nth,T)
        T = self.V_grid.shape[-1]
        pts3d = (np.asarray(x_axis), np.asarray(y_axis), np.asarray(theta_axis))
        kw = dict(bounds_error=False, fill_value=None)

        self._V_slice_interps = [
            RegularGridInterpolator(pts3d, self.V_grid[:, :, :, i], **kw) for i in range(T)
        ]
        # 3-D spatial gradient per time slice (numpy WENO -> [d/dx, d/dy, d/dtheta]).
        self._grad_interps = []
        for i in range(T):
            gx, gy, gth = computeGradients(
                odp_grid, np.ascontiguousarray(self.V_grid[:, :, :, i], dtype=np.float64))
            self._grad_interps.append((
                RegularGridInterpolator(pts3d, gx,  **kw),
                RegularGridInterpolator(pts3d, gy,  **kw),
                RegularGridInterpolator(pts3d, gth, **kw)))

        pts2d = (np.asarray(x_axis), np.asarray(y_axis))
        self._ell_int = RegularGridInterpolator(pts2d, np.asarray(target_sdf_2d), **kw)   # <0 inside target
        self._G_int   = RegularGridInterpolator(pts2d, -np.asarray(obstacle_sdf_2d), **kw)  # >0 inside obstacle

    def value_at_full_BRS(self, s):              # s: (B,3) -> (B,)
        return self._V_slice_interps[0](s)        # slice 0 = fully-grown BRT

    def grad_at_indices(self, s, k_arr):          # s: (B,3), k_arr: (B,) -> (B,3)
        B = s.shape[0]
        out = np.empty((B, 3), dtype=np.float64)
        for k in np.unique(k_arr):
            msk = (k_arr == k)
            gx, gy, gth = self._grad_interps[int(k)]
            out[msk, 0] = gx(s[msk]); out[msk, 1] = gy(s[msk]); out[msk, 2] = gth(s[msk])
        return out

    def find_max_safe_time_index(self, s, delta=0.0):
        """Largest slice index k with V[...,k](s) <= delta (tightest sublevel set
        still containing s). Index 0 = fully-grown (loosest). Fallback 0."""
        T = len(self._V_slice_interps)
        B = s.shape[0]
        t_i = np.zeros(B, dtype=int)
        found = np.zeros(B, dtype=bool)
        for k in range(T - 1, -1, -1):
            new = (~found) & (self._V_slice_interps[k](s) <= delta)
            if new.any():
                t_i[new] = k; found[new] = True
                if found.all():
                    break
        return t_i

    def ell(self, xy):
        return self._ell_int(xy)

    def G(self, xy):
        return self._G_int(xy)


def rollout_cost(cache, s0, car, dt, delta=0.0):
    """Constrained reach-avoid cost under the FNO-induced bang-bang controller
    and worst-case disturbance (`car`):

        J = min_k max( ell(s_k), max_{j<=k} G(s_j) ),  failure iff J >= 0.

    s0: (B, 3) states (x, y, theta) in the region's local frame. Returns (B,).

    Vectorized over the B trajectories: at each integration step k all active
    trajectories advance simultaneously via batched numpy ops. Only the time
    axis (k) carries a data dependency (Euler integration); the per-trajectory
    axis is independent, so optCtrl/optDstb/dynamics are inlined as elementwise
    np.where branches mirroring DubinsCar2.{optCtrl,optDstb,dynamics}_inPython."""
    s   = np.asarray(s0, dtype=np.float64).copy()
    T_slices = len(cache._grad_interps)

    # Cache mode flags / bounds once (avoids per-step attribute lookups).
    u_is_min = (car.uMode == "min")
    d_is_max = (car.dMode == "max")
    speedMin, speedMax = car.speedMin, car.speedMax
    wMin, wMax = car.wMin, car.wMax
    dMax = np.asarray(car.dMax, dtype=np.float64)  # (3,)

    t_start = cache.find_max_safe_time_index(s.astype(np.float32), delta=delta)
    n_steps = T_slices - 1 - t_start
    max_steps = int(n_steps.max()) if t_start.size > 0 else 0

    running_G = cache.G(s[:, :2]).copy()
    cost      = np.maximum(cache.ell(s[:, :2]), running_G)

    for k in range(max_steps):
        active = k < n_steps
        if not active.any():
            break

        # subset of the B trajectories still integrating at step k
        ai = np.flatnonzero(active)
        sa = s[ai]                                   # (n_active, 3) view-copy below

        # k-th time slice per active trajectory -> batched spatial gradient
        slice_idx = t_start[ai] + k
        grad = cache.grad_at_indices(sa.astype(np.float32), slice_idx)  # (n_active, 3)
        gx, gy, gth = grad[:, 0], grad[:, 1], grad[:, 2]
        th = sa[:, 2]
        cos_th, sin_th = np.cos(th), np.sin(th)

        #NOTE This vectorization of rollout trajectory is hard-coded for DubinsCar2.py
        # ---- optCtrl (bang-bang), elementwise over active trajectories ----
        coeff = gx * cos_th + gy * sin_th
        if u_is_min:
            opt_w     = np.where(gth   > 0, wMin,     wMax)
            opt_speed = np.where(coeff > 0, speedMin, speedMax)
        else:
            opt_w     = np.where(gth   < 0, wMin,     wMax)
            opt_speed = np.where(coeff < 0, speedMin, speedMax)

        # ---- optDstb (worst-case wind), elementwise ----
        if d_is_max:
            d1 = np.where(gx  >= 0,  dMax[0], -dMax[0])
            d2 = np.where(gy  >= 0,  dMax[1], -dMax[1])
            d3 = np.where(gth >= 0,  dMax[2], -dMax[2])
        else:
            d1 = np.where(gx  >= 0, -dMax[0],  dMax[0])
            d2 = np.where(gy  >= 0, -dMax[1],  dMax[1])
            d3 = np.where(gth >= 0, -dMax[2],  dMax[2])

        # ---- dynamics + Euler step, written back to the active rows ----
        s[ai, 0] += dt * (opt_speed * cos_th + d1)
        s[ai, 1] += dt * (opt_speed * sin_th + d2)
        s[ai, 2]  = _scenario_wrap_pi(th + dt * (opt_w + d3))

        running_G = np.maximum(running_G, cache.G(s[:, :2]))
        cost      = np.minimum(cost, np.maximum(cache.ell(s[:, :2]), running_G))
    return cost


def sample_states(cache, n_target, delta, rng, grid_min, grid_max,
                  batch=8192, max_tries=400):
    """Uniform rejection sampling of states (x, y, theta) keeping
    V(s, fully-grown) < delta. Returns (n_target, 3)."""
    x_lo = np.array([grid_min[0], grid_min[1], -math.pi])
    x_hi = np.array([grid_max[0], grid_max[1],  math.pi])
    kept, n_kept = [], 0
    for _ in range(max_tries):
        s_cand = rng.uniform(x_lo, x_hi, size=(batch, 3)).astype(np.float32)
        V = cache.value_at_full_BRS(s_cand)
        mask = (V < delta) if math.isfinite(delta) else np.ones_like(V, dtype=bool)
        if mask.any():
            kept.append(s_cand[mask]); n_kept += int(mask.sum())
            if n_kept >= n_target:
                break
    if n_kept == 0:
        raise RuntimeError(
            f"Sampled 0/{n_target} states with V<{delta} after {max_tries} tries "
            f"(sublevel set may be empty on this grid).")
    return np.concatenate(kept, axis=0)[:n_target]


def scenario_delta_hat_worker(V_full, obstacle_sdf, target2d, x_axis, y_axis,
                              theta_array, grid, car, cfg, verbose=True):
    """Run the per-obstacle scenario optimization for a freshly-estimated
    reachable tube and return the certified safe margin delta_hat.

    All inputs are picklable (numpy arrays, odp Grid, DubinsCar2, plain cfg dict),
    so this free function can run either in-process or in a ProcessPoolExecutor
    worker. The parent computes target2d via shapeCylinder and passes it in.

    Robust (k-outlier) certification: with (eps, beta) fixed, the sample budget
    N = required_N(eps, beta) buys a failure budget k_max = max_outliers(N, eps,
    beta). We march delta DOWN a fixed grid from delta_init to delta_floor and
    return the FIRST (= largest, biggest recovered set) delta whose empirical
    failure count k = #{J >= 0} satisfies k <= k_max. By monotonicity of the
    sublevel set in delta, first-pass on a descending grid is the largest
    certifiable set (break-on-first-pass).

    Guarantee (w.p. >= 1 - beta):  Pr_{x in S(delta*)}( V(x,0,c) >= 0 ) <= eps.

    @params
    - V_full       : (Nx, Ny, Nth, T) FNO value tube on `grid` (numpy)
    - obstacle_sdf : RAW obstacle SDF (negative inside), 2-D or 3-D (theta-indep)
    - target2d     : 2-D target SDF (negative inside target) on (x_axis, y_axis)
    - cfg          : dict(eps, beta, max_tries, delta_floor, delta_init,
                          delta_step, seed, dt, grid_min, grid_max)
    @return (delta_hat, report) where delta_hat is a float in
            [delta_floor, delta_init] and report is a dict with the certification
            stats at the returned level: {eps, beta, N, kmax, k, success_rate,
            eps_hat, delta, certified}. k/success_rate/eps_hat are None when no
            sample was evaluated (empty set) or N is below the feasibility floor.
    """
    V_full = np.asarray(V_full, dtype=np.float32)

    obs2d = np.asarray(obstacle_sdf)
    if obs2d.ndim == 3:
        obs2d = obs2d[:, :, 0]

    cache = _ReachValueCache(
        obs2d, V_full, x_axis, y_axis, theta_array, target2d, grid)

    eps, beta = cfg['eps'], cfg['beta']
    N = _scenario_required_N(eps, beta)               # sample budget
    kmax = _scenario_max_outliers(N, eps, beta)       # tolerated failures

    def _report(k, delta, certified):
        """Certification stats at a given level (k=None when no sample eval'd)."""
        eps_hat = _scenario_epsilon_from_Nk(N, k, beta) if k is not None else None
        succ    = (1.0 - k / N) if (k is not None and N > 0) else None
        return dict(eps=eps, beta=beta, N=N, kmax=kmax, k=k,
                    success_rate=succ, eps_hat=eps_hat,
                    delta=float(delta), certified=bool(certified))

    if kmax < 0:
        # N below feasibility floor: even zero failures cannot certify eps.
        # Fall back to the conservative floor (keeps this a total function).
        floor = float(cfg['delta_floor'])
        if verbose:
            print(f"    [scenario] N={N} below feasibility floor for "
                  f"eps={eps}, beta={beta}: k_max<0 -> delta={floor}")
        return floor, _report(None, floor, certified=False)

    rng = np.random.default_rng(cfg['seed'])
    dt = float(cfg['dt'])
    deltas = _scenario_delta_grid(cfg['delta_init'], cfg['delta_floor'], cfg['delta_step'])
    if verbose:
        print(f"    [scenario] eps={eps}, N={N}, beta={beta} -> k_max={kmax}; "
              f"sweeping {len(deltas)} delta levels")

    last_k = None                                      # last evaluated level (for floor fallback)
    for delta in deltas:                               # descending: set shrinks
        try:
            X = sample_states(cache, N, delta, rng,
                              cfg['grid_min'], cfg['grid_max'],
                              max_tries=cfg['max_tries'])
        except RuntimeError:
            continue                                   # empty sublevel set here -> smaller delta
        k = int((rollout_cost(cache, X, car, dt, delta=delta) >= 0.0).sum())
        last_k = k
        passed = (k <= kmax)
        if verbose:
            eps_hat = _scenario_epsilon_from_Nk(N, k, beta)
            print(f"    [scenario] delta={delta:+.4g}: k={k}/{N} "
                  f"(<= {kmax}? {passed})  eps_hat={eps_hat:.3e}")
        if passed:
            return float(delta), _report(k, delta, certified=True)   # largest passing delta

    floor = float(cfg['delta_floor'])                  # nothing passed down to the floor
    return floor, _report(last_k, floor, certified=False)


def _scenario_pool_initializer():
    """ProcessPoolExecutor initializer: pin BLAS/OMP to a single thread in each
    worker so M worker processes don't each spawn cpu_count() BLAS threads
    (oversubscription). Belt-and-suspenders with the parent-side env set in
    HJR_FNO._get_scenario_pool (the parent set, inherited under spawn, is what
    OpenBLAS actually honors at import time)."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
