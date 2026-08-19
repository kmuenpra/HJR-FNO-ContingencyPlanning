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
    """Fifth order WENO approximation - simplified implementation (falls back to ENO2).

    This has always delegated wholesale to ENO2. It used to ALSO build a 3-cell
    ghost-padded copy of `data` and np.diff it, then discard both unread -- one
    wasted array allocation and one full pass per spatial dim, per call.
    Deleting them is bit-identical by construction (nothing read them) and was
    worth 2.24x on gradient construction: 1.060 s -> 0.473 s over 20 reps of the
    17-slice tube. They were ~55% of this stage.

    If real WENO5 is ever implemented here, it goes where those lines were.
    """
    return upwindFirstENO2(grid, data, dim)


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
    """Central-difference gradients via the upwind scheme, along the grid's dims.

    Extra trailing axes beyond ``grid.dims`` are permitted and ride along
    untouched (``add_ghost_cells`` / ``strip_dim`` / ``np.diff`` all slice an
    explicit ``dim`` and infer rank from ``data.ndim``) -- but do NOT use that to
    batch the time axis: callers are faster slice-by-slice, since one slice fits
    in L2 and the whole tube does not. See _ReachValueCache.__init__.
    """
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
        # kept as attributes so sublevel_bbox can map grid indices back to
        # coordinates without being handed the axes separately
        self.x_axis = np.asarray(x_axis)
        self.y_axis = np.asarray(y_axis)
        self.theta_axis = np.asarray(theta_axis)
        pts3d = (np.asarray(x_axis), np.asarray(y_axis), np.asarray(theta_axis))
        kw = dict(bounds_error=False, fill_value=None)

        self._V_slice_interps = [
            RegularGridInterpolator(pts3d, self.V_grid[:, :, :, i], **kw) for i in range(T)
        ]
        # 3-D spatial gradient per time slice (numpy WENO -> [d/dx, d/dy, d/dtheta]),
        # stacked into ONE 4-D vector-valued interpolator over (x, y, theta, k).
        #
        # WHY ONE INTERPOLATOR AND NOT 3*T OF THEM. The old layout held a
        # (gx, gy, gth) triple per time slice, so grad_at_indices had to loop over
        # the distinct time indices present in the batch and issue 3 calls each --
        # up to 3*17 = 51 scipy calls per integration step, each on a small subset
        # of the 435 trajectories. Profiled at 71% of rollout_cost (1.646 s of
        # 2.313 s, 17,940 calls): the cost was in the CALL COUNT, not the data.
        # Stacking collapses that to a single call per step.
        #
        # The k axis is arange(T) and is only ever queried at integer k, so the
        # linear interpolation along it lands exactly on a node -- identical
        # values to indexing the old per-slice interpolators, not an approximation.
        # Memory is unchanged: the same 3*T fields, in one array instead of 3*T.
        self.T = T
        # Per SLICE, deliberately. computeGradients accepts the whole 4-D tube in
        # one call, but that is measurably SLOWER here (0.473 s -> 0.750 s over 20
        # reps): a single slice is 50*50*25*8 B = 500 kB and stays in L2, while the
        # 17-slice array does not. The win in this stage came from deleting
        # upwindFirstWENO5's dead stores (2.24x), not from restructuring the loop.
        grads = np.empty(self.V_grid.shape[:3] + (T, 3), dtype=np.float64)
        for i in range(T):
            gx, gy, gth = computeGradients(
                odp_grid, np.ascontiguousarray(self.V_grid[:, :, :, i], dtype=np.float64))
            grads[:, :, :, i, 0] = gx
            grads[:, :, :, i, 1] = gy
            grads[:, :, :, i, 2] = gth
        self._grad_interp = RegularGridInterpolator(
            pts3d + (np.arange(T, dtype=float),), grads, **kw)

        pts2d = (np.asarray(x_axis), np.asarray(y_axis))
        self._ell_int = RegularGridInterpolator(pts2d, np.asarray(target_sdf_2d), **kw)   # <0 inside target
        self._G_int   = RegularGridInterpolator(pts2d, -np.asarray(obstacle_sdf_2d), **kw)  # >0 inside obstacle

    def value_at_full_BRS(self, s):              # s: (B,3) -> (B,)
        return self._V_slice_interps[0](s)        # slice 0 = fully-grown BRT

    def grad_at_indices(self, s, k_arr):          # s: (B,3), k_arr: (B,) -> (B,3)
        """One vector-valued query at (x, y, theta, k) for the whole batch."""
        q = np.empty((s.shape[0], 4), dtype=np.float64)
        q[:, :3] = s
        q[:, 3] = k_arr
        return self._grad_interp(q)

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
    T_slices = cache.T

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


def sublevel_bbox(cache, delta, grid_min, grid_max):
    """Axis-aligned (x, y) box guaranteed to CONTAIN the continuous sublevel set
    {s : V(s, fully-grown) < delta}. Returns (lo2, hi2) or None if empty.

    WHY THIS IS SOUND
    -----------------
    Uniform sampling on a box B, rejected down to S(delta), is uniform on S(delta)
    -- but ONLY if B contains S(delta). Two details make that hold:

    * ONE CELL OF PADDING, and do not reduce it. ``value_at_full_BRS`` is a LINEAR
      interpolant, so a continuous point can sit below delta while every enclosing
      grid node is at or above it. That can only happen within one cell of a node
      that is itself sub-delta, so one cell is sufficient -- and necessary.

    * THETA IS NOT BOUNDED. The sublevel set can wrap across +-pi; an axis-aligned
      theta interval would then cut part of S(delta) out of B, breaking containment
      and silently invalidating delta_hat. Theta stays full-range. The win is in
      x, y anyway: a 20 x 20 m domain against sets that are usually far smaller.

    The scenario guarantee is therefore untouched -- this changes only the
    PROPOSAL distribution's support, not the distribution being sampled.
    """
    if not math.isfinite(delta):
        return (np.asarray(grid_min[:2], float), np.asarray(grid_max[:2], float))
    mask_xy = (cache.V_grid[:, :, :, 0] < delta).any(axis=2)   # marginalise theta
    if not mask_xy.any():
        return None
    xi = np.flatnonzero(mask_xy.any(axis=1))
    yi = np.flatnonzero(mask_xy.any(axis=0))
    x_ax, y_ax = cache.x_axis, cache.y_axis
    nx, ny = len(x_ax), len(y_ax)
    i0, i1 = max(int(xi[0]) - 1, 0), min(int(xi[-1]) + 1, nx - 1)   # one-cell pad
    j0, j1 = max(int(yi[0]) - 1, 0), min(int(yi[-1]) + 1, ny - 1)
    return (np.array([x_ax[i0], y_ax[j0]], float),
            np.array([x_ax[i1], y_ax[j1]], float))


def sample_states(cache, n_target, delta, rng, grid_min, grid_max,
                  batch=8192, max_tries=400, stats=None):
    """Uniform rejection sampling of states (x, y, theta) keeping
    V(s, fully-grown) < delta. Returns (n_target, 3).

    Proposes over the FULL domain, and falls back to the sublevel set's bounding
    box only if a draw comes up short. That fallback is a measured decision, not
    caution: on this repo's tubes the first 8192-candidate draw always yields the
    435 samples needed (acceptance ~0.4), and sampling is only ~1.3% of a
    certification -- building the box up front was pure overhead. It stays as
    insurance for a tight set, the one case where a full-domain proposal could
    burn max_tries*batch = 3.3M evaluations.

    Mixing proposals across draws is sound: uniform-on-domain restricted to
    S(delta) and uniform-on-box restricted to S(delta) are BOTH uniform on
    S(delta), so the concatenation is still an i.i.d. uniform sample of S(delta).

    ``stats``: optional dict, filled with tries / accept_rate so the cost of this
    stage is visible rather than inferred.
    """
    x_lo = np.array([grid_min[0], grid_min[1], -math.pi])
    x_hi = np.array([grid_max[0], grid_max[1],  math.pi])
    boxed = False

    kept, n_kept, tries, n_drawn = [], 0, 0, 0
    for _ in range(max_tries):
        s_cand = rng.uniform(x_lo, x_hi, size=(batch, 3)).astype(np.float32)
        V = cache.value_at_full_BRS(s_cand)
        mask = (V < delta) if math.isfinite(delta) else np.ones_like(V, dtype=bool)
        tries += 1
        n_drawn += batch
        if mask.any():
            kept.append(s_cand[mask]); n_kept += int(mask.sum())
            if n_kept >= n_target:
                break
        if not boxed:
            # first draw fell short -> tighten the proposal to the sublevel set's
            # bounding box for every subsequent draw
            box = sublevel_bbox(cache, delta, grid_min, grid_max)
            if box is None:
                break                      # empty set; fall through to the raise
            lo2, hi2 = box
            x_lo = np.array([lo2[0], lo2[1], -math.pi])
            x_hi = np.array([hi2[0], hi2[1],  math.pi])
            boxed = True
        # Adapt the draw size to the observed acceptance rate, aiming to finish
        # the remainder in ~2 draws. Clamped so a near-empty set cannot explode
        # the batch and a near-full one cannot degenerate to trickling.
        rate = n_kept / max(n_drawn, 1)
        if rate > 0:
            batch = int(np.clip(math.ceil((n_target - n_kept) / rate / 2),
                                512, 8192))
    if stats is not None:
        stats["sample_tries"] = tries
        stats["sample_accept_rate"] = (n_kept / n_drawn) if n_drawn else 0.0
        stats["sample_boxed"] = boxed
    if n_kept == 0:
        raise RuntimeError(
            f"Sampled 0/{n_target} states with V<{delta} after {max_tries} tries "
            f"(sublevel set may be empty on this grid).")
    return np.concatenate(kept, axis=0)[:n_target]


def count_failures(cache, X, car, dt, delta, kmax, chunk=128, stats=None):
    """Number of rollout failures ``#{J >= 0}``, stopping once it exceeds kmax.

    A level whose failure count passes kmax is REJECTED, and rejected levels are
    discarded -- so the exact count never matters for them and the remaining
    rollouts are pure waste. A level that PASSES still evaluates all N, which the
    certificate requires. delta_hat is therefore bit-identical to counting the
    full sample every time; only the discarded levels get cheaper.

    Returns (k, partial). ``partial=True`` means k is a LOWER BOUND (>= kmax+1),
    not the true count -- callers must not report it as exact.
    """
    k, n = 0, X.shape[0]
    n_chunks = 0
    for a in range(0, n, chunk):
        k += int((rollout_cost(cache, X[a:a + chunk], car, dt, delta=delta) >= 0.0).sum())
        n_chunks += 1
        if k > kmax:
            if stats is not None:
                stats["rollout_chunks"] = stats.get("rollout_chunks", 0) + n_chunks
            return k, True
    if stats is not None:
        stats["rollout_chunks"] = stats.get("rollout_chunks", 0) + n_chunks
    return k, False


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
                          delta_step, seed, dt, grid_min, grid_max,
                          delta_warm)  -- delta_warm optional, see the sweep below
    @return (delta_hat, report) where delta_hat is a float in
            [delta_floor, delta_init] and report is a dict with the certification
            stats at the returned level: {eps, beta, N, kmax, k, k_partial,
            success_rate, eps_hat, delta, certified} plus the cost counters
            {levels_evaluated, sample_tries, sample_accept_rate, rollout_chunks}.
            k/success_rate/eps_hat are None when no sample was evaluated (empty
            set) or N is below the feasibility floor.
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

    stats = {"levels_evaluated": 0, "rollout_chunks": 0}

    def _report(k, delta, certified, partial=False):
        """Certification stats at a given level (k=None when no sample eval'd).

        ``k_partial`` marks a k that early-exit truncated: it is then a LOWER
        BOUND on the failure count, not the count. Only ever set on a level that
        FAILED, so no certificate is reported from a partial count.
        """
        eps_hat = (_scenario_epsilon_from_Nk(N, k, beta)
                   if (k is not None and not partial) else None)
        succ    = (1.0 - k / N) if (k is not None and N > 0 and not partial) else None
        return dict(eps=eps, beta=beta, N=N, kmax=kmax, k=k, k_partial=bool(partial),
                    success_rate=succ, eps_hat=eps_hat,
                    delta=float(delta), certified=bool(certified), **stats)

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
    step = abs(float(cfg['delta_step']))

    # ---- warm start ------------------------------------------------------
    # delta_hat moves slowly between reveals, but the sweep restarted from
    # delta_init every time -- ~11 levels to reach a typical delta_hat = -0.5.
    # Start one level ABOVE the region's previous answer instead, then walk UP
    # while it keeps passing (see below): starting AT the previous answer and
    # only descending would make delta_hat monotonically non-increasing, so a
    # region whose certifiable margin improved could never recover the set
    # volume -- and that costs path length for the rest of the episode.
    #
    # Callers pass delta_warm=None on a region's FIRST certification:
    # safe_margin is initialised to 0, which is itself a valid grid level, so
    # the value alone cannot distinguish "never certified" from "certified at 0".
    warm = cfg.get('delta_warm')
    j_start = 0
    if warm is not None and math.isfinite(warm):
        j_prev = int(round((cfg['delta_init'] - float(warm)) / step))
        j_start = int(np.clip(j_prev - 1, 0, len(deltas) - 1))
    if verbose:
        print(f"    [scenario] eps={eps}, N={N}, beta={beta} -> k_max={kmax}; "
              f"{len(deltas)} delta levels, starting at index {j_start} "
              f"({'warm from ' + format(warm, '+.4g') if warm is not None else 'cold'})")

    evaluated = {}          # level index -> (k, passed, partial), so a level is
                            # never re-run when the walk changes direction

    def _eval(j):
        """Test level j. Returns (k, passed, partial) or None if its set is empty."""
        if j in evaluated:
            return evaluated[j]
        delta = deltas[j]
        try:
            X = sample_states(cache, N, delta, rng, cfg['grid_min'], cfg['grid_max'],
                              max_tries=cfg['max_tries'], stats=stats)
        except RuntimeError:
            evaluated[j] = None                # empty sublevel set -> smaller delta
            return None
        k, partial = count_failures(cache, X, car, dt, delta, kmax, stats=stats)
        stats["levels_evaluated"] += 1
        out = (k, k <= kmax, partial)
        evaluated[j] = out
        if verbose:
            eh = ("n/a" if partial
                  else f"{_scenario_epsilon_from_Nk(N, k, beta):.3e}")
            print(f"    [scenario] delta={delta:+.4g}: k={'>' if partial else ''}{k}/{N} "
                  f"(<= {kmax}? {out[1]})  eps_hat={eh}")
        return out

    last_k, last_partial = None, False
    r = _eval(j_start)
    if r is not None:
        last_k, last_partial = r[0], r[2]

    if r is not None and r[1]:
        # PASSED at the warm level -> walk UP (larger delta, bigger set) while it
        # keeps passing, and return the last level that did. Without this the
        # margin could only ever shrink, so a region whose set GREW would keep a
        # needlessly conservative delta forever.
        best_j, best_k = j_start, r[0]
        j = j_start - 1
        while j >= 0:
            rj = _eval(j)
            if rj is None or not rj[1]:
                break
            best_j, best_k = j, rj[0]
            j -= 1
        return float(deltas[best_j]), _report(best_k, deltas[best_j], certified=True)

    # FAILED (or empty) at the warm level -> descend exactly as the cold sweep does
    for j in range(j_start + 1, len(deltas)):
        rj = _eval(j)
        if rj is None:
            continue
        last_k, last_partial = rj[0], rj[2]
        if rj[1]:
            return float(deltas[j]), _report(rj[0], deltas[j], certified=True)

    floor = float(cfg['delta_floor'])                  # nothing passed down to the floor
    return floor, _report(last_k, floor, certified=False, partial=last_partial)


def _scenario_pool_initializer():
    """ProcessPoolExecutor initializer: pin BLAS/OMP to a single thread in each
    worker so M worker processes don't each spawn cpu_count() BLAS threads
    (oversubscription). Belt-and-suspenders with the parent-side env set in
    HJR_FNO._get_scenario_pool (the parent set, inherited under spawn, is what
    OpenBLAS actually honors at import time)."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
