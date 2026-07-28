"""
Scenario optimization for the REACH-AVOID case — DubinsCar2 + 5-channel FNO3d.

Setting
-------
A trained FNO3d approximates the HJ reach-avoid value function for a 3-state
Dubins car (see HJR_FNO/docs/reach_avoid_problem_summary.md):

    V_tilde(x, y, theta, t ; c)        state (x, y, theta),  c = obstacle SDF

The FNO treats theta as a CONSTANT 5th input channel, so one forward pass gives
V over (x, y, t) at a fixed heading; sweeping all headings gives the full
V(x, y, theta, t).  Input channel order is [sdf, x, y, t, theta] with
sdf = -obstacle (the model was trained on the negated obstacle SDF).

Goal: for each obstacle configuration c, find a corrected scalar threshold
delta_hat such that the recovered set

    S_hat(c) = { s : V_tilde(s, fully-grown, c) < delta_hat }

is, with high probability, contained in the true reach-avoid set of the problem
indexed by c.

Two modes (selected via `mode=`):
  * "per_c":  one scenario optimization PER OBSTACLE CONFIGURATION c in a list
              (the test obstacle SDFs from the .mat).  Bonferroni: each run uses
              confidence beta/K so the joint claim across K obstacles holds at
              confidence (1 - beta).   <-- the main mode
  * "joint":  one delta_hat over the joint distribution of (state, obstacle);
              obstacles are resampled via the data-gen `random_obstacle_set`.

Conventions
-----------
* delta_0 = +inf
* Recovered set / sample region:  V_tilde(s, fully-grown, c) < delta
* Failure (didn't reach OR hit obstacle) iff cost J >= 0
* delta update:  delta = min { V_tilde(s, fully-grown, c) : J(s, c) >= 0 }

Time convention (odp): the .mat tau increases 0 -> T_max (= 8 s).
  V_raw[..., 0]  = fully-grown BRT (full lookback)   <-- "S_hat" thresholds this
  V_raw[..., -1] = terminal/target slice  max(ell, -obstacle)
During a forward rollout, the slice index marches from the per-sample start
toward T-1 (target), tracking time-to-go.

SIGN CONVENTIONS — read before touching any sign
=================================================
The dataset stores odp implicit surfaces (LINEAR SDFs, quadratic=False):

    obstacle_sdf(s) < 0  inside the obstacle,  > 0 in free space   (== mat["constraints"])
    target_sdf(s)   < 0  inside the target,    > 0 outside         (== mat["target_set"])

Reach-avoid value uses the AVOIDANCE function  G = -obstacle_sdf  (> 0 inside
the obstacle), and the realized cost

    J = min_k max( ell(s_k),  max_{j<=k} G(s_j) ),     J >= 0  <=>  failure.

FNO input channel 0 is  -obstacle_sdf  (the model was trained that way), so
`FNOValueModel.predict_full` negates internally — pass the RAW obstacle SDF
(mat["constraints"], negative-inside).  NEVER pre-negate before predict_full.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence, List, Tuple

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
# matplotlib is imported lazily inside the viz helpers (Agg backend) to avoid
# CUDA-context segfaults from interactive backends.

# ── Make repo root + optimized_dp importable ──────────────────────────────────
_THIS         = Path(__file__).resolve()
_HJR_FNO_ROOT = _THIS.parent.parent          # .../HJR_FNO
_REPO_ROOT    = _HJR_FNO_ROOT.parent          # repo root (contains optimized_dp/)
for _p in (_REPO_ROOT, _REPO_ROOT / "optimized_dp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from HJR_FNO.HJR_FNO3d import SpectralConv3d, FNO3d   # noqa: E402  (needed to unpickle the model)
from odp.Grid import Grid                              # noqa: E402
from odp.dynamics.DubinsCar2 import DubinsCar2         # noqa: E402
from odp.solver import computeSpatDerivArray           # noqa: E402

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_CKPT_PATH = str(
    _HJR_FNO_ROOT / "training/model/01_FNO3d_dubins_5ch_tuned.pt"
)
DEFAULT_MAT_PATH = str(
    _HJR_FNO_ROOT / "data_gen/HJB_training_mat/DubinsCar2_50x50x36_reach_avoid.mat"
)


# =============================================================================
# 1. Problem definition
# =============================================================================

@dataclass
class DubinsReachProblem:
    """Concrete reach-avoid problem for the DubinsCar2 + 5-channel FNO3d setup."""

    # Grid axes (must match the .mat the FNO was trained on).
    x_axis:     np.ndarray        # (Nx,)
    y_axis:     np.ndarray        # (Ny,)
    theta_axis: np.ndarray        # (Nth,)  heading (periodic)
    tau:        np.ndarray        # (T,)    monotone 0 -> T_max
    # Fixed target SDF on the (Nx, Ny) grid (theta-independent). <0 inside target.
    target_sdf: np.ndarray        # (Nx, Ny)
    # State-space box for sampling: lower / upper [x, y, theta].
    x_lo: np.ndarray              # (3,)
    x_hi: np.ndarray              # (3,)
    # DubinsCar2 dynamics object (bang-bang opt-ctrl + worst-dstb used by rollout).
    car: DubinsCar2 = field(default_factory=lambda: DubinsCar2(
        uMin=[0.0, -1.0], uMax=[1.0, 1.0], dMax=[0.1, 0.1, 0.1],
        uMode="min", dMode="max",
    ))
    # odp Grid (3D, periodic theta) — required by computeSpatDerivArray.
    odp_grid: Optional[Grid] = None
    # Euler step (matches data tau spacing).
    dt: float = 0.5
    # Optional obstacle sampler for `joint` mode -> (Nx, Ny) SDF (negative inside).
    obstacle_sampler: Optional[Callable[[np.random.Generator], np.ndarray]] = None
    # Optional discrete obstacle list for `per_c` mode (each (Nx,Ny) or (Nx,Ny,Nth)).
    c_list: Optional[List[np.ndarray]] = None

    @property
    def T(self) -> float:
        return float(self.tau[-1])


# =============================================================================
# 2. Value-model wrapper around FNO3d (5-channel, theta swept)
# =============================================================================

class FNOValueModel:
    """Wraps an FNO3d. Input channels are [sdf, x, y, t, theta] with
    sdf = -obstacle (training convention). `predict_full` sweeps every heading
    and returns V over (Nx, Ny, Nth, T)."""

    def __init__(self, fno: torch.nn.Module, device: str = "cuda"):
        self.fno = fno.to(device).eval()
        self.device = device

    @torch.no_grad()
    def predict_full(
        self,
        obstacle_sdf: np.ndarray,   # (Nx, Ny)  RAW obstacle SDF (negative inside obstacle)
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        theta_axis: np.ndarray,
        tau: np.ndarray,
        chunk: int = 12,
    ) -> np.ndarray:                # (Nx, Ny, Nth, T)
        """Build the 5-channel input [sdf, x, y, t, theta] (sdf = -obstacle),
        sweep all headings in batches, and stack -> V(x, y, theta, t)."""
        Nx, Ny = obstacle_sdf.shape
        T   = len(tau)
        Nth = len(theta_axis)

        Xg, Yg  = np.meshgrid(x_axis, y_axis, indexing="ij")            # (Nx,Ny)
        sdf_neg = torch.tensor(-obstacle_sdf.astype(np.float32))         # ch0 = -obstacle
        sdf_3d  = sdf_neg.unsqueeze(-1).expand(Nx, Ny, T)
        x_3d    = torch.tensor(Xg, dtype=torch.float32).unsqueeze(-1).expand(Nx, Ny, T)
        y_3d    = torch.tensor(Yg, dtype=torch.float32).unsqueeze(-1).expand(Nx, Ny, T)
        t_3d    = torch.tensor(np.asarray(tau, np.float32)).view(1, 1, T).expand(Nx, Ny, T)
        base    = torch.stack([sdf_3d, x_3d, y_3d, t_3d,
                               torch.zeros(Nx, Ny, T)], dim=-1)          # (Nx,Ny,T,5)

        V_full = np.empty((Nx, Ny, Nth, T), dtype=np.float32)
        for s in range(0, Nth, chunk):
            ths = np.asarray(theta_axis[s:s + chunk], np.float32)
            b   = base.unsqueeze(0).repeat(len(ths), 1, 1, 1, 1)        # (B,Nx,Ny,T,5)
            for bi, th in enumerate(ths):
                b[bi, :, :, :, 4] = float(th)                          # constant theta channel
            out = self.fno(b.to(self.device))[..., 0].cpu().numpy()    # (B,Nx,Ny,T)
            V_full[:, :, s:s + len(ths), :] = np.moveaxis(out, 0, 2)
        return V_full


class CachedConstraint:
    """Per-obstacle cache: V over (x, y, theta, t) + interpolators for V and the
    3-D spatial gradient (dVdx, dVdy, dVdtheta), plus 2-D ell / G interpolators."""

    def __init__(self, obstacle_sdf: np.ndarray, prob: DubinsReachProblem,
                 model: FNOValueModel):
        if obstacle_sdf.ndim == 3:           # (Nx,Ny,Nth) -> theta-independent slice
            obstacle_sdf = obstacle_sdf[:, :, 0]
        self.obstacle_sdf = obstacle_sdf.astype(np.float32)              # <0 inside obstacle
        self.prob = prob

        if prob.odp_grid is None:
            raise ValueError("DubinsReachProblem.odp_grid is None; build via "
                             "build_problem_from_mat().")

        # V over (Nx, Ny, Nth, T). [...,0]=fully-grown BRT, [...,-1]=target slice.
        V = model.predict_full(self.obstacle_sdf, prob.x_axis, prob.y_axis,
                               prob.theta_axis, prob.tau).astype(np.float32)
        T = V.shape[-1]

        # 3-D spatial derivatives per time slice (deriv_dim 1->x, 2->y, 3->theta).
        dVdx = np.empty_like(V); dVdy = np.empty_like(V); dVdth = np.empty_like(V)
        for k in range(T):
            Vk = np.array(V[:, :, :, k], dtype=np.float32, copy=True)   # (Nx,Ny,Nth)
            dVdx[:, :, :, k]  = computeSpatDerivArray(prob.odp_grid, Vk, deriv_dim=1, accuracy="low")
            dVdy[:, :, :, k]  = computeSpatDerivArray(prob.odp_grid, Vk, deriv_dim=2, accuracy="low")
            dVdth[:, :, :, k] = computeSpatDerivArray(prob.odp_grid, Vk, deriv_dim=3, accuracy="low")

        self.V_grid = V.copy()
        self.V_grid.flags.writeable = False

        pts3d = (prob.x_axis, prob.y_axis, prob.theta_axis)
        kw    = dict(bounds_error=False, fill_value=None)
        self._grad_interps: List[Tuple] = [
            (RegularGridInterpolator(pts3d, dVdx[:, :, :, i],  **kw),
             RegularGridInterpolator(pts3d, dVdy[:, :, :, i],  **kw),
             RegularGridInterpolator(pts3d, dVdth[:, :, :, i], **kw))
            for i in range(T)
        ]
        self._V_slice_interps = [
            RegularGridInterpolator(pts3d, V[:, :, :, i], **kw) for i in range(T)
        ]

        # 2-D (x, y) interpolators — target and avoidance are theta-independent.
        pts2d = (prob.x_axis, prob.y_axis)
        self._ell_int = RegularGridInterpolator(pts2d, prob.target_sdf, **kw)
        self._G_int   = RegularGridInterpolator(pts2d, -self.obstacle_sdf, **kw)  # G = -obstacle (>0 inside obstacle)

    # ── value / gradient queries (state s = (x, y, theta)) ────────────────────
    def value_at_full_BRS(self, s: np.ndarray) -> np.ndarray:
        """V at slice 0 (fully-grown BRT). s: (B,3). Returns (B,)."""
        return self._V_slice_interps[0](s)

    def grad_at_step(self, s: np.ndarray, k: int) -> np.ndarray:
        gx, gy, gth = self._grad_interps[k]
        return np.stack([gx(s), gy(s), gth(s)], axis=-1)               # (B,3)

    def grad_at_indices(self, s: np.ndarray, k_arr: np.ndarray) -> np.ndarray:
        B = s.shape[0]
        out = np.empty((B, 3), dtype=np.float64)
        for k in np.unique(k_arr):
            m = (k_arr == k)
            gx, gy, gth = self._grad_interps[int(k)]
            out[m, 0] = gx(s[m]); out[m, 1] = gy(s[m]); out[m, 2] = gth(s[m])
        return out

    def find_max_safe_time_index(self, s: np.ndarray, delta: float = 0.0) -> np.ndarray:
        """Largest slice index k with V_grid[...,k](s) <= delta (tightest sublevel
        set still containing s). Index 0 = fully-grown (loosest). Fallback 0."""
        T = len(self._V_slice_interps)
        B = s.shape[0]
        t_i   = np.zeros(B, dtype=int)
        found = np.zeros(B, dtype=bool)
        for k in range(T - 1, -1, -1):
            new = (~found) & (self._V_slice_interps[k](s) <= delta)
            if new.any():
                t_i[new] = k; found[new] = True
                if found.all():
                    break
        return t_i

    # ── 2-D scalar fields (evaluated at the (x,y) part of the state) ──────────
    def ell(self, xy: np.ndarray) -> np.ndarray:
        return self._ell_int(xy)

    def G(self, xy: np.ndarray) -> np.ndarray:
        """Avoidance function: > 0 inside the obstacle."""
        return self._G_int(xy)


# =============================================================================
# 3. DubinsCar2 induced policy + rollout cost
# =============================================================================

def _wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def rollout_cost(s0: np.ndarray, cache: CachedConstraint, delta: float = 0.0) -> np.ndarray:
    """Constrained reach-avoid cost under the FNO-induced controller and worst-case
    disturbance (DubinsCar2 bang-bang policies):

        J = min_k max( ell(s_k),  max_{j<=k} G(s_j) ),     failure iff J >= 0.

    s0: (B, 3) states (x, y, theta).  Returns (B,).
    """
    prob = cache.prob
    car  = prob.car
    s    = s0.astype(np.float64).copy()                                 # (B,3)
    T_slices = len(cache._grad_interps)

    t_start = cache.find_max_safe_time_index(s.astype(np.float32), delta=delta)  # (B,)
    n_steps_per_sample = T_slices - 1 - t_start
    max_steps = int(n_steps_per_sample.max()) if t_start.size > 0 else 0

    running_G = cache.G(s[:, :2]).copy()
    cost      = np.maximum(cache.ell(s[:, :2]), running_G)

    for k in range(max_steps):
        active = k < n_steps_per_sample
        if not active.any():
            break
        ai = np.flatnonzero(active)
        slice_idx = t_start[ai] + k
        grad = cache.grad_at_indices(s[ai].astype(np.float32), slice_idx)   # (Bact,3)

        for li, gi in enumerate(ai):
            u = car.optCtrl_inPython(s[gi], grad[li])      # [speed, omega]
            d = car.optDstb_inPython(s[gi], grad[li])      # [d1, d2, d3]
            dx, dy, dth = car.dynamics_inPython(s[gi], u, d)
            s[gi, 0] += prob.dt * dx
            s[gi, 1] += prob.dt * dy
            s[gi, 2]  = _wrap_to_pi(s[gi, 2] + prob.dt * dth)

        running_G = np.maximum(running_G, cache.G(s[:, :2]))
        cost      = np.minimum(cost, np.maximum(cache.ell(s[:, :2]), running_G))
    return cost


# =============================================================================
# 4. Sampling under V_tilde(s, fully-grown, c) < delta
# =============================================================================

def sample_x_under_delta(
    n_target: int,
    delta: float,
    cache: CachedConstraint,
    rng: np.random.Generator,
    batch: int = 8192,
    max_tries: int = 400,
) -> np.ndarray:
    """Uniform rejection sampling of states (x, y, theta) for a FIXED obstacle,
    keeping V_tilde(s, fully-grown, c) < delta.  Returns (n_target, 3)."""
    prob = cache.prob
    kept, n_kept = [], 0
    for _ in range(max_tries):
        s_cand = rng.uniform(prob.x_lo, prob.x_hi, size=(batch, 3)).astype(np.float32)
        V = cache.value_at_full_BRS(s_cand)
        mask = (V < delta) if math.isfinite(delta) else np.ones_like(V, dtype=bool)
        if mask.any():
            kept.append(s_cand[mask]); n_kept += int(mask.sum())
            if n_kept >= n_target:
                break
    if n_kept < n_target:
        if n_kept == 0:
            raise RuntimeError(
                f"Sampled 0/{n_target} states with V<{delta} after {max_tries} tries. "
                f"The sublevel set may be empty on this grid.")
        print(f"  [warn] partial batch: {n_kept}/{n_target} states with V<{delta:.4g} "
              f"(sublevel set too small; guarantee weakened for this iter)")
    return np.concatenate(kept, axis=0)[:n_target]


# =============================================================================
# 4b. Animated visualization of one rollout (x-y projection at the live heading)
# =============================================================================

def _precompute_rollout(cache: "CachedConstraint", s_init: np.ndarray,
                        delta: float = 0.0) -> Tuple[np.ndarray, int]:
    """Run one rollout. Returns (traj (n+1, 3), t_start)."""
    prob = cache.prob; car = prob.car
    T_slices = len(cache._grad_interps)
    s = np.asarray(s_init, dtype=np.float64).reshape(1, 3).copy()
    t_start = int(cache.find_max_safe_time_index(s.astype(np.float32), delta=delta)[0])
    n_local = T_slices - 1 - t_start

    traj = [s.flatten().copy()]
    for k in range(n_local):
        grad = cache.grad_at_step(s.astype(np.float32), t_start + k)
        u = car.optCtrl_inPython(s[0], grad[0])
        d = car.optDstb_inPython(s[0], grad[0])
        dx, dy, dth = car.dynamics_inPython(s[0], u, d)
        s[0, 0] += prob.dt * dx
        s[0, 1] += prob.dt * dy
        s[0, 2]  = _wrap_to_pi(s[0, 2] + prob.dt * dth)
        traj.append(s.flatten().copy())
    return np.asarray(traj, dtype=np.float64), t_start


def visualize_constraint_result(
    cache: "CachedConstraint",
    s_init: np.ndarray,            # (3,) start state (x, y, theta)
    delta_hat: float,
    title_prefix: str = "",
    frame_pause: float = 0.15,
) -> None:
    """Animate ONE rollout for this obstacle and BLOCK until the window is
    closed (then the per_c loop continues to the next obstacle).

    The animation (simple ax.clear() redraw loop) runs in a SEPARATE process
    that imports only numpy + matplotlib, so it never shares this process's
    CUDA context — avoiding the plt SIGSEGV. Each frame (x-y plane) shows:
      * V at the rollout's current heading / time slice (background + magenta V=0),
      * lime  = recovered set  S_hat = {V < delta_hat}  (fully-grown, start heading),
      * red -- = obstacle,  green -- = target,
      * black  = trajectory so far.
    `frame_pause` sets the per-frame delay (seconds).
    """
    import os, sys as _sys, tempfile, subprocess

    traj, t_start = _precompute_rollout(cache, s_init, delta=delta_hat)
    prob   = cache.prob
    Xg, Yg = np.meshgrid(prob.x_axis, prob.y_axis, indexing="ij")
    V_all  = cache.V_grid
    th_ax  = prob.theta_axis
    T      = V_all.shape[-1]
    F      = len(traj)

    # Per-frame V slice: at the trajectory's live heading and marching time slice.
    frames_V = np.empty((F, Xg.shape[0], Xg.shape[1]), dtype=np.float32)
    for k in range(F):
        si  = min(t_start + k, T - 1)
        thi = int(np.argmin(np.abs(th_ax - _wrap_to_pi(traj[k, 2]))))
        frames_V[k] = V_all[:, :, thi, si]
    th0 = int(np.argmin(np.abs(th_ax - _wrap_to_pi(s_init[2]))))
    V_bg = np.ascontiguousarray(V_all[:, :, th0, 0])      # fully-grown BRT at start heading

    # Hand everything to the CUDA-free worker process via a temp .npz.
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.close()
    np.savez(
        tmp.name, Xg=Xg, Yg=Yg, frames_V=frames_V, V_bg=V_bg,
        obstacle=cache.obstacle_sdf, target=prob.target_sdf, traj=traj,
        delta_hat=np.float64(delta_hat), dt=np.float64(prob.dt),
        x_lo=np.asarray(prob.x_lo), x_hi=np.asarray(prob.x_hi),
        title=np.array(title_prefix),
    )
    worker = str(_THIS.parent / "_viz_worker.py")
    print(f"  [viz] {title_prefix.strip()} — close the window to continue...")
    try:
        subprocess.run([_sys.executable, worker, tmp.name, str(frame_pause)])
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# =============================================================================
# 5. Required N from the scenario theorem
# =============================================================================

def required_N(eps: float, beta: float) -> int:
    return int(math.ceil((2.0 / eps) * (math.log(1.0 / beta) + 1.0)))


# =============================================================================
# 6a. PER-OBSTACLE scenario optimization with Bonferroni  ("per_c")
# =============================================================================

def scenario_optimize_per_c(
    model: FNOValueModel,
    prob: DubinsReachProblem,
    c_list: Sequence[np.ndarray],
    eps: float = 1e-2,
    beta: float = 1e-9,
    M: int = 20,
    seed: int = 0,
    verbose: bool = True,
    visualize: bool = False,
    viz_pause: float = 0.1,
    max_tries: int = 400,
    delta_floor: float = -1.40,
    delta_init: float = 0.0,
    step_frac: float = 0.5,
) -> dict:
    """Independent scenario optimization for EACH obstacle configuration in
    c_list. Bonferroni: each run uses beta_k = beta / K so the joint claim
    across K obstacles holds at confidence (1 - beta).

    `delta_init` is the starting threshold (default 0.0): the first iteration
    samples only states INSIDE the learned BRT, {V(s, fully-grown) < delta_init},
    rather than uniformly over the whole state box.

    `step_frac` damps the delta update: instead of jumping straight to the worst
    violator value, delta moves a fraction of the way there,
        delta <- delta + step_frac * (target - delta),
    where target = max(min V[violators], delta_floor).  step_frac=1.0 reproduces
    the exact scenario update (tightest, one-shot); step_frac<1 changes delta
    more gradually (more iterations, smoother). The loop still only declares
    convergence when a sampled batch has zero violators, so the final recovered
    set is valid regardless of step_frac.

    `delta_floor` clamps delta_hat from below: the recovered set is never shrunk
    past {V < delta_floor}. If the worst violator sits below the floor and delta
    has settled onto it, delta_hat is pinned at the floor and the run stops
    (converged=False, floored=True). After each obstacle, a fresh batch is rolled
    out at the final delta_hat to report how many trajectories still fail (J>=0)."""
    K = len(c_list)
    beta_k = beta / K
    N = required_N(eps, beta_k)
    if verbose:
        print(f"[per_c] K={K} obstacles, eps={eps}, beta={beta} "
              f"-> per-run beta={beta_k:.2e}, N={N}")

    rng = np.random.default_rng(seed)
    results: dict = {}

    for k, c_sdf in enumerate(c_list):
        cache = CachedConstraint(c_sdf, prob, model)
        delta = float(delta_init)        # start inside the learned BRT ({V < delta_init})
        history, converged, floored = [], False, False

        for i in range(M):
            try:
                X = sample_x_under_delta(N, delta, cache, rng, max_tries=max_tries)
            except RuntimeError as exc:
                if verbose:
                    print(f"  c#{k} iter {i}: {exc}")
                break

            J = rollout_cost(X, cache, delta=delta)
            violators = J >= 0.0
            n_viol = int(violators.sum())
            if verbose:
                print(f"  c#{k} iter {i}: delta={delta:.6g}, violators={n_viol}/{N}")

            if n_viol == 0:
                converged = True
                history.append({"iter": i, "delta": delta, "violators": 0})
                break

            # Scenario target: the most-negative learned value among violators,
            # clamped so we never aim below the floor.
            raw_delta = float(cache.value_at_full_BRS(X)[violators].min())
            target    = max(raw_delta, delta_floor)
            # Damped step toward the target (step_frac=1.0 -> exact scenario update).
            new_delta = delta + step_frac * (target - delta)

            # Floor-limited stop: worst violator is below the floor and delta has
            # essentially settled onto the floor -> can't certify a clean set.
            if raw_delta <= delta_floor and (delta - new_delta) < 1e-3:
                delta = delta_floor
                floored = True
                history.append({"iter": i, "delta": delta, "violators": n_viol, "floored": True})
                if verbose:
                    print(f"  c#{k} iter {i}: floor-limited (worst violator V={raw_delta:.4g} "
                          f"<= floor {delta_floor:.4g}); pinned and stopping.")
                break

            delta = new_delta
            history.append({"iter": i, "delta": delta, "violators": n_viol})

        # ── Report rollout failures at the final (possibly clamped) delta_hat ──
        n_fail = n_eval = 0
        fail_frac = 0.0
        try:
            Xf = sample_x_under_delta(N, delta, cache, rng, max_tries=max_tries)
            Jf = rollout_cost(Xf, cache, delta=delta)
            n_eval = len(Xf)
            n_fail = int((Jf >= 0.0).sum())
            fail_frac = n_fail / n_eval if n_eval else 0.0
        except RuntimeError:
            pass  # recovered set empty at this delta -> nothing to fail
        if verbose:
            print(f"  c#{k}: delta_hat={delta:.4g} (converged={converged}, floored={floored}) "
                  f"-> rollout failures {n_fail}/{n_eval} ({100*fail_frac:.2f}%)")

        results[k] = {"delta_hat": delta, "converged": converged, "floored": floored,
                      "iters": len(history), "history": history,
                      "n_fail": n_fail, "n_eval": n_eval, "fail_frac": fail_frac}

        if visualize:
            s_init = _pick_viz_state(cache, delta, rng)
            if s_init is not None:
                visualize_constraint_result(cache, s_init, delta,
                                            title_prefix=f"c#{k}  ", frame_pause=viz_pause)

    return {"per_c": results, "N": N, "K": K, "eps": eps, "beta": beta,
            "beta_per_run": beta_k, "mode": "per_c"}


def _pick_viz_state(cache: CachedConstraint, delta: float,
                    rng: np.random.Generator) -> Optional[np.ndarray]:
    """Pick a feasible start state near the recovered-set boundary:
    V_full(s) <= delta (inside S_hat), G(s) < 0 (outside obstacle), V as large
    as possible (closest to delta from below)."""
    try:
        cand = sample_x_under_delta(4000, delta if math.isfinite(delta) else float("inf"),
                                    cache, rng, max_tries=50)
    except RuntimeError:
        return None
    V = cache.value_at_full_BRS(cand)
    G = cache.G(cand[:, :2])
    feas = G < 0
    if not feas.any():
        return None
    pool, Vp = cand[feas], V[feas]
    return pool[int(np.argmax(Vp))]   # closest to delta from below


# =============================================================================
# 6b. JOINT scenario optimization (resamples obstacle each iteration)
# =============================================================================

def scenario_optimize_joint(
    model: FNOValueModel,
    prob: DubinsReachProblem,
    eps: float = 1e-2,
    beta: float = 1e-9,
    M: int = 20,
    seed: int = 0,
    states_per_c: int = 64,
    verbose: bool = True,
) -> dict:
    """One delta_hat over the joint distribution of (state, obstacle). Each
    iteration redraws obstacles via prob.obstacle_sampler."""
    if prob.obstacle_sampler is None:
        raise ValueError("joint mode requires prob.obstacle_sampler")

    N = required_N(eps, beta)
    n_c_per_iter = max(1, math.ceil(N / states_per_c))
    if verbose:
        print(f"[joint] eps={eps}, beta={beta} -> N={N} "
              f"({n_c_per_iter} obstacle-draws x {states_per_c} states)")

    rng = np.random.default_rng(seed)
    delta = float("inf")
    history = []

    for i in range(M):
        all_x, all_idx, caches = [], [], []
        for _ in range(n_c_per_iter):
            c_sdf = prob.obstacle_sampler(rng)
            cache = CachedConstraint(c_sdf, prob, model)
            try:
                xs = sample_x_under_delta(states_per_c, delta, cache, rng)
            except RuntimeError:
                continue
            caches.append(cache)
            all_x.append(xs)
            all_idx.append(np.full(len(xs), len(caches) - 1, dtype=int))

        if not all_x:
            if verbose:
                print(f"  iter {i}: no samples under delta={delta}; converged trivially.")
            break

        X = np.concatenate(all_x, axis=0)
        C_idx = np.concatenate(all_idx, axis=0)
        J = np.empty(len(X), dtype=np.float32)
        V_at_X = np.empty(len(X), dtype=np.float32)
        for ci, cache in enumerate(caches):
            m = C_idx == ci
            if not m.any():
                continue
            J[m] = rollout_cost(X[m], cache, delta=delta).astype(np.float32)
            V_at_X[m] = cache.value_at_full_BRS(X[m]).astype(np.float32)

        violators = J >= 0.0
        n_viol = int(violators.sum())
        if verbose:
            print(f"  iter {i}: delta={delta:.6g}, violators={n_viol}/{len(X)} (N={N})")
        history.append({"iter": i, "delta": delta, "violators": n_viol, "N_drawn": len(X)})

        if n_viol == 0:
            return {"delta_hat": delta, "converged": True, "iters": i + 1,
                    "N": N, "history": history, "mode": "joint", "eps": eps, "beta": beta}
        delta = float(V_at_X[violators].min())

    return {"delta_hat": delta, "converged": False, "iters": M,
            "N": N, "history": history, "mode": "joint", "eps": eps, "beta": beta}


# =============================================================================
# 6c. Convenience dispatcher
# =============================================================================

def scenario_optimize_reach(model: FNOValueModel, prob: DubinsReachProblem,
                            mode: str = "per_c", **kwargs) -> dict:
    if mode == "joint":
        return scenario_optimize_joint(model, prob, **kwargs)
    if mode == "per_c":
        c_list = kwargs.pop("c_list", prob.c_list)
        if c_list is None:
            raise ValueError("per_c mode needs c_list (kwarg or prob.c_list).")
        return scenario_optimize_per_c(model, prob, c_list, **kwargs)
    raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# 7. Helpers: load .mat / FNO, obstacle sampler, problem builder
# =============================================================================

def load_mat(path: str) -> dict:
    import scipy.io as sio
    raw = sio.loadmat(path, squeeze_me=True)
    return {
        "constraints": raw["constraints"].astype(np.float32),  # (M,Nx,Ny,Nth) obstacle SDFs
        "results":     raw["results"].astype(np.float32),      # (M,Nx,Ny,Nth,T) BRT
        "target_set":  raw["target_set"].astype(np.float32),   # (Nx,Ny,Nth) target SDF
        "tau":         raw["tau"].astype(np.float32),          # (T,)
        "x_axis":      raw["x_axis"].astype(np.float32),       # (Nx,)
        "y_axis":      raw["y_axis"].astype(np.float32),       # (Ny,)
        "theta_axis":  raw["theta_axis"].astype(np.float32),   # (Nth,)
    }


def load_fno(ckpt_path: str, device: str) -> torch.nn.Module:
    """Loads a full-model FNO3d save (torch.save(model, ...)).

    The training notebook defined FNO3d/SpectralConv3d inline, so the pickle
    references them as `__main__.FNO3d` / `__main__.SpectralConv3d`. Expose the
    repo classes there so the unpickler can resolve them outside the notebook.
    """
    import __main__
    __main__.FNO3d = FNO3d
    __main__.SpectralConv3d = SpectralConv3d
    net = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.eval()
    return net


def make_obstacle_sampler(odp_grid: Grid):
    """Returns sampler(rng) -> (Nx, Ny) obstacle SDF (negative inside), using the
    SAME random_obstacle_set distribution that produced the training data. The
    obstacle is theta-independent, so we return its (x, y) slice."""
    from HJR_FNO.data_gen.dubins3D_data_gen import random_obstacle_set

    def sampler(rng: np.random.Generator) -> np.ndarray:
        obs3d = np.asarray(random_obstacle_set(odp_grid, rng), dtype=np.float32)  # (Nx,Ny,Nth)
        return obs3d[:, :, 0]

    return sampler


def build_problem_from_mat(mat: dict,
                           obstacle_sampler: Optional[Callable] = None,
                           c_list: Optional[List[np.ndarray]] = None,
                           car: Optional[DubinsCar2] = None) -> DubinsReachProblem:
    Nx, Ny, Nth = len(mat["x_axis"]), len(mat["y_axis"]), len(mat["theta_axis"])
    # 3-D odp grid, periodic theta in [-pi, pi] (matches the data-gen grid).
    odp_grid = Grid(
        minBounds    = np.array([float(mat["x_axis"][0]),  float(mat["y_axis"][0]),  -math.pi]),
        maxBounds    = np.array([float(mat["x_axis"][-1]), float(mat["y_axis"][-1]),  math.pi]),
        dims         = 3,
        pts_each_dim = np.array([Nx, Ny, Nth]),
        periodicDims = [2],
    )
    # Target SDF: theta-independent -> 2-D slice.
    target = mat["target_set"]
    target_sdf = target[:, :, 0] if target.ndim == 3 else target

    if car is None:
        car = DubinsCar2(uMin=[0.0, -1.0], uMax=[1.0, 1.0], dMax=[0.1, 0.1, 0.1],
                         uMode="min", dMode="max")

    return DubinsReachProblem(
        x_axis     = mat["x_axis"],
        y_axis     = mat["y_axis"],
        theta_axis = mat["theta_axis"],
        tau        = mat["tau"],
        target_sdf = target_sdf,
        x_lo = np.array([float(mat["x_axis"][0]),  float(mat["y_axis"][0]),  -math.pi]),
        x_hi = np.array([float(mat["x_axis"][-1]), float(mat["y_axis"][-1]),  math.pi]),
        car        = car,
        odp_grid   = odp_grid,
        dt         = float(mat["tau"][1] - mat["tau"][0]),
        obstacle_sampler = obstacle_sampler,
        c_list     = c_list,
    )


# =============================================================================
# 8. Script entry point
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat",  default=DEFAULT_MAT_PATH)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT_PATH,
                        help="Full-model FNO3d save (torch.save(model, ...)).")
    parser.add_argument("--mode", choices=["joint", "per_c"], default="per_c")
    parser.add_argument("--eps",  type=float, default=1e-2)
    parser.add_argument("--beta", type=float, default=1e-9)
    parser.add_argument("--M",    type=int,   default=100)
    parser.add_argument("--K",    type=int,   default=10,
                        help="Number of test obstacle configs in per_c mode.")
    parser.add_argument("--seed", type=int,   default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--visualize", action="store_true",
                        help="Animate one rollout per obstacle (per_c mode only).")
    parser.add_argument("--viz-pause", type=float, default=0.1)
    parser.add_argument("--max-tries", type=int, default=400)
    parser.add_argument("--delta-floor", type=float, default=-1.40,
                        help="Lower clamp on delta_hat (per_c mode).")
    parser.add_argument("--delta-init", type=float, default=0.0,
                        help="Starting threshold; first iter samples {V<delta_init} (per_c mode).")
    parser.add_argument("--step-frac", type=float, default=0.5,
                        help="Damping on the delta update in [0,1]; 1.0=exact scenario step (per_c mode).")
    args = parser.parse_args()

    mat = load_mat(args.mat)
    fno = load_fno(args.ckpt, args.device)
    model = FNOValueModel(fno, device=args.device)

    if args.mode == "joint":
        prob = build_problem_from_mat(mat)
        prob.obstacle_sampler = make_obstacle_sampler(prob.odp_grid)
        out = scenario_optimize_joint(model, prob, eps=args.eps, beta=args.beta,
                                      M=args.M, seed=args.seed)
        print(f"\ndelta_hat (joint) = {out['delta_hat']:.6g} (converged={out['converged']})")
    else:
        K = min(args.K, mat["constraints"].shape[0])
        c_list = [mat["constraints"][i] for i in range(K)]
        prob = build_problem_from_mat(mat, c_list=c_list)
        out = scenario_optimize_per_c(model, prob, c_list, eps=args.eps, beta=args.beta,
                                      M=args.M, seed=args.seed, visualize=args.visualize,
                                      viz_pause=args.viz_pause, max_tries=args.max_tries,
                                      delta_floor=args.delta_floor, delta_init=args.delta_init,
                                      step_frac=args.step_frac)
        print("\nPer-obstacle results:")
        tot_fail = tot_eval = 0
        for k, info in out["per_c"].items():
            tot_fail += info["n_fail"]; tot_eval += info["n_eval"]
            print(f"  c#{k}: delta_hat={info['delta_hat']:.4g} "
                  f"(converged={info['converged']}, floored={info['floored']}, iters={info['iters']}) "
                  f"-> fails {info['n_fail']}/{info['n_eval']} ({100*info['fail_frac']:.2f}%)")
        if tot_eval:
            print(f"\nTotal rollout failures across {len(out['per_c'])} obstacles: "
                  f"{tot_fail}/{tot_eval} ({100*tot_fail/tot_eval:.2f}%)  "
                  f"[delta_floor={args.delta_floor}]")
