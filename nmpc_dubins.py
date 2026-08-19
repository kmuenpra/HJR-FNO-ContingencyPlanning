"""
Nonlinear MPC tracker for a Dubins/unicycle robot.

WHAT IT IS FOR
--------------
The RRTX tree is GEOMETRIC: bare (x, y) nodes joined by straight edges, so the
committed path has corners a bounded-curvature vehicle cannot drive. The previous
answer (dubins_path.py) pre-smoothed the polyline into Dubins arcs and then chased
a lookahead point with pure pursuit. That is generate-and-test: an arc is either
accepted or rejected wholesale, with no way to DEFORM it toward feasibility, so
~20% of arcs were discarded and replaced by straight edges with discontinuous
headings.

This module replaces that with an optimizer. Constraints are handled inside the
solve, so the tracker bends the trajectory instead of discarding it:

    minimize   sum_k  w_pos |p_k - p_ref,k|^2  +  w_v (v_k - v_ref)^2
                    + w_w  omega_k^2  +  w_dw (omega_k - omega_{k-1})^2
                    + w_term |p_N - p_ref,N|^2
    over       u = (v_0, omega_0, ..., v_{N-1}, omega_{N-1})
    subject to  v_min <= v_k <= v_max,  |omega_k| <= w_max        (bounds)
                slack(p_k) >= 0            reachable-set invariance
                state box                  x/y inside the domain

`slack` is supplied by the caller as a function of the predicted POSITIONS, so
this module knows nothing about HJ reachable sets: RRTX passes a closure over
hjr_fno (margin minus interpolated value). Obstacle avoidance is deliberately NOT
a separate term -- the caller's feasible_region already excludes obstacles.

DYNAMICS
--------
Forward Euler, in the SAME order as Utils.update_robot_position_dubins and
Navigation2DEnv.dynamics -- position steps with the OLD heading, then heading
updates:

    x_{k+1} = x_k + v_k dt cos(theta_k)
    y_{k+1} = y_k + v_k dt sin(theta_k)
    theta_{k+1} = wrap(theta_k + omega_k dt)

Keeping that order matters: it is what makes a behavioural difference between this
planner and the MPPI one attributable to the PLANNERS rather than their trackers.

SOLVER
------
scipy SLSQP (casadi is not installed in this environment). The objective is pure
numpy and cheap, so scipy may finite-difference it. The CONSTRAINT is the
expensive part -- every evaluation interpolates the reachable-set value -- so its
Jacobian is computed here with one forward-difference sweep whose perturbed
rollouts are stacked into a SINGLE batched slack() call. That call has a fixed
overhead which dominates its per-point cost, so N + 2N*N points cost about the
same as one.

Warm-started from the previous solution shifted by one step. If the solve fails or
returns an infeasible first move, `solve` reports it and the caller falls back to
pure pursuit -- a tracker that can refuse to move is worse than a crude one.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

# slack() returns +inf for points outside the value function's domain; SLSQP cannot
# work with infinities, so clamp to a large finite violation instead.
_SLACK_FLOOR = -1e3


def wrap_pi(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


class DubinsNMPC:
    """Receding-horizon tracker. One instance per robot; call solve() each step."""

    def __init__(
        self,
        dt: float = 0.1,
        horizon: int = 10,
        v_min: float = 0.0,
        v_max: float = 1.0,
        w_max: float = 1.0,
        w_pos: float = 10.0,
        w_v: float = 0.1,
        w_w: float = 0.05,
        w_dw: float = 0.5,
        w_term: float = 25.0,
        max_iter: int = 25,
        ftol: float = 1e-3,
        state_box: Optional[Tuple[float, float, float, float]] = None,
        slack_margin: float = 0.0,
        feas_tol: float = 1e-3,
    ):
        self.dt = float(dt)
        self.N = int(horizon)
        self.v_min, self.v_max = float(v_min), float(v_max)
        self.w_max = float(w_max)
        self.w_pos, self.w_v, self.w_w = float(w_pos), float(w_v), float(w_w)
        self.w_dw, self.w_term = float(w_dw), float(w_term)
        self.max_iter, self.ftol = int(max_iter), float(ftol)
        self.state_box = state_box          # (x_min, x_max, y_min, y_max) or None
        # SLSQP satisfies constraints only to ~ftol, so the applied step can sit a whisker
        # outside the set while the solver honestly reports success. Two knobs handle that:
        #   slack_margin -- ask the OPTIMIZER for slack >= margin, i.e. keep a buffer, so a
        #                   tolerance-sized violation of the tightened constraint still
        #                   satisfies the true one. This is how to make an invariance claim.
        #   feas_tol     -- how much violation the post-solve check tolerates before
        #                   declaring the step inadmissible and telling the caller to fall
        #                   back. Set below slack_margin so the buffer absorbs it.
        self.slack_margin = float(slack_margin)
        self.feas_tol = float(feas_tol)

        self._u_prev: Optional[np.ndarray] = None   # warm start
        self._omega_prev: float = 0.0               # for the rate-of-change term

        # diagnostics the caller can read/print
        self.last_status = ""
        self.n_solves = 0
        self.n_failures = 0
        self.n_infeasible_first_step = 0

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def rollout(self, state, u) -> np.ndarray:
        """Integrate the horizon. Returns (N+1, 3) states, row 0 == `state`."""
        u = np.asarray(u, dtype=float).reshape(self.N, 2)
        xs = np.empty((self.N + 1, 3), dtype=float)
        xs[0] = state
        dt = self.dt
        for k in range(self.N):
            x, y, th = xs[k]
            v, w = u[k]
            xs[k + 1, 0] = x + v * dt * math.cos(th)      # OLD heading -- see docstring
            xs[k + 1, 1] = y + v * dt * math.sin(th)
            xs[k + 1, 2] = wrap_pi(th + w * dt)
        return xs

    def _rollout_batch(self, state, U) -> np.ndarray:
        """Vectorized rollout of M control sequences. U is (M, N, 2) -> (M, N+1, 3)."""
        U = np.asarray(U, dtype=float)
        M = U.shape[0]
        xs = np.empty((M, self.N + 1, 3), dtype=float)
        xs[:, 0, :] = np.asarray(state, dtype=float)
        dt = self.dt
        for k in range(self.N):
            x, y, th = xs[:, k, 0], xs[:, k, 1], xs[:, k, 2]
            v, w = U[:, k, 0], U[:, k, 1]
            xs[:, k + 1, 0] = x + v * dt * np.cos(th)
            xs[:, k + 1, 1] = y + v * dt * np.sin(th)
            xs[:, k + 1, 2] = wrap_pi(th + w * dt)
        return xs

    # ------------------------------------------------------------------
    # objective
    # ------------------------------------------------------------------
    def _cost(self, state, u, ref) -> float:
        u = np.asarray(u, dtype=float).reshape(self.N, 2)
        xs = self.rollout(state, u)
        p, v, w = xs[1:, :2], u[:, 0], u[:, 1]

        e = p - ref[1:self.N + 1]
        J = self.w_pos * float(np.sum(e * e))
        J += self.w_term * float(np.sum((p[-1] - ref[self.N]) ** 2))
        J += self.w_v * float(np.sum((v - self.v_max) ** 2))      # prefer full speed
        J += self.w_w * float(np.sum(w * w))
        dw = np.diff(np.concatenate([[self._omega_prev], w]))
        J += self.w_dw * float(np.sum(dw * dw))
        return J

    # ------------------------------------------------------------------
    # constraints
    # ------------------------------------------------------------------
    def _state_box_slack(self, xs) -> np.ndarray:
        """Four slacks per predicted state; empty when no box is configured."""
        if self.state_box is None:
            return np.empty(0)
        x_min, x_max, y_min, y_max = self.state_box
        p = xs[1:, :2]
        return np.concatenate([p[:, 0] - x_min, x_max - p[:, 0],
                               p[:, 1] - y_min, y_max - p[:, 1]])

    def _cons(self, state, u, slack_fn) -> np.ndarray:
        xs = self.rollout(state, u)
        parts = []
        if slack_fn is not None:
            s = np.asarray(slack_fn(xs[1:, :2]), dtype=float) - self.slack_margin
            parts.append(np.maximum(s, _SLACK_FLOOR))
        parts.append(self._state_box_slack(xs))
        return np.concatenate(parts) if parts else np.empty(0)

    def _cons_jac(self, state, u, slack_fn) -> np.ndarray:
        """Forward-difference Jacobian, all perturbed rollouts in ONE slack() call."""
        u = np.asarray(u, dtype=float).reshape(-1)
        n = u.size
        h = 1e-4

        U = np.repeat(u[None, :], n + 1, axis=0)          # row 0 nominal, then n perturbed
        for j in range(n):
            U[j + 1, j] += h
        xs = self._rollout_batch(state, U.reshape(n + 1, self.N, 2))

        if slack_fn is not None:
            pts = xs[:, 1:, :2].reshape(-1, 2)            # ((n+1)*N, 2) -- one batched call
            s = np.asarray(slack_fn(pts), dtype=float).reshape(n + 1, self.N)
            s = np.maximum(s - self.slack_margin, _SLACK_FLOOR)
        else:
            s = np.empty((n + 1, 0))

        box = np.stack([self._state_box_slack(xs[i]) for i in range(n + 1)], axis=0)
        C = np.concatenate([s, box], axis=1)              # (n+1, m)
        return (C[1:] - C[0][None, :]).T / h              # (m, n)

    # ------------------------------------------------------------------
    def solve(self, state, ref, slack_fn: Optional[Callable] = None):
        """One MPC step.

        @params
        - state    : (x, y, theta) current pose
        - ref      : (N+1, 2) desired positions, ref[0] alongside the robot
        - slack_fn : points (K,2) -> slack (K,), >= 0 meaning admissible. None disables
                     the invariance constraint (e.g. HJ_contingency_enable=False).

        @return (u0, predicted_states, ok)
            u0               : (v, omega) to apply now
            predicted_states : (N+1, 3) the plan, for plotting
            ok               : False when the solver failed or the first step is
                               inadmissible -- caller should fall back.
        """
        self.n_solves += 1
        N = self.N
        ref = np.asarray(ref, dtype=float)
        assert ref.shape[0] >= N + 1, f"reference needs {N+1} rows, got {ref.shape[0]}"

        # warm start: shift the previous solution, repeat its tail
        if self._u_prev is not None:
            u0 = np.vstack([self._u_prev[1:], self._u_prev[-1]])
        else:
            u0 = np.column_stack([np.full(N, self.v_max), np.zeros(N)])
        u0 = np.clip(u0, [self.v_min, -self.w_max], [self.v_max, self.w_max]).reshape(-1)

        bounds = [(self.v_min, self.v_max), (-self.w_max, self.w_max)] * N
        cons = [{
            "type": "ineq",
            "fun": lambda z: self._cons(state, z, slack_fn),
            "jac": lambda z: self._cons_jac(state, z, slack_fn),
        }]

        try:
            res = minimize(
                lambda z: self._cost(state, z, ref), u0,
                method="SLSQP", bounds=bounds, constraints=cons,
                options={"maxiter": self.max_iter, "ftol": self.ftol},
            )
            u_opt = np.asarray(res.x, dtype=float)
            ok = bool(res.success)
            self.last_status = str(res.message)
        except Exception as exc:                     # solver blew up -> caller falls back
            self.n_failures += 1
            self.last_status = f"exception: {exc}"
            return None, None, False

        u_opt = np.clip(u_opt.reshape(N, 2),
                        [self.v_min, -self.w_max], [self.v_max, self.w_max])
        xs = self.rollout(state, u_opt)

        # Even a "successful" solve can end slightly infeasible; check the step we are
        # about to APPLY rather than trusting the status flag.
        if slack_fn is not None:
            s0 = float(np.asarray(slack_fn(xs[1:2, :2]), dtype=float)[0])
            if s0 < -self.feas_tol:
                self.n_infeasible_first_step += 1
                self.last_status += " | first step inadmissible"
                ok = False

        if not ok:
            self.n_failures += 1

        self._u_prev = u_opt
        self._omega_prev = float(u_opt[0, 1])
        return u_opt[0].copy(), xs, ok

    def reset(self):
        """Drop the warm start (after a contingency/teleport the old plan is garbage)."""
        self._u_prev = None
        self._omega_prev = 0.0


# ----------------------------------------------------------------------
def reference_from_polyline(pts, n_steps: int, ds: float) -> np.ndarray:
    """Sample a polyline at `ds` arc-length spacing into an (n_steps+1, 2) reference.

    `pts` is the committed waypoint chain (robot first, goal last). Sampling at
    ds = v_max*dt makes the reference reachable at full speed; when the polyline is
    shorter than the horizon the final point repeats, which is what makes the robot
    settle onto the goal instead of overshooting it.
    """
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        raise ValueError("pts must be (M, 2)")
    if pts.shape[0] == 1:
        return np.repeat(pts, n_steps + 1, axis=0)

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    want = np.arange(n_steps + 1, dtype=float) * ds

    out = np.empty((n_steps + 1, 2), dtype=float)
    for i, s in enumerate(want):
        if s >= cum[-1]:
            out[i] = pts[-1]
            continue
        k = int(np.searchsorted(cum, s, side="right"))
        k = max(1, min(k, len(seg)))
        denom = seg[k - 1] if seg[k - 1] > 1e-12 else 1.0
        f = (s - cum[k - 1]) / denom
        out[i] = pts[k - 1] + f * (pts[k] - pts[k - 1])
    return out


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Self-test: control limits, tracking, and a hard constraint the optimizer must
    # respect by BENDING the trajectory (the thing pure pursuit + arc rejection could
    # not do).
    dt, N = 0.1, 12
    mpc = DubinsNMPC(dt=dt, horizon=N, v_min=0.0, v_max=1.0, w_max=1.0,
                     state_box=(-20, 20, -20, 20))

    print("test 1: straight reference, no constraint")
    state = np.array([0.0, 0.0, 0.0])
    ref = reference_from_polyline(np.array([[0.0, 0.0], [20.0, 0.0]]), N, dt * 1.0)
    traj = [state.copy()]
    for _ in range(80):
        u, xs, ok = mpc.solve(state, ref, None)
        assert ok, mpc.last_status
        assert -1.0 - 1e-9 <= u[1] <= 1.0 + 1e-9 and 0.0 - 1e-9 <= u[0] <= 1.0 + 1e-9
        state = mpc.rollout(state, mpc._u_prev)[1]
        traj.append(state.copy())
        ref = reference_from_polyline(np.array([state[:2], [20.0, 0.0]]), N, dt * 1.0)
    traj = np.array(traj)
    print(f"  travelled {traj[-1,0]:.2f} m in x, |y| max = {np.abs(traj[:,1]).max():.2e}"
          f"  (want ~8 m, ~0)")

    print("test 2: reference cuts through a forbidden half-plane y > 1")
    #   slack = 1 - y  ->  the optimizer must keep y <= 1 while chasing a ref at y = 3
    mpc.reset()
    state = np.array([0.0, 0.0, 0.0])
    worst = -np.inf
    for _ in range(60):
        ref = reference_from_polyline(np.array([state[:2], [10.0, 3.0]]), N, dt * 1.0)
        u, xs, ok = mpc.solve(state, ref, lambda p: 1.0 - p[:, 1])
        assert ok, mpc.last_status
        state = mpc.rollout(state, mpc._u_prev)[1]
        worst = max(worst, state[1])
    print(f"  max y reached = {worst:.4f}   (constraint y <= 1)")
    assert worst <= 1.05, "invariance constraint violated"

    print("test 3: infeasible start is reported, not silently driven")
    mpc.reset()
    u, xs, ok = mpc.solve(np.array([0.0, 5.0, 0.0]),
                          reference_from_polyline(np.array([[0.0, 5.0], [5.0, 5.0]]), N, 0.1),
                          lambda p: 1.0 - p[:, 1])          # start already at y=5 > 1
    print(f"  ok={ok}  status={mpc.last_status!r}  (want ok=False)")

    print(f"\nsolves={mpc.n_solves} failures={mpc.n_failures} "
          f"infeasible_first={mpc.n_infeasible_first_step}")
