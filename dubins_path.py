"""
Minimal Dubins shortest-path solver (Shkel & Lumelsky word forms).

WHAT IT IS FOR
--------------
The RRTX tree is GEOMETRIC: its nodes are bare (x, y) points and its edges are
straight lines, so the committed path has sharp corners. A bounded-curvature
vehicle cannot drive a sharp corner -- at speed v with turn-rate limit w_max the
tightest circle it can trace has radius

    rho = v / w_max

This module replaces each corner with the tightest legal curve, turning the
polyline into a reference the vehicle can physically follow. Consumed by
``RRTX.dubins_reference()`` in rrtx_FNO3d_oneGoal.py.

THE IDEA
--------
A Dubins car has three moves: hard left (L), hard right (R), straight (S).
Dubins (1957) proved the shortest path between two POSES is always exactly three
of these, and only six combinations are ever optimal:

    LSL, RSR, LSR, RSL      "swing out, run straight, swing in"
    RLR, LRL                "swerve around"  (only when the poses are close;
                             ~3% of random pairs in testing)

So this is not a search: plug the two poses into six closed-form formulas and
take the smallest. Self-contained -- math + numpy, no external `dubins` package.

CONVENTIONS
-----------
A pose is q = (x, y, theta), theta in radians (any range; wrapped internally).
Segment lengths are returned in NORMALIZED units (unit turning radius): for
'L'/'R' a length is the turn angle in radians, for 'S' it is a distance in units
of rho. Multiply the total by rho for metres. Headings come back wrapped to
[-pi, pi), the convention used by the env, the trackers and the HJR grids.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

_WORDS = ("LSL", "RSR", "LSR", "RSL", "RLR", "LRL")


def _mod2pi(x: float) -> float:
    return x % (2.0 * math.pi)


def wrap_pi(x: float) -> float:
    """Wrap to [-pi, pi)."""
    return (x + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# The six words, in the normalized (alpha, beta, d) frame.
# alpha/beta are the start/end headings measured relative to the straight line
# joining the two points; d is that line's length in units of rho. Each returns
# the three segment lengths (t, p, q), or None if the word is inadmissible.
# ---------------------------------------------------------------------------
def _LSL(a, b, d):
    tmp0 = d + math.sin(a) - math.sin(b)
    p_sq = 2 + d * d - 2 * math.cos(a - b) + 2 * d * (math.sin(a) - math.sin(b))
    if p_sq < 0:
        return None
    tmp1 = math.atan2(math.cos(b) - math.cos(a), tmp0)
    return _mod2pi(tmp1 - a), math.sqrt(p_sq), _mod2pi(b - tmp1)


def _RSR(a, b, d):
    tmp0 = d - math.sin(a) + math.sin(b)
    p_sq = 2 + d * d - 2 * math.cos(a - b) + 2 * d * (math.sin(b) - math.sin(a))
    if p_sq < 0:
        return None
    tmp1 = math.atan2(math.cos(a) - math.cos(b), tmp0)
    return _mod2pi(a - tmp1), math.sqrt(p_sq), _mod2pi(tmp1 - b)


def _LSR(a, b, d):
    p_sq = -2 + d * d + 2 * math.cos(a - b) + 2 * d * (math.sin(a) + math.sin(b))
    if p_sq < 0:
        return None
    p = math.sqrt(p_sq)
    tmp = math.atan2(-math.cos(a) - math.cos(b), d + math.sin(a) + math.sin(b)) \
        - math.atan2(-2.0, p)
    return _mod2pi(tmp - a), p, _mod2pi(tmp - _mod2pi(b))


def _RSL(a, b, d):
    p_sq = d * d - 2 + 2 * math.cos(a - b) - 2 * d * (math.sin(a) + math.sin(b))
    if p_sq < 0:
        return None
    p = math.sqrt(p_sq)
    tmp = math.atan2(math.cos(a) + math.cos(b), d - math.sin(a) - math.sin(b)) \
        - math.atan2(2.0, p)
    return _mod2pi(a - tmp), p, _mod2pi(_mod2pi(b) - tmp)


def _RLR(a, b, d):
    tmp = (6.0 - d * d + 2 * math.cos(a - b) + 2 * d * (math.sin(a) - math.sin(b))) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = _mod2pi(2 * math.pi - math.acos(tmp))
    t = _mod2pi(a - math.atan2(math.cos(a) - math.cos(b),
                               d - math.sin(a) + math.sin(b)) + p / 2.0)
    return t, p, _mod2pi(a - b - t + p)


def _LRL(a, b, d):
    tmp = (6.0 - d * d + 2 * math.cos(a - b) + 2 * d * (math.sin(b) - math.sin(a))) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = _mod2pi(2 * math.pi - math.acos(tmp))
    t = _mod2pi(-a + math.atan2(math.cos(b) - math.cos(a),
                                d + math.sin(a) - math.sin(b)) + p / 2.0)
    # NOTE the last term is (-t + mod2pi(p)); an earlier version had (+2p - t),
    # which gave a wrong endpoint on ~1% of pose pairs -- exactly the ~3% of cases
    # where LRL/RLR win, so it hid easily. Covered by the __main__ test below.
    return t, p, _mod2pi(_mod2pi(b) - a - t + _mod2pi(p))


_SOLVERS = {
    "LSL": _LSL, "RSR": _RSR, "LSR": _LSR,
    "RSL": _RSL, "RLR": _RLR, "LRL": _LRL,
}


# ---------------------------------------------------------------------------
def shortest_path(
    q0: Tuple[float, float, float],
    q1: Tuple[float, float, float],
    rho: float,
) -> Optional[Tuple[float, str, Tuple[float, float, float]]]:
    """Shortest Dubins path from pose q0 to pose q1 at turning radius rho.

    @return (length_in_metres, word, (t, p, q) normalized lengths), or None if no
            word is admissible (should not happen for rho > 0).
    """
    dx, dy = q1[0] - q0[0], q1[1] - q0[1]
    D = math.hypot(dx, dy)
    d = D / rho
    phi = math.atan2(dy, dx) if D > 1e-12 else 0.0
    a = _mod2pi(q0[2] - phi)
    b = _mod2pi(q1[2] - phi)

    best = None
    for word in _WORDS:
        sol = _SOLVERS[word](a, b, d)
        if sol is None:
            continue
        cost = sum(sol)
        if best is None or cost < best[0]:
            best = (cost, word, sol)
    if best is None:
        return None
    return best[0] * rho, best[1], best[2]


def _advance(x, y, th, seg_len, mode):
    """Integrate one segment of `seg_len` (normalized) from (x, y, th)."""
    if mode == "L":
        return (x + math.sin(th + seg_len) - math.sin(th),
                y - math.cos(th + seg_len) + math.cos(th),
                th + seg_len)
    if mode == "R":
        return (x - math.sin(th - seg_len) + math.sin(th),
                y + math.cos(th - seg_len) - math.cos(th),
                th - seg_len)
    return (x + seg_len * math.cos(th), y + seg_len * math.sin(th), th)


def sample_path(q0, word, lengths, rho, ds: float = 0.2) -> np.ndarray:
    """Sample a solved Dubins path at ~`ds` metre arc-length spacing.

    @return (N, 3) poses [x, y, theta]; row 0 == q0, last row == the endpoint.
    """
    total = sum(lengths)
    n = max(2, int(math.ceil(total * rho / max(ds, 1e-6))) + 1)
    ss = np.linspace(0.0, total, n)

    out = np.empty((n, 3), dtype=float)
    for i, s in enumerate(ss):
        x, y, th = 0.0, 0.0, q0[2]      # integrate in the normalized frame
        left = s
        for seg_len, mode in zip(lengths, word):
            step = min(seg_len, left)
            x, y, th = _advance(x, y, th, step, mode)
            left -= step
            if left <= 1e-12:
                break
        out[i] = (q0[0] + rho * x, q0[1] + rho * y, wrap_pi(th))
    return out


def path_between(q0, q1, rho, ds: float = 0.2) -> Optional[np.ndarray]:
    """shortest_path + sample_path in one call. (N, 3) poses, or None."""
    sol = shortest_path(q0, q1, rho)
    if sol is None:
        return None
    _, word, lengths = sol
    return sample_path(q0, word, lengths, rho, ds)


# ---------------------------------------------------------------------------
def waypoint_headings(pts: np.ndarray, theta_start: float) -> np.ndarray:
    """Assign a heading to every waypoint of a geometric polyline.

    A Dubins path needs a heading at BOTH ends, but tree nodes are bare points.
    So: the first waypoint uses the robot's real current heading (the reference
    then leaves from where the robot actually points -- no initial jump), each
    interior waypoint gets the ANGLE BISECTOR of the incoming and outgoing
    segment directions (splitting the turn evenly across the corner instead of
    dumping it all on one side -- at a 90 degree corner, "point at 45 degrees
    while passing through"), and the last inherits the final segment direction.
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    th = np.empty(n, dtype=float)
    th[0] = wrap_pi(theta_start)
    if n == 1:
        return th

    seg = np.arctan2(np.diff(pts[:, 1]), np.diff(pts[:, 0]))   # (n-1,)
    for i in range(1, n - 1):
        # bisector via unit-vector sum: robust to the +-pi wrap
        vx = math.cos(seg[i - 1]) + math.cos(seg[i])
        vy = math.sin(seg[i - 1]) + math.sin(seg[i])
        th[i] = seg[i] if (abs(vx) < 1e-9 and abs(vy) < 1e-9) else math.atan2(vy, vx)
    th[-1] = seg[-1]
    return th


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # regression test: endpoint exactness, length consistency, curvature bound
    rng = np.random.default_rng(0)
    rho = 0.6
    max_end = max_len = max_k = 0.0
    for _ in range(5000):
        q0 = (rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-math.pi, math.pi))
        q1 = (rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-math.pi, math.pi))
        L, word, lens = shortest_path(q0, q1, rho)
        pts = sample_path(q0, word, lens, rho, ds=0.01)
        max_end = max(max_end,
                      math.hypot(pts[-1, 0] - q1[0], pts[-1, 1] - q1[1]),
                      abs(wrap_pi(pts[-1, 2] - q1[2])))
        arc = np.sum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))
        max_len = max(max_len, abs(arc - L) / max(1.0, L))
        dth = np.abs([wrap_pi(b - a) for a, b in zip(pts[:-1, 2], pts[1:, 2])])
        dsg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
        m = dsg > 1e-9
        if m.any():
            max_k = max(max_k, (dth[m] / dsg[m]).max())
    print("5000 random pose pairs:")
    print(f"  max endpoint error : {max_end:.2e}   (want ~1e-14)")
    print(f"  max length error   : {max_len:.2e}   (want <1e-3)")
    print(f"  max curvature      : {max_k:.4f}     (bound 1/rho = {1/rho:.4f})")
