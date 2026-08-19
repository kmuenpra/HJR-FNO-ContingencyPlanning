"""Topological guidance for MPPI.

Wraps the standalone topo_prm planner (topo_prm/) and a Dubins path-tracker to
produce M *ancillary proposal means* for guided multi-group MPPI: TopoPRM finds
M homotopy-distinct paths inside the reachable set, and the tracker turns each
path into a dynamically-feasible control sequence (matching the env dynamics)
that MPPI then perturbs around.

The tracker is a simple pure-pursuit stand-in for the "per-path NMPC" in the
method; it is dynamically feasible (controls clamped to bounds, rolled through
the same unicycle update as the env), which is all MPPI needs from a mean.

Cadence (measured; see the two-tier split below):
  - The ROADMAP is a geometric object over (occupancy AND reachable set), so it
    only needs rebuilding when one of those changes -- i.e. when the lidar
    reveals new obstacles. That is `replan()`, ~640 ms with PRM_QUALITY.
  - The MEANS depend on the robot's current state, so `group_means()` re-tracks
    the cached paths EVERY control step. That is ~2.6 ms for 6 paths.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# make the standalone topo_prm package importable
_TOPO_DIR = Path(__file__).resolve().parents[1] / "topo_prm"
if str(_TOPO_DIR) not in sys.path:
    sys.path.insert(0, str(_TOPO_DIR))
from topo_prm import TopoPRM  # noqa: E402

# Sampling budgets. TopoPRM.create_graph is capped by WALL CLOCK
# (max_sample_time), so these presets are literally "how long may the roadmap
# build take". QUALITY ~640 ms / ~6 paths, FAST ~67 ms / ~3-4 paths. QUALITY is
# the default because a replan only fires on obstacle-reveal steps.
PRM_QUALITY = dict(
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
PRM_FAST = dict(
    PRM_QUALITY,
    resolution=0.4,
    max_sample_num=200,
    max_sample_time=0.05,
    max_raw_path=8,
    reserve_num=4,
)
# WIDE: whole-domain sampling instead of the start->goal ellipse. Needed when the
# feasible (reachable-set) corridor requires a large detour PERPENDICULAR to the
# straight start->goal line -- e.g. a U-shaped reachable set (env_D), where the
# ellipse never samples the bottom of the U and TopoPRM returns 0 paths. Costs
# more (bigger region, more rejection) so it's opt-in; ~1-2 s per replan.
PRM_WIDE = dict(
    PRM_QUALITY,
    sample_mode="domain",
    max_sample_num=3000,
    max_sample_time=2.0,
    reserve_num=8,
)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _np(x) -> np.ndarray:
    """Array view of a torch tensor (any device) or array-like."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class TopoGuidance:
    """Builds topo-PRM paths and tracks them into MPPI ancillary control means."""

    def __init__(
        self,
        obstacle_map,
        horizon: int,
        u_min,
        u_max,
        dt: float = 0.1,
        feasible_fn=None,
        lookahead: float = 1.5,
        track_ds: float = 0.2,
        prm_kwargs: Optional[dict] = None,
        replan_dist: float = 1.0,
        replan_every: int = 40,
        prev_best_patience: int = 20,
        replan_fail_backoff: int = 10,
        topo_mass_floor: float = 0.05,
        min_paths_frac: float = 0.5,
    ) -> None:
        """
        obstacle_map: the env's ObstacleMap (occupancy read live each replan).
            TopoPRM reads its numpy `_map` / `_cell_size` / `_cell_map_origin`
            directly, so no copy or adapter is needed.
        horizon, dt: MPPI horizon and env timestep (for tracking rollouts).
        u_min/u_max: control bounds [v, omega] (torch or array).
        feasible_fn: (N,2)->bool reachable-set membership; masks V(x)>0 as
            occupied so all paths stay inside the reachable set.
        lookahead: pure-pursuit lookahead distance [m].
        track_ds: arc-length spacing [m] the path is resampled to before tracking.
            Must be < lookahead, and small enough that a lookahead point exists;
            shortcut paths can be as coarse as two waypoints.
        prm_kwargs: passed to TopoPRM; defaults to PRM_QUALITY.
        replan_dist: replan once the robot has moved this far [m] from the point
            the cached paths were planned FROM. This is the trigger that matters
            most: every path is built around its start point, and the PRM's
            sampling ellipse is centered on that start->goal segment, so both go
            stale with displacement rather than with time.
        replan_every: periodic floor [control steps]; insurance against a trigger
            nobody thought of.
        prev_best_patience: replan after this many consecutive steps in which the
            ancillary means contributed nothing -- i.e. the previous-MPPI-mean
            group scored best AND the topo groups held less than topo_mass_floor
            of the total weight. Both conditions are needed: "prev is best" alone
            is the HEALTHY steady state (the prev mean is just the refined version
            of the path already being followed), so keying on it alone turns this
            into a second, more aggressive periodic trigger.
        topo_mass_floor: weight-mass share below which the ancillary means count
            as contributing nothing (see prev_best_patience).
        replan_fail_backoff: after a replan finds NO paths, wait this many control
            steps before trying again (an obstacle reveal still retries at once).
            Without it a hopeless world costs a full roadmap build every step.
        min_paths_frac: on an obstacle reveal, cached paths that the new obstacle
            blocked are PRUNED; a full rebuild only happens if fewer than this
            fraction of them survive. This is what keeps a reveal behind the
            robot from costing a ~500 ms rebuild.
        """
        self.obstacle_map = obstacle_map
        self.horizon = int(horizon)
        self.dt = float(dt)
        u_min = u_min.tolist() if torch.is_tensor(u_min) else list(u_min)
        u_max = u_max.tolist() if torch.is_tensor(u_max) else list(u_max)
        self.vmin, self.wmin = float(u_min[0]), float(u_min[1])
        self.vmax, self.wmax = float(u_max[0]), float(u_max[1])
        self.feasible_fn = feasible_fn
        self.lookahead = float(lookahead)
        self.track_ds = float(track_ds)
        self.prm_kwargs = dict(PRM_QUALITY if prm_kwargs is None else prm_kwargs)
        self.replan_dist = float(replan_dist)
        self.replan_every = int(replan_every)
        self.prev_best_patience = int(prev_best_patience)
        self.topo_mass_floor = float(topo_mass_floor)
        self.replan_fail_backoff = int(replan_fail_backoff)
        self.min_paths_frac = float(min_paths_frac)
        self.paths: List[np.ndarray] = []
        self.prm: Optional[TopoPRM] = None
        self.cost_to_go = None  # GeodesicCostToGo, built lazily on first replan
        # trigger state
        self._replan_xy = np.zeros(2)
        self._steps_since = 0
        self._prev_best_streak = 0
        self._fail_cooldown = 0
        # diagnostics
        self.num_replans = 0
        self.num_prunes = 0
        self.last_replan_ms = 0.0
        self.last_check_ms = 0.0
        self.last_track_ms = 0.0
        self.last_reason = ""

    def maybe_replan(
        self,
        state,
        goal_xy,
        obs_revealed: bool,
        group_died: bool = False,
        best_group: Optional[int] = None,
        num_groups: Optional[int] = None,
        topo_mass: Optional[float] = None,
    ) -> bool:
        """Decide whether to rebuild the roadmap, and do it if so.

        Triggers, in priority order:
          init          no roadmap / no cached paths yet.
          moved         robot displaced > replan_dist from where the paths were
                        planned from (paths and sampling ellipse are both built
                        around that point).
          group-died    RBR found a group with no feasible survivors, i.e. a
                        homotopy class died -- go find a replacement.
          topo-stale    the previous-MPPI-mean group has scored best for
                        prev_best_patience steps running; the ancillary means are
                        contributing nothing.
          periodic      replan_every steps elapsed (floor).
          paths-blocked a reveal invalidated too many cached paths (see below).

        An obstacle reveal is handled SELECTIVELY, not as an automatic rebuild:
        the grids are refreshed (~13 ms: ESDF + one batched feasible_fn call) and
        the cached paths re-tested. Blocked paths are pruned; a full rebuild
        (~500 ms) only happens if fewer than min_paths_frac of them survive.

        Args:
            state: current robot state [x, y, theta].
            goal_xy: goal position.
            obs_revealed: env._obs_revealed for this step.
            group_died / best_group / num_groups: solver feedback (_group_dead.
                any(), _selected_group, _num_groups). Optional.
        Returns:
            bool: whether a replan happened this call.
        """
        xy = _np(state).astype(float).reshape(-1)[:2]
        self._steps_since += 1

        # streak of "the ancillary means contributed nothing this step": the prev
        # mean scored best AND the topo groups hold a negligible share of the
        # weight. Either condition alone is normal.
        if best_group is not None and num_groups:
            useless = best_group == num_groups - 1 and (
                topo_mass is None or topo_mass < self.topo_mass_floor
            )
            self._prev_best_streak = self._prev_best_streak + 1 if useless else 0

        reason = None
        if self.prm is None or not self.paths:
            # A replan that found NOTHING must not be retried every step: with no
            # cached paths the "init" trigger re-fires immediately, so the loop
            # pays a full ~0.6 s roadmap build per control step for nothing (this
            # is what a `topo=0 REPLAN[init]` line every step means). Back off and
            # let the contingency behaviour run instead; a lidar reveal
            # (obs_revealed) still forces an immediate retry below, because that
            # is the one event that can actually change the answer.
            if self._fail_cooldown > 0 and not obs_revealed:
                self._fail_cooldown -= 1
                return False
            reason = "init"
        elif float(np.linalg.norm(xy - self._replan_xy)) > self.replan_dist:
            reason = "moved"
        elif group_died:
            reason = "group-died"
        elif self._prev_best_streak >= self.prev_best_patience:
            reason = "topo-stale"
        elif self._steps_since >= self.replan_every:
            reason = "periodic"
        elif obs_revealed and not self._paths_survive_reveal():
            reason = "paths-blocked"

        if reason is None:
            return False
        self.replan(xy, goal_xy, reason)
        return True

    def _paths_survive_reveal(self) -> bool:
        """Refresh the grids against the new occupancy + reachable set, prune the
        cached paths that are now blocked, and report whether enough survived to
        skip a full rebuild. Cheap: TopoPRM ctor (ESDF) + one batched feasible_fn
        call, ~13 ms measured, vs ~500 ms for find_topo_paths."""
        t0 = time.perf_counter()
        prm = self._fresh_prm()
        kept = [p for p in self.paths if self._path_clear(prm, p)]
        self.last_check_ms = 1e3 * (time.perf_counter() - t0)

        if len(kept) < max(1, int(np.ceil(self.min_paths_frac * len(self.paths)))):
            return False
        if len(kept) < len(self.paths):
            self.num_prunes += 1
        # keep the refreshed grids: they now match the live map
        self.prm, self.paths = prm, kept
        return True

    def _fresh_prm(self) -> TopoPRM:
        """A TopoPRM whose ESDF / feasibility grids match the CURRENT map. Does
        not search for paths."""
        prm = TopoPRM(self.obstacle_map, **self.prm_kwargs)
        if self.feasible_fn is not None:
            prm.rasterize_feasible(self.feasible_fn)
        return prm

    @staticmethod
    def _path_clear(prm: TopoPRM, path: np.ndarray) -> bool:
        """Whether every segment of `path` is still obstacle-free AND inside the
        feasible set, using the PRM's combined grid (one lookup per sample)."""
        return all(
            prm.line_visible(a, b)[0] for a, b in zip(path[:-1], path[1:])
        )

    def replan(self, start_xy, goal_xy, reason: str = "manual") -> List[np.ndarray]:
        """Rebuild the roadmap over the CURRENT occupancy + reachable set and
        extract the homotopy-distinct paths. Prefer maybe_replan()."""
        t0 = time.perf_counter()
        start = _np(start_xy).astype(float).reshape(-1)[:2]
        goal = _np(goal_xy).astype(float).reshape(-1)[:2]

        # Geodesic cost-to-go over the feasible corridor (obstacle- + reachable-
        # set-aware). Built once, recomputed here on each replan. It (a) gives the
        # MPPI goal cost -- see navigation2d wiring -- and (b) supplies the
        # goal-reachable feasible cells that bias TopoPRM sampling into the
        # corridor (no rejection waste; covers big perpendicular detours).
        if self.cost_to_go is None:
            from cost_to_go import GeodesicCostToGo

            self.cost_to_go = GeodesicCostToGo(
                self.obstacle_map, feasible_fn=self.feasible_fn
            )
        self.cost_to_go.recompute(goal)

        self.prm = self._fresh_prm()
        cells = self.cost_to_go.reachable_cells()
        if len(cells) == 0:
            # goal not reachable through the feasible set -> no topo path
            self.paths = []
        else:
            self.prm.sample_mode = "corridor"
            self.prm.set_sample_cells(cells)
            self.paths = self.prm.find_topo_paths(start, goal)
        self.last_replan_ms = 1e3 * (time.perf_counter() - t0)
        self.num_replans += 1
        self.last_reason = reason
        # reset trigger state
        self._replan_xy = start
        self._steps_since = 0
        self._prev_best_streak = 0
        # arm the backoff only when the roadmap came back empty
        self._fail_cooldown = 0 if self.paths else self.replan_fail_backoff
        return self.paths

    def group_means(self, state, device, dtype) -> Optional[torch.Tensor]:
        """Track each cached path from the current state into an MPPI mean.
        Returns (M, horizon, dim_control) tensor, or None if no paths."""
        if not self.paths:
            return None
        t0 = time.perf_counter()
        s = _np(state).astype(float)
        means = [self._track(s, p) for p in self.paths]
        out = torch.as_tensor(np.stack(means), device=device, dtype=dtype)
        self.last_track_ms = 1e3 * (time.perf_counter() - t0)
        return out

    def _resample(self, path: np.ndarray) -> np.ndarray:
        """Uniform arc-length resampling to `track_ds` spacing.

        REQUIRED before pure pursuit: shortcut_path() string-pulls a raw graph
        path down to as few as TWO waypoints (e.g. [[-9,-9],[8,8]]), and a
        2-point polyline has no point "lookahead metres ahead" to pursue -- the
        tracker would have to aim at an endpoint tens of metres away, or at the
        path's own origin behind the robot."""
        p = np.asarray(path, dtype=float)
        if len(p) < 2:
            return p
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        if total < 1e-9:
            return p[:1]
        n = max(2, int(np.ceil(total / self.track_ds)) + 1)
        s = np.linspace(0.0, total, n)
        return np.column_stack([np.interp(s, cum, p[:, 0]),
                                np.interp(s, cum, p[:, 1])])

    def _track(self, state: np.ndarray, path: np.ndarray) -> np.ndarray:
        """Pure-pursuit tracking of `path` from `state`=[x,y,theta] -> (H, 2)
        controls [v, omega], rolled with the SAME unicycle update as the env
        (old-theta for the x/y step, then theta update).

        The lookahead point is taken STRICTLY AHEAD of the closest point, by arc
        length along the resampled path. Selecting "the first point at least
        `lookahead` away" is wrong: when the closest waypoint is already farther
        than the lookahead (always true for a sparse shortcut path) it picks that
        waypoint even if it lies BEHIND the robot, so the mean drives backwards
        toward the path origin and spirals at saturated omega.

        A monotonic index guard stops the target from snapping back to an earlier
        part of a path that passes near itself, and the speed is tapered into the
        path end so a mean cannot overshoot the goal and leave the reachable set.
        """
        x, y, th = float(state[0]), float(state[1]), float(state[2])
        pts = self._resample(path)
        if len(pts) < 2:
            return np.zeros((self.horizon, 2), dtype=float)
        end = pts[-1]
        # uniform spacing -> the lookahead is a fixed index offset
        step_ahead = max(1, int(np.ceil(self.lookahead / self.track_ds)))
        last_ci = 0
        u = np.zeros((self.horizon, 2), dtype=float)
        for t in range(self.horizon):
            d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
            ci = max(int(np.argmin(d)), last_ci)  # never move backwards along path
            last_ci = ci
            tgt = pts[min(ci + step_ahead, len(pts) - 1)]
            desired = np.arctan2(tgt[1] - y, tgt[0] - x)
            w = np.clip(_wrap(desired - th) / self.dt, self.wmin, self.wmax)
            # decelerate into the path end (no overshoot past the goal)
            d_end = float(np.hypot(end[0] - x, end[1] - y))
            v = float(np.clip(min(self.vmax, d_end / self.dt), self.vmin, self.vmax))
            u[t, 0], u[t, 1] = v, w
            # env dynamics: use OLD theta for the x/y step, then update theta
            x = x + v * np.cos(th) * self.dt
            y = y + v * np.sin(th) * self.dt
            th = _wrap(th + w * self.dt)
        return u
