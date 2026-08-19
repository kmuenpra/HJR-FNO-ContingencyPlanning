"""Contingency fallback for the MPPI planner.

When the topological guidance finds NO certified route to the goal (``topo=0``),
the robot would otherwise sit still burning a full topo replan per control step.
Instead we do what ``rrtx_FNO3d.py`` does when a tree has no path to the goal:
hand control to ``HJR_FNO.contingency_policy()``, which drives the robot into the
nearest *certified* safe region using the HJ optimal control, and idle there until
a replan finally produces a route.

Design notes
------------
* ``contingency_policy`` is a BLOCKING, whole-trajectory maneuver: it returns the
  entire rollout to the safe region in one call (and senses / updates the
  reachable sets along the way). RRTX consumes it by teleporting the robot to
  ``trajectory[-1]``. Here we cache it and play it back ONE STATE PER CONTROL
  STEP, so the render cadence, the video and the diagnostics stay per-step.
* Playback assigns ``env._robot_state`` directly rather than re-tracking the path
  through ``env.step``. That is deliberate: the trajectory is the certified HJ
  maneuver, and re-tracking it with a pure-pursuit approximation would introduce
  tracking error the certificate does not cover. The cost is that env dynamics
  are bypassed during the maneuver (as they are in RRTX).
* Obstacles sensed inside ``contingency_policy`` update the ORACLE's reachable
  sets internally, but not the env's occupancy grid, and the policy rebinds
  ``utils.obs_circle`` to a private copy -- which silently breaks the
  shared-list contract set up by ``Navigation2DEnv.attach_reachability``. We
  therefore re-sync both explicitly on return (see ``_absorb_obstacles``).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


class ContingencyManager:
    """Drives the robot to the nearest certified safe region and idles there."""

    def __init__(self, env, hjr_fno, verbose: bool = True) -> None:
        self.env = env
        self.hjr = hjr_fno
        self.verbose = verbose

        self._traj: Optional[np.ndarray] = None   # (N, 3) cached rollout
        self._i = 0                               # playback cursor
        self._active = False
        self._idling = False
        # NOTE: for evaluating per-step solve time
        # the oracle's per-pose timings for the cached maneuver (its last_rollout),
        # replayed one per control step alongside the poses
        self._rollout: List[dict] = []
        self.last_step_own_s = 0.0
        # diagnostics
        self.num_triggers = 0
        self.num_failures = 0
        self.steps_driving = 0
        self.steps_idling = 0
        self.last_region: Optional[int] = None

        # Detached figure: contingency_policy calls ax.clear() UNCONDITIONALLY
        # (before its own showplot guard), so handing it the live simulation axes
        # would wipe every frame. A pyplot-free Figure is never shown.
        self._fig = None
        self._ax = None

    # --- Plotting-object surface expected by contingency_policy ---------
    # It reads plotting.obs_circle / .unknown_obs_circle (and copies them) and
    # calls plotting.update_obs(...). Proxy straight through to the env's lists so
    # the policy sees the live world.
    @property
    def obs_circle(self):
        return self.env.obs_circle

    @property
    def unknown_obs_circle(self):
        return self.env.unknown_obs_circle

    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        """True while driving to, or idling at, a safe region."""
        return self._active

    @property
    def idling(self) -> bool:
        """True once the maneuver is finished and we are holding position."""
        return self._idling

    @property
    def rollout_own_total_s(self) -> float:
        """NOTE: for evaluating per-step solve time.

        Per-pose solve time that will be re-charged on LATER control steps, i.e.
        every pose but the first. The driver subtracts this from the block that
        ran the solve: those times were measured INSIDE that block, so charging
        them again as each pose is replayed would count them twice. The first
        pose is excluded because it is replayed in that same block.
        """
        return float(sum(r.get("t_step_s", 0.0) for r in self._rollout[1:]))

    def status(self) -> str:
        if not self._active:
            return ""
        if self._idling:
            return f"CONTINGENCY idle@region{self.last_region}"
        left = 0 if self._traj is None else len(self._traj) - self._i
        return f"CONTINGENCY drive->region{self.last_region} ({left} left)"

    # ------------------------------------------------------------------
    def _ensure_axes(self):
        if self._ax is None:
            from matplotlib.figure import Figure

            self._fig = Figure(figsize=(4, 4))
            self._ax = self._fig.add_subplot()
        return self._fig, self._ax

    def start(self, state) -> bool:
        """Compute the contingency maneuver from ``state``. Returns True if a
        certified safe region was found (so the manager is now active)."""
        if self._active:
            return True
        s = state.detach().cpu().numpy() if torch.is_tensor(state) else np.asarray(state)
        robot_state = [float(s[0]), float(s[1]), float(s[2])]

        fig, ax = self._ensure_axes()
        try:
            # `self` stands in for RRTX's Plotting object: the policy only needs
            # obs_circle / unknown_obs_circle / update_obs from it.
            detected, traj, _code, success, *_ = self.hjr.contingency_policy(
                robot_state, self, fig, ax, showplot=False
            )
        except Exception as e:  # noqa: BLE001 - a failed contingency must not kill the run
            if self.verbose:
                print(f"[contingency] policy raised ({e}); holding position")
            self.num_failures += 1
            return False

        traj = np.atleast_2d(np.asarray(traj, dtype=float))
        self._absorb_obstacles(detected)

        if not success or len(traj) < 2:
            # Robot is not inside ANY region's certified set -> no maneuver exists.
            if self.verbose:
                print("[contingency] no certified safe region reachable; holding position")
            self.num_failures += 1
            return False

        self._traj = traj
        self._i = 1                     # traj[0] is the current state
        # NOTE: for evaluating per-step solve time
        # entry j of last_rollout produced traj[j+1]
        self._rollout = list(getattr(self.hjr, "last_rollout", None) or [])
        self._active = True
        self._idling = False
        self.num_triggers += 1
        # find_feasible_closest_region is vectorized (returns an array even for a
        # single pose), so reduce it to a scalar index for the label.
        reg = np.asarray(
            self.hjr.find_feasible_closest_region(robot_pose=np.array(robot_state[:2]))
        ).reshape(-1)
        self.last_region = int(reg[0]) if reg.size else None
        if self.verbose:
            print(
                f"[contingency] driving to region {self.last_region} "
                f"over {len(traj) - 1} states "
                f"({float(np.linalg.norm(traj[-1, :2] - traj[0, :2])):.2f} m)"
            )
        return True

    # ------------------------------------------------------------------
    def advance(self):
        """Execute one control step of the contingency behaviour.

        While the cached maneuver has states left, teleport the robot onto the
        next one. Once exhausted, hold position (zero control) so the lidar still
        sweeps and newly revealed obstacles can unblock the replan.

        Returns:
            (state, is_goal_reached) -- same contract as ``env.step``.
        """
        if self._traj is not None and self._i < len(self._traj):
            # NOTE: for evaluating per-step solve time
            # This pose's OWN solve cost, so the replay reproduces the per-step
            # distribution instead of collapsing it onto the step that solved.
            # The first pose is charged as part of the block that ran the solve
            # (see rollout_own_total_s), so it reports 0 here.
            self.last_step_own_s = (
                0.0 if self._i <= 1 or self._i - 1 >= len(self._rollout)
                else float(self._rollout[self._i - 1].get("t_step_s", 0.0))
            )
            nxt = self._traj[self._i]
            self._i += 1
            self.steps_driving += 1
            self.env._robot_state = torch.tensor(
                nxt, device=self.env._device, dtype=self.env._dtype
            )
            # run the env's own sensing at the new pose so the occupancy grid and
            # the oracle both learn about anything now in range
            return self.env.step(torch.zeros(2, device=self.env._device))

        # maneuver finished -> idle, but keep sensing
        self.last_step_own_s = 0.0      # NOTE: for evaluating per-step solve time
        self._idling = True
        self.steps_idling += 1
        return self.env.step(torch.zeros(2, device=self.env._device))

    def stop(self) -> None:
        """Leave contingency (a route to the goal exists again)."""
        if self._active and self.verbose:
            print(
                f"[contingency] released after {self.steps_driving} driving / "
                f"{self.steps_idling} idling step(s)"
            )
        self._active = False
        self._idling = False
        self._traj = None
        self._i = 0

    # ------------------------------------------------------------------
    def _absorb_obstacles(self, detected: List) -> None:
        """Fold obstacles sensed during the maneuver into the ENV, and repair the
        shared-list bindings the policy clobbered.

        ``contingency_policy`` updates the oracle's reachable sets itself, but it
        (a) never touches the env's occupancy grid, so the MPPI cost would stay
        blind to the new obstacles, and (b) rebinds ``utils.obs_circle`` /
        ``unknown_obs_circle`` to private copies, breaking the shared identity
        with ``env._known_obs`` / ``env._unknown_obs``.
        """
        env, utils = self.env, getattr(self.hjr, "utils", None)

        for obs in detected or []:
            ox, oy, r = float(obs[0]), float(obs[1]), float(obs[2])
            if (ox, oy, r) not in env._known_obs:
                env._known_obs.append((ox, oy, r))
                env._obstacle_map.add_circle_obstacle(np.array([ox, oy]), r)
            # keep the unknown list consistent (the policy's lidar consumed it
            # from a copy, so the env's own list may still hold the entry)
            env._unknown_obs[:] = [
                o for o in env._unknown_obs
                if not (abs(o[0] - ox) < 1e-9 and abs(o[1] - oy) < 1e-9)
            ]
        if detected:
            env._obstacle_map.convert_to_torch()
            env._obs_revealed = True     # force the guidance to replan/re-check

        # re-point the oracle's helper at the env's lists (see attach_reachability)
        if utils is not None:
            utils.obs_circle = env.obs_circle
            utils.unknown_obs_circle = env.unknown_obs_circle
            utils.sensing_radius = env.lidar_radius

    # ------------------------------------------------------------------
    def update_obs(self, obs_circle, *args) -> None:
        """Stub so this manager can be passed as ``contingency_policy``'s
        ``plotting`` argument (it calls ``plotting.update_obs(...)`` and reads
        ``obs_circle`` / ``unknown_obs_circle``). The real bookkeeping happens in
        ``_absorb_obstacles`` once the maneuver returns."""
        return None

    # Only reached when showplot=True (we pass False), but stubbed so a caller
    # that flips the flag for debugging does not crash on the detached axes.
    def plot_env(self, ax) -> None:
        return None

    def plot_robot(self, ax, xy, radius) -> None:
        return None
