"""Exact-HJ (optimized_dp) reachability oracle for the MPPI framework.

A drop-in alternative to ``HJR_FNO`` for the reachable-set half of the planner:
same questions asked, different machinery answering them. Selected with
``navigation2d.py --reach_backend odp``.

WHAT IS DIFFERENT FROM THE FNO ORACLE
-------------------------------------
* The sets are the EXACT numerical HJI solve, not a surrogate. No model error --
  so no scenario optimization, no ``delta_hat``, no ``N=435`` certification
  rollouts. Those lines will not appear in an ODP run; they are FNO-only.

* The safety margin is ANALYTIC (eval/lipschitz_delta.py):

      delta = L_V * eps_ZOH + L_V * dx * sqrt(2) / 2  ~= 0.411 m

  computed in microseconds from the tube's own gradient, where the FNO's
  equivalent needs hundreds of sampled rollouts per region. ``safe_margin[i] =
  -delta_i``, matching the FNO's sign convention, so anything thresholding against
  ``safe_margin`` works unchanged.

* Every reveal RE-SOLVES. There is no cache: the lidar exposes new obstacles, the
  affected regions' tubes are recomputed from scratch, and the wall-clock cost of
  that (~0.8 s of PDE solve per region, plus ~0.3 s of graph rebuild that
  HJSolver does unconditionally) is real and is reported. That cost is the point
  -- it is what the planner comparison is measuring against the FNO's
  millisecond inference.

* Solving happens in SUBPROCESSES under the `odp` interpreter, because MPPI needs
  torch (env `rrtx`) and HJSolver needs heterocl (env `odp`) and no single env has
  both. See mppi_src/odp_solve_worker.py.

  The workers are PERSISTENT: started once in __init__, reused for every reveal.
  They used to be spawned per reveal, which charged ~0.8 s of `import heterocl`
  plus ~0.77 s of first-solve lazy init to EVERY update -- while the FNO arm's
  own worker pool was already persistent (HJR_FNO3d._get_scenario_pool caches and
  reuses it). Measuring one arm cold and the other warm inflated the FNO's
  apparent advantage on `t_reach`; it is the same class of asymmetry plan doc 7.7
  fixed for parallelism, and it favoured the contribution under test. Startup now
  lands in __init__, where the FNO's model load and CUDA init already sit.

* ``contingency_policy`` is implemented here too, as an HJB rollout driven by
  optimized_dp's own ``DubinsCar2.optCtrl_inPython`` / ``optDstb_inPython`` /
  ``dynamics_inPython`` on interpolated spatial derivatives. ContingencyManager
  drives it unchanged.

Everything the MPPI pipeline uses -- the cost term, RBR resampling, topological
guidance, the contingency fallback and the overlay -- is supported.

FEASIBILITY SEMANTICS
---------------------
Mirrors ``HJR_FNO.points_feasible`` with ``feasibility_source="feasible_region"``:
the theta-marginalised grown slice of the CLOSEST safe region, tested against
``safe_margin``. ``thetas`` is accepted and ignored, exactly as in the FNO path
under that source. Out-of-domain points are infeasible.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.config import CFG  # noqa: E402
from eval.lipschitz_delta import delta_for_field, zoh_epsilon_mppi  # noqa: E402

# The genuine optimized_dp dynamics class, via the shim that stubs heterocl (only
# its HeteroCL solver methods need it; optCtrl_inPython / optDstb_inPython /
# dynamics_inPython are pure numpy). Using the SAME class the solver was run with
# means the contingency controller cannot disagree with the game the tube encodes.
from odp_shim import DubinsCar2  # noqa: E402

# Interpreter that owns heterocl. Overridable so this is not welded to one box.
ODP_PYTHON = os.environ.get(
    "ODP_PYTHON", "/home/kmuenpra/anaconda3/envs/odp/bin/python"
)

LOCAL_HALF = 10.0      # region-local domain is [-10, 10]^2, as in eval/odp_reachset

# how long to wait for a worker to finish importing heterocl + warm up. Measured
# cold start is ~1.6 s; this only bounds a hang.
WORKER_READY_TIMEOUT_S = 120.0
WORKER_LOG_DIR = _REPO / "eval" / "results" / "logs"


class ODPReachOracle:
    """Exact HJ reachable sets over a set of safe regions, updated online.

    Args:
        env: the Navigation2DEnv (only its obstacle bookkeeping is read).
        safe_regions: iterable of (x, y, radius) in world coordinates.
        Tf_reach: reach horizon [s].
        inflate: safety factor on the Lipschitz margin (1.0 = the bound as derived).
        eps_zoh: unmonitored drift per control step [m]; defaults to
            ``zoh_epsilon_mppi()`` = dt*(v_max + |d_xy|) = 0.1141 m.
        verbose: print per-solve timing and the margin breakdown.
    """

    def __init__(self, env=None, safe_regions: Sequence = (), Tf_reach: float = 8.0,
                 inflate: float = 1.0, eps_zoh: Optional[float] = None,
                 verbose: bool = True) -> None:
        self.env = env
        self.safe_regions = np.asarray([tuple(map(float, r)) for r in safe_regions],
                                       dtype=float)
        assert self.safe_regions.ndim == 2 and self.safe_regions.shape[1] == 3, \
            "safe_regions must be (M, 3) of (x, y, radius)"
        self.num_safe_regions = int(self.safe_regions.shape[0])
        self.Tf_reach = float(Tf_reach)
        self.inflate = float(inflate)
        self.eps_zoh = float(eps_zoh) if eps_zoh is not None else zoh_epsilon_mppi()
        self.verbose = bool(verbose)

        # world-frame obstacles revealed so far; grows monotonically
        self.known_obs: List[tuple] = []

        # per-region state
        M = self.num_safe_regions
        self.tube: List[Optional[np.ndarray]] = [None] * M    # (Nx, Ny, Nth, Nt)
        self.grown: List[Optional[np.ndarray]] = [None] * M   # (Nx, Ny, Ntheta)
        self.feasible_region: List[Optional[np.ndarray]] = [None] * M  # (Nx, Ny)
        self._interps: List[Optional[RegularGridInterpolator]] = [None] * M
        self._interp3: List[Optional[RegularGridInterpolator]] = [None] * M
        self._grad: dict = {}          # (region, time_idx) -> 3 interpolators
        self.safe_margin = np.zeros(M, dtype=float)   # = -delta_i (FNO convention)
        self.x_axis = self.y_axis = self.theta_axis = self.time_array = None

        # Same dynamics/game the tubes were solved under (eval/odp_reachset.py
        # reads these from eval/config.yaml too), so contingency_policy's control
        # is the one the value function assumes.
        self.car = DubinsCar2(uMin=list(CFG.u_min), uMax=list(CFG.u_max),
                              dMax=list(CFG.d_max), uMode="min", dMode="max")
        # matches HJR_FNO's ranking of candidate regions in contingency_policy
        self.contingency_heading_weight = 1.0
        self.last_rollout: List[dict] = []

        # Concurrent region solves. Same switch the FNO backend's scenario
        # certification obeys, so the two reachability updates are parallelized
        # or not TOGETHER -- otherwise 2-vs-3 measures the implementation, not
        # the backend. 6 workers on a 14-core machine; each heterocl process is
        # single-threaded and CPU-only, so they neither contend with each other
        # nor with the FNO's GPU work.
        self.solve_parallel = CFG.scenario_parallel
        self.solve_workers = 6

        # online cost accounting -- the number this backend exists to expose
        self.solve_calls = 0
        self.solve_regions_total = 0
        self.solve_time_total = 0.0
        self.certify_time_total = 0.0
        self.worker_starts = 0          # must stay == solve_workers for the whole
        self.pool_startup_s = 0.0       # episode, or the timing is contaminated

        # persistent worker pool
        self._last_solve_s: dict = {}   # region -> HJSolver's own clock, last update
        self._last_n_workers = 0
        self._procs: List[subprocess.Popen] = []
        self._worker_logs: List[Path] = []
        self._pool_n = 0

        if self.verbose:
            print(f"[ODP] exact-HJ backend: {M} safe regions, Tf={self.Tf_reach} s, "
                  f"eps_ZOH={self.eps_zoh:.4f} m, inflate={self.inflate:g}")
            print(f"[ODP] solver subprocess: {ODP_PYTHON}")
        self._start_pool()
        atexit.register(self.shutdown_pool)
        # t = 0: nothing has been revealed yet, so every tube is obstacle-free.
        self._solve(range(M), reason="init")

    # ------------------------------------------------------------------
    # persistent worker pool
    # ------------------------------------------------------------------
    def _start_pool(self) -> None:
        """Spawn the solver processes and block until each has imported heterocl
        and warmed up. Everything expensive-but-once happens here, OUTSIDE the
        measured loop -- the same place the FNO arm pays model load and CUDA init.
        """
        n = max(1, int(self.solve_workers)) if self.solve_parallel else 1
        WORKER_LOG_DIR.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for j in range(n):
            log = WORKER_LOG_DIR / f"odp_worker_{os.getpid()}_{j}.log"
            # stderr goes to a FILE, not a pipe: a chatty child (heterocl warns
            # freely) would otherwise fill the pipe buffer while the parent waits
            # on stdout, and both would deadlock.
            proc = subprocess.Popen(
                [ODP_PYTHON, "-m", "mppi_src.odp_solve_worker", "--serve"],
                cwd=str(_REPO), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=log.open("w"), text=True, bufsize=1)
            self._procs.append(proc)
            self._worker_logs.append(log)
        for j, proc in enumerate(self._procs):
            self._await_ready(j, proc)
        self._pool_n = n
        self.worker_starts = n
        self.pool_startup_s = time.time() - t0
        if self.verbose:
            print(f"[ODP] worker pool: {n} persistent process(es) ready in "
                  f"{self.pool_startup_s:.2f} s (import + warmup, NOT charged to "
                  f"t_reach); logs in {WORKER_LOG_DIR}")

    def _await_ready(self, j: int, proc: subprocess.Popen) -> None:
        """Read the child's stdout until READY, ignoring heterocl chatter."""
        from mppi_src.odp_solve_worker import READY_TOKEN

        deadline = time.time() + WORKER_READY_TIMEOUT_S
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:                       # EOF -> the child died on import
                raise RuntimeError(
                    f"ODP worker {j} exited during startup (rc={proc.poll()}).\n"
                    f"interpreter: {ODP_PYTHON}\n{self._log_tail(j)}")
            if line.startswith(READY_TOKEN):
                return
        raise RuntimeError(f"ODP worker {j} did not become ready within "
                           f"{WORKER_READY_TIMEOUT_S:.0f} s.\n{self._log_tail(j)}")

    def _log_tail(self, j: int, n: int = 2000) -> str:
        try:
            return f"--- worker {j} stderr ---\n{self._worker_logs[j].read_text()[-n:]}"
        except Exception:  # noqa: BLE001
            return f"(no log for worker {j})"

    def _check_alive(self) -> None:
        """A dead worker means the episode's timing is no longer what it claims.
        Fail loudly rather than restarting: a silent restart would re-introduce
        exactly the startup cost this pool exists to remove, and it would never
        show up in the numbers.
        """
        for j, proc in enumerate(self._procs):
            if proc.poll() is not None:
                raise RuntimeError(
                    f"ODP worker {j} died (rc={proc.returncode}) mid-episode. "
                    f"Not restarting -- a restart would silently recharge startup "
                    f"cost to t_reach.\n{self._log_tail(j)}")

    def shutdown_pool(self) -> None:
        """Tear the pool down; safe to call twice (registered with atexit)."""
        for proc in getattr(self, "_procs", []):
            try:
                if proc.poll() is None:
                    try:
                        proc.stdin.write("__ODP_STOP__\n")
                        proc.stdin.flush()
                    except Exception:  # noqa: BLE001 - already gone
                        pass
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except Exception:  # noqa: BLE001 - never raise from teardown
                pass
        self._procs = []
        self._pool_n = 0

    def __del__(self):
        try:
            self.shutdown_pool()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # solving
    # ------------------------------------------------------------------
    def _obs_local(self, m: int) -> List[List[float]]:
        """Known obstacles in region m's local frame, clipped to its domain."""
        cx, cy = self.safe_regions[m, :2]
        out = []
        for ox, oy, r in self.known_obs:
            lx, ly = ox - cx, oy - cy
            if (abs(lx) - r <= LOCAL_HALF) and (abs(ly) - r <= LOCAL_HALF):
                out.append([float(lx), float(ly), float(r)])
        return sorted(out)

    def _spec_for(self, regions: Sequence[int], out: Path) -> dict:
        return {
            "Tf": self.Tf_reach,
            "out": str(out),
            "regions": [
                {"index": int(m), "center": self.safe_regions[m, :2].tolist(),
                 "r_target": float(self.safe_regions[m, 2]),
                 "obs_local": self._obs_local(int(m))}
                for m in regions
            ],
        }

    def _predict_tubes(self, regions: Sequence[int]) -> None:
        """PREDICT phase: produce the value tubes. Always solves; nothing cached.

        Regions are split across ``solve_workers`` CONCURRENT persistent
        subprocesses. HJSolver is built by ``hcl.build`` with no target, i.e. the
        LLVM CPU backend, and is effectively single-threaded -- measured 3.37x on
        4 workers (5.90 s -> 1.75 s for 4 env_D regions), so the split scales at
        ~84% efficiency and never contends with the FNO's GPU work.

        This exists for COMPARABILITY, not just speed: the FNO backend already
        certifies its regions in a 6-worker pool, so leaving the exact solve
        serial made 2-vs-3 measure an implementation asymmetry on top of the
        backend difference it is meant to isolate.

        This method is the ``predict`` hook ReachTimer wraps by name, so it must
        contain the solve and NOTHING ELSE -- ``_refresh`` (which certifies) runs
        after it in ``_solve``, never inside it, or the phases would nest and
        double-count.
        """
        from mppi_src.odp_solve_worker import RESULT_PREFIX

        self._check_alive()
        n_jobs = min(len(regions), self._pool_n)
        # contiguous chunks; region solve cost is roughly uniform, so a simple
        # split balances about as well as anything cleverer
        chunks = [regions[i::n_jobs] for i in range(n_jobs)]
        chunks = [c for c in chunks if c]

        with tempfile.TemporaryDirectory() as td:
            # dispatch to every worker first, so the solves overlap
            for j, chunk in enumerate(chunks):
                spec = self._spec_for(chunk, Path(td) / f"tubes_{j}.npz")
                self._procs[j].stdin.write(json.dumps(spec) + "\n")
                self._procs[j].stdin.flush()

            for j, chunk in enumerate(chunks):
                reply = self._read_result(j, RESULT_PREFIX)
                if not reply.get("ok"):
                    raise RuntimeError(
                        f"ODP worker {j} failed on regions {chunk}: "
                        f"{reply.get('error')}\n{self._log_tail(j)}")
                self._last_solve_s.update(
                    {int(k): float(v) for k, v in reply.get("t_solve_s", {}).items()})

                d = np.load(reply["out"])
                self.x_axis = np.asarray(d["x_axis"], float)
                self.y_axis = np.asarray(d["y_axis"], float)
                self.theta_axis = np.asarray(d["theta_axis"], float)
                self.time_array = np.asarray(d["time_array"], float)
                for m in chunk:
                    self.tube[m] = np.asarray(d[f"V_{m}"], dtype=np.float32)
                    self.grown[m] = self.tube[m][..., 0]   # index 0 = fully grown
                d.close()
        self._last_n_workers = len(chunks)

    def _read_result(self, j: int, prefix: str) -> dict:
        """Read worker j's stdout until the sentinel. Everything before it is
        library chatter (heterocl prints during graph_3D) and is discarded."""
        proc = self._procs[j]
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"ODP worker {j} closed stdout (rc={proc.poll()}); the solve "
                    f"did not complete.\n{self._log_tail(j)}")
            if line.startswith(prefix):
                return json.loads(line[len(prefix):])

    def _solve(self, regions: Sequence[int], reason: str = "") -> None:
        """One reachable-set update: predict the tubes, then certify + refresh.

        The two phases are separate methods so ReachTimer can time them
        independently (plan doc 3.1b). Their sum is below the enclosing
        ``update_obs``; the remainder is interpolator rebuilds.
        """
        regions = [int(m) for m in regions]
        if not regions:
            return

        self._last_solve_s = {}
        self._last_n_workers = 0
        t0 = time.time()
        self._predict_tubes(regions)
        t_predict = time.time() - t0

        t1 = time.time()
        for m in regions:
            self._refresh(m)
        t_refresh = time.time() - t1
        wall = time.time() - t0

        self.solve_calls += 1
        self.solve_regions_total += len(regions)
        self.solve_time_total += wall
        if self.verbose:
            pde = sum(self._last_solve_s.values())
            print(f"  [ODP] re-solved {len(regions)} region(s) "
                  f"{'(' + reason + ') ' if reason else ''}in {wall:.2f} s "
                  f"[{self._last_n_workers} worker(s)] "
                  f"| predict {t_predict:.2f} s (HJSolver {pde:.2f} s summed over "
                  f"regions) + refresh {t_refresh:.2f} s "
                  f"| delta {self.delta.min():.3f}..{self.delta.max():.3f} "
                  f"| episode total {self.solve_time_total:.1f} s "
                  f"over {self.solve_calls} solve(s)")

    def _certify_margin(self, vm: np.ndarray) -> float:
        """CERTIFY phase: the safety margin for region m's field.

        The exact-HJ counterpart of the FNO's scenario optimization, and the
        ``certify`` hook ReachTimer wraps by name. It is analytic rather than
        sampled because ODP's value function is sound -- its whole error budget is
        discretisation, which a Lipschitz argument bounds. The FNO's error
        includes model bias, which no such argument bounds, so that arm pays
        hundreds of rollouts here instead of microseconds. Timing the two under
        the same label is the point: it makes the difference visible rather than
        folding it into one `t_reach` number.
        """
        t0 = time.perf_counter()
        info = delta_for_field(vm, self.x_axis, self.y_axis, self.eps_zoh,
                               inflate=self.inflate)
        self.certify_time_total += time.perf_counter() - t0
        return -info["delta"]

    def _refresh(self, m: int) -> None:
        """Rebuild region m's marginal field, interpolators and Lipschitz margin."""
        vm = self.grown[m].max(axis=2)            # max over theta == FNO's source
        self.feasible_region[m] = vm
        self._interps[m] = RegularGridInterpolator(
            (self.x_axis, self.y_axis), vm, bounds_error=False, fill_value=None)
        # grown-slice value in (x, y, theta), for contingency_policy's entry check
        self._interp3[m] = RegularGridInterpolator(
            (self.x_axis, self.y_axis, self.theta_axis),
            np.asarray(self.grown[m], dtype=float),
            bounds_error=False, fill_value=None)
        # this region's cached per-slice gradients describe the OLD tube
        self._grad = {k: v for k, v in self._grad.items() if k[0] != m}
        # delta is a property of THIS tube, so it moves when the tube does
        self.safe_margin[m] = self._certify_margin(vm)

    @property
    def delta(self) -> np.ndarray:
        """Positive margins; ``safe_margin`` is their negation."""
        return -self.safe_margin

    # ------------------------------------------------------------------
    # the interface MPPI/guidance/env use
    # ------------------------------------------------------------------
    def update_obs(self, detected: Sequence) -> None:
        """Absorb newly revealed obstacles and re-solve the regions they touch.

        Only regions whose local domain the obstacle actually intersects are
        re-solved; the rest cannot be influenced by it. That is what keeps an
        online run affordable -- most reveals touch one or two regions.
        """
        fresh = [tuple(map(float, o)) for o in (detected or [])]
        fresh = [o for o in fresh if o not in set(self.known_obs)]
        if not fresh:
            return
        self.known_obs.extend(fresh)

        touched = set()
        for ox, oy, r in fresh:
            for m in range(self.num_safe_regions):
                cx, cy = self.safe_regions[m, :2]
                if abs(ox - cx) - r <= LOCAL_HALF and abs(oy - cy) - r <= LOCAL_HALF:
                    touched.add(m)
        self._solve(sorted(touched), reason=f"{len(fresh)} obs revealed")

    def find_feasible_closest_region(self, robot_pose, t=None, use_distance=True,
                                     returnList=False):
        """Index of the nearest safe region per query point (Euclidean), matching
        HJR_FNO's signature."""
        pts = np.atleast_2d(np.asarray(robot_pose, float))[:, :2]
        d2 = ((pts[:, None, :] - self.safe_regions[None, :, :2]) ** 2).sum(-1)
        return d2.argmin(axis=1)

    def points_feasible(self, points: np.ndarray, thetas: np.ndarray = None,
                        reachable_set_constraint: bool = True) -> np.ndarray:
        """Per-point feasibility: V_closest(p) <= safe_margin, i.e. V < -delta.

        ``thetas`` is accepted and IGNORED (the field is theta-marginalised), which
        is what HJR_FNO also does under feasibility_source="feasible_region".
        """
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        K = pts.shape[0]
        if not reachable_set_constraint:
            return np.ones(K, dtype=bool)

        closest = np.asarray(self.find_feasible_closest_region(pts)).reshape(-1)
        local = pts - self.safe_regions[closest, :2]
        in_bound = np.all(np.abs(local) <= LOCAL_HALF, axis=1)
        feas = in_bound.copy()
        if not in_bound.any():
            return feas

        for m in np.unique(closest):
            sel = (closest == m) & in_bound
            if not sel.any():
                continue
            vals = self._interps[m](local[sel])
            feas[sel] = vals <= self.safe_margin[m]
        return feas

    def is_feasible(self, v: np.ndarray, reachable_set_constraint: bool = True,
                    thetas: np.ndarray = None) -> bool:
        return bool(np.all(self.points_feasible(
            v, thetas=thetas, reachable_set_constraint=reachable_set_constraint)))

    # ------------------------------------------------------------------
    # contingency: HJB-optimal retreat into a safe region
    # ------------------------------------------------------------------
    def _wrap_theta(self, th) -> float:
        """Wrap a heading into the value grid's own theta range, [-pi, pi)."""
        lo = float(self.theta_axis[0])
        return lo + float(np.mod(float(th) - lo, 2.0 * np.pi))

    def _slice_grad(self, m: int, k: int):
        """Spatial-derivative interpolators for region m's time slice k.

        Built on demand and cached until the tube changes (~2 ms per slice on the
        50x50x25 grid). Slice k rather than the grown slice for a concrete reason:
        the grown slice SATURATES at ~-r_target throughout the interior of the
        reachable set, so |grad V| falls to ~1e-3 there and the bang-bang law
        chatters on numerical noise instead of driving anywhere (measured: 13% of
        rollouts arrived). Slice k is "reachable within k steps", whose gradient
        points at the target -- which is why the FNO policy walks slices too.
        """
        key = (m, k)
        if key not in self._grad:
            V3 = np.asarray(self.tube[m][..., k], dtype=float)
            dx = float(self.x_axis[1] - self.x_axis[0])
            dy = float(self.y_axis[1] - self.y_axis[0])
            dth = float(self.theta_axis[1] - self.theta_axis[0])
            # theta is PERIODIC: pad one slice each side before differencing, or
            # np.gradient's one-sided end stencils give a wrong dV/dtheta at +-pi.
            Vp = np.concatenate([V3[:, :, -1:], V3, V3[:, :, :1]], axis=2)
            gx, gy, gth = np.gradient(Vp, dx, dy, dth)
            axes = (self.x_axis, self.y_axis, self.theta_axis)
            self._grad[key] = tuple(
                RegularGridInterpolator(axes, g[:, :, 1:-1], bounds_error=False,
                                        fill_value=None)
                for g in (gx, gy, gth)
            )
        return self._grad[key]

    def _spat_deriv(self, local_state: np.ndarray, m: int, k: int) -> np.ndarray:
        """grad V at a continuous state, by interpolating slice k's derivatives."""
        q = np.array([[local_state[0], local_state[1],
                       self._wrap_theta(local_state[2])]])
        return np.array([float(g(q)[0]) for g in self._slice_grad(m, k)])

    def contingency_policy(self, robot_state, plotting=None, fig=None, ax=None,
                           showplot: bool = False, special_case: bool = False):
        """Retreat maneuver: drive into the nearest certified safe region.

        Mirrors ``HJR_FNO.contingency_policy``'s CONTRACT -- same argument list,
        same ``(detected, trajectory, code, success, ...)`` return -- so
        mppi_src/contingency.py drives it unchanged.

        The maneuver is the textbook HJB rollout, and every piece of it comes from
        optimized_dp's own DubinsCar2 rather than being re-derived here:

            p     = grad V(x)                        (interpolated, per time slice)
            u*    = car.optCtrl_inPython(x, p)       (bang-bang, uMode="min")
            x    += dt * car.dynamics_inPython(x, u*, 0)

        The rollout is DISTURBANCE-FREE (d = 0), matching HJR_FNO's rollout car
        (config ``d_max_rollout``) so the two arms execute the same physics. The
        tube itself is still solved against the full adversary (``d_max``), so the
        maneuver stays valid -- only the simulated execution is nominal.

        One deliberate difference from the FNO version: it does not sense obstacles
        itself. ContingencyManager.advance() calls env.step() at every pose it
        replays, so the lidar sweeps and update_obs() fires through the normal
        path, re-solving affected tubes mid-maneuver. The returned ``detected``
        list is therefore empty by design, not by omission.
        """
        self.last_rollout = []
        x_r, y_r, theta = (float(v) for v in robot_state[:3])

        # Rank the 3 nearest regions by distance plus heading misalignment, as the
        # FNO policy does: turning around costs time the maneuver may not have.
        d2 = ((np.array([x_r, y_r]) - self.safe_regions[:, :2]) ** 2).sum(1)
        ranked = []
        for idx in np.argsort(d2)[:3]:
            cx, cy = self.safe_regions[idx, :2]
            dist = math.hypot(cx - x_r, cy - y_r)
            dth = ((math.atan2(cy - y_r, cx - x_r) - theta) + math.pi) % (
                2 * math.pi) - math.pi
            ranked.append((int(idx), dist + self.contingency_heading_weight * abs(dth),
                           dist))
        ranked.sort(key=lambda z: (z[1], z[2]))

        # Take the first candidate whose CERTIFIED set actually contains the robot.
        chosen = None
        for idx, _, _ in ranked:
            local = np.array([x_r - self.safe_regions[idx, 0],
                              y_r - self.safe_regions[idx, 1], theta])
            if np.any(np.abs(local[:2]) > LOCAL_HALF):
                continue                        # outside this region's local grid
            q = np.array([[local[0], local[1], self._wrap_theta(theta)]])
            if float(self._interp3[idx](q)[0]) <= self.safe_margin[idx]:
                chosen = (idx, local)
                break
        if chosen is None:
            if self.verbose:
                print("[ODP] contingency: robot is in no region's certified set; "
                      "no maneuver exists")
            return [], np.array([[x_r, y_r, theta]]), 999, False, None, None, None

        m, x = chosen
        dt = float(CFG.dt_c)
        r_stop = 0.8 * float(self.safe_regions[m, 2])   # stop WELL inside, as FNO does
        traj = [[x_r, y_r, theta]]

        # Walk the tube's slices from fully grown (index 0) inward, spending `sub`
        # control steps on each. One pass = one control step at dt_c, so a
        # contingency pose is directly comparable with a nominal one. Increasing
        # index tightens the set toward the target, so this descends through
        # nested sets rather than wandering on one saturated field.
        slice_dt = float(self.time_array[1] - self.time_array[0])
        sub = max(1, int(round(slice_dt / dt)))
        reached = False
        for k in range(len(self.time_array)):
            # NOTE: for evaluating per-step solve time
            # Build (or cache-hit) this slice's gradient once. Its cost is shared by
            # the `sub` control steps below, so it is split evenly across them --
            # same convention as HJR_FNO's computeGradients.
            _t0 = time.perf_counter()
            self._slice_grad(m, k)
            t_grad = time.perf_counter() - _t0

            for j in range(sub):
                _t0 = time.perf_counter()   # NOTE: for evaluating per-step solve time
                p = self._spat_deriv(x, m, k)
                u = self.car.optCtrl_inPython(x, p)

                # d = self.car.optDstb_inPython(x, p)
                d = np.zeros(3, dtype=float)
                
                x = x + dt * np.asarray(self.car.dynamics_inPython(x, u, d), float)
                x[2] = (x[2] + math.pi) % (2 * math.pi) - math.pi
                t_step = time.perf_counter() - _t0
                pose = [x[0] + self.safe_regions[m, 0],
                        x[1] + self.safe_regions[m, 1], x[2]]
                traj.append(pose)
                self.last_rollout.append({"pose": tuple(pose), "dt": dt,
                                          "t_step_s": t_step + t_grad / sub,
                                          "t_grad_s": t_grad if j == 0 else 0.0,
                                          "region": m, "time_idx": k})
                if x[0] ** 2 + x[1] ** 2 < r_stop ** 2:
                    reached = True
                    break
            if reached:
                break

        if self.verbose:
            print(f"[ODP] contingency: region {m}, {len(traj) - 1} steps, "
                  f"{math.hypot(traj[-1][0] - x_r, traj[-1][1] - y_r):.2f} m, "
                  f"{'reached' if reached else 'did NOT reach'}")
        return [], np.asarray(traj, dtype=float), 8, True, None, None, None

    # ------------------------------------------------------------------
    # rendering -- each backend draws its OWN set
    # ------------------------------------------------------------------
    def draw_sets(self, ax, theta: float = None) -> None:
        """Fill each region's certified set {V <= safe_margin} in world frame.

        Defined here rather than in navigation2d so the overlay does not have to
        reach into backend internals: swapping FNO for ODP changes which object
        draws, not the caller.
        """
        if self.x_axis is None:
            return
        X, Y = np.meshgrid(self.x_axis, self.y_axis, indexing="ij")
        for m in range(self.num_safe_regions):
            fr = self.feasible_region[m]
            if fr is None or fr.min() > self.safe_margin[m]:
                continue      # empty set at this margin: nothing to draw
            ax.contourf(
                X + self.safe_regions[m, 0], Y + self.safe_regions[m, 1], fr,
                levels=[fr.min(), self.safe_margin[m]], colors="#ADD8E6", alpha=0.4,
            )

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Online reachability cost of the episode -- the headline this backend
        exists to produce."""
        return {
            "backend": "ODP",
            "solve_calls": self.solve_calls,
            "regions_solved": self.solve_regions_total,
            "solve_time_total_s": self.solve_time_total,
            "certify_time_total_s": self.certify_time_total,
            "delta_mean": float(self.delta.mean()),
            "known_obstacles": len(self.known_obs),
            # Pool health. worker_starts MUST equal the pool size for the whole
            # episode: any restart would silently recharge import + warmup to
            # t_reach, which is the cost the persistent pool exists to remove.
            "worker_starts": self.worker_starts,
            "pool_size": self._pool_n,
            "pool_startup_s": self.pool_startup_s,
        }
