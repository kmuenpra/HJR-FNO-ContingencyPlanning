#!/usr/bin/env python
"""
P0 baseline capture for the two-stage certification rewrite.
See two_stage_certification_impl_plan.md, phase P0.

WHY THIS EXISTS
---------------
The delta-sublevel-set certification in HJR_FNO/scenario_worker.py is being
replaced (descending delta sweep -> Campi-Garatti two-stage). Every later phase
needs to answer "did I change the answer, and did I change the cost?" against a
fixed reference. This script provides both halves:

  --mode dump   Build HJR_FNO once, reveal a scenario's obstacles, and freeze the
                per-region value tubes to .pkl FIXTURES. Needs torch + the FNO
                checkpoint (conda env `rrtx`). Run ONCE.

  --mode run    Load those fixtures and run the certification worker at a given
                (eps, beta). Needs only numpy/scipy -- no torch, no GPU, no FNO.
                Run as often as you like, including after the rewrite.

Freezing the tubes is the point: from P1 onward nothing has to re-run the FNO,
so a regression check is seconds rather than minutes, and old-vs-new is compared
on byte-identical inputs.

COST METRIC
-----------
"Trajectory-steps" = the total number of (trajectory, integration-step) pairs
actually integrated, summed over the whole certification. It is measured by
wrapping _ReachValueCache.grad_at_indices, which is called exactly once per
integration step with the currently-active batch -- so summing its batch size is
exactly the work done. This needs NO source change, which is what keeps P0
read-only with respect to the algorithm.

That metric is what P2's early exit is meant to cut; wall time alone would
confound it with machine load.

USAGE
-----
    conda run -n rrtx python eval/bench_certification.py --mode dump  --scenario env_C --seed 1
    conda run -n rrtx python eval/bench_certification.py --mode run   --scenario env_C --seed 1 \
        --eps 0.1 --beta 1e-9 --beta 1e-6 --label baseline
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# =====================================================================
# fixtures
# =====================================================================
def fixture_dir(out_dir: Path, tag: str) -> Path:
    return Path(out_dir) / tag


def dump_fixtures(scenario: str, seed: int | None, out_dir: Path,
                  verbose: bool = True) -> Path:
    """Build HJR_FNO, reveal every scenario obstacle at once, freeze the tubes.

    Every obstacle is revealed in a SINGLE update_obs call rather than replaying
    a lidar sweep. That is deliberate: it is the worst case for the certifier
    (every region that can be perturbed is, and each tube is as obstacle-laden as
    it will ever get), it is a pure function of (scenario, seed), and it does not
    depend on a planner's trajectory -- so the fixture is reproducible without
    pinning RRTx's behaviour too.

    scenario_enable=False: we want the TUBES, not a certification. The
    certification is run separately in `run_bench` so it can be re-run at
    different (eps, beta) without another GPU pass.
    """
    import torch                                   # noqa: F401  (HJR_FNO needs it)
    import env as env_module
    from eval.scenarios import get_scenario
    from HJR_FNO.HJR_FNO3d import HJR_FNO

    sc = get_scenario(scenario, seed)
    tag = sc.tag if hasattr(sc, "tag") else f"{scenario}_s{seed}"
    if verbose:
        print(f"[bench] scenario={sc.name} seed={seed} tag={tag} "
              f"regions={len(sc.safe_regions)} obstacles={len(sc.obstacles)}")

    _env = env_module.Env(
        safe_regions=[list(r) for r in sc.safe_regions],
        unknown_obs=[list(o) for o in sc.obstacles],
        x_range=sc.x_range, y_range=sc.x_range,
    )
    hjr = HJR_FNO(env=_env, safe_regions=[list(r) for r in sc.safe_regions],
                  Tf_reach=sc.Tf_reach, scenario_enable=False)

    t0 = time.perf_counter()
    changed = hjr.update_obs([list(o) for o in sc.obstacles])
    if verbose:
        print(f"[bench] update_obs: {len(changed)} regions re-predicted "
              f"in {time.perf_counter() - t0:.2f} s -> {sorted(changed)}")
    if not changed:
        raise RuntimeError(
            f"No region saw an obstacle for {tag}. The fixture would be empty; "
            f"pick a scenario/seed whose obstacles fall inside a safe region's grid.")

    fdir = fixture_dir(out_dir, tag)
    fdir.mkdir(parents=True, exist_ok=True)

    # Shared across regions -- stored once, alongside the per-region tubes.
    shared = dict(
        tag=tag, scenario=sc.name, seed=seed,
        target2d=np.asarray(hjr._scenario_target2d()),
        x_axis=np.asarray(hjr.g.grid_points[0]),
        y_axis=np.asarray(hjr.g.grid_points[1]),
        theta_array=np.asarray(hjr.theta_array),
        grid=hjr.g, car=hjr.scenario_car,
        dt=float(hjr.time_array[1] - hjr.time_array[0]),
        grid_min=np.asarray(hjr.grid_min), grid_max=np.asarray(hjr.grid_max),
        # The sweep knobs the CURRENT worker needs. P5 replaces these; kept here
        # so a baseline run reproduces today's behaviour exactly.
        delta_floor=float(hjr.scenario_delta_floor),
        delta_init=float(hjr.scenario_delta_init),
        delta_step=float(hjr.scenario_delta_step),
        max_tries=int(hjr.scenario_max_tries),
        regions=sorted(int(i) for i in changed),
        obs_digest=getattr(sc, "obs_digest", None),
    )
    with open(fdir / "shared.pkl", "wb") as fh:
        pickle.dump(shared, fh, protocol=4)

    for i in changed:
        V = hjr.HJR_sets[i]
        if hasattr(V, "cpu"):
            V = V.cpu().numpy()
        obs = np.asarray(hjr.obs_SDF[i])
        with open(fdir / f"region_{i:02d}.pkl", "wb") as fh:
            pickle.dump(dict(region=int(i),
                             V_full=np.asarray(V, dtype=np.float32),
                             obs_sdf=obs,
                             safe_region=list(hjr.safe_regions[i]),
                             n_obs=len(hjr.obs_list[i])), fh, protocol=4)
        if verbose:
            print(f"[bench]   region {i:2d}: V{tuple(V.shape)} "
                  f"obs{tuple(obs.shape)} n_obs={len(hjr.obs_list[i])}")

    print(f"[bench] fixtures written to {fdir}")
    return fdir


def load_fixtures(out_dir: Path, tag: str):
    fdir = fixture_dir(out_dir, tag)
    if not (fdir / "shared.pkl").exists():
        raise FileNotFoundError(
            f"No fixtures at {fdir}. Run --mode dump first (needs conda env `rrtx`).")
    with open(fdir / "shared.pkl", "rb") as fh:
        shared = pickle.load(fh)
    regions = []
    for p in sorted(fdir.glob("region_*.pkl")):
        with open(p, "rb") as fh:
            regions.append(pickle.load(fh))
    return shared, regions


# =====================================================================
# instrumentation
# =====================================================================
class StepCounter:
    """Counts integrated (trajectory, step) pairs by wrapping grad_at_indices.

    grad_at_indices is called exactly once per Euler step of rollout_cost with
    the batch of trajectories still active at that step, so the sum of its batch
    sizes IS the integration work. Patching the class (not an instance) catches
    the caches the worker builds internally.

    Also counts rollout_cost CALLS and the trajectories handed to them, which
    together give the average steps-per-trajectory -- the number P2's early exit
    should move.
    """

    def __init__(self, worker_mod):
        self.mod = worker_mod
        self.steps = 0
        self.grad_calls = 0
        self.rollout_calls = 0
        self.trajectories = 0
        self._orig_grad = None
        self._orig_rollout = None

    def __enter__(self):
        cache_cls = self.mod._ReachValueCache
        self._orig_grad = cache_cls.grad_at_indices
        self._orig_rollout = self.mod.rollout_cost
        outer = self

        def grad_at_indices(self, s, k_arr):
            outer.steps += int(np.shape(s)[0])
            outer.grad_calls += 1
            return outer._orig_grad(self, s, k_arr)

        def rollout_cost(cache, s0, *a, **kw):
            outer.rollout_calls += 1
            outer.trajectories += int(np.shape(s0)[0])
            return outer._orig_rollout(cache, s0, *a, **kw)

        cache_cls.grad_at_indices = grad_at_indices
        self.mod.rollout_cost = rollout_cost
        return self

    def __exit__(self, *exc):
        self.mod._ReachValueCache.grad_at_indices = self._orig_grad
        self.mod.rollout_cost = self._orig_rollout
        return False

    def as_dict(self):
        return dict(traj_steps=self.steps, grad_calls=self.grad_calls,
                    rollout_calls=self.rollout_calls,
                    trajectories=self.trajectories,
                    steps_per_traj=(self.steps / self.trajectories
                                    if self.trajectories else None))


# =====================================================================
# run
# =====================================================================
def run_bench(out_dir: Path, tag: str, eps_list, beta_list, label: str,
              seed: int = 0, verbose: bool = False) -> dict:
    """Certify every fixture region at each (eps, beta); record delta/report/cost."""
    import HJR_FNO.scenario_worker as sw

    shared, regions = load_fixtures(out_dir, tag)
    print(f"[bench] {tag}: {len(regions)} regions, "
          f"grid {shared['x_axis'].size}x{shared['y_axis'].size}"
          f"x{shared['theta_array'].size}, dt={shared['dt']}")

    results = []
    for eps in eps_list:
        for beta in beta_list:
            N_pred = sw._scenario_required_N(eps, beta)
            kmax = sw._scenario_max_outliers(N_pred, eps, beta)
            print(f"\n[bench] === eps={eps} beta={beta:.0e} -> N={N_pred} k_max={kmax} ===")
            for reg in regions:
                cfg = dict(
                    eps=eps, beta=beta, M=30,
                    max_tries=shared['max_tries'],
                    delta_floor=shared['delta_floor'],
                    delta_init=shared['delta_init'],
                    delta_step=shared['delta_step'],
                    step_frac=0.8, seed=seed, dt=shared['dt'],
                    grid_min=shared['grid_min'], grid_max=shared['grid_max'],
                    delta_warm=None,        # COLD: no warm start, comparable across phases
                )
                with StepCounter(sw) as ctr:
                    t0 = time.perf_counter()
                    delta, rep = sw.scenario_delta_hat_worker(
                        reg['V_full'], reg['obs_sdf'], shared['target2d'],
                        shared['x_axis'], shared['y_axis'], shared['theta_array'],
                        shared['grid'], shared['car'], cfg, verbose=verbose)
                    wall = time.perf_counter() - t0
                cost = ctr.as_dict()
                row = dict(eps=eps, beta=beta, region=reg['region'],
                           n_obs=reg['n_obs'], delta=float(delta),
                           wall_s=round(wall, 3), **cost,
                           report=_jsonable(rep))
                results.append(row)
                print(f"[bench]   region {reg['region']:2d}: "
                      f"delta={delta:+.4g}  {wall:7.2f}s  "
                      f"levels={rep.get('levels_evaluated')}  "
                      f"k={rep.get('k')}/{rep.get('N')}  "
                      f"cert={rep.get('certified')}  "
                      f"traj_steps={cost['traj_steps']:,}  "
                      f"steps/traj={cost['steps_per_traj']:.1f}")

    payload = dict(
        label=label, tag=tag, when=datetime.now().isoformat(timespec="seconds"),
        git=_git_rev(), seed=seed,
        method="sweep(legacy)" if not hasattr(sw, "_stage1_scan") else "two-stage",
        results=results,
        totals=_totals(results),
    )
    out = Path(out_dir) / f"{tag}__{label}.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[bench] -> {out}")
    _print_totals(payload)
    return payload


def _totals(results):
    by = {}
    for r in results:
        k = f"eps{r['eps']}_beta{r['beta']:.0e}"
        t = by.setdefault(k, dict(regions=0, wall_s=0.0, traj_steps=0,
                                  rollout_calls=0, trajectories=0, certified=0))
        t['regions'] += 1
        t['wall_s'] += r['wall_s']
        t['traj_steps'] += r['traj_steps']
        t['rollout_calls'] += r['rollout_calls']
        t['trajectories'] += r['trajectories']
        t['certified'] += int(bool(r['report'].get('certified')))
    for t in by.values():
        t['wall_s'] = round(t['wall_s'], 2)
    return by


def _print_totals(payload):
    print(f"\n[bench] TOTALS ({payload['label']}, method={payload['method']})")
    print(f"{'config':>22} {'reg':>4} {'cert':>5} {'wall_s':>9} "
          f"{'traj_steps':>13} {'trajectories':>13}")
    for k, t in payload['totals'].items():
        print(f"{k:>22} {t['regions']:>4} {t['certified']:>5} {t['wall_s']:>9.2f} "
              f"{t['traj_steps']:>13,} {t['trajectories']:>13,}")


def _jsonable(d):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, float) and not math.isfinite(v):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def _git_rev():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_REPO_ROOT, text=True).strip()
    except Exception:
        return None


# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("dump", "run", "both"), default="both")
    ap.add_argument("--scenario", default="env_C")
    ap.add_argument("--seed", type=int, default=1,
                    help="obstacle-map seed (picks the MAP, see eval/scenarios.py)")
    ap.add_argument("--out-dir", default=str(_REPO_ROOT / "bench" / "certification"))
    ap.add_argument("--tag", default=None, help="override the fixture tag")
    ap.add_argument("--eps", type=float, action="append", default=None)
    ap.add_argument("--beta", type=float, action="append", default=None)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--cert-seed", type=int, default=0,
                    help="rng seed INSIDE the certification (cfg['seed'])")
    ap.add_argument("--verbose", action="store_true",
                    help="per-delta-level trace from the worker")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    tag = a.tag or (f"{a.scenario}_s{a.seed}" if a.seed is not None else a.scenario)

    if a.mode in ("dump", "both"):
        fdir = dump_fixtures(a.scenario, a.seed, out_dir)
        tag = fdir.name

    if a.mode in ("run", "both"):
        run_bench(out_dir, tag,
                  eps_list=a.eps or [0.1],
                  beta_list=a.beta or [1e-9, 1e-6],
                  label=a.label, seed=a.cert_seed, verbose=a.verbose)


if __name__ == "__main__":
    main()
