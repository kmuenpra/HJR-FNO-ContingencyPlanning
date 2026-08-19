# Reverting to the legacy (delta-sweep) certification

Snapshots taken **2026-08-19**, immediately before the two-stage rewrite began
(see [../two_stage_certification_impl_plan.md](../two_stage_certification_impl_plan.md)).
At that moment **neither file was committed** — `git status` showed both as modified with
277 uncommitted lines in `HJR_FNO3d.py` — so these snapshots are the only copy of the exact
pre-rewrite state. Do not delete them until the rewrite is committed and proven.

| snapshot | sha256 (16) | importable? |
|---|---|---|
| `scenario_worker_legacy.py` | `9908c4b15fac3cb4` | **yes** — kept importable on purpose, for old-vs-new A/B in one process (P7.5) |
| `HJR_FNO3d_legacy.py.bak` | `aa410a49c8f29ddd` | no — `.bak` extension deliberately, a second importable copy of this module would be a trap |

---

## The thing that is easy to get wrong

**The two files are coupled. Reverting one alone leaves a broken worker.**

`scenario_delta_hat_worker` takes a plain `cfg` **dict**, not keyword arguments, so a key
mismatch is not a `TypeError` at the call site — it is a `KeyError` deep inside the worker,
inside a `ProcessPoolExecutor`, where `_certify_safe_margins` catches it as a per-region
failure and prints `region i: failed (...); keeping safe_margin = ...`. **Every region
silently keeps its stale margin and the planner keeps running.** That is the failure mode to
watch for: not a crash, a silent loss of certification.

So: **revert both files together, or neither.**

```bash
cd "$(git rev-parse --show-toplevel)"
cp HJR_FNO/scenario_worker_legacy.py  HJR_FNO/scenario_worker.py
cp HJR_FNO/HJR_FNO3d_legacy.py.bak    HJR_FNO/HJR_FNO3d.py
```

Then verify (should print `435`, the legacy N at the old defaults):

```bash
conda run -n rrtx python -c "
from HJR_FNO.HJR_FNO3d import HJR_FNO, _scenario_required_N
print('N =', _scenario_required_N(0.1, 1e-9))"
```

and run one certification end-to-end against the frozen fixtures:

```bash
conda run -n rrtx python eval/bench_certification.py --mode run --tag env_C_s1 \
    --eps 0.1 --beta 1e-9 --label revert-check
```

Expected, from the P0 baseline (§8 of the plan): 7/7 certified,
δ̂ = −0.7, −0.7, −0.5, −0.5, −0.5, −0.7, −0.7, ~0.96 s, 10,520 trajectories.
**Same δ̂ per region = a good revert.**

---

## What the legacy worker requires of `HJR_FNO3d.py`

If you ever revert only `scenario_worker.py` (e.g. to bisect a bug), these are the seven
coupling points that must be in their legacy form. Line numbers are from the snapshot.

### 1. `cfg` dict schema — `_scenario_cfg()`, line ~857

The legacy worker reads these keys and **will `KeyError` without them**:

```python
dict(eps, beta, M, max_tries, delta_floor, delta_init, delta_step,
     step_frac, seed, dt, grid_min, grid_max)
```

plus `delta_warm`, injected **per region** (not in `_scenario_cfg`) by
`_certify_safe_margins`. It is read with `cfg.get('delta_warm')`, so its absence is
survivable (= cold start) — but `delta_init` / `delta_step` / `delta_floor` are read with
`cfg[...]` and are not.

The two-stage cfg drops `delta_init`, `delta_step`, `M`, `step_frac`, `delta_warm` and adds
`gamma`, `delta_start`, `max_attempts`, `n_scale`, `eps_cap`. **Disjoint enough that a
half-revert fails immediately** — which is the one good thing about this coupling.

### 2. Constructor attributes — lines ~624–645

`_scenario_cfg` reads these off `self`; all must exist:

```
scenario_eps  scenario_beta  scenario_M  scenario_max_tries  scenario_delta_floor
scenario_delta_init  scenario_delta_step  scenario_step_frac  scenario_seed
scenario_verbose  scenario_car  scenario_enable  scenario_parallel  scenario_max_workers
```

Legacy defaults: `scenario_eps=0.1`, **`scenario_beta=1e-9`** (a constructor default — the
two-stage version changes it to `1e-6`), `delta_floor=-1.20`, `delta_init=-0.1`,
`delta_step=0.1`, `max_tries=400`, `seed=0`.

### 3. `self._certified_once` — line ~595, used ~994/1005/1020

A per-region `[False] * num_safe_regions`. Exists **only** to gate the warm start: 0 is a
legal delta level, so `safe_margin[i] == 0` cannot distinguish "never certified" from
"certified at 0". The two-stage version deletes it. If you revert the worker without
restoring this, `_certify_safe_margins` raises `AttributeError` on the first reveal.

### 4. `_certify_safe_margins` warm-start injection — line ~993

```python
cfg_i = dict(cfg)
cfg_i['delta_warm'] = self.safe_margin[i] if self._certified_once[i] else None
```

and `self._certified_once[i] = True` on each success, in **both** the serial and the
parallel branch.

### 5. `rollout_cost` signature — `_scenario_rollout_cost`, line ~873

Legacy is `rollout_cost(cache, s0, car, dt, delta=0.0)` and the wrapper forwards `delta=`.
The two-stage version **removes** the `delta` parameter outright (so `t₀*` cannot
accidentally become δ-dependent — see P2 in the plan). Mixing them is a `TypeError`, which
is the intended loud failure.

### 6. `_format_scenario_report` — line ~898

Reads the legacy report keys: `N, k, kmax, k_partial, success_rate, eps_hat, eps,
certified, levels_evaluated, sample_tries, sample_accept_rate, rollout_chunks`.
The two-stage report drops `levels_evaluated` / `kmax` / `k_partial` and adds
`verdict, attempt, m_star, rho_hat, phi, k_aim, k_accept, eps_original, ...`.
A mismatch here is *not* fatal — it reads via `.get()` — so it degrades to `None`s in the
log line rather than raising. **This is the one coupling point that fails quietly.**

### 7. Import list — line ~382

```python
from .scenario_worker import (
    Grid, DubinsCar2,
    upwindFirstENO2, upwindFirstWENO5, add_ghost_cells, strip_dim,
    computeGradients, eval_u,
    _scenario_required_N, _scenario_wrap_pi, _ReachValueCache,
    rollout_cost, sample_states, scenario_delta_hat_worker,
    _scenario_pool_initializer,
)
```

Note `_scenario_required_N` changes arity: legacy `(eps, beta)`, two-stage
`(eps, beta, gamma)`. The two-stage version also adds `_scenario_kmax_aim` and
`_stage1_scan` to this list.

---

## What does *not* need reverting

The `ProcessPoolExecutor` machinery is untouched by the rewrite (plan P6a keeps it as-is):
`_get_scenario_pool`, `shutdown_scenario_pool`, `_scenario_pool_initializer`, the spawn
context, the BLAS pinning, `scenario_parallel` / `scenario_max_workers`. Same for
`_scenario_target2d`, `_scenario_sample`, `sublevel_bbox`, and every consumer of the
*result* (`self.safe_margin[i]` at HJR_FNO3d.py ~1209, ~1356, ~1481) — those read a plain
float and are indifferent to how it was produced.

`HJR_FNO/verfication/scenario_optimization_reach.py` is a separate heterocl-based
standalone with its own `rollout_cost` / `CachedConstraint`. Nothing imports across. Leave
it alone.

---

## Better than any of this

Commit the current state before P1 starts. A `git checkout <sha> -- HJR_FNO/` is a cleaner
revert than two `cp`s, and it also protects the ~277 uncommitted lines in `HJR_FNO3d.py`
that these snapshots exist to cover.
