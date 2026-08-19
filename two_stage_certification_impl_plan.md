> Open with VS Code Markdown Preview (`Ctrl+Shift+V`) to view rendered math.

# Two-Stage Robust Certification — **Implementation Plan**

Companion to [two_stage_certification_plan.md](two_stage_certification_plan.md) (the *theory*). This file is the
*execution* plan: what code changes, in what order, how each phase is verified, and what
is already done.

**Working rule: no phase is executed without explicit approval. Each phase is a separate,
independently revertable change.**

---

## 0. Status board

| phase | scope | status |
|---|---|---|
| P0 | Baseline capture + regression harness | ✅ **done** (2026-08-19) — see §8 |
| P1 | Sizing helpers (`k_aim`, `N(γ)`, tie-safe selector) | ☐ not started |
| P2 | `rollout_cost`: δ-free `t₀*` + sign-resolution early exit | ☐ not started |
| P3 | Stage 1: one-batch scoring + `select_delta_star` | ☐ not started |
| P4 | Stage 2 + 2-attempt driver: new `scenario_delta_hat_worker` | ☐ not started |
| P5 | Parent-side wiring in `HJR_FNO3d.py` (cfg, report, β budget) | ☐ not started |
| P6 | Parallelization across regions (+ optional intra-region) | ☐ not started |
| P7 | Validation, benchmark, acceptance | ☐ not started |

Update this table as phases land. Anything marked ☐ has **not** been written.

### Rollback → **[HJR_FNO/LEGACY_REVERT.md](HJR_FNO/LEGACY_REVERT.md)**

Byte-identical snapshots taken 2026-08-19 before any change. **Neither file was committed**
at that point (277 uncommitted lines in `HJR_FNO3d.py`), so these are the only copies of the
exact starting state.

| snapshot | sha256 (16) | importable? |
|---|---|---|
| `HJR_FNO/scenario_worker_legacy.py` | `9908c4b15fac3cb4` | yes — on purpose, for the P7.5 A/B |
| `HJR_FNO/HJR_FNO3d_legacy.py.bak` | `aa410a49c8f29ddd` | no — `.bak` on purpose |

```bash
cp HJR_FNO/scenario_worker_legacy.py HJR_FNO/scenario_worker.py
cp HJR_FNO/HJR_FNO3d_legacy.py.bak   HJR_FNO/HJR_FNO3d.py    # BOTH, or neither
```

⚠️ **The two files are coupled and a half-revert fails *silently*.** `cfg` is a plain dict,
so a key mismatch raises inside the worker, where `_certify_safe_margins` catches it per
region and prints "failed …; keeping safe_margin" — every region quietly retains its stale
margin while the planner keeps running. `LEGACY_REVERT.md` enumerates all seven coupling
points (cfg schema, constructor attrs, `_certified_once`, warm-start injection,
`rollout_cost` arity, report keys, import list) and the verification command.

**Do not edit the snapshots.** Better still: commit before P1 starts, and revert with
`git checkout <sha> -- HJR_FNO/` instead.

### Blast radius (verified by grep, 2026-08-19)

Only two files change:

- [HJR_FNO/scenario_worker.py](HJR_FNO/scenario_worker.py) — all numerics.
- [HJR_FNO/HJR_FNO3d.py](HJR_FNO/HJR_FNO3d.py) — config, report formatting, pool orchestration.

**Not touched:** `HJR_FNO/verfication/scenario_optimization_reach.py` is a separate
heterocl-based standalone with its *own* `rollout_cost`/`CachedConstraint`; nothing imports
across. `mppi_src/pi_mpc/mppi.py`'s `_rollout_cost_rbr` is an unrelated name collision.
Consumers of the *result* (`self.safe_margin[i]` at `HJR_FNO3d.py:1209, 1356, 1481`) read a
plain float and need no change.

---

## 1. Locked design decisions

Carried over from the design discussion; do not re-litigate during implementation.

| decision | value | rationale |
|---|---|---|
| `γ` | **0.5** | Admissible observed rate `(1−γ)ε = 0.5ε`. `N = 8·ln(1/β)/ε`. |
| `β` | **1e-6** | See §1.1 — β is the *cheap* parameter (`N ∝ ln(1/β)`); 1e-6 costs 33% fewer rollouts than 1e-9 for confidence that is still overwhelming under a union bound. |
| `ε` | **0.1** (unchanged), but see §1.1 | The *expensive* parameter (`N ∝ 1/ε`). This is the knob to move if latency bites. |
| `δ_start` | **−0.05** | Fixed constant, no warm start. δ\* ≤ δ_start, so this caps the certified set slightly inside the FNO's own predicted BRT `{V(·,0)<0}` — a small baked-in conservatism floor. |
| `max_attempts` | **1** (retry machinery still implemented, dormant) | Gives `β_eff = β` — the *full* confidence budget per region, no `β/2` split. §P4.1 defines the single-attempt outcomes. |
| warm start | **removed** | `delta_warm` / `_certified_once` / `_scenario_delta_grid` all retire. δ\* is continuous-valued, not grid-snapped. |
| `k_accept` | reuse existing [`_scenario_max_outliers`](HJR_FNO/scenario_worker.py#L198) | Already the exact `max{k : binom.cdf(k,N,ε) ≤ β}` with the `−1` infeasibility sentinel. |
| `k_aim` | **new** closed form, Campi–Garatti Eq. (8) at d=1 | `floor(εm − sqrt(2εm·ln(1/β)))`, evaluated at each level's own `m = n(δ)`. |
| `t₀*` | `find_max_safe_time_index(s, delta=0.0)` — **δ-free** | Load-bearing: if `t₀*` depended on δ, `failᵢ` would change per level and the one-batch-scores-all-levels argument collapses. |
| early exit | per-trajectory sign resolution (success `cost<0`, or `running_G>0` freezing the min) | Exact, not an approximation — see §P2. |
| retry rule | on 2f-empty **or** `ρ̂ ≥ ε`: set `ε ← ε̂`, rerun from Step 1 | Coded but unreachable at `max_attempts=1`. Flipping the constant to 2 re-enables it *and* forces `β_eff = β/2`. |
| parallel axis | per-region `ProcessPoolExecutor` (unchanged) | Existing machinery already correct; intra-region is optional (P6b). |

### 1.1 Sizing table — why β = 1e-6

`N = ceil( 2·ln(1/β) / (γ²·ε) )`, so at γ = 0.5 this is `N = 8·ln(1/β)/ε`.
**β enters logarithmically, ε enters reciprocally** — β is cheap, ε is expensive.

| ε | β | N = n | k_aim | k_accept | k_acc/N | rollouts (n+N) |
|---|---|---|---|---|---|---|
| 0.05 | 1e-9 | 3316 | 82 | 95 | 0.0286 | 6632 |
| 0.05 | 1e-6 | 2211 | 55 | 64 | 0.0289 | 4422 |
| **0.10** | 1e-9 | 1658 | 82 | 97 | 0.0585 | 3316 |
| **0.10** | 1e-7 | 1290 | 64 | 76 | 0.0589 | 2580 |
| **0.10** | **1e-6** | **1106** | **55** | **65** | 0.0588 | **2212** |
| **0.10** | 1e-4 | 737 | 36 | 44 | 0.0597 | 1474 |
| 0.15 | 1e-6 | 737 | 55 | 66 | 0.0896 | 1474 |
| 0.20 | 1e-6 | 553 | 55 | 67 | 0.1212 | 1106 |
| 0.20 | 1e-4 | 369 | 36 | 46 | 0.1247 | 738 |

`k_aim ≤ k_accept` everywhere (gap 6–18), confirming the stage-1 margin is real at these sizes.

**Recommendation: β = 1e-6.**

- Dropping 1e-9 → 1e-6 cuts `ln(1/β)` from 20.7 to 13.8: **33% fewer rollouts, for free.**
  Going further to 1e-4 saves another 33% but starts to matter under a union bound.
- 1e-6 is still overwhelming in practice. With ~5 safe regions and ~100 reveals per episode
  that is ~500 certificates issued; union bound ⇒ joint failure probability ≤ **5e-4**,
  i.e. the certification pipeline mis-certifies about once per 2000 episodes.
- Below 1e-6 you are paying linearly in `ln(1/β)` for confidence that is already far tighter
  than every other approximation in the stack (FNO error, linear interpolation, Euler `dt`).

**Cost vs the old sweep.** Old: `N=435` per level × `levels_evaluated`. The code notes a cold
sweep took ~11 levels (≈4.8k rollouts) and a warm one 2–4 (≈0.9–1.7k). New at ε=0.1, β=1e-6:
a flat **2212 rollouts**, always, with no warm-start dependence — cheaper than the old cold
sweep, ~1.3–2.5× the warm sweep, and each individual rollout is cheaper after P2's early exit.
The old numbers also bought *no valid guarantee*, so this is not a like-for-like trade.

**If latency still bites, move ε, not β.** ε=0.15 β=1e-6 gives N=737 (1474 rollouts, −33%);
ε=0.2 β=1e-6 gives N=553 (1106 rollouts, −50%). Decide against the P7.4 measured table.

---

## 2. Target API (`scenario_worker.py` after P4)

```python
# ---- sizing (pure, no state) --------------------------------------------
_scenario_required_N(eps, beta, gamma)  -> int      # CHANGED: ceil(2ln(1/b)/(g^2 e))
_scenario_kmax_aim(m, eps, beta)        -> int      # NEW: Eq.(8) at d=1, at m = n(delta)
_scenario_max_outliers(N, eps, beta)    -> int      # UNCHANGED (= k_accept)
_scenario_epsilon_from_Nk(N, k, beta)   -> float    # UNCHANGED (= eps_hat)

# ---- rollout -------------------------------------------------------------
rollout_cost(cache, s0, car, dt)        -> (B,)     # CHANGED: `delta` arg DROPPED
count_failures(cache, X, car, dt, chunk, stats)     # CHANGED: exact count, no kmax early exit
                                        -> (k:int, fail_mask:(n,) bool)

# ---- stage 1 -------------------------------------------------------------
_stage1_scan(v, fail, eps, beta, delta_start)       # NEW: Eq.(2.3)-(2.5), tie-safe
                                        -> (delta_star | None, m_star, k_star)

# ---- driver --------------------------------------------------------------
scenario_delta_hat_worker(V_full, obstacle_sdf, target2d, x_axis, y_axis,
                          theta_array, grid, car, cfg, verbose) -> (delta, report)

# ---- retired -------------------------------------------------------------
_scenario_delta_grid                                # DELETE
```

`cfg` schema after P5:

```python
dict(eps=0.1, beta=1e-6, gamma=0.5, delta_start=-0.05, delta_floor=-1.20,
     max_attempts=1,     # 1 => beta_eff = beta (full budget per region)
     n_scale=1.0,        # stage-1 oversample factor (see P3.3)
     eps_cap=0.25,       # refuse to return a set whose measured eps_hat exceeds this (P4.1)
     max_tries=400, seed=0, dt=..., grid_min=..., grid_max=...)
# GONE: delta_init, delta_step, delta_warm, M, step_frac
```

`report` schema after P4:

```python
dict(certified: bool, delta: float, attempt: int, attempts_used: int,
     eps_target: float,      # the target in force on the LAST attempt (may be eps_hat of attempt 1)
     eps_original: float,    # the user's original eps
     eps_hat: float | None,  # achieved bound at the returned delta
     beta_eff: float, gamma: float,
     n: int, N: int, k_aim: int, k_accept: int,
     m_star: int,            # n(delta*) — stage-1 survivors
     k_stage1: int,          # k(delta*)
     k: int | None,          # stage-2 failure count
     rho_hat: float | None, phi: float | None,
     verdict: str,           # 'accepted' | 'underpowered' | 'misses_target' | 'no_level' | 'infeasible' | 'empty_set'
     # cost counters
     rollouts_stage1: int, rollouts_stage2: int,
     sample_tries: int, sample_accept_rate: float, sample_boxed: bool,
     steps_saved_frac: float)   # early-exit effectiveness
```

---

## P0 — Baseline capture + regression harness

**Goal:** a before/after comparison that is not vibes. Must exist before any numerics change.

1. Write `eval/bench_certification.py` (new, standalone, no RRTx driver):
   - Loads a fixed `(env, seed)` obstacle map, builds `HJR_FNO`, calls `update_obs` once to
     populate `HJR_sets` / `obs_SDF` for all regions.
   - For each region, pickles `(V_full, obs_sdf, target2d, axes, grid, car)` into the
     scratchpad as `region_<i>.pkl`. **These frozen fixtures are the regression input** —
     every later phase runs against the same tubes, so nothing depends on re-running the FNO.
   - Runs the *current* `scenario_delta_hat_worker` on each and records
     `(delta_hat, report, wall_time)` to `bench_baseline.json`.
2. Record: per-region δ̂, total wall time, `levels_evaluated`, `rollout_chunks`, and the
   **total number of rollout trajectory-steps** (add a temporary counter in `rollout_cost`).
3. Note the baseline in §8 of this file.

**Gate:** baseline JSON exists and reruns reproducibly at fixed seed.
**Risk:** none — read-only w.r.t. the algorithm.

---

## P1 — Sizing helpers

**Goal:** pure functions, fully unit-testable, no dependency on the rest of the pipeline.

1. `_scenario_kmax_aim(m, eps, beta)` — Eq. (1.1)/(2.4):
   ```
   floor( eps*m - sqrt(2*eps*m*log(1/beta)) )     ; return -1 for m <= 0
   ```
   May legitimately return negative → the level auto-rejects (even `k=0` fails). Document
   that this is the *intended* auto-rejection of under-sampled levels, not an error path.
2. Change `_scenario_required_N(eps, beta)` → `(eps, beta, gamma)` = `ceil(2ln(1/β)/(γ²ε))`.
   Update the re-export list at [HJR_FNO3d.py:386](HJR_FNO/HJR_FNO3d.py#L386).
3. Delete `_scenario_delta_grid`.
4. Feasibility guards, asserted at call time in P4:
   - `k_aim(N) ≥ 0 ⟺ N ≥ 2ln(1/β)/ε ⟺ γ ≤ 1`
   - `k_accept ≥ 0 ⟺ N ≥ ln β / ln(1−ε)` (the existing `−1` sentinel)
   - always `k_aim ≤ k_accept` — **assert this in a test**; it is the margin the method
     depends on and a silent inversion would break Step 3.

**Tests** (`eval/test_scenario_sizing.py`, plain asserts, no pytest dependency required):
- `k_aim(N,ε,β) ≤ k_accept(N,ε,β)` over a grid of `(ε,β,γ)`.
- `k_accept` monotone non-decreasing in `N`, non-increasing in `β`↓.
- `eps_hat(N, k_accept(N,ε,β), β) ≤ ε` (the reported bound really is inside target).
- `N(γ) ∝ γ⁻²` to within the ceil.
- `k_aim(m)` < 0 exactly when `m < 2ln(1/β)/ε`.

**Gate:** all tests pass. No behaviour change yet (nothing calls the new helpers).

---

## P2 — `rollout_cost`: δ-free `t₀*` + sign-resolution early exit

**Goal:** the single biggest correctness *and* speed change. Isolated so it can be
diffed against P0 numerically.

### P2.1 Drop the `delta` argument

- [scenario_worker.py:364](HJR_FNO/scenario_worker.py#L364): `find_max_safe_time_index(s, delta=delta)` → `delta=0.0`.
- Remove `delta` from `rollout_cost`'s signature entirely (not defaulted — **removed**, so
  any stale caller is a `TypeError`, not a silent wrong answer).
- Update `_scenario_rollout_cost` at [HJR_FNO3d.py:873](HJR_FNO/HJR_FNO3d.py#L873).

Comment to add, verbatim intent: *`t₀*` must not depend on δ. If it did, `failᵢ` would
change per candidate level, and the stage-1 argument that one batch scores every level
would collapse — as would the requirement that π̃ be identical in both stages.*

### P2.2 Per-trajectory sign resolution

With `J = min_k max( ell(s_k), max_{j≤k} G(s_j) )`, failure iff `J ≥ 0`. Maintain
`running_G` and `cost` as today, and add:

```
resolved  = (cost < 0) | (running_G > 0)
integrate = active & ~resolved
```

- **`cost < 0`** ⇒ reached the target with no prior collision ⇒ `J ≤ cost < 0` ⇒ success, final.
- **`running_G > 0`** ⇒ every later term `max(ell, max_{j≤k}G) > 0` ⇒ the running min can
  never decrease again ⇒ `J = cost` **exactly**, sign already determined.

Both are *exact*, not heuristic truncations: the returned `cost` for a resolved trajectory
equals what the full-horizon loop would return. **State this in the docstring** — it is the
claim that makes the certificate unaffected.

⚠️ Sign conventions (repo): `ell < 0` **inside** the target; `G = −obstacle_sdf > 0`
**inside** the obstacle. Do not invert these.

### P2.3 Restrict per-step field evaluations to active rows

[scenario_worker.py:412-413](HJR_FNO/scenario_worker.py#L412-L413) currently evaluate
`cache.G(s[:, :2])` and `cache.ell(s[:, :2])` over the **full** batch every step, even
though only `ai` rows moved. Restrict both to `ai`. Combined with P2.2 the working set
shrinks monotonically, so `grad_at_indices` also gets cheaper each step.

**Preserved as-is (do not regress):**
- the stacked vector-valued RGI over `(x,y,θ,k)` in `_ReachValueCache.__init__` — one scipy
  call per step instead of up to `3·T`;
- the deleted dead stores in `upwindFirstWENO5`;
- per-slice (not whole-tube) gradient construction, which is L2-resident.

### Verification

- **Equivalence test:** on the P0 fixtures, sample 2000 states, run old `rollout_cost`
  (`delta=0`) vs new. Assert `sign(J_old) == sign(J_new)` for **all** samples, and
  `J_old == J_new` (bitwise or `atol=1e-12`) for all — resolution is exact, so equality
  should hold, not just sign agreement. Investigate any mismatch before proceeding.
- **Speed:** record trajectory-steps executed vs P0; report `steps_saved_frac`.

**Gate:** exact-equality test passes; measurable step reduction.

---

## P3 — Stage 1: one batch, one rollout per sample, then `δ*`

**Goal:** replace the descending sweep with a single cumulative pass.

1. **Rewrite `count_failures`** → returns `(k, fail_mask)`, exact, **no `kmax` early exit**.
   Stage 1 needs the per-sample flags and the exact cumulative counts at every level; the
   old early exit is illegal here. (Stage 2 may exit early in principle but runs to
   completion anyway, because `ε̂` for the retry needs the exact `k`.)
2. **`_stage1_scan(v, fail, eps, beta, delta_start)`** implementing Eq. (2.3)–(2.5):

```python
order = np.argsort(v, kind='stable')
vs, fs = v[order], fail[order]
kcum   = np.cumsum(fs)                       # k(delta) after j+1 samples
best   = None
for j in range(len(vs)):
    if j + 1 < len(vs) and vs[j+1] <= vs[j]:
        continue                             # TIE GUARD — see below
    m = j + 1                                # n(delta) = survivors
    if kcum[j] <= _scenario_kmax_aim(m, eps, beta):
        cand = vs[j+1] if j + 1 < len(vs) else delta_start
        best = (cand, m, int(kcum[j]))       # keep the LAST qualifying j
return best
```

Two correctness points to preserve in the comments:

- **Why `v[j+1]` is exactly right.** `S(δ)` uses a *strict* `<`. For any
  `δ ∈ (v[j], v[j+1]]`: all of `v[0..j] ≤ v[j] < δ` are counted, and `v[j+1] ≥ δ` is not.
  So `n(δ)` and `k(δ)` — hence the verdict — are **constant** on that interval, whose right
  endpoint is `v[j+1]` (closed, because of the strict `<`). Larger δ ⇒ larger recovered set
  at identical evidence ⇒ `v[j+1]` is the maximiser. At `j = n−1` the endpoint is
  `δ_start`, the largest δ any data covers.
- **Tie guard.** If `v[j] == v[j+1]` the interval is empty and setting `δ = v[j+1] = v[j]`
  would give `n(δ) ≤ j`, i.e. certifying against inflated evidence. Evaluate only at the end
  of each tie group. Near-measure-zero with a float interpolant; the guard is two characters.
- **Scan the whole array, keep the last pass.** `k(δ)/n(δ)` is not monotone in δ, so the
  qualifying set need not be contiguous — **do not break at the first failure.**

3. **`n_scale` knob.** `n(δ*) < n` always. If δ\* lands deep, the survivor count can drop
   below `2ln(1/β)/ε` and auto-reject purely for lack of resolution. Doc sets `n = N`;
   nothing forbids `n = ceil(n_scale · N)` with `n_scale > 1`. Default `1.0`; expose in cfg;
   cost is linear in stage-1 rollouts. Log `m_star` so under-resolution is visible.

**Tests:**
- Synthetic `(v, fail)` with hand-computed answers, incl. an all-ties array and an
  interleaved case where the qualifying set is non-contiguous.
- Property: `n(δ*) == #{v < δ*}` recomputed independently, and `k(δ*) ≤ k_aim(n(δ*))`.
- Property: no δ' > δ\* in the candidate set satisfies the test.

**Gate:** tests pass; `_stage1_scan` is not yet wired into the driver.

---

## P4 — Stage 2 + 2-attempt driver

**Goal:** rewrite `scenario_delta_hat_worker` end to end.

```
β_eff = β / max_attempts                    # max_attempts=1 => β_eff = β (full budget)
ε_cur = ε
for attempt in 1..max_attempts:
    # --- Step 1: sizing --------------------------------------------------
    N = required_N(ε_cur, β_eff, γ);  n = ceil(n_scale * N)
    k_accept = max_outliers(N, ε_cur, β_eff)
    if k_accept < 0: -> verdict 'infeasible', return δ_floor      # existing branch, kept

    # --- Step 2: stage 1 -------------------------------------------------
    X   = sample_states(cache, n, δ_start, rng, ...)              # RuntimeError -> 'empty_set'
    v   = cache.value_at_full_BRS(X)                              # cached once
    k1, fail = count_failures(cache, X, car, dt)                  # ONE rollout per sample
    sel = _stage1_scan(v, fail, ε_cur, β_eff, δ_start)

    if sel is None:                                               # 2f found nothing
        # ε̂ from the WHOLE stage-1 batch at δ_start. Legitimate: δ_start is fixed
        # a priori, so this batch IS a fresh i.i.d. sample of a pre-committed set,
        # and ε̂ is a valid certificate for S(δ_start) at level β_eff.
        ε_cur = eps_hat(n, k1, β_eff);  δ_last = δ_start;  continue   # -> 5c retry

    δ*, m*, k1* = sel

    # --- Step 3: stage 2, FRESH batch ------------------------------------
    del X, v, fail                                                # discard stage-1 data
    Xp = sample_states(cache, N, δ*, rng, ...)
    k, _ = count_failures(cache, Xp, car, dt)
    ρ̂ = k / N

    # --- Steps 3-4: verdict ----------------------------------------------
    if k <= k_accept:            -> 'accepted', return (δ*, ε̂ = eps_hat(N,k,β_eff))
    elif ρ̂ < ε_cur:              -> 'underpowered'   (5b: report N_new, do NOT auto-raise)
    else:                        -> 'misses_target'  (5c: ε_cur = eps_hat(N,k,β_eff), retry)
    δ_last = δ*
# attempts spent
warn("lower probability of safety accepted"); return (δ_last, ε̂_last, certified=False)
```

### P4.1 Single-attempt outcomes (`max_attempts = 1`)

With the loop running once, the retry branches are dead at runtime and every path must
resolve on the first pass. Four terminal cases, in decreasing order of desirability:

| case | return | `certified` | note |
|---|---|---|---|
| stage 2 `k ≤ k_accept` | `δ*`, `ε̂ ≤ ε` | **True** | the intended path |
| stage 2 `k > k_accept` | `δ*`, `ε̂ > ε` | False | **still a valid certificate at `ε̂`** — δ\* was fixed before that batch (§5a). Warn, return it, record `verdict` ∈ {`underpowered`, `misses_target`}. |
| stage 1 found no level (`sel is None`) | `δ_start`, `ε̂` from the whole stage-1 batch | False | Legitimate: `δ_start` is fixed a priori, so that batch *is* a fresh i.i.d. sample of a pre-committed set. |
| `k_accept < 0`, or `sample_states` raises | `δ_floor`, `ε̂ = None` | False | `verdict` ∈ {`infeasible`, `empty_set`} |

⚠️ **`eps_cap` guard (new).** Rows 2 and 3 can return a set whose measured `ε̂` is terrible
(e.g. 0.5) — the planner would then treat a badly-unsafe set as usable. Guard: if
`ε̂ > cfg['eps_cap']`, fall through to `δ_floor` instead. `δ_floor` is the *smaller* δ,
hence the smaller and more conservative set, so this fails in the safe direction. Default
`eps_cap = 0.25`; log every time it fires.

**Re-enabling retries later** is a one-constant change, but it is **not free**:
`max_attempts = 2` makes `β_eff = β/2`, so `ln(1/β_eff)` grows from 13.8 to 14.5 and
`N: 1106 → 1161` **per attempt** — i.e. up to ~2.1× the rollouts in the worst case, for a
second shot at a larger δ\*. Do not flip it without re-reading §1.1.

### P4.2 Notes to encode

- **Fresh data in stage 2 is not optional.** δ\* was placed where the stage-1 failures were
  *not*, so the stage-1 count at δ\* is an artefact of selection, not evidence.
- **5a is available.** A stage-2 `ε̂` at a δ\* fixed *before* that batch is already a valid
  certificate. The retry is not needed for validity — only to recover a *larger* δ\* under a
  relaxed target. Say so in the report (`verdict`, `eps_target` vs `eps_original`).
- **5b (underpowered)** does **not** auto-raise `N` inside the worker — that would blow the
  per-reveal latency budget unpredictably. Compute and *report*
  `N_new = ceil(2ε·ln(1/β_eff)/(ε−ρ̂)²)` so it can be tuned offline; the attempt still
  consumes its slot and falls through to the `ε ← ε̂` retry.
  (Former 5d folds in here: repeated near-misses across regions ⇒ lower γ at config level.)
- **`_ReachValueCache` is built once** per worker call and reused across both stages and
  both attempts — the gradient tube depends on neither δ nor ε.
- **RNG:** one `default_rng(cfg['seed'])` per worker call, drawn from sequentially. Stage-2
  independence from stage 1 is by construction (fresh draws from the same stream); document
  that a shared stream is fine, only *reuse of the same samples* is forbidden.
- Keep the total-function guarantee: every failure path returns a float, never raises.

**Gate:** runs on the P0 fixtures for all regions; reports look sane; δ̂ recorded for §8
comparison against baseline.

---

## P5 — Parent-side wiring (`HJR_FNO3d.py`)

1. **Constructor** ([HJR_FNO3d.py:624-645](HJR_FNO/HJR_FNO3d.py#L624-L645)):
   - Add `scenario_gamma = 0.5`, `scenario_delta_start = -0.05`,
     `scenario_max_attempts = 1`, `scenario_n_scale = 1.0`, `scenario_eps_cap = 0.25`.
   - **Change `scenario_beta` default `1e-9 → 1e-6`** (§1.1). This is a *default* change and
     must be called out in the commit message — callers that pass `scenario_beta` explicitly
     are unaffected, but nothing in-tree does.
   - Keep `scenario_eps = 0.1`.
   - Delete `scenario_delta_init`, `scenario_delta_step`, `scenario_M`, `scenario_step_frac`.
   - Keep `scenario_delta_floor` as the fallback-only clamp.
   - Resulting sizing: **N = n = 1106, k_aim = 55, k_accept = 65, 2212 rollouts/region.**
2. **Remove `self._certified_once`** ([:595](HJR_FNO/HJR_FNO3d.py#L595), [:994](HJR_FNO/HJR_FNO3d.py#L994), [:1005](HJR_FNO/HJR_FNO3d.py#L1005), [:1020](HJR_FNO/HJR_FNO3d.py#L1020)) — it existed only
   to distinguish "never certified" from "certified at 0" for the warm start, which is gone.
3. **`_scenario_cfg`** ([:857](HJR_FNO/HJR_FNO3d.py#L857)) → new schema (§2).
4. **`_certify_safe_margins`** ([:970](HJR_FNO/HJR_FNO3d.py#L970)) — drop the `cfg_i['delta_warm']`
   per-region branch; `cfg` becomes fully shared, so `_args(i)` only varies `V` and
   `obs_SDF[i]`. Keep the per-region try/except that preserves the old margin on failure.
5. **β budget across regions.** `scenario_beta` is documented as *per region*. The worker
   now splits it by `max_attempts` internally. Add an explicit `scenario_beta_joint: bool`
   — when true, pass `β/num_safe_regions` so the guarantee is joint over regions. Default
   `False` (preserves today's semantics); document the difference where the constant is set.
6. **`_format_scenario_report`** ([:898](HJR_FNO/HJR_FNO3d.py#L898)) — rewrite for the new schema.
   Target one line per region:
   ```
   [scenario] region 3: delta_hat = -0.35 | attempt 1/2 verdict=accepted
     | n=4632 m*=1204 k1=8 | N=4632 k=41/4632 (k_acc=52) rho=0.89%
     | eps_hat=9.4e-03 (target 1.0e-02) phi=0.26 | cert
     | draws=2 acc=0.41 steps_saved=63%
   ```
7. **Import list** ([:382-389](HJR_FNO/HJR_FNO3d.py#L382-L389)) — add `_scenario_kmax_aim`,
   `_stage1_scan`; drop nothing that still exists.

**Gate:** `topo_prm/demo.py` (which sets `scenario_enable=False`) still imports;
`rrtx_FNO3d_oneGoal.py` runs one episode end-to-end without touching the certification path
semantics beyond the new δ̂ values.

---

## P6 — Parallelization

### P6a — Per-region (keep, adjust)

The existing axis is correct and unchanged in structure: `_get_scenario_pool` +
`ProcessPoolExecutor(mp_context=spawn, initializer=_scenario_pool_initializer)` over the M
changed regions, workers torch-free.

Adjustments needed:

1. **Payload size.** Each job pickles a full `(Nx,Ny,Nθ,T)` float32 tube. Unchanged in
   count, but jobs now live ~2× longer (two stages, possibly two attempts), so pool reuse
   matters more. Keep the lazy-reuse logic at [:929](HJR_FNO/HJR_FNO3d.py#L929).
2. **`max_workers = 6`** was sized for the old per-job memory. The new worker holds *two*
   sample batches transiently (stage-1 `X`+`v`+`fail`, then stage-2 `Xp`). At `N = 1106 × 3
   floats` (~13 kB) this is negligible vs the gradient tube (`Nx·Ny·Nθ·T·3·8 B`, ~25 MB at
   50×50×25×17). Re-measure RSS per worker in P7 and re-pick the cap; do not guess.
3. **Per-region seeding** — `cfg['seed'] + region_index`. **Free, and worth doing, but not a
   correctness fix.** See §4 Q4 for the full reasoning; the short version is that it costs
   one integer add, and the earlier claim that it was needed for order-independent
   reproducibility was wrong (each worker already re-seeds from `cfg['seed']` internally at
   [scenario_worker.py:615](HJR_FNO/scenario_worker.py#L615), so completion order never
   mattered). The real benefit is decorrelating the P7.2 validation across regions.
4. **BLAS pinning** stays (`_scenario_pool_initializer` + parent-side env).

### P6b — Intra-region (optional, only if P7 shows a need)

When `len(regions) < max_workers`, cores idle. The rollout batch is embarrassingly parallel
over trajectories, so `count_failures` could fan `X` out in chunks.

Do this **only** if P7 measures a real deficit, and only via a *nested* submit guarded
against pool deadlock (a worker cannot submit to its own pool). Preferred shape: the parent
splits a single-region job into `W` chunk-jobs at the `count_failures` level, each returning
`(k_chunk, fail_mask_chunk)`, and reassembles in order. `_stage1_scan` then runs in the
parent. This changes the worker's decomposition, so it is deliberately deferred.

**Recommendation: default to P6a only.** Two stages already cost ~2 batches; per-region
fan-out plus the P2 early exit is likely enough.

---

## P7 — Validation, benchmark, acceptance

1. **Numerical equivalence where it must hold:** P2's exact-`J` test (already gated).
2. **Statistical sanity — the key experiment.** For each fixture, take the returned δ\* and
   draw a large **independent** batch (`50k`) uniform on `S(δ*)`; measure the true failure
   rate `ρ_true`. Assert `ρ_true ≤ ε` on every fixture, and report the margin.
   Repeat over ≥10 seeds × all regions and report the distribution of `ρ_true`.

   ⚠️ **Be honest about what this can and cannot show.** β = 1e-6 is *not* empirically
   checkable at this sample size — ~50 trials cannot resolve a one-in-a-million exceedance
   rate, and observing zero exceedances is consistent with almost any β. What the experiment
   actually tests is the much more likely failure mode: an **implementation** bug (wrong sign,
   biased sampler, δ\* off by one index) would push `ρ_true` far above ε and show up
   immediately. Treat it as a bug detector, not a validation of β.
3. **Selection-bias check.** Confirm the old method was optimistic: run the *stage-1*
   selection and then measure `ρ_true` at δ\* without stage 2. Expect `ρ_true` to sit
   above the stage-1 empirical rate — this quantifies why stage 2 is required and is worth
   a figure for the writeup.
4. **Cost table — measured.** §1.1 gives the *analytic* `N/k_aim/k_accept`; P7.4 adds the
   columns theory cannot supply: **wall time per region, and the δ̂ actually achieved.**
   Sweep `ε ∈ {0.1, 0.15, 0.2}` at the locked `γ=0.5, β=1e-6`. The question this answers is
   whether raising ε buys speed *without* materially shrinking δ̂ — if δ̂ is flat in ε, take
   the cheaper ε; if δ̂ degrades, keep 0.1. Also spot-check `γ ∈ {0.3, 0.5}` at ε=0.1 to
   confirm 0.5 is not leaving δ̂ on the table.
5. **Speed + δ̂ vs P0 baseline.** Rerun `bench_certification.py --mode run` on the same
   `env_C_s1` fixtures with `--label twostage` and diff against
   `env_C_s1__baseline.json`. Because `scenario_worker_legacy.py` is importable, the old
   and new workers can be run back-to-back **in one process on identical inputs** — add a
   `--legacy` flag to the bench that swaps the module.
   Per §8, the expectation is **~1.5× the trajectory count** at comparable wall time; the
   headline result is δ̂ (does the two-stage recover a set as large as the sweep's −0.5/−0.7?)
   and validity, not speed.
6. **End-to-end regression.** One `rrtx_FNO3d_oneGoal.py` episode at a fixed `(env, seed)`;
   compare path length / replans / collisions against the pre-change run. δ̂ values *will*
   differ (that's the point) — the check is that the planner still behaves sensibly and
   `points_feasible` does not start rejecting everything.

### Acceptance criteria

- [ ] `sign(J)` and `J` identical old-vs-new at `delta=0` on ≥2000 states (P2).
- [ ] `k_aim ≤ k_accept` holds across the tested `(ε,β,γ)` grid.
- [ ] Independent-batch `ρ_true ≤ ε` at the returned δ\* on all fixtures.
- [ ] Rollout trajectory-steps reduced vs P0 baseline (record the factor).
- [ ] One full episode completes with no regression in reached-goal / collision counts.
- [ ] Every failure path returns a float; no new exception escapes to `_certify_safe_margins`.

---

## 3. Risk register

| # | risk | mitigation |
|---|---|---|
| R1 | Per-reveal latency grows | **Retired by measurement** (§8): the legacy sweep costs 0.96 s for 7 regions, so even a 2–4× increase is immaterial. ε remains the lever if a slower machine says otherwise. |
| R2 | `n(δ*)` too small ⇒ `k_aim(m) < 0` ⇒ chronic `verdict='no_level'` | `n_scale` knob (P3.3); log `m_star` every region |
| R3 | Someone re-adds a δ-dependent `t₀*` | `delta` **removed** from the signature (not defaulted) so stale callers raise |
| R4 | Ties in `v` inflate the evidence at δ\* | explicit tie guard + a dedicated unit test |
| R5 | Early exit changes `J` rather than just truncating | exact-equality test in P2, not sign-only |
| R6 | β accounting drifts (per-region vs joint vs per-attempt) | one place computes `β_eff`; put the three-way split in the report dict |
| R7 | An uncertified δ\* with a terrible `ε̂` is returned and silently used as if safe | `eps_cap` guard (P4.1) falls back to `δ_floor`; `eps_original` **and** `eps_target` both in the report; warn loudly. Retry-hides-bad-set is moot at `max_attempts=1`. |
| R8 | Worker RSS × 6 under the larger `N` | measure in P7.5, re-pick `max_workers` |

---

## 4. Questions — resolved

**Q1. γ default → 0.5.** ✔ Locked. Admissible observed rate `0.5ε`; `N = 8·ln(1/β)/ε`.

**Q2. δ_start → −0.05.** ✔ Locked. δ\* ≤ −0.05 always, so the certified set sits strictly
inside the FNO's predicted BRT. Note the interaction with P4.1 row 3: when stage 1 finds no
qualifying level the worker returns `δ_start` itself, so −0.05 (rather than 0.0) also makes
that fallback mildly conservative instead of maximally optimistic.

**Q3. β → 1e-6.** ✔ See §1.1 for the table and reasoning. Headline: `N ∝ ln(1/β)` so β is
the *cheap* parameter — 1e-9 → 1e-6 is a free 33% rollout cut, and 1e-6 still gives
~5e-4 joint failure probability over an entire ~500-certificate episode. **Move ε, not β,
if you need more speed** (`N ∝ 1/ε`: ε=0.15 → −33%, ε=0.2 → −50%).

**Q4. Is per-region seeding bad? Does it cost anything?** **No, and no.**

- *Cost:* zero. `default_rng(cfg['seed'] + i)` versus `default_rng(cfg['seed'])` — one
  integer add, once per worker call. No extra draws, no extra state, no synchronisation.
- *Is it bad?* No. Every region's certificate is a **marginal** statement about its own
  `S(δ*)`, and a union bound over regions does not require independence either — so
  correctness holds under a shared seed *or* per-region seeds.
- *So why do it?* Under a shared seed every region draws the **identical** uniform proposal
  sequence over the same `(x,y,θ)` box. The tubes differ so the accepted subsets differ, but
  the proposals are perfectly correlated across regions. That is harmless for the guarantee
  and actively misleading for validation: P7.2 measures an exceedance rate *across* regions,
  and correlated draws understate its variance.
- *Correction to the earlier note:* per-region seeding is **not** needed for reproducibility.
  Each worker constructs its own `default_rng` from `cfg['seed']` inside the call
  ([scenario_worker.py:615](HJR_FNO/scenario_worker.py#L615)), so results were already
  independent of pool completion order.

**Q5. max_attempts → 1.** ✔ Locked, so `β_eff = β` and every region gets the full 1e-6
budget. The retry machinery is still written (P4), just unreachable; see P4.1 for the four
terminal outcomes and the new `eps_cap` guard, and P4.1's last paragraph for what flipping
the constant back to 2 would cost.

### Still open

**Joint-over-regions β.** `scenario_beta` remains *per region*, so the joint guarantee across
`num_safe_regions` sets is weaker than the constant suggests. §1.1's union-bound estimate
(~5e-4 per episode) says this is comfortable at β=1e-6, so the plan keeps
`scenario_beta_joint = False` by default (P5.5) and leaves the flag available. Revisit only
if the region count grows substantially.

---

## 5. Reference — equations, for implementation

```
(1.3)  N        = ceil( 2 ln(1/b) / (g^2 e) )                        n = ceil(n_scale*N)
(1.1)  k_aim(m) = floor( e*m - sqrt(2*e*m*ln(1/b)) )                 Eq.(8), d=1
(0.1)  k_accept = max{ k : binom.cdf(k, N, e) <= b }                 exact
(2.1)  t0*(x0)  = max{ t : Vtilde(x0,t) < 0 }                        delta-FREE
(2.3)  n(d)     = #{i : v_i < d}     k(d) = #{i : v_i < d, fail_i}   phi = n(d)/n
(2.5)  d*       = max{ d : k(d) <= k_aim(n(d)) }   realised at v[j+1]
(3.1)  accept  iff  k <= k_accept
(3.2)  eps_hat  = min{ e' : binom.cdf(k, N, e') <= b }
(5.2)  N_new    = ceil( 2*e*ln(1/b) / (e - rho_hat)^2 )              report only
(5.3)  g_new    = 1 - rho_hat/e
```

Guarantee at acceptance, w.p. ≥ 1 − β_eff:

```
Pr_{x in S(d*)}( J_pi(x,0) >= 0 ) <= e      and      Pr_{x in S(d*)}( V(x,0) >= 0 ) <= e
```

---

## 6. What each phase deletes

Tracked separately so nothing is orphaned:

- `_scenario_delta_grid` (P1)
- `rollout_cost(..., delta=...)` parameter (P2)
- `count_failures`' `kmax` early-exit + `(k, partial)` return shape (P3)
- the entire warm-start / descending-sweep body of `scenario_delta_hat_worker`, `evaluated`
  dict, `_eval` closure, `levels_evaluated` counter (P4)
- `self._certified_once`, `scenario_delta_init`, `scenario_delta_step`, `scenario_M`,
  `scenario_step_frac`, `cfg['delta_warm']` (P5)

---

## 7. Execution order & dependencies

```
P0 ──> P1 ──> P2 ──> P3 ──> P4 ──> P5 ──> P7
                      │              └──> P6a
                      └── P2 is independently testable against P0 fixtures
                                          P6b only if P7 demands it
```

P1 and P2 do not depend on each other and could be approved together.
Nothing before P4 changes runtime behaviour of the planner.

---

## 8. Measurements log

### P0 baseline — 2026-08-19, `env_C` seed 1, git `d839998`

Harness: [eval/bench_certification.py](eval/bench_certification.py). Fixtures:
`bench/certification/env_C_s1/` (7 regions, all re-predicted; grid 50×50×25, T=17,
dt=0.5). Results: `bench/certification/env_C_s1__baseline.json`.

```bash
conda run -n rrtx python eval/bench_certification.py --mode dump --scenario env_C --seed 1
conda run -n rrtx python eval/bench_certification.py --mode run  --tag env_C_s1 \
    --eps 0.1 --beta 1e-9 --beta 1e-6 --label baseline
```

**Legacy sweep, ε = 0.1, cold start (`delta_warm=None`):**

| β | N | k_max | regions cert. | δ̂ per region | wall (7 reg) | traj-steps | trajectories |
|---|---|---|---|---|---|---|---|
| 1e-9 | 435 | 10 | 7/7 | −0.7, −0.7, −0.5, −0.5, −0.5, −0.7, −0.7 | **0.96 s** | 110,070 | 10,520 |
| 1e-6 | 297 | 7 | 7/7 | −0.7, −0.7, −0.5, −0.5, −0.5, −0.7, −0.7 | **0.87 s** | 88,317 | 8,469 |

δ̂ is identical at both β — the sweep lands on the same grid level either way, so β=1e-6
costs nothing in set volume here. Levels evaluated: 5–7 per region (not the ~11 the source
comment suggests). Mean **10.1–10.8 integration steps per trajectory** out of T−1 = 16
possible, i.e. `t₀*` already starts most rollouts well inside the tube.

### Three things this changes about the plan

1. **Certification is not a bottleneck.** 0.96 s for all 7 regions, ~0.14 s each. The
   §1.1 worry about `N: 435 → 1106` costing latency is largely moot in absolute terms —
   the two-stage version should land around 2–4 s for 7 regions. **R1 is retired.**
   Re-check on a slower machine before treating this as settled.
2. **`trajectories` (10,520) is far below `levels × N` (43 × 435 = 18,705).** The existing
   `count_failures` `kmax` early exit is already discarding ~44% of the work on failed
   levels. So the "rollouts drop from `levels × N` to `n + N`" claim must be measured
   against **10,520**, not 18,705 — the honest comparison for the two-stage at β=1e-6 is
   `7 × (n+N) = 7 × 2212 = 15,484` trajectories, i.e. **~1.5× more trajectories than the
   legacy sweep**, offset by P2's per-trajectory early exit. Do not claim a rollout-count
   win; claim a *validity* win at comparable cost.
3. **P2's early exit has less headroom than assumed** — 10.5/16 steps means at most ~34%
   can come from the `n_steps` bound alone. The real gain must come from the
   `running_G > 0` / `cost < 0` resolution firing early. Measure it, don't assume it.

### Log

| date | phase | config | δ̂ (7 regions) | wall | traj-steps | trajectories |
|---|---|---|---|---|---|---|
| 2026-08-19 | P0 baseline | ε=0.1 β=1e-9 legacy | −0.7×2, −0.5×3, −0.7×2 | 0.96 s | 110,070 | 10,520 |
| 2026-08-19 | P0 baseline | ε=0.1 β=1e-6 legacy | −0.7×2, −0.5×3, −0.7×2 | 0.87 s | 88,317 | 8,469 |
