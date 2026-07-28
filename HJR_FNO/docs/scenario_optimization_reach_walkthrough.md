# How `scenario_optimization_reach.py` Works — DubinsCar2 + 5-channel FNO3d

> Open with VS Code Markdown Preview (`Ctrl+Shift+V`) to view rendered math.

This note explains [scenario_optimization_reach.py](../verfication/scenario_optimization_reach.py)
as it is **currently wired**: a trained `FNO3d` ([../HJR_FNO3d.py](../HJR_FNO3d.py))
approximates the HJ reach-avoid value function for a **3-state Dubins car**, and
scenario optimization certifies a corrected threshold $\hat\delta$ per obstacle
configuration. See also the problem set-up in
[reach_avoid_problem_summary.md](reach_avoid_problem_summary.md).

---

## 0. The setting

We approximate the reach-avoid value function

$$
\tilde V(x, y, \theta, t \,;\, c)
$$

- State $s = (x, y, \theta) \in [-10,10]^2 \times [-\pi, \pi]$ (heading periodic).
- $t \in [0, T_{\max}]$, $T_{\max} = 8$ s, lookback step $\Delta t = 0.5$ s ($T = 17$ slices).
- $c$ = the **obstacle SDF**, `mat["constraints"][i]`, a $(50,50,36)$ array
  (heading-independent). Convention (odp linear SDF): $c(s) < 0$ **inside** the
  obstacle.
- $\ell(x,y)$ = fixed target SDF (disk radius 2 at the origin), $\ell < 0$ inside.

The FNO carries **$\theta$ as a constant 5th input channel**, so one forward pass
gives $V$ over $(x,y,t)$ at a fixed heading; sweeping all headings gives the full
$V(x,y,\theta,t)$. Input channel order is $[\,\text{sdf},x,y,t,\theta\,]$ with
$\text{sdf} = -c$.

**DubinsCar2 dynamics** ([../../optimized_dp/odp/dynamics/DubinsCar2.py](../../optimized_dp/odp/dynamics/DubinsCar2.py)):

$$
\dot x = v\cos\theta + d_1,\quad \dot y = v\sin\theta + d_2,\quad \dot\theta = \omega + d_3,
$$

with $v \in [0,1]$, $\omega \in [-1,1]$, $|d_i| \le 0.1$, `uMode=min`,
`dMode=max` (control reaches the goal / avoids the obstacle; disturbance is
adversarial). The induced controller is bang-bang in the Dubins Hamiltonian.

**Avoidance + cost.** Reach-avoid uses the avoidance function $G = -c$
($G > 0$ inside the obstacle). The realized rollout cost is

$$
J(s_0) \;=\; \min_{k}\ \max\!\Bigl(\ell(x_k,y_k),\ \max_{j\le k} G(x_j,y_j)\Bigr),
\qquad \textbf{failure} \iff J \ge 0 .
$$

We want, per obstacle $c$, a scalar $\hat\delta$ such that the **recovered set**

$$
\boxed{\ \hat{\mathcal S}(c) = \{\, s : \tilde V(s,\ \text{fully-grown},\ c) < \hat\delta \,\}\ }
$$

is contained in the true reach-avoid set with high probability.

> **Time convention (odp).** Slice index $0$ is the **fully-grown** BRT (full
> lookback); index $-1$ is the terminal/target slice $\max(\ell,-c)$.
> $\tilde V(s,\text{fully-grown})$ is slice $0$, queried via
> [`CachedConstraint.value_at_full_BRS`](../verfication/scenario_optimization_reach.py#L233).

---

## 1. Two modes

### Mode A — `per_c` with Bonferroni (the main mode)

`per_c` = **per obstacle configuration**. Run scenario optimization independently
for each $c_k$ in the list (the test obstacles from the `.mat`). With confidence
$\beta_k = \beta/K$ per run, the joint claim across $K$ obstacles holds at $1-\beta$:

$$
\boxed{\;\Pr\!\Bigl(\,\bigcap_{k=1}^{K}\bigl\{\Pr_{s}\!\bigl(J(s)\ge 0 \mid \tilde V(s,\text{fg},c_k) < \hat\delta_k\bigr) \le \varepsilon\bigr\}\Bigr) \;\ge\; 1 - \beta.\;}
$$

Since the per-run budget $N$ scales as $\log(1/\beta_k)$, Bonferroni only adds a
$\log K$ factor.

### Mode B — `joint`

One $\hat\delta$ over the joint distribution of (state, obstacle); obstacles are
resampled each iteration via [`make_obstacle_sampler`](../verfication/scenario_optimization_reach.py#L717)
(reusing the data-gen `random_obstacle_set`). Sample budget

$$
N = \left\lceil \tfrac{2}{\varepsilon}\bigl(\ln \tfrac1\beta + 1\bigr)\right\rceil .
$$

---

## 2. `DubinsReachProblem`

| Field | Meaning |
|---|---|
| `x_axis`, `y_axis`, `theta_axis`, `tau` | Grid axes from the `.mat`. |
| `target_sdf` | $\ell$ on the $(x,y)$ grid (heading-independent slice). |
| `x_lo`, `x_hi` | 3-D sampling box $[-10,10]^2\times[-\pi,\pi]$. |
| `car` | `DubinsCar2` (bang-bang opt-ctrl + worst-dstb). |
| `odp_grid` | 3-D periodic-$\theta$ grid for `computeSpatDerivArray`. |
| `dt` | Euler step = `tau` spacing (0.5 s). |
| `obstacle_sampler` | Callable for `joint` mode. |
| `c_list` | Obstacle SDFs for `per_c` mode. |

Built by [`build_problem_from_mat`](../verfication/scenario_optimization_reach.py#L730).

---

## 3. `FNOValueModel` and `CachedConstraint`

### 3a. `FNOValueModel.predict_full` — [scenario_optimization_reach.py:149](../verfication/scenario_optimization_reach.py#L149)

Builds the 5-channel input $[\,\text{sdf},x,y,t,\theta\,]$ with $\text{sdf}=-c$
(broadcast over $t$; coordinate/time grids; $\theta$ a **constant** channel),
sweeps all headings in batches, and stacks $\to V$ of shape
$(N_x, N_y, N_\theta, T)$. Channel order matches training byte-for-byte.

### 3b. `CachedConstraint(obstacle_sdf, prob, model)` — [scenario_optimization_reach.py:184](../verfication/scenario_optimization_reach.py#L184)

One FNO forward pass per obstacle, then build (all subsequent queries are
vectorised CPU interpolations — no autograd, no per-step network call):

- 3-D `RegularGridInterpolator` of $V$ over $(x,y,\theta)$, **per time slice**.
- 3-D interpolators of the **spatial gradient** $(\partial_x V, \partial_y V,
  \partial_\theta V)$ per time slice, computed with
  [`odp.solver.computeSpatDerivArray`](../../optimized_dp/odp/solver.py#L440)
  (`deriv_dim` $1\to x$, $2\to y$, $3\to\theta$) — the same routine the HJ solver
  uses, so the gradient is consistent with the trained value function.
- 2-D interpolators for $\ell$ and the avoidance function $G = -c$.

> **Cost note.** `computeSpatDerivArray` rebuilds a HeteroCL graph per call: with
> $T = 17$ slices $\times\ 3$ axes that is $\sim 51$ graph builds per obstacle —
> a one-time per-$c$ setup cost; rollouts after that are pure interpolation.

The induced policy (per fixed $c$) is bang-bang in the Dubins Hamiltonian; the
rollout calls `DubinsCar2.optCtrl_inPython`, `optDstb_inPython`, and
`dynamics_inPython` so it matches the HJ-solver logic exactly.

---

## 4. Constrained rollout cost — [`rollout_cost`](../verfication/scenario_optimization_reach.py#L282)

For each $s_0 = (x,y,\theta)$, forward-Euler under the induced controller and
worst-case disturbance, wrapping $\theta$ to $[-\pi,\pi]$, tracking
$G_k = \max_{j\le k} G(x_j,y_j)$ and

$$
J(s_0) = \min_k \max\bigl(\ell(x_k,y_k),\, G_k\bigr).
$$

**Per-sample starting slice.** Each state starts at the *tightest* time slice
whose sublevel set still contains it
([`find_max_safe_time_index`](../verfication/scenario_optimization_reach.py#L250)):
the largest $k$ with $V_{\text{grid}}[\dots,k](s) \le \delta$. The rollout then
marches that index toward $T-1$ (target), encoding decreasing time-to-go.
Failure iff $J \ge 0$.

---

## 5. Sampling under the level set — [`sample_x_under_delta`](../verfication/scenario_optimization_reach.py#L327)

Uniform rejection over the 3-D box, keeping $s$ with
$\tilde V(s,\text{fully-grown}) < \delta$.

> **Update vs. the old code.** $\delta$ now **starts at `delta_init = 0`**, so the
> first iteration samples only states **inside the learned BRT** $\{V < 0\}$ —
> not uniformly over the whole box (which wasted effort on far-field states that
> trivially fail). Set `--delta-init inf` to recover the old behavior.

---

## 6. The per-obstacle loop — [`scenario_optimize_per_c`](../verfication/scenario_optimization_reach.py#L454)

```text
beta_k = beta / K
N      = ceil((2/eps) * (ln(1/beta_k) + 1))
for c_k in c_list:
    cache = CachedConstraint(c_k, prob, model)     # one FNO forward pass
    delta = delta_init        (= 0.0)              # start inside the BRT
    for i in 1..M:
        X         = sample N states with V(s, fg, c_k) < delta
        J         = rollout_cost(X, cache, delta)
        violators = J >= 0
        if no violators: converged; break
        raw_delta = min V(X[violators], fg, c_k)   # worst violator
        target    = max(raw_delta, delta_floor)    # never below floor
        delta     = delta + step_frac * (target - delta)   # DAMPED update
        if floor-limited and step ~ 0: pin at floor; floored; break
    # fresh independent batch -> report rollout failures at final delta_hat
record delta_hat_k, converged, floored, n_fail / n_eval
```

### 6a. Damped $\delta$ update — [scenario_optimization_reach.py:528](../verfication/scenario_optimization_reach.py#L528)

The exact scenario step jumps straight to the worst learned violator value,
$\delta \leftarrow \min\{V(s): J(s)\ge 0\}$, which can be a large drop. We damp it:

$$
\text{target} = \max\bigl(\min_{\text{viol}} V,\ \delta_{\text{floor}}\bigr),
\qquad
\boxed{\ \delta \leftarrow \delta + \rho\,(\text{target} - \delta)\ },\quad \rho = \texttt{step\_frac}.
$$

- $\rho = 1.0$ reproduces the exact one-shot scenario update (tightest).
- $\rho < 1$ (default $0.5$) moves $\delta$ only part-way each iteration → smoother
  changes, **more iterations**. The recovered set may still contain the worst
  violator between iterations, so convergence is iterative — but the loop only
  declares `converged` when a sampled batch has **zero** violators, so the final
  set is valid regardless of $\rho$.

### 6b. Floor clamp — `delta_floor` (default $-1.40$)

$\hat\delta$ is never aimed below `delta_floor`. If the worst violator sits below
the floor and $\delta$ has settled onto it, $\hat\delta$ is **pinned** at the floor
and the run stops with `floored=True`, `converged=False` (you are deliberately
keeping a *larger* recovered set than the guarantee certifies). Pass
`--delta-floor -inf` to disable.

### 6c. Failure report

After each obstacle, a **fresh independent batch** of $N$ states is drawn under
$\{V < \hat\delta\}$ and rolled out; `n_fail = #\{J \ge 0\}` and `fail_frac` are
recorded and printed.

### 6d. `joint` — [scenario_optimization_reach.py:599](../verfication/scenario_optimization_reach.py#L599)

Same idea, but each iteration draws fresh obstacles, builds caches on the fly, and
concatenates $(s, c)$ pairs until $\ge N$ samples. (No floor / damping wired in;
say the word to add them.)

---

## 7. Why a converged run can still report a failure

`converged=True` is decided on the **batch sampled that iteration** (it had zero
violators). The reported `fails n/N` comes from a **different, fresh batch** drawn
afterward (§6c). Independent draws → a rare near-boundary state can fail in one
batch and not the other. This is exactly the scenario guarantee: violation
probability $\le \varepsilon$ **with confidence** $\ge 1-\beta$ — not "zero
failures." With $\varepsilon = 10^{-2}$, an empirical $0.02\%$ is far inside the
bound.

There are **two cutoffs**:

$$
\underbrace{\tilde V(s,\text{fg}) < \hat\delta}_{\text{set membership}}
\qquad\text{and}\qquad
\underbrace{J(s) \ge 0}_{\text{rollout failure}} .
$$

The residual failures are typically states whose $V$ sits *just below* $\hat\delta$
(the recovered-set boundary), where the FNO's approximation error is largest. To
tighten: lower $\varepsilon$ (bigger $N$, lower $\hat\delta$) or subtract a safety
margin from $\hat\delta$.

---

## 8. Visualization — `--visualize`

For each obstacle, after computing $\hat\delta$, one rollout is animated.

- A feasible start state is chosen near the recovered-set boundary by
  [`_pick_viz_state`](../verfication/scenario_optimization_reach.py): inside
  $\hat{\mathcal S}$ ($V < \hat\delta$), outside the obstacle ($G < 0$), with $V$
  as large as possible.
- **CUDA safety.** Interactive matplotlib SIGSEGVs while a CUDA context is live, so
  the animation runs in a **separate process** ([../verfication/_viz_worker.py](../verfication/_viz_worker.py))
  that imports only numpy + matplotlib. `visualize_constraint_result` precomputes
  every frame (pure numpy), dumps a temp `.npz`, and
  `subprocess.run([...])` **blocks until the window is closed** — closing it
  advances to the next obstacle.

Each frame (x-y plane) shows:

| color | set | behavior |
|---|---|---|
| **magenta** | $V(x,t)=0$ — current BRT at the live heading / time slice | shrinks toward target |
| **blue** | $V(x,0)=0$ — fully-grown BRT | fixed |
| **lime** | $V(x,0)=\hat\delta$ — recovered set $\hat{\mathcal S}$ | fixed |
| red `--` | obstacle | fixed |
| green `--` | target | fixed |
| black | rollout trajectory (■ start, ● current) | grows |

---

## 9. End-to-end usage

Default paths ([scenario_optimization_reach.py:90](../verfication/scenario_optimization_reach.py#L90)):

- ckpt: `HJR_FNO/training/model/01_FNO3d_dubins_5ch_tuned.pt`
- mat:  `HJR_FNO/data_gen/HJB_training_mat/DubinsCar2_50x50x36_reach_avoid.mat`

```bash
# under the `tTLT` conda env (has heterocl + torch)
python HJR_FNO/verfication/scenario_optimization_reach.py \
    --mode per_c --K 10 --eps 1e-2 --beta 1e-9 \
    --delta-init 0.0 --step-frac 0.5 --delta-floor -1.40 --visualize
```

| Flag | Meaning | Default |
|---|---|---|
| `--mode` | `per_c` (per obstacle) or `joint` | `per_c` |
| `--eps`, `--beta` | scenario tolerance / confidence | `1e-2`, `1e-9` |
| `--K` | number of test obstacles (`per_c`) | `10` |
| `--delta-init` | starting threshold (first iter samples $\{V<\delta_{\text{init}}\}$) | `0.0` |
| `--step-frac` | damping $\rho$ of the $\delta$ update; `1.0` = exact | `0.5` |
| `--delta-floor` | lower clamp on $\hat\delta$ (`-inf` to disable) | `-1.40` |
| `--visualize` | animate one rollout per obstacle | off |

> **Model load.** The notebook saved a full-model pickle whose class path is
> `__main__.FNO3d`, so [`load_fno`](../verfication/scenario_optimization_reach.py#L702)
> injects `FNO3d`/`SpectralConv3d` into `__main__` before `torch.load`.

### Programmatic recovery of the safe set

```python
from HJR_FNO.verfication.scenario_optimization_reach import (
    load_mat, load_fno, FNOValueModel, build_problem_from_mat,
    CachedConstraint, scenario_optimize_per_c,
)
mat   = load_mat(MAT_PATH)
model = FNOValueModel(load_fno(CKPT_PATH, device), device=device)
c_list = [mat["constraints"][i] for i in range(10)]
prob   = build_problem_from_mat(mat, c_list=c_list)
out    = scenario_optimize_per_c(model, prob, c_list, eps=1e-2, beta=1e-9)

delta_k = out["per_c"][k]["delta_hat"]
cache   = CachedConstraint(c_list[k], prob, model)
def in_recovered_set(s):                  # s: (B, 3) (x, y, theta)
    return cache.value_at_full_BRS(s) < delta_k
```

---

## 10. Source map

- Problem dataclass: `DubinsReachProblem`.
- FNO wrapper + per-obstacle cache: `FNOValueModel.predict_full`, `CachedConstraint`.
- DubinsCar2 induced policy: `rollout_cost` → `DubinsCar2.optCtrl_inPython` /
  `optDstb_inPython` / `dynamics_inPython` per sample.
- Rejection sampler: `sample_x_under_delta` (3-D box).
- Per-obstacle / joint loops: `scenario_optimize_per_c`, `scenario_optimize_joint`.
- Dispatcher + CLI: `scenario_optimize_reach`, `__main__`.
- Obstacle resampler (`joint`): `make_obstacle_sampler` → data-gen
  `random_obstacle_set`.
- Visualization worker: [_viz_worker.py](../verfication/_viz_worker.py).

---

## 11. One-line summary

> Per obstacle SDF: cache the FNO value over $(x,y,\theta,t)$ (one pass,
> $\theta$-swept) and its 3-D spatial gradient; starting from $\delta=0$ (inside
> the BRT), sample states under $\{\tilde V(\cdot,\text{fg},c)<\delta\}$, roll out
> under the Dubins bang-bang controller and worst-case disturbance, count
> reach-or-avoid failures ($J\ge 0$), and **damp** $\delta$ toward the worst
> violator value (clamped at `delta_floor`) until a batch is clean; then report
> the empirical failure rate on a fresh batch.
