> Open with VS Code Markdown Preview (`Ctrl+Shift+V`) to view rendered math.

# Refactor Plan — `HJR_FNO3d.py` compatible with `rrtx.py`

Goal: turn [HJR_FNO/HJR_FNO3d.py](../HJR_FNO3d.py) into a drop-in replacement for the
old [HJR_FNO/HJR_FNO.py](../HJR_FNO.py), built around the **3-D FNO** (θ as a channel,
spectral conv over $(x,y,t)$) instead of the old 1-D-per-(θ,t) model, while keeping the
exact public surface that [rrtx.py](../../rrtx.py) and [plotting.py](../../plotting.py) consume.

---

## ⭐ HANDOFF — READ FIRST (status as of 2026-06-02)

[HJR_FNO/HJR_FNO3d.py](../HJR_FNO3d.py) is **functionally complete and import/smoke-tested** in the
`rrtx` conda env: the module imports, instantiates, predicts (validated against dataset GT, mean
abs err 0.015), runs feasibility checks, and `update_obs` now certifies a **per-region scenario
safe margin**. The parallel entry point [rrtx_FNO3d.py](../../rrtx_FNO3d.py) is repointed to it.
**Not yet run live end-to-end** (the contingency rollout + plotting loop). Details: §8 / §9.

### ✅ DONE
1. **Model + query** — Classes 1–2 (`SpectralConv3d`, `FNO3d`) unchanged; 5-channel query helpers
   (`_build_input_5ch`, `query_FNO3d`, `query_FNO3d_full`) integrated. Loads
   `training/model/01_FNO3d_dubins_5ch_tuned.pt`. **Channel-0 sign fixed** (§9.1, GT-validated).
2. **odp Classes 3–4** — `Grid` + `DubinsCar2` imported (not redefined), via a thin `PlaneDynamics`
   shim. odp path-injected; `heterocl` stubbed when absent; `FNO3d`/`SpectralConv3d` injected into
   `__main__` for unpickling.
3. **Dual θ-grid** — FNO grid `g` (θ∈[-π,π)×36) + fine grid `g_fine` (θ∈[0,2π)×25). A grid-aware
   `_wrap_to_grid_theta(theta, grid)` keeps every θ lookup in the correct range.
4. **Class 5 ported** — `predict`, `shapeCylinder`, `update_obs`, `check_hj_descent`(+`_grid`),
   `eval_value_at_state`, `compute_time_derivative`, `xs_to_rows`/`ys_to_cols`, `is_feasible`,
   `find_feasible_closest_region`, `smooth_value_function_xy`, `contingency_policy`, + odp-adapted
   derivative helpers (`computeGradients`/`eval_u`/ENO/WENO).
5. **`true_reach_obsFree`** — loaded from the EXACT odp `.mat` (`50_50_25_SDF_no_obs.mat`,
   `BRT_all (50,50,25,33)` → time `[::2]` → `(50,50,25,17)`) on `g_fine`, NOT FNO-predicted.
6. **Per-region `safe_margin`** — now a `list` (one entry per safe region). Every reference indexed
   by region across [HJR_FNO3d.py](../HJR_FNO3d.py) (`is_feasible`, `find_feasible_closest_region`,
   `contingency_policy`), [plotting.py](../../plotting.py), and [rrtx_FNO3d.py](../../rrtx_FNO3d.py).
7. **Scenario certification of `safe_margin`** (§9.4) — heterocl-free port of
   [scenario_optimization_reach.py](../verfication/scenario_optimization_reach.py) (`per_c`). In
   `update_obs`, after a region's set is re-estimated, `_scenario_delta_hat(...)` computes a
   probabilistic `delta_hat` and stores it in `self.safe_margin[region]`. Tunable via constructor
   kwargs `scenario_enable / scenario_eps / scenario_beta / scenario_M` (+ attributes for the rest).
8. **`rrtx_FNO3d.py` imports** repointed to `HJR_FNO3d` (the 4 import lines); `rrtx.py` untouched.

### ▶ NEXT — still to do
1. **Run `rrtx_FNO3d.py` live** (in the `rrtx` env) and **trigger a contingency** (mouse handler
   [rrtx_FNO3d.py:1380](../../rrtx_FNO3d.py#L1380)). First real exercise of `contingency_policy`
   + `plotting.plot_reachable_set`.
2. **🔴 FIX `contingency_policy` time convention** — VERIFIED BUG (§9.3): the FNO tube has
   **index 0 = fully-grown BRT, index −1 = target**, but `contingency_policy`'s `is_in_BRS` /
   `tEarliest` logic treats index 0 as the target ("entered the target"). The binary search and
   the backward-time march are very likely inverted. (User is reviewing `contingency_policy`.)
3. **Speed up scenario certification** — production defaults (`eps=1e-2, beta=1e-9` → N≈4344,
   `M=30`) make each updated region's cert heavy (~tens of s, dominated by the per-sample Python
   rollout loop in `_scenario_rollout_cost`). Options: vectorize the rollout over the batch
   (replace the `for li, gi in ai` loop with array ops), reduce N (larger eps), or cert less often.
4. **`plotting.plot_reachable_set`** not yet run; confirm slice indices look right on the new grids.
5. **`.mat` time-ordering sanity check** when `HJR_sets` mixes the loaded obstacle-free tube
   (`g_fine`) and FNO-predicted tubes (`g`) mid-rollout (slice 0 vs −1 = fully-grown vs terminal).
6. **Optional cleanups**: `is_state_feasible` raises `NotImplementedError` (per-point model, dead
   in rrtx); `contingency_policy_newTest` / `is_feasible_old` NOT ported; the obstacle-free branch
   of `is_feasible` still hard-codes `> 0` rather than the per-region margin.

### How to test (conda env — bare `python3` lacks numpy)
```bash
source ~/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH=/home/kmuenpra/git/HJR-FNO-ContingencyPlanning
conda run -n rrtx python -c "import HJR_FNO.HJR_FNO3d as m; print('ok')"
# Fast scenario smoke (cheap eps/beta/M so update_obs returns quickly):
#   hjr = m.HJR_FNO(env, safe_regions, Tf_reach=6.0, device='cpu',
#                   scenario_eps=0.1, scenario_beta=1e-3, scenario_M=6)
#   hjr.update_obs([(3.0, 0.0, 1.5)]);  print(hjr.safe_margin)
```
Envs available: `rrtx`, `HJR-FNO`, `odp` (only `odp` has `heterocl`; the module stubs it elsewhere).

---

## 0. Constraints / ground rules

- **Classes 1 & 2** (`SpectralConv3d`, `FNO3d`) — already present, keep as-is.
- **FNOQuery function** — *you will provide the correct one* (the notebook's
  `query_FNO3d_full`, sweeping all θ → `(Nx, Ny, Nθ, T)`). **I will not write it until you hand it over.**
- **Classes 3 & 4** (`Grid`, `Plane`) — *import from odp*, do **not** redefine:
  - `from odp.Grid import Grid` → [optimized_dp/odp/Grid/GridProcessing.py:6](../../optimized_dp/odp/Grid/GridProcessing.py#L6)
  - `from odp.dynamics.DubinsCar2 import DubinsCar2` → [optimized_dp/odp/dynamics/DubinsCar2.py:13](../../optimized_dp/odp/dynamics/DubinsCar2.py#L13)
- **Class 5** (`HJR_FNO`) — rewrite to load `HJR_FNO/training/model/01_FNO3d_dubins_5ch_tuned.pt` and use the 3-D query.
- **Ignore** the Overlap / Area utilities (`compute_area_2d`, `compute_overlap_2d`,
  `check_overlap_threshold_translate_phi2`, `build_overlap_index_list`).

---

## 1. Public surface that MUST keep working

Collected from every `hjr_fno.<member>` reference in [rrtx.py](../../rrtx.py) and [plotting.py](../../plotting.py).

**Attributes**

$$
\begin{aligned}
&\texttt{Tf\_reach},\ \texttt{safe\_margin},\ \texttt{num\_safe\_regions},\ \texttt{safe\_regions},\\
&\texttt{grid\_min},\ \texttt{grid\_max},\ \texttt{N},\ \texttt{N\_fine},\\
&\texttt{g},\ \texttt{g\_fine},\quad \texttt{X},\ \texttt{Y},\ \texttt{X\_fine},\ \texttt{Y\_fine},\\
&\texttt{theta\_array},\ \texttt{theta\_array\_fine},\ \texttt{time\_array},\ \texttt{time\_array\_fine},\\
&\texttt{HJR\_sets},\ \texttt{obs\_list},\ \texttt{feasible\_region},\ \texttt{true\_reach\_obsFree},\ \texttt{utils}.
\end{aligned}
$$

**Methods**

$$
\begin{aligned}
&\texttt{update\_obs(obs\_cir)} \\
&\texttt{is\_feasible(v,\ reachable\_set\_constraint)} \quad (v:\ (M,2)\ \text{array}) \\
&\texttt{is\_state\_feasible(robot\_state,\ theta\_array,\ reachable\_set\_constraint)} \\
&\texttt{find\_feasible\_closest\_region(robot\_pose,\ \dots,\ returnList)} \\
&\texttt{contingency\_policy(robot\_state,\ plotting,\ fig,\ ax,\ \dots)} \\
&\texttt{xs\_to\_rows(xs,\ N)},\quad \texttt{ys\_to\_cols(ys,\ N)} \\
&\texttt{predict(\dots)},\quad \texttt{shapeCylinder(\dots)} \ \text{(internal, but used by update\_obs)}
\end{aligned}
$$

`utils` must expose `sensing_radius`, `lidar_detected`, `update_obs`, `obs_boundary`,
`unknown_obs_circle` (same `Utils` object as today — unchanged).

---

## 2. The core API mismatch: odp `Grid` vs the old hand-rolled `Grid`

This is the single biggest source of churn. The old code uses `.N`, `.dim`, `.vs[i]` (1-D),
`.xs` (full meshgrid). The odp `Grid` differs:

| Old (`HJR_FNO.Grid`) | odp `Grid` | Notes |
|---|---|---|
| `Grid(min, max, N, periodic_dims)` | `Grid(minBounds, maxBounds, dims, pts_each_dim, periodicDims)` | **extra `dims` arg**, `dims` is 3rd positional |
| `g.N` (array) | `g.pts_each_dim` | rename everywhere |
| `g.dim` | `g.dims` | rename |
| `g.periodic_dims` | `g.pDim` | rename |
| `g.vs[i]` → 1-D coord array | `g.grid_points[i]` | **odp `g.vs[i]` is reshaped for broadcasting**, not 1-D! |
| `g.xs[i]` → full meshgrid | *(absent)* | must build via `np.meshgrid(...indexing="ij")` |
| `g.axis` | *(absent)* | rebuild locally if needed |

Also: periodic dim handling differs. odp **excludes** the upper bound for periodic dims
($[-\pi,\pi)$, 36 pts). The old code used $[0,2\pi]$ with 25 pts. **The new model was trained on
$\theta\in[-\pi,\pi)$, 36 pts** ([dubins3D_data_gen.py:23](../data_gen/dubins3D_data_gen.py#L23)) — so the new
`theta_array` convention changes (see §6, open question).

**Plan:** add a tiny private adapter so the rest of the class reads cleanly:

```text
self._vs(g, i)   -> g.grid_points[i]          # 1-D axis
self._xs(g)      -> cached np.meshgrid(...)    # full meshgrid, indexing="ij"
```

and keep `self.N` / `self.N_fine` as plain numpy arrays (rrtx/plotting read these directly),
constructing odp grids from them.

---

## 3. `Plane` → `DubinsCar2` mapping

The old `Plane` API and the substitute calls on `DubinsCar2`
([DubinsCar2.py:102-162, 207](../../optimized_dp/odp/dynamics/DubinsCar2.py#L102)):

| Old `Plane` call | `DubinsCar2` equivalent |
|---|---|
| `plane.optCtrl(t, x, deriv, 'min')` | `car.optCtrl_inPython(state, spat_deriv)` → `np.array([speed, ω])` (uMode is an **instance attr**, not an arg) |
| `plane.optDstb(t, x, deriv, 'max')` | `car.optDstb_inPython(state, spat_deriv)` |
| `plane.dynamics(t, x, u, d)` | `car.dynamics_inPython(state, control, disturbance)` |
| `plane.updateState(u, dt, d)` | **no direct match** — `forward()` ignores disturbance. Need a small wrapper. |
| `plane.optCtrl_grid(deriv, θgrid, 'min')` | **no match** — vectorized control. Need a helper (or vectorize `optCtrl_inPython`). |
| `plane.x` (mutable state) | `car.x` exists, but `forward()` returns next state instead of mutating |

**Decisions needed (see §6):**
- DubinsCar2's `forward()` does **not** add disturbance and wraps θ to $[0,2\pi)$. The old
  `updateState` did Euler with disturbance and no wrap. → I'll add a private
  `_euler_step(state, u, d, dt)` helper inside `HJR_FNO` rather than fight the dynamics class.
- `optCtrl_grid` is only used by `check_hj_descent_grid` (currently commented out in
  `update_obs`). If we keep that method, I'll add a local vectorized control helper.

Construct the car to mirror the data-gen settings:

$$
\texttt{DubinsCar2}(uMin=[0,-1],\ uMax=[1,1],\ dMax=[0.1,0.1,0.1],\ uMode=\texttt{"min"},\ dMode=\texttt{"max"})
$$

(Old code used `wMax=1.5`, `dMax=[0,0,0]` — **mismatch** with the trained model's $\pm1$ yaw and wind; see §6.)

---

## 4. Model loading & `predict()` rewrite

**Loading.** Replace the hard-coded paths with:

```text
save_path = Path(__file__).resolve().parent / "training/model/01_FNO3d_dubins_5ch_tuned.pt"
self.model = torch.load(save_path, weights_only=False, map_location=device).eval()
```

The pickle needs `FNO3d` / `SpectralConv3d` importable — they live in **this** module, so unpickling
works once `rrtx.py` imports from `HJR_FNO3d` (see §5).

**`predict()`.** The old method built a `(T·TH, Npts, 5)` tensor and looped a `DataLoader` with the
1-D model. The new 3-D model consumes a *volume* per θ:

$$
f_\phi:\ \mathbb{R}^{N_x\times N_y\times T\times 5}\ \longrightarrow\ \mathbb{R}^{N_x\times N_y\times T\times 1},
\qquad \text{channels}=[\,\text{sdf},x,y,t,\theta\,].
$$

New `predict(sdf_input, theta_hyparam, time_hyparam)` becomes a thin wrapper over **your**
`query_FNO3d_full`, returning the same shape the rest of the code expects:

$$
\boxed{\ \hat V\in\mathbb{R}^{N_x\times N_y\times N_\theta\times T}\ }
$$

so `HJR_sets[i]`, slicing `[...,θ_slice, t_slice]`, and `feasible_region` all keep working unchanged.

---

## 5. Import fixes — in `rrtx_FNO3d.py` (NOT `rrtx.py`)

**Strategy decided:** `rrtx.py` stays on the OLD model; the 3-D model gets a parallel entry point
**[rrtx_FNO3d.py](../../rrtx_FNO3d.py)** (currently an exact copy). Edit the 4 import lines there —
see the table in the HANDOFF section at the top of this doc for the exact before/after.

Notes confirmed during implementation:
- `Grid` is imported in rrtx but **never used** → just drop it (no need to re-import from odp).
- `FNO1d`/`SpectralConv1d` (L45) were only for unpickling the old model → **delete the line**;
  `HJR_FNO3d.__init__` self-injects `FNO3d`/`SpectralConv3d` into `__main__` for unpickling.

---

## 6. Open questions — ✅ ALL RESOLVED (decisions baked into the code)

1. ✅ **FNOQuery**: provided by user; integrated. **Key gotcha:** channel-0 sign (see §9.1).
2. ✅ **θ convention**: adopted $[-\pi,\pi)$ / 36 pts (taken directly from the odp grid).
3. ✅ **`N` / `N_fine`**: both `[50,50,36,17]` (model is resolution-flexible; dual-grid logic
   kept but identical resolution).
4. ✅ **Dynamics**: `v\in[0,1]$, $\omega\in[\pm1]$, `uMode=min`, `dMode=max`; **`dMax=0` in the
   rollout** (no simulated wind — `optCtrl` is disturbance-independent). `wMax=1.0` (not 1.5).
5. ✅ **`true_reach_obsFree`**: loaded from the EXACT odp `.mat`
   (`50_50_25_SDF_no_obs.mat`, time `[::2]`→17) on a **dedicated fine grid** `g_fine`
   (θ∈[0,2π)×25), distinct from the FNO grid `g` (θ∈[-π,π)×36). A grid-aware
   `_wrap_to_grid_theta(theta, grid)` keeps θ lookups in each grid's own range so the two
   resolutions/conventions never conflict. *(Was previously an OOD model prediction — fixed.)*
6. ✅ **`check_hj_descent_grid` / `optCtrl_grid`**: KEPT (vectorized control implemented in the
   `PlaneDynamics` shim). Still commented out at the `update_obs` call site, as before.

---

## 7. Proposed file skeleton (Step 1 deliverable)

Order preserved; signatures kept identical to the old class unless noted. **No bodies yet** for
the parts that depend on your answers (marked ⏳).

```python
# ── imports ──────────────────────────────────────────────────────────────
import os, math, warnings
from pathlib import Path
from typing import Tuple, List, Dict, Union, Iterable, Optional
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from .neural_utils import *

from odp.Grid import Grid                       # Class 3  (imported, not redefined)
from odp.dynamics.DubinsCar2 import DubinsCar2  # Class 4  (imported, not redefined)

# ── Class 1: SpectralConv3d ─────────────────────────────────────  (unchanged)
class SpectralConv3d(nn.Module): ...

# ── Class 2: FNO3d ──────────────────────────────────────────────  (unchanged)
class FNO3d(nn.Module): ...

# ── FNOQuery (3-D) ──────────────────────────────────────────────  ⏳ YOU PROVIDE
def query_FNO3d(...): ...
def query_FNO3d_full(...): ...   # sweep θ -> (Nx, Ny, Nθ, T)

# ── (derivative helpers retained from old file) ─────────────────
def upwindFirstENO2(grid, data, dim): ...
def upwindFirstWENO5(grid, data, dim): ...
def add_ghost_cells(data, dim, stencil): ...
def strip_dim(data, dim, left, right): ...
def computeGradients(grid, data): ...   # uses grid.grid_points / grid.dx (odp)
def eval_u(grid, gradients, x): ...

# ── Class 5: HJR_FNO ────────────────────────────────────────────
class HJR_FNO:
    def __init__(self, env, safe_regions, Tf_reach, device='cuda'): ...   # loads new .pt, builds odp grids, DubinsCar2

    # --- grid adapters ---
    def _axis(self, g, i): ...        # -> g.grid_points[i]
    def _mesh(self, g): ...           # cached np.meshgrid(indexing="ij")

    # --- dynamics helpers (replace Plane methods) ---
    def _opt_ctrl(self, state, deriv): ...      # car.optCtrl_inPython
    def _opt_dstb(self, state, deriv): ...      # car.optDstb_inPython
    def _euler_step(self, state, u, d, dt): ... # Euler w/ disturbance

    # --- prediction ---
    def predict(self, sdf_input, theta_hyparam, time_hyparam): ...   # wraps query_FNO3d_full ⏳
    def shapeCylinder(self, ignoreDims=None, center=None, radius=1.0, g=None): ...
    def update_obs(self, obs_cir: List): ...

    # --- HJ residual / descent ---
    def eval_value_at_state(self, grid, data_slice, x): ...
    def compute_time_derivative(self, grid, closest_idx, t_idx, x, dt): ...
    def check_hj_descent(self, grid, data_safe, closest_idx, t_idx, x, dt): ...
    def check_hj_descent_grid(self, grid, V_raw, obs_sdf, dt): ...   # ⏳ keep? (§6.6)

    # --- coordinate / feasibility ---
    def xs_to_rows(self, xs, N=None): ...
    def ys_to_cols(self, ys, N=None): ...
    def is_state_feasible(self, robot_pose, theta_array, t=None, reachable_set_constraint=True): ...
    def is_feasible(self, v, reachable_set_constraint=True): ...
    def find_feasible_closest_region(self, robot_pose, t=None, use_distance=True, returnList=False): ...

    # --- contingency planning ---
    def smooth_value_function_xy(self, data_union, sigma_xy=1.0): ...
    def contingency_policy(self, robot_state, plotting, fig, ax, showplot=True, special_case=False): ...

# (Overlap / Area utilities intentionally omitted — §0)
# (No __main__ demo unless you want the super-res GIF carried over)
```

---

## 8. Implementation order (after §6 answers)

1. ✅ **Skeleton** (this doc) — signatures + import wiring decided.
2. ✅ Grid/dynamics adapters + `__init__` (grids, car, model load, safe-region bookkeeping).
3. ✅ `predict()` on top of `query_FNO3d_full`; `shapeCylinder`, `update_obs`.
4. ✅ `computeGradients`/`eval_u` ported to odp `Grid` accessors.
5. ✅ Feasibility (`is_feasible`, `find_feasible_closest_region`, `xs_to_rows`,
   `ys_to_cols`) + `feasible_region` build. (`is_state_feasible` raises — needs a
   per-point model; not used by rrtx.)
6. ✅ `check_hj_descent` (+ `check_hj_descent_grid`) and `contingency_policy`.
7. ✅ `rrtx_FNO3d.py` import edits done (4 lines repointed to `HJR_FNO3d`; AST OK).
8. ✅ **Per-region `safe_margin`** — converted to a `list`; all reference sites indexed by region
   (`HJR_FNO3d.py`, `plotting.py`, `rrtx_FNO3d.py`).
9. ✅ **Scenario certification** of `safe_margin` wired into `update_obs` (§9.4); `eps/beta/M`
   exposed as `HJR_FNO.__init__` kwargs.
10. ⏳ **TODO** run `rrtx_FNO3d.py` live + trigger a contingency; **fix the `contingency_policy`
    time-convention bug** (§9.3); speed up the scenario rollout. (See HANDOFF ▶ NEXT.)

---

## 9. Findings & decisions

Everything below is implemented in [HJR_FNO/HJR_FNO3d.py](../HJR_FNO3d.py) and verified in the
`rrtx` conda env (module import → instantiate → `predict`/`is_feasible`/`update_obs` + scenario cert).

**Decisions taken (from §6 recommendations):** FNO grid `g` = θ∈[-π,π)×36, `N=[50,50,36,17]`;
fine grid `g_fine` = θ∈[0,2π)×25, `N_fine=[50,50,25,17]` (matches the exact obstacle-free
`.mat`); `DubinsCar2(uMin=[0,-1], uMax=[1,1], dMax=[0,0,0], uMode="min", dMode="max")` (rollout
has no simulated wind; `wMax=1.0` matches training); `true_reach_obsFree` **loaded from the exact
odp `.mat`** (`50_50_25_SDF_no_obs.mat`), with a grid-aware `_wrap_to_grid_theta` so the two
θ-conventions never conflict.

**Three things that bit us (now fixed in code):**

1. **Channel-0 sign convention** — the single most important finding.
   `build_dataset` feeds `extract_input(data1) = -constraints` into channel 0, i.e.
   $$\text{network ch}_0 = -\,\text{constraints (negated obstacle SDF)}.$$
   `query_FNO3d_full` computes `ch0 = -constraint_sdf` internally, so its `constraint_sdf`
   argument must be the **original obstacle SDF** `+constraints` (what `shapeCylinder`
   returns, negative inside the obstacle). `predict()` therefore passes `constraint_sdf = sdf`
   (NOT `-sdf`). Validated against dataset GT:
   $$\boxed{\ \text{max abs err}=0.38,\quad \text{mean abs err}=0.015\ }\quad(\text{vs }6.8/1.4\text{ with the wrong sign}).$$
   No `ChannelScaler` is applied — the model is RAW-input (confirmed in the notebook).

2. **`heterocl` dependency** — `DubinsCar2.py` does `import heterocl` at module top, but the
   `rrtx`/`HJR-FNO` envs lack it (only the `odp` env has it). We use only the pure-Python
   methods, so the module installs a `heterocl` stub in `sys.modules` when it's missing.
   Also injects `optimized_dp/` onto `sys.path` (odp is path-injected, not pip-installed).

3. **Pickle `__main__` lookup** — the checkpoint was saved with `FNO3d`/`SpectralConv3d` in
   `__main__`. `__init__` injects them into `__main__` before `torch.load`, so loading works
   from any entry point (this also means rrtx no longer *needs* to import them for unpickling —
   only the `HJR_FNO` import line must change to `HJR_FNO3d`).

4. **Scenario-certified, per-region `safe_margin`** (the safety guarantee). `update_obs` now, after
   re-estimating a region's tube, runs a heterocl-free port of `scenario_optimization_reach.py`
   (`per_c`) to find `delta_hat` s.t. the recovered set `{V(·,fully-grown) < delta_hat}` is — w.h.p.
   `(1−beta)` at violation level `eps` — contained in the true reach-avoid set under the
   FNO-induced bang-bang controller + **worst-case wind** (`scenario_car`, `dMax=0.1`). The one
   heterocl dependency (`computeSpatDerivArray`) is replaced by the numpy `computeGradients`.
   Cost scales with `N = ⌈(2/eps)(ln(1/beta)+1)⌉` (≈4344 at defaults) × `M` rounds; the per-sample
   rollout loop is the bottleneck (see HANDOFF ▶ NEXT #3). Knobs: `scenario_eps/beta/M` (ctor) +
   `scenario_delta_floor/delta_init/step_frac/max_tries/seed/enable` (attrs).

5. **🔴 FNO time convention (verified) — `contingency_policy` is inverted.** Measured `{V<0}`
   fraction: **0.466 at slice 0** vs **0.029 at slice −1** ⇒
   $$\boxed{\ V[\dots,0]=\text{fully-grown BRT (loosest)},\quad V[\dots,-1]=\text{terminal/target}\ }$$
   The scenario code uses this convention correctly. But `contingency_policy`'s `is_in_BRS` +
   `tEarliest` binary search treat **slice 0 as the target** ("Trajectory has entered the target!"),
   which is backwards — its time march is very likely inverted. **Not yet fixed** (under review).

---

### Summary

The heavy lifting is **(a)** the odp-`Grid` attribute renames + meshgrid rebuilds, **(b)** swapping
`Plane` for `DubinsCar2` (with a small Euler/disturbance wrapper since `forward()` drops the
disturbance), and **(c)** re-pointing `predict()` at your 3-D `query_FNO3d_full`. The public API and
return shapes stay identical so `rrtx.py`/`plotting.py` need only their **import lines** changed.
