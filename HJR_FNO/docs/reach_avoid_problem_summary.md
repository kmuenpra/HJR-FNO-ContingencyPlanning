> Open with VS Code Markdown Preview (`Ctrl+Shift+V`) to view rendered math.

# Dubins Car Reach–Avoid + FNO — Problem & Pipeline Summary

Full context for the current project: a Hamilton–Jacobi (HJ) **reach-avoid**
problem for a 3-state Dubins car, whose backward reachable tube (BRT) is learned
by a **5-channel Fourier Neural Operator** so the tube can be queried for many
obstacle layouts (and any heading) without re-running the PDE solver.

Source files:
- Data generation / ground-truth solver: [../data_gen/dubins3D_data_gen.py](../data_gen/dubins3D_data_gen.py)
- FNO model + query: [../HJR_FNO3d.py](../HJR_FNO3d.py)
- Training notebook: [../training/HJRNO_training_Plane3D_xyt.ipynb](../training/HJRNO_training_Plane3D_xyt.ipynb)
- Dynamics class: [../../optimized_dp/odp/dynamics/DubinsCar2.py](../../optimized_dp/odp/dynamics/DubinsCar2.py)
- Reference example: [../../optimized_dp/examples/dubins_3d_reach_avoid.py](../../optimized_dp/examples/dubins_3d_reach_avoid.py)

---

## 1. Dynamics

State $s = (x, y, \theta)$ — planar position and heading. Controls $u = (v, \omega)$
(speed, yaw-rate); disturbance $d = (d_1, d_2, d_3)$ (wind):

$$
\begin{aligned}
\dot x &= v\cos\theta + d_1, \\
\dot y &= v\sin\theta + d_2, \\
\dot\theta &= \omega + d_3.
\end{aligned}
$$

Bounds and game sense ([dubins3D_data_gen.py:32](../data_gen/dubins3D_data_gen.py#L32)):

$$
v \in [0, 1], \quad \omega \in [-1, 1], \quad \lvert d_i\rvert \le 0.1, \quad
\texttt{uMode}=\min,\ \ \texttt{dMode}=\max.
$$

Control **minimizes** the value (drives toward the target while avoiding the
obstacle); disturbance is adversarial (**maximizes**). Both are bang-bang in the
Dubins Hamiltonian.

---

## 2. Grid

A single fixed grid over $(x, y, \theta)$ ([dubins3D_data_gen.py:22](../data_gen/dubins3D_data_gen.py#L22)):

$$
\boxed{X = [-10, 10]\times[-10, 10]\times[-\pi, \pi], \quad
(N_x, N_y, N_\theta) = (50, 50, 36),\ \ \theta \text{ periodic.}}
$$

Time horizon ([dubins3D_data_gen.py:55](../data_gen/dubins3D_data_gen.py#L55)):
lookback $T_{\max} = 8$ s, step $\Delta t = 0.5$ s, so
$\tau = (0, 0.5, \dots, 8)$ has $T = 17$ slices.

---

## 3. Target set $\ell$, obstacle set $g$, and the BRT

All sets are stored as implicit surfaces (negative inside). With
`quadratic=False`, they are **linear signed distances** (range scales with
distance, not distance²), which keeps value magnitudes small.

**Target** $\ell$ — fixed across all samples ([dubins3D_data_gen.py:45](../data_gen/dubins3D_data_gen.py#L45)):
a cylinder at the origin, radius $2$, infinite in $\theta$ (`ignore_dims=[2]`):

$$
\ell(x,y) = \sqrt{x^2 + y^2} - 2, \qquad \{\ell \le 0\} = \text{goal disk}.
$$

**Obstacle** $g$ — random per sample ([dubins3D_data_gen.py:68](../data_gen/dubins3D_data_gen.py#L68)):
the union of $n \sim \mathcal{U}\{1,\dots,5\}$ cylinders (infinite in $\theta$),
each with radius $r \sim \mathcal{U}[0.5, 2]$, center drawn in $[-10,10]^2$ but
kept $\ge 2$ from the origin (rejection-sampled) so the goal stays clear:

$$
g(x,y) = \min_{k} \Bigl(\sqrt{(x-c^k_x)^2 + (y-c^k_y)^2} - r_k\Bigr),
\qquad \{g \le 0\} = \text{obstacle}.
$$

**Reach-avoid value.** For trajectory $\xi$ under optimal control / worst
disturbance, the value is

$$
V(s, t) = \min_{u(\cdot)}\max_{d(\cdot)}\;
\min_{\tau \in [0,t]} \max\!\Bigl(\ell(\xi(\tau)),\ \max_{\sigma \le \tau} g(\xi(\sigma))\Bigr),
$$

and the **backward reachable tube** is the sub-zero level set

$$
\boxed{\ \mathrm{BRT}(t) = \{\, s : V(s,t) \le 0 \,\}\ } \quad
(\text{reach the goal within } t \text{ while never entering the obstacle}).
$$

**Solver.** `HJSolver` is initialized with $V_0 = \max(\ell, -g)$ and steps the
reach-avoid HJ variational inequality with modes
`TargetSetMode="minVWithV0"`, `ObstacleSetMode="maxVWithObstacle"`
([dubins3D_data_gen.py:137](../data_gen/dubins3D_data_gen.py#L137)).
Time convention (odp): slice index $0$ = fully-grown BRT (full lookback),
index $-1$ = terminal/target slice $\max(\ell, -g)$.

Each generated sample stores:

$$
\texttt{constraints} \in \mathbb{R}^{M\times N_x\times N_y\times N_\theta}, \qquad
\texttt{results} \in \mathbb{R}^{M\times N_x\times N_y\times N_\theta\times T}.
$$

---

## 4. FNO model architecture

A 4-block **3-D Fourier Neural Operator** ([../HJR_FNO3d.py](../HJR_FNO3d.py)),
spectral-convolving over the grid dims $(x, y, t)$.

$$
f_\phi : \underbrace{\mathbb{R}^{N_x\times N_y\times T\times 5}}_{\text{5 input channels}}
\;\longrightarrow\;
\mathbb{R}^{N_x\times N_y\times T\times 1}\ (=V).
$$

- **Lifting** `Conv3d(5 → width)`, pointwise.
- **4 Fourier blocks**: each $u \mapsto \sigma\!\big(\mathcal{K}(u) + W u\big)$, where
  $W$ is a pointwise `Conv3d` and $\mathcal{K}$ is `SpectralConv3d` — a 3-D
  `rfftn`, truncation to the lowest modes, a learned complex multiply on the
  four low-frequency corners $(\pm k_x, \pm k_y, +k_t)$, then `irfftn`.
- **Projection** `Conv3d(width → 1)`.

Hyperparameters ([../training/HJRNO_training_Plane3D_xyt.ipynb](../training/HJRNO_training_Plane3D_xyt.ipynb)):

$$
\text{modes} = (16, 16, 8), \quad \text{width} = 32, \quad
\texttt{in\_channels} = 5 \;\;\Rightarrow\;\; \approx 3.36\times 10^{7}\ \text{params.}
$$

Because spectral convolution is discretization-invariant, the model queries at
**any** $(N_x, N_y, T)$ satisfying the mode minimums $N_x, N_y \ge 32$ and
$T \ge 15$.

---

## 5. Training-data construction

The key modeling choice: **$\theta$ is treated as a constant input channel, not a
grid dimension.** Each (sample, heading) pair becomes one training example, and
the network predicts the value over $(x, y, t)$ at that fixed $\theta$:

$$
\text{input channels } = [\,\text{sdf},\ x,\ y,\ t,\ \theta\,], \qquad
\text{sdf} = -g \ (\text{negated obstacle}).
$$

- Channel 0 is the obstacle SDF (broadcast over $t$); channels 1–3 are the
  coordinate/time grids; channel 4 is the **constant** heading value.
- Output is $V(x, y, t)$ at that heading, shape $(N_x, N_y, T, 1)$.
- With all $N_\theta = 36$ headings and $T = 17$ slices, $N$ samples expand to
  $N \times 36$ examples; split $90\%/10\%$ train/test.
- Data are currently used **raw** (the optional `ChannelScaler` normalization was
  removed for simplicity; coordinates are $\pm 10$, $\theta \in [-\pi,\pi]$,
  $t \in [0,8]$).

**Loss** (data-driven; [../training/HJRNO_training_Plane3D_xyt.ipynb](../training/HJRNO_training_Plane3D_xyt.ipynb)):

$$
\mathcal{L} = \underbrace{\frac{\lVert \hat V - V^*\rVert_2}{\lVert V^*\rVert_2}}_{\text{relative }L_2}
+ \lambda_{\text{bnd}}\,\underbrace{\frac{\sum w\,(\hat V - V^*)^2}{\sum w}}_{w=\exp(-V^{*2}/2\sigma^2),\ \text{focus on }\{V=0\}}
+ \lambda_{\text{term}}\,\underbrace{\big\lVert \hat V|_{\text{target slice}} - \max(\ell, -g)\big\rVert^2}_{\text{terminal anchor}}.
$$

with $\lambda_{\text{bnd}} = 1$, $\lambda_{\text{term}} = 10^{-4}$, $\sigma = 1$.

**No PDE residual.** A Hamilton–Jacobi residual term is intentionally omitted:
the Dubins HJ-VI needs $\partial V/\partial\theta$, but $\theta$ is a fixed
constant channel here, so a single example has no neighboring headings to
finite-difference. Training is therefore purely supervised against the odp
solver output, with the boundary-weighted term concentrating accuracy on the
decision-relevant level set $\{V = 0\}$.

---

## 6. Inference / query

Given any obstacle SDF, `query_FNO3d_full` ([../HJR_FNO3d.py](../HJR_FNO3d.py))
sweeps all headings (batched) and stacks the per-$\theta$ predictions into the
full 4-D tube

$$
\hat V \in \mathbb{R}^{N_x\times N_y\times N_\theta\times T},
$$

recovering the learned BRT $\{\hat V \le 0\}$ over position, heading, and
lookback time — the object the downstream contingency-planning task consumes.
