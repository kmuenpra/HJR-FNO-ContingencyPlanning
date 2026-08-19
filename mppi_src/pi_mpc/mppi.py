"""
Kohei Honda, 2023.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from scipy.optimize import brentq, minimize_scalar
from torch.distributions.multivariate_normal import MultivariateNormal


class MPPI(nn.Module):
    """Model Predictive Path Integral Control (MPPI) solver.

    Reference:
        Williams et al., "Information Theoretic MPC for Model-Based
        Reinforcement Learning", IEEE T-RO, 2017.
    """

    def __init__(
        self,
        horizon: int,
        num_samples: int,
        dim_state: int,
        dim_control: int,
        dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        cost_func: Callable[[torch.Tensor, torch.Tensor, Dict], torch.Tensor],
        u_min: torch.Tensor,
        u_max: torch.Tensor,
        sigmas: torch.Tensor,
        lambda_: float | str,
        lbps_delta: float = 0.01,
        essps_target_ess: Optional[float] = None,
        lambda_min: float = 0.01,
        lambda_max: float = 10.0,
        exploration: float = 0.0,
        use_rbr: bool = False,
        constraint_func: Optional[
            Callable[[torch.Tensor, Dict], torch.Tensor]
        ] = None,
        prev_group_frac: float = 0.4,
        use_sg_filter: bool = False,
        sg_window_size: int = 5,
        sg_poly_order: int = 3,
        device=torch.device("cuda"),
        dtype=torch.float32,
        seed: int = 42,
    ) -> None:
        """Initialize MPPI controller.

        Args:
            horizon: Predictive horizon length (number of timesteps).
            num_samples: Number of trajectory samples to generate.
            dim_state: Dimension of state vector.
            dim_control: Dimension of control input vector.
            dynamics: Dynamics model function: (state, action) -> next_state.
            cost_func: Cost function: (state, action, info) -> cost.
            u_min: Minimum control bounds, shape (dim_control,).
            u_max: Maximum control bounds, shape (dim_control,).
            sigmas: Noise standard deviation for each control dimension,
                shape (dim_control,).
            lambda_: Temperature parameter (float) or auto-tuning method
                ('MPO', 'LBPS', 'ESSPS').

        Auto-lambda args:
            lbps_delta: Confidence parameter for LBPS (0 < delta < 1).
                Higher values give tighter bounds. Default: 0.01.
            essps_target_ess: Target effective sample size for ESSPS.
                Default: num_samples / 10.
            lambda_min: Minimum lambda for optimization bounds. Default: 0.01.
            lambda_max: Maximum lambda for optimization bounds. Default: 10.0.

        Sampling args:
            exploration: Fraction of purely random samples (0 to 1).
                Default: 0.0 (all samples inherit from previous solution).

        Robust-pipeline args (use_rbr is the MASTER switch: with use_rbr=False
        this is plain Williams-2017 MPPI and every feature below is inert,
        including set_group_means()):
            use_rbr: Propagate the batch one step at a time and, after each step,
                resample any constraint-violating particle onto a uniformly
                chosen feasible one (donor state/control prefix copied; the
                violator's own future noise kept). Keeps all rollouts inside the
                feasible set. Also enables guided multi-group sampling and the
                safety filter on control selection. Requires constraint_func.
                Default: False (plain MPPI, behavior byte-identical to no-RBR).
            constraint_func: hard feasibility indicator
                (state (num_samples, dim_state), info) -> bool (num_samples,),
                True where the state is feasible. Only used when use_rbr=True.

        Guided multi-group args (only active after set_group_means(means)):
            prev_group_frac: fraction of the sample budget reserved for the
                previous-MPPI-mean group (the LAST group); the remaining budget
                is split equally across the M ancillary means. Default: 0.4.

        Filtering args:
            use_sg_filter: Apply Savitzky-Golay filter for smoothing.
                Default: False.
            sg_window_size: Window size for SG filter (must be odd).
                Larger values give smoother output. Default: 5.
            sg_poly_order: Polynomial order for SG filter.
                Smaller values give smoother output. Default: 3.

        Device args:
            device: Torch device ('cuda' or 'cpu'). Default: 'cuda'.
            dtype: Torch data type. Default: torch.float32.
            seed: Random seed for reproducibility. Default: 42.
        """

        super().__init__()

        # torch seed
        torch.manual_seed(seed)

        # check dimensions
        assert u_min.shape == (dim_control,)
        assert u_max.shape == (dim_control,)
        assert sigmas.shape == (dim_control,)
        # assert num_samples % batch_size == 0 and num_samples >= batch_size

        # device and dtype
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype

        # set parameters
        self._horizon = horizon
        self._num_samples = num_samples
        self._dim_state = dim_state
        self._dim_control = dim_control
        self._dynamics = dynamics
        self._cost_func = cost_func
        self._u_min = u_min.clone().detach().to(self._device, self._dtype)
        self._u_max = u_max.clone().detach().to(self._device, self._dtype)
        self._sigmas = sigmas.clone().detach().to(self._device, self._dtype)
        self._exploration = exploration
        self._use_rbr = use_rbr
        self._constraint_func = constraint_func
        # Single master gate for the whole robust pipeline: resampling-based
        # rollouts, group-confined resampling, guided multi-group sampling and
        # the control-selection safety filter. When this is False the solver is
        # plain Williams-2017 MPPI and set_group_means() is a no-op, so nothing a
        # caller does can change the simplest path.
        self._robust = bool(use_rbr) and constraint_func is not None
        # RBR diagnostics (updated in _rollout_cost_rbr) + safety-filter selection
        self._rbr_resample_frac = 0.0
        self._rbr_min_num_safe = num_samples
        self._rbr_num_allunsafe = 0
        self._safety_selection = "mean"  # 'mean' | 'sample' | 'fallback'

        # ---- guided multi-group state (inactive until set_group_means) -------
        # _group_means: (M, horizon, dim_control) ancillary means, or None for
        #   the single-group (plain) case, which stays byte-identical.
        # _group_of: (num_samples,) group index per sample, fixed by the budget
        #   allocation; group G-1 is always the previous-MPPI-mean group.
        # _group_of_eff: group labels AFTER RBR resampling -- a particle reseeded
        #   from a dead group's donor adopts the donor's group, so this is the
        #   label that describes the trajectory actually rolled out.
        self._prev_group_frac = float(prev_group_frac)
        self._group_means: Optional[torch.Tensor] = None
        self._num_groups = 1
        self._group_of = torch.zeros(
            num_samples, dtype=torch.long, device=self._device
        )
        self._group_of_eff = self._group_of
        # diagnostics
        self._group_mass: Optional[torch.Tensor] = None
        self._group_score: Optional[torch.Tensor] = None
        self._group_counts: Optional[torch.Tensor] = None
        self._group_dead: Optional[torch.Tensor] = None
        self._selected_group = 0

        self._use_sg_filter = use_sg_filter
        self._sg_window_size = sg_window_size
        self._sg_poly_order = sg_poly_order

        # noise distribution
        self._covariance = torch.zeros(
            self._horizon,
            self._dim_control,
            self._dim_control,
            device=self._device,
            dtype=self._dtype,
        )
        self._covariance[:, :, :] = torch.diag(sigmas**2).to(self._device, self._dtype)
        self._inv_covariance = torch.zeros_like(
            self._covariance, device=self._device, dtype=self._dtype
        )
        for t in range(1, self._horizon):
            self._inv_covariance[t] = torch.inverse(self._covariance[t])

        zero_mean = torch.zeros(dim_control, device=self._device, dtype=self._dtype)
        self._noise_distribution = MultivariateNormal(
            loc=zero_mean, covariance_matrix=self._covariance
        )

        self._sample_shape = torch.Size([self._num_samples])

        # sampling with reparameting trick
        self._action_noises = self._noise_distribution.rsample(
            sample_shape=self._sample_shape
        )

        zero_mean_seq = torch.zeros(
            self._horizon, self._dim_control, device=self._device, dtype=self._dtype
        )
        self._perturbed_action_seqs = torch.clamp(
            zero_mean_seq + self._action_noises, self._u_min, self._u_max
        )

        self._previous_action_seq = zero_mean_seq

        # Initialize Savitzky-Golay filter
        self._coeffs = self._savitzky_golay_coeffs(
            self._sg_window_size, self._sg_poly_order
        )
        self._actions_history_for_sg = torch.zeros(
            self._horizon - 1, self._dim_control, device=self._device, dtype=self._dtype
        )  # previous inputted actions for sg filter

        # inner variables
        self._state_seq_batch = torch.zeros(
            self._num_samples,
            self._horizon + 1,
            self._dim_state,
            device=self._device,
            dtype=self._dtype,
        )
        self._weights = torch.zeros(
            self._num_samples, device=self._device, dtype=self._dtype
        )
        self._optimal_state_seq = torch.zeros(
            self._horizon + 1, self._dim_state, device=self._device, dtype=self._dtype
        )

        # auto lambda tuning
        self._lambda: float | str = lambda_
        self._lbps_delta = lbps_delta
        self._essps_target_ess = (
            essps_target_ess if essps_target_ess is not None else num_samples / 10
        )
        self._lambda_min = lambda_min
        self._lambda_max = lambda_max

        if self._lambda == "MPO":
            self._auto_lambda = "MPO"
            self._lambda = 1.0  # initial value
            self._mpo_epsilon = 0.1
            self.log_temperature = torch.nn.Parameter(
                torch.log(
                    torch.tensor([self._lambda], device=self._device, dtype=self._dtype)
                )
            )
            self.optimizer = torch.optim.Adam([self.log_temperature], lr=0.2)
        elif self._lambda == "LBPS":
            self._auto_lambda = "LBPS"
        elif self._lambda == "ESSPS":
            self._auto_lambda = "ESSPS"
        elif isinstance(self._lambda, float):
            self._auto_lambda = None
        else:
            raise ValueError(
                "lambda_ must be 'MPO', 'LBPS', 'ESSPS', or a float value."
            )

    def reset(self):
        """
        Reset the previous action sequence.
        """
        self._previous_action_seq = torch.zeros(
            self._horizon, self._dim_control, device=self._device, dtype=self._dtype
        )
        self._actions_history_for_sg = torch.zeros(
            self._horizon - 1, self._dim_control, device=self._device, dtype=self._dtype
        )  # previous inputted actions for sg filter

    def set_group_means(self, means: Optional[torch.Tensor]) -> None:
        """Install M ancillary proposal means for guided multi-group sampling.

        Call once per control step, BEFORE forward(). The M means (e.g. control
        sequences tracking M homotopy-distinct topo-PRM paths) become groups
        0..M-1; the previous-MPPI mean is always the LAST group (index M), so
        there are G = M+1 groups. The sample budget is split by
        _allocate_groups, each sample is perturbed around ITS OWN group's mean,
        RBR resampling is confined within groups, and control selection is
        winner-take-all over groups (see forward()).

        No-op unless use_rbr=True (with a constraint_func): guided sampling is
        part of the robust pipeline, so plain MPPI cannot be perturbed by it.

        Args:
            means: (M, horizon, dim_control) tensor, or None to return to plain
                single-group MPPI (byte-identical to never calling this).
        """
        if not self._robust:
            return
        if means is None:
            if self._num_groups != 1:
                self._num_groups = 1
                self._group_of = torch.zeros(
                    self._num_samples, dtype=torch.long, device=self._device
                )
                self._group_of_eff = self._group_of
            self._group_means = None
            return

        means = torch.as_tensor(means, device=self._device, dtype=self._dtype)
        assert means.ndim == 3 and means.shape[1:] == (
            self._horizon,
            self._dim_control,
        ), f"group means must be (M, {self._horizon}, {self._dim_control})"
        assert self._exploration == 0.0, (
            "exploration>0 replaces a tail slice of the batch with raw zero-mean "
            "noise, which would silently overwrite the last group; unsupported "
            "with group means."
        )

        self._group_means = means
        num_groups = means.shape[0] + 1
        if num_groups != self._num_groups:
            self._num_groups = num_groups
            self._group_of = self._allocate_groups(num_groups)
        self._group_of_eff = self._group_of

    def _allocate_groups(self, num_groups: int) -> torch.Tensor:
        """Sample budget -> per-sample group label, shape (num_samples,).

        The previous-MPPI-mean group (LAST index) gets prev_group_frac of the
        budget; the rest is split as equally as possible across the ancillary
        means, with the remainder handed to the earliest groups. Labels are
        contiguous blocks, which keeps the index_select in sampling cheap.
        """
        n = self._num_samples
        if num_groups <= 1:
            return torch.zeros(n, dtype=torch.long, device=self._device)

        n_prev = int(round(self._prev_group_frac * n))
        # every group needs at least one sample (donor pool + weight mass)
        n_prev = max(1, min(n_prev, n - (num_groups - 1)))
        rest, num_anc = n - n_prev, num_groups - 1
        base, rem = divmod(rest, num_anc)
        counts = [base + (1 if i < rem else 0) for i in range(num_anc)] + [n_prev]
        return torch.repeat_interleave(
            torch.arange(num_groups, device=self._device),
            torch.tensor(counts, device=self._device),
        )

    def forward(
        self, state: torch.Tensor, info: Dict = {}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve the optimal control problem for one timestep.

        Algorithm steps:
            1. Sample action sequences around previous solution
            2. Rollout dynamics for all samples in parallel
            3. Compute trajectory costs (stage + terminal)
            4. Update temperature parameter (if auto-lambda enabled)
            5. Compute importance weights using softmax
            6. Compute optimal action as weighted average
            7. Apply smoothing filter (if enabled)

        Args:
            state: Current state vector, shape (dim_state,).
            info: Optional dictionary for passing additional context
                to the cost function (e.g., reference trajectory).

        Returns:
            Tuple of (optimal_action_seq, optimal_state_seq):
                - optimal_action_seq: Shape (horizon, dim_control)
                - optimal_state_seq: Shape (horizon + 1, dim_state)
        """
        assert state.shape == (self._dim_state,)

        if not torch.is_tensor(state):
            state = torch.tensor(state, device=self._device, dtype=self._dtype)
        else:
            if state.device != self._device or state.dtype != self._dtype:
                state = state.to(self._device, self._dtype)

        mean_action_seq = self._previous_action_seq.clone().detach()

        # =================================================================
        # Step 1: Sample action sequences
        # =================================================================
        # Generate noise samples using reparametrization trick
        self._action_noises = self._noise_distribution.rsample(
            sample_shape=self._sample_shape
        )

        if self._group_means is None:
            # ---- plain single-mean sampling (unchanged) ----
            # Split samples: (1-exploration) inherit from previous, rest are random
            threshold = int(self._num_samples * (1 - self._exploration))
            inherited_samples = mean_action_seq + self._action_noises[:threshold]
            self._perturbed_action_seqs = torch.cat(
                [inherited_samples, self._action_noises[threshold:]]
            )
        else:
            # ---- guided multi-group sampling ----
            # Groups 0..M-1 are the ancillary (topo-PRM) means, group M is the
            # previous-MPPI mean. Each sample is perturbed around ITS OWN mean.
            all_means = torch.cat(
                [self._group_means, mean_action_seq.unsqueeze(0)], dim=0
            )  # (G, horizon, dim_control)
            means_per_sample = all_means.index_select(0, self._group_of)
            self._perturbed_action_seqs = means_per_sample + self._action_noises

        # Enforce control limits
        self._perturbed_action_seqs = torch.clamp(
            self._perturbed_action_seqs, self._u_min, self._u_max
        )

        # =================================================================
        # Step 2 & 3: Rollout dynamics and compute trajectory costs
        # =================================================================
        # effective group labels default to the allocation; RBR may relabel
        # particles it reseeds across groups (see _rollout_cost_rbr)
        self._group_of_eff = self._group_of
        self._group_dead = None
        if self._robust:
            # Resampling-Based Rollouts: step-wise rollout that resamples any
            # constraint-violating particle onto a feasible one. Advertise the
            # hard constraint so the cost can skip a redundant feasibility term.
            info["rbr_active"] = True
            costs = self._rollout_cost_rbr(state, info)
        else:
            # ---- plain MPPI (unchanged) ----
            self._state_seq_batch[:, 0, :] = state.repeat(self._num_samples, 1)

            for t in range(self._horizon):
                self._state_seq_batch[:, t + 1, :] = self._dynamics(
                    self._state_seq_batch[:, t, :],
                    self._perturbed_action_seqs[:, t, :],
                )

            costs = torch.zeros(
                self._num_samples, self._horizon, device=self._device, dtype=self._dtype
            )
            action_costs = torch.zeros(
                self._num_samples, self._horizon, device=self._device, dtype=self._dtype
            )
            initial_state = self._state_seq_batch[:, 0, :]
            for t in range(self._horizon):
                prev_index = t - 1 if t > 0 else 0
                prev_state = self._state_seq_batch[:, prev_index, :]
                prev_action = self._perturbed_action_seqs[:, prev_index, :]
                # info update
                info["prev_state"] = prev_state
                info["prev_action"] = prev_action
                info["initial_state"] = initial_state
                info["t"] = t
                costs[:, t] = self._cost_func(
                    self._state_seq_batch[:, t, :],
                    self._perturbed_action_seqs[:, t, :],
                    info,
                )
                action_costs[:, t] = (
                    mean_action_seq[t]
                    @ self._inv_covariance[t]
                    @ self._perturbed_action_seqs[:, t].T
                )

            prev_state = self._state_seq_batch[:, -2, :]
            info["prev_state"] = prev_state
            zero_action = torch.zeros(
                self._num_samples,
                self._dim_control,
                device=self._device,
                dtype=self._dtype,
            )
            terminal_costs = self._cost_func(
                self._state_seq_batch[:, -1, :], zero_action, info
            )
            # In the original paper, the action cost is added to consider KL div. penalty,
            # but it is easier to tune without it
            # please see: https://arxiv.org/abs/2511.08019
            costs = (
                torch.sum(costs, dim=1) + terminal_costs
                # + torch.sum(self._lambda * action_costs, dim=1)
            )

        # =================================================================
        # Step 4: Update temperature (auto-lambda)
        # =================================================================
        if self._auto_lambda == "LBPS":
            # Lower-Bound Policy Search
            # ref: https://openreview.net/forum?id=HbGgF93Ppoy
            result = minimize_scalar(
                lambda lambda_: self._lbps_objective(lambda_, costs.detach()),
                bounds=(self._lambda_min, self._lambda_max),
                method="bounded",
            )
            self._lambda = result.x

        elif self._auto_lambda == "ESSPS":
            # Effective Sample Size Policy Search
            # ref: https://openreview.net/forum?id=HbGgF93Ppoy
            ess_at_min = self._compute_ess(
                torch.softmax(-costs.detach() / self._lambda_min, dim=0)
            )
            ess_at_max = self._compute_ess(
                torch.softmax(-costs.detach() / self._lambda_max, dim=0)
            )

            if self._essps_target_ess <= ess_at_min:
                self._lambda = self._lambda_min
            elif self._essps_target_ess >= ess_at_max:
                self._lambda = self._lambda_max
            else:
                self._lambda = brentq(
                    lambda lambda_: self._essps_objective(lambda_, costs.detach()),
                    self._lambda_min,
                    self._lambda_max,
                )

        # =================================================================
        # Step 5: Compute importance weights
        # =================================================================
        # w_i = softmax(-cost_i / lambda) = exp(-cost_i/lambda) / sum(exp(-cost_j/lambda))
        self._weights = torch.softmax(-costs / self._lambda, dim=0)

        # =================================================================
        # Step 6: Compute optimal action as weighted average
        # =================================================================
        # Weighted mean over ALL trajectories from ALL groups. With group means
        # active this is a mixture-proposal average, so it may fall between
        # homotopy classes and leave the (non-convex) safe set -- that is exactly
        # what the safety filter in Step 8 checks and repairs.
        optimal_action_seq = torch.sum(
            self._weights.view(self._num_samples, 1, 1) * self._perturbed_action_seqs,
            dim=0,
        )

        if self._group_means is not None:
            # DIAGNOSTICS ONLY (not used for control): per-group weight mass, and
            # mass PER SAMPLE. Raw mass scales with a group's sample budget, so
            # with prev_group_frac=0.4 the previous-mean group has a ~4x head
            # start at M=6; the normalized score is the budget-free comparison of
            # how good a TYPICAL sample of each mode is. `_selected_group` is the
            # argmax of that score and is used only for reporting / the overlay.
            groups = self._group_of_eff
            mass = torch.zeros(
                self._num_groups, device=self._device, dtype=self._dtype
            )
            mass.index_add_(0, groups, self._weights)
            counts = torch.bincount(groups, minlength=self._num_groups)
            score = mass / counts.clamp(min=1).to(self._dtype)
            score = torch.where(counts > 0, score, torch.full_like(score, -1.0))
            self._group_mass = mass
            self._group_score = score
            self._group_counts = counts
            self._selected_group = int(torch.argmax(score))

        mean_action_seq = optimal_action_seq

        if self._auto_lambda == "MPO":
            # auto-tune temperature parameter
            # Refer E step of MPO algorithm:
            # https://arxiv.org/pdf/1806.06920
            for _ in range(1):
                self.optimizer.zero_grad()
                temperature = torch.nn.functional.softplus(self.log_temperature)
                cost_logsumexp = torch.logsumexp(-costs / temperature, dim=0)
                loss = temperature * (self._mpo_epsilon + torch.mean(cost_logsumexp))
                loss.backward()
                self.optimizer.step()
            self._lambda = torch.exp(self.log_temperature).item()

        # calculate new covariance
        # https://arxiv.org/pdf/2104.00241
        # covariance = torch.sum(
        #     self._weights.view(self._num_samples, 1, 1)
        #     * (self._perturbed_action_seqs - optimal_action_seq) ** 2,
        #     dim=0,
        # )  # T x dim_control

        # small_cov = 1e-6 * torch.eye(
        #     self._dim_control, device=self._device, dtype=self._dtype
        # )
        # self._covariance = torch.diag_embed(covariance) + small_cov

        # for t in range(1, self._horizon):
        #     self._inv_covariance[t] = torch.inverse(self._covariance[t])
        # zero_mean = torch.zeros(self._dim_control, device=self._device, dtype=self._dtype)
        # self._noise_distribution = MultivariateNormal(
        #     loc=zero_mean, covariance_matrix=self._covariance
        # )

        # =================================================================
        # Step 7: Apply smoothing filter (optional)
        # =================================================================
        if self._use_sg_filter:
            # Concatenate history with new optimal sequence for filtering
            prolonged_action_seq = torch.cat(
                [
                    self._actions_history_for_sg,
                    optimal_action_seq,
                ],
                dim=0,
            )

            # Apply SG filter for each control dimension
            filtered_action_seq = torch.zeros_like(
                prolonged_action_seq, device=self._device, dtype=self._dtype
            )
            for i in range(self._dim_control):
                filtered_action_seq[:, i] = self._apply_savitzky_golay(
                    prolonged_action_seq[:, i], self._coeffs
                )

            # Extract filtered horizon-length sequence
            optimal_action_seq = filtered_action_seq[-self._horizon :]

        # =================================================================
        # Step 8: Control selection (safety filter), predict, update history
        # =================================================================
        # Warm-start next iteration with the weighted MEAN (unchanged).
        self._previous_action_seq = optimal_action_seq

        # Safety filter is active ONLY in the robust pipeline: the weighted mean
        # over ALL groups can leave the (non-convex) safe set, so the EXECUTED
        # sequence is chosen separately. Without RBR this is a no-op and the
        # executed sequence IS the weighted mean, so the plain path is
        # byte-identical to Williams-2017.
        if self._robust:
            executed_seq = self._safety_filter(state, optimal_action_seq, costs)
        else:
            executed_seq = optimal_action_seq

        expanded_optimal_action_seq = executed_seq.repeat(1, 1, 1)
        optimal_state_seq = self._states_prediction(state, expanded_optimal_action_seq)

        # Update action history for Savitzky-Golay filter (with executed action)
        optimal_action = executed_seq[0]
        self._actions_history_for_sg = torch.cat(
            [self._actions_history_for_sg[1:], optimal_action.view(1, -1)]
        )

        return executed_seq, optimal_state_seq

    def _rollout_cost_rbr(self, state: torch.Tensor, info: Dict) -> torch.Tensor:
        """Resampling-Based Rollouts: step-wise rollout with mid-rollout
        resampling of constraint-violating particles.

        At each step: accumulate the stage cost, propagate one step, evaluate the
        hard constraint on the newly reached state, then replace each infeasible
        particle by a UNIFORMLY chosen feasible donor -- copying only the donor's
        state/control PREFIX (and accumulated cost) while KEEPING the violator's
        own future controls (suffix noise). Mutates self._state_seq_batch and
        self._perturbed_action_seqs in place; returns the summed cost (N,)."""
        N, T, device, dtype = (
            self._num_samples,
            self._horizon,
            self._device,
            self._dtype,
        )

        self._state_seq_batch[:, 0, :] = state.repeat(N, 1)
        cost_acc = torch.zeros(N, device=device, dtype=dtype)
        initial_state = self._state_seq_batch[:, 0, :]
        all_idx = torch.arange(N, device=device)
        total_resampled = 0
        min_num_safe = N
        num_allunsafe = 0

        # Working group labels: a particle that gets reseeded from ANOTHER group
        # (because its own group died) adopts the donor's group, which the
        # gather below does for free. Confined resampling then stays consistent
        # on later steps, and the final labels describe the trajectories that
        # were actually rolled out.
        num_groups = self._num_groups
        group_id = self._group_of.clone()
        group_dead = torch.zeros(num_groups, dtype=torch.bool, device=device)

        for t in range(T):
            x_t = self._state_seq_batch[:, t, :]
            u_t = self._perturbed_action_seqs[:, t, :]

            # stage cost on the current (already-feasible) state, before propagating
            info["prev_state"] = self._state_seq_batch[:, t - 1 if t > 0 else 0, :]
            info["prev_action"] = self._perturbed_action_seqs[:, t - 1 if t > 0 else 0, :]
            info["initial_state"] = initial_state
            info["t"] = t
            cost_acc = cost_acc + self._cost_func(x_t, u_t, info)

            # propagate one step
            x_next = self._dynamics(x_t, u_t)
            self._state_seq_batch[:, t + 1, :] = x_next

            # hard constraint indicator on the newly reached state
            safe = self._constraint_func(x_next, info)
            safe = safe.to(device=device, dtype=torch.bool).view(-1)
            safe_idx = torch.nonzero(safe, as_tuple=False).view(-1)
            num_safe = safe_idx.numel()

            min_num_safe = min(min_num_safe, num_safe)
            if num_safe == 0:
                num_allunsafe += 1

            # Resample only when there is at least one donor AND one violator.
            # num_safe == 0 -> no donor, leave batch untouched (plain-MPPI
            # fallback this step); num_safe == N -> nothing to resample.
            if 0 < num_safe < N:
                resample_idx = all_idx.clone()

                if num_groups == 1:
                    unsafe_idx = torch.nonzero(~safe, as_tuple=False).view(-1)
                    donors = safe_idx[
                        torch.randint(num_safe, (unsafe_idx.numel(),), device=device)
                    ]
                    resample_idx[unsafe_idx] = donors
                    total_resampled += int(unsafe_idx.numel())
                else:
                    # CONFINED resampling: a violator may only clone a survivor
                    # that shares its proposal mean, so a group cannot leak into
                    # another's homotopy class. A group with NO survivors is dead:
                    # it is reseeded from the global survivor pool (its budget is
                    # recycled rather than wasted) and, via the gather, its
                    # particles adopt the donor's group label -- i.e. the dead
                    # homotopy class is removed from the mixture.
                    for gi in range(num_groups):
                        in_g = group_id == gi
                        viol_idx = torch.nonzero(in_g & ~safe, as_tuple=False).view(-1)
                        if viol_idx.numel() == 0:
                            continue
                        pool = torch.nonzero(in_g & safe, as_tuple=False).view(-1)
                        if pool.numel() == 0:
                            group_dead[gi] = True
                            pool = safe_idx  # global reseed (num_safe > 0 here)
                        resample_idx[viol_idx] = pool[
                            torch.randint(pool.numel(), (viol_idx.numel(),), device=device)
                        ]
                        total_resampled += int(viol_idx.numel())

                # copy donor PREFIX only (states x_0..x_{t+1}, controls u_0..u_t,
                # accumulated cost); suffix controls u_{t+1..} are left untouched
                # so each rescued particle keeps its own future noise.
                self._state_seq_batch[:, : t + 2, :] = self._state_seq_batch[
                    resample_idx, : t + 2, :
                ]
                self._perturbed_action_seqs[:, : t + 1, :] = (
                    self._perturbed_action_seqs[resample_idx, : t + 1, :]
                )
                cost_acc = cost_acc[resample_idx]
                group_id = group_id[resample_idx]

        # terminal cost on the final (feasible) state
        info["prev_state"] = self._state_seq_batch[:, -2, :]
        zero_action = torch.zeros(N, self._dim_control, device=device, dtype=dtype)
        terminal_costs = self._cost_func(
            self._state_seq_batch[:, -1, :], zero_action, info
        )

        self._rbr_resample_frac = total_resampled / float(N * T)
        self._rbr_min_num_safe = min_num_safe
        self._rbr_num_allunsafe = num_allunsafe
        self._group_of_eff = group_id
        self._group_dead = group_dead
        return cost_acc + terminal_costs

    def _safety_filter(
        self, state: torch.Tensor, mean_seq: torch.Tensor, costs: torch.Tensor
    ) -> torch.Tensor:
        """Select the control sequence to EXECUTE (robust pipeline only).

        The MPPI weighted mean is taken over all trajectories from ALL groups, so
        it may violate the contingency constraint: the safe control set is
        non-convex, and an average of safe sequences need not be safe. So:

          1. simulate ONE step from the mean and test the value function. If
             V < -delta, execute the mean.
          2. otherwise execute the first control of the lowest-cost SAMPLED
             trajectory that satisfies V < -delta at EVERY step.
          3. if no sample qualifies, fall back to the mean (flagged 'fallback').

        The V < -delta test is exactly `constraint_func`: points_feasible
        thresholds the value at `safe_margin - feasibility_buffer`, and
        safe_margin IS the certified -delta (negative, e.g. -0.6). So there is no
        separate value plumbing and the filter can never disagree with the
        constraint RBR enforced during the rollouts.

        Note on cost: branch 2 evaluates the constraint on all N*(H+1) rollout
        states in one batched call. RBR has already driven most of them feasible,
        so this is a correctness backstop rather than the common path."""
        cf = self._constraint_func

        # (1) one-step check on the weighted mean
        x1 = self._dynamics(state.view(1, -1), mean_seq[0].view(1, -1))
        if bool(cf(x1, {}).to(torch.bool).view(-1)[0]):
            self._safety_selection = "mean"
            return mean_seq

        # (2) lowest-cost sample feasible at every step.
        # Index 0 is the CURRENT state, shared by every sample and not a decision
        # variable, so it is excluded: if the robot is momentarily outside the set
        # (V >= -delta), including it would disqualify all N samples and disarm
        # the filter exactly when it is needed most. Steps 1..H are what the
        # candidate controls actually determine.
        N = self._num_samples
        Hs = self._state_seq_batch.shape[1]
        flat = self._state_seq_batch.reshape(N * Hs, self._dim_state)
        feas = cf(flat, {}).to(torch.bool).view(N, Hs)
        feasible_all = feas[:, 1:].all(dim=1)
        if bool(feasible_all.any()):
            masked = torch.where(
                feasible_all, costs, torch.full_like(costs, float("inf"))
            )
            best = int(torch.argmin(masked))
            self._safety_selection = "sample"
            return self._perturbed_action_seqs[best]

        self._safety_selection = "fallback"
        return mean_seq

    def diagnostics(self) -> str:
        """One-line summary of the last forward(): group selection, per-group
        weight mass and RBR resampling. Empty-ish when nothing is active."""
        parts = []
        if self._group_means is not None and self._group_counts is not None:
            last = self._num_groups - 1
            sel = self._selected_group
            tag = "prev" if sel == last else f"topo{sel}"
            mass = " ".join(f"{m:.2f}" for m in self._group_mass.tolist())
            # 'best' is the diagnostic argmax of mass-per-sample, NOT a control
            # decision -- the executed sequence comes from the safety filter.
            parts.append(f"G={self._num_groups} best={tag} mass=[{mass}]")
            if self._group_dead is not None and bool(self._group_dead.any()):
                dead = torch.nonzero(self._group_dead).view(-1).tolist()
                parts.append(f"dead={dead}")
        if self._robust:
            parts.append(
                f"sel={self._safety_selection} "
                f"resample={100 * self._rbr_resample_frac:.1f}% "
                f"minsafe={self._rbr_min_num_safe} "
                f"allunsafe={self._rbr_num_allunsafe}"
            )
        return " | ".join(parts)

    def get_top_samples(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the top-weighted trajectory samples.

        Retrieves the trajectories with highest importance weights from the
        most recent forward pass. Useful for visualization or debugging.

        Args:
            num_samples: Number of top samples to retrieve.

        Returns:
            Tuple of (top_samples, top_weights):
                - top_samples: Shape (num_samples, horizon + 1, dim_state)
                - top_weights: Shape (num_samples,), sorted descending
        """
        assert num_samples <= self._num_samples

        # large weights are better
        top_indices = torch.topk(self._weights, num_samples).indices

        top_samples = self._state_seq_batch[top_indices]
        top_weights = self._weights[top_indices]

        top_samples = top_samples[torch.argsort(top_weights, descending=True)]
        top_weights = top_weights[torch.argsort(top_weights, descending=True)]

        return top_samples, top_weights

    def get_samples_from_posterior(
        self, optimal_solution: torch.Tensor, state: torch.Tensor, num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert num_samples <= self._num_samples

        # Posterior distribution of MPPI
        # Covariance is the same as noise distribution
        posterior_distribution = MultivariateNormal(
            loc=optimal_solution, covariance_matrix=self._covariance
        )

        # sampling control input sequence from posterior
        samples = posterior_distribution.sample(sample_shape=torch.Size([num_samples]))

        # get state sequence from sampled control input sequence
        predictive_states = self._states_prediction(state, samples)

        return samples, predictive_states

    def _states_prediction(
        self, state: torch.Tensor, action_seqs: torch.Tensor
    ) -> torch.Tensor:
        state_seqs = torch.zeros(
            action_seqs.shape[0],
            self._horizon + 1,
            self._dim_state,
            device=self._device,
            dtype=self._dtype,
        )
        state_seqs[:, 0, :] = state
        # expanded_optimal_action_seq = action_seq.repeat(1, 1, 1)
        for t in range(self._horizon):
            state_seqs[:, t + 1, :] = self._dynamics(
                state_seqs[:, t, :], action_seqs[:, t, :]
            )
        return state_seqs

    def _compute_ess(self, weights: torch.Tensor) -> float:
        """Compute Effective Sample Size (ESS).

        ESS = 1 / sum(weights^2)
        Range: 1 <= ESS <= N
        """
        return 1.0 / torch.sum(weights**2).item()

    def _lbps_objective(self, lambda_: float, costs: torch.Tensor) -> float:
        """LBPS objective function (Lower-Bound Policy Search).

        Maximizes the lower bound of expected return:
        J_LB(λ) = E[R] - penalty
        where penalty = R_range * sqrt((1-δ)/δ) / sqrt(ESS)

        Returns negative value for minimization.
        """
        weights = torch.softmax(-costs / lambda_, dim=0)
        ess = self._compute_ess(weights)

        # Expected return (negative cost)
        expected_return = -torch.sum(weights * costs).item()

        # Penalty term
        cost_range = (costs.max() - costs.min()).item()
        penalty = (
            cost_range
            * math.sqrt((1 - self._lbps_delta) / self._lbps_delta)
            / math.sqrt(ess)
        )

        return -(expected_return - penalty)  # Negative for minimization

    def _essps_objective(self, lambda_: float, costs: torch.Tensor) -> float:
        """ESSPS objective function (Effective Sample Size Policy Search).

        Returns ESS - target_ess for root finding.
        """
        weights = torch.softmax(-costs / lambda_, dim=0)
        ess = self._compute_ess(weights)
        return ess - self._essps_target_ess

    def _savitzky_golay_coeffs(self, window_size: int, poly_order: int) -> torch.Tensor:
        """
        Compute the Savitzky-Golay filter coefficients using PyTorch.

        Parameters:
        - window_size: The size of the window (must be odd).
        - poly_order: The order of the polynomial to fit.

        Returns:
        - coeffs: The filter coefficients as a PyTorch tensor.
        """
        # Ensure the window size is odd and greater than the polynomial order
        if window_size % 2 == 0 or window_size <= poly_order:
            raise ValueError("window_size must be odd and greater than poly_order.")

        # Generate the Vandermonde matrix of powers for the polynomial fit
        half_window = (window_size - 1) // 2
        indices = torch.arange(
            -half_window, half_window + 1, dtype=self._dtype, device=self._device
        )
        A = torch.vander(indices, N=poly_order + 1, increasing=True)

        # Compute the pseudo-inverse of the matrix
        pseudo_inverse = torch.linalg.pinv(A)

        # The filter coefficients are given by the first row of the pseudo-inverse
        coeffs = pseudo_inverse[0]

        return coeffs

    def _apply_savitzky_golay(
        self, y: torch.Tensor, coeffs: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the Savitzky-Golay filter to a 1D signal using the provided coefficients.

        Parameters:
        - y: The input signal as a PyTorch tensor.
        - coeffs: The filter coefficients as a PyTorch tensor.

        Returns:
        - y_filtered: The filtered signal.
        """
        # Pad the signal at both ends to handle the borders
        pad_size = len(coeffs) // 2
        y_padded = torch.cat([y[:pad_size].flip(0), y, y[-pad_size:].flip(0)])

        # Apply convolution
        y_filtered = torch.conv1d(
            y_padded.view(1, 1, -1), coeffs.view(1, 1, -1), padding="valid"
        )

        return y_filtered.view(-1)
