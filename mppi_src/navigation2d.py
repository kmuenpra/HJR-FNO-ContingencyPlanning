import sys
import time
from pathlib import Path

# import gymnasium
import fire
import numpy as np
import torch
import tqdm

from envs.navigation_2d import Navigation2DEnv
from pi_mpc.mppi import MPPI

'''
Executable command:

source /home/kmuenpra/git/mppi_playground/.venv/bin/activate
    uv run python3 mppi_src/navigation2d.py --save_mode True

'''


def _load_hjr_fno_class():
    """Import the HJR_FNO class from the sibling ``HJR_FNO/`` package.

    ``HJR_FNO3d.py`` uses relative imports (``from .neural_utils import *``,
    ``from .scenario_worker import ...``), so it must be imported as a sub-module
    of the real on-disk ``HJR_FNO`` package — not loaded standalone. Its
    constructor also does absolute imports (``import utils``, which in turn pulls
    in ``env`` / ``plotting`` / ``rrtx``, all living at the repo root), so the
    repo root must be on ``sys.path``. Inserting it first satisfies both: the
    package import resolves ``HJR_FNO`` and the absolute imports resolve ``utils``
    and friends.
    """
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from HJR_FNO.HJR_FNO3d import HJR_FNO

    return HJR_FNO


def main(
    save_mode: bool = False,
    use_hjr: bool = True,
    hjr_reach_time: float = 8.0,
    use_rbr: bool = True,
):
    env = Navigation2DEnv()

    # Optional Hamilton-Jacobi reachability oracle. Construction can fail if the
    # HJR-FNO dependencies (neural_utils.py, scenario_worker.py, utils.py, the
    # trained model and .mat tubes) are not present; in that case we just skip
    # the reachable-set overlay and run the planner as usual.
    hjr_fno = None
    if use_hjr:
        try:
            HJR_FNO = _load_hjr_fno_class()
            hjr_fno = HJR_FNO(
                env=env,
                safe_regions=env.safe_regions,
                Tf_reach=hjr_reach_time,
            )
            # Keep per-region safe-margin certification serial. The spawned
            # scenario workers would re-import the on-disk HJR_FNO package via
            # PYTHONPATH; running serially sidesteps that machinery and is plenty
            # fast for the handful of safe regions here.
            hjr_fno.scenario_parallel = False
            # Use the theta-marginalized feasible_region (max over theta at the
            # Tf_reach slice) for feasibility, so the cost boundary matches the
            # filled feasible-set overlay instead of the heading-sliced tube.
            hjr_fno.feasibility_source = "feasible_region"
        except Exception as e:  # noqa: BLE001 - any import/IO/dependency failure
            print(f"[HJR-FNO] disabled (could not initialize): {e}")
            hjr_fno = None

    # let the env feed lidar-detected obstacles to the oracle inside step()
    if hjr_fno is not None:
        env.attach_reachability(hjr_fno)

    # solver. With RBR enabled, constraint-violating rollouts are resampled onto
    # feasible ones mid-rollout (see MPPI._rollout_cost_rbr) instead of being
    # penalized after the fact, so no samples are wasted outside the reachable
    # set. RBR only applies when a reachability oracle is attached; without one
    # (use_hjr=False or init failure) points_safe still enforces collision-freedom.
    solver = MPPI(
        horizon=30,
        num_samples=3000,
        dim_state=3,
        dim_control=2,
        dynamics=env.dynamics,
        cost_func=env.cost_function,
        u_min=env.u_min,
        u_max=env.u_max,
        # sigmas: per-step control-noise std. NOTE controls are clamped to
        # v in [0,1], omega in [-1,1]; sigmas far above that range (e.g. 20)
        # saturate almost every sample to the control-box corners (bang-bang),
        # which REDUCES effective diversity. A value near the control range
        # (~0.3-0.5) plus noise_beta gives the widest useful spread.
        sigmas=torch.tensor([0.5,0.6]), #([0.5, 0.5]),
        # noise_beta: temporal correlation of the sampling noise (0=white).
        # This is the main lever for spatial spread: correlated steering noise
        # produces sustained turns that fan the rollouts apart, whereas white
        # noise just jitters and averages back to a straight line.
        noise_beta=0.95,
        exploration=0.2,   # 20% of samples drawn fresh, not warm-started
        lambda_="ESSPS",
        use_rbr=use_rbr,
        constraint_func=env.points_safe if use_rbr else None,
    )

    def hjr_overlay(ax):
        """Draw each safe region's reachable-set boundary, translated by the
        region position. Plotted as the contour of the value tube at the
        region's certified safe margin."""
        if hjr_fno is None:
            return

        # heading used to slice the (x, y, theta, time) reachable tube
        theta = float(state[2].item())

        for i in range(hjr_fno.num_safe_regions):
            reachable_set = hjr_fno.HJR_sets[i]
            if torch.is_tensor(reachable_set):
                reachable_set = reachable_set.cpu().numpy()

            # If there's no obstacle yet, use the exact (pre-computed) reachable
            # set on the FINE grid; once obstacles are detected, use the
            # FNO-predicted set on the COARSE grid (theta/time/X-Y all differ).
            if not hjr_fno.obs_list[i]:
                theta_array = hjr_fno.theta_array_fine
                time_array = hjr_fno.time_array_fine
                X, Y = hjr_fno.X_fine, hjr_fno.Y_fine
            else:
                theta_array = hjr_fno.theta_array
                time_array = hjr_fno.time_array
                X, Y = hjr_fno.X, hjr_fno.Y

            theta_slice = np.argmin(np.abs(theta_array - theta))
            # index 0 = fully grown: flip the ascending-time argmin
            time_slice = (len(time_array) - 1) - np.argmin(
                np.abs(time_array - hjr_reach_time)
            )

            reachable_set_slice = reachable_set[..., theta_slice, time_slice]

            #delta-sublevel set of FNO reachable set (theta-sliced at current heading)
            ax.contour(
                X + hjr_fno.safe_regions[i][0],
                Y + hjr_fno.safe_regions[i][1],
                reachable_set_slice,
                levels=[hjr_fno.safe_margin[i]],
                colors="#191970",
                linewidths=2,
                linestyles="solid",
            )

            # filled feasible set {feasible_region <= safe_margin} (max over
            # theta at the Tf_reach slice; theta-marginalized, 2-D). Same grid
            # as the reachable set (g_fine before obstacles, g after), so X/Y
            # line up. Skip if the set is empty (margin below the field min).
            fr = hjr_fno.feasible_region[i]
            if torch.is_tensor(fr):
                fr = fr.cpu().numpy()
            if hjr_fno.safe_margin[i] > fr.min():
                ax.contourf(
                    X + hjr_fno.safe_regions[i][0],
                    Y + hjr_fno.safe_regions[i][1],
                    fr,
                    levels=[fr.min(), hjr_fno.safe_margin[i]],
                    colors="#ADD8E6",
                    alpha=0.4,
                )

    state = env.reset()
    max_steps = 500
    total_time = 0.0
    step_count = 0
    for i in range(max_steps):
        start = time.time()
        action_seq, state_seq = solver.forward(
            state=state, info={"hjr_fno": hjr_fno}
        )
        end = time.time()
        total_time += end - start
        step_count += 1

        # RBR collapse diagnostics: if the batch is being pinned inside a narrow
        # feasible tube we expect min#safe to crash toward 0, resample_frac to
        # climb, and mean_v / planned path length to shrink over MPC steps.
        if use_rbr and hjr_fno is not None:
            seg = state_seq[0, 1:, :2] - state_seq[0, :-1, :2]
            planned_len = float(torch.linalg.norm(seg, dim=1).sum())
            mean_v = float(action_seq[:, 0].mean())  # mean commanded speed over horizon
            print(
                f"[RBR] step {i:3d} | resample_frac={solver._rbr_resample_frac:5.2f} "
                f"| min#safe={solver._rbr_min_num_safe:5d}/{solver._num_samples} "
                f"| all_unsafe_steps={solver._rbr_num_allunsafe:2d} "
                f"| mean_v={mean_v:5.3f} | planned_len={planned_len:6.2f}"
            )

        state, is_goal_reached = env.step(action_seq[0, :])

        is_collisions = env.collision_check(state=state_seq)

        top_samples, top_weights = solver.get_top_samples(num_samples=300)

        if save_mode:
            env.render(
                predicted_trajectory=state_seq,
                is_collisions=is_collisions,
                top_samples=(top_samples, top_weights),
                mode="rgb_array",
                overlay_fn=hjr_overlay,
            )
            # progress bar
            if i == 0:
                pbar = tqdm.tqdm(total=max_steps, desc="recording video")
            pbar.update(1)

        else:
            env.render(
                predicted_trajectory=state_seq,
                is_collisions=is_collisions,
                top_samples=(top_samples, top_weights),
                mode="human",
                overlay_fn=hjr_overlay,
            )
        if is_goal_reached:
            print("Goal Reached!")
            break

    average_time = total_time / step_count
    print("average solve time: {:.3f} ms".format(average_time * 1000))
    env.close()  # close window and save video if save_mode is True


if __name__ == "__main__":
    fire.Fire(main)
