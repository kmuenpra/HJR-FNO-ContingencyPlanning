import sys
import time
from pathlib import Path

# import gymnasium
import fire
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt

from envs.navigation_2d import Navigation2DEnv
from guidance import PRM_FAST, PRM_QUALITY, PRM_WIDE, TopoGuidance
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


def _load_scenario(name: str, seed: int = None):
    """Look up a shared evaluation scenario from ``eval/scenarios.py`` by name
    (e.g. "env_A"), or load one from a .json path.

    ``seed`` selects the OBSTACLE LAYOUT: None gives the hand-authored baseline
    list, an int gives a random layout that is a pure function of (name, seed) --
    so this arm and every other arm at the same --seed solve the same world. See
    eval/scenarios.py's ``_obs_rng`` for why that holds despite each arm seeding
    and consuming the global rng differently."""
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from eval.scenarios import get_scenario

    return get_scenario(name, seed)


def main(
    save_mode: bool = False,
    use_hjr: bool = True,
    reach_backend: str = "fno",
    hjr_reach_time: float = 8.0,
    odp_inflate: float = 1.0,
    use_rbr: bool = True,
    use_topo: bool = True,
    topo_fast: bool = False,
    topo_wide: bool = False,
    scenario: str = None,
    horizon: int = 50,
    num_samples: int = 1800,
    max_steps: int = 1000,
    out_path: str = None,
    seed: int = None,
    arm: str = None,
    log_dir: str = None,
    no_log: bool = False,
    no_render: bool = False,
    mem_profile: bool = False,
):
    """
    reach_backend: which oracle answers the reachability question.
        "fno" (default) -- the HJR-FNO surrogate, with its scenario-optimization
            safe margin (the `[scenario] region N: delta_hat = ...` lines).
        "odp"  -- the EXACT numerical HJ solve, re-solved online on every lidar
            reveal, with the ANALYTIC Lipschitz margin (delta ~ 0.411 m). No
            scenario optimization runs in this mode. Solving happens in a
            subprocess under the `odp` interpreter (set $ODP_PYTHON to override),
            because MPPI needs torch and HJSolver needs heterocl and no single
            conda env has both.
    odp_inflate: safety factor on the ODP Lipschitz margin (reach_backend="odp").
    use_topo: guided multi-group MPPI. A topological PRM finds M homotopy-distinct
        paths inside the reachable set and a Dubins pure-pursuit tracker turns each
        into an ancillary mean; MPPI samples around all M means plus its own
        previous mean (M+1 groups) and commits to one group (winner-take-all).
    topo_fast: use the cheap PRM preset (~67 ms, 3-4 paths) instead of the default
        quality preset (~640 ms, ~6 paths). Only matters on replan steps, which
        fire only when the lidar reveals new obstacles.
    topo_wide: sample the WHOLE domain instead of a start->goal ellipse. Needed
        when the feasible reachable-set corridor is a big detour perpendicular to
        the straight line (e.g. env_D's U-shaped set, where the ellipse finds 0
        paths). Slower per replan (~1-2 s). Overrides topo_fast.
    horizon: MPPI rollout horizon H, in control steps. Lookahead distance is
        H * v_max * dt_c, so H=40 sees 4 m. The knob for the compute/quality
        sweep (RRTX's counterpart is --nodes). It lands in the log filename and
        in the summary JSON's `knob`, so a sweep's runs cannot overwrite each
        other and the scorer groups them separately.
    num_samples: MPPI rollout count N. Per-step cost scales as N*(H+1) in
        BOTH halves of the step (GPU rollout and CPU feasibility), so it is the
        other compute knob. NOTE the sampling density per control dimension is
        N/(2H), so holding N fixed while sweeping H makes the estimator sparser
        at large H -- state that if you sweep H at fixed N.
    max_steps: episode timeout, in control steps. On timeout the run stops and the
        video / final plot are still written (see the finally block below).
    out_path: gif path; defaults to video/navigation_2d_<scenario or seed>.gif, with
        the final frame also saved as <...>_final.png.
    seed: seeds torch + numpy + random, so this run can be paired with the RRTX
        run on the same seed. It ALSO selects the scenario's obstacle layout
        (eval/scenarios.py): with --seed the obstacles are generated randomly and
        reproducibly from (scenario, seed) -- identical in every arm -- and
        WITHOUT it the scenario's hand-authored baseline list is used. The safe
        regions, start and goal are hand-picked either way.
    arm/log_dir/no_log: per-step CSV (eval/episode_log.py). ``arm`` is the label
        in the filename; it defaults to a name derived from use_rbr/use_topo.
    no_render: skip drawing entirely -- the RRTX driver's --no_plot. Use for
        timing runs: rendering is outside the timed block, so it does not corrupt
        t_planner, but it dominates wall clock and its frame buffer inflates the
        memory numbers. Overrides save_mode (no frames are captured).
    """
    if seed is not None:
        import random

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # A shared scenario (eval/scenarios.py) fixes the map, safe regions,
    # obstacles, start and goal so this run is comparable to the RRTX run on the
    # same scenario. Without it the env falls back to its own random map.
    # --seed picks the obstacle layout as well as the planner's rng, so a run is
    # reproducible end to end and every arm at seed X sees the same map.
    sc = _load_scenario(scenario, seed) if scenario else None
    if sc is not None:
        hjr_reach_time = sc.Tf_reach
        print(f"[scenario] {sc.tag}: {len(sc.safe_regions)} safe regions, "
              f"{len(sc.obstacles)} obstacles, domain +/-{sc.domain}"
              + (f", obstacles RANDOM (seed {sc.obs_seed}, digest {sc.obs_digest})"
                 if sc.obs_seed is not None else ", obstacles hand-authored"))
    env = Navigation2DEnv(scenario=sc)

    # Optional Hamilton-Jacobi reachability oracle. Construction can fail if the
    # HJR-FNO dependencies (neural_utils.py, scenario_worker.py, utils.py, the
    # trained model and .mat tubes) are not present; in that case we just skip
    # the reachable-set overlay and run the planner as usual.
    hjr_fno = None
    if use_hjr and reach_backend == "odp":
        # Exact-HJ backend. Deliberately NOT wrapped in a try/except: if the
        # solver subprocess cannot run, silently falling back to "no reachability"
        # would produce a run that looks fine and enforces nothing.
        from odp_oracle import ODPReachOracle

        hjr_fno = ODPReachOracle(
            env=env,
            safe_regions=env.safe_regions,
            Tf_reach=hjr_reach_time,
            inflate=odp_inflate,
        )
    elif use_hjr and reach_backend != "fno":
        raise ValueError(f"unknown reach_backend {reach_backend!r}; use 'fno' or 'odp'")
    elif use_hjr:
        try:
            HJR_FNO = _load_hjr_fno_class()
            hjr_fno = HJR_FNO(
                env=env,
                safe_regions=env.safe_regions,
                Tf_reach=hjr_reach_time,
            )
            # scenario_parallel comes from eval/config.yaml, same as RRTX -- it
            # must match or t_reach is not comparable between the arms. (The old
            # comment here claimed the spawned workers could not import the
            # HJR_FNO package; _get_scenario_pool injects the repo root into
            # PYTHONPATH itself, and this module has the __main__ guard spawn
            # needs, so the concern was stale.)
            # Use the theta-marginalized feasible_region (max over theta at the
            # Tf_reach slice) for feasibility, so the cost boundary matches the
            # filled blue feasible-set overlay.
            hjr_fno.feasibility_source = "feasible_region"
        except Exception as e:  # noqa: BLE001 - any import/IO/dependency failure
            print(f"[HJR-FNO] disabled (could not initialize): {e}")
            hjr_fno = None

    # let the env feed lidar-detected obstacles to the oracle inside step()
    if hjr_fno is not None:
        env.attach_reachability(hjr_fno)

    # solver
    solver = MPPI(
        horizon=horizon,
        num_samples=num_samples,
        dim_state=3,
        dim_control=2,
        dynamics=env.dynamics,
        cost_func=env.cost_function,
        u_min=env.u_min,
        u_max=env.u_max,
        sigmas=torch.tensor([0.5,0.5]), #([0.5, 0.5]),
        lambda_="ESSPS",
        # RBR + safety-filter control selection (active only when use_rbr=True;
        # use_rbr=False leaves the solver behavior byte-identical to plain MPPI).
        use_rbr=use_rbr,
        constraint_func=env.points_safe if use_rbr else None,
    )

    # Topological guidance: M homotopy-distinct ancillary means for MPPI.
    # The roadmap is rebuilt only when the lidar reveals new obstacles (the
    # occupancy AND reachable set change together); the Dubins tracker reruns
    # every control step so the means always start from the current state.
    # use_rbr is the master switch for the whole robust pipeline, so guidance is
    # only built alongside it; with use_rbr=False this runs plain Williams MPPI.
    guide = None
    if use_topo and not use_rbr:
        print("[topo] ignored: --use_topo requires --use_rbr True")
    if use_topo and use_rbr:
        guide = TopoGuidance(
            env._obstacle_map,
            horizon=solver._horizon,
            u_min=env.u_min,
            u_max=env.u_max,
            dt=0.1,
            feasible_fn=env.points_feasible_xy,
            prm_kwargs=(
                PRM_WIDE if topo_wide else (PRM_FAST if topo_fast else PRM_QUALITY)
            ),
        )

    def topo_overlay(ax):
        """Draw the cached topological paths (the ancillary proposal means),
        highlighting the best-SCORING group. That ranking is diagnostic only --
        the executed control is the safety-filtered weighted mean over all
        groups, not a single group's mean."""
        if guide is None or not guide.paths:
            return
        cmap = plt.get_cmap("tab10")
        sel = solver._selected_group
        for k, p in enumerate(guide.paths):
            chosen = k == sel
            ax.plot(
                p[:, 0],
                p[:, 1],
                color=cmap(k % 10),
                lw=2.5 if chosen else 1.0,
                ls="solid" if chosen else "dashed",
                alpha=0.95 if chosen else 0.5,
                zorder=8,
            )

    def hjr_overlay(ax):
        """Draw each safe region's reachable-set boundary, translated by the
        region position. Plotted as the contour of the value tube at the
        region's certified safe margin."""
        if hjr_fno is None:
            return

        # Backend-agnostic path: an oracle that knows how to draw itself does so,
        # and this function stops reaching into its internals. The FNO oracle has
        # no draw_sets(), so it falls through to the FNO-specific code below.
        if hasattr(hjr_fno, "draw_sets"):
            hjr_fno.draw_sets(ax, theta=float(state[2].item()))
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

            # Wrap the heading into the ARRAY's own periodic range before the
            # nearest-index lookup: the fine (obstacle-free) grid's theta_array is
            # [0, 2pi) while the FNO grid's is [-pi, pi), and the robot heading is
            # always [-pi, pi) (angle_normalize). Without this, a negative heading
            # picks index 0 of the fine grid and the wrong slice gets drawn.
            _lo = float(theta_array[0])
            theta_slice = np.argmin(
                np.abs(theta_array - (_lo + (theta - _lo) % (2 * np.pi)))
            )
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
            # theta at the Tf_reach slice; theta-marginalized, 2-D) -- this is
            # the set the cost now enforces (feasibility_source="feasible_region").
            # Same grid as the reachable set (g_fine before obstacles, g after),
            # so X/Y line up. Skip if the set is empty (margin below field min).
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

    def overlay(ax):
        hjr_overlay(ax)
        topo_overlay(ax)

    # HJ contingency fallback: used when the guidance can find no certified route
    # to the goal, so the robot retreats to the nearest safe region instead of
    # idling. Requires the oracle (the policy lives on it) and the guidance (its
    # empty path list is the trigger).
    contingency = None
    if guide is not None and hjr_fno is not None:
        from contingency import ContingencyManager

        contingency = ContingencyManager(env, hjr_fno)

    state = env.reset()
    total_time = 0.0
    step_count = 0
    is_goal_reached = False
    pbar = None

    # NOTE: for evaluating per-step solve time
    rec = rt = fc = mem = None
    if not no_log:
        from eval.episode_log import EpisodeRecorder
        from eval.profiling import FeasCounter, MemoryProbe, ReachTimer

        # baseline AFTER the model load, so it measures the shared backend
        # tracemalloc taxes every allocation; opt-in (see --mem_profile)
        mem = MemoryProbe(trace_cpu=mem_profile).baseline()
        rt = ReachTimer(hjr_fno)        # wraps update_obs in place
        fc = FeasCounter(hjr_fno)       # wraps points_feasible in place
        # The backend belongs in the default name: without it a hand-run
        # `--reach_backend odp` writes timing_scramppi_fno_*.csv and silently
        # OVERWRITES arm 2's file with arm 3's numbers. run_experiments.sh passes
        # --arm explicitly, so this only bites interactive runs -- which is
        # exactly when it would go unnoticed.
        # Naming follows the names already on disk and in run_experiments.sh:
        # the scramppi family always carries the backend (scramppi_fno /
        # scramppi_hjr); the plain-mppi family names FNO implicitly (arm 1 is
        # "mppi", not "mppi_fno") and only marks the odp variant.
        _odp = reach_backend == "odp"
        rec = EpisodeRecorder(
            arm or ("scramppi_" + ("hjr" if _odp else "fno") if (use_rbr and use_topo)
                    else "mppi_rbr" + ("_hjr" if _odp else "") if use_rbr
                    else "mppi" + ("_hjr" if _odp else "")),
            sc.name if sc is not None else f"seed{env._seed}",
            seed=seed,
            knob={"H": solver._horizon, "N": solver._num_samples},
            out_dir=log_dir,
        )
        rec.start(*(float(v) for v in state.cpu()))
    ctg_own_carry = 0.0                 # NOTE: for evaluating per-step solve time

    def _render(state_seq, is_collisions, top):
        """One frame, in whichever mode the run was launched with."""
        nonlocal pbar
        if no_render:
            return
        if save_mode:
            env.render(
                predicted_trajectory=state_seq,
                is_collisions=is_collisions,
                top_samples=top,
                mode="rgb_array",
                overlay_fn=overlay,
            )
            if pbar is None:
                pbar = tqdm.tqdm(total=max_steps, desc="recording video")
            pbar.update(1)
        else:
            env.render(
                predicted_trajectory=state_seq,
                is_collisions=is_collisions,
                top_samples=top,
                mode="human",
                overlay_fn=overlay,
            )

    try:
        for i in range(max_steps):
            if rec is not None:                     # NOTE: for evaluating per-step solve time
                rt.reset_step()
                fc.reset_step()
            start = time.perf_counter()
            if guide is not None:
                # replan the roadmap only on obstacle-reveal steps; re-track the
                # cached paths into fresh ancillary means every step
                replanned = guide.maybe_replan(
                    state,
                    env._goal_pos.cpu().numpy(),
                    env._obs_revealed,
                    group_died=bool(
                        solver._group_dead is not None and solver._group_dead.any()
                    ),
                    best_group=solver._selected_group,
                    num_groups=solver._num_groups,
                    topo_mass=(
                        float(solver._group_mass[:-1].sum())
                        if solver._group_mass is not None
                        else None
                    ),
                )
                solver.set_group_means(
                    guide.group_means(state, solver._device, solver._dtype)
                )
                # Use the geodesic cost-to-go (built/updated inside the replan) as
                # MPPI's goal term, so moving along the feasible corridor (e.g.
                # down a U's left column) is correctly rewarded and the topo
                # homotopy class wins the weighting. Same object each step; its
                # field is refreshed in place on replan.
                env.cost_to_go = guide.cost_to_go

                # No certified route to the goal (topo=0)? Rather than idling in
                # place while a hopeless replan re-fires, fall back to the HJ
                # contingency policy: drive to the nearest certified safe region
                # and wait there until a replan succeeds. Mirrors RRTX's
                # contingency_triggered path.
                if contingency is not None:
                    if not guide.paths:
                        contingency.start(state)
                        # NOTE: for evaluating per-step solve time
                        # the maneuver's per-pose times were measured inside the
                        # solve this block is about to be charged for; subtract
                        # them so replaying them later does not double-count
                        ctg_own_carry = contingency.rollout_own_total_s
                    elif contingency.active:
                        contingency.stop()

            # ---- contingency owns the robot: skip the MPPI solve entirely ----
            if contingency is not None and contingency.active:
                state, is_goal_reached = contingency.advance()
                if rec is not None:                 # NOTE: for evaluating per-step solve time
                    ReachTimer.sync()
                end = time.perf_counter()
                total_time += end - start
                step_count += 1
                if rec is not None:                 # NOTE: for evaluating per-step solve time
                    # This step costs the replay overhead PLUS the share of the
                    # solve attributable to its pose. That share was measured in
                    # an earlier block, so it adds to t_s here (t_own_s alone
                    # carves out of the block rather than adding to it) and is
                    # subtracted once, via ctg_own_carry, from the block that
                    # actually ran the solve.
                    own = contingency.last_step_own_s
                    rec.mark(*(float(v) for v in state.cpu()),
                             mode="idle" if contingency.idling else "contingency",
                             t_own_s=own)
                    rec.charge(
                        t_s=max(0.0, (end - start) + rt.step_s + own - ctg_own_carry),
                        t_reach_s=rt.step_s,
                        t_predict_s=rt.step_predict_s, t_certify_s=rt.step_certify_s,
                        n_events=rt.step_events, n_feas=fc.step_n)
                    ctg_own_carry = 0.0
                state_seq = state.view(1, 1, -1).repeat(1, 2, 1)  # 1-state stub
                is_collisions = env.collision_check(state=state_seq)
                if i % 10 == 0:
                    print(f"[{i:4d}] {(end - start) * 1e3:6.1f}ms | "
                          f"{contingency.status()} | topo={len(guide.paths)}")
                _render(state_seq, is_collisions, None)
                if is_goal_reached:
                    print("Goal Reached!")
                    break
                continue

            action_seq, state_seq = solver.forward(
                state=state, info={"hjr_fno": hjr_fno}
            )
            if rec is not None:                     # NOTE: for evaluating per-step solve time
                ReachTimer.sync()                   # flush GPU before stopping the clock
            end = time.perf_counter()
            total_time += end - start
            step_count += 1

            # diagnostics: group selection / RBR resampling. Printed on replan
            # steps and every 10th step, so it stays readable.
            diag = solver.diagnostics()
            if diag and (i % 10 == 0 or (guide is not None and replanned)):
                extra = ""
                if guide is not None:
                    extra = (
                        f" | topo={len(guide.paths)}"
                        f" track={guide.last_track_ms:.1f}ms"
                        + (
                            f" REPLAN[{guide.last_reason}]="
                            f"{guide.last_replan_ms:.0f}ms"
                            if replanned
                            else ""
                        )
                    )
                print(f"[{i:4d}] {(end - start) * 1e3:6.1f}ms | {diag}{extra}")

            # update_obs fires INSIDE env.step, so its cost is added from the
            # ReachTimer rather than by timing env.step (which also does lidar +
            # rasterization -- simulator work RRTX has no counterpart to).
            state, is_goal_reached = env.step(action_seq[0, :])

            if rec is not None:                     # NOTE: for evaluating per-step solve time
                rec.mark(*(float(v) for v in state.cpu()), mode="nominal")
                rec.charge(t_s=(end - start) + rt.step_s, t_reach_s=rt.step_s,
                           t_predict_s=rt.step_predict_s, t_certify_s=rt.step_certify_s,
                           n_events=rt.step_events, n_feas=fc.step_n)

            is_collisions = env.collision_check(state=state_seq)

            top_samples, top_weights = solver.get_top_samples(num_samples=300)

            _render(state_seq, is_collisions, (top_samples, top_weights))
            if is_goal_reached:
                print("Goal Reached!")
                break
        else:
            # loop ran to max_steps without reaching the goal
            print(f"TIMEOUT: goal not reached within {max_steps} steps "
                  f"(final distance {float(torch.norm(state[:2] - env._goal_pos)):.2f} m)")
    finally:
        # Always report and always write the video / final plot, whether the run
        # succeeded, timed out, crashed or was interrupted.
        if pbar is not None:
            pbar.close()
        if step_count:
            print("average solve time: {:.3f} ms".format(total_time / step_count * 1000))
        print(f"steps: {step_count}, goal reached: {bool(is_goal_reached)}")
        # Online reachability cost. Only the ODP backend reports it, because only
        # it pays a measurable one (the FNO's is folded into per-step inference).
        if hjr_fno is not None and hasattr(hjr_fno, "stats"):
            s = hjr_fno.stats()
            print(f"[ODP] reachability: {s['solve_time_total_s']:.1f} s total over "
                  f"{s['solve_calls']} solve(s) / {s['regions_solved']} region-solves, "
                  f"{s['known_obstacles']} obstacles revealed, "
                  f"mean delta {s['delta_mean']:.3f}")
        if rec is not None:                         # NOTE: for evaluating per-step solve time
            rec.finish(
                goal_reached=bool(is_goal_reached),
                final_goal_dist=float(torch.norm(state[:2] - env._goal_pos)),
                mem=mem.measure(),
                contingency_triggers=getattr(contingency, "num_triggers", 0),
            )
            rec.print_summary()
            rec.to_csv()
        if save_mode:
            # sc.tag, not sc.name: two seeds of the same env are two different
            # maps, so keying the gif on the name alone would silently overwrite.
            tag = sc.tag if sc is not None else f"seed{env._seed}"
            gif = out_path or f"video/navigation_2d_{tag}.gif"
            Path(gif).parent.mkdir(parents=True, exist_ok=True)
            # env.render() clears the axes after grabbing each frame, so the live
            # figure is blank by now -- save the LAST RECORDED frame as the still.
            if env._rendered_frames:
                from PIL import Image

                still = gif.rsplit(".", 1)[0] + "_final.png"
                Image.fromarray(env._rendered_frames[-1]).save(still)
                print(f"[save_mode] wrote final plot to {still}")
            env.close(path=gif)  # writes the gif
            print(f"[save_mode] wrote {len(env._rendered_frames)} frames to {gif}")
        else:
            env.close()


if __name__ == "__main__":
    # fire SILENTLY DROPS flags it does not recognise, so a typo like
    # `--use_rbr Ture` would quietly run a different arm with nothing in the log
    # to say so. Validate the names against main()'s signature first.
    import inspect

    _known = set(inspect.signature(main).parameters)
    _given = [a.split("=", 1)[0].lstrip("-").replace("-", "_")
              for a in sys.argv[1:] if a.startswith("--")]
    _bad = [f for f in _given if f not in _known]
    if _bad:
        raise SystemExit(
            "unknown flag(s): " + ", ".join("--" + f for f in _bad) + "\n"
            "available: " + ", ".join("--" + k for k in sorted(_known))
        )
    fire.Fire(main)
