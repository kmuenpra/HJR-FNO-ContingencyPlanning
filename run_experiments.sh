#!/usr/bin/env bash
# =============================================================================
# Batch runner for the 4-arm x 4-environment evaluation.
#
#   bash run_experiments.sh                 # default: 3 seeds, arms 1/2/4, all envs
#   ARMS="2 4" ENVS=env_A SEEDS=1 bash run_experiments.sh      # one cell
#   ARMS=3 bash run_experiments.sh                             # the slow ODP arm
#   SCORE_ONLY=1 bash run_experiments.sh                       # just re-score
#   SKIP_SETUP=1 bash run_experiments.sh       # no ODP ground truth (no scoring)
#
# --seed selects the OBSTACLE LAYOUT as well as the planners' rng: seed X gives
# the same random-but-reproducible map to every arm (eval/scenarios.py), and the
# hand-authored baseline map is what you get with no --seed at all. Ground truth
# is therefore per (env, seed) and is built by prepare_ground_truth.sh in the
# pre-flight below.
#
# Bash (not sh): uses an associative array. Run from the repo root.
# Each run logs to eval/results/logs/ and writes its own per-step CSV; a failed
# run is reported and SKIPPED rather than killing the batch.
# =============================================================================
set -uo pipefail

RRTX=${RRTX:-/home/kmuenpra/anaconda3/envs/rrtx/bin/python}
ODP=${ODP:-/home/kmuenpra/anaconda3/envs/odp/bin/python}

ENVS=${ENVS:-"env_A env_B env_C env_D"}
SEEDS=${SEEDS:-"1 2 3"}
ARMS=${ARMS:-"1 2 4"}          # 3 = SCRAMPPI-HJR (slow: re-solves every reveal)
SCORE_ONLY=${SCORE_ONLY:-0}
# --seed selects the obstacle layout too, so every seed is a different map and
# needs its own ground truth (prepare_ground_truth.sh, run in the pre-flight).
# SKIP_SETUP=1 postpones that -- the runs still execute, but cannot be scored.
SKIP_SETUP=${SKIP_SETUP:-0}

# Planner capacity, pinned rather than inherited from whatever the source
# defaults happen to be today. Both land in the log filename and in the summary
# JSON's `knob`, so every run records the capacity it actually had.
# Vary these only in the sweep (sweep_knobs.sh); hold them fixed for the arm
# comparison, or 1-vs-2-vs-4 confounds the planner with its budget.
HORIZON=${HORIZON:-30}         # MPPI rollout horizon H
NSAMP=${NSAMP:-1000}           # MPPI rollout count N
NODES=${NODES:-3000}           # RRTX initial tree samples n

# --max_steps per env ~ 3x the certified optimum / (v_max * dt_c). A run that
# times out at 3x the certified length is a real failure, not a budget artefact.
declare -A STEPS=([env_A]=1100 [env_B]=900 [env_C]=850 [env_D]=1400)
# MAX_STEPS=8 overrides all of them -- for a smoke test that only checks the
# plumbing. Any real number must come from the table above.
MAX_STEPS=${MAX_STEPS:-0}

cd "$(dirname "$0")" || exit 1
LOGS=eval/results/logs
mkdir -p "$LOGS"

# -- pre-flight: a config mismatch invalidates every number after it ----------
# Per seed, because --seed now selects the OBSTACLE LAYOUT as well as the
# planners' rng (eval/scenarios.py): each seed is a different map, so it needs
# its own assertions and its own ground truth.
if [ "$SCORE_ONLY" = 0 ]; then
    echo "=== pre-flight ==="
    $RRTX -m eval.episode_log > /dev/null || { echo "episode_log FAILED"; exit 1; }
    for S in $SEEDS; do
        $RRTX -m eval.assert_config --fast -q --seed "$S" \
            || { echo "assert_config FAILED at seed $S"; exit 1; }
    done
    echo "  config + accounting invariants OK (seeds: $SEEDS)"

    # L*_cert and the fully-revealed ODP masks, per (env, seed). Cached, so this
    # is an `ls` per cell on a re-run. Needs the `odp` env (heterocl); without it
    # the runs still execute but cannot be scored, hence the loud warning.
    if [ "$SKIP_SETUP" = 0 ]; then
        echo "=== ground truth (per env x seed, cached) ==="
        ENVS="$ENVS" SEEDS="$SEEDS" ODP="$ODP" bash prepare_ground_truth.sh \
            || echo "  WARNING: ground truth incomplete -- some cells will not score"
    else
        echo "  SKIP_SETUP=1: no ground truth built; scoring will report missing masks"
    fi
fi

run() {   # run <label> <logfile> <command...>
    local label=$1 log=$2; shift 2
    printf '  %-34s ' "$label"
    local t0=$SECONDS
    if "$@" > "$log" 2>&1; then
        printf 'ok   %4ds  %s\n' "$((SECONDS - t0))" \
            "$(grep -m1 'episode  :' "$log" | sed 's/^ *episode *: *//')"
    else
        printf 'FAIL %4ds  (see %s)\n' "$((SECONDS - t0))" "$log"
    fi
}

if [ "$SCORE_ONLY" = 0 ]; then
  for E in $ENVS; do
    N=${STEPS[$E]}; [ "$MAX_STEPS" != 0 ] && N=$MAX_STEPS
    W=""; [ "$E" = env_D ] && W="--topo_wide True"   # env_D: ellipse PRM finds 0 paths
    echo; echo "=== $E  (max_steps=$N) ==="
    for S in $SEEDS; do
      for A in $ARMS; do
        case $A in
          1) run "arm1 mppi        $E s$S" "$LOGS/mppi_${E}_s${S}.log" \
               $RRTX mppi_src/navigation2d.py --scenario "$E" --seed "$S" \
               --max_steps "$N" --horizon "$HORIZON" --num_samples "$NSAMP" \
               --use_rbr False --use_topo False --no_render True $W ;;
          2) run "arm2 scramppi_fno $E s$S" "$LOGS/scramppi_fno_${E}_s${S}.log" \
               $RRTX mppi_src/navigation2d.py --scenario "$E" --seed "$S" \
               --max_steps "$N" --horizon "$HORIZON" --num_samples "$NSAMP" \
               --use_rbr True --use_topo True --no_render True $W ;;
          3) run "arm3 scramppi_hjr $E s$S" "$LOGS/scramppi_hjr_${E}_s${S}.log" \
               $RRTX mppi_src/navigation2d.py --scenario "$E" --seed "$S" \
               --max_steps "$N" --horizon "$HORIZON" --num_samples "$NSAMP" \
               --use_rbr True --use_topo True \
               --reach_backend odp --arm scramppi_hjr --no_render True $W ;;
          4) run "arm4 rrtx_fno     $E s$S" "$LOGS/rrtx_fno_${E}_s${S}.log" \
               $RRTX rrtx_FNO3d_oneGoal.py "$E" --seed "$S" \
               --max_steps "$N" --nodes "$NODES" --no_plot ;;
        esac
      done
    done
  done
fi

echo; echo "=== scoring ==="
$RRTX -m eval.score_trajectory
