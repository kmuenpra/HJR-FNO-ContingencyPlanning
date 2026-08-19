#!/usr/bin/env bash
# =============================================================================
# Compute/quality sweep: MPPI horizon H and RRTX initial nodes n.
#
#   bash sweep_knobs.sh                          # both arms, env_A + env_D, 3 seeds
#   ARMS=2 ENVS=env_A bash sweep_knobs.sh        # MPPI horizon only
#   ARMS=4 ENVS=env_D SEEDS="1 2 3 4 5" bash sweep_knobs.sh
#   HS="10 20 40" NODES="500 2000" bash sweep_knobs.sh
#   NSAMPS="450 900 1800" ARMS=2 bash sweep_knobs.sh    # sweep N instead of H
#   SCORE_ONLY=1 bash sweep_knobs.sh             # re-score + re-plot
#
# Standalone: changes nothing in the planners. Each run is an ordinary episode
# with one knob overridden; the knob rides in the log filename (see
# EpisodeRecorder.knob_tag) so runs cannot overwrite each other, and
# eval/score_trajectory groups by it so the output is a CURVE, not one average.
#
# The two knobs are NOT commensurable -- H is a horizon, n is a node count. Do
# not put them on a shared x-axis. The comparable axis is MEASURED COMPUTE; see
# eval/plot_sweep.py, which plots RTF and bar_L against each knob separately and
# then both arms together against measured ms.
# =============================================================================
set -uo pipefail

RRTX=${RRTX:-/home/kmuenpra/anaconda3/envs/rrtx/bin/python}
ODP=${ODP:-/home/kmuenpra/anaconda3/envs/odp/bin/python}
# --seed selects the obstacle layout too, so ground truth is per (env, seed);
# SKIP_SETUP=1 postpones building it (runs execute but cannot be scored).
SKIP_SETUP=${SKIP_SETUP:-0}

# env_A: clean compute scaling (constraint cost +0.00 m, no replanning cascades).
# env_D: the quality response -- the only env where the knob should change the
#        OUTCOME (56 m certified detour vs a 21.5 m straight line).
ENVS=${ENVS:-"env_A env_D"}
SEEDS=${SEEDS:-"1 2 3"}
ARMS=${ARMS:-"2 4"}

# Doubling, to bracket each arm's own RTF=1 crossing. Measured beforehand:
# MPPI ~110 ms at H=50 (RTF~1.1), RRTX 35-74 ms at n=800-3000 (RTF 0.35-0.75),
# so RRTX needs to go past 4000 to cross at all.
HS=${HS:-"10 20 40 80"}          # MPPI horizons to sweep
NODES=${NODES:-"500 1000 2000 4000 8000"}   # RRTX initial-tree sizes to sweep

# MPPI has a SECOND knob, the rollout count N. Per-step cost scales as N*(H+1) in
# both halves of the step, so N and H are not independent -- sweep ONE at a time
# and pin the other, or the curve is uninterpretable.
#   NSAMP  : N held fixed while H is swept (the usual mode)
#   NSAMPS : set this to a LIST to sweep N instead, at HORIZON held fixed
# Sampling density per control dimension is N/(2H): at N=900 that falls 22.5 -> 9.0
# across H=20..50, so a fixed-N H-sweep answers "at this budget, which H is best?"
# rather than "what does H do?". State that when reporting it.
NSAMP=${NSAMP:-900}
NSAMPS=${NSAMPS:-""}
HORIZON=${HORIZON:-40}           # H held fixed when sweeping N

declare -A STEPS=([env_A]=1100 [env_B]=900 [env_C]=850 [env_D]=1400)
MAX_STEPS=${MAX_STEPS:-0}
SCORE_ONLY=${SCORE_ONLY:-0}

cd "$(dirname "$0")" || exit 1
LOGS=eval/results/logs
mkdir -p "$LOGS"

if [ "$SCORE_ONLY" = 0 ]; then
    echo "=== pre-flight ==="
    # per seed: --seed selects the obstacle layout as well as the rng, so each
    # seed is a different map with its own assertions and its own ground truth
    for S in $SEEDS; do
        $RRTX -m eval.assert_config --fast -q --seed "$S" \
            || { echo "assert_config FAILED at seed $S"; exit 1; }
    done
    echo "  ok (seeds: $SEEDS)"
    if [ "$SKIP_SETUP" = 0 ]; then
        echo "=== ground truth (per env x seed, cached) ==="
        ENVS="$ENVS" SEEDS="$SEEDS" ODP="$ODP" bash prepare_ground_truth.sh \
            || echo "  WARNING: ground truth incomplete -- some cells will not score"
    fi
fi

run() {   # run <label> <logfile> <command...>
    local label=$1 log=$2; shift 2
    printf '  %-40s ' "$label"
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
    N_STEPS=${STEPS[$E]}; [ "$MAX_STEPS" != 0 ] && N_STEPS=$MAX_STEPS
    W=""; [ "$E" = env_D ] && W="--topo_wide True"
    echo; echo "=== $E  (max_steps=$N_STEPS) ==="
    for S in $SEEDS; do
      for A in $ARMS; do
        case $A in
          2) if [ -n "$NSAMPS" ]; then          # sweep N, pin H
               for NSMP in $NSAMPS; do
                 run "arm2 scramppi_fno $E N=$NSMP s$S" \
                     "$LOGS/sweep_scramppi_fno_${E}_N${NSMP}_s${S}.log" \
                     $RRTX mppi_src/navigation2d.py --scenario "$E" --seed "$S" \
                     --max_steps "$N_STEPS" --horizon "$HORIZON" \
                     --num_samples "$NSMP" \
                     --use_rbr True --use_topo True --no_render True $W
               done
             else                                # sweep H, pin N
               for H in $HS; do
                 run "arm2 scramppi_fno $E H=$H s$S" \
                     "$LOGS/sweep_scramppi_fno_${E}_H${H}_s${S}.log" \
                     $RRTX mppi_src/navigation2d.py --scenario "$E" --seed "$S" \
                     --max_steps "$N_STEPS" --horizon "$H" \
                     --num_samples "$NSAMP" \
                     --use_rbr True --use_topo True --no_render True $W
               done
             fi ;;
          4) for NN in $NODES; do
               run "arm4 rrtx_fno     $E n=$NN s$S" \
                   "$LOGS/sweep_rrtx_fno_${E}_n${NN}_s${S}.log" \
                   $RRTX rrtx_FNO3d_oneGoal.py "$E" --seed "$S" \
                   --max_steps "$N_STEPS" --nodes "$NN" --no_plot
             done ;;
          *) echo "  (arm $A has no knob in this sweep -- skipped)" ;;
        esac
      done
    done
  done
fi

echo; echo "=== scoring ==="
$RRTX -m eval.score_trajectory

echo; echo "=== sweep curves ==="
$RRTX -m eval.plot_sweep
