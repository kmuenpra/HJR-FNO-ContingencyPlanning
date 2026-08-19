#!/usr/bin/env bash
# =============================================================================
# Per-seed ground truth for the performance metrics. Runs in the `odp` conda env
# (heterocl), ONCE per (env, seed), and caches.
#
# Two artifacts, both functions of the OBSTACLE GEOMETRY -- so with random
# obstacles (--seed, eval/scenarios.py) they are per (env, seed), not per env:
#
#   eval/results/baseline_astar_s<seed>.json   L*_cert, the denominator of bar_L
#   eval/results/gt_masks_<env>_s<seed>.npz    exact reach-avoid masks, for VR
#
# Both carry the map's obstacle digest and score_trajectory checks it, so a
# mismatch raises instead of quietly producing a plausible wrong number.
#
#   bash prepare_ground_truth.sh                      # all envs, seeds 1..10
#   SEEDS="1 2 3" ENVS=env_A bash prepare_ground_truth.sh
#   SEEDS=base bash prepare_ground_truth.sh           # the hand-authored map
#   FORCE=1 bash prepare_ground_truth.sh              # rebuild, ignore the cache
#   bash prepare_ground_truth.sh --status             # just print the coverage
#
# run_experiments.sh and sweep_knobs.sh call this for their own ENVS/SEEDS, so
# there is ONE implementation of the caching rule.
#
# Bash (not sh). Run from the repo root. ~17 s per cell, ~12 min for 4 x 10.
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")" || exit 1

ODP=${ODP:-/home/kmuenpra/anaconda3/envs/odp/bin/python}
ENVS=${ENVS:-"env_A env_B env_C env_D"}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}   # `base` = the hand-authored obstacles
FORCE=${FORCE:-0}
RES=${RES:-0.1}                          # baseline_astar A* grid resolution

RESULTS=eval/results
LOGS=$RESULTS/logs
mkdir -p "$LOGS"

STATUS_ONLY=0
[ "${1:-}" = "--status" ] && STATUS_ONLY=1

# -- paths, matching eval/score_trajectory.py's mask_path / baseline_json_path --
mask_file() {  # mask_file <env> <seed>
    if [ "$2" = base ]; then echo "$RESULTS/gt_masks_$1.npz"
    else echo "$RESULTS/gt_masks_$1_s$2.npz"; fi
}
baseline_file() {  # baseline_file <seed>
    if [ "$1" = base ]; then echo "$RESULTS/baseline_astar.json"
    else echo "$RESULTS/baseline_astar_s$1.json"; fi
}
seed_flag() {  # seed_flag <seed>
    if [ "$1" = base ]; then echo ""; else echo "--seed $1"; fi
}
# Has this seed's baseline JSON got an entry for this env yet? The file is shared
# by the four envs (read-modify-write), so existence alone is not enough.
baseline_has() {  # baseline_has <env> <seed>
    local f; f=$(baseline_file "$2")
    [ -f "$f" ] || return 1
    local key="$1"; [ "$2" = base ] || key="$1_s$2"
    grep -q "\"$key\"" "$f"
}

status_table() {
    echo
    printf '%-8s' "seed"; for E in $ENVS; do printf '%-16s' "$E"; done; echo
    printf '%-8s' "----"; for E in $ENVS; do printf '%-16s' "--------------"; done; echo
    local n_ok=0 n_tot=0
    for S in $SEEDS; do
        printf '%-8s' "$S"
        for E in $ENVS; do
            local b=. m=.
            baseline_has "$E" "$S" && b="L"
            [ -f "$(mask_file "$E" "$S")" ] && m="V"
            n_tot=$((n_tot + 1))
            [ "$b$m" = "LV" ] && n_ok=$((n_ok + 1))
            printf '%-16s' "$b$m"
        done
        echo
    done
    echo "  L = L*_cert present, V = ODP masks present"
    echo "  complete: $n_ok / $n_tot cell(s)"
    [ "$n_ok" -eq "$n_tot" ]
}

if [ "$STATUS_ONLY" = 1 ]; then
    status_table; exit $?
fi

if [ ! -x "$ODP" ]; then
    echo "ERROR: no odp interpreter at $ODP"
    echo "  Both steps need heterocl, which only the \`odp\` env has."
    echo "  Set \$ODP, or run with SKIP_SETUP=1 to postpone (runs then cannot be scored)."
    exit 2
fi

run() {   # run <label> <logfile> <command...>
    local label=$1 log=$2; shift 2
    printf '  %-42s ' "$label"
    local t0=$SECONDS
    if "$@" > "$log" 2>&1; then
        printf 'ok   %4ds\n' "$((SECONDS - t0))"
        return 0
    fi
    printf 'FAIL %4ds  (see %s)\n' "$((SECONDS - t0))" "$log"
    tail -3 "$log" | sed 's/^/      | /'
    return 1
}

n_fail=0
n_built=0
n_cached=0
T0=$SECONDS
for S in $SEEDS; do
    echo; echo "=== seed $S ==="
    SF=$(seed_flag "$S")
    for E in $ENVS; do
        # 1. L*_cert -- the bar_L denominator. Per env, because baseline_astar
        #    rewrites its JSON whole; the --dump path merges what is already there.
        if [ "$FORCE" = 0 ] && baseline_has "$E" "$S"; then
            printf '  %-42s cached\n' "baseline_astar $E s$S"
            n_cached=$((n_cached + 1))
        else
            run "baseline_astar $E s$S" "$LOGS/gt_${E}_s${S}_baseline.log" \
                $ODP -m eval.baseline_astar --scenario "$E" $SF \
                     --res "$RES" --dump \
                || n_fail=$((n_fail + 1))
            n_built=$((n_built + 1))
        fi

        # 2. the fully-revealed ODP masks -- the violation-rate ground truth
        MF=$(mask_file "$E" "$S")
        if [ "$FORCE" = 0 ] && [ -f "$MF" ]; then
            printf '  %-42s cached\n' "gt_masks $E s$S"
            n_cached=$((n_cached + 1))
        else
            run "gt_masks $E s$S" "$LOGS/gt_${E}_s${S}_masks.log" \
                $ODP -m eval.score_trajectory --build_masks --scenario "$E" $SF \
                || n_fail=$((n_fail + 1))
            n_built=$((n_built + 1))
        fi
    done
done

echo
echo "=== ground truth: $n_built built, $n_cached cached, $n_fail failed"\
     "in $((SECONDS - T0))s ==="
status_table
if [ "$n_fail" -gt 0 ]; then
    echo "  $n_fail step(s) FAILED -- those cells cannot be scored"
    exit 1
fi
exit 0
