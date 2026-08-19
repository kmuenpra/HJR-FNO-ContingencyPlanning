"""Batch HJ solve worker -- runs under the `odp` interpreter, called by odp_oracle.

WHY A SUBPROCESS
----------------
The two conda envs are disjoint in exactly the wrong way:

    rrtx : torch + fire, NO heterocl      <- MPPI runs here
    odp  : heterocl, NO torch             <- HJSolver runs here

MPPI cannot import heterocl and the solver cannot import torch, so the ODP
reachability backend has to cross a process boundary. This module is the far side:
it reads a JSON spec, solves every requested region through
``eval.odp_reachset.solve_region`` (the SAME function the hindsight baseline uses,
so the two backends cannot drift), and writes one npz for the parent to load.

    $ODP -m mppi_src.odp_solve_worker spec.json     # one-shot
    $ODP -m mppi_src.odp_solve_worker --serve       # persistent (the online path)

TWO MODES, AND WHY THE PERSISTENT ONE EXISTS
--------------------------------------------
One-shot mode pays, on EVERY reveal, costs a deployed system pays once per
mission:

    fork/exec                 ~0.1 s   <- artifact of the conda env split
    import heterocl           ~0.80 s  <- once per process
    first-solve lazy init     ~0.77 s  <- once per process
    graph_3D / hcl.build      ~0.30 s  <- per HJSolver call (see below)
    the HJ PDE solve          ~0.82 s  <- the number the comparison wants

Measured: three back-to-back solve_region calls in one process took 1.895 /
1.131 / 1.119 s. So ~0.77 s of the first call is one-time, and steady state is
~1.12 s. Charging the one-time part to every update inflated `t_reach` for the
ODP arm by roughly 0.7-0.8 s per event -- against an FNO arm whose own worker
pool is already persistent (HJR_FNO3d._get_scenario_pool caches and reuses it).
That asymmetry flattered the FNO, i.e. it favoured the contribution under test.

Serve mode moves import and first-solve into pool startup, outside the timed
loop, matching where the FNO's model load and CUDA init already sit (plan doc
3.1's "drop k <= 1"). What it CANNOT remove is the ~0.30 s graph rebuild:
HJSolver calls hcl.init() and graph_3D unconditionally on every invocation
(optimized_dp/odp/solver.py:81,161), even though the compiled kernel depends
only on (dynamics, grid, TargetSetMode, accuracy) -- all fixed for the whole
mission. Obstacles enter as DATA, not into the graph. That 0.30 s is a known
floor of this implementation and should be reported as such rather than as a
property of exact HJ reachability.

THE NPZ IS TRANSPORT, NOT A CACHE -- IN BOTH MODES. The parent deletes it
immediately after loading, and every request re-solves. See the NO CACHING
section of eval/odp_reachset.py -- online solve cost is a measured quantity
here, so caching it would falsify the very number the planner comparison
reports. Serve mode holds NO per-request state; do not add a tube cache here.

PROTOCOL (serve mode), line-oriented over stdin/stdout:

    <- {"regions": [...], "Tf": ..., "out": "/tmp/.../tubes.npz"}   one JSON line
    -> __ODP_RESULT__ {"ok": true, "out": ..., "t_solve": [...], ...}

Responses are SENTINEL-PREFIXED because heterocl prints to stdout during
graph_3D ("Optimizing"), which would desynchronise a bare line protocol. The
parent scans for the prefix and discards everything else. stderr is redirected
to a per-worker log file by the parent, so a chatty child cannot fill a pipe
buffer and deadlock.

The FULL tube (every time slice) is returned, ~4 MB per region. The feasibility
test alone would only need the grown slice V[..., 0], but ``contingency_policy``
needs the time-indexed ones: the grown slice SATURATES at ~-r_target across the
whole interior of the reachable set (every state there can already reach the
target, so the min-over-time value is the same), leaving grad V ~ 1e-3 with no
usable descent direction. Slice k -- "reachable within k steps" -- is what has a
gradient to follow.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Response sentinel. heterocl prints during graph_3D, so the parent cannot treat
# every stdout line as protocol -- it scans for this prefix and drops the rest.
RESULT_PREFIX = "__ODP_RESULT__ "
READY_TOKEN = "__ODP_READY__"


def _run_spec(spec: dict) -> dict:
    """Solve every region in one spec and write the npz. Returns the reply dict.

    Holds no state between calls: every region is re-solved from its obstacle
    list, which is the measured cost the ODP arm exists to report.
    """
    from eval.odp_reachset import set_grid, solve_region

    if spec.get("grid"):
        nx, ny, nth, nt = spec["grid"]
        set_grid(n_x=nx, n_y=ny, n_theta=nth, n_time=nt)

    out, t_solve, t_wall = {}, {}, {}
    for r in spec["regions"]:
        idx = int(r["index"])
        _t0 = time.perf_counter()
        tube = solve_region(
            r["obs_local"], r_target=float(r["r_target"]), Tf=float(spec["Tf"]),
            center=r["center"], quadratic=bool(spec.get("quadratic", False)),
            accuracy=spec.get("accuracy", "medium"), verbose=False,
        )
        t_wall[idx] = time.perf_counter() - _t0
        # full tube (x, y, theta, t); index 0 = fully grown
        out[f"V_{idx}"] = np.asarray(tube["V"], dtype=np.float32)
        out[f"solve_time_{idx}"] = np.float64(tube["solve_time"])
        t_solve[idx] = float(tube["solve_time"])
        out["x_axis"] = tube["x_axis"]
        out["y_axis"] = tube["y_axis"]
        out["theta_axis"] = tube["theta_axis"]
        out["time_array"] = tube["time_array"]

    np.savez(spec["out"], **out)
    return {"ok": True, "out": spec["out"],
            "regions": [int(r["index"]) for r in spec["regions"]],
            # solve_time is HJSolver's own clock; wall additionally carries the
            # ~0.30 s graph_3D rebuild that HJSolver does on every call.
            "t_solve_s": t_solve, "t_wall_s": t_wall}


def main(spec_path: str) -> int:
    """One-shot mode: solve one spec and exit. Kept for eval/odp_reachset and
    any caller that is not the online oracle."""
    _run_spec(json.loads(Path(spec_path).read_text()))
    return 0


def serve() -> int:
    """Persistent mode: import once, warm up once, then answer specs forever.

    The warmup solve is what moves the ~0.77 s first-call lazy initialisation out
    of the measured loop. It solves an obstacle-free region, so it exercises the
    same code path without depending on any scenario.
    """
    t0 = time.perf_counter()
    from eval.odp_reachset import solve_region       # noqa: F401  (import cost)

    try:
        solve_region([], r_target=1.0, Tf=1.0, center=[0.0, 0.0], verbose=False)
    except Exception as exc:  # noqa: BLE001 - warmup is best-effort
        print(f"[odp_worker] warmup solve failed: {exc}", file=sys.stderr)
    print(f"{READY_TOKEN} {time.perf_counter() - t0:.3f}", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "__ODP_STOP__":
            break
        try:
            reply = _run_spec(json.loads(line))
        except Exception as exc:  # noqa: BLE001 - report, do not die
            import traceback

            traceback.print_exc(file=sys.stderr)
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(RESULT_PREFIX + json.dumps(reply), flush=True)
    return 0


if __name__ == "__main__":
    if "--serve" in sys.argv[1:]:
        raise SystemExit(serve())
    raise SystemExit(main(sys.argv[1]))
