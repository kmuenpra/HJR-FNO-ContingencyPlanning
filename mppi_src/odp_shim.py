"""odp shim — re-export the real ``Grid`` and ``DubinsCar2`` from optimized_dp.

The MPPI framework needs odp's ``Grid`` and ``DubinsCar2`` but optimized_dp is
path-injected (not pip-installed), and ``DubinsCar2`` imports ``heterocl`` at
module top (only its HeteroCL GPU-solver methods need it). This module does the
same bootstrap that HJR_FNO/scenario_worker.py does, so callers can simply::

    from .odp_shim import Grid, DubinsCar2

and get the genuine optimized_dp classes regardless of whether heterocl is
installed in the active conda env.
"""

import sys as _sys
from pathlib import Path

# optimized_dp lives at the repo root (mppi_src/ is one level below it).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ODP_ROOT = str(_REPO_ROOT / "optimized_dp")
if _ODP_ROOT not in _sys.path:
    _sys.path.insert(0, _ODP_ROOT)

# Harmless stub when heterocl is absent (we only use the pure-Python methods).
try:
    import heterocl  # noqa: F401
except Exception:
    import types as _types

    _sys.modules["heterocl"] = _types.ModuleType("heterocl")

from odp.Grid import Grid  # noqa: E402  (re-exported)
from odp.dynamics.DubinsCar2 import DubinsCar2  # noqa: E402  (re-exported)

__all__ = ["Grid", "DubinsCar2"]
