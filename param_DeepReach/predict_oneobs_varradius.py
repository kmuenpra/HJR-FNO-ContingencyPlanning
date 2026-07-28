"""
Self-contained predictor for the parameter-conditioned Dubins reach-avoid model "varR_v1"
(VARIABLE-radius, one obstacle).

Dependencies: ONLY `modules.py` (the SIREN network) + torch + numpy.
Does NOT import dataio / diff_operators / utils / loss_functions (no scipy/torchvision/matplotlib).

The model is V_theta(x, t; c, r) for a Dubins car reaching a fixed target disk (radius 2 at the
origin) while avoiding ONE circular obstacle whose CENTER c=(cx,cy) AND RADIUS r are BOTH
conditioning parameters (r in [r_min, r_max] = [0.5, 1.5]). The reach-avoid set is
{ (x,y,theta) : V <= 0 } for a given time-to-go t, obstacle center c, and radius r.

varR_v1 was trained with the difference model (V = NN + max{l,g}); this predictor reconstructs V
the same way. Difference from predict_oneobs.py: the network input has a 7th feature r' and the
obstacle SDF uses the per-query radius r instead of a fixed 1.0.
See summary_docs/dubins_oneobstacle_model_summary.md.

Usage
-----
    import predict_oneobs_varradius as P
    model = P.load_model('dubinsObstacles_scripts/logs/varR_v1/checkpoints/model_final.pth')

    # value at one or many states (arrays broadcast), fixed obstacle center c, radius r, time t:
    V = P.predict_value(model, x=0.3, y=-0.2, theta=0.0, cx=4.0, cy=0.0, r=1.0, t=8.0)  # V<=0 => in set

    # reach-avoid set on a grid (for plotting):
    V_grid, extent = P.reach_avoid_grid(model, t=8.0, theta=0.0, cx=4.0, cy=0.0, r=1.0, sidelen=300)
    brt = V_grid <= 0

Requirements
------------
- Python packages: only torch and numpy. (No scipy/torchvision/matplotlib/dataio.)
- GPU optional: load_model uses CUDA if available, else CPU.
- The checkpoint is a plain state_dict, so it loads across torch versions (e.g. param_cond_brt).

Arguments to remember when calling
----------------------------------
- cx, cy -- obstacle center (physical, in [-10,10]); r -- obstacle radius (physical, in [0.5,1.5]).
- t      -- time-to-go (physical, in [0,8]).
- x, y   -- physical position in [-10,10]; theta -- heading in radians.
- r_min/r_max default to 0.5/1.5 (matches varR_v1). Only change if you retrain with a different range.
"""
import os
import sys
import math

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))   # so `import modules` works
import modules


# --- varR_v1 constants (must match dataio.ReachabilityDubinsCarOneObstacleVarRadiusSource) ----
ALPHA_POS = 10.0             # positions [-10,10] -> [-1,1]
ALPHA_ANGLE = 1.2 * math.pi  # heading  [-1.2pi,1.2pi] -> [-1,1]
ALPHA_CENTER = 10.0          # obstacle center [-10,10] -> [-1,1]
ALPHA_TIME = 8.0             # tMax; physical time [0,8] -> [0,1]
R_TARGET = 2.0               # target disk radius at origin
R_MIN, R_MAX = 0.5, 1.5      # conditioned obstacle-radius range
R_ALPHA = 0.5 * (R_MAX - R_MIN)   # radius (r - r_beta)/r_alpha -> [-1,1]   (= 0.5)
R_BETA = 0.5 * (R_MAX + R_MIN)    #                                          (= 1.0)
VAL_MEAN, VAL_VAR, VAL_NORM_TO = 5.0, 8.0, 0.02   # value normalization
DOMAIN = 10.0
_EPS = 1e-12

# network architecture (must match training)
IN_FEATURES = 7             # [t, x', y', theta', cx', cy', r']
HIDDEN, N_HIDDEN_LAYERS = 512, 3


def load_model(ckpt_path, device=None):
    """Build SingleBVPNet and load varR_v1 weights. Accepts a raw state_dict (model_final.pth /
    model_current.pth) or a bundled epoch checkpoint (model_epoch_*.pth)."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = modules.SingleBVPNet(in_features=IN_FEATURES, out_features=1, type='sine',
                                 mode='mlp', hidden_features=HIDDEN, num_hidden_layers=N_HIDDEN_LAYERS)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:   # epoch checkpoint bundles optimizer etc.
        state = state['model']
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def terminal_cost(x, y, cx, cy, r):
    """max{ l, g } in physical units. l = ||p||-R_target (neg inside target);
    g = r - ||p-c|| (pos inside obstacle, uses the per-query radius r). Inputs are torch tensors
    (physical units); cx, cy, r are scalars (broadcast)."""
    lx = torch.sqrt(x * x + y * y + _EPS) - R_TARGET
    gx = float(r) - torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + _EPS)
    return torch.maximum(lx, gx)


@torch.no_grad()
def predict_value(model, x, y, theta, cx, cy, r, t, diffModel=True, device=None):
    """Physical value V(x,t;c,r). x,y,theta may be scalars or array-likes (broadcast together);
    cx,cy,r,t are scalars. Returns a numpy array (or float for scalar input).
    The reach-avoid set is { V <= 0 }."""
    device = device or next(model.parameters()).device
    scalar_in = np.ndim(x) == 0 and np.ndim(y) == 0 and np.ndim(theta) == 0
    xa, ya, tha = np.broadcast_arrays(np.asarray(x, dtype=np.float32),
                                      np.asarray(y, dtype=np.float32),
                                      np.asarray(theta, dtype=np.float32))
    out_shape = xa.shape
    xt = torch.from_numpy(np.ascontiguousarray(xa).reshape(-1))
    yt = torch.from_numpy(np.ascontiguousarray(ya).reshape(-1))
    tht = torch.from_numpy(np.ascontiguousarray(tha).reshape(-1))

    N = xt.shape[0]
    coords = torch.empty(N, IN_FEATURES, dtype=torch.float32)
    coords[:, 0] = float(t) / ALPHA_TIME
    coords[:, 1] = xt / ALPHA_POS
    coords[:, 2] = yt / ALPHA_POS
    coords[:, 3] = tht / ALPHA_ANGLE
    coords[:, 4] = float(cx) / ALPHA_CENTER
    coords[:, 5] = float(cy) / ALPHA_CENTER
    coords[:, 6] = (float(r) - R_BETA) / R_ALPHA                     # normalized radius

    out = model({'coords': coords.to(device)})['model_out'].squeeze(-1).cpu()
    V = out * VAL_VAR / VAL_NORM_TO + VAL_MEAN                       # un-normalize
    if diffModel:                                                   # add back the terminal base
        V = V + terminal_cost(xt, yt, float(cx), float(cy), float(r)) - VAL_MEAN
    V = V.numpy()
    return float(V[0]) if scalar_in else V.reshape(out_shape)


@torch.no_grad()
def reach_avoid_grid(model, t, theta, cx, cy, r, sidelen=300, diffModel=True, device=None):
    """Evaluate V on a sidelen x sidelen (x,y) grid over [-DOMAIN,DOMAIN]^2 at fixed time/heading
    /obstacle center/radius. Returns (V_grid [sidelen,sidelen], extent=(-D,D,-D,D)). { V_grid <= 0 }
    is the reach-avoid set slice. V_grid is indexed [ix, iy]; use V_grid.T for imshow(origin='lower')."""
    axis = np.linspace(-DOMAIN, DOMAIN, sidelen, dtype=np.float32)
    gx, gy = np.meshgrid(axis, axis, indexing='ij')
    V = predict_value(model, gx.reshape(-1), gy.reshape(-1), theta, cx, cy, r, t,
                      diffModel=diffModel, device=device)
    return V.reshape(sidelen, sidelen), (-DOMAIN, DOMAIN, -DOMAIN, DOMAIN)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Quick check of the varR_v1 (variable-radius) predictor.')
    ap.add_argument('--ckpt', default='dubinsObstacles_scripts/logs/varR_v1/checkpoints/model_final.pth')
    args = ap.parse_args()

    m = load_model(args.ckpt)
    print('loaded', args.ckpt)
    # sanity points at t=8, obstacle at (4,0); sweep the radius to see the carved hole grow
    for r in [R_MIN, 1.0, R_MAX]:
        print('  obstacle radius r = %.2f:' % r)
        for (x, y, th) in [(-6.0, 0.0, math.pi), (4.0, 0.0, 0.0), (0.0, 0.0, 0.0)]:
            v = predict_value(m, x, y, th, cx=4.0, cy=0.0, r=r, t=8.0)
            tag = 'IN reach-avoid set' if v <= 0 else 'outside'
            note = '(obstacle interior -> should be >0)' if (x, y) == (4.0, 0.0) else ''
            print('    (x=%+.1f y=%+.1f th=%+.2f): V=%+.3f  [%s] %s' % (x, y, th, v, tag, note))
        Vg, ext = reach_avoid_grid(m, t=8.0, theta=0.0, cx=4.0, cy=0.0, r=r, sidelen=200)
        print('    grid reach-avoid fraction = %.3f' % float((Vg <= 0).mean()))
