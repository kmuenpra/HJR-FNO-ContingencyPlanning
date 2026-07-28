"""
Self-contained predictor for the parameter-conditioned Dubins reach-avoid model "oneobs2".

Dependencies: ONLY `modules.py` (the SIREN network) + torch + numpy.
Does NOT import dataio / diff_operators / utils / loss_functions (no scipy/torchvision/matplotlib).

The model is V_theta(x, t; c) for a Dubins car reaching a fixed target disk (radius 2 at the
origin) while avoiding ONE circular obstacle of fixed radius, conditioned on its center c=(cx,cy).
The reach-avoid set is { (x,y,theta) : V <= 0 } for a given time-to-go t and obstacle center c.

oneobs2 was trained with the difference model (V = NN + max{l,g}); this predictor reconstructs V
the same way. See summary_docs/dubins_oneobstacle_model_summary.md.

Usage
-----
    import predict_oneobs2 as P
    model = P.load_model('dubinsObstacles_scripts/logs/oneobs2/checkpoints/model_final.pth')

    # value at one or many states (arrays broadcast), fixed obstacle center c and time-to-go t:
    V = P.predict_value(model, x=0.3, y=-0.2, theta=0.0, cx=4.0, cy=0.0, t=8.0)   # V<=0 => in set

    # reach-avoid set on a grid (for plotting):
    V_grid, extent = P.reach_avoid_grid(model, t=8.0, theta=0.0, cx=4.0, cy=0.0, sidelen=300)
    brt = V_grid <= 0
"""






'''
Run it
Built-in sanity check (verifies it loads + prints a few predictions):


python param_DeepReach/predict_oneobs.py --ckpt param_DeepReach/model_final.pth


In your own code:

import predict_oneobs2 as P     # run from ~/mymodel, or add it to sys.path

model = P.load_model('model_final.pth')

# value at a single state (V <= 0  ⇒  inside the reach-avoid set)
V = P.predict_value(model, x=0.3, y=-0.2, theta=0.0, cx=4.0, cy=0.0, t=8.0)

# many states at once (x,y,theta broadcast; cx,cy,t scalar)
import numpy as np
V = P.predict_value(model, x=np.array([0.3, -5.0]), y=np.array([-0.2, 1.0]),
                    theta=0.0, cx=4.0, cy=0.0, t=8.0)

# whole set on a grid (for plotting / level set)
V_grid, extent = P.reach_avoid_grid(model, t=8.0, theta=0.0, cx=4.0, cy=0.0, sidelen=300)
brt = V_grid <= 0

Requirements in the new location:
- Python packages: only torch and numpy. (No scipy/torchvision/matplotlib/dataio.)
- GPU is optional — load_model uses CUDA if available, else CPU automatically.
- The checkpoint is a plain state_dict, so it loads across torch versions; any env with torch+numpy works (e.g. param_cond_brt).


Arguments to remember when calling:
- cx, cy — obstacle center (physical, in [-10,10]); t — time-to-go (physical, in [0,8]).
- x, y — physical position in [-10,10]; theta — heading in radians.
- obstacle_radius=1.0 is the default (matches oneobs2). Only change it if you retrain with a different fixed radius.
'''
import os
import sys
import math

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))   # so `import modules` works
import modules


# --- oneobs2 constants (must match dataio.ReachabilityDubinsCarOneObstacleSource) -------------
ALPHA_POS = 10.0            # positions [-10,10] -> [-1,1]
ALPHA_ANGLE = 1.2 * math.pi  # heading  [-1.2pi,1.2pi] -> [-1,1]
ALPHA_CENTER = 10.0         # obstacle center [-10,10] -> [-1,1]
ALPHA_TIME = 8.0            # tMax; physical time [0,8] -> [0,1]
R_TARGET = 2.0              # target disk radius at origin
OBSTACLE_RADIUS = 1.0       # fixed obstacle radius
VAL_MEAN, VAL_VAR, VAL_NORM_TO = 5.0, 8.0, 0.02   # value normalization
DOMAIN = 10.0
_EPS = 1e-12

# network architecture (must match training)
IN_FEATURES = 6            # [t, x', y', theta', cx', cy']
HIDDEN, N_HIDDEN_LAYERS = 512, 3


def load_model(ckpt_path, device=None):
    """Build SingleBVPNet and load oneobs2 weights. Accepts a raw state_dict (model_final.pth /
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


def terminal_cost(x, y, cx, cy, obstacle_radius=OBSTACLE_RADIUS):
    """max{ l, g } in physical units. l = ||p||-R_target (neg inside target);
    g = R_obs - ||p-c|| (pos inside obstacle). Inputs are torch tensors (physical units)."""
    lx = torch.sqrt(x * x + y * y + _EPS) - R_TARGET
    gx = obstacle_radius - torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + _EPS)
    return torch.maximum(lx, gx)


@torch.no_grad()
def predict_value(model, x, y, theta, cx, cy, t,
                  obstacle_radius=OBSTACLE_RADIUS, diffModel=True, device=None):
    """Physical value V(x,t;c). x,y,theta may be scalars or array-likes (broadcast together);
    cx,cy,t,obstacle_radius are scalars. Returns a numpy array (or float for scalar input).
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

    out = model({'coords': coords.to(device)})['model_out'].squeeze(-1).cpu()
    V = out * VAL_VAR / VAL_NORM_TO + VAL_MEAN                       # un-normalize
    if diffModel:                                                   # add back the terminal base
        V = V + terminal_cost(xt, yt, float(cx), float(cy), obstacle_radius) - VAL_MEAN
    V = V.numpy()
    return float(V[0]) if scalar_in else V.reshape(out_shape)


@torch.no_grad()
def reach_avoid_grid(model, t, theta, cx, cy, sidelen=300,
                     obstacle_radius=OBSTACLE_RADIUS, diffModel=True, device=None):
    """Evaluate V on a sidelen x sidelen (x,y) grid over [-DOMAIN,DOMAIN]^2 at fixed time/heading
    /obstacle center. Returns (V_grid [sidelen,sidelen], extent=(-D,D,-D,D)). { V_grid <= 0 } is
    the reach-avoid set slice. V_grid is indexed [ix, iy]; use V_grid.T for imshow(origin='lower')."""
    axis = np.linspace(-DOMAIN, DOMAIN, sidelen, dtype=np.float32)
    gx, gy = np.meshgrid(axis, axis, indexing='ij')
    V = predict_value(model, gx.reshape(-1), gy.reshape(-1), theta, cx, cy, t,
                      obstacle_radius=obstacle_radius, diffModel=diffModel, device=device)
    return V.reshape(sidelen, sidelen), (-DOMAIN, DOMAIN, -DOMAIN, DOMAIN)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Quick check of the oneobs2 predictor.')
    ap.add_argument('--ckpt', default='dubinsObstacles_scripts/logs/oneobs2/checkpoints/model_final.pth')
    ap.add_argument('--obstacle_radius', type=float, default=OBSTACLE_RADIUS)
    args = ap.parse_args()

    m = load_model(args.ckpt)
    print('loaded', args.ckpt)
    # sanity points at t=8, obstacle at (4,0), heading toward target
    for (x, y, th) in [(-6.0, 0.0, math.pi), (4.0, 0.0, 0.0), (0.0, 0.0, 0.0)]:
        v = predict_value(m, x, y, th, cx=4.0, cy=0.0, t=8.0, obstacle_radius=args.obstacle_radius)
        tag = 'IN reach-avoid set' if v <= 0 else 'outside'
        note = '(obstacle interior -> should be >0)' if (x, y) == (4.0, 0.0) else ''
        print('  (x=%+.1f y=%+.1f th=%+.2f): V=%+.3f  [%s] %s' % (x, y, th, v, tag, note))
    Vg, ext = reach_avoid_grid(m, t=8.0, theta=0.0, cx=4.0, cy=0.0, sidelen=200)
    print('grid: %d x %d, reach-avoid fraction = %.3f' % (Vg.shape[0], Vg.shape[1], float((Vg <= 0).mean())))
