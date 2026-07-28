# =========================
# Standard library imports
# =========================
import os
import time
import math
import operator
import warnings
from functools import reduce, partial
from timeit import default_timer
from pathlib import Path
from typing import Tuple, List, Dict, Union, Iterable, Optional

warnings.filterwarnings("ignore")

# =========================
# Third-party imports
# =========================
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from scipy.ndimage import gaussian_filter

# =========================
# PyTorch imports
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset

# =========================
# Local / project imports
# =========================
from .neural_utils import *


# =========================
# Module-level config
# =========================
# Default device + trained-model path used as defaults by the query helpers.
device    = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PATH = str(Path(__file__).resolve().parent / "training" / "model" / "01_FNO3d_dubins_5ch_tuned.pt")


################################################################
#  3D Fourier layer
################################################################
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.

        modes1: number of Fourier modes to keep along x-dimension
        modes2: number of Fourier modes to keep along y-dimension
        modes3: number of Fourier modes to keep along z/t-dimension
                (at most floor(N/2) + 1 for the last spatial dim)
        """
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)

        # Four weight tensors to cover the four low-frequency corners in 3D.
        # rfftn halves only the LAST axis (z/t), so kz >= 0 always.
        # The four corners span (±kx, ±ky, +kz):
        #   weights1 → (+kx, +ky, +kz)
        #   weights2 → (-kx, +ky, +kz)
        #   weights3 → (+kx, -ky, +kz)
        #   weights4 → (-kx, -ky, +kz)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3,
                                    dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        # (batch, in_ch, x, y, z), (in_ch, out_ch, x, y, z) -> (batch, out_ch, x, y, z)
        # Contracts over in_ch ('i'), keeps all three spatial freq axes ('x','y','z')
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # x shape: (batch, in_ch, X, Y, Z)

        # 1. 3D real FFT → (batch, in_ch, X, Y, Z//2+1)
        #    rfftn halves only the LAST axis (z/t), giving the non-redundant
        #    half of the conjugate-symmetric 3D spectrum.
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # 2. Allocate output frequency tensor (same half-spectrum shape)
        out_ft = torch.zeros(batchsize, self.out_channels,
                             x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             device=x.device, dtype=torch.cfloat)

        # 3. Multiply the four low-frequency corners by learned weights.
        #
        #    (+kx, +ky, +kz)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3],
                             self.weights1)
        #    (-kx, +ky, +kz)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3],
                             self.weights2)
        #    (+kx, -ky, +kz)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3],
                             self.weights3)
        #    (-kx, -ky, +kz)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3],
                             self.weights4)

        # 4. Inverse 3D real FFT back to physical space.
        #    s=(X, Y, Z) ensures the output spatial size matches the input
        #    exactly, compensating for the halved last axis.
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


################################################################
#  FNO3d — 4-layer Fourier Neural Operator for 3D problems
################################################################
class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, in_channels=3):
        super(FNO3d, self).__init__()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desired channel dimension by self.lifting.
        2. 4 layers of the integral operators u' = (W + K)(u).
               W defined by self.w*    (pointwise Conv3d, kernel=1)
               K defined by self.conv* (SpectralConv3d)
        3. Project from channel space to output space by self.projection.

        input:  solution + coordinates (e.g. a(x,y,t), x, y, t)  — arbitrary c_in channels
        input shape:  (batch, X, Y, T, in_channels)
        output: solution / quantity of interest on the same (X, Y, T) grid
        output shape: (batch, X, Y, T, 1)

        in_channels: number of input features (default 3; e.g. 4 if you concatenate
                     [a(x,y,t), x, y, t] as for the HJI reach-avoid setup)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width  = width

        # ------- Lifting -------
        # Maps in_channels → width.  Conv3d(kernel=1) = pointwise operation,
        # identical in role to the 2D version but now over a 3D spatio-temporal grid.
        self.lifting    = nn.Conv3d(in_channels, self.width, 1)

        # ------- Fourier blocks -------
        # Spectral (K) paths
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)

        # Bypass (W) paths — pointwise Conv3d
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        # ------- Projection -------
        self.projection = nn.Conv3d(self.width, 1, 1)

    def forward(self, x):
        # x: (batch, X, Y, T, in_channels)

        # --- Lifting ---
        # Permute to channel-first for Conv3d: (batch, in_channels, X, Y, T)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.lifting(x)                          # → (batch, width, X, Y, T)

        # --- Fourier Block 0 ---
        x1 = self.conv0(x)                           # spectral path  K(u)
        x2 = self.w0(x)                              # bypass path    W(u)
        x  = x1 + x2
        x  = F.relu(x)

        # --- Fourier Block 1 ---
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x  = x1 + x2
        x  = F.relu(x)

        # --- Fourier Block 2 ---
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x  = x1 + x2
        x  = F.relu(x)

        # --- Fourier Block 3 ---
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x  = x1 + x2
        x  = F.relu(x)

        # --- Projection ---
        x = self.projection(x)                       # → (batch, 1, X, Y, T)
        x = x.permute(0, 2, 3, 4, 1)                 # → (batch, X, Y, T, 1)
        return x


################################################################
#  Query a trained FNO3d (5-channel: SDF, x, y, t, theta) — RAW, resolution-flexible
#
#  query_FNO3d        : one fixed theta  -> (Nx, Ny, T)
#  query_FNO3d_full   : sweep all theta  -> (Nx, Ny, Nth, T)   (full 3D BRT)
#
#  Resolution-invariant: pass any x_axis / y_axis / tau / theta_axis. The FNO
#  works at any grid size that satisfies its mode minimums:
#     Nx, Ny >= 2*modes1 (= 32 for modes1=16),   T >= 2*modes3 - 1 (= 15 for modes3=8)
################################################################

def _build_input_5ch(constraint_sdf, theta_vals, x_axis, y_axis, tau):
    """Return a torch tensor (B, Nx, Ny, T, 5) for B = len(theta_vals).
       Channels [sdf, x, y, t, theta] in RAW units (matches training)."""
    x_axis = np.asarray(x_axis, np.float32)
    y_axis = np.asarray(y_axis, np.float32)
    tau    = np.asarray(tau,    np.float32)
    theta_vals = np.atleast_1d(np.asarray(theta_vals, np.float32))
    Nx, Ny, T, B = len(x_axis), len(y_axis), len(tau), len(theta_vals)
    assert constraint_sdf.shape == (Nx, Ny), \
        f"constraint_sdf {constraint_sdf.shape} != grid ({Nx},{Ny})"

    sdf_neg = torch.tensor(-np.asarray(constraint_sdf, np.float32))     # ch0 = -g
    Xg, Yg  = np.meshgrid(x_axis, y_axis, indexing="ij")
    sdf_3d = sdf_neg.unsqueeze(-1).expand(Nx, Ny, T)
    x_3d   = torch.tensor(Xg).unsqueeze(-1).expand(Nx, Ny, T)
    y_3d   = torch.tensor(Yg).unsqueeze(-1).expand(Nx, Ny, T)
    t_3d   = torch.tensor(tau).view(1, 1, T).expand(Nx, Ny, T)

    base = torch.stack([sdf_3d, x_3d, y_3d, t_3d,
                        torch.zeros(Nx, Ny, T)], dim=-1)               # (Nx,Ny,T,5)
    x_in = base.unsqueeze(0).repeat(B, 1, 1, 1, 1)                     # (B,Nx,Ny,T,5)
    for b, th in enumerate(theta_vals):
        x_in[b, :, :, :, 4] = float(th)                               # constant theta channel
    return x_in


def query_FNO3d(constraint_sdf, theta_val, model=None,
                x_axis=None, y_axis=None, tau=None,
                return_all_times=True, t_idx=-1, device=device):
    """Single fixed-theta query -> (Nx, Ny, T) (or (Nx,Ny) if return_all_times=False).

    x_axis / y_axis / tau must be provided explicitly when called outside a
    notebook context (no dataset globals available in this module).
    """
    if model is None:
        model = torch.load(SAVE_PATH, map_location=device, weights_only=False)
    model.eval()
    if x_axis is None or y_axis is None or tau is None:
        raise ValueError("query_FNO3d: x_axis, y_axis and tau must be provided.")

    x_in = _build_input_5ch(constraint_sdf, [theta_val], x_axis, y_axis, tau)
    with torch.no_grad():
        V = model(x_in.to(device))[0, :, :, :, 0].float().cpu().numpy()   # (Nx,Ny,T)
    return V if return_all_times else V[:, :, t_idx]


def query_FNO3d_full(constraint_sdf, model=None,
                     x_axis=None, y_axis=None, theta_axis=None, tau=None,
                     device=device, chunk=12):
    """Full 3D BRT: sweep every theta, stack -> (Nx, Ny, Nth, T).
       theta is batched (chunked) so it's one (or a few) forward passes, not Nth.

    x_axis / y_axis / theta_axis / tau must be provided explicitly when called
    outside a notebook context (no dataset globals available in this module).
    """
    if model is None:
        model = torch.load(SAVE_PATH, map_location=device, weights_only=False)
    model.eval()
    if x_axis is None or y_axis is None or theta_axis is None or tau is None:
        raise ValueError("query_FNO3d_full: x_axis, y_axis, theta_axis and tau must be provided.")
    theta_axis = np.asarray(theta_axis, np.float32)
    Nx, Ny, Nth, T = len(x_axis), len(y_axis), len(theta_axis), len(tau)

    V_full = np.empty((Nx, Ny, Nth, T), dtype=np.float32)
    for s in range(0, Nth, chunk):
        th_chunk = theta_axis[s:s + chunk]
        x_in = _build_input_5ch(constraint_sdf, th_chunk, x_axis, y_axis, tau)  # (B,Nx,Ny,T,5)
        with torch.no_grad():
            out = model(x_in.to(device))[..., 0].float().cpu().numpy()          # (B,Nx,Ny,T)
        V_full[:, :, s:s + len(th_chunk), :] = np.moveaxis(out, 0, 2)
    return V_full


# =====================================================================
# Torch-free numerics + scenario-optimization logic live in scenario_worker.py
# (importable without torch, so a ProcessPoolExecutor can run scenario opt in
# spawned worker processes). It also performs the odp sys.path injection +
# heterocl stub and re-exports odp's Grid / DubinsCar2. Import everything back
# here so existing module-level references keep resolving unchanged — there is
# exactly ONE implementation of each, living in scenario_worker.py.
# =====================================================================
from .scenario_worker import (
    Grid, DubinsCar2,
    upwindFirstENO2, upwindFirstWENO5, add_ghost_cells, strip_dim,
    computeGradients, eval_u,
    _scenario_required_N, _scenario_wrap_pi, _ReachValueCache,
    rollout_cost, sample_states, scenario_delta_hat_worker,
    _scenario_pool_initializer,
)


# ---------------------------------------------------------------------
# Thin shim that exposes the OLD `Plane` API on top of odp's DubinsCar2.
# All optimal-control / disturbance / dynamics math is delegated to
# DubinsCar2; this only adds the mutable-state + Euler-update convenience
# that the contingency rollout relied on, so the ported methods below read
# almost verbatim against the original HJR_FNO.py.
# ---------------------------------------------------------------------
class PlaneDynamics:
    def __init__(self, car: DubinsCar2, x0=(0.0, 0.0, 0.0)):
        self.car   = car
        self.x     = np.array(x0, dtype=float)
        self.vrange = [car.speedMin, car.speedMax]
        self.wMax   = car.wMax
        self.dMax   = np.array(car.dMax, dtype=float)

    def dynamics(self, t, x, u, d):
        return np.array(self.car.dynamics_inPython(x, u, d), dtype=float)

    def optCtrl(self, t, x, deriv, uMode='min'):
        self.car.uMode = uMode
        return np.asarray(self.car.optCtrl_inPython(x, deriv), dtype=float)

    def optDstb(self, t, x, deriv, dMode='max'):
        self.car.dMode = dMode
        return np.asarray(self.car.optDstb_inPython(x, deriv), dtype=float)

    def updateState(self, u, dt, d):
        dx = self.dynamics(0, self.x, u, d)
        self.x = self.x + dx * dt

    def optCtrl_grid(self, deriv, theta_grid, uMode='min'):
        """Vectorized bang-bang control over a whole grid (DubinsCar2 has no
        vectorized variant, so the original Plane logic is reproduced)."""
        Vx, Vy, Vtheta = deriv[0], deriv[1], deriv[2]
        det1 = Vx * np.cos(theta_grid) + Vy * np.sin(theta_grid)
        if uMode == 'max':
            v_opt     = np.where(det1   >= 0, self.vrange[1], self.vrange[0])
            omega_opt = np.where(Vtheta >= 0, self.wMax,      -self.wMax)
        else:  # 'min'
            v_opt     = np.where(det1   >= 0, self.vrange[0], self.vrange[1])
            omega_opt = np.where(Vtheta >= 0, -self.wMax,     self.wMax)
        return v_opt, omega_opt


#---------------------------------------
# Class 5: Neural Operator-based HJ reachability  (FNO3d backend)
#---------------------------------------
class HJR_FNO:
    """
    Wrapper class for Hamilton-Jacobi Reachability prediction using a trained
    3-D Fourier Neural Operator (FNO3d, theta as a constant input channel).

    Public surface (attributes + methods) is kept identical to the original
    HJR_FNO.py so rrtx.py / plotting.py only need their import lines changed.
    """

    def __init__(self, env, safe_regions, Tf_reach, device='cuda',
                 scenario_enable=True, scenario_eps=0.05, scenario_beta=1e-9,
                 scenario_M=30):
        # Import here to avoid a circular dependency (utils imports plotting/env)
        import utils as utils_module

        self.device = device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu'

        # ---- Load the trained 3-D FNO ------------------------------------
        save_path = SAVE_PATH
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"HJR-FNO model not found at {save_path}")
        print("Loading saved HJR-FNO (FNO3d) model...")
        # The checkpoint was pickled with FNO3d / SpectralConv3d defined in
        # __main__ (the training script). Make them resolvable there so
        # torch.load works regardless of the entry point (rrtx, notebook, test).
        import __main__ as _main
        _main.FNO3d = FNO3d
        _main.SpectralConv3d = SpectralConv3d
        self.model = torch.load(save_path, map_location=self.device, weights_only=False)
        self.model.to(self.device)
        self.model.eval()

        # ---- Environment / utils (unchanged) -----------------------------
        self.env = env
        self.utils = utils_module.Utils(environment=env)

        # ---- Dynamics: odp DubinsCar2 (matches data-gen settings) --------
        #   v in [0,1], w in [-1,1], uMode=min, dMode=max.
        #   dMax=0 for the ROLLOUT (optCtrl is disturbance-independent; the
        #   simulated environment applies no wind). wMax=1.0 matches the value
        #   function the model was trained on (old code used 1.5).
        self.car = DubinsCar2(
            x=[0, 0, 0],
            uMin=[0.0, -1.0], uMax=[1.0, 1.0],
            dMax=[0.0, 0.0, 0.0],
            uMode="min", dMode="max",
        )
        self.plane = PlaneDynamics(self.car)
        self.wMax   = self.plane.wMax
        self.vrange = self.plane.vrange
        self.dMax   = self.plane.dMax

        # ---- Grids (odp Grid). TWO grids with DIFFERENT theta conventions: -
        #   g      (FNO model):  theta in [-pi, pi), 36 pts  — matches training.
        #   g_fine (precomputed): theta in [0, 2pi),  25 pts — matches the EXACT
        #                         obstacle-free BRT .mat loaded below.
        # They are intentionally distinct; this is what N_fine / g_fine is for.
        # The dual-grid code branches on obstacle presence (g_fine when a region
        # has no obstacles yet, g once the FNO has been queried), and every
        # theta lookup wraps into the *grid's own* range via _wrap_to_grid_theta.
        self.pd = [2]  # theta is periodic

        # FNO grid (theta in [-pi, pi), 36 pts)
        self.grid_min = np.array([-10.0, -10.0, -math.pi])
        self.grid_max = np.array([ 10.0,  10.0,  math.pi])
        self.N = np.array([40, 40, 30, 17])              # [x, y, theta, time]
        self.g = Grid(self.grid_min, self.grid_max, 3, self.N[:3], self.pd)

        # Fine grid — matches the precomputed obstacle-free .mat (theta in
        # [0, 2pi), 25 pts; the .mat is 50x50x25x33, time subsampled ::2 -> 17).
        self.grid_min_fine = np.array([-10.0, -10.0, 0.0])
        self.grid_max_fine = np.array([ 10.0,  10.0, 2 * math.pi])
        self.N_fine = np.array([50, 50, 25, 17])
        self.g_fine = Grid(self.grid_min_fine, self.grid_max_fine, 3, self.N_fine[:3], self.pd)

        # ---- Coordinate axes (theta taken from each grid's own range) ----
        self.theta_array      = np.asarray(self.g.grid_points[2], dtype=float)       # [-pi, pi), 36
        self.theta_array_fine = np.asarray(self.g_fine.grid_points[2], dtype=float)  # [0, 2pi), 25
        self.theta_min = float(self.theta_array.min())
        self.theta_max = float(self.theta_array.max())

        self.t0 = 0
        self.tf = 8  # finite-time horizon (must match training tau and the .mat time span)
        self.time_res = self.N[3]
        self.time_array      = np.linspace(self.t0, self.tf, self.N[3])
        self.time_array_fine = np.linspace(self.t0, self.tf, self.N_fine[3])

        # ---- Spatial meshgrids (used by plotting.py) ---------------------
        self.X_fine, self.Y_fine = np.meshgrid(
            self.g_fine.grid_points[0], self.g_fine.grid_points[1], indexing="ij")
        self.X, self.Y = np.meshgrid(
            self.g.grid_points[0], self.g.grid_points[1], indexing="ij")
        self.env_extent = [self.grid_min[0], self.grid_max[0],
                           self.grid_min[1], self.grid_max[1]]

        # ---- Safe regions ------------------------------------------------
        self.num_safe_regions = len(safe_regions)
        self.safe_regions = np.array(safe_regions)

        # Per-region obstacle bookkeeping (obstacle SDFs live on the FNO grid g)
        self.obs_SDF  = [np.empty(tuple(self.N[:3])) for _ in range(self.num_safe_regions)]
        self.obs_list = [[] for _ in range(self.num_safe_regions)]

        # Target/safe set (same for every region): cylinder r=2 at origin
        self.safeSet_SDF = -self.shapeCylinder(ignoreDims=[2], center=[0, 0, 0], radius=2)

        # ---- Obstacle-free reachable tube (EXACT, precomputed) -----------
        # Loaded from the odp HJSolver result (GROUND TRUTH), NOT predicted by
        # the FNO — a no-obstacle scene is out-of-distribution for the model.
        # It lives on the FINE grid (theta in [0, 2pi), 25 pts; time 33 -> ::2
        # -> 17), which is exactly why g_fine / N_fine differ from g / N.
        mat_path = Path(__file__).resolve().parent / "training" / "HJB_training_mat" / "50_50_25_SDF_no_obs.mat"
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Obstacle-free BRT .mat not found at {mat_path}")
        print(f"Loading exact obstacle-free reachable tube from {mat_path} ...")
        mat_data = loadmat(str(mat_path))
        data_safe = np.asarray(mat_data['BRT_all'][0][0])   # (50, 50, 25, 33)
        data_safe = data_safe[..., ::2]                     # -> (50, 50, 25, 17)
        # The precomputed .mat stores time with index -1 = fully-grown; reverse it so
        # index 0 = fully-grown, matching the FNO-predicted convention (see _grown_time_index).
        data_safe = data_safe[..., ::-1]
        assert data_safe.shape[:3] == tuple(self.N_fine[:3]), (
            f"obstacle-free .mat shape {data_safe.shape[:3]} != N_fine {tuple(self.N_fine[:3])}")
        self.true_reach_obsFree = np.ascontiguousarray(data_safe)   # (Nx, Ny, Nth_fine, T) numpy

        # ---- Initialize reachable sets (obstacle-free, on the FINE grid) -
        self.HJR_sets = [self.true_reach_obsFree.copy() for _ in range(self.num_safe_regions)]
        print("Finished initializing reachable sets for contingency plan.")

        # ---- Feasible region (min over theta at Tf_reach slice) ----------
        self.Tf_reach = Tf_reach
        self.feasible_region = []
        Tf_slice = self._grown_time_index(self.time_array_fine)  # index 0 = fully grown
        for reach_i in self.HJR_sets:
            self.feasible_region.append(np.max(reach_i[..., Tf_slice], axis=2))

        # Frozen obstacle-free snapshot used by is_feasible() — never updated.
        self.feasible_region_init = [fr.copy() for fr in self.feasible_region]

        # Per-reachable-set safe margin (one entry per safe region): ensures the
        # safe set is within V(x,y) <= safe_margin < 0. Initialized to the same
        # tuned value for every region; can be set independently per region.
        # NOTE: every reference to self.safe_margin must now index by region.
        # When scenario certification is enabled, update_obs() overwrites the
        # entry for any region whose set is re-estimated (see _scenario_delta_hat).
        self.safe_margin = [0 for _ in range(self.num_safe_regions)]

        # Source for feasibility values: "HJR_sets" (4D, theta-interpolated -> Option B)
        # or "feasible_region" (2D, theta-marginalized snapshot). One-line switch.
        self.feasibility_source = "feasible_region"
        # Safety buffer subtracted from safe_margin in the feasibility test (tune later).
        self.feasibility_buffer = 0.0 
        # Region indices whose reachable set was re-predicted in the most recent
        # update_obs(); shared across all RRTX trees for non-local tree re-validation.
        self._last_changed_regions = []

        # Interpolator caches (Fix #1): building a RegularGridInterpolator per is_feasible call
        # was a major cost. Cache per region and reuse; invalidate changed regions in update_obs.
        #   _interp_cache[region]      = (source_tag, interp)  -> Check 2b (HJR_sets/feasible_region)
        #   _interp_init_cache[region] = interp                -> Check 2a (frozen obstacle-free, g_fine)
        self._interp_cache = {}
        self._interp_init_cache = {}

        # ---- Scenario optimization config (per-obstacle delta_hat) -------
        # When a region's reachable set is re-estimated in update_obs(), run a
        # scenario optimization ("per_c") to certify a probabilistic safe margin
        # delta_hat and store it as self.safe_margin[region].
        #   N = required_N(eps, beta) samples per iteration (scenario theorem).
        #   With eps=1e-2, beta=1e-9 -> N ~ 4344, so each call is fairly heavy
        #   (M rounds of N rejection-sampled rollouts). Tune eps up / M down for
        #   speed, or set scenario_enable=False to keep the static margin.
        #   For a JOINT guarantee across all regions, use beta/num_safe_regions.
        self.scenario_enable      = scenario_enable
        self.scenario_eps         = scenario_eps   # admissible violation level
        self.scenario_beta        = scenario_beta  # confidence (per region)
        self.scenario_M           = scenario_M     # (unused by the robust sweep; kept for API compat)
        self.scenario_max_tries   = 400      # rejection-sampling attempts per batch
        self.scenario_delta_floor = -1.20    # lower clamp / bottom of the delta sweep
        self.scenario_delta_init  = 0.0      # top of the delta sweep ({V < delta_init})
        self.scenario_delta_step  = 0.05     # descending delta-grid resolution (robust sweep)
        self.scenario_step_frac   = 0.8      # (unused by the robust sweep; kept for API compat)
        self.scenario_seed        = 0
        self.scenario_verbose     = False    # True -> print per-delta-level trace (k/N, eps_hat)
        # Worst-case-disturbance car for the certification rollout: dMax matches
        # the value function's training (0.1), unlike self.car (dMax=0 rollout).
        self.scenario_car = DubinsCar2(
            uMin=[0.0, -1.0], uMax=[1.0, 1.0], dMax=[0.1, 0.1, 0.1],
            uMode="min", dMode="max",
        )

        # ---- Parallel scenario optimization across changed regions -------
        # When >1 region is re-estimated in a single update_obs(), certify their
        # delta_hat concurrently in a spawned ProcessPoolExecutor (workers are
        # torch-free: see scenario_worker.py). Set scenario_parallel=False to
        # force the serial path. max_workers capped for RAM headroom.
        self.scenario_parallel    = True
        self.scenario_max_workers = 6
        self._scenario_pool   = None
        self._scenario_pool_n = 0
        import atexit
        atexit.register(self.shutdown_scenario_pool)

    # =================================================================
    # Small helpers
    # =================================================================
    @staticmethod
    def _wrap_to_grid_theta(theta, grid):
        """Wrap an angle into the periodic theta range of `grid`,
        i.e. [grid.min[2], grid.min[2] + 2*pi). This matters because the FNO
        grid uses theta in [-pi, pi) while the precomputed obstacle-free grid
        (g_fine) uses [0, 2pi); a single fixed wrap would mis-index one of them.
        """
        lo = float(grid.min[2])
        return lo + (np.asarray(theta, dtype=float) - lo) % (2 * np.pi)

    def _grown_time_index(self, time_array):
        """Time-slice index for horizon self.Tf_reach under the convention
        index 0 = fully-grown reachable set (index increases -> shrinks to target).
        Returns 0 when Tf_reach == max(time_array); flips the ascending-time argmin.
        """
        T = len(time_array)
        return (T - 1) - int(np.argmin(np.abs(np.asarray(time_array) - self.Tf_reach)))

    # =================================================================
    # Prediction
    # =================================================================
    def predict(self, sdf_input, theta_hyparam, time_hyparam, g=None):
        """
        Predict a time-varying reachable set from an obstacle SDF.

        @params
        - sdf_input     : obstacle SDF g (negative inside obstacle), 2-D (Nx,Ny)
                          or 3-D (Nx,Ny,Ntheta) (theta-independent -> slice 0 used)
        - theta_hyparam : 1-D array of heading values
        - time_hyparam  : 1-D array of time values
        - g             : grid (default self.g)

        @return torch.Tensor of shape (Nx, Ny, Ntheta, T)
        """
        if g is None:
            g = self.g

        sdf = sdf_input
        if torch.is_tensor(sdf):
            sdf = sdf.cpu().numpy()
        sdf = np.asarray(sdf, dtype=np.float32)
        if sdf.ndim == 3:
            sdf = sdf[:, :, 0]   # obstacle is theta-independent

        # Training convention: network channel0 = -constraints (negated obstacle
        # SDF). query_FNO3d_full computes channel0 = -constraint_sdf internally,
        # so we pass the ORIGINAL obstacle SDF (= constraints = what shapeCylinder
        # returns, negative inside the obstacle). Verified against the dataset GT
        # (mean abs err ~0.015) — see docs/HJR_FNO3d_refactor_plan.md.
        V_full = query_FNO3d_full(
            constraint_sdf=sdf,
            model=self.model,
            x_axis=np.asarray(g.grid_points[0], dtype=np.float32),
            y_axis=np.asarray(g.grid_points[1], dtype=np.float32),
            theta_axis=np.asarray(theta_hyparam, dtype=np.float32).ravel(),
            tau=np.asarray(time_hyparam, dtype=np.float32).ravel(),
            device=self.device,
        )  # (Nx, Ny, Ntheta, T)

        return torch.from_numpy(np.asarray(V_full, dtype=np.float32))

    def shapeCylinder(self, ignoreDims=None, center=None, radius=1.0, g=None):
        """Cylindrical signed distance field on the grid (negative inside)."""
        if g is None:
            g = self.g
        dim = g.dims

        if ignoreDims is None:
            ignoreDims = []
        if center is None:
            center = np.zeros(dim)
        ignoreDims = set(ignoreDims)

        # odp Grid: g.vs[i] is reshaped for broadcasting, so the sum below
        # naturally broadcasts up to the full grid shape.
        dist_squared = np.zeros(tuple(g.pts_each_dim))
        for i in range(dim):
            if i not in ignoreDims:
                dist_squared = dist_squared + (g.vs[i] - center[i]) ** 2

        dist = np.sqrt(dist_squared)
        sdf = dist - radius
        return sdf

    def update_obs(self, obs_cir: List):
        """Update obstacle SDFs with newly detected obstacles and re-predict."""
        # NOTE only circular obstacles (training assumed cylinders).
        changed_regions = []  # region indices whose reachable set is re-predicted in this call
        for i in range(self.num_safe_regions):
            x_offset, y_offset, _ = self.safe_regions[i]
            update_HJR_set = False

            for obs in obs_cir:
                x, y, r = obs
                center = np.array([x - x_offset, y - y_offset, 0])
                within_bound = np.all(center >= self.grid_min) and np.all(center <= self.grid_max)

                if within_bound:
                    update_HJR_set = True
                    obs_sdf = self.shapeCylinder(ignoreDims=[2], center=center, radius=r)
                    if not self.obs_list[i]:
                        self.obs_SDF[i] = obs_sdf
                    else:
                        self.obs_SDF[i] = np.minimum(self.obs_SDF[i], obs_sdf)
                    self.obs_list[i].append(obs)

            #NOTE when predicting FNO, we use g(x) <= 0, since that is how FNO was trained (negative inside <-> obstacle collision)
            # During reach-avoid theory, g(x) >= 0 is the right convention for being inside the obstacles shape
            if update_HJR_set:
                changed_regions.append(i)  # mark region for downstream non-local tree re-validation
                self.HJR_sets[i] = self.predict(
                    sdf_input=self.obs_SDF[i],
                    theta_hyparam=self.theta_array,
                    time_hyparam=self.time_array,
                )

                # Refresh feasible_region for this region from the newly predicted set:
                # max over theta at the fully-grown slice. NOTE now on the coarse FNO grid
                # (g), unlike the obstacle-free init (g_fine); plotting/consumers must pick
                # the grid by obs_list[i]. feasible_region_init stays the frozen g_fine snapshot.
                reach_pred = self.HJR_sets[i]
                if torch.is_tensor(reach_pred):
                    reach_pred = reach_pred.cpu().numpy()
                Tf_slice = self._grown_time_index(self.time_array)
                self.feasible_region[i] = np.max(reach_pred[..., Tf_slice], axis=2)

                # its HJR_sets/feasible_region just changed, so the cached interp is stale.
                self._interp_cache.pop(i, None)

        # Phase 2: certify a probabilistic safe margin (delta_hat) for every
        # region re-estimated above, via scenario optimization. Each region is
        # independent, so when >1 changed (and scenario_parallel) they run
        # concurrently in a spawned, torch-free worker pool; otherwise serial.
        # Reuses the just-computed value tubes (self.HJR_sets[i]).
        if self.scenario_enable and changed_regions:
            self._certify_safe_margins(changed_regions)

        # feasible_region[i] is refreshed above for re-predicted regions (coarse grid,
        # obstacle-aware). feasible_region_init remains the frozen obstacle-free g_fine
        # snapshot used by is_feasible() Check 2a. HJR_sets is queried directly in
        # is_feasible() Check 2b for the reach-avoid sublevel check.

        self._last_changed_regions = changed_regions  # record for tree re-validation (shared across all RRTX trees)
        return changed_regions

    # =================================================================
    # Scenario optimization -> per-region safe margin (delta_hat)
    #
    # The numerical heavy lifting lives in torch-free free functions in
    # scenario_worker.py (rollout_cost / sample_states / scenario_delta_hat_worker)
    # so it can run in spawned worker processes. The methods below are thin
    # wrappers that pull config/grid/car off `self` and delegate, keeping the
    # original call sites working and the serial path byte-identical to parallel.
    # =================================================================
    def _scenario_cfg(self):
        """Pack the picklable scalars/arrays the worker needs (no `self`, no torch)."""
        return dict(
            eps=self.scenario_eps, beta=self.scenario_beta, M=self.scenario_M,
            max_tries=self.scenario_max_tries, delta_floor=self.scenario_delta_floor,
            delta_init=self.scenario_delta_init, delta_step=self.scenario_delta_step,
            step_frac=self.scenario_step_frac, seed=self.scenario_seed,
            dt=float(self.time_array[1] - self.time_array[0]),
            grid_min=np.asarray(self.grid_min), grid_max=np.asarray(self.grid_max),
        )

    def _scenario_target2d(self):
        """2-D target SDF (cylinder r=2 at origin, <0 inside) on the FNO grid g.
        Same for every region, so consumers can compute it once."""
        return self.shapeCylinder(ignoreDims=[2], center=[0, 0, 0], radius=2, g=self.g)[:, :, 0]

    def _scenario_rollout_cost(self, cache, s0, delta=0.0):
        """Thin wrapper -> scenario_worker.rollout_cost (vectorized reach-avoid cost)."""
        dt = float(self.time_array[1] - self.time_array[0])
        return rollout_cost(cache, s0, self.scenario_car, dt, delta=delta)

    def _scenario_sample(self, cache, n_target, delta, rng, batch=8192, max_tries=400):
        """Thin wrapper -> scenario_worker.sample_states (rejection sampling V<delta)."""
        return sample_states(cache, n_target, delta, rng,
                             self.grid_min, self.grid_max,
                             batch=batch, max_tries=max_tries)

    def _scenario_delta_hat(self, V_full, obstacle_sdf, verbose=True):
        """Thin wrapper -> scenario_worker.scenario_delta_hat_worker. Runs the
        per-obstacle scenario optimization IN-PROCESS and returns delta_hat
        (the worker's (delta, report) is unwrapped to just delta here for
        backward compat; use scenario_delta_hat_worker directly for the report)."""
        if torch.is_tensor(V_full):
            V_full = V_full.cpu().numpy()
        delta, _report = scenario_delta_hat_worker(
            np.asarray(V_full, dtype=np.float32), obstacle_sdf, self._scenario_target2d(),
            self.g.grid_points[0], self.g.grid_points[1], self.theta_array,
            self.g, self.scenario_car, self._scenario_cfg(), verbose=verbose)
        return delta

    @staticmethod
    def _format_scenario_report(i, delta, rep):
        """One-line per-region certification summary: N, k, success rate, eps."""
        base = f"  [scenario] region {i}: delta_hat = {delta:.4g}"
        if rep is None:
            return base
        N, k, kmax = rep.get('N'), rep.get('k'), rep.get('kmax')
        cert = "certified" if rep.get('certified') else "FLOOR (uncertified)"
        if k is None:
            return (f"{base} | N={N} k=n/a k_max={kmax} | "
                    f"eps_hat=n/a (target {rep.get('eps'):.1e}) | {cert}")
        sr, eh = rep.get('success_rate'), rep.get('eps_hat')
        return (f"{base} | N={N} k={k}/{N} (k_max={kmax}) "
                f"success={sr:.2%} | eps_hat={eh:.3e} (target {rep.get('eps'):.1e}) | {cert}")

    # =================================================================
    # Parallel orchestration of per-region scenario optimization
    # =================================================================
    def _get_scenario_pool(self, n_jobs):
        """Lazily create (and reuse) a spawned ProcessPoolExecutor sized for the
        number of concurrent regions. Workers are torch-free (scenario_worker)."""
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp

        n = max(1, min(int(n_jobs), int(self.scenario_max_workers)))
        if self._scenario_pool is not None and self._scenario_pool_n >= n:
            return self._scenario_pool
        if self._scenario_pool is not None:
            self._scenario_pool.shutdown(wait=False, cancel_futures=True)
            self._scenario_pool = None

        # Ensure spawned children can import the HJR_FNO package: under spawn the
        # child starts a fresh interpreter and does NOT inherit the parent's
        # runtime sys.path edits, so put the repo root on PYTHONPATH (inherited).
        repo_root = str(Path(__file__).resolve().parent.parent)
        pp = os.environ.get("PYTHONPATH", "")
        if repo_root not in pp.split(os.pathsep):
            os.environ["PYTHONPATH"] = repo_root + (os.pathsep + pp if pp else "")
        # Pin BLAS/OMP to 1 thread in children to avoid oversubscription. The
        # parent's BLAS is already loaded so this no-ops here, but children spawn
        # fresh and inherit these env vars before importing numpy/OpenBLAS.
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(v, "1")

        ctx = mp.get_context("spawn")
        self._scenario_pool = ProcessPoolExecutor(
            max_workers=n, mp_context=ctx, initializer=_scenario_pool_initializer)
        self._scenario_pool_n = n
        return self._scenario_pool

    def shutdown_scenario_pool(self):
        """Tear down the worker pool (registered with atexit; safe to call twice)."""
        pool = getattr(self, "_scenario_pool", None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
            self._scenario_pool = None
            self._scenario_pool_n = 0

    def _certify_safe_margins(self, regions):
        """Certify delta_hat for each changed region and store in self.safe_margin.
        Runs concurrently in the worker pool when >1 region changed (and
        scenario_parallel); otherwise serial. Per-region failures keep the old
        margin (same fallback as the original inline code)."""
        regions = list(regions)

        # Shared (picklable) inputs — same for every region, computed once.
        target2d = self._scenario_target2d()
        cfg      = self._scenario_cfg()
        x_axis   = np.asarray(self.g.grid_points[0])
        y_axis   = np.asarray(self.g.grid_points[1])
        theta    = np.asarray(self.theta_array)

        def _args(i):
            V = self.HJR_sets[i]
            if torch.is_tensor(V):
                V = V.cpu().numpy()
            return (np.asarray(V, dtype=np.float32), np.asarray(self.obs_SDF[i]),
                    target2d, x_axis, y_axis, theta, self.g, self.scenario_car, cfg)

        # Serial path: 1 region, or parallel disabled. Runs the worker in-process
        # so we get the (delta, report) tuple directly.
        if len(regions) <= 1 or not self.scenario_parallel:
            for i in regions:
                try:
                    delta, rep = scenario_delta_hat_worker(*_args(i), self.scenario_verbose)
                    self.safe_margin[i] = delta
                    print(self._format_scenario_report(i, delta, rep))
                except Exception as exc:
                    print(f"  [scenario] region {i}: failed ({exc}); "
                          f"keeping safe_margin = {self.safe_margin[i]}")
            return

        # Parallel path: one job per region, results collected as (delta, report).
        pool = self._get_scenario_pool(len(regions))
        futs = {i: pool.submit(scenario_delta_hat_worker, *_args(i), self.scenario_verbose)
                for i in regions}
        for i, f in futs.items():
            try:
                delta, rep = f.result()
                self.safe_margin[i] = delta
                print(self._format_scenario_report(i, delta, rep))
            except Exception as exc:
                print(f"  [scenario] region {i}: failed ({exc}); "
                      f"keeping safe_margin = {self.safe_margin[i]}")

    def check_hj_descent_grid(self, grid, V_raw, obs_sdf, dt):
        """For every time slice, compute the HJ residual and mask the set."""
        if torch.is_tensor(V_raw):
            V_raw = V_raw.cpu().numpy()
        Nt = V_raw.shape[-1]

        if obs_sdf is not None:
            if torch.is_tensor(obs_sdf):
                obs_sdf = obs_sdf.cpu().numpy()
            g = -obs_sdf
            V = np.maximum(V_raw, g[..., np.newaxis])
        else:
            V = V_raw.copy()

        V_t = np.empty_like(V)
        V_t[..., 1:-1] = (V[..., 2:] - V[..., :-2]) / (2 * dt)
        V_t[...,    0] = (V[...,  1] - V[...,   0]) / dt
        V_t[...,   -1] = (V[..., -1] - V[...,  -2]) / dt

        theta_grid = grid.vs[2]   # odp reshaped (1,1,Ntheta) -> broadcasts
        ham = np.empty_like(V)
        for t_idx in range(Nt):
            V_slice = V[..., t_idx]
            Deriv = computeGradients(grid, V_slice)
            Vx, Vy, Vtheta = Deriv[0], Deriv[1], Deriv[2]
            v_opt, omega_opt = self.plane.optCtrl_grid([Vx, Vy, Vtheta], theta_grid, uMode='min')
            ham[..., t_idx] = (
                Vx * v_opt * np.cos(theta_grid)
                + Vy * v_opt * np.sin(theta_grid)
                + Vtheta * omega_opt
            )

        hj_values = V_t + ham
        V_masked = np.where(hj_values <= 0, V, np.inf)
        return V_masked

    # =================================================================
    # Coordinate <-> index helpers
    # =================================================================
    def ys_to_cols(self, ys: np.ndarray, N=None) -> np.ndarray:
        if N is None:
            N = self.N
        num_rows, num_cols = N[:2]
        y_cell_size = (self.env_extent[3] - self.env_extent[2]) / num_rows
        cols = ((ys - self.env_extent[2]) / y_cell_size).astype(int)
        np.clip(cols, 0, num_cols - 1, out=cols)
        return cols

    def xs_to_rows(self, xs: np.ndarray, N=None) -> np.ndarray:
        if N is None:
            N = self.N
        num_rows, num_cols = N[:2]
        self.x_cell_size = (self.env_extent[1] - self.env_extent[0]) / num_cols
        rows = ((xs - self.env_extent[0]) / self.x_cell_size).astype(int)
        np.clip(rows, 0, num_rows - 1, out=rows)
        return rows

    def is_state_feasible(self, robot_pose, theta_array, t=None, reachable_set_constraint=True):
        """LEGACY per-point feasibility query.

        The original implementation fed a (Ntheta, Npts, 5) point-cloud tensor
        straight into the network. The FNO3d operator is volume-based
        ((B, Nx, Ny, T, 5)), so that path no longer applies. This method is NOT
        used by rrtx.py (it is commented out in is_feasible_ray); the active
        feasibility check is is_feasible(). Kept for API compatibility.
        """
        if not reachable_set_constraint:
            return True
        raise NotImplementedError(
            "is_state_feasible requires a per-point FNO; the FNO3d operator is "
            "volume-based. Use is_feasible() (the active rrtx feasibility check)."
        )

    def _get_region_interp(self, region):
        """Cached Check-2b interpolator for `region` under the current feasibility_source.
        Rebuilt when the region's data changed (cache popped in update_obs) or the source switched.
          "HJR_sets"       -> 3D (x, y, theta) interp on the periodically-padded theta axis.
          "feasible_region"-> 2D (x, y) interp (g for obstacle regions, g_fine otherwise)."""
        
        #Check if the RegularGridInterpolator cache already exist -> to save memory by not defining this at every call
        cached = self._interp_cache.get(region)
        if cached is not None and cached[0] == self.feasibility_source:
            return cached[1]

        if self.feasibility_source == "HJR_sets":
            reach_i = self.HJR_sets[region]
            if torch.is_tensor(reach_i):
                reach_i = reach_i.cpu().numpy()
            Tf_slice = self._grown_time_index(self.time_array)              # index 0 = fully grown
            vol = reach_i[..., Tf_slice]                                    # (Nx, Ny, Ntheta)
            vol = np.concatenate([vol, vol[:, :, :1]], axis=2)             # periodic theta pad (V(theta_0) at +2pi)
            theta_padded = np.concatenate([self.theta_array, [self.theta_array[0] + 2 * np.pi]])
            interp = RegularGridInterpolator(
                (self.g.grid_points[0], self.g.grid_points[1], theta_padded),
                vol, bounds_error=False, fill_value=None)
        else:
            # feasible_region is fine-grid for obstacle-free regions, coarse once re-predicted.
            fr_grid = self.g if self.obs_list[region] else self.g_fine
            interp = RegularGridInterpolator(
                (fr_grid.grid_points[0], fr_grid.grid_points[1]),
                self.feasible_region[region], bounds_error=False, fill_value=None)

        self._interp_cache[region] = (self.feasibility_source, interp)
        return interp

    def _get_init_interp(self, region):
        """Cached Check-2a interpolator over the frozen obstacle-free snapshot (g_fine).
        feasible_region_init never changes, so this is built once per region."""
        interp = self._interp_init_cache.get(region)
        if interp is None:
            interp = RegularGridInterpolator(
                (self.g_fine.grid_points[0], self.g_fine.grid_points[1]),
                self.feasible_region_init[region], bounds_error=False, fill_value=None)
            self._interp_init_cache[region] = interp
        return interp

    def feasibility_values(self, points: np.ndarray, thetas: np.ndarray = None) -> np.ndarray:
        """Interpolated reachable-set value V at each (x,y[,theta]); source set by
        self.feasibility_source ("HJR_sets" -> Option B theta-interpolation,
        "feasible_region" -> 2D theta-marginalized snapshot). Out-of-bounds -> +inf.
        Uses cached per-region interpolators (Fix #1)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        assert points.shape[1] == 2
        K = points.shape[0]

        # B1 heading per point (used only by "HJR_sets"); wrap onto g's periodic theta range
        if thetas is None:
            thetas = np.zeros(K)
        theta_q = self._wrap_to_grid_theta(np.atleast_1d(np.asarray(thetas, dtype=float)), self.g)

        # closest safe region per point, then transform to that region's local frame
        closest_idx = np.asarray(self.find_feasible_closest_region(robot_pose=points)).reshape(-1)
        local_positions = points - self.safe_regions[closest_idx, :2]

        # out-of-bounds points are always infeasible -> +inf sentinel (above any margin)
        within_bound_mask = np.all((local_positions >= -10) & (local_positions <= 10), axis=1)
        vals = np.full(K, np.inf)

        for region in np.unique(closest_idx):
            sel = (closest_idx == region) & within_bound_mask
            if not np.any(sel):
                continue
            interp = self._get_region_interp(region)
            if self.feasibility_source == "HJR_sets":
                pts = np.column_stack([local_positions[sel, 0], local_positions[sel, 1], theta_q[sel]])
            else:
                pts = local_positions[sel, :2]
            vals[sel] = interp(pts)

        return vals

    def points_feasible(self, points: np.ndarray, thetas: np.ndarray = None,
                        reachable_set_constraint=True) -> np.ndarray:
        """Per-point boolean feasibility (vectorized), replicating is_feasible's
        Check-1 (bounds) / Check-2a (obstacle-free, feasible_region_init) /
        Check-2b (obstacle, source) split, but returning one bool PER POINT.
        theta is consumed ONLY by the "HJR_sets" source (Option B); for the
        theta-independent "feasible_region" source it is never computed/used.
        Uses the cached interpolators (Fix #1)."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        K = pts.shape[0]
        if not reachable_set_constraint:
            return np.ones(K, dtype=bool)

        # Closest region (Euclidean), local frame, bounds. Out-of-bounds -> infeasible.
        closest_idx = np.asarray(self.find_feasible_closest_region(robot_pose=pts)).reshape(-1)
        local = pts - self.safe_regions[closest_idx, :2]
        in_bound = np.all((local >= -10) & (local <= 10), axis=1)
        feas = in_bound.copy()
        if not in_bound.any():
            return feas

        # theta only needed for the HJR_sets source
        use_theta = (self.feasibility_source == "HJR_sets")
        if use_theta:
            if thetas is None:
                thetas = np.zeros(K)
            theta_q = self._wrap_to_grid_theta(np.atleast_1d(np.asarray(thetas, dtype=float)), self.g)

        for region in np.unique(closest_idx):
            sel = (closest_idx == region) & in_bound
            if not sel.any():
                continue
            thresh = self.safe_margin[region] - self.feasibility_buffer
            if self.obs_list[region]:
                # Check 2b: cached source interpolator (3D theta query for HJR_sets, else 2D)
                interp = self._get_region_interp(region)
                q = (np.column_stack([local[sel, 0], local[sel, 1], theta_q[sel]])
                     if use_theta else local[sel, :2])
                vals = interp(q)
            else:
                # Check 2a: cached frozen obstacle-free interpolator (g_fine, theta-independent)
                vals = self._get_init_interp(region)(local[sel, :2])
            feas[sel] = vals <= thresh

        return feas

    def is_feasible(self, v: np.ndarray, reachable_set_constraint=True, thetas: np.ndarray = None) -> bool:
        # Feasible iff every query point passes (defined via the per-point primitive so the
        # single-call and batched paths can never drift).
        v = np.asarray(v)
        assert v.ndim == 2 and v.shape[1] == 2
        return bool(np.all(self.points_feasible(v, thetas=thetas,
                                                reachable_set_constraint=reachable_set_constraint)))

    def find_feasible_closest_region(self, robot_pose, t=None, use_distance=True, returnList=False):
        """Find the closest feasible safe region (NOTE use distance-based for now)"""
        # if use_distance:

        robots  = np.atleast_2d(robot_pose)          # (M, 2)
        centers = self.safe_regions[:, :2]           # (N, 2)
        dist2 = np.sum((robots[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        sorted_indices = np.argsort(dist2, axis=1)
        if returnList:
            return sorted_indices
        return sorted_indices[:, 0]


    # =================================================================
    # HJ residual / descent
    # =================================================================
    def eval_value_at_state(self, grid, data_slice, x):
        x_eval = np.array(x, dtype=float)
        x_eval[2] = self._wrap_to_grid_theta(x_eval[2], grid)
        interp = RegularGridInterpolator(
            grid.grid_points, data_slice, bounds_error=False, fill_value=None)
        return float(interp(x_eval))

    def compute_time_derivative(self, grid, closest_idx, t_idx, x, dt):
        data_t = self.HJR_sets[closest_idx][:, :, :, t_idx]
        if torch.is_tensor(data_t):
            data_t = data_t.cpu().numpy()
        data_prev = self.HJR_sets[closest_idx][:, :, :, t_idx - 1]
        if torch.is_tensor(data_prev):
            data_prev = data_prev.cpu().numpy()
        V_t    = self.eval_value_at_state(grid, data_t, x)
        V_prev = self.eval_value_at_state(grid, data_prev, x)
        return (V_t - V_prev) / dt

    def check_hj_descent(self, grid, data_safe, closest_idx, t_idx, x, dt):
        data_union = data_safe[:, :, :, t_idx]
        if torch.is_tensor(data_union):
            data_union = data_union.cpu().numpy()

        if self.obs_list[closest_idx]:
            data_union = np.maximum(data_union, -self.obs_SDF[closest_idx])

        V_val = self.eval_value_at_state(grid, data_union, x)

        Deriv = computeGradients(grid, data_union)
        grad = eval_u(grid, Deriv, x)               # [Vx, Vy, Vtheta]
        v, omega = self.plane.optCtrl(0, x, grad, 'min')

        Vx, Vy, Vtheta = grad
        theta = x[2]
        ham_term = (Vx * v * np.cos(theta) + Vy * v * np.sin(theta) + Vtheta * omega)

        V_t = self.compute_time_derivative(grid, closest_idx, t_idx, x, dt)
        hj_value = V_t + ham_term

        if self.obs_list[closest_idx]:
            g_val = self.eval_value_at_state(grid, -self.obs_SDF[closest_idx], x)
        else:
            g_val = None

        return hj_value, g_val, V_val

    # =================================================================
    # Contingency planning
    # =================================================================
    def smooth_value_function_xy(self, data_union, sigma_xy=1.0):
        data_smooth = np.empty_like(data_union)
        for k in range(data_union.shape[2]):
            data_smooth[:, :, k] = gaussian_filter(data_union[:, :, k], sigma=sigma_xy, mode='nearest')
        return data_smooth

    def contingency_policy(self, robot_state: List, plotting, fig: Figure, ax: Axes,
                           showplot=True, special_case=False):

        closest_idx_list = self.find_feasible_closest_region(
            robot_pose=np.array(robot_state[:2]), returnList=True)
        assert closest_idx_list is not None, "No feasible safe region found for contingency!"

        x_r, y_r, theta = robot_state
        closest_idx_list = closest_idx_list[0]
        top3 = closest_idx_list[:3]

        # Reorder the 3 closest regions by smallest heading deviation
        heading_deviation = []
        for idx in top3:
            x_g, y_g = self.safe_regions[idx][:2]
            dx = x_g - x_r
            dy = y_g - y_r
            theta_des = math.atan2(dy, dx)
            delta_theta = ((theta_des - theta) + np.pi) % (2 * np.pi) - np.pi
            heading_deviation.append((idx, abs(delta_theta)))
        heading_deviation.sort(key=lambda z: z[1])
        reordered_top3 = [idx for idx, _ in heading_deviation]

        # ---- Region selection -------------------------------------------------
        for closest_idx in reordered_top3:
            print(f"Choosing set {closest_idx}")

            # Robot in this region's LOCAL frame (reachable sets stored region-at-origin).
            x_r_local = x_r - self.safe_regions[closest_idx][0]
            y_r_local = y_r - self.safe_regions[closest_idx][1]
            self.plane.x = np.array([x_r_local, y_r_local, theta])

            data_safe = self.HJR_sets[closest_idx]
            if torch.is_tensor(data_safe):
                data_safe = data_safe.cpu().numpy()


            delta = self.safe_margin[closest_idx]   # scenario-certified threshold (delta_hat)

            # ---------------------------------
            # Check if V(x,0) <= delta 
            # This ensures that there exist contingency policy
            # ---------------------------------

            # Pick discretization based on obstacle presence.
            if self.obs_list[closest_idx]:
                time_array = self.time_array
                grid = self.g
            else:
                time_array = self.time_array_fine
                grid = self.g_fine

            tauLength   = len(time_array)
            subSamples  = 8
            dtSmall     = (time_array[1] - time_array[0]) / subSamples
            theta_slice = int(np.argmin(np.abs(
                grid.grid_points[2] - self._wrap_to_grid_theta(self.plane.x[2], grid))))
            
            # Robot's grid cell in the local frame.
            row = int(self.xs_to_rows(np.array([self.plane.x[0]]), N=grid.pts_each_dim)[0])
            col = int(self.ys_to_cols(np.array([self.plane.x[1]]), N=grid.pts_each_dim)[0])
            if not (0 <= row < data_safe.shape[0] and 0 <= col < data_safe.shape[1]):
                print(f"  set {closest_idx}: robot off-grid; trying next")
                continue

            # V(robot) across all time slices at the robot's (x, y, theta) cell.
            V_robot = data_safe[row, col, theta_slice, :]

            # Entry check: robot must lie in the certified set {V(., fully-grown) <= delta}.
            if V_robot[0] > delta:
                print(f"  set {closest_idx}: robot outside certified set "
                      f"(V_full={V_robot[0]:.3f} > delta={delta:.3f}); trying next closest set")
                continue

            # ---------------------------------
            # If V(x,0) <= delta 
            # Find the smallest time slice of the reachable set that still contains the robot poses
            # ---------------------------------

            # Time-to-go index: tightest sublevel set still containing the robot
            # (largest slice k with V[...,k] <= delta). Non-empty since V_robot[0] <= delta.
            tEarliest = 0
            # tEarliest = int(np.flatnonzero(V_robot <= delta).max())

            if tEarliest >= tauLength - 1:
                print("Trajectory has entered the target!")
                return [], np.array([robot_state]), 0, True, None, None, None

            break   # feasible region found; proceed to the rollout

        else:
            # No heading-ranked candidate's certified set contains the robot.
            print("No certified safe region found for contingency.")
            return [], np.array([robot_state]), 999, False, None, None, None
        
        # ---- end region selection ----



        # Index 0 means the reeachable set is the largest
        time_idx = tEarliest
        trajectory = np.array([x_r, y_r, theta])

        # Pick discretization based on obstacle presence.
        if self.obs_list[closest_idx]:
            time_array = self.time_array
            grid = self.g
        else:
            time_array = self.time_array_fine
            grid = self.g_fine

        #get most up-to-date obstacles list, in preparation to detecting more obstacles during contingency plan
        obs_circle = plotting.obs_circle.copy()
        unknown_obs_circle = plotting.unknown_obs_circle.copy()
        self.utils.update_obs(obs_circle, [], [], unknown_obs_circle)
        detected_obs_list = []

        smallTol = 1E-1

        #TODO remove this 
        self.grad_smooth = None
        self.alpha = 0.3
        self.grad_eps = 1e-4

        #TODO our goal is to remove the fallback plan (use probabilistic safety instead)
        fallback_plan = False
        success = True
        V_val_failed = None
        g_val_failed = None
        ham_failed = None

        # while t > 0:
        while time_idx < len(time_array):


            # ---------------------------------
            # Extract the reachable set, and take max{ V(x,0), g(x) } to enhance reach-avoid property
            # ---------------------------------

            #Define current time slice            
            brt_time_slice = time_idx

            #load the reachable set and safe_margin
            data_safe = self.HJR_sets[closest_idx]
            if torch.is_tensor(data_safe):
                data_safe = data_safe.cpu().numpy()
            safe_margin = self.safe_margin[closest_idx]

            #Extract the current time slice
            data_union = data_safe[:, :, :, brt_time_slice]

            #Ensure the reachable set always be less than the obstacles (never run into it)
            if self.obs_list[closest_idx]:
                data_union = np.maximum(data_union, -self.obs_SDF[closest_idx]) #NOTE negate the obs_SDF such that g(x)>=0 holds for being inside the obstacles


            # NOTE no need to verify Hamiltonian again, since scenario optimization (self.safe_margin already guarantees reachibility)

            # ham_term, g_val, V_val = self.check_hj_descent(
            #     grid, data_safe, closest_idx, brt_time_slice, self.plane.x, t - t_next)

            # if g_val is not None:
            #     V_val = max(V_val, g_val)
            #     obstacle_term = g_val - V_val
            # else:
            #     obstacle_term = 0

            # pde_tol = 15e-3
            # fallback_plan = False
            # if ham_term > 0:
            #     fallback_plan = True
            #     print("Using gradient of true set")
            #     # Use true (obstacle-free) reachable set, intersected with obstacles
            #     time_array = self.time_array_fine
            #     grid = self.g_fine

            #     time_idx = int(np.argmin(np.abs(time_array - t)))
            #     t_next = time_array[time_idx - 1]
            #     brt_time_slice = time_idx

            #     data_union_true = self.true_reach_obsFree[:, :, :, brt_time_slice]
            #     if self.obs_list[closest_idx]:
            #         obs_SDF = None
            #         for obs in self.obs_list[closest_idx]:
            #             x, y, r = obs
            #             center = np.array([x - self.safe_regions[closest_idx][0],
            #                                y - self.safe_regions[closest_idx][1], 0])
            #             if obs_SDF is None:
            #                 obs_SDF = self.shapeCylinder(ignoreDims=[2], center=center, radius=r, g=grid)
            #             else:
            #                 obs_SDF = np.minimum(obs_SDF, self.shapeCylinder(ignoreDims=[2], center=center, radius=r, g=grid))
            #         data_union_true = np.maximum(data_union_true, -obs_SDF)
            #     data_union = data_union_true


            # ---------------------------------
            # Apply HJB optimal control to return to safe region
            # ---------------------------------
            Deriv = computeGradients(grid, data_union)
            for j in range(subSamples):
                deriv = eval_u(grid, Deriv, self.plane.x)
                u = self.plane.optCtrl(time_array[time_idx], self.plane.x, deriv, 'min')
                d = self.plane.optDstb(time_array[time_idx], self.plane.x, deriv, 'max')
                self.plane.updateState(u, dtSmall, d)

            x_r = self.plane.x[0] + self.safe_regions[closest_idx][0]
            y_r = self.plane.x[1] + self.safe_regions[closest_idx][1]
            theta = self.plane.x[2]
            theta_slice = int(np.argmin(np.abs(grid.grid_points[2] - self._wrap_to_grid_theta(theta, grid))))


            trajectory = np.vstack((trajectory, np.array([x_r, y_r, theta])))

            # Sense for new obstacles (global coordinates)
            _, new_obs = self.utils.lidar_detected(robot_position=(x_r, y_r))

            #NOTE Here, we double check such that 
            # new_obs doesn't have any duplicates of the obstacles that was already discovered
            detected_obs = [obs for obs in new_obs if obs not in self.utils.obs_circle]


            if len(detected_obs) > 0:
                print("Update reachable set with newly detected obstacles: ", detected_obs)
                self.update_obs(detected_obs)
                for obs in detected_obs:
                    detected_obs_list.append(obs)
                    obs_circle.append(obs)

                 # Update obstacles for plotting and collision checking
                plotting.update_obs(obs_circle, [], [], self.utils.unknown_obs_circle)
                self.utils.update_obs(obs_circle, [], [], self.utils.unknown_obs_circle)


            # TODO we might still need some form of fallback plans incase the new obstacles make the contingency plan infeasible.

            # if fallback_plan or len(detected_obs) > 0:
            #     if self.obs_list[closest_idx]:
            #         time_array = self.time_array
            #         grid = self.g
            #     else:
            #         time_array = self.time_array_fine
            #         grid = self.g_fine

            #     if len(detected_obs) > 0:
            #         theta_slice = int(np.argmin(np.abs(grid.grid_points[2] - self._wrap_to_grid_theta(self.plane.x[2], grid))))
            #         tauLength = len(time_array)
            #         subSamples = 8
            #         dtSmall = (time_array[1] - time_array[0]) / subSamples

            #         upper = tauLength - 1
            #         lower = 0

            #         def is_in_BRS(time_idx):
            #             rows = self.xs_to_rows(np.array([self.plane.x[0]]), N=grid.pts_each_dim)
            #             cols = self.ys_to_cols(np.array([self.plane.x[1]]), N=grid.pts_each_dim)
            #             row = int(rows[0]); col = int(cols[0])
            #             if not (0 <= row < data_safe.shape[0] and 0 <= col < data_safe.shape[1]):
            #                 return False
            #             return data_safe[row, col, theta_slice, time_idx] <= self.safe_margin[closest_idx]

            #         tEarliest = tauLength
            #         while lower <= upper:
            #             mid = (lower + upper) // 2
            #             if is_in_BRS(mid):
            #                 tEarliest = mid
            #                 upper = mid - 1
            #             else:
            #                 lower = mid + 1

            #         if tEarliest == 0:
            #             print("Trajectory has entered the target!")
            #             return [], np.array([robot_state]), first_time - t, True, None, None, None
            #         if tEarliest < tauLength:
            #             first_time += time_array[tEarliest] - t
            #             t = time_array[tEarliest]


            # ---------------------------------
            # Plot the trajectory toward the target set
            # ---------------------------------
            

            ax.clear()
            fig.suptitle(rf"$\theta = {self.plane.x[2]:.2f}\,\mathrm{{rad}},\; t = {time_array[time_idx]:.2f}\,\mathrm{{s}}$")

            #Local frame 
            ax.set_xlim(self.grid_min[0] + self.safe_regions[closest_idx][0],
                        self.grid_max[0] + self.safe_regions[closest_idx][0])
            ax.set_ylim(self.grid_min[1] + self.safe_regions[closest_idx][1],
                        self.grid_max[1] + self.safe_regions[closest_idx][1])

            # Arrow to show robot's heading
            arrow_len = 0.03 * max(self.env.x_range[1] - self.env.x_range[0],
                                   self.env.y_range[1] - self.env.y_range[0])
            dx = arrow_len * np.cos(theta)
            dy = arrow_len * np.sin(theta)

            # trajectory (inclusing past states)
            ax.quiver(x_r, y_r, dx, dy, angles="xy", scale_units="xy", scale=1,
                      color="red", width=0.006, zorder=10)
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2.5, label='Trajectory', zorder=5)

            #NOTE we redefine data_safe again, since in the old code, we iterate through different time slices to see if there is alternative fallback plans
            # Right now we assume we dont need any, but keep it redefined anyways
            
            data_safe = self.HJR_sets[closest_idx]
            if torch.is_tensor(data_safe):
                data_safe = data_safe.cpu().numpy()

            brt_time_slice = time_idx
            data_union = data_safe[:, :, :, brt_time_slice]
            if self.obs_list[closest_idx]:
                data_union = np.maximum(data_union, -self.obs_SDF[closest_idx])

            #Plot the countour heat map of the value function over local grid
            if showplot:
                X2d, Y2d = np.meshgrid(grid.grid_points[0], grid.grid_points[1], indexing="ij")
                Z = data_union[..., theta_slice]
                Z_masked = np.ma.masked_where(Z > 0, Z)

                ax.contourf(
                    X2d + self.safe_regions[closest_idx][0],
                    Y2d + self.safe_regions[closest_idx][1],
                    Z_masked, levels=50, cmap="Blues_r",
                    vmin=np.min(Z), vmax=np.max(Z), alpha=0.7)

                ax.contour(
                    X2d + self.safe_regions[closest_idx][0],
                    Y2d + self.safe_regions[closest_idx][1],
                    Z, levels=[0], colors='#191970', linewidths=2, linestyles='solid')

                plotting.plot_env(ax)
                plotting.plot_robot(ax, [x_r, y_r], self.utils.sensing_radius)
                plt.pause(0.3)

        #     # ---- Fallback-plan checking ----
        #     ham_term, g_val, V_val = self.check_hj_descent(
        #         grid, data_safe, closest_idx, brt_time_slice, self.plane.x, t - t_next)
        #     if g_val is not None:
        #         V_val = max(V_val, g_val)
        #         obstacle_term = g_val - V_val
        #     else:
        #         obstacle_term = 0

        #     if obstacle_term > 0:
        #         success = False
        #         g_val_failed = g_val
        #     if V_val > 0 and (self.plane.x[0] ** 2 + self.plane.x[1] ** 2) > (2) ** 2:
        #         success = False
        #         V_val_failed = V_val
        #     if ham_term > 0 and not success:
        #         ham_failed = ham_term

            time_idx += 1

        if (self.plane.x[0] ** 2 + self.plane.x[1] ** 2) < (2) ** 2:
            return (detected_obs_list, trajectory, 8, success,
                V_val_failed, g_val_failed, ((self.plane.x[0] ** 2 + self.plane.x[1] ** 2) - 4))

        # self.grad_smooth = None
        # V_val_failed = V_val
        # g_val_failed = g_val
        # ham_failed = ham_term

        return (detected_obs_list, trajectory, 8, success,
                V_val_failed, g_val_failed, ((self.plane.x[0] ** 2 + self.plane.x[1] ** 2) - 4))
