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
from typing import Tuple, List, Dict, Union

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
# import utils
# import plotting


# =========================
# Reproducibility
# =========================
torch.manual_seed(0)
np.random.seed(0)


#---------------------
# 1D Fourier Neural Operator Class
#---------------------
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
                
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
                
        self.lifting = nn.Conv1d(5, self.width, 1)
        self.projection = nn.Conv1d(self.width, 1, 1)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)


    def forward(self, x):
        
        #Lifting
        x = x.permute(0, 2, 1)
        x = self.lifting(x)
        
        #Fourier Block 0
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)
        
        #Fourier Block 1
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)
        
        #Fourier Block 2
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)
        
        #Fourier Block 3
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.relu(x)
        
        #Projection
        x = self.projection(x)
        x = x.permute(0, 2, 1)

        return x

#---------------------------------------
# Grid structure for HJB computation
#---------------------------------------

class Grid:
    def __init__(self, min_vals, max_vals, N, periodic_dims=None):
        self.min = np.array(min_vals)
        self.max = np.array(max_vals)
        self.N = np.array(N)
        self.dim = len(min_vals)
        self.periodic_dims = periodic_dims if periodic_dims is not None else []
        
        # Grid spacing
        self.dx = (self.max - self.min) / (self.N - 1)
        
        # Create grid coordinates
        self.vs = []
        for i in range(self.dim):
            self.vs.append(np.linspace(self.min[i], self.max[i], self.N[i]))
        
        # Create meshgrid
        self.xs = np.meshgrid(*self.vs, indexing='ij')
        
        # Axis bounds for plotting
        self.axis = [self.min[0], self.max[0], self.min[1], self.max[1]]

    def __str__(self):
        return (
            f"Grid:\n"
            + f"  max: {self.max}\n"
            + f"  min: {self.min}\n"
            + f"  pts_each_dim: {self.N}\n"
            + f"  pDim: {self.periodic_dims}\n"
            + f"  dx: {self.dx}\n"
        )
        
#---------------------------------------
# Plane dynamics class
#---------------------------------------        
class Plane:
    
    def __init__(self, x0, wMax, vrange, dMax):
        self.x = np.array(x0, dtype=float)
        self.wMax = wMax
        self.vrange = vrange
        self.dMax = np.array(dMax)
        self.nx = 3
        self.nu = 2
        self.nd = 3
        
    def dynamics(self, t, x, u, d):
        """Compute dynamics: dx/dt = f(x, u, d)"""
        dx = np.zeros(3)
        dx[0] = u[0] * np.cos(x[2]) + d[0]
        dx[1] = u[0] * np.sin(x[2]) + d[1]
        dx[2] = u[1] + d[2]
        return dx
    
    def optCtrl(self, t, x, deriv, uMode='min'):
        """Optimal control"""
        u = np.zeros(2)
        det1 = deriv[0] * np.cos(x[2]) + deriv[1] * np.sin(x[2])
        
        # print("det1", det1)
        
        v_range = self.vrange
        
        if uMode == 'max':
            u[0] = v_range[1] if det1 >= 0 else v_range[0]
            u[1] = self.wMax if deriv[2] >= 0 else -self.wMax
        elif uMode == 'min':
            u[0] = v_range[0] if det1 >= 0 else v_range[1]
            u[1] = -self.wMax if deriv[2] >= 0 else self.wMax
        
        return u
    
    def optDstb(self, t, x, deriv, dMode='max'):
        """Optimal disturbance"""
        d = np.zeros(3)
        normDeriv12 = np.sqrt(deriv[0]**2 + deriv[1]**2)
        
        if normDeriv12 > 0:
            if dMode == 'max':
                d[0] = self.dMax[0] * deriv[0] / normDeriv12
                d[1] = self.dMax[1] * deriv[1] / normDeriv12
            elif dMode == 'min':
                d[0] = -self.dMax[0] * deriv[0] / normDeriv12
                d[1] = -self.dMax[1] * deriv[1] / normDeriv12
        
        if dMode == 'max':
            d[2] = self.dMax[2] if deriv[2] >= 0 else -self.dMax[2]
        elif dMode == 'min':
            d[2] = -self.dMax[2] if deriv[2] >= 0 else self.dMax[2]
        
        return d
    
    def updateState(self, u, dt, d):
        """Update state using Euler integration"""
        dx = self.dynamics(0, self.x, u, d)
        self.x = self.x + dx * dt
        
#---------------------------------------
# Derivative Function from HelperOC Toolbox
# https://github.com/HJReachability/helperOC
#---------------------------------------

def upwindFirstENO2(grid: Grid, data: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Second order ENO approximation of first derivative"""
    dxInv = 1.0 / grid.dx[dim]
    
    # Add ghost cells (simple periodic or extrapolation)
    stencil = 2
    gdata = add_ghost_cells(data, dim, stencil)
    
    # First divided differences
    D1 = dxInv * np.diff(gdata, axis=dim)
    
    # Second divided differences
    D2 = 0.5 * dxInv * np.diff(D1, axis=dim)
    
    # Strip extra entries from D1
    D1 = strip_dim(D1, dim, 1, 1)
    
    # Create left and right approximations
    derivL = strip_dim(D1, dim, 0, 1)
    derivR = strip_dim(D1, dim, 1, 0)
    
    # Add second order corrections
    D2_left = strip_dim(D2, dim, 0, 2)
    D2_right = strip_dim(D2, dim, 1, 1)
    
    derivL = derivL + grid.dx[dim] * D2_left
    derivR = derivR - grid.dx[dim] * D2_right
    
    return derivL, derivR


def upwindFirstWENO5(grid: Grid, data: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fifth order WENO approximation - simplified implementation"""
    dxInv = 1.0 / grid.dx[dim]
    stencil = 3
    
    # Add ghost cells
    gdata = add_ghost_cells(data, dim, stencil)
    
    # Compute first divided differences
    D1 = dxInv * np.diff(gdata, axis=dim)
    
    # For simplicity, use second order ENO as base
    derivL, derivR = upwindFirstENO2(grid, data, dim)
    
    return derivL, derivR


def add_ghost_cells(data: np.ndarray, dim: int, stencil: int) -> np.ndarray:
    """Add ghost cells by extrapolation or periodic boundary"""
    # Simple extrapolation for ghost cells
    shape = list(data.shape)
    shape[dim] += 2 * stencil
    
    gdata = np.zeros(shape)
    
    # Copy original data
    slices = [slice(None)] * data.ndim
    slices[dim] = slice(stencil, -stencil)
    gdata[tuple(slices)] = data
    
    # Extrapolate boundaries
    for i in range(stencil):
        # Left boundary
        slices_src = [slice(None)] * data.ndim
        slices_src[dim] = stencil
        slices_dst = [slice(None)] * data.ndim
        slices_dst[dim] = stencil - i - 1
        gdata[tuple(slices_dst)] = gdata[tuple(slices_src)]
        
        # Right boundary
        slices_src[dim] = -stencil - 1
        slices_dst[dim] = -stencil + i
        gdata[tuple(slices_dst)] = gdata[tuple(slices_src)]
    
    return gdata


def strip_dim(data: np.ndarray, dim: int, left: int, right: int) -> np.ndarray:
    """Strip entries from left and right along dimension"""
    slices = [slice(None)] * data.ndim
    slices[dim] = slice(left, -right if right > 0 else None)
    return data[tuple(slices)]


def computeGradients(grid: Grid, data: np.ndarray) -> List[np.ndarray]:
    """Compute gradients using upwind scheme"""
    derivC = []
    
    for dim in range(grid.dim):
        derivL, derivR = upwindFirstWENO5(grid, data, dim)
        # Central difference
        deriv = 0.5 * (derivL + derivR)
        derivC.append(deriv)
    
    return derivC


def eval_u(grid: Grid, gradients: List[np.ndarray], x: np.ndarray) -> np.ndarray:
    """Evaluate gradient at point x using interpolation"""
    deriv = np.zeros(grid.dim)
    
    for dim in range(grid.dim):
        # Handle periodic dimensions
        x_eval = x.copy()
        if dim in grid.periodic_dims:
            period = grid.max[dim] - grid.min[dim]
            while x_eval[dim] > grid.max[dim]:
                x_eval[dim] -= period
            while x_eval[dim] < grid.min[dim]:
                x_eval[dim] += period
        
        # Create interpolator
        interp = RegularGridInterpolator(
            grid.vs, gradients[dim], 
            bounds_error=False, fill_value=None
        )
        
        # Evaluate
        deriv[dim] = interp(x_eval)
    
    # If NaN, use nearest neighbor
    if np.any(np.isnan(deriv)):
        for dim in range(grid.dim):
            idx = np.argmin(np.abs(grid.vs[dim] - x[dim]))
            if dim == 0:
                deriv[dim] = gradients[dim][idx, :, :].mean()
            elif dim == 1:
                deriv[dim] = gradients[dim][:, idx, :].mean()
            else:
                deriv[dim] = gradients[dim][:, :, idx].mean()
    
    return deriv
    
#---------------------------------------
# Class: Neural Operator-based HJ reachaiblity 
#--------------------------------------- 
    
class HJR_FNO:
    """
    Wrapper class for Hamilton–Jacobi Reachability
    prediction using a trained Fourier Neural Operator (FNO).

    Assumes:
    - Goal-rooted reachable set
    - 3D Dubins state: (x, y, theta)
    - Time-varying reachable set prediction
    """

    def __init__(self, env, safe_regions, Tf_reach,  device='cuda'):
        """
        Initialize the HJR-FNO model and grid.

        Parameters
        ----------
        save_path : str
            Path to the saved FNO model
        device : str
            'cuda' or 'cpu'
        """
        
        # Import here to avoid circular dependency
        import utils as utils_module  
          
        self.device = device
        
        save_path = Path(__file__).resolve().parent / "model/hjrno_dubins_best_so_far"

        if not os.path.exists(save_path):
            raise FileNotFoundError(f"HJR-FNO model not found at {save_path}")

        print("Loading saved HJR-FNO model...")
        self.model = torch.load(save_path, weights_only=False)
        self.model.to(self.device)
        self.model.eval()
        
        #Define utils
        self.env = env #This should be coincide with SFF_star.plotting.env such that it is shared globally
        self.utils = utils_module.Utils(environment=env)
        
        
        #Define 3D Plane dynamics
        x_init = np.array([0, 0, 0])
        self.wMax = 1
        self.vrange = [0.03, 1]
        self.dMax = [0, 0, 0]
        self.plane = Plane(x_init, self.wMax, self.vrange, self.dMax)
        
        
        # Define grid 
        self.grid_min = np.array([-10.0, -10.0, 0.0])
        self.grid_max = np.array([10.0, 10.0, 2 * math.pi])
        self.N = np.array([100, 100, 15, 17]) #Dimension for [x,y,theta] ;  original (50,50,25)
        self.pd = [2]  # theta is periodic

        self.g = Grid(
            self.grid_min,
            self.grid_max,
            self.N[:3],
            self.pd
        )        
        
        
        # Define the offset to the center of safe region 
        # (since training data assume the safe region locates at the origin)
        self.num_safe_regions = len(safe_regions)
        self.safe_regions = np.array(safe_regions)
        
        # At first there is no obstacles (i.e., self.obs_list = [])
        # Use exact reachable set precomputed ahead of time (dim 50x50x25x33)
        
        mat_data = loadmat('test/HJB_training_mat/50_50_25_SDF_no_obs.mat')  # Update this path
        # mat_data = loadmat('test/HJB_training_mat/Plane_NoObs_10steps.mat')  # Update this path
        data_safe = mat_data['BRT_all'][0][0] #Exact reachable set for no obstacle case
        self.N_fine = np.array([50,50,25,33]) #21]
        self.g_fine = Grid(
            self.grid_min,
            self.grid_max,
            self.N_fine[:3],
            self.pd
        )
        # NOTE Pre-computed set "data_safe is discretized to (50,50,25,33) == (x,y,theta,time)
        
        self.obs_SDF = [ np.empty(self.N[:3]) for i in range(self.num_safe_regions)]
        self.obs_list = [ [] for i in range(self.num_safe_regions)] #store all obstacles seen so far for each safe region
        
        # Precompute spatial meshgrid (XY only)
        self.X_fine, self.Y_fine = np.meshgrid(
            np.linspace(self.grid_min[0], self.grid_max[0], self.N_fine[0]),
            np.linspace(self.grid_min[1], self.grid_max[1], self.N_fine[1]),
            indexing="ij"
        )
        
        self.env_extent = [self.grid_min[0], self.grid_max[0], self.grid_min[1], self.grid_max[1]]

        
        self.X, self.Y = np.meshgrid(
            self.g.vs[0],
            self.g.vs[1],
            indexing="ij"
        )
        self.X_flat = self.X.reshape(-1)
        self.Y_flat = self.Y.reshape(-1)

        self.allGridPoints = self.g.N[0] * self.g.N[1]
    
        
        #Theta discretization
        self.theta_min = 0
        self.theta_max = 2*math.pi
        self.theta_array = np.linspace(self.theta_min, self.theta_max, self.N[2])
        self.theta_array_fine = np.linspace(self.theta_min, self.theta_max, self.N_fine[2])
        
        #Time discretization
        self.t0 = 0
        self.tf = 8 #Finite time reachability
        self.time_res = self.N[3] #time resolution
        self.time_array = np.linspace(self.t0, self.tf, self.time_res)
        self.time_array_fine = np.linspace(self.t0, self.tf, self.N_fine[3])
        
        
        #initilized HJR_set        
        self.HJR_sets = [data_safe for i in range(self.num_safe_regions) ]        
        print("Finished initializing reachable sets for contingency plan.")
        
        #Last time slice to define reachable region within "Tf_reach" sec.
        self.Tf_reach = Tf_reach
        
        
        #Update feasible region for sampling
        self.feasible_region = []
        Tf_slice = np.argmin(np.abs(self.time_array_fine - self.Tf_reach))
        for reach_i in self.HJR_sets:
            self.feasible_region.append(np.max(reach_i[...,Tf_slice], axis=2))
            
            
        
        self.safe_margin = -0.5 #ensure safe set is within V(x,y) <= safe_margin < 0
        
        
    # Single-query reachable set prediction
    def predict(self, sdf_input, theta_hyparam, time_hyparam):
        """
        Predict a time-varying reachable set from a signed distance field.

        @params
        - sdf_input: torch.Tensor, SDF over (x, y, θ)
        - theta_hyparam: torch.Tensor, θ discretization
        - time_hyparam: torch.Tensor, time discretization

        @return
        - pred_reshaped: torch.Tensor, reachable set of shape (Nx, Ny, Nθ, Nt)
        """


        g = self.g

        assert (
            sdf_input.shape[0] == g.N[0]
            and sdf_input.shape[1] == g.N[1]
        ), (
            f"sdf_input spatial dimensions must be "
            f"({g.N[0]}, {g.N[1]}), "
            f"but got {sdf_input.shape[:2]}"
        )

        if torch.is_tensor(theta_hyparam):
            theta_hyparam = theta_hyparam.to(dtype=torch.float32)
        else:
            theta_hyparam = torch.as_tensor(theta_hyparam, dtype=torch.float32).flatten()

        if torch.is_tensor(time_hyparam):
            time_hyparam = time_hyparam.to(dtype=torch.float32)
        else:
            time_hyparam = torch.as_tensor(time_hyparam, dtype=torch.float32).flatten()


        if sdf_input.ndim == 2:
            sdf_input = sdf_input.unsqueeze(-1)
        elif sdf_input.ndim != 3:
            raise ValueError(
                f"sdf_input must be 2D or 3D, but got shape {sdf_input.shape}"
            )

        if sdf_input.shape[2] != theta_hyparam.numel():
            raise ValueError(
                f"Expected len(theta_hyparam) == {sdf_input.shape[2]}, "
                f"but got {theta_hyparam.numel()} "
                f"(sdf_input shape={sdf_input.shape})"
            )
            
        theta_hyparam = theta_hyparam.flatten()
        time_hyparam = time_hyparam.flatten()

        TH = len(theta_hyparam)
        T = len(time_hyparam)

        batch_size = T * TH

        # ---------------------------------------
        # Construct query tensor: (batch, spatial, channels)
        # 5 Channels = (SDF, x, y, theta, time)
        # ---------------------------------------

        xx_query = torch.empty(
            batch_size,
            self.allGridPoints,
            5,
            dtype=torch.float32
        )

        j = 0
        for t_i in range(T):
            for th_i in range(TH):

                sdf_slice = sdf_input[:, :, th_i].reshape(-1)

                xx_query[j, :, 0] = torch.tensor(sdf_slice)
                xx_query[j, :, 1] = torch.tensor(self.X_flat)
                xx_query[j, :, 2] = torch.tensor(self.Y_flat)
                xx_query[j, :, 3] = theta_hyparam[th_i]
                xx_query[j, :, 4] = time_hyparam[t_i]

                j += 1

        #Run FNO inference
        pred = torch.zeros(batch_size, self.allGridPoints)

        loader = DataLoader(
            TensorDataset(xx_query),
            batch_size=1,
            shuffle=False
        )

        with torch.no_grad():
            for idx, (x,) in enumerate(loader):
                x = x.to(self.device)
                out = self.model(x)
                pred[idx] = out.squeeze().cpu()

        # Reshape to (Nx, Ny, Ntheta, Nt)
        pred_reshaped = torch.zeros(
            self.g.N[0],
            self.g.N[1],
            TH,
            T
        )

        idx = 0
        for t_i in range(T):
            for th_i in range(TH):
                pred_reshaped[:, :, th_i, t_i] = pred[idx].reshape(
                    self.g.N[0],
                    self.g.N[1]
                )
                idx += 1

        return pred_reshaped
    
    
    def shapeCylinder(self, ignoreDims=None, center=None, radius=1.0):
        """
        Create a cylindrical signed distance function on the grid.

        @params
        - ignoreDims: list[int], ignored dimensions
        - center: array-like, cylinder center
        - radius: float, cylinder radius

        @return
        - sdf: signed distance field with shape g.N
        """


        g = self.g
        dim = g.dim

        # Default arguments
        if ignoreDims is None:
            ignoreDims = []

        if center is None:
            center = np.zeros(dim)

        ignoreDims = set(ignoreDims)

        # Allocate signed distance array
        data_shape = tuple(g.N)
        dist_squared = np.zeros(data_shape)

        # Accumulate squared distance over ACTIVE dimensions
        for i in range(dim):
            if i not in ignoreDims:
                dist_squared += (g.xs[i] - center[i])**2

        # Euclidean norm in active dimensions
        dist = np.sqrt(dist_squared)

        # Signed distance function (negative inside)
        sdf = dist - radius

        return sdf
    
    def update_obs(self, obs_cir:List):
        '''
        Update Obstacles's signed distance field, and Predict the reachable set.
        
        @param obs_cir: Newly Detected obstacles
        @type obs_cir: List
        '''
        # NOTE For now, Only consider circular obstacle since training data only consider spherical obstacles
        # NOTE this function updates SDF for every newly-detected obstacle only. It does not have memory of all obstacles seen in the past
        
        
        # Iterate through each safe region that we want to update
        for i in range(self.num_safe_regions):
            x_offset, y_offset , _ = self.safe_regions[i]
            
            update_HJR_set = False 
            
            for obs in obs_cir:
                x,y,r = obs
                
                center = np.array([x - x_offset, y - y_offset, 0])
                within_bound = np.all(center >= self.grid_min) and np.all(center <= self.grid_max)
                
                if within_bound: #only care about signed distance field within the grid range of the reachable set
                    
                    #Need to update reachable set
                    update_HJR_set = True
                
                    #Create signed distance field in the local frame
                    obs_sdf = self.shapeCylinder(ignoreDims=[2], center=center, radius=r)
                    
                    #Union of all observed obstacles
                    if not self.obs_list[i]:
                        self.obs_SDF[i] = obs_sdf
                    else:
                        self.obs_SDF[i] = np.minimum(self.obs_SDF[i], obs_sdf)

                    self.obs_list[i].append(obs) #store the newly detected obstacle
                    
            #if new obstacle lies within the grid range of the reachable set, we need to update the reachable set
            if update_HJR_set:
                #Predict HJRNO reachable set (again)
                # (Nx, Ny, Ntheta, Nt)
                self.HJR_sets[i] = self.predict(sdf_input=self.obs_SDF[i], theta_hyparam=self.theta_array, time_hyparam=self.time_array)
            
        #Update feasible region (find miminal set with respect to all theta slices) for RRT planning
        for i, reach_i in enumerate(self.HJR_sets):
            
            #Find time slice of the reachable set
            if self.obs_list[i]:
                reach_i = reach_i.cpu().numpy()
                Tf_slice = np.argmin(np.abs(self.time_array - self.Tf_reach))
                                     
            #Special case: No obstacle yet, we use exact (pre-computed) reachable set with finer discretization
            else:
                Tf_slice = np.argmin(np.abs(self.time_array_fine - self.Tf_reach))
                
            self.feasible_region[i] = np.max(reach_i[..., Tf_slice], axis=2)
                    
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
    
    def is_state_feasible(self, robot_pose:np.array, theta_array:np.array, t=None, reachable_set_constraint=True) -> bool:
        
                
        #If no reachable set constraint, always feasible
        if reachable_set_constraint == False:
            return True
        
        if t is None:
            t= self.Tf_reach
                        
        closest_idx = self.find_feasible_closest_region(robot_pose, t=t)
        # print(closest_idx)
        
        if closest_idx is None:
            return False
        
        
        # ------------------------------------------------------------
        # Robot positions (GLOBAL)
        # ------------------------------------------------------------
        robots_xy = robot_pose                        # (M, 2)
        safe_centers = self.safe_regions[closest_idx, :2]      # (M, 2)

        # Robot positions in LOCAL frame
        robots_local = robots_xy - safe_centers                # (M, 2)
        
        within_bounds = np.all(
            (robots_local >= -10) & (robots_local <= 10)
        )
        
        if not within_bounds:
            return False

        # ------------------------------------------------------------
        # Build obstacle lists PER ROBOT in LOCAL frame
        # ------------------------------------------------------------
        centers_local_list = []
        radii_list = []

        for i, k in enumerate(closest_idx):
            if len(self.obs_list[k]) == 0:
                # Dummy obstacle already in LOCAL frame (so far away that it does not affect SDF)
                centers_local_list.append([[10.0, 10.0]])
                radii_list.append([1.0])
            else:
                obs = np.asarray(self.obs_list[k])              # (Mi, 3)
                centers_global = obs[:, :2]                      # (Mi, 2)
                radii = obs[:, 2]                                # (Mi,)

                # Convert obstacle centers to LOCAL frame
                centers_local = centers_global - safe_centers[i]

                centers_local_list.append(centers_local)
                radii_list.append(radii)

        # ------------------------------------------------------------
        # Stack and align
        # ------------------------------------------------------------
        centers_local_all = np.vstack(centers_local_list)       # (ΣMi, 2)
        radii_all = np.concatenate(radii_list)                  # (ΣMi,)

        obs_counts = np.array([len(r) for r in radii_list])

        robots_local_rep = np.repeat(robots_local, obs_counts, axis=0)

        # ------------------------------------------------------------
        # Compute SDF
        # ------------------------------------------------------------
        sdf_all = np.linalg.norm(
            centers_local_all - robots_local_rep,
            axis=1
        ) - radii_all

        split_idx = np.cumsum(obs_counts)[:-1]
        obs_sdf = np.minimum.reduceat(sdf_all, np.r_[0, split_idx])  # (M,)
        
        # ------------------------------------------------------------
        # Build xx_query tensor
        # Shapes:
        # obs_sdf        : (32,)
        # robots_local   : (32, 2)
        # theta_array    : (TH,)
        # t              : scalar
        
        TH = len(theta_array)

        obs_sdf_t = torch.from_numpy(obs_sdf).float().to(self.device)
        robots_t  = torch.from_numpy(robots_local).float().to(self.device)
        theta_t   = torch.from_numpy(theta_array).float().to(self.device)

        xx_query = torch.empty((TH, 32, 5), device=self.device)

        xx_query[:, :, 0] = obs_sdf_t[None, :]
        xx_query[:, :, 1] = robots_t[:, 0][None, :]
        xx_query[:, :, 2] = robots_t[:, 1][None, :]
        xx_query[:, :, 3] = theta_t[:, None]
        xx_query[:, :, 4] = float(t)



        # ------------------------------------------------------------
        # Run FNO inference
        # ------------------------------------------------------------
        with torch.no_grad():
            out = self.model(xx_query)

        is_feasible = torch.all(out <= self.safe_margin).item()
        return is_feasible
    
    def is_feasible(self, v: np.ndarray, reachable_set_constraint=True) -> bool:

        if not reachable_set_constraint:
            return True

        v = np.asarray(v)
        assert v.ndim == 2 and v.shape[1] == 2

        # ----------------------------------
        # Closest region per point
        # ----------------------------------
        closest_idx = self.find_feasible_closest_region(robot_pose=v)
        closest_idx = np.asarray(closest_idx).reshape(-1)

        # ----------------------------------
        # Transform to local frame
        # ----------------------------------
        safe_centers = self.safe_regions[closest_idx, :2]
        local_positions = v - safe_centers        # (K,2)

        # ----------------------------------
        # Bounds check (early exit)
        # ----------------------------------
        within_bound_mask = np.all(
            (local_positions >= -10) & (local_positions <= 10),
            axis=1
        )

        if not np.all(within_bound_mask):
            return False

        # ----------------------------------
        # Obstacle presence per region
        # ----------------------------------
        obs_nonempty = np.fromiter(
            (len(obs) > 0 for obs in self.obs_list),
            dtype=bool,
            count=len(self.obs_list),
        )

        has_obs = obs_nonempty[closest_idx]   # (K,)

        idx_empty    = np.where(~has_obs)[0]
        idx_nonempty = np.where(has_obs)[0]

        # ----------------------------------
        # Check obstacle-free regions (fine grid)
        # ----------------------------------
        if idx_empty.size > 0:
            local_empty = local_positions[idx_empty]

            # rows_e = self.xs_to_rows(local_empty[:, 0], N=self.N_fine).astype(int)
            # cols_e = self.ys_to_cols(local_empty[:, 1], N=self.N_fine).astype(int)            


            # #Collect all look-up values from the HJB sets
            # vals_e = np.array([
            #     self.feasible_region[closest_idx[i]][rows_e[k], cols_e[k]]
            #     for k, i in enumerate(idx_empty)
            # ])
            
            
            # # Check feasibility: inside reachable set
            # if np.any( vals_e > self.safe_margin):
            #     return False
            
            
            '''
            use RegularGridInterpolator
            '''
            region_indices = closest_idx[idx_empty]

            vals_e = np.zeros(len(local_empty))

            for region in np.unique(region_indices):
                mask = region_indices == region
                pts = local_empty[mask]

                interp = RegularGridInterpolator(
                    (self.g_fine.vs[0], self.g_fine.vs[1]),
                    self.feasible_region[region],
                    bounds_error=False,
                    fill_value=None
                )

                vals_e[mask] = interp(pts)
                
            # Check feasibility: inside reachable set
            if np.any( vals_e > self.safe_margin):
                return False


        # ----------------------------------
        # Check obstacle regions (coarse grid / FNO)
        # ----------------------------------
        if idx_nonempty.size > 0:
            local_nonempty = local_positions[idx_nonempty]

            # rows_n = self.xs_to_rows(local_nonempty[:, 0], N=self.N).astype(int)
            # cols_n = self.ys_to_cols(local_nonempty[:, 1], N=self.N).astype(int)

            # vals_n = np.array([
            #     self.feasible_region[closest_idx[i]][rows_n[k], cols_n[k]]
            #     for k, i in enumerate(idx_nonempty)
            # ])
            
            #  # Check feasibility: inside reachable set
            # if np.any( vals_n > self.safe_margin):
            #     return False
            
            
            '''
            use RegularGridInterpolator
            '''
            region_indices = closest_idx[idx_nonempty]

            vals_n = np.zeros(len(idx_nonempty))

            for region in np.unique(region_indices):
                mask = region_indices == region
                pts = local_nonempty[mask]

                interp = RegularGridInterpolator(
                    (self.g.vs[0], self.g.vs[1]),
                    np.maximum( self.feasible_region[region], -self.obs_SDF[region][...,0]),
                    bounds_error=False,
                    fill_value=None
                )

                vals_n[mask] = interp(pts)
                
            # Check feasibility: inside reachable set
            if np.any( vals_n > self.safe_margin):
                return False


        return True


            
        
    def is_feasible_old(self, v:Tuple, reachable_set_constraint=True) -> bool:
        
        #TODO instead of finding the closest index, it might be better to predict for HJR-FNO at the current state (might be faster)
        #                                           (implement this with is_feasible_ray altogether)
        
        #If no reachable set constraint, always feasible
        if reachable_set_constraint == False:
            return True
        
        closest_idx = self.find_feasible_closest_region(robot_pose=np.array([v[0], v[1]]))
        closest_idx = closest_idx[0]
        safe_centers = self.safe_regions[closest_idx, :2]      # (M, 2)

        # Robot positions in LOCAL frame
        robots_local = np.array([v[0], v[1]]) - safe_centers     # (M, 2)
        
        #Check within bounds
        within_bounds = np.all(
            (robots_local >= -10) & (robots_local <= 10)
        )
        
        if not within_bounds:
            return False
        
        #Define discretization based on obstacle presence
        if not self.obs_list[closest_idx]:
                N = self.N_fine #no obstacles, use exact reachable set with pre-defined discretization
        else:
            N = self.N #user-defined discretization for Neural Operator
            
        #find closest matrix indices to the Node
        rows = self.xs_to_rows(np.array([robots_local[0]]), N=N)
        cols = self.ys_to_cols(np.array([robots_local[1]]), N=N)
        
        row = int(rows[0])
        col = int(cols[0])
        
        return self.feasible_region[closest_idx][row, col] <= self.safe_margin
        
        
        '''OLD CODE'''
        # for i, feasible_i in enumerate(self.feasible_region):
            
        #     if not self.obs_list[i]:
        #         N = self.N_fine #no obstacles, use exact reachable set with pre-defined discretization
        #     else:
        #         N = self.N #user-defined discretization for Neural Operator
            
        #     #find closest matrix indices to the Node
        #     rows = self.xs_to_rows(np.array([v[0] - self.safe_regions[i][0]]), N=N)
        #     cols = self.ys_to_cols(np.array([v[1] - self.safe_regions[i][1]]), N=N)
            
        #     row = int(rows[0])
        #     col = int(cols[0])
            
        #     if feasible_i[row, col] <= self.safe_margin:
        #         return True
        
        # return False

    def find_feasible_closest_region(self, robot_pose:np.array, t=None, use_distance=True, returnList=False):        
        
        '''
        Find the closest feasible safe region based on euclidean distance
        - can pass in array of robot states for batch processing
        '''
        if use_distance:
            # Extract positions
            robots  = np.atleast_2d(robot_pose)              # (M, 2)
            centers = self.safe_regions[:, :2]         # (N, 2)

            # Compute squared distances using broadcasting
            # Result shape: (M, N)
            dist2 = np.sum(
                (robots[:, None, :] - centers[None, :, :])**2,
                axis=2
            )

            sorted_indices = np.argsort(dist2, axis=1)   # (M, N)
            
            if returnList:
                return sorted_indices
            else:
                return sorted_indices[:, 0]      # (M,)



        else:
            '''
            TODO: Make this handle batch of robot states as well
            
            Find the closest feasible safe region based on look-up values from HJB reachable set
            - more expensive
            '''
            x_r, y_r, theta = robot_pose

            feasible_regions = []
            HJB_values = []
            
            if t is None:
                t = self.Tf_reach
                
            # robot's position is local frame of each safe region
            local_positions = np.array([x_r, y_r]) - self.safe_regions[:, :2] 
            
            # only consider safe regions that the robot's position is within local grid bounds
            within_bound_mask = (np.abs(local_positions) <= 10).all(axis=1)     

            local_positions_filtered = local_positions[within_bound_mask]
            
            #boolean checking which safe regions have obstacles
            obs_nonempty = np.fromiter(
                (len(obs) > 0 for obs in self.obs_list),
                dtype=bool,
                count=len(self.obs_list),
            )
            
            idx_empty = np.where(within_bound_mask & ~obs_nonempty)[0] 
            idx_nonempty = np.where(within_bound_mask & obs_nonempty)[0]
            
            #smaller subset of local positions
            local_empty     = local_positions[idx_empty]
            local_nonempty  = local_positions[idx_nonempty]
            
            # Convert continuous position to grid indices  
            # - no obstacles, use exact reachable set with pre-computed finer discretization       
            rows_e = self.xs_to_rows(local_empty[:, 0], N=self.N_fine)
            cols_e = self.ys_to_cols(local_empty[:, 1], N=self.N_fine)

            rows_e = rows_e.astype(int)
            cols_e = cols_e.astype(int)
            
            theta_slice_f = np.argmin(np.abs(self.theta_array_fine - theta))
            time_slice_f  = np.argmin(np.abs(self.time_array_fine  - t))

            
            # - obstacles present, use Neural Operator reachable set with coarse discretization
            rows_n = self.xs_to_rows(local_nonempty[:, 0], N=self.N)
            cols_n = self.ys_to_cols(local_nonempty[:, 1], N=self.N)

            rows_n = rows_n.astype(int)
            cols_n = cols_n.astype(int)
            
            theta_slice_c = np.argmin(np.abs(self.theta_array - theta))
            time_slice_c  = np.argmin(np.abs(self.time_array  - t))



            #Collect all look-up values from the HJB sets
            vals_e = np.array([
                self.HJR_sets[i][rows_e[k], cols_e[k], theta_slice_f, time_slice_f]
                for k, i in enumerate(idx_empty)
            ])
            
            vals_n = np.array([
                self.HJR_sets[i][rows_n[k], cols_n[k], theta_slice_c, time_slice_c]
                for k, i in enumerate(idx_nonempty)
            ])
            
            # Check feasibility: inside reachable set
            feasible_e = vals_e <= self.safe_margin
            feasible_n = vals_n <= self.safe_margin

            feasible_indices = np.concatenate([
                idx_empty[feasible_e],
                idx_nonempty[feasible_n],
            ])

            HJB_values = np.concatenate([
                vals_e[feasible_e],
                vals_n[feasible_n],
            ])
            
            # Choose the feasible region with smallest value fucntion
            if len(feasible_indices) > 0:
                closest_idx = feasible_indices[int(np.argmin(HJB_values))]
                return closest_idx
            else:
                return None # no feasible region found

        
        
    def eval_value_at_state(self, grid, data_slice, x):
        interp = RegularGridInterpolator(
            grid.vs,
            data_slice,
            bounds_error=False,
            fill_value=None
        )
        val = interp(x)
        return float(val)


    def compute_time_derivative(self, grid, closest_idx, t_idx, x, dt):
        # current slice
        data_t = self.HJR_sets[closest_idx][:, :, :, t_idx]
        if torch.is_tensor(data_t):
            data_t = data_t.cpu().numpy()

        # previous slice (earlier time)
        data_prev = self.HJR_sets[closest_idx][:, :, :, t_idx - 1]
        if torch.is_tensor(data_prev):
            data_prev = data_prev.cpu().numpy()

        V_t     = self.eval_value_at_state(grid, data_t, x)
        V_prev  = self.eval_value_at_state(grid, data_prev, x)

        return (V_t - V_prev) / dt
    
    def check_hj_descent(self, 
                     grid,
                     closest_idx,
                     t_idx,
                     x,
                     dt):

        # 1. Load spatial slice
        data_safe = self.HJR_sets[closest_idx]
        if torch.is_tensor(data_safe):
            data_safe = data_safe.cpu().numpy()

        data_union = data_safe[:, :, :, t_idx]
        
        # print("data_safe shape:", data_safe.shape)
        # print("data_union shape:", data_union.shape)
        # print("grid dims:", len(grid.vs))


        # 2. Evaluate value function at x
        V_val = self.eval_value_at_state(grid, data_union, x)

        # 3. Spatial gradient
        Deriv = computeGradients(grid, data_union)
        grad = eval_u(grid, Deriv, x)   # [Vx, Vy, Vtheta]

        v, omega = self.plane.optCtrl(0, x, grad, 'min')

        Vx, Vy, Vtheta = grad
        theta = x[2]

        # 4. Hamiltonian term  ∇V · f
        dot_term = (
            Vx * v * np.cos(theta)
            + Vy * v * np.sin(theta)
            + Vtheta * omega
        )

        # 5. Time derivative
        V_t = self.compute_time_derivative(
            grid,
            closest_idx,
            t_idx,
            x,
            dt
        )

        hj_value = dot_term + V_t   # ∂t V + H

        # 6. Obstacle term g(x) - V(x,t)
        if self.obs_list[closest_idx]:
            g_val = self.eval_value_at_state(grid, self.obs_SDF[closest_idx], x)
            obstacle_term = -g_val - V_val
        else:
            obstacle_term = 0
        

        # 7. Full HJI-VI residual
        # residual = max(min(hj_value, np.inf), obstacle_term)

        # If you also have target ℓ(x), replace np.inf with (ℓ(x) - V_val)

        # is_safe = residual <= 0.0

        return hj_value, obstacle_term



    '''
    For Contingency Planning toward safe region
    '''
    
    

    def contingency_policy_old(self, robot_state:List, plotting, fig:Figure, ax:Axes):
    
        
        closest_idx_list = self.find_feasible_closest_region(robot_pose=np.array(robot_state[:2]), returnList=True)
        assert closest_idx_list is not None , "No feasible safe region found for contingency!"
        
        x_r, y_r, theta = robot_state
        
        closest_idx_list = closest_idx_list[0]
        
        # Take first 3 closest indices
        top3 = closest_idx_list[:3]

        heading_deviation = []

        for idx in top3:
            x_g, y_g = self.safe_regions[idx][:2]

            dx = x_g - x_r
            dy = y_g - y_r

            theta_des = math.atan2(dy, dx)
            delta_theta = ((theta_des - theta) + np.pi) % (2*np.pi) - np.pi

            heading_deviation.append((idx, abs(delta_theta)))

        # Sort by smallest angular deviation
        heading_deviation.sort(key=lambda x: x[1])

        # Extract reordered indices
        reordered_top3 = [idx for idx, _ in heading_deviation]
        
 
        for i in range(len(reordered_top3)):
            
            closest_idx = reordered_top3[i]
            
            #Update state in dynamics w.r.t. local frame
            
            x_r_local = x_r - self.safe_regions[closest_idx][0]
            y_r_local = y_r - self.safe_regions[closest_idx][1]
            
            
            robot_state_local = np.array([x_r_local, y_r_local, theta])
            self.plane.x = robot_state_local
            
            #Define reachable set
            data_safe = self.HJR_sets[closest_idx]  
            if torch.is_tensor(data_safe):
                data_safe = data_safe.cpu().numpy()
            
            '''
            Find tEarliest: the smallest time index where the current state is in the BRS
            This represents the earliest time the robot can reach the target
            '''
            
            # Determine which time array to use, based on different discretization
            if self.obs_list[closest_idx]:
                time_array = self.time_array  # HJR-FNO reachable set with user-defined discretization
                grid = self.g
            else:
                time_array = self.time_array_fine  # finer discretization for no obstacle case
                grid = self.g_fine
            
            theta_slice = np.argmin(np.abs(grid.vs[2] - self.plane.x[2]))
            tauLength = len(time_array)
            subSamples = 4
            dtSmall = (time_array[1] - time_array[0]) / subSamples
            
            # Binary search for the smallest time index where state is in BRS
            upper = tauLength - 1
            lower = 0
            
            def is_in_BRS(time_idx):
                """Check if current state is inside BRS at given time index"""
                # Convert continuous position to grid indices
                rows = self.xs_to_rows(np.array([self.plane.x[0]]), N=grid.N)
                cols = self.ys_to_cols(np.array([self.plane.x[1]]), N=grid.N)
                row = int(rows[0])
                col = int(cols[0])
                
                # Check bounds
                if not (0 <= row < data_safe.shape[0] and 0 <= col < data_safe.shape[1]):
                    return False
                    
                return data_safe[row, col, theta_slice, time_idx] <= self.safe_margin
            
            # Binary search to find the minimum time index where state is in BRS
            tEarliest = tauLength  # Default: not in any BRS
            while lower <= upper:
                mid = (lower + upper) // 2
                if is_in_BRS(mid):
                    tEarliest = mid  # Found a valid index, try to find smaller
                    upper = mid - 1
                else:
                    lower = mid + 1
                    
            # Check if trajectory has reached the target (smallest set)
            if tEarliest == 0:
                print("Trajectory has entered the target!")
                return [], np.array([robot_state])  # Already at target, no contingency needed
            
            # Check if robot is in any BRS (if not, we have a problem)
            if tEarliest >= tauLength - 1:
                print("Warning: Robot state is near boundary of reachable set!")
                continue
            
            break
        #-----  End of iterating over potential closest feasible region ---- 
        
    
    
        # State-based time tracking
        t = time_array[tEarliest] #time_array[tEarliest+1]
        t_max = self.tf
        trajectory = np.array([x_r, y_r, theta])

        
        #Transfer all obstacles seen so far
        obs_circle = plotting.obs_circle.copy()
        unknown_obs_circle = plotting.unknown_obs_circle.copy()
        self.utils.update_obs(obs_circle, self.utils.obs_boundary, [], unknown_obs_circle)
        
        #Store only new obstacles detected during contingency
        detected_obs_list = []
        
        # #Colorbar 
        # cax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        # cbar = None
            
        while t > 0:
            
            
                            
            # Find next time step
            time_idx = np.argmin(np.abs(time_array - t))
        
            t_next = time_array[time_idx - 1] # step back in time
                        
            # Work BACKWARD in time: start from the largest set (end) toward target (beginning)
            # Find the time slice by going backward: map forward simulation time to backward BRT time
            # time_remaining = t_max - t  # How much time left until target
            # brt_time_slice = np.argmin(np.abs(time_array - time_remaining))
            brt_time_slice = time_idx
            
            # time_remaining_after_next = self.tf - t_next  # How much time left until target
            # brt_time_next_slice = np.argmin(np.abs(time_array - time_remaining_after_next))
            
            # # Convert continuous position to grid indices
            # rows = self.xs_to_rows(np.array([x_r_local]), N=grid.N)
            # cols = self.ys_to_cols(np.array([y_r_local]), N=grid.N)
            # row = int(rows[0])
            # col = int(cols[0])
            
            # if data_safe[row, col, theta_slice, brt_time_slice] <= 0 and data_safe[row, col, theta_slice, brt_time_next_slice] <= 0:
            #     #If already inside the safe reachable set, keep shrinking the reachable set until we find the minimal-time slice
            #     t = t_next
            #     continue
            
            
            """
            Check Vdot <= 0
            """
            hj_value, obstacle_term = self.check_hj_descent(
                grid,
                closest_idx,
                brt_time_slice,
                self.plane.x,
                t - t_next
            )

            print(f"t = {t}, DtV+H: {hj_value}, g-V: {obstacle_term}")

            
            
            
            """
            Apply HJB optimal control to return to safe region, given then the minimal-time reachable set is found
            """
            
            data_safe = self.HJR_sets[closest_idx]  
            if torch.is_tensor(data_safe):
                data_safe = data_safe.cpu().numpy()
            
            # Use the reachable set at this backward time
            data_union = data_safe[:, :, :, brt_time_slice]
            
            if self.obs_list[closest_idx]:
                data_union = np.maximum(data_union, -self.obs_SDF[closest_idx])
                # data_union = np.minimum(data_union, self.shapeCylinder(ignoreDims=[2], center=[0,0,0], radius=2))
            
            # Compute gradients
            Deriv = computeGradients(grid, data_union) 
            for j in range(subSamples):
                deriv = eval_u(grid, Deriv, self.plane.x)
                
                # print("derivative", deriv)
                
                # Compute optimal control
                u = self.plane.optCtrl(time_array[time_idx], self.plane.x, deriv, 'min')
                # print("control", u)
                d = self.plane.optDstb(time_array[time_idx], self.plane.x, deriv, 'max')
                
                # Update state
                self.plane.updateState(u, dtSmall, d)
                                
    
            x_r = self.plane.x[0] +  self.safe_regions[closest_idx][0]
            y_r = self.plane.x[1] +  self.safe_regions[closest_idx][1]
            theta = self.plane.x[2]
            theta_slice = np.argmin(np.abs(grid.vs[2] - theta))
            
            #Sense for new obstacles
            # - use global coordinate
            _ , detected_obs = self.utils.lidar_detected(robot_position=(x_r, y_r))
            # NOTE: utils.unknown_obs is updated within lidar_detected()
            
            # # Store trajectory (Must be global position) NOTE plane.x is in the local frame of safe region
            trajectory = np.vstack((  trajectory , np.array([x_r, y_r, theta])))

            t = t_next

            #Update reachable set
            if len(detected_obs) > 0:
                print("Update reachable set with newly detected obstacles: ", detected_obs)
                
                self.update_obs(detected_obs) # Update all reachable sets with newly detected obstacles
                for obs in detected_obs:
                    detected_obs_list.append(obs) #update only new obstacles detected during contingency
                    obs_circle.append(obs) #update global obs_circle for ALL obstacles seen so far
                    
                # Determine which time array to use, based on different discretization
                if self.obs_list[closest_idx]:
                    time_array = self.time_array #HJR-FNO reachable set with user-defined discretization
                    grid = self.g
                    
                else:
                    time_array = self.time_array_fine #finer discretization for no obstacle case (since we precomputed this)
                    grid = self.g_fine
                dtSmall = (time_array[1] - time_array[0]) / subSamples
                    
            #Update obstacles for plotting and collision checking   
            plotting.update_obs(obs_circle, self.utils.obs_boundary, [], self.utils.unknown_obs_circle) # for plotting obstacles
            self.utils.update_obs(obs_circle, self.utils.obs_boundary, [], self.utils.unknown_obs_circle) # for collision checking
                
                
            # # Plot contingency plan
            # if cbar is not None:
            #     cbar.ax.cla()     # clear axis first
            #     cbar.remove()
            #     cbar = None
            #     cax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
                
            ax.clear()
            
            fig.suptitle(f"HJR-FNO Contincgency\n Safe Region: {self.safe_regions[closest_idx][:2]} | Time to Target: {self.tf - t:.2f}s")

            # restore static axis properties
            # ax.set_xlim(self.env.x_range[0], self.env.x_range[1] + 1)
            # ax.set_ylim(self.env.y_range[0], self.env.y_range[1] + 1)
            ax.set_xlim(self.grid_min[0] + self.safe_regions[closest_idx][0], self.grid_max[0] + self.safe_regions[closest_idx][0])
            ax.set_ylim(self.grid_min[1] + self.safe_regions[closest_idx][1], self.grid_max[1] + self.safe_regions[closest_idx][1])

            # draw environment
            plotting.plot_env(ax)
            
            # draw robot + lidar + heading
            plotting.plot_robot(ax, [x_r, y_r], self.utils.sensing_radius)
            
            arrow_len = 0.03 * max(self.env.x_range[1] - self.env.x_range[0],  self.env.y_range[1] - self.env.y_range[0])
            dx = arrow_len * np.cos(theta)
            dy = arrow_len * np.sin(theta)

            ax.quiver(
                x_r, y_r,          # base position
                dx, dy,            # direction vector
                angles="xy",
                scale_units="xy",
                scale=1,
                color="red",
                width=0.006,
                zorder=10
            )
            
            ax.plot(trajectory[:,0], trajectory[:,1], 
                   'r-', linewidth=2.5, label='Trajectory', zorder=5)

            # print(f"Point: {(trajectory[-1,0], trajectory[-1,1])}, Feasible? {self.is_feasible((trajectory[-1,0], trajectory[-1,1]))}")
            
            #Load reachable set again, in case the discretization changes
            data_safe = self.HJR_sets[closest_idx]  
            if torch.is_tensor(data_safe):
                data_safe = data_safe.cpu().numpy()
            brt_time_slice = np.argmin(np.abs(time_array - t))
            data_union = data_safe[:, :, :, brt_time_slice]
            
            # plot reachable set at current heading
            Z = data_union[..., theta_slice]

            # mask out values > 0
            Z_masked = np.ma.masked_where(Z > 0, Z)
            
            # rows = self.xs_to_rows(np.array([self.plane.x[0]]), N=grid.N)
            # cols = self.ys_to_cols(np.array([self.plane.x[1]]), N=grid.N)
            # row = int(rows[0])
            # col = int(cols[0])
            
            # Z_masked[row, col] = 100  # ensure current position is visible

            contourf = ax.contourf(
                grid.xs[0][..., 0] + self.safe_regions[closest_idx][0],
                grid.xs[1][..., 0] + self.safe_regions[closest_idx][1],
                Z_masked,
                levels=50,
                cmap="Blues_r",
                vmin=np.min(Z),          # keep original scale
                vmax=np.max(Z),
                alpha=0.7
            )

            
            #colorbar
            # create fresh colorbar
            # cbar = fig.colorbar(contourf, cax=cax)
            # cbar.set_label("Value Function", fontsize=8)
            
            
            CS = ax.contour(
                    grid.xs[0][...,0] + self.safe_regions[closest_idx][0],
                    grid.xs[1][...,0] + self.safe_regions[closest_idx][1],
                    Z ,
                    levels=[self.safe_margin],
                    colors='magenta',
                    linewidths=2
            )   
            
            CS2 = ax.contour(
                    grid.xs[0][...,0] + self.safe_regions[closest_idx][0],
                    grid.xs[1][...,0] + self.safe_regions[closest_idx][1],
                    Z ,
                    levels=[0],
                    colors='green',
                    linewidths=2
            )   
            
            
            
            ax.grid(True)
            
            plt.pause(0.3) #original 0.3s            
        # if cbar is not None:
        #     cbar.ax.cla()     # clear axis first
        #     cbar.remove()
        #     cbar = None


        return detected_obs_list, trajectory
    
    
    def contingency_policy(self, robot_state:List, plotting, fig:Figure, ax:Axes):
    
        
        closest_idx_list = self.find_feasible_closest_region(robot_pose=np.array(robot_state[:2]), returnList=True)
        assert closest_idx_list is not None , "No feasible safe region found for contingency!"
        
        x_r, y_r, theta = robot_state
        
        closest_idx_list = closest_idx_list[0]
        
        # Take first 3 closest indices
        top3 = closest_idx_list[:3]

        heading_deviation = []

        for idx in top3:
            x_g, y_g = self.safe_regions[idx][:2]

            dx = x_g - x_r
            dy = y_g - y_r

            theta_des = math.atan2(dy, dx)
            delta_theta = ((theta_des - theta) + np.pi) % (2*np.pi) - np.pi

            heading_deviation.append((idx, abs(delta_theta)))

        # Sort by smallest angular deviation
        heading_deviation.sort(key=lambda x: x[1])

        # Extract reordered indices
        reordered_top3 = [idx for idx, _ in heading_deviation]
        
 
        for i in range(len(reordered_top3)):
            
            closest_idx = reordered_top3[i]
            
            #Update state in dynamics w.r.t. local frame
            
            x_r_local = x_r - self.safe_regions[closest_idx][0]
            y_r_local = y_r - self.safe_regions[closest_idx][1]
            
            
            robot_state_local = np.array([x_r_local, y_r_local, theta])
            self.plane.x = robot_state_local
            
            #Define reachable set
            data_safe = self.HJR_sets[closest_idx]  
            if torch.is_tensor(data_safe):
                data_safe = data_safe.cpu().numpy()
            
            '''
            Find tEarliest: the smallest time index where the current state is in the BRS
            This represents the earliest time the robot can reach the target
            '''
            
            # Determine which time array to use, based on different discretization
            if self.obs_list[closest_idx]:
                time_array = self.time_array  # HJR-FNO reachable set with user-defined discretization
                grid = self.g
            else:
                time_array = self.time_array_fine  # finer discretization for no obstacle case
                grid = self.g_fine
            
            theta_slice = np.argmin(np.abs(grid.vs[2] - self.plane.x[2]))
            tauLength = len(time_array)
            subSamples = 4
            dtSmall = (time_array[1] - time_array[0]) / subSamples
            
            # Binary search for the smallest time index where state is in BRS
            upper = tauLength - 1
            lower = 0
            
            def is_in_BRS(time_idx):
                """Check if current state is inside BRS at given time index"""
                # Convert continuous position to grid indices
                rows = self.xs_to_rows(np.array([self.plane.x[0]]), N=grid.N)
                cols = self.ys_to_cols(np.array([self.plane.x[1]]), N=grid.N)
                row = int(rows[0])
                col = int(cols[0])
                
                # Check bounds
                if not (0 <= row < data_safe.shape[0] and 0 <= col < data_safe.shape[1]):
                    return False
                    
                return data_safe[row, col, theta_slice, time_idx] <= self.safe_margin
            
            # Binary search to find the minimum time index where state is in BRS
            tEarliest = tauLength  # Default: not in any BRS
            while lower <= upper:
                mid = (lower + upper) // 2
                if is_in_BRS(mid):
                    tEarliest = mid  # Found a valid index, try to find smaller
                    upper = mid - 1
                else:
                    lower = mid + 1
                    
            # Check if trajectory has reached the target (smallest set)
            if tEarliest == 0:
                print("Trajectory has entered the target!")
                return [], np.array([robot_state])  # Already at target, no contingency needed
            
            # Check if robot is in any BRS (if not, we have a problem)
            if tEarliest >= tauLength - 1:
                print("Warning: Robot state is near boundary of reachable set!")
                continue
            
            break
        #-----  End of iterating over potential closest feasible region ---- 
        
    
    
        # State-based time tracking
        t = time_array[tEarliest] #time_array[tEarliest+1]
        t_max = self.tf
        trajectory = np.array([x_r, y_r, theta])

        
        #Transfer all obstacles seen so far
        obs_circle = plotting.obs_circle.copy()
        unknown_obs_circle = plotting.unknown_obs_circle.copy()
        self.utils.update_obs(obs_circle, self.utils.obs_boundary, [], unknown_obs_circle)
        
        #Store only new obstacles detected during contingency
        detected_obs_list = []
        
        # #Colorbar 
        # cax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        # cbar = None
            
        while t > 0:
            
            
                            
            # Find next time step
            time_idx = np.argmin(np.abs(time_array - t))
        
                        
            # Work BACKWARD in time: start from the largest set (end) toward target (beginning)
            # Find the time slice by going backward: map forward simulation time to backward BRT time
            # time_remaining = t_max - t  # How much time left until target
            # brt_time_slice = np.argmin(np.abs(time_array - time_remaining))
            brt_time_slice = time_idx
            
            # time_remaining_after_next = self.tf - t_next  # How much time left until target
            # brt_time_next_slice = np.argmin(np.abs(time_array - time_remaining_after_next))
            
            # # Convert continuous position to grid indices
            # rows = self.xs_to_rows(np.array([x_r_local]), N=grid.N)
            # cols = self.ys_to_cols(np.array([y_r_local]), N=grid.N)
            # row = int(rows[0])
            # col = int(cols[0])
            
            # if data_safe[row, col, theta_slice, brt_time_slice] <= 0 and data_safe[row, col, theta_slice, brt_time_next_slice] <= 0:
            #     #If already inside the safe reachable set, keep shrinking the reachable set until we find the minimal-time slice
            #     t = t_next
            #     continue
            
            
            """
            Check Vdot <= 0
            """
            t_next = time_array[time_idx - 1] # step back in time
            hj_value, obstacle_term = self.check_hj_descent(
                grid,
                closest_idx,
                brt_time_slice,
                self.plane.x,
                t - t_next
            )

            print(f"t = {t}, DtV+H: {hj_value}, g-V: {obstacle_term}")

            
            
            
            """
            Apply HJB optimal control to return to safe region, given then the minimal-time reachable set is found
            """
            
            data_safe = self.HJR_sets[closest_idx]  
            if torch.is_tensor(data_safe):
                data_safe = data_safe.cpu().numpy()
            
            # Use the reachable set at this backward time
            data_union = data_safe[:, :, :, brt_time_slice]
            
            if self.obs_list[closest_idx]:
                data_union = np.maximum(data_union, -self.obs_SDF[closest_idx])
                # data_union = np.minimum(data_union, self.shapeCylinder(ignoreDims=[2], center=[0,0,0], radius=2))
            
            # Compute gradients
            Deriv = computeGradients(grid, data_union) 
            for j in range(subSamples):
                deriv = eval_u(grid, Deriv, self.plane.x)
                
                # print("derivative", deriv)
                
                # Compute optimal control
                u = self.plane.optCtrl(time_array[time_idx], self.plane.x, deriv, 'min')
                # print("control", u)
                d = self.plane.optDstb(time_array[time_idx], self.plane.x, deriv, 'max')
                
                # Update state
                self.plane.updateState(u, dtSmall, d)
                                
    
            x_r = self.plane.x[0] +  self.safe_regions[closest_idx][0]
            y_r = self.plane.x[1] +  self.safe_regions[closest_idx][1]
            theta = self.plane.x[2]
            theta_slice = np.argmin(np.abs(grid.vs[2] - theta))
            
            #Sense for new obstacles
            # - use global coordinate
            _ , detected_obs = self.utils.lidar_detected(robot_position=(x_r, y_r))
            # NOTE: utils.unknown_obs is updated within lidar_detected()
            
            # # Store trajectory (Must be global position) NOTE plane.x is in the local frame of safe region
            trajectory = np.vstack((  trajectory , np.array([x_r, y_r, theta])))

            #Update reachable set
            if len(detected_obs) > 0:
                print("Update reachable set with newly detected obstacles: ", detected_obs)
                
                self.update_obs(detected_obs) # Update all reachable sets with newly detected obstacles
                for obs in detected_obs:
                    detected_obs_list.append(obs) #update only new obstacles detected during contingency
                    obs_circle.append(obs) #update global obs_circle for ALL obstacles seen so far
                    
                # Determine which time array to use, based on different discretization
                if self.obs_list[closest_idx]:
                    time_array = self.time_array #HJR-FNO reachable set with user-defined discretization
                    grid = self.g
                    
                else:
                    time_array = self.time_array_fine #finer discretization for no obstacle case (since we precomputed this)
                    grid = self.g_fine
                dtSmall = (time_array[1] - time_array[0]) / subSamples
                    
            #Update obstacles for plotting and collision checking   
            plotting.update_obs(obs_circle, self.utils.obs_boundary, [], self.utils.unknown_obs_circle) # for plotting obstacles
            self.utils.update_obs(obs_circle, self.utils.obs_boundary, [], self.utils.unknown_obs_circle) # for collision checking
                

            
            #Load reachable set again, in case the discretization changes
            data_safe = self.HJR_sets[closest_idx]  
            if torch.is_tensor(data_safe):
                data_safe = data_safe.cpu().numpy()

            #This is for plotting later
            brt_time_slice = np.argmin(np.abs(time_array - t))
            data_union = data_safe[:, :, :, brt_time_slice]
            
            
            '''Plotting'''
            
            ax.clear()
            
            fig.suptitle(f"HJR-FNO Contincgency\n Safe Region: {self.safe_regions[closest_idx][:2]} | Time to Target: {t:.2f}s")

            # restore static axis properties
            # ax.set_xlim(self.env.x_range[0], self.env.x_range[1] + 1)
            # ax.set_ylim(self.env.y_range[0], self.env.y_range[1] + 1)
            ax.set_xlim(self.grid_min[0] + self.safe_regions[closest_idx][0], self.grid_max[0] + self.safe_regions[closest_idx][0])
            ax.set_ylim(self.grid_min[1] + self.safe_regions[closest_idx][1], self.grid_max[1] + self.safe_regions[closest_idx][1])

            # draw environment
            plotting.plot_env(ax)
            
            # draw robot + lidar + heading
            plotting.plot_robot(ax, [x_r, y_r], self.utils.sensing_radius)
            
            arrow_len = 0.03 * max(self.env.x_range[1] - self.env.x_range[0],  self.env.y_range[1] - self.env.y_range[0])
            dx = arrow_len * np.cos(theta)
            dy = arrow_len * np.sin(theta)

            ax.quiver(
                x_r, y_r,          # base position
                dx, dy,            # direction vector
                angles="xy",
                scale_units="xy",
                scale=1,
                color="red",
                width=0.006,
                zorder=10
            )
            
            ax.plot(trajectory[:,0], trajectory[:,1], 
                   'r-', linewidth=2.5, label='Trajectory', zorder=5)
            

            
            # plot reachable set at current heading
            Z = data_union[..., theta_slice]

            # mask out values > 0
            Z_masked = np.ma.masked_where(Z > 0, Z)
            
            # rows = self.xs_to_rows(np.array([self.plane.x[0]]), N=grid.N)
            # cols = self.ys_to_cols(np.array([self.plane.x[1]]), N=grid.N)
            # row = int(rows[0])
            # col = int(cols[0])
            
            # Z_masked[row, col] = 100  # ensure current position is visible

            contourf = ax.contourf(
                grid.xs[0][..., 0] + self.safe_regions[closest_idx][0],
                grid.xs[1][..., 0] + self.safe_regions[closest_idx][1],
                Z_masked,
                levels=50,
                cmap="Blues_r",
                vmin=np.min(Z),          # keep original scale
                vmax=np.max(Z),
                alpha=0.7
            )

            
            #colorbar
            # create fresh colorbar
            # cbar = fig.colorbar(contourf, cax=cax)
            # cbar.set_label("Value Function", fontsize=8)
            
            
            CS = ax.contour(
                    grid.xs[0][...,0] + self.safe_regions[closest_idx][0],
                    grid.xs[1][...,0] + self.safe_regions[closest_idx][1],
                    Z ,
                    levels=[self.safe_margin],
                    colors='magenta',
                    linewidths=2
            )   
            
            CS2 = ax.contour(
                    grid.xs[0][...,0] + self.safe_regions[closest_idx][0],
                    grid.xs[1][...,0] + self.safe_regions[closest_idx][1],
                    Z ,
                    levels=[0],
                    colors='green',
                    linewidths=2
            )   
            
            
            
            ax.grid(True)
            
            plt.pause(0.3) #original 0.3s        
            
            
            
            '''Recompute minimum time (for next time step)'''
            
            theta_slice = np.argmin(np.abs(grid.vs[2] - self.plane.x[2]))
            tauLength = len(time_array)
            
            # Binary search for the smallest time index where state is in BRS
            upper = tauLength - 1
            lower = 0
            
            def is_in_BRS(time_idx):
                """Check if current state is inside BRS at given time index"""
                # Convert continuous position to grid indices
                rows = self.xs_to_rows(np.array([self.plane.x[0]]), N=grid.N)
                cols = self.ys_to_cols(np.array([self.plane.x[1]]), N=grid.N)
                row = int(rows[0])
                col = int(cols[0])
                
                # Check bounds
                if not (0 <= row < data_safe.shape[0] and 0 <= col < data_safe.shape[1]):
                    return False
                    
                return data_safe[row, col, theta_slice, time_idx] <= self.safe_margin
            
            # Binary search to find the minimum time index where state is in BRS
            tEarliest = tauLength  # Default: not in any BRS
            while lower <= upper:
                mid = (lower + upper) // 2
                if is_in_BRS(mid):
                    tEarliest = mid  # Found a valid index, try to find smaller
                    upper = mid - 1
                else:
                    lower = mid + 1
        
            #Update time slice for the next control step
            t = time_array[tEarliest]


        return detected_obs_list, trajectory