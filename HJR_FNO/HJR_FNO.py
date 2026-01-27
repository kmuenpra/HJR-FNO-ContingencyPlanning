import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import operator
from functools import reduce
from functools import partial
from timeit import default_timer

from utils import *

torch.manual_seed(0)
np.random.seed(0)

import math
from typing import List, Union
import warnings

import os
from tqdm import tqdm
import math
from torch.utils.data import DataLoader, TensorDataset

from pathlib import Path

from scipy.interpolate import RegularGridInterpolator

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

#---------------------
# Grid Class 
#---------------------

class Grid:
    def __init__(
        self,
        minBounds: List,
        maxBounds: List,
        dims: int,
        pts_each_dim: List,
        periodicDims: List = [],
    ):
        """
        Args:
            minBounds (list): The lower bounds of each dimension in the grid
            maxBounds (list): The upper bounds of each dimension in the grid
            dims (int): The dimension of grid
            pts_each_dim (list): The number of points for each dimension in the grid
            periodicDim (list, optional): A list of periodic dimentions (0-indexed). Defaults to [].
        """
        assert len(minBounds) == len(maxBounds) == len(pts_each_dim) == dims

        self.max = np.array(maxBounds)
        self.min = np.array(minBounds)
        self.dims = dims
        self.pts_each_dim = np.array(pts_each_dim)
        self.pDim = periodicDims

        # Exclude the upper bounds for periodic dimensions is not included
        # e.g. [-pi, pi)
        for dim in self.pDim:
            self.max[dim] = self.min[dim] + (self.max[dim] - self.min[dim]) * (
                1 - 1 / self.pts_each_dim[dim]
            )
        self.dx = (self.max - self.min) / (self.pts_each_dim - 1.0)

        """
        Below is re-shaping the self.vs so that we can make use of broadcasting
        self.vs[i] is reshape into (1,1, ... , pts_each_dim[i], ..., 1) such that pts_each_dim[i] is used in ith position
        """
        self.vs = []
        """
        self.grid_points is same as self.vs; however, it is not reshaped. 
        self.grid_points[i] is a numpy array with length pts_each_dim[i] 
        """
        self.grid_points = []
        for i in range(dims):
            tmp = np.linspace(self.min[i], self.max[i], num=self.pts_each_dim[i])
            broadcast_map = np.ones(self.dims, dtype=int)
            broadcast_map[i] = self.pts_each_dim[i]
            self.grid_points.append(tmp)

            # in order to add our range of points to our grid
            # we need to modify the shape of tmp in order to match
            # the size of the grid for one of the axis
            tmp = np.reshape(tmp, tuple(broadcast_map))
            self.vs.append(tmp)

    def __str__(self):
        return (
            f"Grid:\n"
            + f"  max: {self.max}\n"
            + f"  min: {self.min}\n"
            + f"  pts_each_dim: {self.pts_each_dim}\n"
            + f"  pDim: {self.pDim}\n"
            + f"  dx: {self.dx}\n"
        )

    def get_index(self, state: np.ndarray):
        """ Returns a tuple of the closest index of each state in the grid

        Args:
            state (tuple): state of dynamic object

        TODO: Deprecate this method
        """
        warnings.warn(
            "get_index is deprecated and will be removed in a future version. Use get_indices instead.",
            DeprecationWarning,
            stacklevel=2  # This shows where the deprecated function was called
        )        
        return self.get_indices(state)

    def get_value(self, V, state):
        """Obtain the approximate value of a state

        Assumes that the state is within the bounds of the grid

        Args:
            V (np.array): value function of solved HJ PDE 
            state (tuple): state of dynamic object

        Returns:
            [float]: V(state)

        TODO: Deprecate this method
        """
        warnings.warn(
            "get_index is deprecated and will be removed in a future version. Use get_indices instead.",
            DeprecationWarning,
            stacklevel=2  # This shows where the deprecated function was called
        )
        return self.get_values(V, state)

    def get_indices(self, states: np.ndarray) -> np.ndarray:
        """Returns a tuple of the closest indices of each state in the grid

        Args:
            states (np.ndarray): states of dynamical system, shape (self.dims,) or 
                                 (N, self.dims)

        Returns:
            np.ndarray: indices of each state, shape (self.dims,) or (N, self.dims)

        TODO: Handle periodic dimensions correctly
        """
        indices = np.round((states - self.min) / self.dx)
        indices = np.clip(indices, 0, self.pts_each_dim - 1)

        return tuple(indices.astype(int).T)

    def get_values(self, V: np.ndarray, states: np.ndarray) -> Union[float, np.ndarray]:
        """
        Obtains the approximate value of a state using nearest neighbour interpolation

        Out-of-bounds state components will be clipped to the boundary of grid

        Args:
            V (np.array): value function of solved HJ PDE
            state (np.ndarray): states, shape (self.dims,) or (N, self.dims)

        Returns:
            [float or np.ndarray]: Value(s) at states, scalar or shape (N,)
        """
        indices = self.get_indices(states)
        return V[indices]
    
#---------------------
# Class: Neural Operator-based HJ reachaiblity 
#--------------------- 
    
class HJR_FNO:
    """
    Wrapper class for Hamilton–Jacobi Reachability
    prediction using a trained Fourier Neural Operator (FNO).

    Assumes:
    - Goal-rooted reachable set
    - 3D Dubins state: (x, y, theta)
    - Time-varying reachable set prediction
    """

    def __init__(self, safe_regions, Tf_reach,  device='cuda'):
        """
        Initialize the HJR-FNO model and grid.

        Parameters
        ----------
        save_path : str
            Path to the saved FNO model
        device : str
            'cuda' or 'cpu'
        """
        self.device = device
        
        save_path = Path(__file__).resolve().parent / "model/hjrno_dubins"

        if not os.path.exists(save_path):
            raise FileNotFoundError(f"HJR-FNO model not found at {save_path}")

        print("Loading saved HJR-FNO model...")
        self.model = torch.load(save_path, weights_only=False)
        self.model.to(self.device)
        self.model.eval()
        
        
        # Define grid 
        self.grid_min = np.array([-10.0, -10.0, 0.0])
        self.grid_max = np.array([10.0, 10.0, 2 * math.pi])
        self.dims = 3
        self.N = np.array([40, 40, 20]) #Dimension for [x,y,theta] ;  original (50,50,25)
        self.pd = [2]  # theta is periodic

        self.g = Grid(
            self.grid_min,
            self.grid_max,
            self.dims,
            self.N,
            self.pd
        )
        
        
        
        # Define the offset to the center of safe region 
        # (since training data assume the safe region locates at the origin)
        self.num_safe_regions = len(safe_regions)
        self.safe_regions = safe_regions
        self.obs_SDF = [ self.shapeCylinder(ignoreDims=[2], center=np.array([-10,-10,0]), radius=1.0) for i in range(self.num_safe_regions)]
        self.obs_list = [ [] for i in range(self.num_safe_regions)] #store all obstacles seen so far for each safe region
        
        # Params for mapping spatial coordiantes to matrix indices
        self.env_extent = [self.grid_min[0], self.grid_max[0], self.grid_min[1], self.grid_max[1]]
        self.num_rows, self.num_cols = self.N[:2]
        self.x_cell_size = (self.env_extent[1] - self.env_extent[0]) / self.num_cols
        self.y_cell_size = (self.env_extent[3] - self.env_extent[2]) / self.num_rows

        # Precompute spatial meshgrid (XY only)
        self.X, self.Y = np.meshgrid(
            self.g.grid_points[0],
            self.g.grid_points[1],
            indexing="ij"
        )
        self.X_flat = self.X.reshape(-1)
        self.Y_flat = self.Y.reshape(-1)

        self.allGridPoints = self.g.pts_each_dim[0] * self.g.pts_each_dim[1]
    
        
        #Theta discretization
        self.theta_min = 0
        self.theta_max = 2*math.pi
        self.theta_array = np.linspace(self.theta_min, self.theta_max, self.N[2])
        
        #Time discretization
        self.t0 = 0
        self.tf = 8 #Finite time reachability
        self.time_res = 15 #time resolution
        self.time_array = np.linspace(self.t0, self.tf, self.time_res)
        
        
        #initilized HJR_set
        # self.feasible_set = np.column_stack((np.empty(0), np.empty(0)))
        
        HJR_set_init = self.predict(
            sdf_input=self.obs_SDF[0], 
             # TODO Need to retrain HJR-FNO for the case where there are no obstacles, what does the reachable set look like???
             # TODO Technically, we can just solve for the reachable set, pre-planning
            theta_hyparam=self.theta_array, 
            time_hyparam=self.time_array)
        
        self.HJR_sets = [HJR_set_init for i in range(self.num_safe_regions) ]        
        print("Finished initializing reachable sets for contingency plan.")
        
        #Last time slice to define reachable region within "Tf_reach" sec.
        self.Tf_reach = Tf_reach
        self.Tf_slice = np.argmin(np.abs(self.time_array - self.Tf_reach))
        
        #Update feasible region for sampling
        self.feasible_region = []
        for reach_i in self.HJR_sets:
            self.feasible_region.append(np.max(reach_i[...,self.Tf_slice].cpu().numpy(), axis=2))
            
            
        
        self.safe_margin = 0.0 #ensure safe set is within V(x,y) <= safe_margin < 0
        
        
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
            sdf_input.shape[0] == g.pts_each_dim[0]
            and sdf_input.shape[1] == g.pts_each_dim[1]
        ), (
            f"sdf_input spatial dimensions must be "
            f"({g.pts_each_dim[0]}, {g.pts_each_dim[1]}), "
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
            self.g.pts_each_dim[0],
            self.g.pts_each_dim[1],
            TH,
            T
        )

        idx = 0
        for t_i in range(T):
            for th_i in range(TH):
                pred_reshaped[:, :, th_i, t_i] = pred[idx].reshape(
                    self.g.pts_each_dim[0],
                    self.g.pts_each_dim[1]
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
        - sdf: signed distance field with shape g.pts_each_dim
        """


        g = self.g
        dim = g.dims

        # Default arguments
        if ignoreDims is None:
            ignoreDims = []

        if center is None:
            center = np.zeros(dim)

        ignoreDims = set(ignoreDims)

        # Allocate signed distance array
        data_shape = tuple(g.pts_each_dim)
        dist_squared = np.zeros(data_shape)

        # Accumulate squared distance over ACTIVE dimensions
        for i in range(dim):
            if i not in ignoreDims:
                dist_squared += (g.vs[i] - center[i])**2

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
                    self.obs_SDF[i] = np.minimum(self.obs_SDF[i], obs_sdf)
                    self.obs_list[i].append(obs) #store the newly detected obstacle
                    
            #if new obstacle lies within the grid range of the reachable set, we need to update the reachable set
            if update_HJR_set:
                #Predict HJRNO reachable set (again)
                # (Nx, Ny, Ntheta, Nt)
                self.HJR_sets[i] = self.predict(sdf_input=self.obs_SDF[i], theta_hyparam=self.theta_array, time_hyparam=self.time_array)
            
        #Update feasible region (find miminal set with respect to all theta slices) for RRT planning
        for i, reach_i in enumerate(self.HJR_sets):
            self.feasible_region[i] = np.max(reach_i[...,self.Tf_slice].cpu().numpy(), axis=2)
            
    # def update_feasible_set(self, theta_slice, time_slice):
    #     '''
    #     Find all feasible points for RRT samplings with V(x,y) <= 0 (i.e., inside the reachable set)
    #     '''
        
    #     X_all = []
    #     Y_all = []

    #     for i, reach_i in enumerate(self.HJR_sets):
    #         V_xy = reach_i[:, :, theta_slice, time_slice]      # (50, 50)
    #         mask = V_xy <= self.safe_margin                                   # boolean
    #         idx = torch.nonzero(mask, as_tuple=False)          # (K, 2)

    #         if idx.numel() == 0:
    #             continue

    #         X_vals = self.X[idx[:, 0], idx[:, 1]] + self.safe_regions[i][0] #translate coordinates to centered at each safe_region
    #         Y_vals = self.Y[idx[:, 0], idx[:, 1]] + self.safe_regions[i][1] #translate coordinates to centered at each safe_region

    #         X_all.append(X_vals)
    #         Y_all.append(Y_vals)

    #     # concatenate into single arrays
    #     if X_all:
    #         X_all = np.concatenate(X_all)
    #         Y_all = np.concatenate(Y_all)
    #     else:
    #         X_all = np.empty(0)
    #         Y_all = np.empty(0)
            
    #     self.feasible_set = np.column_stack((X_all, Y_all))
        
    def ys_to_cols(self, ys: np.ndarray) -> np.ndarray:
        cols = ((ys - self.env_extent[2]) / self.y_cell_size).astype(int)
        np.clip(cols, 0, self.num_cols - 1, out=cols)
        return cols

    def xs_to_rows(self, xs: np.ndarray) -> np.ndarray:
        rows = ((xs - self.env_extent[0]) / self.x_cell_size).astype(int)
        np.clip(rows, 0, self.num_rows - 1, out=rows)
        return rows
        
    def is_feasible(self, v, theta_slice, time_slice):
        
        #TODO instead of finding the closest index, it might be better to predict for HJR-FNO at the current state (might be faster)
        #                                           (implement this with is_feasible_ray altogether)
        
        #check if the node 'v' is inside one of the safe reachable set
        for i, feasible_i in enumerate(self.feasible_region):
            
            #find closest matrix indices to the Node
            rows = self.xs_to_rows(np.array([v.x - self.safe_regions[i][0]]))
            cols = self.ys_to_cols(np.array([v.y - self.safe_regions[i][1]]))
            
            row = int(rows[0])
            col = int(cols[0])
            
            if feasible_i[row, col] <= self.safe_margin:
                return True
        
        return False
        
    '''
    For Contingency Planning toward safe region
    '''
    
    # def compute_hjb_gradients(self):
    #     """
    #     Compute spatial derivatives of V(x,y,theta) at a fixed time slice.

    #     V: (Nx, Ny, Ntheta) ndarray
    #     Returns: dVdx, dVdy, dVdtheta
    #     """
        
        
    #     self.N = np.array([50, 50, 25]) #Dimension for [x,y,theta] ;  original (50,50,25)

    #     self.g = Grid(
    #         self.grid_min,
    #         self.grid_max,
    #         self.dims,
    #         self.N,
    #         self.pd
    #     )
        
    #     dx = self.g.dx[0]
    #     dy = self.g.dx[1]
    #     dtheta = self.g.dx[2]
        
    #     dVdx, dVdy, dVdtheta = np.gradient(
    #         V, dx, dy, dtheta, edge_order=2
    #     )

    #     return dVdx, dVdy, dVdtheta

        
            

