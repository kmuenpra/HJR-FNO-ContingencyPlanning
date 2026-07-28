import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')
from matplotlib.colors import LinearSegmentedColormap



class Grid:
    """Grid structure for HJB computation"""
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


class Plane:
    """Plane dynamics class"""
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
        
        if uMode == 'max':
            u[0] = self.vrange[1] if det1 >= 0 else self.vrange[0]
            u[1] = self.wMax if deriv[2] >= 0 else -self.wMax
        elif uMode == 'min':
            u[0] = self.vrange[0] if det1 >= 0 else self.vrange[1]
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


def shapeCylinder(grid: Grid, ignore_dim: int, center: np.ndarray, radius: float) -> np.ndarray:
    """Create a cylindrical target set"""
    data = np.zeros(grid.xs[0].shape)
    
    for i in range(grid.dim):
        if i != ignore_dim - 1:  # MATLAB uses 1-indexing
            data += (grid.xs[i] - center[i])**2
    
    data = radius - np.sqrt(data)
    return data


def main():
    """Main function to run HJB trajectory optimization"""
    
    # Create grid
    grid_min = [-10, -10, 0]
    grid_max = [10, 10, 2*np.pi]
    N = [50, 50, 25]
    g = Grid(grid_min, grid_max, N, periodic_dims=[2])
    
    # Define target set (cylinder)
    R_target = 2.0
    center_target = np.array([0, 0])
    data0 = shapeCylinder(g, 3, center_target, R_target)
    
    # Load precomputed reachable set
    # Replace with your .mat file path
    try:
        mat_data = loadmat('test/HJB_training_mat/50_50_25_SDF_no_obs.mat')  # Update this path
        data_safe = mat_data['BRT_all'][0][0]
        print(f"Loaded reachable set with shape: {data_safe.shape}")
        tMax = 8
        dt = 0.25
        tau = np.arange(0, tMax + dt, dt)
    except FileNotFoundError:
        print("Warning: Could not load .mat file. Using target set as reachable set.")
        # Use target set as a simple reachable set for demonstration
        tMax = 8
        dt = 0.25
        tau = np.arange(0, tMax + dt, dt)
        data_safe = np.repeat(data0[:, :, :, np.newaxis], len(tau), axis=3)
    
    # Initialize plane
    x_init = np.array([-4, 6, 0])
    wMax = 1.0
    vrange = [0, 1]
    dMax = [0, 0, 0]
    plane = Plane(x_init, wMax, vrange, dMax)
    
    # Simulation parameters
    dt = 0.25
    tMax = 8
    Nsteps = int(tMax / dt)
    
    # Storage for trajectory
    state_traj = np.zeros((2, Nsteps))
    
    # Time-stepping loop
    print(f"Running trajectory optimization for {Nsteps} steps...")
    
    # Start from the end time and work backward
    num_time_slices = data_safe.shape[3]
    
    for k in range(Nsteps):
        t = k * dt
        
        # Current state
        x0, y0, th0 = plane.x
        
        # Interpolate to find value at current state
        ix = np.argmin(np.abs(g.vs[0] - x0))
        iy = np.argmin(np.abs(g.vs[1] - y0))
        ith = np.argmin(np.abs(g.vs[2] - (th0 % (2*np.pi))))
        
        # Work BACKWARD in time: start from the largest set (end) toward target (beginning)
        # Find the time slice by going backward: map forward simulation time to backward BRT time
        time_remaining = tMax - t  # How much time left until target
        brt_time_idx = int(time_remaining / dt)
        brt_time_idx = np.clip(brt_time_idx, 0, num_time_slices - 1)
        
        # Use the reachable set at this backward time
        # Early in simulation (small k) -> large time_remaining -> large brt_time_idx -> big set
        # Late in simulation (large k) -> small time_remaining -> small brt_time_idx -> small set
        data_union = data_safe[:, :, :, brt_time_idx]
        
        # Store trajectory
        state_traj[:, k] = plane.x[:2]
        
        # Compute gradients
        Deriv = computeGradients(g, data_union)
        deriv = eval_u(g, Deriv, plane.x)
        
        # Compute optimal control
        u = plane.optCtrl(t, plane.x, deriv, 'min')
        d = plane.optDstb(t, plane.x, deriv, 'max')
        
        # Update state
        plane.updateState(u, dt, d)
        
        if k % 5 == 0:
            print(f"Step {k}/{Nsteps}: x={plane.x[:2]}, θ={plane.x[2]:.2f}, BRT time index={brt_time_idx}")
    
    print("Trajectory optimization complete!")
    
    # Create animation
    create_animation(g, data0, data_safe, state_traj, dt, 'test/fig_results/hjb_trajectory.gif')


def create_animation(grid, data0, data_safe, state_traj, dt, filename='test/fig_results/trajectory.gif'):
    """Create animated GIF of trajectory with value function gradient map"""
    
    # Project to 2D (theta slice at 0)
    theta_idx = 0
    data0_2d = data0[:, :, theta_idx]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    Nsteps = state_traj.shape[1]
    num_time_slices = data_safe.shape[3]
    tMax = (num_time_slices - 1) * dt
    
    # Compute global min/max for consistent colormap across all frames
    vmin = np.min(data_safe)
    vmax = np.max(data_safe)
    
    def update(frame):
        ax.clear()
        ax.set_xlim(grid.axis[0], grid.axis[1])
        ax.set_ylim(grid.axis[2], grid.axis[3])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        
        # Current forward simulation time
        t_forward = frame * dt
        # Corresponding backward reachable tube time
        t_backward = tMax - t_forward
        ax.set_title(f'Forward Time = {t_forward:.2f} s | BRT Time Remaining = {t_backward:.2f} s', 
                    fontsize=14, fontweight='bold')
        
        # Get current state for theta slice
        if frame < Nsteps:
            theta_current = state_traj[1, frame]  # Get actual position
            # Map backward in time for BRT visualization
            time_remaining = tMax - t_forward
            brt_time_idx = int(time_remaining / dt)
            brt_time_idx = np.clip(brt_time_idx, 0, num_time_slices - 1)
        else:
            brt_time_idx = 0  # Target set
        
        # Use theta = 0 slice for visualization (or you can make it state-dependent)
        theta_slice_idx = 0
        
        if data_safe.ndim == 4:
            # Show BRT shrinking: large set at early frames, small set at late frames
            data_slice = data_safe[:, :, theta_slice_idx, brt_time_idx]
        else:
            data_slice = data0_2d
        
        # Plot value function as gradient map (blue to white)
        X, Y = grid.xs[0][:, :, 0], grid.xs[1][:, :, 0]
        
        # Create custom colormap: blue (low values) to white (high values)
        colors = ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#f7fbff', 'white']
        n_bins = 100
        cmap = "gnuplot" #LinearSegmentedColormap.from_list('blue_white', colors, N=n_bins)
        
        # Plot filled contour (gradient map)
        contourf = ax.contourf(X, Y, data_slice, levels=50, cmap=cmap, 
                              vmin=vmin, vmax=vmax, alpha=0.8)
        
        # Add colorbar
        if frame == 0:  # Only create colorbar once
            cbar = plt.colorbar(contourf, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Value Function', fontsize=11)
        
        # Plot zero level set contours
        # BRT boundary (shrinks over time)
        contour_brt = ax.contour(X, Y, data_slice, levels=[0], colors='magenta', 
                                linewidths=2.5, linestyles='--', 
                                label=f'BRT Boundary (t={t_backward:.1f}s)')
        
        # Target set boundary (stays constant)
        contour_target = ax.contour(X, Y, data0_2d, levels=[0], colors='green', 
                                   linewidths=3, label='Target Set')
        
        # Plot trajectory up to current frame
        if frame > 0:
            ax.plot(state_traj[0, :frame], state_traj[1, :frame], 
                   'r-', linewidth=2.5, label='Trajectory', zorder=5)
        
        # Plot current position
        if frame < Nsteps:
            ax.plot(state_traj[0, frame], state_traj[1, frame], 
                   'ro', markersize=12, markerfacecolor='red', 
                   markeredgecolor='darkred', markeredgewidth=2,
                   label='Current Position', zorder=6)
        
        # Add legend with better positioning
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Create animation
    print("Creating animation frames...")
    anim = FuncAnimation(fig, update, frames=Nsteps, interval=dt*1000, repeat=True)
    
    # Save as GIF
    print(f"Saving animation to {filename}...")
    writer = PillowWriter(fps=int(1/dt))
    anim.save(filename, writer=writer)
    print(f"Animation saved to {filename}")
    
    plt.close()


if __name__ == "__main__":
    main()