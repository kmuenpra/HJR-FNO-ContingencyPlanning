"""Geodesic cost-to-go over the feasible (reachable-set) grid.

Replaces MPPI's greedy straight-line goal term `||x - goal||` with the distance
to the goal measured *through the feasible corridor* (obstacle- and reachable-set
aware). A state "down the left column" of a U-shaped feasible set is then
genuinely closer to the goal than one pinned against the top wall, so MPPI's
softmax weighting rewards following the good homotopy class instead of penalizing
the detour -- which is what lets the topological guidance actually take effect.

Built once per feasible-set change (obstacle reveal), like the topo replan:
  ctg = GeodesicCostToGo(obstacle_map, feasible_fn=env.points_feasible_xy)
  ctg.recompute(goal_xy)            # ~ms on a 300x300 grid
  goal_cost = ctg.value_torch(state[:, :2])   # per-step, on-device lookup

Uses skimage.graph.MCP_Geometric (obstacle-aware geodesic) -- available in the
`rrtx` conda env.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from skimage.graph import MCP_Geometric


class GeodesicCostToGo:
    def __init__(self, obstacle_map, feasible_fn=None, unreachable_cost: float = 1e3):
        """
        obstacle_map: env ObstacleMap (grid geometry + occupancy read live).
        feasible_fn: (N,2)->bool reachable-set membership; cells failing it are
            treated as barriers (same as obstacles). None -> obstacles only.
        unreachable_cost: finite cost [m] assigned to cells the goal can't reach
            through the feasible set (keeps the MPPI softmax finite).
        """
        self.map = obstacle_map
        self.cell_size = float(obstacle_map._cell_size)
        self.origin = np.asarray(obstacle_map._cell_map_origin, dtype=float)  # (2,)
        self._occ = np.asarray(obstacle_map._map)  # (Nx, Ny), 1 = occupied
        self.Nx, self.Ny = self._occ.shape
        self.feasible_fn = feasible_fn
        self.unreachable_cost = float(unreachable_cost)

        self.C: np.ndarray | None = None  # (Nx, Ny) geodesic distance-to-goal [m]
        self.goal_xy: np.ndarray | None = None
        self._C_t = None  # cached device tensor
        self._C_dev = None

    # ------------------------------------------------------------------
    def _feasible_grid(self) -> np.ndarray:
        """Boolean free grid: obstacle-free AND (if given) feasible_fn True."""
        free = self._occ == 0
        if self.feasible_fn is not None:
            ii, jj = np.meshgrid(np.arange(self.Nx), np.arange(self.Ny), indexing="ij")
            xs = (ii.ravel() - self.origin[0]) * self.cell_size
            ys = (jj.ravel() - self.origin[1]) * self.cell_size
            feas = np.asarray(
                self.feasible_fn(np.column_stack([xs, ys])), dtype=bool
            ).reshape(self.Nx, self.Ny)
            free = free & feas
        return free

    def _world_to_cell(self, xy) -> tuple:
        gx = int(round(xy[0] / self.cell_size + self.origin[0]))
        gy = int(round(xy[1] / self.cell_size + self.origin[1]))
        return gx, gy

    def _nearest_free_cell(self, cell, free) -> tuple:
        gx, gy = cell
        gx = int(min(max(gx, 0), self.Nx - 1))
        gy = int(min(max(gy, 0), self.Ny - 1))
        if free[gx, gy]:
            return gx, gy
        idx = np.argwhere(free)
        if len(idx) == 0:
            return gx, gy  # nothing free (degenerate); caller handles all-inf
        d2 = (idx[:, 0] - gx) ** 2 + (idx[:, 1] - gy) ** 2
        k = int(np.argmin(d2))
        return int(idx[k, 0]), int(idx[k, 1])

    # ------------------------------------------------------------------
    def recompute(self, goal_xy) -> "GeodesicCostToGo":
        """(Re)build the geodesic distance-to-goal field over the current
        feasible grid. Call whenever obstacles / the reachable set change."""
        self.goal_xy = np.asarray(goal_xy, dtype=float)[:2]
        free = self._feasible_grid()
        # MCP_Geometric: cost of traversing each cell; inf = barrier (impassable).
        costs = np.where(free, 1.0, np.inf).astype(float)
        gcell = self._nearest_free_cell(self._world_to_cell(self.goal_xy), free)

        mcp = MCP_Geometric(costs)
        cumulative, _ = mcp.find_costs([gcell])  # geodesic cost from goal, cell units
        C = np.asarray(cumulative, dtype=float) * self.cell_size  # -> meters
        C[~np.isfinite(C)] = self.unreachable_cost  # unreachable -> finite penalty
        self.C = C
        self._C_t = None  # invalidate device cache
        return self

    # ------------------------------------------------------------------
    def reachable_cells(self) -> np.ndarray:
        """World (x, y) centers of feasible cells reachable from the goal
        (finite geodesic cost). Used to bias TopoPRM sampling into the corridor."""
        if self.C is None:
            return np.empty((0, 2))
        idx = np.argwhere(self.C < self.unreachable_cost)
        if len(idx) == 0:
            return np.empty((0, 2))
        xs = (idx[:, 0] - self.origin[0]) * self.cell_size
        ys = (idx[:, 1] - self.origin[1]) * self.cell_size
        return np.column_stack([xs, ys])

    def value(self, pts) -> np.ndarray:
        """Cost-to-go [m] at world points (numpy, nearest-cell)."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        gx = np.round(pts[:, 0] / self.cell_size + self.origin[0]).astype(int)
        gy = np.round(pts[:, 1] / self.cell_size + self.origin[1]).astype(int)
        inb = (gx >= 0) & (gx < self.Nx) & (gy >= 0) & (gy < self.Ny)
        out = np.full(pts.shape[0], self.unreachable_cost)
        out[inb] = self.C[gx[inb], gy[inb]]
        return out

    def value_torch(self, xy):
        """Cost-to-go [m] at world points (torch, on xy.device, nearest-cell).
        Out-of-map points get unreachable_cost."""
        if self._C_t is None or self._C_dev != xy.device:
            self._C_t = torch.as_tensor(self.C, device=xy.device, dtype=xy.dtype)
            self._C_dev = xy.device
        gx = torch.round(xy[:, 0] / self.cell_size + float(self.origin[0])).long()
        gy = torch.round(xy[:, 1] / self.cell_size + float(self.origin[1])).long()
        inb = (gx >= 0) & (gx < self.Nx) & (gy >= 0) & (gy < self.Ny)
        v = self._C_t[gx.clamp(0, self.Nx - 1), gy.clamp(0, self.Ny - 1)]
        return torch.where(inb, v, torch.full_like(v, self.unreachable_cost))
