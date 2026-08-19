"""
Topological PRM — a 2D Python/numpy port of the Fast-Planner topo_prm
(HKUST-Aerial-Robotics/Fast-Planner, topo_prm.cpp / topo_prm.h).

It produces several *topologically distinct* (different-homotopy-class) paths
between a start and goal over a 2D occupancy grid, using a Visibility-PRM graph
plus a UVD (uniform visibility deformation) equivalence check to keep only
distinct classes.

The C++ original depends on ROS / Eigen / PCL / a RayCaster / an EDTEnvironment.
Here those are replaced by:
  - an ESDF built once from the occupancy grid (scipy distance_transform_edt),
  - a line-visibility check that samples the segment on the grid.
The algorithm (createGraph -> searchPaths -> shortcutPaths -> pruneEquivalent
-> selectShortPaths) is otherwise a faithful port, specialized to 2D.

Standalone: depends only on numpy + scipy + this package's ObstacleMap. The only
environment coupling is `_get_dist` / `line_visible`; swap those (e.g. to also
require a state to lie inside an HJR reachable set) to reuse this under MPPI.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


class GraphNode:
    """A roadmap node: either a visibility GUARD or a CONNECTOR between guards."""

    GUARD = "guard"
    CONNECTOR = "connector"

    def __init__(self, pos: np.ndarray, node_type: str, node_id: int) -> None:
        self.pos = np.asarray(pos, dtype=float)
        self.type = node_type
        self.id = node_id
        self.neighbors: List["GraphNode"] = []


class TopoPRM:
    """Topological PRM planner over a 2D ObstacleMap."""

    def __init__(
        self,
        obstacle_map,
        clearance: float = 0.5,
        resolution: float = 0.1,
        sample_inflate: Tuple[float, float] = (1.0, 6.0),
        sample_mode: str = "ellipse",
        max_sample_num: int = 2000,
        max_sample_time: float = 2.0,
        max_raw_path: int = 200,
        max_path_depth: int = 30,
        ratio_to_short: float = 3.0,
        reserve_num: int = 6,
        short_cut_num: int = 1,
        feasible_fn=None,
        seed: int = 0,
    ) -> None:
        """
        Args:
            obstacle_map: ObstacleMap providing the numpy occupancy grid.
            clearance: min obstacle distance [m] for SAMPLED nodes (rejection).
            resolution: discretization step [m] for line-visibility / UVD.
            sample_inflate: (along, perp) [m] padding of the start->goal ellipse
                (only used when sample_mode="ellipse").
            sample_mode: "ellipse" samples a rotated box around the start->goal
                segment (Fast-Planner default; fast when the useful paths hug the
                straight line). "domain" samples uniformly over the whole map
                bounds -- needed when the feasible corridor requires a large
                detour PERPENDICULAR to start->goal (e.g. a U-shaped reachable
                set), which the ellipse never covers. "domain" needs a larger
                max_sample_num for comparable density over the bigger region.
            max_sample_num / max_sample_time: sampling budget (count / seconds).
            max_raw_path: cap on enumerated raw paths (DFS).
            max_path_depth: cap on graph-path node count (DFS safety).
            ratio_to_short: keep a path only if its length < ratio * shortest.
            reserve_num: max number of topo paths returned.
            short_cut_num: string-pulling shortcut iterations.
            feasible_fn: optional callable (N,2)->(N,) bool. When given, a world
                point counts as free only if it is BOTH obstacle-clear AND
                feasible_fn(pt) is True. Use it to constrain every node / edge /
                homotopy deformation to lie inside a set (e.g. an HJR reachable
                set via hjr_fno.points_feasible) -- i.e. treat "outside the set"
                as an obstacle.
            seed: RNG seed.
        """
        self.map = obstacle_map
        self.cell_size = float(obstacle_map._cell_size)
        self.origin = np.asarray(obstacle_map._cell_map_origin, dtype=float)  # (2,)
        self._occ = np.asarray(obstacle_map._map)  # (Nx, Ny), 1 = occupied
        self.Nx, self.Ny = self._occ.shape

        # ESDF: distance [m] from each free cell to the nearest obstacle cell.
        # distance_transform_edt gives distance to the nearest ZERO; feeding the
        # FREE mask (1 where free) yields, per free cell, distance to obstacles;
        # occupied cells -> 0. This replaces EDTEnvironment.evaluateEDT.
        free = (self._occ == 0).astype(np.uint8)
        self._dist = distance_transform_edt(free) * self.cell_size  # (Nx, Ny) [m]

        # Precomputed boolean "free" grid used by the fast line-visibility path
        # (obstacle-free at thresh 0; ANDed with the feasibility grid once
        # rasterize_feasible is called). One combined lookup per sample.
        self._free_grid = self._dist > 0.0  # (Nx, Ny) bool

        self.clearance = float(clearance)
        self.resolution = float(resolution)
        self.sample_inflate = np.asarray(sample_inflate, dtype=float)
        self.sample_mode = str(sample_mode)
        # map bounds [m] for "domain" sampling (fall back to grid extent)
        self._xlim = list(getattr(obstacle_map, "x_lim",
                                  [-0.5 * self.Nx * self.cell_size, 0.5 * self.Nx * self.cell_size]))
        self._ylim = list(getattr(obstacle_map, "y_lim",
                                  [-0.5 * self.Ny * self.cell_size, 0.5 * self.Ny * self.cell_size]))
        self.max_sample_num = int(max_sample_num)
        self.max_sample_time = float(max_sample_time)
        self.max_raw_path = int(max_raw_path)
        self.max_path_depth = int(max_path_depth)
        self.ratio_to_short = float(ratio_to_short)
        self.reserve_num = int(reserve_num)
        self.short_cut_num = int(short_cut_num)
        self.feasible_fn = feasible_fn
        self._feas_grid: Optional[np.ndarray] = None  # set by rasterize_feasible
        # world (x,y) cell centers to draw samples from in sample_mode="corridor"
        # (e.g. the goal-reachable feasible cells of a geodesic cost-to-go).
        self._sample_cells: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(seed)

        # sampling frame (set per query in _setup_sampling)
        self._translation = np.zeros(2)
        self._rotation = np.eye(2)
        self._sample_r = np.ones(2)

        # exposed for visualization / debugging
        self.graph: List[GraphNode] = []
        self.start_node: Optional[GraphNode] = None
        self.goal_node: Optional[GraphNode] = None
        self.raw_paths: List[np.ndarray] = []
        self.short_paths: List[np.ndarray] = []
        self.final_paths: List[np.ndarray] = []

    # =================================================================
    # Environment oracle (grid ESDF + line visibility)
    # =================================================================
    def _get_dist(self, pts: np.ndarray) -> np.ndarray:
        """Obstacle distance [m] at each world point (nearest-cell lookup).
        Out-of-map points return -1 (treated as collision)."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        gx = np.round(pts[:, 0] / self.cell_size + self.origin[0]).astype(int)
        gy = np.round(pts[:, 1] / self.cell_size + self.origin[1]).astype(int)
        inb = (gx >= 0) & (gx < self.Nx) & (gy >= 0) & (gy < self.Ny)
        out = np.full(pts.shape[0], -1.0)
        out[inb] = self._dist[gx[inb], gy[inb]]
        return out

    def _feasible(self, pts: np.ndarray) -> np.ndarray:
        """Reachable-set (or any external) membership mask for world points.
        All-True when no feasible_fn was supplied."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        if self.feasible_fn is None:
            return np.ones(pts.shape[0], dtype=bool)
        return np.asarray(self.feasible_fn(pts), dtype=bool).reshape(-1)

    def rasterize_feasible(self, feasible_fn) -> None:
        """Bake a set-membership constraint (e.g. hjr.points_feasible) onto the
        grid ONCE, so every later feasibility query is an O(1) grid lookup
        instead of a per-point interpolation. Call this whenever the underlying
        set (reachable set) updates, then re-run find_topo_paths.

        Cost: a single BATCHED feasible_fn call over all Nx*Ny cell centers,
        replacing the thousands of tiny per-query calls the PRM would otherwise
        make. This is the main lever for re-running the PRM online."""
        ii, jj = np.meshgrid(np.arange(self.Nx), np.arange(self.Ny), indexing="ij")
        xs = (ii.ravel() - self.origin[0]) * self.cell_size
        ys = (jj.ravel() - self.origin[1]) * self.cell_size
        pts = np.column_stack([xs, ys])
        feas = np.asarray(feasible_fn(pts), dtype=bool).reshape(self.Nx, self.Ny)
        self._feas_grid = feas
        self.feasible_fn = self._grid_feasible  # subsequent queries hit the grid
        # fold feasibility into the combined free grid (obstacle-free AND feasible)
        self._free_grid = (self._dist > 0.0) & feas

    def _grid_feasible(self, pts: np.ndarray) -> np.ndarray:
        """O(1) nearest-cell lookup into the rasterized feasibility grid.
        Out-of-map points are infeasible."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        gx = np.round(pts[:, 0] / self.cell_size + self.origin[0]).astype(int)
        gy = np.round(pts[:, 1] / self.cell_size + self.origin[1]).astype(int)
        inb = (gx >= 0) & (gx < self.Nx) & (gy >= 0) & (gy < self.Ny)
        out = np.zeros(pts.shape[0], dtype=bool)
        out[inb] = self._feas_grid[gx[inb], gy[inb]]
        return out

    def line_visible(
        self, p1: np.ndarray, p2: np.ndarray, thresh: float = 0.0
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """True if the straight segment p1->p2 stays clear of obstacles AND
        (if feasible_fn is set) inside the feasible set -- i.e. every sampled
        point has obstacle distance > thresh and is feasible. Returns
        (visible, first_blocking_point_or_None)."""
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        length = float(np.linalg.norm(p2 - p1))
        n = max(2, int(math.ceil(length / self.resolution)) + 1)
        ts = np.linspace(0.0, 1.0, n)[:, None]
        pts = p1[None, :] + ts * (p2 - p1)[None, :]  # (n, 2)

        if thresh == 0.0:
            # fast path: one combined-grid lookup (obstacle-free AND feasible).
            gx = np.round(pts[:, 0] / self.cell_size + self.origin[0]).astype(int)
            gy = np.round(pts[:, 1] / self.cell_size + self.origin[1]).astype(int)
            inb = (gx >= 0) & (gx < self.Nx) & (gy >= 0) & (gy < self.Ny)
            ok = np.zeros(n, dtype=bool)
            ok[inb] = self._free_grid[gx[inb], gy[inb]]
        else:
            ok = (self._get_dist(pts) > thresh) & self._feasible(pts)

        blocked = ~ok
        if np.any(blocked):
            first = int(np.argmax(blocked))
            return False, pts[first]
        return True, None

    # =================================================================
    # Sampling
    # =================================================================
    def _setup_sampling(self, start: np.ndarray, goal: np.ndarray) -> None:
        """Build the rotated ellipse (box) region aligned with start->goal."""
        self._translation = 0.5 * (start + goal)
        diff = goal - start
        dist = float(np.linalg.norm(diff))
        if dist < 1e-6:
            xdir = np.array([1.0, 0.0])
        else:
            xdir = diff / dist
        ydir = np.array([-xdir[1], xdir[0]])  # +90 deg
        self._rotation = np.column_stack([xdir, ydir])  # cols = axes
        self._sample_r = np.array(
            [0.5 * dist + self.sample_inflate[0], self.sample_inflate[1]]
        )

    def set_sample_cells(self, cells: Optional[np.ndarray]) -> None:
        """Provide the world (x,y) cell centers for sample_mode="corridor"
        (typically the goal-reachable feasible cells from a cost-to-go field).
        Sampling then draws from these cells + sub-cell jitter, so essentially
        every sample lands in the feasible corridor (no rejection waste)."""
        self._sample_cells = None if cells is None else np.asarray(cells, dtype=float)

    def _sample_point(self) -> Optional[np.ndarray]:
        """One uniform sample in the sampling region; None if rejected (too close
        to an obstacle / outside the feasible set / out of map)."""
        if (
            self.sample_mode == "corridor"
            and self._sample_cells is not None
            and len(self._sample_cells)
        ):
            i = int(self._rng.integers(len(self._sample_cells)))
            jitter = self._rng.uniform(-0.5, 0.5, size=2) * self.cell_size
            pt = self._sample_cells[i] + jitter
        elif self.sample_mode == "domain":
            pt = np.array([
                self._rng.uniform(self._xlim[0], self._xlim[1]),
                self._rng.uniform(self._ylim[0], self._ylim[1]),
            ])
        else:  # "ellipse": rotated box around the start->goal segment
            r = self._rng.uniform(-1.0, 1.0, size=2) * self._sample_r
            pt = self._translation + self._rotation @ r
        if self._get_dist(pt[None, :])[0] <= self.clearance:
            return None
        if not self._feasible(pt[None, :])[0]:
            return None
        return pt

    # =================================================================
    # Graph construction (Visibility PRM)
    # =================================================================
    def _find_visib_guards(self, pt: np.ndarray) -> List[GraphNode]:
        """Up to two guards visible from pt (early-stops at 2, as in the
        original findVisibGuard)."""
        visible: List[GraphNode] = []
        for node in self.graph:
            if node.type != GraphNode.GUARD:
                continue
            if self.line_visible(pt, node.pos)[0]:
                visible.append(node)
                if len(visible) == 2:
                    break
        return visible

    def _need_connection(
        self, g1: GraphNode, g2: GraphNode, pt: np.ndarray
    ) -> bool:
        """Whether a connector at `pt` between g1,g2 is a topologically NEW
        link. If an existing connector between the same guards is UVD-equivalent
        and longer, replace it with the shorter `pt` (and report not-new)."""
        path1 = [g1.pos, pt, g2.pos]
        for c in g1.neighbors:
            if c.type != GraphNode.CONNECTOR:
                continue
            if g2 not in c.neighbors:
                continue
            path2 = [g1.pos, c.pos, g2.pos]
            if self.same_topo_path(path1, path2):
                if self.path_length(path1) < self.path_length(path2):
                    c.pos = np.asarray(pt, dtype=float)  # keep the shorter one
                return False
        return True

    def create_graph(self, start: np.ndarray, goal: np.ndarray) -> None:
        """Build the visibility roadmap between start and goal."""
        self.graph = []
        self.start_node = GraphNode(start, GraphNode.GUARD, 0)
        self.goal_node = GraphNode(goal, GraphNode.GUARD, 1)
        self.graph.append(self.start_node)
        self.graph.append(self.goal_node)

        self._setup_sampling(np.asarray(start, float), np.asarray(goal, float))

        node_id = 2
        t0 = time.time()
        n_sampled = 0
        while (
            n_sampled < self.max_sample_num
            and (time.time() - t0) < self.max_sample_time
        ):
            n_sampled += 1
            pt = self._sample_point()
            if pt is None:
                continue

            guards = self._find_visib_guards(pt)
            if len(guards) == 0:
                self.graph.append(GraphNode(pt, GraphNode.GUARD, node_id))
                node_id += 1
            elif len(guards) == 2:
                if not self._need_connection(guards[0], guards[1], pt):
                    continue
                conn = GraphNode(pt, GraphNode.CONNECTOR, node_id)
                node_id += 1
                conn.neighbors = [guards[0], guards[1]]
                guards[0].neighbors.append(conn)
                guards[1].neighbors.append(conn)
                self.graph.append(conn)
            # 1 or >2 visible guards -> redundant sample, skip

    # =================================================================
    # Path search (DFS enumeration of raw paths)
    # =================================================================
    def search_paths(self) -> List[np.ndarray]:
        """Enumerate raw start->goal paths through the graph (DFS)."""
        raw: List[List[np.ndarray]] = []

        def dfs(node: GraphNode, path: List[GraphNode], visited: set) -> None:
            if len(raw) >= self.max_raw_path:
                return
            if len(path) > self.max_path_depth:
                return
            if node is self.goal_node:
                raw.append([n.pos.copy() for n in path])
                return
            for nb in node.neighbors:
                if id(nb) in visited:
                    continue
                visited.add(id(nb))
                path.append(nb)
                dfs(nb, path, visited)
                path.pop()
                visited.discard(id(nb))

        dfs(self.start_node, [self.start_node], {id(self.start_node)})
        return [np.asarray(p, dtype=float) for p in raw]

    # =================================================================
    # Topological equivalence (UVD) + geometry helpers
    # =================================================================
    @staticmethod
    def path_length(path) -> float:
        p = np.asarray(path, dtype=float)
        if len(p) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    def _discretize(self, path, n: int) -> np.ndarray:
        """Resample a polyline to n points evenly spaced by arc length."""
        p = np.asarray(path, dtype=float)
        if len(p) == 1:
            return np.repeat(p, n, axis=0)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        if total < 1e-9:
            return np.repeat(p[:1], n, axis=0)
        targets = np.linspace(0.0, total, n)
        x = np.interp(targets, cum, p[:, 0])
        y = np.interp(targets, cum, p[:, 1])
        return np.column_stack([x, y])

    def same_topo_path(self, path1, path2, thresh: float = 0.0) -> bool:
        """UVD check: two paths are the same homotopy class if, after resampling
        both to equal length, every pair of corresponding points can 'see' each
        other (the straight interpolation between them is collision-free)."""
        len1 = self.path_length(path1)
        len2 = self.path_length(path2)
        max_len = max(len1, len2)
        n = max(2, int(math.ceil(max_len / self.resolution)))
        pts1 = self._discretize(path1, n)
        pts2 = self._discretize(path2, n)
        for a, b in zip(pts1, pts2):
            if not self.line_visible(a, b, thresh)[0]:
                return False
        return True

    # =================================================================
    # Shortcut (string-pulling) + selection
    # =================================================================
    def shortcut_path(self, path: np.ndarray) -> np.ndarray:
        """Greedy string-pulling: drop interior waypoints whose removal keeps the
        path collision-free. Never adds detours, so it cannot change homotopy
        class -- it only cleans up the raw graph path."""
        result = np.asarray(path, dtype=float)
        for _ in range(max(1, self.short_cut_num)):
            n = max(2, int(math.ceil(self.path_length(result) / self.resolution)))
            dpath = self._discretize(result, n)
            pulled = [dpath[0]]
            for i in range(2, len(dpath)):
                if not self.line_visible(pulled[-1], dpath[i])[0]:
                    pulled.append(dpath[i - 1])
            pulled.append(dpath[-1])
            result = np.asarray(pulled, dtype=float)
        return result

    def prune_equivalent(self, paths: List[np.ndarray]) -> List[np.ndarray]:
        """Keep one representative (the shortest) per homotopy class."""
        ordered = sorted(paths, key=self.path_length)
        kept: List[np.ndarray] = []
        for p in ordered:
            if not any(self.same_topo_path(p, q) for q in kept):
                kept.append(p)
        return kept

    def select_short_paths(self, paths: List[np.ndarray]) -> List[np.ndarray]:
        """Sort by length, keep the shortest plus any within ratio_to_short of
        it, capped at reserve_num."""
        if not paths:
            return []
        ordered = sorted(paths, key=self.path_length)
        min_len = self.path_length(ordered[0])
        selected = [
            p
            for p in ordered
            if self.path_length(p) <= self.ratio_to_short * min_len
        ]
        return selected[: self.reserve_num]

    # =================================================================
    # Top-level
    # =================================================================
    def find_topo_paths(
        self, start, goal
    ) -> List[np.ndarray]:
        """Return a list of topologically-distinct waypoint paths start->goal.
        Each path is an (M, 2) array of world-frame [x, y] waypoints."""
        start = np.asarray(start, dtype=float)[:2]
        goal = np.asarray(goal, dtype=float)[:2]

        self.create_graph(start, goal)
        self.raw_paths = self.search_paths()
        if not self.raw_paths:
            self.short_paths = []
            self.final_paths = []
            return []

        self.short_paths = [self.shortcut_path(p) for p in self.raw_paths]
        pruned = self.prune_equivalent(self.short_paths)
        self.final_paths = self.select_short_paths(pruned)
        return self.final_paths
