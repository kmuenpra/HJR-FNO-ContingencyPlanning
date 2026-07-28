# =========================
# Standard library imports
# =========================
import os
import sys
import time
import math
import heapq
import itertools
import traceback
import random
import numbers
from collections import deque
from collections.abc import Sequence
from typing import Dict, List, Tuple, Callable, Optional
from pathlib import Path

# =========================
# Third-party imports
# =========================
import numpy as np
import kdtree
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as patches
from scipy.ndimage import distance_transform_edt
from scipy.io import loadmat
from skimage.measure import marching_cubes

import dill
import yaml

# =========================
# Local / project imports
# =========================
import env
import plotting
import utils
import atsp

from HJR_FNO.HJR_FNO import FNO1d, SpectralConv1d


class Node(Sequence):

    _id_counter = itertools.count()

    def __init__(self, n, lmc=np.inf, cost_to_goal=np.inf):
        self.id: int = next(Node._id_counter)
        self.n = n
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.children = set([])
        self.cost_to_goal = cost_to_goal
        self.lmc = lmc
        self.infinite_dist_nodes = set([])
        self.N_o_plus = set([])
        self.N_o_minus = set([])
        self.N_r_plus = set([])
        self.N_r_minus = set([])
        self.active = False

    def __eq__(self, other):
        return id(self) == id(other) or \
            self.get_key() == other.get_key() or \
            math.hypot(self.x - other.x, self.y - other.y) < 1e-6

    def __lt__(self, other):
        return 1

    def __hash__(self):
        return hash(self.n)

    def __getitem__(self, i):
        return self.n[i]

    def __len__(self):
        return 2

    def all_out_neighbors(self):
        return self.N_o_plus.union(self.N_r_plus)

    def all_in_neighbors(self):
        return self.N_o_minus.union(self.N_r_minus)

    def set_parent(self, new_parent):
        if self.parent:
            try:
                self.parent.children.remove(self)
            except:
                print("")
        self.parent = new_parent
        new_parent.children.add(self)

    def get_key(self):
        return (min(self.cost_to_goal, self.lmc), self.cost_to_goal)

    def cull_neighbors(self, r):
        N_r_plus_list = list(self.N_r_plus)
        for u in N_r_plus_list:
            if self.cost_to_goal == 0.0 and self.lmc == 0.0:
                print("Goal is a neighbor node!")
            else:
                try:
                    if (not self.parent or (self.parent and self.parent != u)) and r < self.distance(u):
                        N_r_plus_list.remove(u)
                        try:
                            u.N_r_minus.remove(self)
                        except KeyError:
                            pass
                except AttributeError:
                    print("Position of none with no parent", (self.x, self.y))
                    traceback.print_exc()
        self.N_r_plus = set(N_r_plus_list)

    def update_LMC(self, orphan_nodes, r, epsilon, utils):
        self.cull_neighbors(r)
        lmcs = [(u, self.distance(u) + u.lmc) for u in (self.all_out_neighbors() - orphan_nodes) if u.parent and u.parent != self]
        if not lmcs:
            return
        p_prime, lmc_prime = min(lmcs, key=lambda x: x[1])
        if lmc_prime < self.lmc and not utils.is_collision(self, p_prime):
            self.lmc = lmc_prime
            self.set_parent(p_prime)

    def distance(self, other):
        return np.inf if other in self.infinite_dist_nodes else math.hypot(self.x - other.x, self.y - other.y)

    def clone(self):
        v = Node((self.x, self.y))
        v.cost_to_goal = self.cost_to_goal
        v.lmc = self.lmc
        return v


class RRTX:

    from HJR_FNO.HJR_FNO import HJR_FNO, Grid

    def __init__(
        self,
        x_start: Tuple[float, float],
        x_goal: Tuple[float, float],
        other_goals: List,
        other_goals_id: List,
        heading: float,
        lidar_range: float,
        step_len: float,
        gamma_FOS: float,
        epsilon: float,
        bot_sample_rate: float,
        iter_max: int,
        safe_regions: List[Sequence[float]],
        hjr_fno: HJR_FNO,
        HJ_contingency_enable: bool,
        fig: Figure,
        ax: Axes,
        plotting: plotting.Plotting,
        # ── Webots integration ──────────────────────────────────────────────
        # Optional callback: lidar_fn(robot_position, unknown_obs_circle, lidar_range)
        #                    -> (remaining_unknown_obs, detected_obs)
        # When None, falls back to self.utils.lidar_detected() (standalone mode).
        lidar_fn: Optional[Callable] = None,
        # Optional callback: pose_fn() -> [x, y, theta]
        # When provided, robot_state is synced from this source each step.
        pose_fn: Optional[Callable] = None,
        # Optional callback: drive_fn(waypoint_x, waypoint_y) -> reached:bool
        # When provided, physical motion is delegated to this function.
        drive_fn: Optional[Callable] = None,
        ) -> None:

        self.s_goal = Node(x_goal, lmc=0.0, cost_to_goal=0.0)
        self.s_bot = None
        self._pending_reset_target = None
        self._pending_reset_heading = None

        self.prob_q = 0.9
        self.other_goals_id = other_goals_id
        self.other_goals = []
        for g in other_goals:
            self.other_goals.append(Node((g[0], g[1])))

        self.env = env.Env(safe_regions=safe_regions)
        self.plotting = plotting
        self.utils = utils.Utils(environment=self.env)

        self.step_len = step_len
        self.epsilon = epsilon
        self.bot_sample_rate = bot_sample_rate
        self.search_radius = 0.0
        self.iter_max = iter_max
        self.kd_tree = kdtree.create([self.s_goal])
        sys.setrecursionlimit(3000)
        self.all_nodes_coor = []
        self.tree_nodes = [self.s_goal]
        self.orphan_nodes = set([])
        self.Q = []
        self.path_to_goal = np.array([False for _ in range(len(other_goals))])
        self.robot_path_to_goal = False

        self.robot_state = [0, 0, 0]
        self.robot_speed = 0.6
        self.lidar_range = lidar_range
        self.utils.sensing_radius = lidar_range

        self.Tf_reach = hjr_fno.Tf_reach
        self.hjr_fno = hjr_fno
        self.hjr_fno.utils.sensing_radius = lidar_range
        self.safe_regions = safe_regions
        self.HJ_contingency_enable = HJ_contingency_enable
        self.contingency_triggered = False

        self.path = []
        self.path_node = []
        self.multi_paths = [[] for _ in range(len(self.other_goals))]

        self.fig = fig
        self.ax = ax
        self.nodes_scatter = self.ax.scatter([], [], s=4, c='gray', alpha=0.5)
        self.edge_col = LineCollection([], colors='blue', linewidths=0.5)
        self.path_col = LineCollection([], colors='red', linewidths=1.0)
        self.ax.add_collection(self.edge_col)
        self.ax.add_collection(self.path_col)

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        self.unknown_obs_circle = self.env.unknown_obs_circle

        self.invalid_nodes = None

        self.d = 2
        self.zeta_d = np.pi
        self.gamma_FOS = gamma_FOS
        self.update_gamma()

        # ── Webots callbacks (None = standalone/simulation mode) ────────────
        self._lidar_fn = lidar_fn
        self._pose_fn  = pose_fn
        self._drive_fn = drive_fn

    # ─────────────────────────────────────────────────────────────────────────
    # Internal lidar dispatch
    # Called from planning_with_robot() at exactly the point where
    # self.utils.lidar_detected() was previously called.
    # ─────────────────────────────────────────────────────────────────────────
    def _lidar_detected(self):
        """
        Returns (remaining_unknown_obs, detected_obs) as lists of [x, y, r].

        In standalone mode  → delegates to utils.lidar_detected() (unchanged).
        In Webots mode      → delegates to the injected lidar_fn callback.
        """
        if self._lidar_fn is not None:
            remaining, detected = self._lidar_fn(
                self.robot_position,
                self.unknown_obs_circle,
                self.lidar_range,
            )
        else:
            remaining, detected = self.utils.lidar_detected(self.robot_position)

        # Keep all three references in sync regardless of which path was taken
        self.unknown_obs_circle          = remaining
        self.utils.unknown_obs_circle    = remaining
        self.env.unknown_obs_circle      = remaining
        self.plotting.unknown_obs_circle = remaining

        return remaining, detected

    # ─────────────────────────────────────────────────────────────────────────
    # Internal pose sync
    # ─────────────────────────────────────────────────────────────────────────
    def _sync_pose(self):
        """
        If a pose_fn callback is registered (Webots GPS/Compass), overwrite
        robot_state with the real sensor reading.  Otherwise does nothing
        (standalone: robot_state is maintained by update_robot_position_dubins).
        """
        if self._pose_fn is not None:
            x, y, theta = self._pose_fn()
            self.robot_state    = [x, y, theta]
            self.robot_position = [x, y]

    def _dist_to_goal(self, node: Node, goal: Tuple[float, float]) -> float:
        dx = node.x - goal[0]
        dy = node.y - goal[1]
        return math.hypot(dx, dy)

    def planning(self, iter_max=None, robots_plan=False):

        if iter_max is None:
            iter_max = self.iter_max

        plt.gca().set_aspect('equal', adjustable='box')
        self.edge_col.set_animated(True)
        self.path_col.set_animated(True)
        plt.show(block=False)
        self.ax.draw_artist(self.edge_col)
        self.fig.canvas.blit(self.ax.bbox)

        for i in range(iter_max):

            self.search_radius = self.shrinking_ball_radius()

            if np.random.random() < self.prob_q:
                v = self.random_node()
            else:
                candidates = [g for g, reached in zip(self.other_goals, self.path_to_goal) if not reached]
                if candidates:
                    v = random.choice(candidates)
                elif robots_plan and (not self.robot_path_to_goal):
                    v = Node((self.s_bot.x, self.s_bot.y))
                else:
                    v = self.random_node()

            v_nearest = self.nearest(v)
            v = self.saturate(v_nearest, v)

            if v and not self.utils.is_collision(v_nearest, v) and self.is_feasible_ray(v_nearest, v):
                self.extend(v, v_nearest, robots_plan=robots_plan)
                if v.parent is not None:
                    self.rewire_neighbours(v, robots_plan=robots_plan)
                    self.reduce_inconsistency(robots_plan=robots_plan)

            if robots_plan and self.s_bot.cost_to_goal < np.inf:
                self.robot_path_to_goal = True

            for j, goal_j in enumerate(self.other_goals):
                if goal_j.cost_to_goal < np.inf:
                    self.path_to_goal[j] = True

    def reset_robot_position(self, new_position: Tuple[float, float], heading: float = None):
        new_node = Node(new_position)
        nearest  = self.nearest(new_node)
        dist     = math.hypot(nearest.x - new_node.x, nearest.y - new_node.y)

        SNAP_THRESHOLD = 0.5

        if dist < SNAP_THRESHOLD and nearest.lmc < np.inf:
            self.s_bot = nearest
            self._pending_reset_target = None
        else:
            new_node   = self.saturate(nearest, new_node)
            V_near     = self.near(new_node)
            if not V_near:
                V_near = [nearest]
            V_near_free = [u for u in V_near if not self.utils.is_collision(u, new_node)]

            if not V_near_free:
                self._pending_reset_target  = Node(new_position)
                self._pending_reset_heading = heading
                self.robot_position = list(new_position)
                self.robot_state = [new_position[0], new_position[1],
                                    heading if heading is not None else self.robot_state[2]]
                self.robot_path_to_goal = False
                self.path = []
                return False

            self._pending_reset_target = None
            self.find_parent(new_node, V_near_free)

            if new_node.parent is None:
                return False

            self.add_node(new_node)

            for u in V_near_free:
                new_node.N_o_plus.add(u)
                new_node.N_o_minus.add(u)
                u.N_r_plus.add(new_node)
                u.N_r_minus.add(new_node)

            self.rewire_neighbours(new_node)
            self.reduce_inconsistency()
            self.s_bot = new_node
            self._pending_reset_target = None

        self.robot_position = [self.s_bot.x, self.s_bot.y]
        self.robot_state = [self.s_bot.x, self.s_bot.y,
                            heading if heading is not None else self.robot_state[2]]
        self.robot_path_to_goal = self.s_bot.lmc < np.inf
        self.update_path(self.s_bot)
        return self._pending_reset_target is None

    def reset_robot_v2(self, current_state, iter_max=500):
        self.robot_path_to_goal = False
        self.path = []
        self.s_bot = Node((current_state[0], current_state[1]))
        self.robot_state    = current_state
        self.robot_position = current_state[:2]

        v_nearest = self.nearest(self.s_bot)
        if math.hypot(v_nearest.x - self.s_bot.x, v_nearest.y - self.s_bot.y) < 1e-3:
            self.s_bot = v_nearest

        if self.s_bot.lmc < np.inf:
            self.verify_queue(self.s_bot)

        if self.s_bot.cost_to_goal < np.inf:
            self.robot_path_to_goal = True
            self.update_path(self.s_bot)
        else:
            i = 0
            while (not self.robot_path_to_goal) and i <= iter_max:
                if np.random.random() > self.prob_q:
                    v = Node((current_state[0], current_state[1]))
                else:
                    v = self.random_node()

                v_nearest = self.nearest(v)
                v = self.saturate(v_nearest, v)

                if v and not self.utils.is_collision(v_nearest, v) and self.is_feasible_ray(v_nearest, v):
                    self.extend(v, v_nearest, robots_plan=True)
                    if v is self.s_bot and self.s_bot.lmc < np.inf:
                        self.verify_queue(self.s_bot)
                    if v.parent is not None:
                        self.rewire_neighbours(v, robots_plan=True)
                        self.reduce_inconsistency(robots_plan=True)

                if self.s_bot.cost_to_goal < np.inf:
                    self.robot_path_to_goal = True
                    self.update_path(self.s_bot)

                i += 1
                if i % 200 == 0:
                    print(f"replanning robot's path iteration {i}")

    def update_robot_heading(self, new_heading=None):
        if new_heading is not None:
            self.robot_state[2] = new_heading
        elif self.s_bot.parent is not None:
            dx = self.s_bot.parent.x - self.s_bot.x
            dy = self.s_bot.parent.y - self.s_bot.y
            self.robot_state[2] = math.atan2(dy, dx)

    def planning_with_robot(self, steps=10):

        new_obs_flag       = False
        all_new_obs        = []
        traversed_distance = 0.0

        for step_idx in range(steps):

            if self._pending_reset_target is not None:
                self._pending_reset_target = Node((self.robot_position[0],
                                                   self.robot_position[1]))
                self._try_connect_pending()

            if self.s_bot.cost_to_goal == np.inf:
                self.robot_path_to_goal = False
                if self.s_bot.lmc < np.inf:
                    self.verify_queue(self.s_bot)

            if self.contingency_triggered:
                return all_new_obs, new_obs_flag, traversed_distance

            if self.robot_path_to_goal:

                if self.s_bot.cost_to_goal == 0.0 and self.s_bot.lmc == 0.0:
                    return all_new_obs, new_obs_flag, traversed_distance

                # ── Pose: sync from real sensor (Webots) or integrate Dubins ──
                if self._pose_fn is not None:
                    self._sync_pose()
                else:
                    self.robot_state = self.utils.update_robot_position_dubins(
                        self.robot_state,
                        [self.s_bot.parent.x, self.s_bot.parent.y],
                        0.01, v=self.robot_speed
                    )
                    self.robot_position = self.robot_state[:2]

                # ── Physical drive command (Webots only, no-op in standalone) ─
                if self._drive_fn is not None and self.s_bot.parent is not None:
                    self._drive_fn(self.s_bot.parent.x, self.s_bot.parent.y)

                # ── Lidar: Webots sensor or software simulation ────────────────
                # This is the single call site that was previously:
                #   self.unknown_obs_circle, detected_obs = self.utils.lidar_detected(self.robot_position)
                _, detected_obs = self._lidar_detected()

                if len(detected_obs) > 0:
                    all_new_obs += detected_obs
                    new_obs_flag = True

                    print(f"\n ----- Rewiring Trees due to {len(detected_obs)} "
                          f"newly-detected obstacle(s) ----- ")
                    print("Obstacle location: ", detected_obs)

                    if self.HJ_contingency_enable:
                        self.hjr_fno.update_obs(detected_obs)

                    for obs in detected_obs:
                        self.update_obstacles(obs, robots_plan=True, print_time=True)
                        self.update_path(self.s_bot)

                if self.s_bot.parent is not None:
                    if math.hypot(self.robot_position[0] - self.s_bot.parent.x,
                                  self.robot_position[1] - self.s_bot.parent.y) < 0.5:
                        traversed_distance += self.s_bot.distance(self.s_bot.parent)
                        self.s_bot = self.s_bot.parent

            # ── Tree expansion ────────────────────────────────────────────────
            self.search_radius = self.shrinking_ball_radius()

            if np.random.random() < self.prob_q:
                v = self.random_node()
            else:
                candidates = [g for g, reached in zip(self.other_goals, self.path_to_goal)
                              if not reached]
                if not self.robot_path_to_goal:
                    v = Node((self.s_bot.x, self.s_bot.y))
                elif candidates:
                    v = random.choice(candidates)
                else:
                    v = self.random_node()

            v_nearest = self.nearest(v)
            v = self.saturate(v_nearest, v)

            if v and not self.utils.is_collision(v_nearest, v) and self.is_feasible_ray(v_nearest, v):
                self.extend(v, v_nearest, robots_plan=True)
                if v.parent is not None:
                    self.rewire_neighbours(v, robots_plan=True)
                    self.reduce_inconsistency(robots_plan=True)

            if self.s_bot.cost_to_goal < np.inf:
                self.robot_path_to_goal = True

            for j, goal_j in enumerate(self.other_goals):
                if goal_j.cost_to_goal < np.inf:
                    self.path_to_goal[j] = True

            if step_idx % 10 == 0:
                self.fig.canvas.flush_events()

        return all_new_obs, new_obs_flag, traversed_distance

    def extend(self, v, v_nearest, robots_plan=False):
        V_near = self.near(v)
        if not V_near:
            V_near.append(v_nearest)
        self.find_parent(v, V_near)
        if not v.parent:
            return
        self.add_node(v, robots_plan=robots_plan)
        for u in V_near:
            if not self.utils.is_collision(u, v) and self.is_feasible_ray(u, v):
                v.N_o_plus.add(u)
                v.N_o_minus.add(u)
                u.N_r_plus.add(v)
                u.N_r_minus.add(v)

    def update_obstacles(self, obs_cir, robots_plan=False, print_time=False):
        exec_time_start = time.time()
        self.add_new_obstacle(obs_cir)
        if print_time:
            print("Added new obstacles:", time.time() - exec_time_start, "s")

        exec_time_start = time.time()
        self.propagate_descendants(robots_plan=robots_plan)
        if print_time:
            print("Propagate Descendants:", time.time() - exec_time_start, "s")

        exec_time_start = time.time()
        if robots_plan:
            self.verify_queue(self.s_bot)
        for g in self.other_goals:
            self.verify_queue(g)
        if print_time:
            print("Verify Queue:", time.time() - exec_time_start, "s")

        exec_time_start = time.time()
        self.reduce_inconsistency(robots_plan=robots_plan)
        if print_time:
            print("Reduce Inconsistency:", time.time() - exec_time_start, "s")

    def add_new_obstacle(self, obs):
        self.obs_circle.append(obs)
        self.plotting.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle, self.unknown_obs_circle)
        self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle, self.unknown_obs_circle)
        self.update_gamma()

        E_O = [(v, u)
               for v in self.kd_tree.search_nn_dist((obs[0], obs[1]), obs[2] + self.search_radius)
               for u in v.all_out_neighbors()
               if self.utils.is_intersect_circle(*self.utils.get_ray(v, u), obs[:2], obs[2]) or not self.is_feasible_ray(v, u)]

        print("Invalidated nodes", len(E_O))
        self.invalid_nodes = E_O

        for v, u in E_O:
            v.infinite_dist_nodes.add(u)
            u.infinite_dist_nodes.add(v)
            if v.parent and v.parent == u:
                self.verify_orphan(v)

        heapq.heapify(self.Q)

    def verify_orphan(self, v):
        key = self.node_in_queue(v)
        if key is not None:
            self.Q.remove((key, v))
        self.orphan_nodes.add(v)

    def propagate_descendants(self, robots_plan=False):
        if not self.orphan_nodes:
            return
        orphan_queue = deque(list(self.orphan_nodes))
        while orphan_queue:
            node = orphan_queue.pop()
            for child in node.children:
                orphan_queue.append(child)
                self.orphan_nodes.add(child)

        for v in self.orphan_nodes:
            for u in (v.all_out_neighbors().union(set([v.parent]))) - self.orphan_nodes:
                u.cost_to_goal = np.inf
                self.verify_queue(u)
        heapq.heapify(self.Q)

        for v in self.orphan_nodes:
            v.cost_to_goal = np.inf
            v.lmc = np.inf
            if v.parent:
                v.infinite_dist_nodes.add(v.parent)
                v.parent.infinite_dist_nodes.add(v)
                v.parent.children.remove(v)
                v.parent = None
            try:
                self.tree_nodes.remove(v)
                self.kd_tree.remove(v)
            except ValueError:
                pass

        if robots_plan:
            if self.s_bot in self.orphan_nodes or np.isinf(self.s_bot.cost_to_goal):
                print('robot node got orphaned')
                self.robot_path_to_goal = False
                self.path = []

        for j, goal_j in enumerate(self.other_goals):
            if goal_j in self.orphan_nodes or np.isinf(goal_j.cost_to_goal):
                self.path_to_goal[j] = False
                self.multi_paths[j] = []

        self.orphan_nodes = set([])

    def verify_queue(self, v):
        key = self.node_in_queue(v)
        if key is not None:
            self.Q.remove((key, v))
        heapq.heappush(self.Q, (v.get_key(), v))

    def reduce_inconsistency(self, robots_plan=False):
        while len(self.Q) > 0 and any(
            self.Q[0][0] < v.get_key()
            or v.lmc != v.cost_to_goal
            or v in {node for _, node in self.Q}
            for v in (self.other_goals + ([self.s_bot] if robots_plan else []))
        ):
            try:
                v = heapq.heappop(self.Q)[1]
            except TypeError:
                print('something went wrong with the queue')

            if v.cost_to_goal - v.lmc > self.epsilon:
                v.update_LMC(self.orphan_nodes, self.search_radius, self.epsilon, self.utils)
                self.rewire_neighbours(v, robots_plan=robots_plan)

            v.cost_to_goal = v.lmc

    def add_node(self, node_new, robots_plan=False):
        self.all_nodes_coor.append(np.array([node_new.x, node_new.y]))
        self.tree_nodes.append(node_new)
        self.kd_tree.add(node_new)

        if robots_plan:
            if node_new == self.s_bot:
                self.s_bot = node_new
                self.robot_path_to_goal = True
                self.update_path(self.s_bot)
                return

        for j in range(len(self.other_goals)):
            if node_new == self.other_goals[j]:
                self.other_goals[j] = node_new
                self.path_to_goal[j] = True
                self.update_multi_paths(node_new, j)

    def saturate(self, v_nearest, v):
        dist, theta = self.get_distance_and_angle(v_nearest, v)
        dist = min(self.step_len, dist)
        node_new = Node((v_nearest.x + dist * math.cos(theta),
                         v_nearest.y + dist * math.sin(theta)))
        return node_new

    def find_parent(self, v, U):
        costs = [math.sqrt((v.x - u.x)**2 + (v.y - u.y)**2) + u.lmc for u in U]
        if not costs:
            return
        min_idx  = int(np.argmin(costs))
        best_u   = U[min_idx]
        if not self.utils.is_collision(best_u, v):
            v.set_parent(best_u)
            v.lmc = costs[min_idx] + best_u.lmc
        else:
            del U[min_idx]
            self.find_parent(v, U)

    def rewire_neighbours(self, v, robots_plan=False):
        if v.cost_to_goal - v.lmc > self.epsilon:
            v.cull_neighbors(self.search_radius)
            for u in v.all_in_neighbors() - set([v.parent]):
                if u.lmc > v.distance(u) + v.lmc and \
                        not self.utils.is_collision(u, v) and self.is_feasible_ray(u, v):
                    u.lmc = v.distance(u) + v.lmc
                    u.set_parent(v)
                    if u.cost_to_goal - u.lmc > self.epsilon:
                        self.verify_queue(u)

        if robots_plan:
            self.update_path(self.s_bot)

        for j, goal_j in enumerate(self.other_goals):
            self.update_multi_paths(goal_j, j)

    def random_node(self, robots_plan=False):
        delta = self.utils.delta

        if self._pending_reset_target is not None:
            if np.random.random() < 0.5:
                return Node((self._pending_reset_target.x, self._pending_reset_target.y))

        if (robots_plan) and (not self.robot_path_to_goal) and (np.random.random() < self.bot_sample_rate):
            return Node((self.s_bot.x, self.s_bot.y))

        if not self.HJ_contingency_enable:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))
        else:
            idx    = np.random.randint(len(self.safe_regions))
            x, y, _ = self.safe_regions[idx]
            mu     = np.array([x, y])
            sigma  = 5.5
            cov    = sigma**2 * np.eye(2)
            sample = np.random.multivariate_normal(mu, cov)
            return Node((sample[0], sample[1]))

    def _try_connect_pending(self):
        target   = Node((self.robot_position[0], self.robot_position[1]))
        nearest  = self.nearest(target)
        saturated = self.saturate(nearest, target)

        V_near = self.near(saturated)
        if not V_near:
            V_near = [nearest]

        V_near_free = [u for u in V_near if not self.utils.is_collision(u, saturated)]
        if not V_near_free:
            return

        self.find_parent(saturated, V_near_free)
        if saturated.parent is None:
            return

        self.add_node(saturated)

        for u in V_near_free:
            saturated.N_o_plus.add(u)
            saturated.N_o_minus.add(u)
            u.N_r_plus.add(saturated)
            u.N_r_minus.add(saturated)

        self.rewire_neighbours(saturated)
        self.reduce_inconsistency()

        self.s_bot = saturated
        self._pending_reset_target = None

        heading = self._pending_reset_heading
        self.robot_state = [self.s_bot.x, self.s_bot.y,
                            heading if heading is not None else self.robot_state[2]]
        self.robot_position = [self.s_bot.x, self.s_bot.y]
        self.robot_path_to_goal = self.s_bot.lmc < np.inf
        self.update_path(self.s_bot)

        print(f"[_try_connect_pending] Connected at "
              f"({self.s_bot.x:.2f}, {self.s_bot.y:.2f}), lmc={self.s_bot.lmc:.3f}")

    def update_gamma(self):
        mu_X_free = (self.x_range[1] - self.x_range[0]) * (self.y_range[1] - self.y_range[0]) * 1/2
        for (_, _, r) in self.obs_circle:
            mu_X_free -= np.pi * r ** 2
        self.gamma = self.gamma_FOS * (2 * (1 + 1/self.d))**(1/self.d) * (mu_X_free/self.zeta_d)**(1/self.d)

    def shrinking_ball_radius(self):
        return min(self.step_len, self.gamma * np.log(len(self.tree_nodes)+1) / len(self.tree_nodes))

    def near(self, v):
        return self.kd_tree.search_nn_dist((v.x, v.y), self.search_radius)

    def nearest(self, v):
        return self.kd_tree.search_nn((v.x, v.y))[0].data

    def update_path(self, node):
        self.path = []
        self.path_node = []
        while node.parent:
            self.path_node.append(node)
            self.path.append(np.array([[node.x, node.y], [node.parent.x, node.parent.y]]))
            node = node.parent

    def update_multi_paths(self, node, idx):
        self.multi_paths[idx] = []
        while node.parent:
            self.multi_paths[idx].append(np.array([[node.x, node.y], [node.parent.x, node.parent.y]]))
            node = node.parent

    def node_in_queue(self, node):
        if not self.Q:
            return None
        keys, nodes = list(zip(*self.Q))
        try:
            idx = nodes.index(node)
            return keys[idx]
        except ValueError:
            return None

    def is_feasible_ray(self, start: Node, end: Node):
        o, d = self.utils.get_ray(start, end)
        t_vals  = np.linspace(0, 1, 4)
        positions = o + t_vals[:, None] * d
        return self.hjr_fno.is_feasible(v=positions, reachable_set_constraint=self.HJ_contingency_enable)

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    @staticmethod
    def get_distance(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy)


class SFF_star:

    def __init__(
        self,
        start_goal_index: int,
        x_goal: List,
        heading: float,
        lidar_range: float,
        step_len: float,
        gamma_FOS: float,
        epsilon: float,
        bot_sample_rate: float,
        iter_max: int,
        safe_regions: List[Sequence[float]],
        HJ_contingency_enable: bool,
        # ── Webots callbacks (all optional; None = standalone mode) ─────────
        # lidar_fn(robot_position, unknown_obs_circle, lidar_range)
        #          -> (remaining_unknown_obs, detected_obs)  each as [[x,y,r],...]
        lidar_fn: Optional[Callable] = None,
        # pose_fn() -> [x, y, theta]
        pose_fn: Optional[Callable]  = None,
        # drive_fn(wx, wy) -> reached:bool
        drive_fn: Optional[Callable] = None,
    ) -> None:

        from HJR_FNO.HJR_FNO import HJR_FNO

        assert start_goal_index < len(x_goal), "start_goal_index index out of range"

        self.HJ_contingency_enable = HJ_contingency_enable
        self.robot_is_isolated     = False

        self.start_goal_index = start_goal_index
        x_start   = x_goal[start_goal_index]
        self.iter_max = iter_max

        self.env      = env.Env(safe_regions=safe_regions)
        self.plotting = plotting.Plotting(x_start, x_goal, safe_regions=safe_regions)

        self.Tf_reach  = 8
        self.hjr_fno   = HJR_FNO(env=self.env, safe_regions=safe_regions, Tf_reach=self.Tf_reach)
        self.current_state  = [x_start[0], x_start[1], heading]
        self.lidar_range    = lidar_range

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.suptitle(f"HJR-FNO Contingency")
        self.ax.set_xlim(self.env.x_range[0], self.env.x_range[1]+1)
        self.ax.set_ylim(self.env.y_range[0], self.env.y_range[1]+1)

        self.show_subplots = False

        self.waiting_for_first_click  = False
        self.waiting_for_second_click = False
        self.contingency_complete     = False
        self.resume_planning          = False

        # ── Store Webots callbacks to pass down to each tree ────────────────
        self._lidar_fn = lidar_fn
        self._pose_fn  = pose_fn
        self._drive_fn = drive_fn

        self.rrtx_trees = {}
        self.n_tree     = len(x_goal)
        q, r = divmod(self.iter_max, self.n_tree)
        iter_list = [q + 1 if i < r else q for i in range(self.n_tree)]

        self.sub_iter_count = 1000
        for i, target_i in enumerate(x_goal):
            other_goals    = [g for j, g in enumerate(x_goal) if j != i]
            other_goals_id = [j for j in range(len(x_goal)) if j != i]

            self.rrtx_trees[i] = RRTX(
                x_start=x_start,
                x_goal=target_i,
                other_goals=other_goals,
                other_goals_id=other_goals_id,
                heading=heading,
                lidar_range=lidar_range,
                step_len=step_len,
                gamma_FOS=gamma_FOS,
                epsilon=epsilon,
                bot_sample_rate=bot_sample_rate,
                iter_max=iter_list[i],
                safe_regions=safe_regions,
                hjr_fno=self.hjr_fno,
                HJ_contingency_enable=self.HJ_contingency_enable,
                fig=self.fig,
                ax=self.ax,
                plotting=self.plotting,
                # pass Webots callbacks into every tree
                lidar_fn=lidar_fn,
                pose_fn=pose_fn,
                drive_fn=drive_fn,
            )

        self.robotState_isSync = [False for _ in range(self.n_tree)]

        self.D = np.full((self.n_tree, self.n_tree), np.inf)
        np.fill_diagonal(self.D, 0.0)

        cmap = plt.get_cmap("hsv")
        self.colorList = [cmap(i) for i in np.linspace(0, 1, len(x_goal), endpoint=False)]

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if self.HJ_contingency_enable:
            print("Adversary detected!")
            for tree_k in self.rrtx_trees.values():
                tree_k.contingency_triggered = True
        else:
            print("Contingency constraint disabled. No action taken.")

    def update_distance_matrix(self, sequence_visited, robot_position_costs: dict = None):
        visited_set  = set(sequence_visited)
        unvisited    = [tid for tid in self.rrtx_trees.keys() if tid not in visited_set]
        include_robot = robot_position_costs is not None
        nodes        = (["robot"] + unvisited) if include_robot else unvisited
        m            = len(nodes)
        idx_map      = {node: k for k, node in enumerate(nodes)}
        D_pat        = np.full((m, m), np.inf)

        for a in nodes:
            for b in nodes:
                ia, ib = idx_map[a], idx_map[b]
                if a == b:
                    D_pat[ia, ib] = 0.0
                    continue
                if a == "robot":
                    if b in robot_position_costs:
                        D_pat[ia, ib] = robot_position_costs[b]
                    continue
                if b == "robot":
                    D_pat[ia, ib] = np.inf
                    continue
                tree_b = self.rrtx_trees[b]
                if a in tree_b.other_goals_id:
                    local_j = tree_b.other_goals_id.index(a)
                    cost    = tree_b.other_goals[local_j].cost_to_goal
                    D_pat[ia, ib] = cost

        self.D         = D_pat
        self.D_idx_map = idx_map
        self.D_nodes   = nodes

    def print_distance_matrix(self, precision=3):
        labels = ["robot" if node == "robot" else f"goal_{node}" for node in self.D_nodes]
        n = len(labels)
        header = "          " + "".join(f"{lbl:^12}" for lbl in labels)
        print(header)
        print("          " + "-" * (12 * n))
        for i, row_lbl in enumerate(labels):
            row_str = f"{row_lbl:>8} |"
            for j in range(n):
                val = self.D[i, j]
                row_str += f"{'∞':^12}" if np.isinf(val) else f"{val:^12.{precision}f}"
            print(row_str)

    def solve_atsp_held_karp(self, distance_matrix, start=[0], hamiltonian_cycle=True):
        if type(start) == int:
            start = [start]
        assert type(start) == list
        return atsp.held_karp(cost=distance_matrix.tolist(), prefix=start, hamiltonian_cycle=hamiltonian_cycle)

    def compute_tour_distance(self, tour, prev_id=None, curr_id=None, traversed_distance=0.0):
        total_cost = 0.0
        for i, j in zip(tour[:-1], tour[1:]):
            if i == j:
                continue
            if prev_id is not None and curr_id is not None:
                if i == prev_id and j == curr_id:
                    remaining = self.rrtx_trees[j].s_bot.cost_to_goal
                    if np.isinf(remaining):
                        return np.inf
                    total_cost += traversed_distance + remaining
                    continue
            tree_i = self.rrtx_trees[i]
            if j not in tree_i.other_goals_id:
                return np.inf
            k = tree_i.other_goals_id.index(j)
            d = tree_i.other_goals[k].cost_to_goal
            if np.isinf(d):
                return np.inf
            total_cost += d
        return total_cost

    def rotate_tour(self, tour, start_id):
        if start_id not in tour:
            raise ValueError(f"Tour does not contain start_id={start_id}")
        if tour[0] == tour[-1]:
            tour = tour[:-1]
        idx     = tour.index(start_id)
        rotated = tour[idx:] + tour[:idx] + [start_id]
        return rotated

    def init_trees(self, showPlot=True):
        for i in range(self.n_tree):
            print(f"Initialize the tree {i}")
            self.rrtx_trees[i].planning()

            if showPlot:
                self.ax.clear()
                self.ax.set_xlim(self.env.x_range[0], self.env.x_range[1] + 1)
                self.ax.set_ylim(self.env.y_range[0], self.env.y_range[1] + 1)
                self.plotting.plot_env(self.ax, colorList=self.colorList)

                for i, tree_i in self.rrtx_trees.items():
                    if tree_i.all_nodes_coor:
                        nodes = np.array(tree_i.all_nodes_coor)
                        self.ax.scatter(nodes[:, 0], nodes[:, 1], s=4, c='gray', alpha=0.5)
                    self.edges = []
                    for node in tree_i.tree_nodes:
                        if node.parent:
                            self.edges.append(np.array([[node.parent.x, node.parent.y], [node.x, node.y]]))
                    if self.edges:
                        edge_col = LineCollection(self.edges, colors=self.colorList[i], linewidths=0.5, alpha=0.2)
                        self.ax.add_collection(edge_col)
                    for j, goal_j in enumerate(tree_i.other_goals):
                        if tree_i.path_to_goal[j]:
                            path_col = LineCollection(tree_i.multi_paths[j], colors=self.colorList[i], linewidths=2.5)
                            self.ax.add_collection(path_col)
                    if self.HJ_contingency_enable:
                        self.plotting.plot_reachable_set(self.ax, self.hjr_fno, theta=tree_i.robot_state[2], time=tree_i.Tf_reach)
                    plt.pause(0.001)

    def planning(self, hamiltonian_cycle=False, showPlot=True):

        heading = 0.0
        prev_heading = heading
        self.current_state = [self.rrtx_trees[self.start_goal_index].s_goal.x,
                               self.rrtx_trees[self.start_goal_index].s_goal.y, heading]

        print("\nInitilize robot position for each tree")
        for i in range(self.n_tree):
            connected = self.rrtx_trees[i].reset_robot_position(
                (self.current_state[0], self.current_state[1]), heading=None)
            self.rrtx_trees[i].update_robot_heading()
            self.robotState_isSync[i] = connected
            self.rrtx_trees[i].prob_q = 0.9

        sequence_visited = [self.start_goal_index]
        sequence_to_visit = []

        robot_position_costs = {
            tid: self.rrtx_trees[tid].s_bot.cost_to_goal
            for tid in range(self.n_tree)
            if tid not in set(sequence_visited)
        }
        print(f"robot_position_costs: {robot_position_costs}")

        if hamiltonian_cycle:
            self.update_distance_matrix(sequence_visited=[], robot_position_costs=None)
            start_idx = self.D_idx_map[self.start_goal_index]
            min_cost, tour_indices = atsp.held_karp(self.D.tolist(), prefix=[start_idx],
                                                     hamiltonian_cycle=hamiltonian_cycle)
            optimal_tour = [self.D_nodes[i] for i in tour_indices[:-1]]
        else:
            self.update_distance_matrix(sequence_visited=sequence_visited,
                                         robot_position_costs=robot_position_costs)
            min_cost, tour_indices = atsp.held_karp(self.D.tolist(), prefix=[0],
                                                     hamiltonian_cycle=hamiltonian_cycle)
            optimal_tour = [self.D_nodes[i] for i in tour_indices if self.D_nodes[i] != "robot"]

        print(f"Optimal tour: {[self.start_goal_index] + optimal_tour}")

        prev_id       = self.start_goal_index
        original_tour = ([self.start_goal_index] + optimal_tour.copy()
                         if not hamiltonian_cycle else optimal_tour.copy())
        optimal_tour  = original_tour
        sequence_to_visit = optimal_tour.copy()
        sequence_visited  = [self.start_goal_index]

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.suptitle(f"HJR-FNO Contingency\nOptimal Tour: {optimal_tour}\n"
                          f"Visited: {sequence_visited}\nTo Visit: {sequence_to_visit}")
        prev_plotting = time.time()

        print("Start Robot's Plan Execution")
        state_history = []

        for i in range(1, len(optimal_tour)):

            traversed_distance = 0.0
            needs_reset = True

            for plan_iter in range(self.iter_max):

                id = optimal_tour[i]

                if needs_reset:
                    connected = self.rrtx_trees[id].reset_robot_position(
                        (self.current_state[0], self.current_state[1]), heading=None)
                    self.rrtx_trees[id].update_robot_heading()
                    self.robotState_isSync[id] = connected
                    needs_reset = False

                    new_heading    = self.rrtx_trees[id].robot_state[2]
                    heading_change = abs(utils.Utils.wrap_angle(new_heading - prev_heading))
                    heading_change_deg = math.degrees(heading_change)

                    path_infeasible = False
                    if self.rrtx_trees[id].path:
                        all_points = []
                        t_vals = np.linspace(0, 1, 3)
                        for seg in self.rrtx_trees[id].path:
                            orig  = seg[0]
                            direc = seg[1] - seg[0]
                            points = orig + t_vals[:, None] * direc
                            all_points.append(points)
                        waypoints = np.unique(np.vstack(all_points), axis=0)
                        path_infeasible = not self.hjr_fno.is_feasible(
                            v=waypoints, reachable_set_constraint=self.HJ_contingency_enable)
                        if path_infeasible:
                            print(f"[Path check] New path for tree {id} is infeasible.")

                    HEADING_THRESHOLD_DEG = 120.0
                    if heading_change_deg > HEADING_THRESHOLD_DEG or path_infeasible:
                        print(f"[Heading check] Large heading change: "
                              f"{heading_change_deg:.1f} deg. Orphaning s_bot of tree {id}.")
                        if self.rrtx_trees[id].s_bot is not None \
                        and self.rrtx_trees[id].s_bot.parent is not None:
                            self.rrtx_trees[id].verify_orphan(self.rrtx_trees[id].s_bot)
                            self.rrtx_trees[id].propagate_descendants(robots_plan=True)
                            self.rrtx_trees[id].robot_path_to_goal = False
                            self.rrtx_trees[id].path = []
                            self.rrtx_trees[id]._pending_reset_target = Node(
                                (self.current_state[0], self.current_state[1]))
                            self.rrtx_trees[id]._pending_reset_heading = self.current_state[2]
                            self.robotState_isSync[id] = False

                    visited_set = set(sequence_visited)
                    for k, tree_k in self.rrtx_trees.items():
                        if k == id or k in visited_set \
                        or (k == self.start_goal_index and not hamiltonian_cycle):
                            continue
                        connected_k = tree_k.reset_robot_position(
                            (self.current_state[0], self.current_state[1]), heading=None)
                        tree_k.update_robot_heading()
                        self.robotState_isSync[k] = connected_k
                        if not connected_k:
                            if tree_k.s_bot is not None and tree_k.s_bot.parent is not None:
                                tree_k.verify_orphan(tree_k.s_bot)
                                tree_k.propagate_descendants(robots_plan=True)
                            tree_k.robot_position = [self.current_state[0], self.current_state[1]]
                            tree_k.robot_state    = [self.current_state[0], self.current_state[1],
                                                     tree_k.robot_state[2]]
                            tree_k.robot_path_to_goal = False
                            tree_k.path = []
                            tree_k._pending_reset_target = Node(
                                (self.current_state[0], self.current_state[1]))
                            tree_k._pending_reset_heading = tree_k.robot_state[2]

                    print(f"\n{'='*50}")
                    print(f"Starting planning from Target #{prev_id} to Target #{id}")
                    print(f"{'='*50}\n")

                new_obs, new_obs_flag, distance_moved = self.rrtx_trees[id].planning_with_robot(steps=10)
                traversed_distance += distance_moved
                self.current_state = self.rrtx_trees[id].robot_state
                state_history.append(self.current_state.copy() + [id])

                if not self.rrtx_trees[id].robot_path_to_goal and (plan_iter % 3 == 0):
                    print("robot's position", (self.rrtx_trees[id].s_bot.x, self.rrtx_trees[id].s_bot.y))
                    print("is feasible?", self.hjr_fno.is_feasible(v=np.atleast_2d(self.current_state[:2])))
                    print("robot's Path to goal", self.rrtx_trees[id].robot_path_to_goal)
                    print("Robot's cost to goal", self.rrtx_trees[id].s_bot.cost_to_goal)
                    print("Robot's LMC cost", self.rrtx_trees[id].s_bot.lmc)
                    print("Path List", self.rrtx_trees[id].path)
                    print("Pending Target", self.rrtx_trees[id]._pending_reset_target)

                if self.rrtx_trees[id].contingency_triggered and self.HJ_contingency_enable:
                    detected_obs_during_contingency, contingency_trajectory, _, _, _, _, _ = \
                        self.hjr_fno.contingency_policy(self.current_state, self.plotting, self.fig, self.ax)
                    state_history.extend(contingency_trajectory.tolist() + [id])
                    self.current_state = contingency_trajectory[-1]
                    print("Position after contingency", self.current_state)
                    for traj_i in range(len(contingency_trajectory) - 1):
                        x0, y0, _ = contingency_trajectory[traj_i]
                        x1, y1, _ = contingency_trajectory[traj_i + 1]
                        traversed_distance += np.hypot(x1 - x0, y1 - y0)
                    if len(detected_obs_during_contingency) > 0:
                        new_obs += detected_obs_during_contingency
                        new_obs_flag = True
                        for obs in detected_obs_during_contingency:
                            self.rrtx_trees[id].update_obstacles(obs, robots_plan=True)
                    if len(contingency_trajectory) > 1:
                        connected = self.rrtx_trees[id].reset_robot_position(
                            (self.current_state[0], self.current_state[1]), heading=None)
                        self.rrtx_trees[id].update_robot_heading()
                        self.robotState_isSync[id] = connected
                    for tree_k in self.rrtx_trees.values():
                        tree_k.contingency_triggered = False
                    visited_set = set(sequence_visited)
                    self.robot_is_isolated = all(
                        np.isinf(self.rrtx_trees[tid].s_bot.cost_to_goal)
                        for tid in range(self.n_tree) if tid not in visited_set)
                    if self.robot_is_isolated:
                        print("\n[INFO] Planning for all unvisited trees until paths are restored...")
                        unvisited_set  = set(sequence_to_visit)
                        recovery_iter  = 0
                        while not all(self.rrtx_trees[tid].robot_path_to_goal for tid in unvisited_set):
                            for tid in unvisited_set:
                                if not self.rrtx_trees[tid].robot_path_to_goal:
                                    self.rrtx_trees[tid].planning(iter_max=100, robots_plan=True)
                                    self.rrtx_trees[tid].robot_path_to_goal = (
                                        self.rrtx_trees[tid].s_bot.cost_to_goal < np.inf)
                            recovery_iter += 1
                            if recovery_iter % 10 == 0:
                                print(f"[Recovery iter {recovery_iter}]")
                                for tid in unvisited_set:
                                    print(f"  Tree {tid}: {self.rrtx_trees[tid].robot_path_to_goal}, "
                                          f"cost={self.rrtx_trees[tid].s_bot.cost_to_goal:.3f}")
                            if recovery_iter >= 100:
                                print("[WARNING] Recovery limit reached.")
                                break
                        recovered       = [tid for tid in unvisited_set if self.rrtx_trees[tid].robot_path_to_goal]
                        still_isolated  = [tid for tid in unvisited_set if not self.rrtx_trees[tid].robot_path_to_goal]
                        print(f"\n[Recovery] Recovered: {recovered}  Still isolated: {still_isolated}")
                        if recovered:
                            self.robot_is_isolated = False
                            needs_reset = True

                for k, tree_k in self.rrtx_trees.items():
                    if k == id \
                    or (len(sequence_visited) > 1 and k in sequence_visited[1:]) \
                    or (k == self.start_goal_index and not hamiltonian_cycle):
                        continue
                    connected = tree_k.reset_robot_position(
                        (self.current_state[0], self.current_state[1]), heading=None)
                    tree_k.update_robot_heading()
                    self.robotState_isSync[k] = connected

                if new_obs_flag:
                    updateObs_time_start = time.time()
                    for k, tree_k in self.rrtx_trees.items():
                        if k == id \
                        or (len(sequence_visited) > 1 and k in sequence_visited[1:]) \
                        or (k == self.start_goal_index and not hamiltonian_cycle):
                            continue
                        for obs in new_obs:
                            tree_k.update_obstacles(obs, robots_plan=True)
                        if tree_k.s_bot is not None and tree_k.s_bot.cost_to_goal < np.inf:
                            tree_k.update_path(tree_k.s_bot)
                        else:
                            tree_k.path = []
                        tree_k.robot_path_to_goal = tree_k.s_bot.cost_to_goal < np.inf
                        for j, goal_j in enumerate(tree_k.other_goals):
                            if goal_j.cost_to_goal < np.inf:
                                tree_k.update_multi_paths(goal_j, j)
                            else:
                                tree_k.multi_paths[j] = []
                            tree_k.path_to_goal[j] = goal_j.cost_to_goal < np.inf
                    print(f"\nRewire other trees: {time.time() - updateObs_time_start} s\n")

                    remaining_count = len(sequence_to_visit)
                    should_replan = (not self.rrtx_trees[id].robot_path_to_goal) or \
                                    (hamiltonian_cycle and remaining_count >= 2) or \
                                    (not hamiltonian_cycle and remaining_count >= 2)

                    if should_replan:
                        visited_set = set(sequence_visited)
                        robot_position_costs = {
                            tid: self.rrtx_trees[tid].s_bot.cost_to_goal
                            for tid in range(self.n_tree) if tid not in visited_set
                        }
                        all_paths_broken = all(np.isinf(c) for c in robot_position_costs.values())
                        if all_paths_broken:
                            print("\n[WARNING] Robot is isolated.")
                            self.robot_is_isolated = True
                            for tid in range(self.n_tree):
                                if tid not in visited_set:
                                    self.rrtx_trees[tid].contingency_triggered = True
                        else:
                            self.robot_is_isolated = False
                            self.update_distance_matrix(sequence_visited=sequence_visited,
                                                         robot_position_costs=robot_position_costs)
                            self.print_distance_matrix(precision=3)
                            min_cost, tour_indices = atsp.held_karp(self.D.tolist(), prefix=[0],
                                                                     hamiltonian_cycle=hamiltonian_cycle)
                            if tour_indices:
                                new_tour_remaining = [self.D_nodes[i] for i in tour_indices
                                                      if self.D_nodes[i] != "robot"]
                                if hamiltonian_cycle:
                                    new_tour_remaining.append(self.start_goal_index)
                                new_tour = sequence_visited + new_tour_remaining
                                if new_tour_remaining != sequence_to_visit:
                                    print("\nNew optimal tour found!")
                                    optimal_tour      = new_tour
                                    sequence_to_visit = new_tour_remaining
                                    new_next_id       = sequence_to_visit[0] if sequence_to_visit else None
                                    if new_next_id != id:
                                        print(f"Next target changed: {id} → {new_next_id}")
                                        needs_reset = True
                                        traversed_distance = 0.0
                                    else:
                                        print(f"Next target unchanged ({id}), no reset needed.")
                                else:
                                    print("Tour unchanged after replanning.")
                            else:
                                print("Held-Karp failed.")
                                self.robot_is_isolated = True
                                for tid in range(self.n_tree):
                                    if tid not in visited_set:
                                        self.rrtx_trees[tid].contingency_triggered = True
                    else:
                        visited_set = set(sequence_visited)
                        robot_position_costs = {
                            tid: self.rrtx_trees[tid].s_bot.cost_to_goal
                            for tid in range(self.n_tree) if tid not in visited_set
                        }
                        self.update_distance_matrix(sequence_visited=sequence_visited,
                                                     robot_position_costs=robot_position_costs)
                        self.print_distance_matrix(precision=3)

                    print(f"\nNew Optimal Tour: {sequence_visited + sequence_to_visit}")
                    print(f"Targets visited:   {sequence_visited}")
                    print(f"Remaining to visit: {sequence_to_visit}")
                    print(f"\nOriginal Tour: {original_tour}")

                elapsed_plotting = time.time() - prev_plotting
                if elapsed_plotting >= 0.2 and showPlot:
                    prev_plotting = time.time()
                    self.ax.clear()
                    self.fig.suptitle(f"HJR-FNO Contingency\nOptimal Tour: {optimal_tour}\n"
                                      f"Visited: {sequence_visited}\nTo Visit: {sequence_to_visit}")
                    self.ax.set_xlim(self.env.x_range[0], self.env.x_range[1] + 1)
                    self.ax.set_ylim(self.env.y_range[0], self.env.y_range[1] + 1)
                    self.plotting.plot_env(self.ax, colorList=self.colorList)
                    if self.rrtx_trees[id].all_nodes_coor:
                        nodes = np.array(self.rrtx_trees[id].all_nodes_coor)
                        self.ax.scatter(nodes[:, 0], nodes[:, 1], s=4, c='gray', alpha=0.5)
                    self.edges = []
                    for node in self.rrtx_trees[id].tree_nodes:
                        if node.parent:
                            self.edges.append(np.array([[node.parent.x, node.parent.y], [node.x, node.y]]))
                    if self.edges:
                        edge_col = LineCollection(self.edges, colors='blue', linewidths=0.3, alpha=0.45)
                        self.ax.add_collection(edge_col)
                    for j, goal_j in enumerate(self.rrtx_trees[id].other_goals):
                        if self.rrtx_trees[id].path_to_goal[j]:
                            path_col = LineCollection(self.rrtx_trees[id].multi_paths[j],
                                                      colors=self.colorList[id], linewidths=2.5, alpha=0.7)
                            self.ax.add_collection(path_col)
                    if self.rrtx_trees[id].path:
                        path_col = LineCollection(self.rrtx_trees[id].path, colors='black', linewidths=1.5)
                        self.ax.add_collection(path_col)
                    self.plotting.plot_robot(self.ax, self.rrtx_trees[id].robot_position,
                                             self.rrtx_trees[id].lidar_range)
                    if self.HJ_contingency_enable:
                        self.plotting.plot_reachable_set(self.ax, self.hjr_fno,
                                                          self.rrtx_trees[id].robot_state[2],
                                                          self.rrtx_trees[id].Tf_reach)
                    plt.pause(0.001)

                if self.rrtx_trees[id].s_bot.cost_to_goal == 0.0 and self.rrtx_trees[id].s_bot.lmc == 0.0:
                    print("Successfully reach the goal!")
                    self.current_state = self.rrtx_trees[id].robot_state
                    break

                prev_heading = self.current_state[2]

            prev_id = id
            self.current_state = self.rrtx_trees[id].robot_state
            heading = self.rrtx_trees[id].robot_state[2]
            sequence_visited.append(sequence_to_visit.pop(0))

        print('\nFinal Tour (target_id): ', optimal_tour)
        print('Final Tour cost', self.compute_tour_distance(optimal_tour))
        print('\nOriginal Tour (target_id): ', original_tour)
        print('Original Tour cost', self.compute_tour_distance(original_tour))
        print("Tour Completed!")

        for i in range(len(self.hjr_fno.safe_regions)):
            obs = np.array(self.hjr_fno.obs_list[i])
            xs, ys = self.hjr_fno.safe_regions[i][:2]
            obs_local = obs.copy()
            obs_local[:, 0] -= xs
            obs_local[:, 1] -= ys
            print(obs_local.tolist())

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(self.env.x_range[0], self.env.x_range[1] + 1)
        self.ax.set_ylim(self.env.y_range[0], self.env.y_range[1] + 1)
        self.plotting.plot_env(self.ax, colorList=None)
        data     = np.vstack(state_history)
        x_traj   = data[:, 0]
        y_traj   = data[:, 1]
        goal_ids = data[-1, 3].astype(int)
        self.ax.scatter(self.plotting.xG[goal_ids][0], self.plotting.xG[goal_ids][1],
                        marker='*', s=300, c='red', edgecolors='black', linewidths=1.5, zorder=10)
        for i in range(len(x_traj) - 1):
            self.ax.plot([x_traj[i], x_traj[i+1]], [y_traj[i], y_traj[i+1]],
                         color=self.colorList[goal_ids[i]], linewidth=2)
        if self.HJ_contingency_enable:
            self.plotting.plot_reachable_set(self.ax, self.hjr_fno,
                                              self.rrtx_trees[id].robot_state[2],
                                              self.rrtx_trees[id].Tf_reach)
        self.ax.scatter(x_traj[0], y_traj[0], color='red', s=60, zorder=5)
        self.ax.scatter(x_traj[-1], y_traj[-1], color='red', s=60, zorder=5)
        plt.show()

        return data

    def generate_random_obstacles(self, env, N, r_min=1, r_max=1.5,
                                   min_dist_between=3.0, goals=None, min_dist_to_goal=3.0,
                                   origin_safe_radius=3.0, start_max_radius=7,
                                   start_min_dist_to_obs=2, max_attempts=1000):
        obs_list = []
        attempts = 0
        while len(obs_list) < N and attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(env.x_range[0] + r_max, env.x_range[1] - r_max)
            y = np.random.uniform(env.y_range[0] + r_max, env.y_range[1] - r_max)
            r = np.random.uniform(r_min, r_max)
            if math.hypot(x, y) < origin_safe_radius + r:
                continue
            too_close_to_obs = any(math.hypot(x - ox, y - oy) < min_dist_between + r + or_
                                   for ox, oy, or_ in obs_list)
            if too_close_to_obs:
                continue
            if goals is not None:
                too_close_to_goal = any(math.hypot(x - gx, y - gy) < min_dist_to_goal + r
                                        for g in goals for gx, gy in [g[:2]])
                if too_close_to_goal:
                    continue
            obs_list.append([x, y, r])
        if len(obs_list) < N:
            print(f"[generate_random_obstacles] Warning: only generated {len(obs_list)}/{N} obstacles.")
        start_state = None
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            angle  = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(5, start_max_radius)
            sx = radius * math.cos(angle)
            sy = radius * math.sin(angle)
            if env is not None:
                if not (env.x_range[0] < sx < env.x_range[1] and env.y_range[0] < sy < env.y_range[1]):
                    continue
            too_close = any(math.hypot(sx - ox, sy - oy) < or_ + start_min_dist_to_obs
                            for ox, oy, or_ in obs_list)
            if too_close:
                continue
            heading    = math.atan2(0.0 - sy, 0.0 - sx)
            start_state = [sx, sy, heading]
            print(f"[generate_random_obstacles] Start state ({sx:.2f}, {sy:.2f}, "
                  f"{math.degrees(heading):.1f} deg) found after {attempts} attempts.")
            break
        return obs_list, start_state

    def test_case_contingency_plan(self, _fig=None, _ax=None, num_obs=1, special_case=False):
        from HJR_FNO.HJR_FNO import HJR_FNO
        new_safe_region = [[0, 0, 2]]
        _env = env.Env(safe_regions=new_safe_region)
        _env.x_range = (-8, 8)
        _env.y_range = (-8, 8)
        if not special_case:
            obs_list, self.current_state = self.generate_random_obstacles(
                env=_env, N=num_obs, r_min=1, r_max=1.5,
                min_dist_between=2.0, origin_safe_radius=3.5)
            known_ratio = 0
            indices     = np.random.permutation(len(obs_list))
            n_known     = max(1, int(len(obs_list) * known_ratio))
            known_obs   = [obs_list[i] for i in indices[:n_known]]
            unknown_obs = [obs_list[i] for i in indices[n_known:]]
            print(f"Known obstacles:   {len(known_obs)}")
            print(f"Unknown obstacles: {len(unknown_obs)}")
            _env.obs_circle        = known_obs
            _env.unknown_obs_circle = unknown_obs
        else:
            self.current_state = [-7, 2, np.deg2rad(-np.pi/6)]
            known_obs = [[-7, 6, 1.2], [-6, -1, 1.2], [5, -4, 1.3], [6, 5, 1.3]]
            _env.obs_circle = known_obs
            _env.unknown_obs_circle = [[0, 4, 1.2], [-2, -3, 1.0]]
        Tf_reach  = 8
        _hjr_fno  = HJR_FNO(env=_env, safe_regions=new_safe_region, Tf_reach=Tf_reach)
        _hjr_fno.utils.sensing_radius = self.lidar_range
        if known_obs:
            _hjr_fno.update_obs(known_obs)
        _plotting = plotting.Plotting(self.current_state[:2], [[0.0, 0.0]],
                                      safe_regions=new_safe_region, _env=_env)
        if _fig is None:
            _fig, _ax = plt.subplots(figsize=(8, 8))
            _fig.suptitle(f"HJR-FNO Contingency")
            _ax.set_xlim(_env.x_range[0], _env.x_range[1]+1)
            _ax.set_ylim(_env.y_range[0], _env.y_range[1]+1)
        _, _, TReach, success, V_val, g_Val, ham_term = _hjr_fno.contingency_policy(
            self.current_state, _plotting, _fig, _ax,
            showplot=True, special_case=special_case)
        _ax.clear()
        plt.pause(0.01)
        return _fig, _ax, TReach, success, V_val, g_Val, ham_term

    def check_delta_clear_overlap(self, phi_i, phi_j, center_i, center_j,
                                   x_local, y_local, delta):
        cx_i, cy_i, _ = center_i
        cx_j, cy_j, _ = center_j
        Nx, Ny = phi_i.shape
        dx = x_local[1] - x_local[0]
        dy = y_local[1] - y_local[0]
        shift_x = int(round((cx_j - cx_i) / dx))
        shift_y = int(round((cy_j - cy_i) / dy))
        pad_x = abs(shift_x)
        pad_y = abs(shift_y)
        phi_j_padded = np.full((Nx + 2*pad_x, Ny + 2*pad_y), np.inf)
        phi_j_padded[pad_x:pad_x+Nx, pad_y:pad_y+Ny] = phi_j
        start_x = pad_x - shift_x
        start_y = pad_y - shift_y
        phi_j_aligned = phi_j_padded[start_x:start_x+Nx, start_y:start_y+Ny]
        overlap_mask  = (phi_i <= self.hjr_fno.safe_margin) & (phi_j_aligned <= self.hjr_fno.safe_margin)
        if not np.any(overlap_mask):
            return False, None
        dist    = distance_transform_edt(overlap_mask, sampling=(dx, dy))
        max_dist = np.max(dist)
        if max_dist < delta:
            return False, None
        max_idx = np.unravel_index(np.argmax(dist), dist.shape)
        x0 = x_local[max_idx[0]] + cx_i
        y0 = y_local[max_idx[1]] + cy_i
        return True, (x0, y0)


def main():
    x_goal = [(-17, 16), (-11, -14), (-10, 5), (5, 10), (14, 18), (15, -7)]
    start_goal_index = 0
    safe_region = [[-15, 19, 2], [-10, -9, 2], [-7, 13, 2],
                   [-5, 2, 2], [3, 8.5, 2], [12, 1, 2],
                   [12, 15, 2], [12, -10, 2]]

    sff = SFF_star(
        start_goal_index=start_goal_index,
        x_goal=x_goal,
        heading=0.0,
        lidar_range=5.5,
        step_len=3.0,
        gamma_FOS=20.0,
        epsilon=0.05,
        bot_sample_rate=0.10,
        iter_max=18000,
        safe_regions=safe_region,
        HJ_contingency_enable=True,
        # lidar_fn, pose_fn, drive_fn all default to None → standalone mode
    )

    showPlot = True
    sff.init_trees(showPlot=showPlot)

    plan_starttime = time.time()
    state_history  = sff.planning(hamiltonian_cycle=False, showPlot=showPlot)
    plan_elapsedtime = time.time() - plan_starttime

    state_history = np.vstack(state_history)
    xy   = state_history[:, :2]
    diff = np.diff(xy, axis=0)
    total_distance = np.sum(np.linalg.norm(diff, axis=1))
    print("Total XY distance:", total_distance)
    print("Elapsed Time", plan_elapsedtime)

    output_dir  = "/home/kmuenpra/git/HJR-FNO-ContingencyPlanning/exp_results"
    os.makedirs(output_dir, exist_ok=True)
    file_path   = os.path.join(output_dir, "state_history.csv")
    np.savetxt(file_path, state_history, delimiter=",", header="x,y,theta,goal_id", comments="")
    print(f"Saved to: {file_path}")


if __name__ == '__main__':
    main()