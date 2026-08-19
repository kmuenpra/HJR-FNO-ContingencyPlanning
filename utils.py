"""
utils for collision check
@author: huiming zhou
"""

import math
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")

# from Sampling_based_Planning.rrt_2D import env
# from Sampling_based_Planning.rrt_2D.rrt import Node
import env
from rrtx import Node
from eval.config import CFG


class Utils:
    def __init__(self, environment:env.Env):
        self.env = environment

        self.delta = 0.5
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        self.unknown_obs_circle = self.env.unknown_obs_circle
        
        # Lidar sensing radius used by lidar_detected(). Defaults to the shared
        # constant (eval/config.yaml) rather than a local literal: every planner
        # overwrites this attribute anyway, and the old 2.0 m default meant any
        # Utils instance that nobody remembered to overwrite sensed at a third of
        # the real range while looking perfectly healthy. Per-scenario overrides
        # (env_C uses 8.0) are applied by the caller, which knows the scenario.
        self.sensing_radius = CFG.lidar_radius

    def update_obs(self, obs_cir, obs_bound, obs_rec, unknown_obs_cir):
        self.obs_circle = obs_cir
        self.obs_boundary = obs_bound
        self.obs_rectangle = obs_rec
        self.unknown_obs_circle = unknown_obs_cir

    def get_obs_vertex(self):
        delta = self.delta
        obs_list = []

        for (ox, oy, w, h) in self.obs_rectangle:
            vertex_list = [[ox - delta, oy - delta],
                           [ox + w + delta, oy - delta],
                           [ox + w + delta, oy + h + delta],
                           [ox - delta, oy + h + delta]]
            obs_list.append(vertex_list)

        return obs_list

    def is_intersect_rec(self, start, end, o, d, a, b):
        v1 = [o[0] - a[0], o[1] - a[1]]
        v2 = [b[0] - a[0], b[1] - a[1]]
        v3 = [-d[1], d[0]]

        div = np.dot(v2, v3)

        if div == 0:
            return False

        t1 = np.linalg.norm(np.cross(v2, v1)) / div
        t2 = np.dot(v1, v3) / div

        if t1 >= 0 and 0 <= t2 <= 1:
            shot = Node((o[0] + t1 * d[0], o[1] + t1 * d[1]))
            dist_obs = self.get_dist(start, shot)
            dist_seg = self.get_dist(start, end)
            if dist_obs <= dist_seg:
                return True

        return False

    def is_intersect_circle(self, o, d, a, r):
        
        
        delta = self.delta
        
        ox, oy = o #starting node 
        dx, dy = d #direction vector
        cx, cy = a #circle center

        # Endpoint 1 inside circle
        if (ox - cx)**2 + (oy - cy)**2 <= (r + delta)**2:
            return True

        # Endpoint 2 inside circle
        ex, ey = ox + dx, oy + dy
        if (ex - cx)**2 + (ey - cy)**2 <= (r + delta)**2:
            return True
        
        #Projection of circle center onto the ray
        d2 = np.dot(d, d)
        
        if d2 < 1e-12:
            return False

        t = np.dot([a[0] - o[0], a[1] - o[1]], d) / d2

        if 0 <= t <= 1:
            shot = Node((o[0] + t * d[0], o[1] + t * d[1]))
            if self.get_dist(shot, Node(a)) <= r + delta:
                return True

        return False

    def is_collision(self, start, end):
        if self.is_inside_obs(start) or self.is_inside_obs(end):
            return True

        o, d = self.get_ray(start, end)
        # obs_vertex = self.get_obs_vertex()

        # for (v1, v2, v3, v4) in obs_vertex:
        #     if self.is_intersect_rec(start, end, o, d, v1, v2):
        #         return True
        #     if self.is_intersect_rec(start, end, o, d, v2, v3):
        #         return True
        #     if self.is_intersect_rec(start, end, o, d, v3, v4):
        #         return True
        #     if self.is_intersect_rec(start, end, o, d, v4, v1):
        #         return True

        for (x, y, r) in self.obs_circle:
            if self.is_intersect_circle(o, d, [x, y], r):
                return True

        return False
    
    def inside_reachable_sets(self, HJB_set):
        raise NotImplemented

    def is_inside_obs(self, node):
        delta = self.delta

        for (x, y, r) in self.obs_circle:
            if math.hypot(node.x - x, node.y - y) <= r + delta:
                return True

        # for (x, y, w, h) in self.obs_rectangle:
        #     if 0 <= node.x - (x - delta) <= w + 2 * delta \
        #             and 0 <= node.y - (y - delta) <= h + 2 * delta:
        #         return True

        for (x, y, w, h) in self.obs_boundary:
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                return True

        return False

    @staticmethod
    def update_robot_position(pos, dest, speed, dt):
        dist = speed * dt
        angle = math.atan2(dest[1] - pos[1], dest[0] - pos[0])
        return [pos[0] + dist * math.cos(angle), pos[1] + dist * math.sin(angle)]

    @staticmethod
    def get_ray(start, end):
        orig = [start.x, start.y]
        direc = [end.x - start.x, end.y - start.y]
        return orig, direc

    @staticmethod
    def get_dist(start, end):
        return math.hypot(end.x - start.x, end.y - start.y)

    @staticmethod
    def get_dist_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)
    
    @staticmethod
    def wrap_angle(theta):
        #Wrap angle to domain [-pi, pi]
        return (theta + math.pi) % (2 * math.pi) - math.pi 

    @staticmethod
    def update_robot_position_dubins(state, dest, dt, v=1.0, w_max=1, stop_at=None):
        """
        Dubins/unicycle update with bounded angular velocity: one pure-pursuit
        control step toward ``dest``.

        Deliberately kept identical, formula for formula, to the pure-pursuit
        tracker used for MPPI's topological guidance (mppi_src/guidance.py
        ``TopoGuidance._track``) and to ``Navigation2DEnv.dynamics``, so that a
        behavioural difference between the two planners is attributable to the
        PLANNERS and not to their controllers:

          omega = clip(wrap(atan2(dest - p) - theta) / dt, +-w_max)
          x, y  = x, y + v*dt*(cos, sin)(theta)      <-- OLD theta
          theta = wrap(theta + omega*dt)

        Note the integration order: x/y step with the OLD heading and only then
        is theta updated. This used to be the other way round (theta first, x/y
        with the NEW heading), which turns slightly "early" compared with the env
        and accumulates a systematic difference in turn geometry.

        @params
        - state   : [x, y, theta]
        - dest    : [x_d, y_d]   the pursued point (a waypoint / lookahead point)
        - dt      : integration timestep [s]. NOTE omega saturates whenever the
                    heading error exceeds w_max*dt, so a small dt makes this a
                    bang-bang turner; use the same dt as the env (0.1) to get
                    comparable behaviour.
        - v       : commanded speed [m/s]
        - w_max   : angular-rate bound [rad/s]
        - stop_at : optional (x, y) of the FINAL target. When given, the speed is
                    tapered as v = min(v, ||stop_at - p|| / dt) so the robot
                    cannot overshoot it -- the same taper the topo tracker applies
                    at its path end.
        @return [x_new, y_new, theta_new] with theta_new wrapped to [-pi, pi),
                the convention used by the env, the tracker and the HJR grids.
        """

        x, y, theta = state
        theta = Utils.wrap_angle(theta)          # accept either convention as input
        dx = dest[0] - x
        dy = dest[1] - y

        desired_theta = math.atan2(dy, dx)
        heading_error = Utils.wrap_angle(desired_theta - theta)

        omega = max(-w_max, min(w_max, heading_error / dt))

        # decelerate into the final target so a step cannot overshoot it
        if stop_at is not None:
            d_end = math.hypot(stop_at[0] - x, stop_at[1] - y)
            v = max(0.0, min(v, d_end / dt))

        # x/y step uses the OLD heading, then theta is updated (env convention)
        x_new = x + v * dt * math.cos(theta)
        y_new = y + v * dt * math.sin(theta)
        theta_new = Utils.wrap_angle(theta + omega * dt)

        return [x_new, y_new, theta_new]

        
        
    
    def lidar_detected(self, robot_position):
        """
        @description 
        Simulate a circular lidar sensor.
        The obstacle is detected when the obstacle center lies inside the sensing radius

        Detection rule:
        An obstacle is detected when the obstacle center lies within
        the sensing radius, i.e. d <= sensing_radius.
        
        @params 
        - robot_position : (x, y) "Current robot position" 
        - sensing_radius : float "Lidar sensing radius" 
        
        @return 
        - self.unknown_obs_circle: list "Remaining Unknown obstacles (for plotting)"
        - detected_obstacles : list "Obstacles that satisfy the detection condition"
        """

        if not self.unknown_obs_circle:
            return [], []

        obs = np.asarray(self.unknown_obs_circle, dtype=float)
        x_r, y_r = robot_position

        # Distance to obstacle centers
        d = np.sqrt((obs[:, 0] - x_r)**2 + (obs[:, 1] - y_r)**2)

        # Detection condition
        detected_mask = d <= self.sensing_radius

        if not np.any(detected_mask):
            return self.unknown_obs_circle, []

        # Split detected vs remaining in ONE operation
        detected_obstacles = obs[detected_mask].tolist()
        remaining_obstacles = obs[~detected_mask].tolist()

        # Update unknown obstacle list IN PLACE so the shared store identity is preserved
        # (all trees + plotting reference the same list; rebinding would diverge them).
        self.unknown_obs_circle[:] = remaining_obstacles

        return self.unknown_obs_circle, detected_obstacles

