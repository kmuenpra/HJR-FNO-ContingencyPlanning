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
from typing import Dict, List, Tuple
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

# NOTE: no module-level import of the FNO classes is needed for unpickling —
# HJR_FNO3d.__init__ injects FNO3d / SpectralConv3d into __main__ before torch.load.



class Node(Sequence):
    
    # global counter shared by all Node instances
    _id_counter = itertools.count()
    
    # inherits from Sequence to support indexing and thus kd-tree support
    def __init__(self, n, lmc=np.inf, cost_to_goal=np.inf):
        
        # Unique, stable identifier
        self.id: int = next(Node._id_counter)
        
        self.n = n # make iterable for kd-tree insertion
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.children = set([])
        self.cost_to_goal = cost_to_goal
        self.lmc = lmc
        self.infinite_dist_nodes = set([]) # set of nodes u where d_pi(v,u) has been set to infinity after adding an obstacle
        self.N_o_plus = set([]) # outgoing original neighbours
        self.N_o_minus = set([]) # incoming original neighbours
        self.N_r_plus = set([]) # outgoing running in neighbours
        self.N_r_minus = set([]) # incoming running in neighbours
        self.active = False

    def __eq__(self, other):
                
        # this is required for the checking if a node is "in" self.Q, but idk what the condition should be
        return id(self) == id(other) or \
            self.get_key() == other.get_key() or \
            math.hypot(self.x - other.x, self.y - other.y) < 1e-6 #NOTE i chnange the tolerance to 1e-6 from 1e-4
        # return self.get_key() == other.get_key()
        # return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        # this is just in case their keys are the same, so a ValueError is not thrown
        return 1

    def __hash__(self):
        # this is required for storing Nodes to sets
        return hash(self.n)

    def __getitem__(self, i):
        # this is required for kd-tree insertion
        return self.n[i]
    
    def __len__(self):
        # this is required for kd-tree insertion
        return 2

    def all_out_neighbors(self):
        return self.N_o_plus.union(self.N_r_plus)
    
    def all_in_neighbors(self):
        return self.N_o_minus.union(self.N_r_minus)
   
    def set_parent(self, new_parent):
        # if a parent exists already
        
        if self.parent:
            try:
                self.parent.children.remove(self)
            except:
                print("")
                # traceback.print_stack(limit=5)
                # print('KeyError in set_parent()')
                # print('Node', (self.x, self.y))
                # print('cost', (self.cost_to_goal, self.lmc))
                # print("Node's Parent", (self.parent.x, self.parent.y))
                # for child in self.parent.children:
                #     print("- children", (child.x, child.y))
        
        self.parent = new_parent
        new_parent.children.add(self)

    def would_create_cycle(self, new_parent):
        """True if setting self.parent = new_parent would form a cycle (new_parent is self
        or a descendant of self). Walks new_parent's ancestor chain by identity; the seen
        guard terminates even if the tree is already transiently cyclic."""
        n = new_parent
        seen = set()
        while n is not None and id(n) not in seen:
            if n is self:
                return True
            seen.add(id(n))
            n = n.parent
        return False

    def get_key(self):
        return (min(self.cost_to_goal, self.lmc), self.cost_to_goal)

    def cull_neighbors(self, r):
        # Algorithm 3
        N_r_plus_list = list(self.N_r_plus) # can't remove from set while iterating over it
        
        for u in N_r_plus_list:
            # switched order of conditions in if statement to be faster
            if self.cost_to_goal == 0.0 and self.lmc == 0.0:
                print("Goal is a neighbor node!")
                
            else:
                try:
                    if (not self.parent or (self.parent and self.parent != u)) and r < self.distance(u):
                        N_r_plus_list.remove(u)
                        try:
                            u.N_r_minus.remove(self)
                        except KeyError:
                            # print('KeyError in RRTX.cull_neighbors(), skipping remove')
                            pass
                except AttributeError:
                    # print("AtrributeError, self.parent", self.parent)
                    print("Position of none with no parent", (self.x, self.y))
                    # print("Goal location might intersect with the obstacle")
                    traceback.print_exc() 
                

        self.N_r_plus = set(N_r_plus_list)

    def update_LMC(self, orphan_nodes, r, epsilon, utils):
        # Algorithm 14
        # pass in orphan nodes from main code, make sure the set is maintained properly
        self.cull_neighbors(r)
        # candidates (u, d_pi(v,u)+lmc(u)) in increasing lmc order; take the first improving,
        # collision-free, acyclic one.
        cands = sorted(
            ((u, self.distance(u) + u.lmc)
             for u in (self.all_out_neighbors() - orphan_nodes) if u.parent and u.parent != self),
            key=lambda x: x[1],
        )
        for p_prime, lmc_prime in cands:
            if lmc_prime >= self.lmc:        # sorted -> no further improvement possible
                break
            if utils.is_collision(self, p_prime):
                continue
            # Option B: a finite & consistent candidate with lmc < self.lmc cannot be a descendant
            # of self (lmc monotonicity), so skip the cycle walk; only check when costs are
            # inf / inconsistent (the regime where monotonicity — and acyclicity — can break).
            safe = math.isfinite(p_prime.lmc) and p_prime.cost_to_goal == p_prime.lmc
            if not safe and self.would_create_cycle(p_prime):
                continue
            self.lmc = lmc_prime
            self.set_parent(p_prime)
            break

    def distance(self, other):
        return np.inf if other in self.infinite_dist_nodes else math.hypot(self.x - other.x, self.y - other.y)
        
    def clone(self):
        v = Node((self.x, self.y))
        v.cost_to_goal = self.cost_to_goal
        v.lmc = self.lmc
        return v

    

class RRTX:
    
    from HJR_FNO.HJR_FNO3d import HJR_FNO
    
    def __init__(
        self,
        x_start: Tuple[float, float],
        x_goal: Tuple[float, float],
        goal_id: int,
        other_goals:List,
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
        environment: env.Env = None,
        ) -> None:
        
        # Start and Goal
        self.s_goal = Node(x_goal, lmc=0.0, cost_to_goal=0.0)
        self.goal_id = goal_id
        # self.s_goal.active = True
        
        # self.s_start = Node(x_start)
        self.s_bot = None
        self._pending_reset_target = None
        self._pending_reset_heading = None
        
        #For multi-goal tree expansion (SFF*)
        self.prob_q = 0.9
        self.other_goals_id = other_goals_id
        self.other_goals = []
        for g in other_goals:
            self.other_goals.append(Node((g[0],g[1])))
                        
        
        
        # RRTx configs
        # Use the shared Env (single obstacle store) when provided; fall back to a private one
        # only if constructed standalone. obs_circle below is then a reference to the shared list,
        # so in-place updates by any tree are seen by all trees + plotting.
        self.env = environment if environment is not None else env.Env(safe_regions=safe_regions)
        self.plotting = plotting #plotting.Plotting(x_start, x_goal, safe_regions=safe_regions)
        self.utils = utils.Utils(environment=self.env)
        
        self.step_len = step_len
        self.epsilon = epsilon
        self.bot_sample_rate = bot_sample_rate
        self.search_radius = 0.0
        self.iter_max = iter_max
        self.kd_tree = kdtree.create([self.s_goal])
        sys.setrecursionlimit(3000) # for the kd-tree cus it searches recursively
        self.all_nodes_coor = []
        self.tree_nodes = [self.s_goal] # this is V_T in the paper
        self.orphan_nodes = set([]) # this is V_T^C in the paper, i.e., nodes that have been disconnected from tree due to obstacles
        self.Q = [] # priority queue of ComparableNodes
        self.path_to_goal = np.array([False for _ in range(len(other_goals))])
        self.robot_path_to_goal = False
        
        #State and Sensor
        self.robot_state = [0,0,0]
        # self.robot_position = [self.s_bot.x, self.s_bot.y]
        self.robot_speed = 0.6 # m/s
        self.lidar_range = lidar_range
        self.utils.sensing_radius = lidar_range
        
        #HJR-FNO configs
        self.Tf_reach = hjr_fno.Tf_reach #must be less than 8s (underapproximation of the training data)
        self.hjr_fno = hjr_fno #HJR_FNO(safe_regions=safe_regions, Tf_reach=self.Tf_reach)
        self.hjr_fno.utils.sensing_radius = lidar_range
        self.safe_regions = safe_regions
        self.HJ_contingency_enable = HJ_contingency_enable
        self.contingency_triggered = False 
        
            
        #Plotting
        self.path = [] #robot's path
        self.path_node = []
        self.multi_paths = [[] for _ in range(len(self.other_goals))]
        self.multi_path_nodes = [[] for _ in range(len(self.other_goals))]

        self.fig = fig
        self.ax = ax
        # self.fig, self.ax = plt.subplots(figsize=(12, 8))
        # self.fig.suptitle('RRTX')
        # self.ax.set_xlim(self.env.x_range[0], self.env.x_range[1]+1)
        # self.ax.set_ylim(self.env.y_range[0], self.env.y_range[1]+1)   
           
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

        # for gamma computation
        self.d = 2 # dimension of the state space
        self.zeta_d = np.pi # volume of the unit d-ball in the d-dimensional Euclidean space
        self.gamma_FOS = gamma_FOS # factor of safety so that gamma > expression from Theorem 38 of RRT* paper
        self.update_gamma() # initialize gamma

        

    def _dist_to_goal(self, node: Node, goal: Tuple[float, float]) -> float:
        dx = node.x - goal[0]
        dy = node.y - goal[1]
        return math.hypot(dx, dy)


    def planning(self, iter_max=None, robots_plan=False):

        # set seed for reproducibility
        # np.random.seed(0)
        
        if iter_max is None:
            iter_max = self.iter_max

        # animation stuff
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
                
                #randomly sample the goal that hasn't been reached, OR the robot's current position
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
        """
        Reset the robot to a new position without breaking the tree structure.
        If no collision-free connection exists yet, a pending target is stored and
        the planner will bias sampling toward it until a path is established.
        """
        new_node = Node(new_position)

        # --- Step 1: Find nearest tree node ---
        nearest = self.nearest(new_node)
        dist = math.hypot(nearest.x - new_node.x, nearest.y - new_node.y)

        SNAP_THRESHOLD = 0.5

        if dist < SNAP_THRESHOLD and nearest.lmc < np.inf:
            # Reuse an existing well-connected node
            self.s_bot = nearest
            self._pending_reset_target = None

        else:
            # Saturate so it's within step_len of nearest
            new_node = self.saturate(nearest, new_node)

            V_near = self.near(new_node)
            if not V_near:
                V_near = [nearest]

            V_near_free = [u for u in V_near if not self.utils.is_collision(u, new_node)]

            if not V_near_free:
                # No collision-free connection yet — store as pending target.
                # random_node() will bias sampling toward it until connected.
                
                # print("[reset_robot_position] No collision-free neighbors found. "
                #     "Storing as pending target; planner will grow toward it.")
                self._pending_reset_target = Node(new_position)  # store original, unsaturated
                self._pending_reset_heading = heading
                # Update robot state visually/logically even if not yet in tree
                self.robot_position = list(new_position)
                self.robot_state = [new_position[0], new_position[1],
                                    heading if heading is not None else self.robot_state[2]]
                self.robot_path_to_goal = False
                self.path = []
                return False

            # Has collision-free neighbors — wire into tree normally
            self._pending_reset_target = None
            self.find_parent(new_node, V_near_free)

            if new_node.parent is None:
                # print("[reset_robot_position] find_parent failed despite free neighbors, aborting.")
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

        # --- Step 2: Update robot state ---
        self.robot_position = [self.s_bot.x, self.s_bot.y]
        self.robot_state = [self.s_bot.x, self.s_bot.y,
                            heading if heading is not None else self.robot_state[2]]

        self.robot_path_to_goal = self.s_bot.lmc < np.inf
        self.update_path(self.s_bot)
        
        return self._pending_reset_target is None 

        # print(f"[reset_robot_position] Robot reset to ({self.s_bot.x:.2f}, {self.s_bot.y:.2f}), "
        #     f"lmc={self.s_bot.lmc:.3f}, path_to_goal={self.robot_path_to_goal}")
         
         
                    
    
    # def set_new_goal(self, target_id:int, new_pose:Tuple):
    #     '''
    #     Set the current target with target_id to the new position 
    #     Assume new_pose := Tuple(x,y) is obstacle free
    #     '''
        
    #     #set new target location
    #     j = self.other_goals_id.index(target_id)
    #     self.other_goals[j] = Node((new_pose[0], new_pose[1]))
        
    #     #find and extend nearest node toward the new target location
    #     v_nearest = self.nearest(self.other_goals[j] )
    #     v = self.saturate(v_nearest, self.other_goals[j])
        
    #     if v and not self.utils.is_collision(v_nearest, v) and self.is_feasible_ray(v_nearest, v):
                    
    #             self.extend(v, v_nearest)
                
    #             if v.parent is not None:
    #                 self.rewire_neighbours(v)
    #                 self.reduce_inconsistency()

        
    #     if self.other_goals[j].cost_to_goal < np.inf:
    #         self.path_to_goal[j] = True
    #     else:
    #         self.path_to_goal[j] = False
            
    #     #expand the tree until new path is found
    #     # TODO there should be a better solution than this while loop
    #     while not self.path_to_goal.all():
            
    #         self.planning(iter_max=100)
        
        
                    
    def reset_robot_v2(self, current_state, iter_max=500):          
        
        '''
        When updating s_bot Node,
        1. check if there is an existing node close enough to current_state
        2. If the tree has not been expanded enough, replanning until there is a path from s_bot to goal
        '''
        
        self.robot_path_to_goal = False
        self.path = []
          
        self.s_bot = Node((current_state[0], current_state[1]))
        self.robot_state = current_state
        self.robot_position = current_state[:2]
        
        v_nearest = self.nearest(self.s_bot)
        
        if math.hypot(v_nearest.x - self.s_bot.x, v_nearest.y - self.s_bot.y) < 1e-3: #equal if position are close within tolerance
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
                    v= self.random_node()
                    # closest_idx = self.hjr_fno.find_feasible_closest_region(robot_pose=np.atleast_2d([current_state[0], current_state[1]]))
                    
                    # #mean
                    # x, y, _ = self.safe_regions[closest_idx[0]]
                    # mu = np.array([x, y])

                    # # Isotropic covariance
                    # sigma = 5.5
                    # cov = sigma**2 * np.eye(2)

                    # sample = np.random.multivariate_normal(mu, cov)
                    # v =  Node((sample[0], sample[1]))
                    
                v_nearest = self.nearest(v)
                v = self.saturate(v_nearest, v)
                
                if v and not self.utils.is_collision(v_nearest, v) and self.is_feasible_ray(v_nearest, v):
                        
                    self.extend(v, v_nearest, robots_plan=True)
                    
                    #doesnt hurt to do this
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
                    
        # if not self.robot_path_to_goal:

        #     # find nearest node with finite cost
        #     valid_nodes = [v for v in self.tree_nodes if v.cost_to_goal < np.inf]

        #     if not valid_nodes:
        #         print("No valid goal-connected nodes exist")
        #         return

        #     # pick closest among them
        #     v_nearest = min(
        #         valid_nodes,
        #         key=lambda v: np.hypot(v.x - self.s_bot.x, v.y - self.s_bot.y)
        #     )

        #     self.s_bot = v_nearest
        #     self.robot_path_to_goal = True
        #     self.update_path(self.s_bot)

                    
    def update_robot_heading(self, new_heading=None):
        
        if new_heading is not None:           
            self.robot_state[2] = new_heading
            
        elif self.s_bot.parent is not None:
            dx = self.s_bot.parent.x - self.s_bot.x
            dy = self.s_bot.parent.y - self.s_bot.y
            heading = math.atan2(dy, dx)
            self.robot_state[2] = heading
            
            
    
    # def reset_robot(self, current_state, iter_max=100):

    #     self.s_bot = Node((current_state[0], current_state[1])) #robot start at the same location but does not have parents
    #     self.path = []
    #     self.robot_path_to_goal = False
    #     # self.curr_tree_idx = j
        
        
    #     # #Connect to an existing tree
    #     # if self.path_to_goal[j]:
    #     #     self.robot_path_to_goal = True
    #     #     self.update_path(self.s_bot)
        
    #     #expand the tree until new path is found
    #     # TODO there should be a better solution than this while loop
                
    #     # v = self.random_node(robots_plan=True)
    #     v_nearest = self.nearest(self.s_bot)
        
    #     v = self.saturate(v_nearest, self.s_bot)
            
    #     if v and not self.utils.is_collision(v_nearest, v) and self.is_feasible_ray(v_nearest, v):
                
    #         self.extend(v, v_nearest, robots_plan=True)
            
    #         #doesnt hurt to do this
    #         if v is self.s_bot and self.s_bot.lmc < np.inf:
    #             self.verify_queue(self.s_bot)
            
    #         if v.parent is not None:
    #             self.rewire_neighbours(v, robots_plan=True)
    #             self.reduce_inconsistency(robots_plan=True)

        
    #     if self.s_bot.cost_to_goal < np.inf:
    #         self.robot_path_to_goal = True
                
    #     if (not self.robot_path_to_goal) or (not all(self.path_to_goal)): #suppose to be while True
            
    #         self.planning(iter_max=iter_max, robots_plan=True)
    #         print("replanning for 1000 iterations")
        
        # self.robot_state = [self.s_bot.x, self.s_bot.y, current_state[2]]
        # self.robot_position = [self.s_bot.x, self.s_bot.y]
                 
    def planning_with_robot(self , steps=10):
        
        new_obs_flag = False
        all_new_obs = []
        traversed_distance = 0.0
                        
        for step_idx in range(steps): #set plotting Hz
            
            # Attempt to connect a pending reset target each iteration
            if self._pending_reset_target is not None:
                # Always update pending target to current robot position
                # before attempting connection — robot may have moved since it was set
                self._pending_reset_target = Node((self.robot_position[0], 
                                                self.robot_position[1]))
                self._try_connect_pending()
                    
            if self.s_bot.cost_to_goal == np.inf:
                self.robot_path_to_goal = False
                
                if self.s_bot.lmc < np.inf:
                    self.verify_queue(self.s_bot)
                # print("======== Robot has infinite cost ========")
                # print(self.s_bot.cost_to_goal)
                # print(self.s_bot.lmc)
                # print(self.path)
                # print(self.s_bot.parent)
                
            if self.contingency_triggered:
                return all_new_obs, new_obs_flag, traversed_distance
        
            # if there is path to goal and run_time > 5s, then start moving the robot
            if self.robot_path_to_goal:

                
                #Terminate when reach the goal
                if self.s_bot.cost_to_goal == 0.0 and self.s_bot.lmc == 0.:
                    return all_new_obs, new_obs_flag, traversed_distance
                
                # Update robot position            
                self.robot_state = self.utils.update_robot_position_dubins(self.robot_state, [self.s_bot.parent.x, self.s_bot.parent.y], 0.01, v=self.robot_speed)
                self.robot_position = self.robot_state[:2]
                
                # Lidar radial detection of the obstacles
                self.unknown_obs_circle, detected_obs = self.utils.lidar_detected(self.robot_position)
                # NOTE self.unknown_obs_circle == self.env.unknown_obs_circle == self.utils.unknown_obs_circle
                # self.plotting.unknwown_obs_circle must be updated independently >>> implement this in RRTX.update_obstacles()
                
                if len(detected_obs) > 0:
                    
                    #sharing new obstacles found with other RRTX-tree in TSP loop
                    all_new_obs += detected_obs
                    new_obs_flag = True

                    
                    print(f"\n ----- Rewiring Trees {self.goal_id} due to {len(detected_obs)} newly-detected obstacle(s) ----- ")
                    print("Obstacle location: ", detected_obs)
                    
                    # TODO update HJB reachable set with the detected_obs here
                    exec_time_start = time.time()
                    
                    if self.HJ_contingency_enable:
                        self.hjr_fno.update_obs(detected_obs)
                        print("Update Reachable set:", time.time() - exec_time_start, "s")
                    
                    # update known obstacles within the environment (all at once: one tree-repair)
                    self.update_obstacles(detected_obs, robots_plan=True, print_time=True)
                
                # update node that robot is currently at
                if self.s_bot.parent is not None:
                    if math.hypot(self.robot_position[0] - self.s_bot.parent.x,
                                self.robot_position[1] - self.s_bot.parent.y) < 0.5:
                        
                        traversed_distance += self.s_bot.distance(self.s_bot.parent)
                        self.s_bot = self.s_bot.parent
                        
            ''' Expand the tree for more optimal path'''            
            self.search_radius = self.shrinking_ball_radius()
            
            if np.random.random() < self.prob_q:
            
                v = self.random_node()
                
            else:
                
                #randomly sample the goal that hasn't been reached, OR the robot's current position
                candidates = [g for g, reached in zip(self.other_goals, self.path_to_goal) if not reached]
                    
                if (not self.robot_path_to_goal):
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
                    
            # Allow matplotlib to process events (including mouse clicks)
            # This is crucial for the click handler to work
            if step_idx % 10 == 0:  # Still skip some iterations
                self.fig.canvas.flush_events()
                
                
        return all_new_obs, new_obs_flag, traversed_distance


    def extend(self, v, v_nearest, robots_plan=False):
        # Algorithm 2
        V_near = self.near(v)

        ### THIS WAS NOT IN PAPER, BUT IN JULIA CODE
        if not V_near:
            V_near.append(v_nearest)

        self.find_parent(v, V_near)
        if not v.parent:
            return
        self.add_node(v, robots_plan=robots_plan)
        # child has already been added to parent's children in call to set_parent()
        for u in V_near:
            # collisions are symmetric for us
            if not self.utils.is_collision(u, v) and self.is_feasible_ray(u,v):
                v.N_o_plus.add(u)
                v.N_o_minus.add(u)
                u.N_r_plus.add(v)
                u.N_r_minus.add(v)
                
    def update_obstacles(self, obs_cir, robots_plan=False, print_time=False, record=True):
        #update_obstacles(self, event, obs_cir):

        # Algorithm 8
        # x, y = int(event.xdata), int(event.ydata)

        # print("current search radius", self.search_radius)

        # record=False -> graph repair only (obstacle store is shared; do not re-record).
        exec_time_start = time.time()
        self.add_new_obstacle(obs_cir, record=record)   #TODO: This function sometimes takes a bit of time
        if print_time:
            print("Added new obstacles:", time.time() - exec_time_start, "s")
        
        exec_time_start = time.time()
        self.propagate_descendants(robots_plan=True)
        if print_time:
            print("Propagate Descendants:", time.time() - exec_time_start, "s")
        
        exec_time_start = time.time()
        # if robots_plan:
        self.verify_queue(self.s_bot)
            
        for g in self.other_goals:
            self.verify_queue(g)
        if print_time:
            print("Verify Queue:", time.time() - exec_time_start, "s")
        
        exec_time_start = time.time()
        self.reduce_inconsistency(robots_plan=True)
        if print_time:
            print("Reduce Inconsistency:", time.time() - exec_time_start, "s")

    def _revalidate_candidates(self, changed_regions):
        """3a/3b: tree nodes whose closest safe region was re-predicted AND that now fail the buffered feasibility test (B1: parent->node heading)."""
        if not self.HJ_contingency_enable or not changed_regions or not self.tree_nodes:
            return set()

        pts = np.array([[n.x, n.y] for n in self.tree_nodes])
        # 3a: keep only nodes whose closest safe region changed (others cannot have flipped)
        node_region = np.asarray(self.hjr_fno.find_feasible_closest_region(robot_pose=pts)).reshape(-1)
        cand_mask = np.isin(node_region, changed_regions)
        if not np.any(cand_mask):
            return set()

        cand_nodes = [n for n, m in zip(self.tree_nodes, cand_mask) if m]
        cand_pts = pts[cand_mask]
        cand_region = node_region[cand_mask]
        # B1: score each node along the edge it lives on (parent -> node); root falls back to robot
        # heading. Only needed for the HJR_sets source — skip entirely when theta-independent.
        cand_thetas = None
        if self.hjr_fno.feasibility_source == "HJR_sets":
            cand_thetas = np.array([
                math.atan2(n.y - n.parent.y, n.x - n.parent.x) if n.parent is not None else self.robot_state[2]
                for n in cand_nodes
            ])
        # 3b: select nodes that fail the buffered feasibility test (value above per-region safe_margin - buffer); widen buffer to widen the band
        vals = self.hjr_fno.feasibility_values(cand_pts, thetas=cand_thetas)
        thresholds = np.array([self.hjr_fno.safe_margin[r] - self.hjr_fno.feasibility_buffer for r in cand_region])
        return {n for n, val, th in zip(cand_nodes, vals, thresholds) if val > th}

    def add_new_obstacle(self, obs, record=True):
        # Algorithm 12
        # x, y, r = obs
        # print("Osbstacle at: x =", x, ", y =", y, ", r = ", r)

        # Accept either a single obstacle [x, y, r] or a list of obstacles [[x,y,r], ...]
        # and process them together in one pass (one tree-repair, one is_feasible_ray per edge).
        obs_list = [obs] if np.ndim(obs) == 1 else list(obs)

        # record=True only at the detection site: the obstacle store (obs_circle) is now SHARED
        # across all trees + plotting, so re-recording in other trees would duplicate it.
        if record:
            self.obs_circle.extend(obs_list)
            self.plotting.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle, self.unknown_obs_circle) # for plotting obstacles
            self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle, self.unknown_obs_circle) # for collision checking
        self.update_gamma() # per-tree: free space volume changed, so gamma must change too

        # self.path.append(np.array([[node.x, node.y], [node.parent.x, node.parent.y]]))
        # for edge in self.path:
        #     print(edge)
            # print(self.utils.get_ray(v, u), obs[:2], obs[2]) or not self.is_feasible_ray(v,u))

        # NOTE Collect all directed node pair (v->u) that intersects with the obstacles
        #
        # for all nodes 'v' in tree_nods
        #       for all nodes 'u' in neighborhood of 'v' (include static and running nodes)
        #               check if the edge v -> u intersected with
        # Candidate node set:
        #  - geom_nodes: obstacle-local nodes (widened to step_len so long straddling edges can't slip through), over ALL obstacles
        #  - margin_nodes: HJR feasibility re-validation when FNO repredicts feasible regions
        #  - path_nodes (record=True only): the detection tree also re-validates the paths between goals (for solving ATSP)
        #    other trees (record=False) rely on geom+margin since their paths get rerouted anyway.
        geom_nodes = set()
        for o in obs_list:
            geom_nodes.update(self.kd_tree.search_nn_dist((o[0], o[1]), o[2] + self.step_len))
        margin_nodes = self._revalidate_candidates(self.hjr_fno._last_changed_regions)
        nodes_to_check = geom_nodes | margin_nodes

        # If this is the first time obstacles ever been recorded, then check if there are any paths to other goals that get invalidated.
        if record:
            nodes_to_check |= {v for nodes in self.multi_path_nodes for v in nodes} | set(self.path_node)

        # Enumerate candidate directed edges once. (v,u) and (u,v) are distinct under HJR_sets
        # (heading differs), so they are kept separate.
        edges = [(v, u) for v in nodes_to_check for u in v.all_out_neighbors()]

        # Pass 1 — geometric (cheap, no FNO): edges intersecting ANY obstacle are invalid; the rest
        # go to the batched feasibility pass (this preserves the old `intersect OR not feasible` short-circuit).
        geom_invalid, remaining = [], []
        for v, u in edges:
            o, d = self.utils.get_ray(v, u)
            if any(self.utils.is_intersect_circle(o, d, ob[:2], ob[2]) for ob in obs_list):
                geom_invalid.append((v, u))
            else:
                remaining.append((v, u, o, d))

        # Pass 2 — ONE batched feasibility call over all remaining edges' samples (Fix #2).
        feas_invalid = []
        if self.HJ_contingency_enable and remaining:
            use_theta = (self.hjr_fno.feasibility_source == "HJR_sets")   # theta only matters for Option B
            t_vals = np.linspace(0, 1, 4)
            all_pts = []
            all_thetas = [] if use_theta else None
            for v, u, o, d in remaining:
                all_pts.append(o + t_vals[:, None] * d)                   # (4, 2) samples along the edge
                if use_theta:
                    all_thetas.append(np.full(4, math.atan2(u.y - v.y, u.x - v.x)))
            all_pts = np.vstack(all_pts)                                  # (R*4, 2)
            thetas = np.concatenate(all_thetas) if use_theta else None
            feas = self.hjr_fno.points_feasible(
                all_pts, thetas=thetas, reachable_set_constraint=self.HJ_contingency_enable
            ).reshape(len(remaining), 4)
            feas_invalid = [(remaining[k][0], remaining[k][1]) for k in range(len(remaining)) if not feas[k].all()]

        E_O = geom_invalid + feas_invalid
        
        
        # E_O = E_O + E_1
        
        print("Invalidated nodes", len(E_O))
        self.invalid_nodes = E_O
        
        # To preserve graph structure:
        # instead of removing (v->u) from the neighbor set, make the edge haviing infinite cost
        for v, u in E_O:
            v.infinite_dist_nodes.add(u)
            u.infinite_dist_nodes.add(v)
            if v.parent and v.parent == u:
                self.verify_orphan(v)
                # should theoretically check if the robot is on this edge now, but we do not
                # v.parent.children.remove(v) # these two lines are from the Julia code
                # v.parent = None 
                
        heapq.heapify(self.Q) # reheapify after removing a bunch of elements and ruining queue

    def verify_orphan(self, v):
        # Algorithm 10
        # if v is in Q, remove it from Q and add it to orphan_nodes
        key = self.node_in_queue(v)
        if key is not None:
            self.Q.remove((key, v))
        self.orphan_nodes.add(v)

    def propagate_descendants(self, robots_plan=False):
        
        # ------------------
        # NOTE Orphan nodes are all nodes that is disconnected from the goal nodes due to newly-observed obstacles
        # ------------------
        
        # Algorithm 9
        if not self.orphan_nodes:
            return
        # recursively add children of nodes in orphan_nodes to orphan_nodes using BFS
        orphan_queue = deque(list(self.orphan_nodes))
        while orphan_queue:
            node = orphan_queue.pop()
            for child in node.children:
                orphan_queue.append(child)
                self.orphan_nodes.add(child)

        
        # put all outgoing neighbours of orphan nodes in Q and tell them to rewire
        for v in self.orphan_nodes:
            for u in (v.all_out_neighbors().union(set([v.parent]))) - self.orphan_nodes:
                u.cost_to_goal = np.inf
                self.verify_queue(u)
        heapq.heapify(self.Q) # reheapify after keys changed to re-sort queue

        # clear orphans, set their costs to infinity, empty their parent
        for v in self.orphan_nodes:
            # self.orphan_nodes.remove(v)
            v.cost_to_goal = np.inf
            v.lmc = np.inf
            if v.parent:
                v.infinite_dist_nodes.add(v.parent)
                v.parent.infinite_dist_nodes.add(v)
                v.parent.children.remove(v)
                v.parent = None

            try:
                self.tree_nodes.remove(v) # NOT IN THE PSEUDOCODE
                self.kd_tree.remove(v)
                
            except ValueError:
                pass
            
        # check if robot node got orphaned
        if robots_plan:
            if self.s_bot in self.orphan_nodes or np.isinf(self.s_bot.cost_to_goal):
                print('robot node got orphaned')
                self.robot_path_to_goal = False
                self.path = []
        
        # else:
        #Check if path between goal got orphaned
        for j, goal_j in enumerate(self.other_goals):
            if goal_j in self.orphan_nodes or np.isinf(goal_j.cost_to_goal):
                self.path_to_goal[j] = False
                self.multi_paths[j] = []
                self.multi_path_nodes[j] = []

        self.orphan_nodes = set([]) # reset orphan_nodes to empty set

    def verify_queue(self, v):
        # Algorithm 13
        # this does not do the updating, it is done after all changes are made (in propagate_descendants)
        # if v is in Q, update its cost and position, otherwise just add it
        key = self.node_in_queue(v)
        # if v is already in Q, remove it first before adding it with updated cost
        if key is not None:
            self.Q.remove((key, v))
        heapq.heappush(self.Q, (v.get_key(), v))

    def reduce_inconsistency(self, robots_plan=False):
        # Algorithm 5
              
        # while 
        # 1. prioty queue is not empty
        # AND
        # 2. (
        #       - either robot's node is inconsistent
        #       OR
        #       - any of the other goal nodes is inconsistent
        #    )
        
        # while len(self.Q) > 0  and (
        #         (
        #             robots_plan and (
        #                 self.Q[0][0] < self.s_bot.get_key()
        #                 or self.s_bot.lmc != self.s_bot.cost_to_goal
        #                 or np.isinf(self.s_bot.cost_to_goal)
        #                 or self.s_bot in {node for _, node in self.Q}
        #             )
        #         )
        #         or
        #         (
        #             any(
        #                 self.Q[0][0] < g.get_key()
        #                 or g.lmc != g.cost_to_goal
        #                 or np.isinf(g.cost_to_goal)
        #                 or g in {node for _, node in self.Q}
        #                 for g in self.other_goals
        #             )
        #         )
        #     ):
    
        
        
        while len(self.Q) > 0 and any(
                                self.Q[0][0] < v.get_key() 
                                or v.lmc != v.cost_to_goal 
                                or v in {node for _, node in self.Q}
                                
                                for v in (
                                    self.other_goals + ([self.s_bot] if robots_plan else [])
                                )
                            ):
                
        

            try:
                v = heapq.heappop(self.Q)[1]
            except TypeError:
                print('something went wrong with the queue')
        
            if v.cost_to_goal - v.lmc > self.epsilon:
                v.update_LMC(self.orphan_nodes, self.search_radius, self.epsilon, self.utils)
                self.rewire_neighbours(v, robots_plan=robots_plan) #find better paths through v

            v.cost_to_goal = v.lmc

        # Tree is now consistent — safe to recompute paths (no transient parent cycles).
        self.refresh_paths(robots_plan=robots_plan)

        # if robots_plan:
        #     assert not (
        #         self.s_bot.lmc < np.inf and self.s_bot.cost_to_goal == np.inf
        #     ), "Robot left inconsistent: finite lmc but infinite cost_to_goal"

            

    def add_node(self, node_new, robots_plan=False):
        self.all_nodes_coor.append(np.array([node_new.x, node_new.y])) # for plotting
        self.tree_nodes.append(node_new)
        
        #update priority queue with distance to other target locations
        # node_new.active = True
        # for goal_j, heap in self.Q_other_goals.items():
        #     d = self._dist_to_goal(node_new, goal_j)
        #     heapq.heappush(heap, (d, node_new.id, node_new))
        
        self.kd_tree.add(node_new)
        
        # if new node is at start, then path to goal is found
        if robots_plan:
            
            if node_new == self.s_bot:
                self.s_bot = node_new
                self.robot_path_to_goal = True
                self.update_path(self.s_bot) # update path to goal for plotting
                
                # self.other_goals[self.curr_tree_idx] = self.s_bot
                # self.update_multi_paths(self.s_bot, self.curr_tree_idx)
                # self.path_to_goal[self.curr_tree_idx] = True
                return
        # else:
            
        for j in range(len(self.other_goals)):
            if node_new == self.other_goals[j]:
                self.other_goals[j] = node_new
                self.path_to_goal[j] = True
                self.update_multi_paths(node_new, j)
        
        

    def saturate(self, v_nearest, v):
        '''
        It creates a new node located exactly one step (or less) in that direction of v_nearest (from kd_tree)
        '''
        dist, theta = self.get_distance_and_angle(v_nearest, v)
        
        dist = min(self.step_len, dist)
        node_new = Node((v_nearest.x + dist * math.cos(theta),
                         v_nearest.y + dist * math.sin(theta)))
        return node_new

    def find_parent(self, v, U):
        # Algorithm 6
        # skip collision check because it is done in "near()"
        costs = [math.sqrt((v.x - u.x)**2 + (v.y - u.y)**2) + u.lmc for u in U]
        if not costs:
            return
        min_idx = int(np.argmin(costs))
        best_u = U[min_idx]
        if not self.utils.is_collision(best_u, v):
            v.set_parent(best_u)
            v.lmc = costs[min_idx] + best_u.lmc
        else:
            del U[min_idx]
            self.find_parent(v, U)
        

    def rewire_neighbours(self, v, robots_plan=False):
        #NOTE remove is_feasible_ray in rewire_neighbor
        
        # Algorithm 4
        if v.cost_to_goal - v.lmc > self.epsilon:
            v.cull_neighbors(self.search_radius)
            for u in v.all_in_neighbors() - set([v.parent]):
                if u.lmc > v.distance(u) + v.lmc and \
                        not self.utils.is_collision(u, v) and self.is_feasible_ray(u,v): # added collision check (Julia)
                    # Option B: a finite & consistent v with v.lmc < u.lmc cannot be a descendant
                    # of u, so skip the cycle walk; only check when costs are inf / inconsistent.
                    safe = math.isfinite(v.lmc) and v.cost_to_goal == v.lmc
                    if not safe and u.would_create_cycle(v):
                        continue
                    u.lmc = v.distance(u) + v.lmc
                    u.set_parent(v)
                    if u.cost_to_goal - u.lmc > self.epsilon:
                        self.verify_queue(u) #add to priority queue, if the node is inconsistent

        # NOTE path/multi_path are NOT refreshed here (mid-rewire the tree can be transiently
        # cyclic). They are refreshed once in refresh_paths() after reduce_inconsistency converges.

    def random_node(self, robots_plan=False):
        
        delta = self.utils.delta
        
        # Strongly bias toward pending reset target until it's connected
        if self._pending_reset_target is not None:
            if np.random.random() < 0.5:   # 50% pull toward target; tune as needed
                return Node((self._pending_reset_target.x, self._pending_reset_target.y))

        # if path to goal is not found,
        # returns a node located exactly at the robot’s current position (no randomness) with probability of bot_sample_rate
        
        if (robots_plan) and (not self.robot_path_to_goal) and (np.random.random() < self.bot_sample_rate):
            return Node((self.s_bot.x, self.s_bot.y))
        
        #------------------------
        
        # uniform random Node inside the env space (if HJ contingency is disabled)
        
        if not self.HJ_contingency_enable:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                        np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))
        
        #------------------------
        
        
        else:
            #multi gaussian distribution
            
            # Choose which Gaussian to sample from
            idx = np.random.randint(len(self.safe_regions))
            x, y, _ = self.safe_regions[idx]
            mu = np.array([x, y])

            # Isotropic covariance
            sigma = 5.5
            cov = sigma**2 * np.eye(2)

            sample = np.random.multivariate_normal(mu, cov)
            return Node((sample[0], sample[1]))
        
        
    def _try_connect_pending(self):
        """
        Called each planning iteration when a pending reset target exists.
        Attempts to wire the target into the tree now that more nodes may exist nearby.
        On success, sets s_bot and clears the pending target.
        """
        # Always use current robot position
        target = Node((self.robot_position[0], self.robot_position[1]))
        nearest = self.nearest(target)
        saturated = self.saturate(nearest, target)

        V_near = self.near(saturated)
        if not V_near:
            V_near = [nearest]

        V_near_free = [u for u in V_near if not self.utils.is_collision(u, saturated)]
        if not V_near_free:
            return  # still blocked, keep biasing

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

        print(f"[_try_connect_pending] Pending target connected at "
            f"({self.s_bot.x:.2f}, {self.s_bot.y:.2f}), lmc={self.s_bot.lmc:.3f}")

    def update_gamma(self):
        '''
        computes and updates gamma required for shrinking ball radius
        - gamma depends on the free space volume, so changes when obstacles are added or removed
        - this assumes that obstacles don't overlap
        '''
        
        # TODO check update gamme
        
        mu_X_free = (self.x_range[1] - self.x_range[0]) * (self.y_range[1] - self.y_range[0]) *1/2 #* 1/3
        for (_, _, r) in self.obs_circle:
            mu_X_free -= np.pi * r ** 2
        # for (_, _, w, h) in self.obs_rectangle:
        #     mu_X_free -= w * h

        self.gamma = self.gamma_FOS * (2 * (1 + 1/self.d))**(1/self.d) * (mu_X_free/self.zeta_d)**(1/self.d) # optimality condition from Theorem 38 of RRT* paper

    def shrinking_ball_radius(self):
        '''
        Computes and returns the radius for the shrinking ball
        '''
        return min(self.step_len, self.gamma * np.log(len(self.tree_nodes)+1) / len(self.tree_nodes))

    def near(self, v):
        return self.kd_tree.search_nn_dist((v.x, v.y), self.search_radius)

    def nearest(self, v):
        '''
        This function finds and returns the tree node whose (x,y) position is closest to the query point v.
        '''
        return self.kd_tree.search_nn((v.x, v.y))[0].data

    def refresh_paths(self, robots_plan=False):
        """Recompute the robot path and all multi-goal paths. Call ONLY after the tree is
        consistent (e.g. end of reduce_inconsistency) — never mid-rewire, where parent
        pointers can be transiently cyclic and the path walks could loop forever."""
        if robots_plan:
            self.update_path(self.s_bot)
        for j, goal_j in enumerate(self.other_goals):
            self.update_multi_paths(goal_j, j)

    def update_path(self, node):
        self.path = []
        self.path_node = []
        seen = set()  # guard against transient parent cycles (mid-rewire)
        while node.parent and node not in seen:
            seen.add(node)
            self.path_node.append(node)
            self.path.append(np.array([[node.x, node.y], [node.parent.x, node.parent.y]]))
            node = node.parent

    def update_multi_paths(self, node, idx):
        self.multi_paths[idx] = []
        self.multi_path_nodes[idx] = []

        seen = set()  # guard against transient parent cycles (mid-rewire)
        while node.parent and node not in seen:
            seen.add(node)
            self.multi_path_nodes[idx].append(node)
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
        
    def is_feasible_ray(self, start:Node, end:Node):
        
        o, d = self.utils.get_ray(start, end)
        
        ''' Use query from HJR-FNO model to check feasibility of the ray (however, expensive beacuse we need spatial size of at least 32)'''
        # #crate batch size of 16 based on start and end nodes
        # # query the value of HJ reachable set from HJR-FNO
        # t_vals = np.linspace(0, 1, 3)
        # theta_array = self.robot_state[2] + np.array([-np.pi/4, 0.0, np.pi/4])

        # positions = o + t_vals[:, None] * d                  # (32, 2)
        # feasible =  self.hjr_fno.is_state_feasible(robot_state= positions, theta_array=theta_array, reachable_set_constraint=self.HJ_contingency_enable)
        # return feasible
        
        #----------------------------------------

        # NOTE old method, using look up table of predicted HJ reachable set
        
        # for t in np.linspace(0,1,3):
            
        #     if not self.hjr_fno.is_feasible(v= (o[0] + t * d[0], o[1] + t * d[1]) , reachable_set_constraint=self.HJ_contingency_enable):
        #         return False

        # return True
        
         #----------------------------------------
    
        t_vals = np.linspace(0, 1, 4)
        positions = o + t_vals[:, None] * d
        # B1: edge heading, applied to every sampled point — only needed by the HJR_sets source.
        thetas = None
        if self.hjr_fno.feasibility_source == "HJR_sets":
            thetas = np.full(positions.shape[0], math.atan2(end.y - start.y, end.x - start.x))
        return bool(np.all(self.hjr_fno.points_feasible(
            positions, thetas=thetas, reachable_set_constraint=self.HJ_contingency_enable)))

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
    
    
    
    
'''
Space Filling Forest (SFF*) Algorithm (modified with RRTX)

Paper: "Multi-Goal Path Planning Using Multiple Random Trees" (Janoš et al., 2021)
'''
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
        HJ_contingency_enable:bool,
        ) -> None:
        
        from HJR_FNO.HJR_FNO3d import HJR_FNO
        
        assert start_goal_index < len(x_goal), "start_goal_index index out of range"
        
        
        #All configs
        self.HJ_contingency_enable = HJ_contingency_enable  #enable contingency constraint in RRTX tree planning
        self.robot_is_isolated = False
        
        self.start_goal_index = start_goal_index
        x_start = x_goal[start_goal_index]
        self.iter_max = iter_max
        
        self.env = env.Env(safe_regions=safe_regions)
        # Single shared obstacle store: Plotting and every RRTX tree use this same Env, so
        # obstacles detected by the active tree (in-place extend of env.obs_circle) are visible
        # to all trees + plotting instead of living in per-tree private Env copies.
        self.plotting = plotting.Plotting(x_start, x_goal, safe_regions=safe_regions, _env=self.env)
        
        #HJR-FNO configs
        self.Tf_reach = 8
        self.hjr_fno = HJR_FNO(env=self.env, safe_regions=safe_regions, Tf_reach=self.Tf_reach)
        self.current_state = [x_start[0], x_start[1], heading]
        self.lidar_range = lidar_range

        # plotting
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.suptitle(f"HJR-FNO Contingency")
        self.ax.set_xlim(self.env.x_range[0], self.env.x_range[1]+1)
        self.ax.set_ylim(self.env.y_range[0], self.env.y_range[1]+1)    
        
        self.show_subplots = False
        
        # Add flags for click handling
        self.waiting_for_first_click = False
        self.waiting_for_second_click = False
        self.contingency_complete = False
        self.resume_planning = False  # Signal to restart the plan_iter loop

        #Define different tree branches rooted at each target location
        self.rrtx_trees = {}
        self.n_tree = len(x_goal)
        q, r = divmod(self.iter_max, self.n_tree)
        iter_list =  [q + 1 if i < r else q for i in range(self.n_tree)]

        self.sub_iter_count = 1000
        for i, target_i in enumerate(x_goal):
            
            other_goals = [g for j, g in enumerate(x_goal) if j != i]
            other_goals_id = [j for j in range(len(x_goal)) if j != i]
            
            self.rrtx_trees[i] = RRTX(
                x_start=x_start,
                x_goal=target_i,
                goal_id = i,
                other_goals = other_goals,
                other_goals_id = other_goals_id,
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
                fig= self.fig,
                ax= self.ax,
                plotting=self.plotting,
                environment=self.env,   # shared obstacle store across all trees + plotting
            )
            
        self.robotState_isSync = [False for _ in range(self.n_tree)]
            
        # Define distance matrix 
        # D_ij means i-th tree distance to j-th goal; if optimal path is not found, D_ij = np.inf
        self.D = np.full((self.n_tree, self.n_tree), np.inf)
        np.fill_diagonal(self.D, 0.0)

        
        #Color map
        cmap = plt.get_cmap("hsv")
        self.colorList = [cmap(i) for i in np.linspace(0, 1, len(x_goal), endpoint=False)]
        
    def on_click(self, event):
        """Handle mouse click events for contingency planning."""
        if event.inaxes != self.ax:
            return
        
        if self.HJ_contingency_enable:
            print("Adversary detected!")
        
            for tree_k in self.rrtx_trees.values():
                tree_k.contingency_triggered = True
        else:
            print("Contingency constraint disabled. No action taken.")
        
    def update_distance_matrix(self, sequence_visited, robot_position_costs: dict = None):
        """
        Build distance matrix for ATSP over unvisited goals only,
        plus one extra row/col representing the robot's CURRENT position.

        D[i][j] = cost to travel FROM node_i TO node_j.

        Since tree_b is rooted at goal_b, the cost from goal_a → goal_b
        is found inside tree_b as the cost_to_goal of the node at goal_a's location.

        Parameters
        ----------
        sequence_visited : List[int]
            Tree IDs already visited (in order). Used to exclude from optimization.

        robot_position_costs : dict, optional
            {tree_id: cost} representing the cost from the robot's CURRENT physical
            position to each unvisited goal_id, as reported by each tree's s_bot.
            If None, the robot row is omitted (goal-to-goal matrix only).
        """

        visited_set = set(sequence_visited)

        # Unvisited goals only — these are what Held-Karp optimizes over
        unvisited = [tid for tid in self.rrtx_trees.keys() if tid not in visited_set]

        # If robot position is provided as a separate node, prepend a virtual
        # "robot" row/col at index 0
        include_robot = robot_position_costs is not None
        nodes = (["robot"] + unvisited) if include_robot else unvisited

        m = len(nodes)
        idx_map = {node: k for k, node in enumerate(nodes)}
        D_pat = np.full((m, m), np.inf)

        for a in nodes:
            for b in nodes:
                ia, ib = idx_map[a], idx_map[b]

                # Self-distance
                if a == b:
                    D_pat[ia, ib] = 0.0
                    continue

                # Robot row: cost from robot's current position to goal_b
                if a == "robot":
                    if b in robot_position_costs:
                        D_pat[ia, ib] = robot_position_costs[b]
                    # robot col (b == "robot") stays inf — robot is not a destination
                    continue

                if b == "robot":
                    # Cannot travel TO the robot's position as a goal
                    D_pat[ia, ib] = np.inf
                    continue

                # Goal-to-goal: cost from goal_a → goal_b
                # Look inside tree_b for the node at goal_a's location
                tree_b = self.rrtx_trees[b]
                if a in tree_b.other_goals_id:
                    local_j = tree_b.other_goals_id.index(a)
                    cost = tree_b.other_goals[local_j].cost_to_goal
                    D_pat[ia, ib] = cost  # may be inf if path is broken by obstacles

        self.D = D_pat
        self.D_idx_map = idx_map
        self.D_nodes = nodes  # ["robot", unvisited_id_1, unvisited_id_2, ...]

            

    def print_distance_matrix(self, precision=3):

        # Map matrix indices to readable labels
        labels = ["robot" if node == "robot" else f"goal_{node}" 
                for node in self.D_nodes]
        n = len(labels)

        # Header
        header = "          " + "".join(f"{lbl:^12}" for lbl in labels)
        print(header)
        print("          " + "-" * (12 * n))

        # Rows
        for i, row_lbl in enumerate(labels):
            row_str = f"{row_lbl:>8} |"
            for j in range(n):
                val = self.D[i, j]
                if np.isinf(val):
                    row_str += f"{'∞':^12}"
                else:
                    row_str += f"{val:^12.{precision}f}"
            print(row_str)

    
    def solve_atsp_held_karp(self, distance_matrix, start=[0], hamiltonian_cycle=True):
        """
        Solve an Asymmetric TSP directly from a distance matrix using Simulated Annealing.

        Returns
        -------
        solution, cost
        """
        
        if type(start) == int:
            start = [start]
        assert type(start) == list, "start must be an integer or a list of integers"

        return atsp.held_karp(cost=distance_matrix.tolist() , prefix=start, hamiltonian_cycle=hamiltonian_cycle)
    
    def compute_tour_distance(
        self,
        tour,
        prev_id=None,
        curr_id=None,
        traversed_distance=0.0,
    ):
        """
        Compute total tour distance.
        If (prev_id -> curr_id) matches the robot's current motion,
        use traversed_distance + s_bot.cost_to_goal instead of tree distances.
        """

        total_cost = 0.0

        for i, j in zip(tour[:-1], tour[1:]):

            # skip degenerate self-loop
            if i == j:
                continue

            # ---- SPECIAL CASE: robot currently moving from prev_id -> curr_id ----
            if prev_id is not None and curr_id is not None:
                if i == prev_id and j == curr_id:
                    remaining = self.rrtx_trees[j].s_bot.cost_to_goal
                    if np.isinf(remaining):
                        return np.inf
                    total_cost += traversed_distance + remaining
                    continue

            # ---- DEFAULT CASE: use precomputed tree distances ----
            tree_i = self.rrtx_trees[i]

            if j not in tree_i.other_goals_id:
                return np.inf  # unreachable

            k = tree_i.other_goals_id.index(j)
            d = tree_i.other_goals[k].cost_to_goal

            if np.isinf(d):
                return np.inf

            total_cost += d

        return total_cost


    
    def rotate_tour(self, tour, start_id):
        """
        Rotate a cyclic ATSP tour so that it starts and ends at start_id.

        Example:
            tour      = [2, 3, 4, 0, 1, 2]
            start_id = 3
            result   = [3, 4, 0, 1, 2, 3]
        """
        if start_id not in tour:
            raise ValueError(f"Tour does not contain start_id={start_id}")

        # Remove repeated last element if tour is already closed
        if tour[0] == tour[-1]:
            tour = tour[:-1]

        idx = tour.index(start_id)
        rotated = tour[idx:] + tour[:idx] + [start_id]
        return rotated
        
    def init_trees(self, showPlot=True):        

        for i in range(self.n_tree):
            
            print(f"Initialize the tree {i}")
        
            self.rrtx_trees[i].planning()
            
            if showPlot:
                # ================= PLOTTING =======================
                                    
                # clear axes
                self.ax.clear()
                
                # restore static axis properties
                self.ax.set_xlim(self.env.x_range[0], self.env.x_range[1] + 1)
                self.ax.set_ylim(self.env.y_range[0], self.env.y_range[1] + 1)

                # draw environment
                self.plotting.plot_env(self.ax, colorList=self.colorList)

                for i, tree_i in self.rrtx_trees.items():
                    
                    # draw tree nodes
                    if tree_i.all_nodes_coor:
                        nodes = np.array(tree_i.all_nodes_coor)
                        self.ax.scatter(nodes[:, 0], nodes[:, 1], s=4, c='gray', alpha=0.5)
                        
                    #get all edges
                    self.edges = []
                    for node in tree_i.tree_nodes:
                        if node.parent:
                            self.edges.append(np.array([[node.parent.x, node.parent.y], [node.x, node.y]]))
            
                    # draw tree edges
                    if self.edges:
                        edge_col = LineCollection(self.edges, colors=self.colorList[i], linewidths=0.5, alpha=0.2)
                        self.ax.add_collection(edge_col)

                    # draw path to goal
                    for j, goal_j in enumerate(tree_i.other_goals):
                        if tree_i.path_to_goal[j]:
                            path_col = LineCollection(tree_i.multi_paths[j], colors=self.colorList[i], linewidths=2.5)
                            self.ax.add_collection(path_col)
                    
                    # plot reachable set at current heading
                    if self.HJ_contingency_enable:
                        self.plotting.plot_reachable_set(self.ax, self.hjr_fno, theta=tree_i.robot_state[2], time=tree_i.Tf_reach)

                    # force redraw
                    plt.pause(0.001)
                
                    # ================= END OF PLOTTING =======================
                    
        # # -------------------------------------------------
        # # show intial plot (for K=2 goals case)
        # # -------------------------------------------------
        
        # self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        # # restore static axis properties
        # self.ax.set_xlim(self.env.x_range[0], self.env.x_range[1] + 1)
        # self.ax.set_ylim(self.env.y_range[0], self.env.y_range[1] + 1)

        # # draw environment
        # self.plotting.plot_env(self.ax, colorList=None)
        
            
        # if self.HJ_contingency_enable:
        #     self.plotting.plot_reachable_set(self.ax, self.hjr_fno, theta=0, time=self.hjr_fno.Tf_reach)
        
        
        # #=====
        # #Extra D_local, and inscribed ball B_delta
        # #=====
        
        
        # side_length = 20
        # half = side_length / 2.0
        
        # square = patches.Rectangle(
        #     (self.hjr_fno.safe_regions[0][0] - half, self.hjr_fno.safe_regions[0][1] - half),   # bottom-left corner
        #     side_length,
        #     side_length,
        #     linewidth=1,
        #     edgecolor='blue',
        #     facecolor='none'        # no fill
        # )
        
        # #plot goal and robot 
        # self.plotting.plot_robot(self.ax, self.plotting.xG[0], lidar_range=0, plot_lidar=False)
        
        # self.ax.scatter(
        #     self.plotting.xG[-1][0], self.plotting.xG[-1][1],
        #     marker='*',
        #     s=300,                    # size (adjust as needed)
        #     c='red',         # face color
        #     edgecolors='black',       # outline color
        #     linewidths=1.5,
        #     zorder=10
        # )
        
        # self.ax.add_patch(square)
        
        # for i in range(len(self.hjr_fno.safe_regions) - 1):
        #     j = i +1
            
        #     # Define local grid once
        #     Nx, Ny = 50, 50
        #     x_local = np.linspace(-10, 10, Nx)
        #     y_local = np.linspace(-10, 10, Ny)

        #     delta = 1.5

        #     valid, ball_center = self.check_delta_clear_overlap(
        #         self.hjr_fno.feasible_region[i],
        #         self.hjr_fno.feasible_region[j],
        #         self.hjr_fno.safe_regions[i],
        #         self.hjr_fno.safe_regions[j],
        #         x_local,
        #         y_local,
        #         delta
        #     )

        #     if valid:
        #         print("δ-clear overlap exists at:", ball_center)

                
        #         circle = patches.Circle(
        #             ball_center,
        #             delta,
        #             facecolor='orange',
        #             edgecolor='orange',
        #             alpha=0.5
        #         )
        #         self.ax.add_patch(circle)
        #     else:
        #         print("No δ-clear overlap exists")
                        
                        
        # #plot true reachable set
        
        # theta_slice = 0
        # time_slice = np.argmin(np.abs(self.hjr_fno.time_array_fine - self.hjr_fno.Tf_reach))
        # reachable_set_slice = self.hjr_fno.true_reach_obsFree[..., theta_slice, time_slice]
        
        # for i in range(len(self.hjr_fno.safe_regions)):
            
        #     CS = self.ax.contour(
        #         self.hjr_fno.X_fine + self.hjr_fno.safe_regions[i][0],
        #         self.hjr_fno.Y_fine + self.hjr_fno.safe_regions[i][1],
        #         reachable_set_slice,
        #         levels=[0],
        #         colors='green',
        #         linewidths=2,
        #         linestyles='solid'
        #     )
            
        # # Increase tick label size
        # self.ax.tick_params(axis='both', labelsize=25)
        
        # plt.show()


    def debug_plot_reachable_constraint(self, state, tag=""):
        """Visualize the feasibility constraint actually enforced by is_feasible() at `state`.
        For each safe region it shows the feasible sublevel set ({V <= safe_margin} for
        obstacle regions at the robot heading via Option B; {V_init <= 0} for obstacle-free
        regions), the robot position, and the robot's interpolated V. An EMPTY fill means
        the constraint is too strict (e.g. scenario delta_hat pushed safe_margin too low)."""
        import torch
        hjr = self.hjr_fno
        x_r, y_r = float(state[0]), float(state[1])
        theta = float(state[2]) if len(state) > 2 else 0.0

        fig, ax = plt.subplots(figsize=(8, 8))
        self.plotting.plot_env(ax)  # obstacles + boundary

        for i in range(hjr.num_safe_regions):
            cx, cy = hjr.safe_regions[i][:2]
            # Always plot HJR_sets (the constraint actually enforced) at the safe_margin[i] sublevel.
            reach = hjr.HJR_sets[i]
            if torch.is_tensor(reach):
                reach = reach.cpu().numpy()
            level = 0 #hjr.safe_margin[i]
            # obstacle-free regions hold the fine-grid precomputed tube ([0,2pi)); re-predicted
            # obstacle regions hold the coarse FNO grid ([-pi,pi)). Pick grid/axes accordingly.
            if not hjr.obs_list[i]:
                th_q = float(hjr._wrap_to_grid_theta(theta, hjr.g_fine))
                theta_slice = int(np.argmin(np.abs(hjr.theta_array_fine - th_q)))
                Tf_slice = hjr._grown_time_index(hjr.time_array_fine)  # index 0 = fully grown
                X, Y = hjr.X_fine + cx, hjr.Y_fine + cy
            else:
                th_q = float(hjr._wrap_to_grid_theta(theta, hjr.g))
                theta_slice = int(np.argmin(np.abs(hjr.theta_array - th_q)))
                Tf_slice = hjr._grown_time_index(hjr.time_array)  # index 0 = fully grown
                X, Y = hjr.X + cx, hjr.Y + cy
            Z = np.asarray(reach[..., theta_slice, Tf_slice])

            # filled feasible area + boundary; flag empty sublevel sets in red
            if level > Z.min():
                # ax.contourf(X, Y, Z, levels=[Z.min(), level], colors='#ADD8E6', alpha=0.4)
                ax.contour(X, Y, Z, levels=[level], colors='#191970', linewidths=2)
                ax.text(cx, cy, f"r{i}\nm={level:.2f}", color='navy', fontsize=8, ha='center')
            else:
                ax.text(cx, cy, f"r{i} EMPTY\nmin={Z.min():.2f} > m={level:.2f}",
                        color='red', fontsize=8, ha='center', weight='bold')

        # robot marker + the exact value/threshold the feasibility gate sees
        v_robot = hjr.feasibility_values(np.atleast_2d([x_r, y_r]), thetas=np.array([theta]))[0]
        feasible = hjr.is_feasible(np.atleast_2d([x_r, y_r]), thetas=np.array([theta]))
        ax.plot(x_r, y_r, 'r*', markersize=18, label=f"robot V={v_robot:.3f} feasible={feasible}")
        ax.set_title(f"Reachable-set constraint @ theta={theta:.2f} rad  {tag}")
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        plt.show()

    def planning(self, hamiltonian_cycle=False, showPlot=True):
        
        '''
        Planning using TSP-RRTX with Contingency Handling
        
        if hamiltonian_cycle = True, the tour will return to the starting target
        otherwise, it will only visit each target once
        '''
        
        
        #-----------------
        # Set robot states
        # ----------------

        #Set robot position to the first target location of the routing sequence
        heading = 0.0
        prev_heading = heading
        self.current_state = [self.rrtx_trees[self.start_goal_index].s_goal.x, self.rrtx_trees[self.start_goal_index].s_goal.y, heading]
        
        # Simulate Robot's plan
        print("\nInitilize robot position for each tree")
        
        for i in range(self.n_tree):
            
            # self.rrtx_trees[i].reset_robot_v2(current_state=self.current_state)        
            connected = self.rrtx_trees[i].reset_robot_position((self.current_state[0], self.current_state[1]), heading=None)    
            self.rrtx_trees[i].update_robot_heading()
            
            self.robotState_isSync[i] = connected
            
            self.rrtx_trees[i].prob_q = 0.9
            
            
        #-----------------
        # Solve initial TSP tour
        # ----------------
        
        '''
        Held-Karp Algorithm
        '''
        sequence_visited = [self.start_goal_index]
        sequence_to_visit = []        
        
        # At call site in planning():
        robot_position_costs = {
            tid: self.rrtx_trees[tid].s_bot.cost_to_goal
            for tid in range(self.n_tree)
            if tid not in set(sequence_visited)
        }
        
        print(f"robot_position_costs: {robot_position_costs}")

        if hamiltonian_cycle:
            
            self.update_distance_matrix(
                sequence_visited=[],
                robot_position_costs=None
            )

            start_idx = self.D_idx_map[self.start_goal_index]  # correct index of goal 0

            min_cost, tour_indices = atsp.held_karp(
                self.D.tolist(),
                prefix=[start_idx],
                hamiltonian_cycle=hamiltonian_cycle
            )
            optimal_tour = [self.D_nodes[i] for i in tour_indices[:-1]]  # strip trailing repeat

        else:
            self.update_distance_matrix(
                sequence_visited=sequence_visited,
                robot_position_costs=robot_position_costs
            )
            
            # Held-Karp starts from index 0 (the robot node)
            min_cost, tour_indices = atsp.held_karp(
                self.D.tolist(),
                prefix=[0],  # 0 = robot's current position
                hamiltonian_cycle=hamiltonian_cycle
            )
            
            # if hamiltonian_cycle:
            #     # Strip leading "robot", but re-append start_goal_index at the end
            #     optimal_tour = [self.D_nodes[i] for i in tour_indices 
            #                     if self.D_nodes[i] != "robot"]
            #     optimal_tour.append(self.start_goal_index)
            # else:
            optimal_tour = [self.D_nodes[i] for i in tour_indices 
                            if self.D_nodes[i] != "robot"]

        print(f"Optimal tour: {[self.start_goal_index] + optimal_tour}")
        
        
        # Before the outer for loop, initialize:
        prev_id = self.start_goal_index
        
        if hamiltonian_cycle:
            original_tour = optimal_tour.copy() 
        else:
            original_tour = [self.start_goal_index] + optimal_tour.copy() 
            
        optimal_tour = original_tour
        
        sequence_to_visit = optimal_tour.copy()   # [0, 2, 3, 4]
        sequence_visited = [self.start_goal_index]  # [1]
        
        
        #-----------------
        # Configure Mouse-click event (for Contingency planning)
        # ----------------
        
            
        #Add mouse click event handler (resemble adversarial event) which triggers Contingency planning
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click) 
        self.fig.suptitle(f"HJR-FNO Contingency\nOptimal Tour: {optimal_tour}\nVisited: {sequence_visited}\nTo Visit: {sequence_to_visit}")
        prev_plotting = time.time()
            
            
        #-----------------
        # Main Planning loop
        # TSP-RRTX with Contingency Handling
        # ----------------
            
        print("Start Robot's Plan Execution")
                
        
        state_history = []

        # Ensure `id` is always defined, even when the traversal loop below never
        # runs (single-goal case: optimal_tour == [start_goal_index], so there is
        # no leg to traverse). Downstream plotting (e.g. the reachable-set draw)
        # references `id`, so default it to the tour's starting target.
        id = optimal_tour[0] if optimal_tour else self.start_goal_index

        for i in range(1, len(optimal_tour)):
            
            traversed_distance = 0.0
            needs_reset = True  # use flag instead of plan_iter == 0 (fixes the plan_iter=0 bug from before)
                        
            for plan_iter in range(self.iter_max):
                
                id = optimal_tour[i] 
                
                if needs_reset:
                    # Reset active tree
                    connected = self.rrtx_trees[id].reset_robot_position(
                        (self.current_state[0], self.current_state[1]), heading=None)
                    self.rrtx_trees[id].update_robot_heading()
                    self.robotState_isSync[id] = connected
                    needs_reset = False
                    #---------------
                    
                    # --- Check 1: Heading change ---
                    new_heading = self.rrtx_trees[id].robot_state[2]
                    heading_change = abs(utils.Utils.wrap_angle(new_heading - prev_heading))
                    heading_change_deg = math.degrees(heading_change)
                    
                    # --- Check 2: Path feasibility ---
                    path_infeasible = False
                    if self.rrtx_trees[id].path:

                        all_points = []
                        t_vals = np.linspace(0, 1, 3)  # start, mid, end per segment

                        for seg in self.rrtx_trees[id].path:
                            orig  = seg[0]  # [x0, y0]
                            direc = seg[1] - seg[0]  # [dx, dy]
                            # interpolated points along this segment: shape (3, 2)
                            points = orig + t_vals[:, None] * direc
                            all_points.append(points)

                        # Stack all into one
                        waypoints = np.unique(np.vstack(all_points), axis=0)

                        path_infeasible = not self.hjr_fno.is_feasible(
                            v=waypoints,
                            reachable_set_constraint=self.HJ_contingency_enable
                        )

                        if path_infeasible:
                            print(f"[Path check] New path for tree {id} is infeasible "
                                f"— treads over constrained region.")

                    # --- Trigger orphan if either condition met ---

                    HEADING_THRESHOLD_DEG = 120.0
                    if heading_change_deg > HEADING_THRESHOLD_DEG or path_infeasible:
                        print(f"[Heading check] Large heading change detected: "
                            f"{heading_change_deg:.1f} deg > {HEADING_THRESHOLD_DEG} deg. "
                            f"Orphaning s_bot of tree {id} to trigger rewire.")

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
                    #---------------

                    # Immediately sync ALL other non-visited trees to current state
                    visited_set = set(sequence_visited)
                    for k, tree_k in self.rrtx_trees.items():
                        if k == id \
                        or k in visited_set \
                        or (k == self.start_goal_index and not hamiltonian_cycle):
                            continue
                        connected_k = tree_k.reset_robot_position(
                            (self.current_state[0], self.current_state[1]), heading=None)
                        tree_k.update_robot_heading()
                        self.robotState_isSync[k] = connected_k
                        
                        #If the robot's position cannot be sync up
                        if not connected_k:
                          
                            # Too far or no path — orphan s_bot so cost_to_goal becomes inf naturally
                            if tree_k.s_bot is not None and tree_k.s_bot.parent is not None:
                                tree_k.verify_orphan(tree_k.s_bot)
                                tree_k.propagate_descendants(robots_plan=True)

                            tree_k.robot_position = [self.current_state[0], self.current_state[1]]
                            tree_k.robot_state = [self.current_state[0], self.current_state[1],
                                                tree_k.robot_state[2]]
                            tree_k.robot_path_to_goal = False
                            tree_k.path = []
                            tree_k._pending_reset_target = Node((self.current_state[0],
                                                                self.current_state[1]))
                            tree_k._pending_reset_heading = tree_k.robot_state[2]
                        # print(f"[Sync failed] Tree {k} — nearest too far "
                        #     f"(dist={dist_to_nearest:.2f}) or no path. "
                        #     f"s_bot orphaned, cost_to_goal=inf.")

                    print(f"\n{'='*50}")
                    print(f"Starting planning from Target #{prev_id} to Target #{id}")
                    print(f"{'='*50}\n")
                                                    
                #Update robot position
                new_obs, new_obs_flag, distance_moved = self.rrtx_trees[id].planning_with_robot(steps=10)
                traversed_distance += distance_moved   # uncomment this
                self.current_state = self.rrtx_trees[id].robot_state
                state_history.append(self.current_state.copy() + [id])
                
                if not self.rrtx_trees[id].robot_path_to_goal and (plan_iter % 3 == 0):
                    print("robot's position", (self.rrtx_trees[id].s_bot.x, self.rrtx_trees[id].s_bot.y))
                    print("is feasible?", self.hjr_fno.is_feasible(v=np.atleast_2d(self.current_state[:2])))
                    print("robot's Path to goal", self.rrtx_trees[id].robot_path_to_goal)
                    print("Robot's cost to goal" , self.rrtx_trees[id].s_bot.cost_to_goal)
                    print("Robot's LMC cost" , self.rrtx_trees[id].s_bot.lmc)
                    print("Path List", self.rrtx_trees[id].path)
                    print("Pending Target", self.rrtx_trees[id]._pending_reset_target)

                    # DEBUG: visualize the reachable-set feasibility constraint at the robot's state
                    # self.debug_plot_reachable_constraint(self.current_state, tag=f"tree {id}")

                    
                '''
                Handling contingency during robot's plan execution
                '''
                if self.rrtx_trees[id].contingency_triggered and self.HJ_contingency_enable:
                    
                    detected_obs_during_contingency, contingency_trajectory, _, _, _, _, _ = self.hjr_fno.contingency_policy(
                            self.current_state, 
                            self.plotting, 
                            self.fig, 
                            self.ax
                    )
                    state_history.extend([list(s) + [id] for s in contingency_trajectory.tolist()])
                    
                    #Update state and traversed distance so far
                    self.current_state = contingency_trajectory[-1]
                    
                    print("Position after contingency", self.current_state)
                    
                    for traj_i in range(len(contingency_trajectory) - 1):
                        x0, y0, _ = contingency_trajectory[traj_i]
                        x1, y1, _ = contingency_trajectory[traj_i + 1]
                        traversed_distance += np.hypot(x1 - x0, y1 - y0)
                    
                    if len(detected_obs_during_contingency) > 0:
                        new_obs += detected_obs_during_contingency
                        new_obs_flag = True
                        
                        #For the current tree, update all obstacles detected during contingency at once
                        self.rrtx_trees[id].update_obstacles(detected_obs_during_contingency, robots_plan=True)
                            
                    if len(contingency_trajectory) > 1:
                        #Reset robot position in the current tree                                    
                        # self.rrtx_trees[id].reset_robot_v2(current_state=self.current_state)
                        connected = self.rrtx_trees[id].reset_robot_position((self.current_state[0], self.current_state[1]), heading=None)   
                        self.rrtx_trees[id].update_robot_heading()
                        
                        self.robotState_isSync[id] = connected
                            
                    #Reset the contingency trigger flag
                    for tree_k in self.rrtx_trees.values():
                        tree_k.contingency_triggered = False
                        
                    # Re-evaluate isolation status AFTER reset_robot_position
                    # since the contingency trajectory may have moved robot to a reachable position
                    visited_set = set(sequence_visited)
                    self.robot_is_isolated = all(
                        np.isinf(self.rrtx_trees[tid].s_bot.cost_to_goal)
                        for tid in range(self.n_tree)
                        if tid not in visited_set
                    )
                        
                        
                    '''
                    After Contingency, if the robot still can't reach anywhere, keep re-planning for every remaining tree
                    '''
                    if self.robot_is_isolated:
                        print("\n[INFO] Planning for all unvisited trees until paths are restored...")

                        unvisited_set = set(sequence_to_visit)

                        # Keep planning until all unvisited trees have robot_path_to_goal = True
                        recovery_iter = 0
                        while not all(self.rrtx_trees[tid].robot_path_to_goal for tid in unvisited_set):

                            for tid in unvisited_set:
                                if not self.rrtx_trees[tid].robot_path_to_goal:
                                    self.rrtx_trees[tid].planning(iter_max=100, robots_plan=True)

                                    # Update flag
                                    self.rrtx_trees[tid].robot_path_to_goal = (
                                        self.rrtx_trees[tid].s_bot.cost_to_goal < np.inf
                                    )

                            recovery_iter += 1

                            if recovery_iter % 10 == 0:
                                print(f"[Recovery iter {recovery_iter}] Path status:")
                                for tid in unvisited_set:
                                    print(f"  Tree {tid}: robot_path_to_goal = "
                                        f"{self.rrtx_trees[tid].robot_path_to_goal}, "
                                        f"cost = {self.rrtx_trees[tid].s_bot.cost_to_goal:.3f}")

                            # Safety cutoff to avoid infinite loop
                            if recovery_iter >= 100:
                                print("[WARNING] Recovery iteration limit reached. "
                                    "Some trees may still be isolated.")
                                break

                        # Check which trees recovered
                        recovered = [tid for tid in unvisited_set
                                    if self.rrtx_trees[tid].robot_path_to_goal]
                        still_isolated = [tid for tid in unvisited_set
                                        if not self.rrtx_trees[tid].robot_path_to_goal]

                        print(f"\n[Recovery complete]")
                        print(f"  Recovered trees:      {recovered}")
                        print(f"  Still isolated trees: {still_isolated}")

                        if recovered:
                            self.robot_is_isolated = False
                            needs_reset = True  # re-trigger reset with updated paths
                    
                '''
                1. Always update robot state within other trees
                 
                2. If any new obstacles were detected during plan execution/contingency planning,
                    repair every other tree's graph against them so optimal routing stays valid.
                '''
                updateObs_time_start = time.time()

                for k, tree_k in self.rrtx_trees.items():

                    # NOTE: skip rewiring RRTX tree if
                    # 1. k is the current active tree (already updated)
                    # 2. k is already visited (excluding the starting goal, in case we want a hamiltonian cycle)
                    # 3. k is the starting goal, and we don't need to return back to the start
                    if k == id \
                    or (len(sequence_visited) > 1 and k in sequence_visited[1:]) \
                    or (k == self.start_goal_index and not hamiltonian_cycle):
                        continue

                    connected = tree_k.reset_robot_position(
                        (self.current_state[0], self.current_state[1]), heading=None)
                    tree_k.update_robot_heading()
                    self.robotState_isSync[k] = connected

                    # If there is new obstacle found, optimal routing might change.
                    # Repair this tree's graph against the new obstacles. 
                    # 
                    # NOTE The obstacle store is shared, so record=False (no re-recording); this only
                    # invalidates this tree's own edges + rewires. reduce_inconsistency
                    # (inside update_obstacles) refreshes the tree's path/multi_paths
                    # via refresh_paths, so we only set flags.
                    if new_obs_flag:
                        # ----------------- Graph repair in other trees (shared store) -----------------
                        tree_k.update_obstacles(new_obs, robots_plan=True, record=False)

                        # paths/multi_paths already refreshed by refresh_paths(); just update flags
                        tree_k.robot_path_to_goal = (
                            tree_k.s_bot is not None and tree_k.s_bot.cost_to_goal < np.inf
                        )
                        if not tree_k.robot_path_to_goal:
                            tree_k.path = []
                        for j, goal_j in enumerate(tree_k.other_goals):
                            tree_k.path_to_goal[j] = goal_j.cost_to_goal < np.inf
                        # ----------------- End of graph repair in other trees -----------------

                if new_obs_flag:
                    print(f"\nRewire other trees: {time.time() - updateObs_time_start} s\n")
                        

                        
                    # ----------------- Solve ATSP again -----------------
                    
                    remaining_count = len(sequence_to_visit)
                    
                    #Held-Karp needs at least 2 targets to optimize (excluding the final target of the tour that we must return back to)
                    should_replan = (
                        not self.rrtx_trees[id].robot_path_to_goal
                    ) or (
                        hamiltonian_cycle and remaining_count >= 2
                    ) or (
                        not hamiltonian_cycle and remaining_count >= 2
                    )

                    if should_replan:                    
                    
                        
                        '''
                        Held-Karp Algorithm
                        '''
                        
                        visited_set = set(sequence_visited)

                        # Cost from robot's current position to each unvisited goal
                        robot_position_costs = {
                            tid: self.rrtx_trees[tid].s_bot.cost_to_goal
                            for tid in range(self.n_tree)
                            if tid not in visited_set
                        }
                        
                        # Check if robot is completely isolated
                        all_paths_broken = all(
                            np.isinf(cost)
                            for cost in robot_position_costs.values()
                        )

                        if all_paths_broken:
                            print("\n[WARNING] Robot is isolated — no path exists to any remaining goal.")
                            self.robot_is_isolated = True
                            
                            # Trigger contingency for all trees with remaining unvisited goals
                            for tid in range(self.n_tree):
                                if tid not in visited_set:
                                    self.rrtx_trees[tid].contingency_triggered = True

                        else:
                            self.robot_is_isolated = False

                            self.update_distance_matrix(
                                sequence_visited=sequence_visited,
                                robot_position_costs=robot_position_costs
                            )
                            self.print_distance_matrix(precision=3)


                            #COMPUTE OPTIMAL TOUR
                            min_cost, tour_indices = atsp.held_karp(
                                self.D.tolist(),
                                prefix=[0],  # 0 = robot's current position
                                hamiltonian_cycle=hamiltonian_cycle
                            )

                            if tour_indices:
                                new_tour_remaining = [self.D_nodes[i] for i in tour_indices
                                                    if self.D_nodes[i] != "robot"]
                                if hamiltonian_cycle:
                                    new_tour_remaining.append(self.start_goal_index)

                                # Reconstruct full tour = visited prefix + remaining sequence
                                # This ensures len(new_tour) == len(optimal_tour) always
                                new_tour = sequence_visited + new_tour_remaining

                                if new_tour_remaining != sequence_to_visit:  # compare only the remaining part
                                    print("\nNew optimal tour found after replanning due to new obstacles!")
                                    optimal_tour = new_tour
                                    sequence_to_visit = new_tour_remaining
                                    
                                    # Only reset if the immediate next target changed
                                    new_next_id = sequence_to_visit[0] if sequence_to_visit else None
                                    current_next_id = id  # current active target

                                    if new_next_id != current_next_id:
                                        print(f"Next target changed: {current_next_id} → {new_next_id}")
                                        needs_reset = True
                                        traversed_distance = 0.0
                                    else:
                                        print(f"Next target unchanged ({current_next_id}), no reset needed.")

                                else:
                                    print("Tour unchanged after replanning.")
                            else:
                                print("Held-Karp failed to find a new optimal tour.")
                                self.robot_is_isolated = True
                                
                                # Trigger contingency for all trees with remaining unvisited goals
                                for tid in range(self.n_tree):
                                    if tid not in visited_set:
                                        self.rrtx_trees[tid].contingency_triggered = True
                            
                        #-------------- END OF ALL_PATH_BROKEN    
                        

                    else:
                        # Not enough targets left to replan — just print current matrix
                        visited_set = set(sequence_visited)
                        robot_position_costs = {
                            tid: self.rrtx_trees[tid].s_bot.cost_to_goal
                            for tid in range(self.n_tree)
                            if tid not in visited_set
                        }
                        self.update_distance_matrix(
                            sequence_visited=sequence_visited,
                            robot_position_costs=robot_position_costs
                        )
                        self.print_distance_matrix(precision=3)
                        
                    #------------------ END OF SHOULD_REPLAN
                    
                    
                    print(f"\nNew Optimal Tour: {sequence_visited + sequence_to_visit}")
                    print(f"Targets visited:   {sequence_visited}")
                    print(f"Remaining to visit: {sequence_to_visit}")
                    print(f"\nOriginal Tour: {original_tour}")

                    # ----------------- End of solve ATSP  ---------------
                    
                        
                    
                
                # only update the plot at 5 Hz
                elapsed_plotting = time.time() - prev_plotting
                if elapsed_plotting >= 0.2 and showPlot:
                    prev_plotting = time.time()
                
                    # ========================= PLOTTING =======================
                                                
                    # clear axes
                    self.ax.clear()
                    
                    self.fig.suptitle(f"HJR-FNO Contingency\nOptimal Tour: {optimal_tour}\nVisited: {sequence_visited}\nTo Visit: {sequence_to_visit}")


                    # restore static axis properties
                    self.ax.set_xlim(self.env.x_range[0], self.env.x_range[1] + 1)
                    self.ax.set_ylim(self.env.y_range[0], self.env.y_range[1] + 1)

                    # draw environment
                    self.plotting.plot_env(self.ax, colorList=self.colorList)

                    # draw tree nodes
                    if self.rrtx_trees[id].all_nodes_coor:
                        nodes = np.array(self.rrtx_trees[id].all_nodes_coor)
                        self.ax.scatter(nodes[:, 0], nodes[:, 1], s=4, c='gray', alpha=0.5)
                        
                    #get all edges
                    self.edges = []
                    for node in self.rrtx_trees[id].tree_nodes:
                        if node.parent:
                            self.edges.append(np.array([[node.parent.x, node.parent.y], [node.x, node.y]]))

                    # draw tree edges
                    if self.edges:
                        edge_col = LineCollection(self.edges, colors='blue', linewidths=0.3, alpha=0.45)
                        self.ax.add_collection(edge_col)
                        
                        
                    # draw path: goal to goal
                    for j, goal_j in enumerate(self.rrtx_trees[id].other_goals):
                        if self.rrtx_trees[id].path_to_goal[j]:
                            path_col = LineCollection(self.rrtx_trees[id].multi_paths[j], colors=self.colorList[id], linewidths=2.5, alpha=0.7)
                            self.ax.add_collection(path_col)
                        

                    # draw path: robot to goal
                    if self.rrtx_trees[id].path:
                        path_col = LineCollection(self.rrtx_trees[id].path, colors='black', linewidths=1.5)
                        self.ax.add_collection(path_col)

                    # draw robot + lidar
                    self.plotting.plot_robot(self.ax, self.rrtx_trees[id].robot_position, self.rrtx_trees[id].lidar_range)
                    
                    # plot reachable set at current heading
                    if self.HJ_contingency_enable:
                        self.plotting.plot_reachable_set(self.ax, self.hjr_fno, self.rrtx_trees[id].robot_state[2], self.rrtx_trees[id].Tf_reach)
                    
                    
                    if new_obs_flag and  self.show_subplots:
                                
                        # self.ax.clear()
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.set_xlim(self.env.x_range[0], self.env.x_range[1] + 1)
                        ax.set_ylim(self.env.y_range[0], self.env.y_range[1] + 1)
                        self.plotting.plot_env(ax, colorList=self.colorList)     
                        self.plotting.plot_robot(ax, self.rrtx_trees[id].robot_position, self.rrtx_trees[id].lidar_range)
                        if self.HJ_contingency_enable:
                            self.plotting.plot_reachable_set(ax, self.hjr_fno, self.rrtx_trees[id].robot_state[2], self.rrtx_trees[id].Tf_reach)
                                            
                        # draw path to goal
                        if self.rrtx_trees[id].robot_path_to_goal:
                            path_col = LineCollection(self.rrtx_trees[id].path, colors='k', linewidths=3, alpha=0.7)
                            ax.add_collection(path_col)
                                    

                        
                    # force redraw
                    plt.pause(0.001)
                    
                    # ================= END OF PLOTTING =======================
                    
                if new_obs_flag:
                    
                    #Print new distance matrix between each goals after replanning
                    # print("\n")
                    # print("Global location:", self.current_state[:2])
                    # for k, tree_k in self.rrtx_trees.items():
                    #     print(f"Tree {k} robot's cost to goal: {tree_k.s_bot.cost_to_goal}, Reachable? {tree_k.robot_path_to_goal}")
                    #     print(f"- Robot's Position of inside the tree: {(tree_k.s_bot.x, tree_k.s_bot.y)}")
                    #     print(f"- current search radius: {tree_k.search_radius}")
                    #     print(f"- RobotState_isSync: {self.robotState_isSync[k]}")
                    #     # print(f"--- Number of nodes:", len(tree_k.tree_nodes))
                    # print("\n")
                    
                    if  self.show_subplots:
                        plt.show()
                        self.fig, self.ax = plt.subplots(figsize=(8, 8))
                    
                #Terminate when reach the goals
                if self.rrtx_trees[id].s_bot.cost_to_goal == 0.0 and self.rrtx_trees[id].s_bot.lmc == 0.0:
                    print("Successfully reach the goal!")
                    self.current_state = self.rrtx_trees[id].robot_state
                    break
                
                # At the end of each plan_iter, store heading
                prev_heading = self.current_state[2]
                
                    
            ###### END OF PLANNING LOOP FROM prev_id TO id ######
                
            prev_id = id
            self.current_state = self.rrtx_trees[id].robot_state
            heading = self.rrtx_trees[id].robot_state[2]
            sequence_visited.append(sequence_to_visit.pop(0))  #remove the reached target from the sequence
            
        ###### END OF FOR LOOP THROUGH OPTIMAL TOUR ######
        
        
        print('\Final Tour (target_id): ' , optimal_tour)
        print('Final Tour cost', self.compute_tour_distance(optimal_tour))
        
        print('\nOriginal Tour (target_id): ' , original_tour)
        print('Original Tour cost', self.compute_tour_distance(original_tour))

        print("Tour Completed!")

        # Single-goal case: the traversal loop never appended any state (the robot
        # already starts at the only goal). Seed the history with the current state
        # so the downstream np.vstack / plotting / return value stay well-defined.
        if not state_history:
            state_history.append(list(self.current_state) + [self.start_goal_index])

        for i in range(len(self.hjr_fno.safe_regions)):
            obs = np.array(self.hjr_fno.obs_list[i])        # shape (N, 3)

            # No obstacles detected in this region → np.array([]) is 1-D, so the
            # (:, 0) indexing below would fail. Skip empty regions.
            if obs.ndim != 2 or obs.shape[0] == 0:
                continue

            xs, ys = self.hjr_fno.safe_regions[i][:2]

            obs_local = obs.copy()
            obs_local[:, 0] -= xs
            obs_local[:, 1] -= ys

            obs_local = obs_local.tolist()       # convert back to list if needed
            print(obs_local)
        
        # -------------------------------------------------
        # show final plot
        # -------------------------------------------------
        
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        # restore static axis properties
        self.ax.set_xlim(self.env.x_range[0], self.env.x_range[1] + 1)
        self.ax.set_ylim(self.env.y_range[0], self.env.y_range[1] + 1)

        # draw environment
        self.plotting.plot_env(self.ax, colorList=None)
        
        data = np.vstack(state_history)
        x_traj = data[:, 0]
        y_traj = data[:, 1]
        goal_ids = data[:, 3].astype(int)
        final_goal_id = goal_ids[-1]

        self.ax.scatter(
            self.plotting.xG[final_goal_id][0], self.plotting.xG[final_goal_id][1],
            marker='*',
            s=300,                    # size (adjust as needed)
            c='red',         # face color
            edgecolors='black',       # outline color
            linewidths=1.5,
            zorder=10
        )

        
        for i in range(len(x_traj) - 1):

            self.ax.plot(
                [x_traj[i], x_traj[i+1]],
                [y_traj[i], y_traj[i+1]],
                color=self.colorList[goal_ids[i]],
                linewidth=2
            )
            
        if self.HJ_contingency_enable:
            self.plotting.plot_reachable_set(self.ax, self.hjr_fno, self.rrtx_trees[id].robot_state[2], self.rrtx_trees[id].Tf_reach)

        # Optional: plot start/end markers
        self.ax.scatter(x_traj[0], y_traj[0], color='red', s=60, zorder=5)
        self.ax.scatter(x_traj[-1], y_traj[-1], color='red', s=60, zorder=5)
        
        
        
        plt.show()
        
        
        
        return data
        
        
        
    def generate_random_obstacles(self, env, N, r_min=1, r_max=1.5,
                               min_dist_between=3.0,
                               goals=None, min_dist_to_goal=3.0,
                               origin_safe_radius=3.0,
                               start_max_radius=7,
                               start_min_dist_to_obs=2,
                               max_attempts=1000):
        """
        Generate N random circular obstacles and a valid start point (x, y).

        Returns
        -------
        obs_list    : List of [x, y, r]
        start_point : Tuple(x, y) within start_max_radius from origin,
                    clear of all obstacles
        """
        obs_list = []
        attempts = 0

        while len(obs_list) < N and attempts < max_attempts:
            attempts += 1

            x = np.random.uniform(env.x_range[0] + r_max, env.x_range[1] - r_max)
            y = np.random.uniform(env.y_range[0] + r_max, env.y_range[1] - r_max)
            r = np.random.uniform(r_min, r_max)

            if math.hypot(x, y) < origin_safe_radius + r:
                continue

            too_close_to_obs = any(
                math.hypot(x - ox, y - oy) < min_dist_between + r + or_
                for ox, oy, or_ in obs_list
            )
            if too_close_to_obs:
                continue

            if goals is not None:
                too_close_to_goal = any(
                    math.hypot(x - gx, y - gy) < min_dist_to_goal + r
                    for g in goals
                    for gx, gy in [g[:2]]
                )
                if too_close_to_goal:
                    continue

            obs_list.append([x, y, r])

        if len(obs_list) < N:
            print(f"[generate_random_obstacles] Warning: only generated "
                f"{len(obs_list)}/{N} obstacles after {max_attempts} attempts.")

        start_state = None
        attempts = 0
        while attempts < max_attempts:
            attempts += 1

            angle  = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(5, start_max_radius)
            sx = radius * math.cos(angle)
            sy = radius * math.sin(angle)

            if env is not None:
                if not (env.x_range[0] < sx < env.x_range[1] and
                        env.y_range[0] < sy < env.y_range[1]):
                    continue

            too_close = any(
                math.hypot(sx - ox, sy - oy) < or_ + start_min_dist_to_obs
                for ox, oy, or_ in obs_list
            )
            if too_close:
                continue

            # Heading points toward origin (0, 0)
            heading = math.atan2(0.0 - sy, 0.0 - sx)

            start_state = [sx, sy, heading]
            print(f"[generate_random_obstacles] Start state "
                  f"({sx:.2f}, {sy:.2f}, {math.degrees(heading):.1f} deg) "
                  f"found after {attempts} attempts.")
            break

        # if start_state is None:
        #     print("[generate_random_obstacles] Warning: could not find valid "
        #           "start state, defaulting to [0, 0, 0].")
        #     start_state = [0.0, 0.0, 0.0]

        return obs_list, start_state

        
    def test_case_contingency_plan(self, _fig=None, _ax=None, num_obs=1, special_case=False):
    
        
        from HJR_FNO.HJR_FNO3d import HJR_FNO
        
        new_safe_region = [[0,0,2]] #x , y, r
        
        #New environment
        _env = env.Env(safe_regions=new_safe_region) 
        _env.x_range = (-8, 8)
        _env.y_range = (-8, 8)
        
        if not special_case:
            #generate random obstacles set
            number_of_obs = num_obs
            obs_list, self.current_state = self.generate_random_obstacles(
                env=_env,
                N=number_of_obs,
                r_min=1,
                r_max=1.8,#1.8,
                min_dist_between=2.0, 
                origin_safe_radius=1.3
            )
            
            # np.random.seed(0)
            # rng = np.random.default_rng()
            # known_ratio = rng.uniform() 
            known_ratio = 0 #np.random.uniform()

            indices = np.random.permutation(len(obs_list))
            n_known = max(1, int(len(obs_list) * known_ratio))

            print(self.current_state)
            known_obs   = [obs_list[i] for i in indices[:n_known]]
            unknown_obs = [obs_list[i] for i in indices[n_known:]]

            print(f"Known obstacles:   {len(known_obs)}")
            print(f"Unknown obstacles: {len(unknown_obs)}")
                
            _env.obs_circle = known_obs
            _env.unknown_obs_circle = unknown_obs   
        
        else:
            
            # self.current_state = [7,1,np.deg2rad(160)]
            
            # known_obs = [[-7,6,1.5], [-6, -6, 1.5],[0,-6,1.5]]
            # _env.obs_circle = known_obs
            # _env.unknown_obs_circle = [
            #                            [5,-2,1.5],
            #                            [2,7,1.5],
            #                            [0,4,1.5],
            #                            [8,5,1.5],
            #                            ]
            
            self.current_state = [-7,2,np.deg2rad(-np.pi/6)]
            
            known_obs = [[-7,6,1.2], [-6,-1,1.2], [5,-4,1.3], [6,5,1.3],]
            _env.obs_circle = known_obs
            _env.unknown_obs_circle = [
                                       [0,4,1.2],
                                       [-2,-3,1.0]
                                       ]
          
            
        
        #HJR-FNO configs
        Tf_reach = 8
        _hjr_fno = HJR_FNO(env=_env, safe_regions=new_safe_region, Tf_reach=Tf_reach)
        _hjr_fno.utils.sensing_radius = self.lidar_range
        if known_obs:
            _hjr_fno.update_obs(known_obs)
        
        
         # plotting
        _plotting = plotting.Plotting(self.current_state[:2], [[0.0, 0.0]], safe_regions=new_safe_region, _env=_env)
        
        if _fig is None:
            _fig, _ax = plt.subplots(figsize=(8, 8))
            _fig.suptitle(f"HJR-FNO Contingency")
            _ax.set_xlim(_env.x_range[0], _env.x_range[1]+1)
            _ax.set_ylim(_env.y_range[0], _env.y_range[1]+1)   

            
        _, _, TReach, success, V_val, g_Val, ham_term = _hjr_fno.contingency_policy(
            self.current_state, 
            _plotting, 
            _fig, 
            _ax,
            showplot=True,
            special_case = special_case
        )
        
        # _ax.clear()   # moved to caller so frames can be captured before clearing
        plt.pause(0.01)


        return _fig, _ax, TReach, success, V_val, g_Val, ham_term
    


    
    def check_delta_clear_overlap(self, phi_i, phi_j,
                              center_i, center_j,
                              x_local, y_local,
                              delta):   

        cx_i, cy_i, _ = center_i
        cx_j, cy_j, _ = center_j

        Nx, Ny = phi_i.shape
        dx = x_local[1] - x_local[0]
        dy = y_local[1] - y_local[0]

        # Compute relative shift in grid units
        shift_x = int(round((cx_j - cx_i) / dx))
        shift_y = int(round((cy_j - cy_i) / dy))

        # Create padded version of phi_j
        pad_x = abs(shift_x)
        pad_y = abs(shift_y)

        phi_j_padded = np.full(
            (Nx + 2*pad_x, Ny + 2*pad_y),
            np.inf
        )

        # Insert phi_j into padded array
        phi_j_padded[
            pad_x:pad_x+Nx,
            pad_y:pad_y+Ny
        ] = phi_j

        # Extract region aligned with phi_i
        start_x = pad_x - shift_x
        start_y = pad_y - shift_y

        phi_j_aligned = phi_j_padded[
            start_x:start_x+Nx,
            start_y:start_y+Ny
        ]

        # Compute overlap.
        # safe_margin is per-region; this method receives value arrays (phi_i/phi_j)
        # and centers (center_i/center_j) rather than region indices, so recover each
        # region's index by matching its center to hjr_fno.safe_regions.
        regions_xy = self.hjr_fno.safe_regions[:, :2]
        idx_i = int(np.argmin(np.sum((regions_xy - np.asarray(center_i)[:2]) ** 2, axis=1)))
        idx_j = int(np.argmin(np.sum((regions_xy - np.asarray(center_j)[:2]) ** 2, axis=1)))
        overlap_mask = (phi_i <= self.hjr_fno.safe_margin[idx_i]) & (phi_j_aligned <= self.hjr_fno.safe_margin[idx_j])

        if not np.any(overlap_mask):
            return False, None

        # Distance transform
        dist = distance_transform_edt(overlap_mask, sampling=(dx, dy))

        max_dist = np.max(dist)

        if max_dist < delta:
            return False, None

        max_idx = np.unravel_index(np.argmax(dist), dist.shape)
        
        print("Max stuff")
        print(np.argmax(dist))
        print(dist.shape)
        print(max_idx)

        # Compute global coordinates of ball center
        x0 = x_local[max_idx[0]] + cx_i
        y0 = y_local[max_idx[1]] + cy_i

        return True, (x0, y0)
        
def main():
    
    # #load configs
    # with open("config.yaml", "r") as f:
    #     cfg = yaml.safe_load(f)
        
    # x_start = (-18, 23, 0)  # Starting node
    # x_goal = [(-18, 23),  (-11, -14),  (5, 10), (15, 20), (15,-15)]  # Goal node
    # x_goal = [(-17, 16),  (-11, -14),  (-10, 5), (5,10), (14, 18), (15,-7)]  # Goal node
    # x_goal = [(-18, 23), (15,-15)]  # Goal node
    # x_goal = [(15, 20), (15,-15)]
    
    
    #case 1 (K=6):
    # x_goal = [(-17, 16),  (-11, -14), (5,10), (14, 18), (15,-7)] 
    x_goal = [
        (-12.94, 17),
        ( -8.44, -5.77),
        # (  3.81,  8.73),
        # ( 10.56, 14.73),
        # ( 11, -7),
    ]
    start_goal_index=0
    # safe_region = [[-15, 19, 2],
    #                 [-10, -9, 2],
    #                 [-7, 13, 2],
    #                 [-5, 2, 2],
    #                 [3, 8.5, 2],
    #                 [12, 1, 2],
    #                 [12, 15, 2],
    #                 [12, -10, 2]]
    safe_region = [
        [-11.06, 15.48, 2],
        [ -7.31, -5.95, 2],
        [ -5.06, 11.00, 2],
        [ -3.56,  2.75, 2],
        [  2.31,  7.63, 2],
        [  9.19,  1.63, 2],
        [  9.19, 12.13, 2],
        [  9.19, -6.70, 2],
    ]

    obs_cir = [
        [-5.0,   4.0,  1.5],
        [-6.0,  -6.0,  2.0],
        [ 7.0,   7.0,  1.0],
        [-10.0, -10.0, 1.8],
        [10.0,   6.0,  1.3],
        [-3.5,   7.0,  1.3],
        [ 0.0,  12.0,  1.6],
        [-2.0, -14.0,  1.6],
        [ 4.0,  -6.0,  1.4],
        [10.0, -14.0,  1.3],
    ]

    safe_cir = [
        [-15, 19, 2],
        [-10, -9, 2],
        [-7, 13, 2],
        [-5, 2, 2],
        [3, 8.5, 2],
        [12, 1, 2],
        [12, 15, 2],
        [12, -10, 2],
    ]

    filtered_obs = []

    for ox, oy, orad in obs_cir:
        intersects = False

        for sx, sy, srad in safe_cir:
            d = np.hypot(ox - sx, oy - sy)

            if d < orad + srad:
                intersects = True
                break

        if not intersects:
            filtered_obs.append([ox, oy, orad])

    print(filtered_obs)
    
    
    # #case 2 (K=1):
    # x_goal = [(-10, 7),(16,-7)] 
    # start_goal_index=0
    # safe_region = [[-6, 6, 2],
    #                 [5.5, 5, 2],
    #                 [13, -4, 2]]
    
        
    # step_len = 3.0
    # n0 = 2000
    # mu_free = 2500/2 
    # d =2
    
    # gamma_target = step_len * n0 / np.log(n0)
    
    # zeta_d = np.pi  # for d=2
    # constant = (2 * (1 + 1/d))**(1/d) * (mu_free / zeta_d)**(1/d)
    # gamma_FOS = gamma_target / constant




    sff = SFF_star(
        start_goal_index=start_goal_index, 
        x_goal=x_goal, 
        heading=0.0,
        lidar_range=8, # 12, #, #4.8
        step_len= 3.0,  #3.0
        gamma_FOS = 20.0,#100.0,
        epsilon=0.05,
        bot_sample_rate=0.10,  
        iter_max=1000, #18000,
        safe_regions = safe_region,
        HJ_contingency_enable = True

        )
    
    
    #====================================================
    #TSP-RRTX
    showPlot = True
    #====================================================
    
    sff.init_trees(showPlot=showPlot)
    
    plan_starttime = time.time()
    state_history = sff.planning(hamiltonian_cycle=False, showPlot=showPlot)
    plan_elapsedtime = time.time() - plan_starttime
    
    state_history = np.vstack(state_history)  # (T, 3)
    xy = state_history[:, :2]                # extract x,y
    diff = np.diff(xy, axis=0)               # (T-1, 2)
    segment_lengths = np.linalg.norm(diff, axis=1)
    total_distance = np.sum(segment_lengths)

    print("Total XY distance:", total_distance)
    print("Elapsed Time", plan_elapsedtime)
    

    output_dir = "/home/kmuenpra/git/HJR-FNO-ContingencyPlanning/exp_results"
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, "state_history.csv")

    np.savetxt(
        file_path,
        state_history,
        delimiter=",",
        header="x,y,theta,goal_id",
        comments=""
    )

    print(f"Saved to: {file_path}")
        
    
    # #====================================================
    # #Testing contingency
    # #====================================================
    
    # _fig, _ax, TReach, success, _, _, _ = sff.test_case_contingency_plan()


    # output_path = "exp_results/contingency_results_7Obs.txt"
    # num_iter = 10
    # num_obs = 6

    # # Ensure directory exists
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # # # Determine starting iteration index
    # # start_iter = 0
    # # file_exists = os.path.exists(output_path)

    # # if file_exists:
    # #     with open(output_path, "r") as f:
    # #         lines = f.readlines()
    # #         if len(lines) > 1:  # header + at least one entry
    # #             last_line = lines[-1].strip()
    # #             if last_line:  # avoid empty trailing line
    # #                 start_iter = int(last_line.split(",")[0]) + 1

    # # # Open in append mode
    # # with open(output_path, "a") as f:

    # #     # Write header only if file is new
    # #     if not file_exists:
    # #         f.write("iter,TReach,success,V_val,g_val,ham_term\n")

    # import imageio.v2 as imageio
    # frames = []

    # for k in range(num_iter):
    #     _fig, _ax, TReach, success, V_val, g_Val, ham_term = sff.test_case_contingency_plan(
    #         _fig, _ax, num_obs=num_obs, special_case=False
    #     )

    #     _fig.canvas.draw()
    #     frame = np.asarray(_fig.canvas.buffer_rgba()).copy()
    #     frames.append(frame)
    #     _ax.clear()
    #     _ax.set_xlim(-8, 9)
    #     _ax.set_ylim(-8, 9)

    #     # f.write(
    #     #     f"{iter_idx},"
    #     #     f"{float(TReach):.6f},"
    #     #     f"{bool(success)},"
    #     #     f"{float(V_val) if V_val is not None else 'None'},"
    #     #     f"{float(g_Val) if g_Val is not None else 'None'},"
    #     #     f"{float(ham_term) if ham_term is not None else 'None'}\n"
    #     # )

    # # gif_path = "exp_results/contingency_runs_7Obs.gif"
    # # imageio.mimsave(gif_path, frames, duration=0.8, loop=0)
    # # print(f"Saved animation to: {gif_path}")
        


if __name__ == '__main__':
    main()

'''
TODO:
-  Benchmark for the case, where the contingecy safe set is not a constraint any more

- overlapness of the reachable set, should be atleast 1 step len + reachable set's shrinking size after dt
>>>> instead of finding cloest index, find the set which has minimum time tEaliest.


- ATSP doesnt return the optimal tour sometimes (maybe due to the SA parameters)
- robot_path_to_goal flag not updated correctly after replanning with new obstacles (cause error because the robot plan to move while there is no path to goal)
- Implement remove_obstacles function in RRTX class
- self.s_bot --->>> Check verify_queue after resetting robot position
- Check the cost of each self.other_goals after resetting robot position
- Fix the heading during the reset_robot to be the heading toward the first parent (Note this assumption in the code)
- when checking for the is_feasible, make it faster by checking whether the obstacles center in local frame exceeds the grid_space of HJR-FNO
    - Also try to use the official ray, rather than using np.linspace()
- Maybe retrain HJRNO model with the case where there is no obstacles, or simply pre-compute it to be used in the experiment
    >>>> when Nodel.parent is None, and Node.x and Node.y is the goal location >>> simply executed toward the goal.
- control policy (try simple switch for now), and think about condition for contingency behavior
'''