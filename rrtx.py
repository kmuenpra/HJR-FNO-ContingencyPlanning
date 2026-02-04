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
import dill

# =========================
# Local / project imports
# =========================
import env
import plotting
import utils
import atsp

# Import these at module level for PyTorch unpickling
# This is safe because we're not importing HJR_FNO class itself
from HJR_FNO.HJR_FNO import FNO1d, SpectralConv1d



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
            math.hypot(self.x - other.x, self.y - other.y) < 1e-4 #NOTE i chnange the tolerance to 1e-6 from 1e-4
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
        # list of tuples: ( u, d_pi(v,u)+lmc(u) )
        lmcs = [(u, self.distance(u) + u.lmc) for u in (self.all_out_neighbors() - orphan_nodes) if u.parent and u.parent != self]
        if not lmcs:
            return
        p_prime, lmc_prime = min(lmcs, key=lambda x: x[1])
        if lmc_prime < self.lmc and not utils.is_collision(self, p_prime): # added collision check, not in pseudocode
            self.lmc = lmc_prime # lmc update is done in Julia code
            self.set_parent(p_prime) # not sure if we need this or literally just set the parent manually without propagating

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
        ) -> None:
        
        # Start and Goal
        self.s_goal = Node(x_goal, lmc=0.0, cost_to_goal=0.0)
        # self.s_goal.active = True
        
        # self.s_start = Node(x_start)
        self.s_bot = None
        
        #For multi-goal tree expansion (SFF*)
        self.prob_q = 0.9
        self.other_goals_id = other_goals_id
        self.other_goals = []
        for g in other_goals:
            self.other_goals.append(Node((g[0],g[1])))
                        
        
        
        # RRTx configs
        self.env = env.Env(safe_regions=safe_regions)
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
        self.multi_paths = [[] for _ in range(len(self.other_goals))]

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
                    
    
    def set_new_goal(self, target_id:int, new_pose:Tuple):
        '''
        Set the current target with target_id to the new position 
        Assume new_pose := Tuple(x,y) is obstacle free
        '''
        
        #set new target location
        j = self.other_goals_id.index(target_id)
        self.other_goals[j] = Node((new_pose[0], new_pose[1]))
        
        #find and extend nearest node toward the new target location
        v_nearest = self.nearest(self.other_goals[j] )
        v = self.saturate(v_nearest, self.other_goals[j])
        
        if v and not self.utils.is_collision(v_nearest, v) and self.is_feasible_ray(v_nearest, v):
                    
                self.extend(v, v_nearest)
                
                if v.parent is not None:
                    self.rewire_neighbours(v)
                    self.reduce_inconsistency()

        
        if self.other_goals[j].cost_to_goal < np.inf:
            self.path_to_goal[j] = True
        else:
            self.path_to_goal[j] = False
            
        #expand the tree until new path is found
        # TODO there should be a better solution than this while loop
        while not self.path_to_goal.all():
            
            self.planning(iter_max=100)
        
        
                    
    def reset_robot_v2(self, current_state, iter_max=100):          
        
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
        
        if v_nearest == self.s_bot: #equal if position are close within tolerance
            self.s_bot = v_nearest
            
        if self.s_bot.cost_to_goal < np.inf:
            self.robot_path_to_goal = True
            self.update_path(self.s_bot)
            
        else:
            
            i = 0
            
            while (not self.robot_path_to_goal) and i <= iter_max:
            
                if np.random.random() > self.prob_q:
                    v = self.s_bot
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
                    
                i += 1
                
                if i == 100:
                    print(f"replanning robot's path iteration {i}")
                    
    def update_robot_heading(self, new_heading=None):
        
        if new_heading is not None:           
            self.robot_state[2] = new_heading
            
        elif self.s_bot.parent is not None:
            dx = self.s_bot.parent.x - self.s_bot.x
            dy = self.s_bot.parent.y - self.s_bot.y
            heading = math.atan2(dy, dx)
            self.robot_state[2] = heading
            
            
    
    def reset_robot(self, current_state, iter_max=100):

        self.s_bot = Node((current_state[0], current_state[1])) #robot start at the same location but does not have parents
        self.path = []
        self.robot_path_to_goal = False
        # self.curr_tree_idx = j
        
        
        # #Connect to an existing tree
        # if self.path_to_goal[j]:
        #     self.robot_path_to_goal = True
        #     self.update_path(self.s_bot)
        
        #expand the tree until new path is found
        # TODO there should be a better solution than this while loop
                
        # v = self.random_node(robots_plan=True)
        v_nearest = self.nearest(self.s_bot)
        
        v = self.saturate(v_nearest, self.s_bot)
            
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
                
        if (not self.robot_path_to_goal) or (not all(self.path_to_goal)): #suppose to be while True
            
            self.planning(iter_max=iter_max, robots_plan=True)
            print("replanning for 1000 iterations")
            


        
        self.robot_state = [self.s_bot.x, self.s_bot.y, current_state[2]]
        self.robot_position = [self.s_bot.x, self.s_bot.y]
                 
    def planning_with_robot(self , steps=10):
        
        new_obs_flag = False
        all_new_obs = []
        traversed_distance = 0.0
                        
        for step_idx in range(steps): #set plotting Hz
            
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

                    
                    print(f"\n ----- Rewiring Trees due to {len(detected_obs)} newly-detected obstacle(s) ----- ")
                    print("Obstacle location: ", detected_obs)
                    
                    # TODO update HJB reachable set with the detected_obs here
                    exec_time_start = time.time()
                    
                    if self.HJ_contingency_enable:
                        self.hjr_fno.update_obs(detected_obs)
                        print("Update Reachable set:", time.time() - exec_time_start, "s")
                    
                    # update known obstacles withint the environment
                    for obs in detected_obs:
                        self.update_obstacles(obs, robots_plan=True)
                
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
                
    def update_obstacles(self, obs_cir, robots_plan=False):
        #update_obstacles(self, event, obs_cir):
        
        # Algorithm 8
        # x, y = int(event.xdata), int(event.ydata)
        
        # print("current search radius", self.search_radius)
        
        exec_time_start = time.time()
        self.add_new_obstacle(obs_cir)   #TODO: This function sometimes takes a bit of time
        print("Added new obstacles:", time.time() - exec_time_start, "s")
        
        exec_time_start = time.time()
        self.propagate_descendants(robots_plan=robots_plan)
        print("Propagate Descendants:", time.time() - exec_time_start, "s")
        
        exec_time_start = time.time()
        if robots_plan:
            self.verify_queue(self.s_bot)
            
        for g in self.other_goals:
            self.verify_queue(g)
        print("Verify Queue:", time.time() - exec_time_start, "s")
        
        exec_time_start = time.time()
        self.reduce_inconsistency(robots_plan=robots_plan)
        print("Reduce Inconsistency:", time.time() - exec_time_start, "s")

    def add_new_obstacle(self, obs):
        # Algorithm 12
        # x, y, r = obs
        # print("Osbstacle at: x =", x, ", y =", y, ", r = ", r)
        
        self.obs_circle.append(obs)
        self.plotting.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle, self.unknown_obs_circle) # for plotting obstacles
        self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle, self.unknown_obs_circle) # for collision checking
        self.update_gamma() # free space volume changed, so gamma must change too
        
        # NOTE Collect all directed node pair (v->u) that intersects with the obstacles
        #
        # for all nodes 'v' in tree_nods 
        #       for all nodes 'u' in neighborhood of 'v' (include static and running nodes)
        #               check if the edge v -> u intersected with 
        E_O = [(v, u) 
                # for v in self.tree_nodes
                for v in self.kd_tree.search_nn_dist((obs[0], obs[1]), obs[2] + self.search_radius)
                    for u in v.all_out_neighbors() 
                        if self.utils.is_intersect_circle(*self.utils.get_ray(v, u), obs[:2], obs[2])] # or not self.is_feasible_ray(v,u)]
        
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

            #NOTE I remove this from the pseudocode because it messes up the kd-tree structure
            # try:
            #     self.tree_nodes.remove(v) # NOT IN THE PSEUDOCODE
            #     self.kd_tree.remove(v)
                
            # except ValueError:
            #     pass
            
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
        # Algorithm 4
        if v.cost_to_goal - v.lmc > self.epsilon:
            v.cull_neighbors(self.search_radius)
            for u in v.all_in_neighbors() - set([v.parent]):
                if u.lmc > v.distance(u) + v.lmc and \
                        not self.utils.is_collision(u, v) and self.is_feasible_ray(u,v): # added collision check (Julia)
                    u.lmc = v.distance(u) + v.lmc
                    u.set_parent(v)
                    if u.cost_to_goal - u.lmc > self.epsilon:       
                        self.verify_queue(u) #add to priority queue, if the node is inconsistent

        if robots_plan:
            self.update_path(self.s_bot) # update path to goal for plotting
            # self.other_goals[self.curr_tree_idx] = self.s_bot
            # self.update_multi_paths(self.s_bot, self.curr_tree_idx)
            
        # else:
        for j, goal_j in enumerate(self.other_goals):
            self.update_multi_paths(goal_j, j)

    def random_node(self, robots_plan=False):
        
        delta = self.utils.delta

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

    def update_path(self, node):
        self.path = []
        while node.parent:
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
        
    def is_feasible_ray(self, start:Node, end:Node):
        
        o, d = self.utils.get_ray(start, end)
        
        ''' Use query from HJR-FNO model to check feasibility of the ray (however, expensive beacuse we need spatial size of at least 32)'''
        # #crate batch size of 16 based on start and end nodes
        # # query the value of HJ reachable set from HJR-FNO
        # t_vals = np.linspace(0, 1, 32)
        # theta_array = self.robot_state[2] + np.array([-np.pi/4, 0.0, np.pi/4])

        # positions = o + t_vals[:, None] * d                  # (32, 2)
        # feasible =  self.hjr_fno.is_state_feasible(robot_state= positions, theta_array=theta_array, reachable_set_constraint=self.HJ_contingency_enable)
        # return feasible

        # NOTE old method, using look up table of predicted HJ reachable set
        
        for t in np.linspace(0,1,3):
            
            if not self.hjr_fno.is_feasible(v= (o[0] + t * d[0], o[1] + t * d[1]) , reachable_set_constraint=self.HJ_contingency_enable):
                return False

        return True

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
        ) -> None:
        
        from HJR_FNO.HJR_FNO import HJR_FNO
        
        assert start_goal_index < len(x_goal), "start_goal_index index out of range"
        
        self.start_goal_index = start_goal_index
        x_start = x_goal[start_goal_index]
        self.iter_max = iter_max
        
        self.env = env.Env(safe_regions=safe_regions)
        self.plotting = plotting.Plotting(x_start, x_goal, safe_regions=safe_regions)
        
        #HJR-FNO configs
        self.Tf_reach = 7.5 #must be less than 8s (underapproximation of the training data)
        self.hjr_fno = HJR_FNO(env=self.env, safe_regions=safe_regions, Tf_reach=self.Tf_reach)
        self.current_state = [x_start[0], x_start[1], heading]
        self.HJ_contingency_enable = False  #enable contingency constraint in RRTX tree planning

        # plotting
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
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
            )
            
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
        
    def update_distance_matrix(self, pattern=None, print_flag=True, precision=3):
                
                
        """
        Build a distance matrix (based on a pattern) used to solve ATSP.

        pattern examples:
        None                 -> full goal-goal matrix (default ; equivalent to [0, 1, ..., n-1])
        [0, 1, 2]            -> subset of goals
        ["robot", 0, 1]      -> robot + selected goals
        """

        # ---------- Default pattern ----------
        if pattern is None:
            pattern = list(self.rrtx_trees.keys())  # [0, 1, ..., n-1]

        # ---------- Validate pattern ----------
        valid_goals = set(self.rrtx_trees.keys())
        for p in pattern:
            if p != "robot" and p not in valid_goals:
                raise ValueError(f"Invalid pattern entry: {p}")

        # ---------- Build index mapping ----------
        idx_map = {p: k for k, p in enumerate(pattern)}
        m = len(pattern)

        D_pat = np.full((m, m), np.inf)

        # ---------- Fill matrix ----------
        for a in pattern:
            for b in pattern:
                ia, ib = idx_map[a], idx_map[b]

                # self-distance
                if a == b:
                    D_pat[ia, ib] = 0.0
                    continue

                # robot → goal
                if a == "robot" and b != "robot":
                    tree_b = self.rrtx_trees[b]
                    D_pat[ia, ib] = tree_b.s_bot.cost_to_goal
                    continue

                # goal → robot
                if a != "robot" and b == "robot":
                    D_pat[ia, ib] = np.inf
                    continue

                # goal → goal
                tree_a = self.rrtx_trees[a]
                if b in tree_a.other_goals_id:
                    local_j = tree_a.other_goals_id.index(b)
                    D_pat[ia, ib] = tree_a.other_goals[local_j].cost_to_goal

        self.D = D_pat
        
        if print_flag:
            self.print_distance_matrix(pattern=pattern, precision=precision)
            

    def print_distance_matrix(self, pattern=None, precision=3):
        
        labels = pattern if pattern is not None else list(range(self.D.shape[0]))
        n = len(labels)

        # ---- Header ----
        header = "      " + "".join(f"{str(lbl):^10}" for lbl in labels)
        print(header)
        print("      " + "-" * (10 * n))

        # ---- Rows ----
        for i, row_lbl in enumerate(labels):
            row_str = f"{str(row_lbl):>4} |"
            for j in range(n):
                val = self.D[i, j]
                if np.isinf(val):
                    row_str += f"{'∞':^10}"
                else:
                    row_str += f"{val:^10.{precision}f}"
            print(row_str)

            
    def solve_atsp_SA(self, distance_matrix, learning_plot=False, silent_mode=True):
        """
        Solve an Asymmetric TSP directly from a distance matrix using Simulated Annealing.

        Returns
        -------
        solution, cost
        """
        # # Optional: sanity checks
        # n = len(distance_matrix)
        # assert all(len(row) == n for row in distance_matrix), "Matrix must be square"
        
            
        D_int = np.where(
            np.isinf(distance_matrix),
            9_999_999,
            np.round(distance_matrix * 1000)
        ).astype(int)


        _atsp = atsp.SA(D_int.tolist(), learning_plot=learning_plot, silent_mode=silent_mode)
        return _atsp.solve()
    
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
        
    def init_trees(self):        

        for i in range(self.n_tree):
            
            print(f"Initialize the tree {i}")
        
            self.rrtx_trees[i].planning()
            
                
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
                
        
    def planning(self, hamiltonian_cycle=False):
        
        '''
        Planning using TSP-RRTX with Contingency Handling
        
        if hamiltonian_cycle = True, the tour will return to the starting target
        otherwise, it will only visit each target once
        '''
        
        
        #-----------------
        # Solve initial TSP tour
        # ----------------
        
        self.update_distance_matrix() #update self.D
        
        '''
        Simulated Annealing
        '''
        # res = self.solve_atsp_SA(distance_matrix=self.D, learning_plot=False)
        
        # print('\nOPTIMIZED COST:', res[1])
        # print('Optimal Tour (target_id): ' + ' '.join(map(str, res[0])))
        
        # sequence_to_visit = res[0].copy() #this will change to reflect remaining sequence to visit targets
        # optimal_tour = res[0].copy() #optimal tour sequence
        # original_tour = res[0].copy()
        
        # #FIRST sequence of the tour
        # prev_id = sequence_to_visit[0]
        # sequence_to_visit.pop(0)
        
        '''
        Held-Karp Algorithm
        '''
        prev_id = self.start_goal_index
        
        min_cost, optimal_tour = self.solve_atsp_held_karp(distance_matrix=self.D, start=self.start_goal_index, hamiltonian_cycle=hamiltonian_cycle)
        print('\nOPTIMIZED COST:', min_cost)
        print('Optimal Tour (target_id): ', optimal_tour )
                
        sequence_to_visit = optimal_tour.copy() #this will change to reflect remaining sequence to visit targets
        sequence_visited = []
        sequence_visited.append(sequence_to_visit.pop(0)) #remove the starting target from the remaining sequence
        
        original_tour = optimal_tour.copy() #Store original tour to observe changes later
        
        #-----------------
        # Set robot states
        # ----------------

        #Set robot position to the first target location of the routing sequence
        heading = 0.0
        self.current_state = [self.rrtx_trees[prev_id].s_goal.x, self.rrtx_trees[prev_id].s_goal.y, heading]
        
        # Simulate Robot's plan
        print("\nInitilize robot position for each tree")
        
        for i in range(self.n_tree):
            
            self.rrtx_trees[i].reset_robot_v2(current_state=self.current_state)            
            self.rrtx_trees[i].prob_q = 0.9
            
        #Add mouse click event handler (resemble adversarial event) which triggers Contingency planning
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click) 
        self.fig.suptitle(f"HJR-FNO Contingency\nOptimal Tour: {optimal_tour}\nVisited: {sequence_visited}\nTo Visit: {sequence_to_visit}")
        prev_plotting = time.time()
            
            
        #-----------------
        # Main Planning loop
        # TSP-RRTX with Contingency Handling
        # ----------------
            
        print("Start Robot's Plan Execution")
                
        for i in range(1, len(optimal_tour)):
            
            traversed_distance = 0.0
            
            for plan_iter in range(self.iter_max):
                
                # # ========== NEW: Check if waiting for clicks ==========
                # # If contingency is running, keep processing events until done
                # while self.waiting_for_first_click:
                #     self.fig.canvas.flush_events()
                #     plt.pause(0.01)
                
                # # If contingency is done, wait for second click before proceeding
                # while self.waiting_for_second_click:
                #     self.fig.canvas.flush_events()
                #     plt.pause(0.01)
                
                # # After second click, restart this plan_iter loop from the beginning
                # if self.resume_planning:
                #     print("Restarting planning iteration after contingency...")
                #     self.resume_planning = False
                #     continue  # Restart from top of plan_iter loop
                # # ========== END NEW CODE ==========
                
    
                id = optimal_tour[i]
            
                if plan_iter == 0:
                    #set the current_state to the previous target location                                    
                    self.rrtx_trees[id].reset_robot_v2(current_state=self.current_state)
                    self.rrtx_trees[id].update_robot_heading()
                                            
                    print("====================================================\n")
                    print(f"Starting planning from Target #{prev_id} to Target #{id}")
                    print("\n====================================================")
                    
                #if path exists, then execute robot's plan
                if self.rrtx_trees[id].robot_path_to_goal:
                            
                    new_obs, new_obs_flag, distance_moved = self.rrtx_trees[id].planning_with_robot(steps=10) #propagate 10 steps
                    traversed_distance += distance_moved
                    
                    self.current_state = self.rrtx_trees[id].robot_state
                    
                #otherwise, keep replanning until path is found
                else:
                    
                    self.rrtx_trees[id].reset_robot_v2(current_state=self.current_state)
                    self.rrtx_trees[id].update_robot_heading()
                    continue
                
                
                '''
                Handling contingency during robot's plan execution
                '''
                if self.rrtx_trees[id].contingency_triggered and self.HJ_contingency_enable:
                    
                    detected_obs_during_contingency, contingency_trajectory = self.hjr_fno.contingency_policy(
                            self.current_state, 
                            self.plotting, 
                            self.fig, 
                            self.ax
                    )
                    
                    #Update state and traversed distance so far
                    self.current_state = contingency_trajectory[-1]
                    
                    for traj_i in range(len(contingency_trajectory) - 1):
                        x0, y0, _ = contingency_trajectory[traj_i]
                        x1, y1, _ = contingency_trajectory[traj_i + 1]
                        traversed_distance += np.hypot(x1 - x0, y1 - y0)
                    
                    if len(detected_obs_during_contingency) > 0:
                        new_obs += detected_obs_during_contingency
                        
                        #For the current tree, update only new obstacles detected during contingency
                        for obs in detected_obs_during_contingency:
                            self.rrtx_trees[id].update_obstacles(obs, robots_plan=True)
                            
                    #Reset robot position in the current tree                                    
                    self.rrtx_trees[id].reset_robot_v2(current_state=self.current_state)
                    self.rrtx_trees[id].update_robot_heading()
                            
                    #Reset the contingency trigger flag
                    for tree_k in self.rrtx_trees.values():
                        tree_k.contingency_triggered = False
                    
                
                '''
                For all new obstacles detected during robot's plan execution/contingency plan
                - Update all other trees with the new obstacles and rewire if necessary
                '''    
                #if there is new obstacle found, optimal routing might changes
                if new_obs_flag:
                                                            
                    updateObs_time_start = time.time()
                    
                    # Update: the path of k-th tree with the 'id'-target being current robot's position
                    for k, tree_k in self.rrtx_trees.items():
                        
                        # print("================")
                        # print("update tree", k)
                        # print("current search radius", tree_k.search_radius)
                        # print("================")
                        
                        # Skip replanning for trees which is currently being visited (already rewired inside planning_with_robot)
                        # and trees that have already been visited in this tour
                        if k == id or (len(sequence_visited) > 1 and k in sequence_visited[1:]):
                            
                            # print(f"Robot's path to target #{k} exist? {tree_k.robot_path_to_goal}")
                            # print("Robot's cost to goal" , tree_k.s_bot.cost_to_goal)
                            # print("Robot's LMC cost" , tree_k.s_bot.lmc)
                            # print(f"Other target's path to target #{k} exist? {tree_k.path_to_goal}")
                            # print(len(tree_k.tree_nodes), "nodes in the tree after replanning")
                            continue

                        #reset robot position in the other trees
                        tree_k.reset_robot_v2(current_state=self.current_state)
                        
                        # ----------------- Update Obstacles in other trees -----------------
                        #NOTE Below is the same as updated_obstacle() function in RRTX class
                        
                        # This is similar to RRTX.updated_obstacle() but here we are updating all paths between targets + robot's current pose, rather than just the path to the robot 
                        for obs in new_obs:
                            
                            #Update list of obstacle and unknown obstacles in RRTX.utils.obs == RRTX.env.obs == RRTX.obs
                            #NOTE since RRTX.plotting is shared globally, the obs and unknown_obs is already updated during the rrtx_trees[id].planning_with_robot
                            tree_k.obs_circle.append(obs)
                            tree_k.utils.update_obs(tree_k.obs_circle, tree_k.obs_boundary, tree_k.obs_rectangle, self.rrtx_trees[id].unknown_obs_circle)
                            
                            #Update orphan node list:
                            tree_k.update_gamma() # free space volume changed, so gamma must change too
                            
                            E_O = [(v, u) 
                                    # for v in tree_k.tree_nodes
                                    for v in tree_k.kd_tree.search_nn_dist((obs[0], obs[1]), obs[2] + tree_k.search_radius)
                                        for u in v.all_out_neighbors() 
                                            if tree_k.utils.is_intersect_circle(*tree_k.utils.get_ray(v, u), obs[:2], obs[2])]
                            #
                            # To preserve graph structure:
                            # instead of removing (v->u) from the neighbor set, make the edge haviing infinite cost
                            for v, u in E_O:
                                v.infinite_dist_nodes.add(u)
                                u.infinite_dist_nodes.add(v)
                                if v.parent and v.parent == u:
                                    tree_k.verify_orphan(v)
                                    # should theoretically check if the robot is on this edge now, but we do not
                                    # v.parent.children.remove(v) # these two lines are from the Julia code
                                    # v.parent = None 
                                    
                            heapq.heapify(tree_k.Q) # reheapify after removing a bunch of elements and ruining queue

                        tree_k.propagate_descendants(robots_plan=True)      
                        
                        tree_k.verify_queue(tree_k.s_bot)
                        
                        for goal_j in tree_k.other_goals:
                            tree_k.verify_queue(goal_j)
                                                        
                        tree_k.reduce_inconsistency(robots_plan=True)
                        
                        # ----------------- End of Update Obstacles in other trees -----------------
                        
                        
                        
                        
                        
                        #------------------ Replanning -------------------------
                            
                        #update robot path and other target paths
                        tree_k.update_path(tree_k.s_bot) # update path to goal for plotting
                        
                        if tree_k.s_bot.cost_to_goal < np.inf:
                            tree_k.robot_path_to_goal = True
                        
                        for j, goal_j in enumerate(tree_k.other_goals):
                            tree_k.update_multi_paths(goal_j, j)
                            
                            if goal_j.cost_to_goal < np.inf:
                                tree_k.path_to_goal[j] = True
                                
                        # print(f"Robot's path to target #{k} exist? {tree_k.robot_path_to_goal}")
                        # print("Robot's cost to goal" , tree_k.s_bot.cost_to_goal)
                        # print("Robot's LMC cost" , tree_k.s_bot.lmc)
                        # print(f"Other target's path to target #{k} exist? {tree_k.path_to_goal}")
                        # print(len(tree_k.tree_nodes), "nodes in the tree after replanning")
                            
                        #------------------ End of Replanning ------------------
                    
                    
                    print(f"\nRewire other trees: {time.time() - updateObs_time_start} s\n")
                        

                        
                    # ----------------- Solve ATSP again -----------------
                    
                    if (hamiltonian_cycle and len(sequence_to_visit[:-1]) >= 2) or (not hamiltonian_cycle and len(sequence_to_visit) >= 2): 
                    #Held-Karp needs at least 2 targets to optimize (excluding the final target of the tour that we must return back to)
                        
                        '''
                        Held-Karp Algorithm
                        '''
                        
                        self.update_distance_matrix(print_flag=False) #update self.D based on the remaining targets to visit
                        
                        #From rrtx[prev_id] to other tree, the distance would be rrtx[prev_id] -> robot -> other tree
                        for tmp_id in range(self.n_tree):
                            if tmp_id == prev_id:
                                continue
                            self.D[prev_id, tmp_id] = traversed_distance + self.rrtx_trees[tmp_id].s_bot.cost_to_goal   
                        self.print_distance_matrix(precision=3)                     
                        
                        min_cost, new_tour = self.solve_atsp_held_karp(distance_matrix=self.D, start=sequence_visited, hamiltonian_cycle=hamiltonian_cycle)
                        
                        if len(new_tour) == self.n_tree:
                            
                            if new_tour != optimal_tour:
                                plan_iter = 0 #reset plan_iter to 0 to replan toward the new target in the new sequence
                                print("\nNew optimal tour found after replanning due to new obstacles!")
                                
                            optimal_tour = new_tour
                        
                            sequence_to_visit = optimal_tour.copy()
                            sequence_visited = sequence_to_visit[:i]
                            sequence_to_visit = sequence_to_visit[i:]
                        else:
                            print("Held-Karp failed to find a new optimal tour")
                            
                        '''
                        [Ignored] Simulated Annealing
                        '''
                        # sorted_seq = ["robot"] + sorted(sequence_to_visit[:-1])
                        # self.update_distance_matrix(pattern= sorted_seq) #update self.D based on the remaining targets to visit
                        
                        # res = self.solve_atsp_SA(distance_matrix=self.D, learning_plot=False, silent_mode=True)
                        
                        # #NOTE ATSP solve return the tour indices based on the pattern order of the distance matrix
                        # #     which is different from the original target ids, that we assigned for each rrtx tree
                        # tour_idx = res[0].copy() 
                        # tour_idx = self.rotate_tour(tour_idx, start_idx=0) #rotate the new tour to start from robot position (index 0)
                        
                        # #update the sequence to visit, based on the indices of the new tour
                        # new_sequence = [
                        #     sorted_seq[t_i]
                        #     for t_i in tour_idx
                        #     if t_i != 0 #ignore robot index
                        # ]
                        
                        # if new_sequence != sequence_to_visit[:-1]:
                        #     plan_iter = 0 #reset plan_iter to 0 to replan toward the new target in the new sequence
                            
                        #     optimal_tour[i:-1] = new_sequence #update the remaining optimal tour
                        #     sequence_to_visit[:-1] = new_sequence #update the remaining sequence to visit
                    else:
                        
                        self.update_distance_matrix(print_flag=False) #update self.D based on the remaining targets to visit
                        self.D[prev_id, id] = traversed_distance + self.rrtx_trees[id].s_bot.cost_to_goal   
                        self.print_distance_matrix(precision=3)
                    
                    
                    print('\nNew Optimal Tour (target_id): ' , optimal_tour)
                    print('New Tour cost', self.compute_tour_distance(optimal_tour, prev_id=prev_id, curr_id=optimal_tour[i], traversed_distance=traversed_distance))
                    print('Targets visited (target_id): ', sequence_visited)
                    print("Remaining sequence to visit (target_id): ", sequence_to_visit)
                    
                    print('\nOriginal Tour (target_id): ' , original_tour)
                    print('Original Tour cost', self.compute_tour_distance(original_tour, prev_id=prev_id, curr_id=original_tour[i], traversed_distance=traversed_distance))

                    # ----------------- End of solve ATSP  ---------------
                        
                    
                
                # only update the plot at 5 Hz
                elapsed_plotting = time.time() - prev_plotting
                if elapsed_plotting >= 0.2:
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
                        edge_col = LineCollection(self.edges, colors='blue', linewidths=0.3, alpha=0.4)
                        self.ax.add_collection(edge_col)

                    # draw path to goal
                    if self.rrtx_trees[id].path:
                        path_col = LineCollection(self.rrtx_trees[id].path, colors='black', linewidths=1.5)
                        self.ax.add_collection(path_col)

                    # draw robot + lidar
                    self.plotting.plot_robot(self.ax, self.rrtx_trees[id].robot_position, self.rrtx_trees[id].lidar_range)
                    
                    # plot reachable set at current heading
                    if self.HJ_contingency_enable:
                        self.plotting.plot_reachable_set(self.ax, self.hjr_fno, self.rrtx_trees[id].robot_state[2], self.rrtx_trees[id].Tf_reach)
                    
                    
                    if new_obs_flag and  self.show_subplots:
                        
                        #plot k-th tree updated path toward j-th target
                        for k in range(self.n_tree): 
                            
                            # self.ax.clear()
                            fig, ax = plt.subplots(figsize=(8, 8))
                                                        
                            fig.suptitle(f"tree {k} replanning due to new obstacle from robot at Target#{prev_id} to Target#{id}")

                            ax.set_xlim(self.env.x_range[0], self.env.x_range[1] + 1)
                            ax.set_ylim(self.env.y_range[0], self.env.y_range[1] + 1)
                            self.plotting.plot_env(ax, colorList=self.colorList)     
                            self.plotting.plot_robot(ax, self.rrtx_trees[id].robot_position, self.rrtx_trees[id].lidar_range)
                            if self.HJ_contingency_enable:
                                self.plotting.plot_reachable_set(ax, self.hjr_fno, self.rrtx_trees[id].robot_state[2], self.rrtx_trees[id].Tf_reach)
                                                
                            # draw path to goal
                            for j in range(len(self.rrtx_trees[k].other_goals)):
                                if self.rrtx_trees[k].path_to_goal[j]:
                                    path_col = LineCollection(self.rrtx_trees[k].multi_paths[j], colors=self.colorList[k], linewidths=3, alpha=0.7)
                                    ax.add_collection(path_col)
                                    
                            if self.rrtx_trees[k].robot_path_to_goal:
                                    path_col = LineCollection(self.rrtx_trees[k].path, colors=self.colorList[k], linewidths=2.5)
                                    ax.add_collection(path_col)
                                    
                                    
                            # draw tree nodes
                            if self.rrtx_trees[k].all_nodes_coor:
                                nodes = np.array(self.rrtx_trees[k].all_nodes_coor)
                                ax.scatter(nodes[:, 0], nodes[:, 1], s=4, c='gray', alpha=1)
                                
                            #get all edges
                            edges = []
                            for node in self.rrtx_trees[k].tree_nodes:
                                if node.parent:
                                    edges.append(np.array([[node.parent.x, node.parent.y], [node.x, node.y]]))

                            # draw tree edges
                            if edges:
                                edge_col = LineCollection(edges, colors='blue', linewidths=0.3, alpha=1)
                                ax.add_collection(edge_col)

                        
                    # force redraw
                    plt.pause(0.001)
                    
                    # ================= END OF PLOTTING =======================
                    
                if new_obs_flag:
                    
                    #Print new distance matrix between each goals after replanning
                    print("\n")
                    for k, tree_k in self.rrtx_trees.items():
                        print(f"Tree {k} robot's cost to goal: {tree_k.s_bot.cost_to_goal}, Reachable? {tree_k.robot_path_to_goal}")
                    print("\n")
                    
                    if  self.show_subplots:
                        plt.show()
                        self.fig, self.ax = plt.subplots(figsize=(8, 8))
                    
                #Terminate when reach the goals
                if self.rrtx_trees[id].s_bot.cost_to_goal == 0.0 and self.rrtx_trees[id].s_bot.lmc == 0.0:
                    print("Successfully reach the goal!")
                    break
                
                    
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

def main():
    x_start = (-18, 23, 0)  # Starting node
    x_goal = [(-18, 23),  (-11, -17),  (5, 12), (15, 20), (15,-15)]  # Goal node
    #                      (-5, -1)
    # x_goal = [(-20, 23), (15,-15)]  # Goal node



    sff = SFF_star(
        start_goal_index=0, 
        x_goal=x_goal, 
        heading=0.0,
        lidar_range=5.5,
        step_len= 3.0, 
        gamma_FOS = 20.0,#100.0,
        epsilon=0.05,
        bot_sample_rate=0.10,  
        iter_max=12000,
        safe_regions=[[-5, -1, 2],
                      [6, 8.5, 2],
                      [-10, -14, 2],
                      [18,1,2],
                      [15,15,2],
                      [-7,13,2],
                      [18,-12,2],
                      [-15,23,2]],
        )
    
    sff.init_trees()
    sff.planning()
    


if __name__ == '__main__':
    main()

'''
TODO:
-  Benchmark for the case, where the contingecy safe set is not a constraint any more
- For Held-Karp Algorithm, implement the case where the we dont have to return to the starting point
- is_state_feasible() directly use HJR-FNO to check feasibility instead of table look up


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