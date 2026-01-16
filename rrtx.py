import os
import sys
import time
import math
import heapq
from typing import Dict, List, Tuple
import itertools    
from collections import deque
from collections.abc import Sequence
import kdtree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import env, plotting, utils, queue_utils

from typing import Tuple, List, Sequence
import numbers

from HJR_FNO import HJR_FNO, FNO1d, SpectralConv1d, Grid

import traceback

import random



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
            math.hypot(self.x - other.x, self.y - other.y) < 1e-6
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
                print('KeyError in set_parent()')
                print('Node', (self.x, self.y))
                print('cost', (self.cost_to_goal, self.lmc))
                print("Node's Parent", (self.parent.x, self.parent.y))
                for child in self.parent.children:
                    print("- children", (child.x, child.y))
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
                    if self.parent != u and r < self.distance(u):
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
        
        
    

class RRTX:
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
        env: env.Env,
        plotting: plotting.Plotting,
        util,
        hjr_fno: HJR_FNO,
        fig: Figure,
        ax: Axes
        ) -> None:
        
        # Start and Goal
        self.s_goal = Node(x_goal, lmc=0.0, cost_to_goal=0.0)
        self.s_goal.active = True
        
        self.s_start = Node(x_start)
        self.s_bot = self.s_start
        
        #For multi-goal tree expansion (SFF*)
        self.prob_q = 0.9
        self.other_goals_id = other_goals_id
        self.other_goals = []
        for g in other_goals:
            self.other_goals.append(Node((g[0],g[1])))
            
        self.curr_tree_idx = 0 #This will be set when robot is decided which other target locations it is starting on
            
        
        
        # RRTx configs
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
        self.robot_state = [self.s_bot.x, self.s_bot.y, heading]
        self.robot_position = [self.s_bot.x, self.s_bot.y]
        self.robot_speed = 0.6 # m/s
        self.lidar_range = lidar_range
        
        #HJR-FNO configs
        self.Tf_reach = hjr_fno.Tf_reach #must be less than 8s (underapproximation of the training data)
        self.hjr_fno = hjr_fno #HJR_FNO(safe_regions=safe_regions, Tf_reach=self.Tf_reach)
        self.safe_regions = safe_regions
        
        #Convert current robot's attitude to indices (slice of the reachable set)
        self.theta_slice = np.argmin(np.abs(self.hjr_fno.theta_array - (self.robot_state[2] % (2*np.pi))))
        self.time_slice = np.argmin(np.abs(self.hjr_fno.time_array - self.Tf_reach))
        # self.hjr_fno.update_feasible_set(self.theta_slice, self.time_slice)
        
        self.path = [] #robot's path
        self.multi_paths = [[] for _ in range(len(self.other_goals))]

        self.env = env #env.Env(safe_regions=safe_regions)
        self.plotting = plotting #plotting.Plotting(x_start, x_goal, safe_regions=safe_regions)
        self.utils = util #utils.Utils()

        # plotting
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
        
        # PriorityQueue with respect to other goal location
        #
        # Q_ij structure:
        # key   -> j-th goal (tuple)
        # value -> heap of (dist, node_id, node)
        # self.Q_other_goals: Dict[Tuple[float, float], List[Tuple[float, int, Node]]] = {goal_j: [] for goal_j in other_goals}
        self.Q_other_goals = {}

        for goal_j in self.other_goals:
            d = math.hypot(
                self.s_goal.x - goal_j.x,
                self.s_goal.y - goal_j.y
            )
            self.Q_other_goals[goal_j] = [
                (d, self.s_goal.id, self.s_goal)
            ]
            heapq.heapify(self.Q_other_goals[goal_j])
        

    def _dist_to_goal(self, node: Node, goal: Tuple[float, float]) -> float:
        dx = node.x - goal[0]
        dy = node.y - goal[1]
        return math.hypot(dx, dy)


    def planning(self):

        # set seed for reproducibility
        # np.random.seed(0)

        # set up event handling
        # self.fig.canvas.mpl_connect('button_press_event', self.update_obstacles)

        # animation stuff
        plt.gca().set_aspect('equal', adjustable='box')
        self.edge_col.set_animated(True)
        self.path_col.set_animated(True)
        plt.show(block=False)
        plt.pause(0.1)
        self.ax.draw_artist(self.edge_col)
        self.fig.canvas.blit(self.ax.bbox)
        start_time = time.time()
        prev_plotting = time.time()
        first_time = True

        for i in range(self.iter_max):
            
            # update robot position
            run_time = time.time() - start_time
            
            # if there is path to goal and run_time > 5s, then start moving the robot
            if self.path_to_goal.any() and run_time > 5:
                # timing stuff
                if first_time:
                    prev_time = time.time()
                    first_time = False
                elapsed_time = time.time() - prev_time
                prev_time = time.time()
                
                #Terminate when reach the goal
                if self.s_bot.cost_to_goal == 0.0 and self.s_bot.lmc == 0.:
                    print("Successfully reach the goal!")
                    break

                # Update robot position
                # self.robot_position = self.utils.update_robot_position(
                #     self.robot_position, 
                #     [self.s_bot.parent.x, self.s_bot.parent.y], 
                #     self.robot_speed, 0.01)
                
                '''
                self.robot_state = self.utils.update_robot_position_dubins(self.robot_state, [self.s_bot.parent.x, self.s_bot.parent.y], 0.01, v=self.robot_speed)
                self.robot_position = self.robot_state[:2]
                
                #Convert current robot's attitude to indices (slice of the reachable set)
                self.theta_slice = np.argmin(np.abs(self.hjr_fno.theta_array - (self.robot_state[2] % (2*np.pi))))
                
                
                # TODO time_slice would only update when we start contingency planning
                # self.time_slice = np.argmin(np.abs(self.hjr_fno.time_array - self.Tf_reach))
                
                # Lidar radial detection of the obstacles
                self.unknown_obs_circle, detected_obs = self.utils.lidar_detected(self.robot_position, sensing_radius=self.lidar_range)
                
                if len(detected_obs) > 0:
                    
                    print(f" ----- Rewiring Trees due to {len(detected_obs)} newly-detected obstacle(s) ----- ")
                    
                    # TODO update HJB reachable set with the detected_obs here
                    exec_time_start = time.time()
                    self.hjr_fno.update_obs(detected_obs)
                    print("Update Reachable set:", time.time() - exec_time_start, "s")
                    
                    # update known obstacles withint the environment
                    for obs in detected_obs:
                        self.update_obstacles(obs)
                '''       
                
                    
                                    
                # As the robot propagates (agnostic to its dynamics), 
                # Check if the robot is close enough to the its parent node,
                # If so, then set the parent node as the current node associated with the robot (i.e., s_bot)
                #
                # NOTE: - The tree is rooted at the goal
                #       - Each node’s parent is closer to the goal
                #       - Following parents walks the path backward along the tree
                
                '''
                # update node that robot is "at"
                if math.hypot(self.robot_position[0] - self.s_bot.parent.x,
                              self.robot_position[1] - self.s_bot.parent.y) < 1:
                    self.s_bot = self.s_bot.parent
                '''
                
            # animate
            '''
            if run_time > 5 or run_time < 0.01:

                # only update the plot at 5 Hz
                elapsed_plotting = time.time() - prev_plotting
                if elapsed_plotting >= 0.2:
                    prev_plotting = time.time()
                    self.edges = []
                    for node in self.tree_nodes:
                        if node.parent:
                            self.edges.append(np.array([[node.parent.x, node.parent.y], [node.x, node.y]]))
                   
                   
                    
                    # ================= PLOTTING =======================
                                        
                    # clear axes
                    self.ax.clear()

                    # restore static axis properties
                    self.ax.set_xlim(self.env.x_range[0], self.env.x_range[1] + 1)
                    self.ax.set_ylim(self.env.y_range[0], self.env.y_range[1] + 1)

                    # draw environment
                    self.plotting.plot_env(self.ax)

                    # draw tree nodes
                    if self.all_nodes_coor:
                        nodes = np.array(self.all_nodes_coor)
                        self.ax.scatter(nodes[:, 0], nodes[:, 1], s=4, c='gray', alpha=0.5)

                    # draw tree edges
                    if self.edges:
                        edge_col = LineCollection(self.edges, colors='blue', linewidths=0.5, alpha=0.5)
                        self.ax.add_collection(edge_col)

                    # draw path to goal
                    if self.path:
                        path_col = LineCollection(self.path, colors='red', linewidths=1.0)
                        self.ax.add_collection(path_col)

                    # draw robot + lidar
                    self.plotting.plot_robot(self.ax, self.robot_position, self.lidar_range)
                    
                    # plot reachable set at current heading
                    self.plotting.plot_reachable_set(self.ax, self.hjr_fno, self.theta_slice, self.time_slice)

                    # force redraw
                    plt.pause(0.001)
                    
                    # ================= END OF PLOTTING =======================
                '''

            self.search_radius = self.shrinking_ball_radius()
            
            #only randomly sampled from inside reachable sets
            # self.hjr_fno.update_feasible_set(self.theta_slice, self.time_slice)
            if np.random.random() < self.prob_q:
             
                v = self.random_node()
                v_nearest = self.nearest(v)
                
                v = self.saturate(v_nearest, v)
                
            else:
                
                goal_j = random.choice(self.other_goals)
                
                if goal_j.parent is not None:
                    continue
                else:
                    # v_nearest = self.get_closest_node_to_goal(goal_j)
                    v_nearest = self.nearest(goal_j)
                    
                    v = self.saturate(v_nearest, goal_j)
                

            if v and not self.utils.is_collision(v_nearest, v) and self.is_feasible_ray(v_nearest, v):
                    
                self.extend(v, v_nearest)
                
                if v.parent is not None:
                    self.rewire_neighbours(v)
                    self.reduce_inconsistency()

            '''
            if self.s_bot.cost_to_goal < np.inf:
                self.path_to_goal = True
            '''
            
            for j, goal_j in enumerate(self.other_goals):
                if goal_j.cost_to_goal < np.inf:
                    # print(f"RRTX planning loop, node {goal_j.x, goal_j.y}, cost_to_go {goal_j.cost_to_goal}")
                    self.path_to_goal[j] = True
            
                    
    
    def reset_robot(self, start_id:int, heading:float):
        '''
        Here we assume the robot is reset at the goal location and when it is free of obstacles
        '''
        j = self.other_goals_id.index(start_id)
        self.s_bot = self.other_goals[j]
        self.curr_tree_idx = j
        
        
        #Connect to an existing tree
        if self.path_to_goal[j]:
            self.robot_path_to_goal = True
            self.update_path(self.s_bot)
        
        self.robot_state = [self.s_bot.x, self.s_bot.y, heading]
        self.robot_position = [self.s_bot.x, self.s_bot.y]
                 
    def planning_with_robot(self , steps=10):

        # animation stuff
        # plt.gca().set_aspect('equal', adjustable='box')
        # self.edge_col.set_animated(True)
        # self.path_col.set_animated(True)
        # plt.show(block=False)
        # plt.pause(0.1)
        # self.ax.draw_artist(self.edge_col)
        # self.fig.canvas.blit(self.ax.bbox)
        # start_time = time.time()
        # prev_plotting = time.time()
        # first_time = True
        
        new_obs_flag = False
                
        for _ in range(steps): #set plotting Hz
            
            
            # # update robot position
            # run_time = time.time() - start_time
        
            # if there is path to goal and run_time > 5s, then start moving the robot
            if self.robot_path_to_goal:
                # # timing stuff
                # if first_time:
                #     prev_time = time.time()
                #     first_time = False
                # elapsed_time = time.time() - prev_time
                # prev_time = time.time()
                
                #Terminate when reach the goal
                if self.s_bot.cost_to_goal == 0.0 and self.s_bot.lmc == 0.:
                    return new_obs_flag
                
                # Update robot position            
                self.robot_state = self.utils.update_robot_position_dubins(self.robot_state, [self.s_bot.parent.x, self.s_bot.parent.y], 0.01, v=self.robot_speed)
                self.robot_position = self.robot_state[:2]
                
                #Convert current robot's attitude to indices (slice of the reachable set)
                self.theta_slice = np.argmin(np.abs(self.hjr_fno.theta_array - (self.robot_state[2] % (2*np.pi))))
                
                
                
                # TODO time_slice would only update when we start contingency planning
                # self.time_slice = np.argmin(np.abs(self.hjr_fno.time_array - self.Tf_reach))
                
                # Lidar radial detection of the obstacles
                self.unknown_obs_circle, detected_obs = self.utils.lidar_detected(self.robot_position, sensing_radius=self.lidar_range)
                
                if len(detected_obs) > 0:
                    
                    print(f" ----- Rewiring Trees due to {len(detected_obs)} newly-detected obstacle(s) ----- ")
                    
                    # TODO update HJB reachable set with the detected_obs here
                    exec_time_start = time.time()
                    self.hjr_fno.update_obs(detected_obs)
                    print("Update Reachable set:", time.time() - exec_time_start, "s")
                    
                    # update known obstacles withint the environment
                    for obs in detected_obs:
                        self.update_obstacles(obs, robots_plan=True)
                        
                    new_obs_flag = True
                 
                
                    
                                    
                # As the robot propagates (agnostic to its dynamics), 
                # Check if the robot is close enough to the its parent node,
                # If so, then set the parent node as the current node associated with the robot (i.e., s_bot)
                #
                # NOTE: - The tree is rooted at the goal
                #       - Each node’s parent is closer to the goal
                #       - Following parents walks the path backward along the tree
                
                
                # update node that robot is "at"
                if math.hypot(self.robot_position[0] - self.s_bot.parent.x,
                              self.robot_position[1] - self.s_bot.parent.y) < 1:
                    self.s_bot = self.s_bot.parent
                    self.other_goals[self.curr_tree_idx] = self.s_bot
                
            '''  
            # animate
            if run_time > 2 or run_time < 0.01:

                # only update the plot at 5 Hz
                elapsed_plotting = time.time() - prev_plotting
                if elapsed_plotting >= 0.2:
                    prev_plotting = time.time()
                    self.edges = []
                    for node in self.tree_nodes:
                        if node.parent:
                            self.edges.append(np.array([[node.parent.x, node.parent.y], [node.x, node.y]]))
                   
                   
                    
                    # ================= PLOTTING =======================
                                        
                    # clear axes
                    self.ax.clear()

                    # restore static axis properties
                    self.ax.set_xlim(self.env.x_range[0], self.env.x_range[1] + 1)
                    self.ax.set_ylim(self.env.y_range[0], self.env.y_range[1] + 1)

                    # draw environment
                    self.plotting.plot_env(self.ax)

                    # draw tree nodes
                    if self.all_nodes_coor:
                        nodes = np.array(self.all_nodes_coor)
                        self.ax.scatter(nodes[:, 0], nodes[:, 1], s=4, c='gray', alpha=0.5)

                    # draw tree edges
                    if self.edges:
                        edge_col = LineCollection(self.edges, colors='blue', linewidths=0.5, alpha=0.5)
                        self.ax.add_collection(edge_col)

                    # draw path to goal
                    if self.path:
                        path_col = LineCollection(self.path, colors='red', linewidths=1.0)
                        self.ax.add_collection(path_col)

                    # draw robot + lidar
                    self.plotting.plot_robot(self.ax, self.robot_position, self.lidar_range)
                    
                    # plot reachable set at current heading
                    self.plotting.plot_reachable_set(self.ax, self.hjr_fno, self.theta_slice, self.time_slice)

                    # force redraw
                    plt.pause(0.001)
                    
                    # ================= END OF PLOTTING =======================
            '''

            self.search_radius = self.shrinking_ball_radius()
            
            #only randomly sampled from inside reachable sets
            # self.hjr_fno.update_feasible_set(self.theta_slice, self.time_slice)
             
            v = self.random_node(robots_plan=True)
            v_nearest = self.nearest(v)
            
            v = self.saturate(v_nearest, v)
                
            if v and not self.utils.is_collision(v_nearest, v) and self.is_feasible_ray(v_nearest, v):
                    
                self.extend(v, v_nearest, robots_plan=True)
                
                if v.parent is not None:
                    self.rewire_neighbours(v, robots_plan=True)
                    self.reduce_inconsistency(robots_plan=True)

            
            if self.s_bot.cost_to_goal < np.inf:
                self.robot_path_to_goal = True
                self.path_to_goal[self.curr_tree_idx] = True
                
        return new_obs_flag
            
            
            
                
                
    #finding nearest nodes to other targets
    def get_closest_node_to_goal(self, goal_j):
        heap = self.Q_other_goals[goal_j]
        
        # print(f"Priority Queue Q_ij for goal {goal_j}:")
        # print("(distance, node_id, (x, y))")

        # for dist, node_id, node in sorted(heap, key=lambda x: x[0]):
        #     if node.active:
        #         print(f"{dist:.3f}, {node_id}, ({node.x:.2f}, {node.y:.2f})")

        while heap:
            dist, node_id, node = heap[0]
            if node.active: 
                return node
            heapq.heappop(heap)  # discard invalid node

        return None


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
        
        exec_time_start = time.time()
        self.add_new_obstacle(obs_cir)   #TODO: This function sometimes takes a bit of time
        print("Added new obstacles:", time.time() - exec_time_start, "s")
        
        exec_time_start = time.time()
        self.propagate_descendants(robots_plan=robots_plan)
        print("Propagate Descendants:", time.time() - exec_time_start, "s")
        
        exec_time_start = time.time()
        self.verify_queue(self.s_bot)
        print("Verify Queue:", time.time() - exec_time_start, "s")
        
        exec_time_start = time.time()
        self.reduce_inconsistency(robots_plan=robots_plan)
        print("Reduce Inconsistency:", time.time() - exec_time_start, "s")

    def add_new_obstacle(self, obs):
        # Algorithm 12
        x, y, r = obs
        # print("Osbstacle at: x =", x, ", y =", y, ", r = ", r)
        self.obs_circle.append(obs)
        self.plotting.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle, self.unknown_obs_circle) # for plotting obstacles
        self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for collision checking
        self.update_gamma() # free space volume changed, so gamma must change too

        # Get all edges that intersect the new circle obstacle
        # E_O = [(v, u) for v in self.tree_nodes if (u:=v.parent) and self.utils.is_intersect_circle(*self.utils.get_ray(v, u), obs[:2], obs[2])]
        
        # NOTE Collect all directed node pair (v->u) that intersects with the obstacles
        #
        # for all nodes 'v' in tree_nods 
        #       for all nodes 'u' in neighborhood of 'v' (include static and running nodes)
        #               check if the edge v -> u intersected with 
        E_O = [(v, u) 
                for v in self.tree_nodes
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

        
        # check if robot node got orphaned
        if robots_plan:
            if self.s_bot in self.orphan_nodes:
                print('robot node got orphaned')
                self.robot_path_to_goal = False
        
        # else:
        #Check if path between goal got orphaned
        self.path_to_goal[[i for i, g in enumerate(self.other_goals) if g in self.orphan_nodes]] = False

        
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
                v.active = False
                
            except ValueError:
                pass

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
              
        # while len(self.Q) > 0 and (self.Q[0][0] < self.s_bot.get_key() \
        #         or self.s_bot.lmc != self.s_bot.cost_to_goal or np.isinf(self.s_bot.cost_to_goal) \
        #         or self.s_bot in list(zip(*self.Q))[1]):
            
        # while len(self.Q) > 0 and ( any(  [self.Q[0][0] < g.get_key()
        #                                 or g.lmc != g.cost_to_goal
        #                                 or np.isinf(g.cost_to_goal)
        #                                 or g in list(zip(*self.Q))[1]
                                        
        #                                 for g in self.other_goals]) ) and ():
        
        while len(self.Q) > 0  and (
                (
                    robots_plan and (
                        self.Q[0][0] < self.s_bot.get_key()
                        or self.s_bot.lmc != self.s_bot.cost_to_goal
                        or np.isinf(self.s_bot.cost_to_goal)
                        or self.s_bot in {node for _, node in self.Q}
                    )
                )
                or
                (
                    any(
                        self.Q[0][0] < g.get_key()
                        or g.lmc != g.cost_to_goal
                        or np.isinf(g.cost_to_goal)
                        or g in {node for _, node in self.Q}
                        for g in self.other_goals
                    )
                )
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
        self.all_nodes_coor.append(np.array([node_new.x, node_new.y])) # for plotting
        self.tree_nodes.append(node_new)
        
        #update priority queue with distance to other target locations
        node_new.active = True
        for goal_j, heap in self.Q_other_goals.items():
            d = self._dist_to_goal(node_new, goal_j)
            heapq.heappush(heap, (d, node_new.id, node_new))
        
        self.kd_tree.add(node_new)
        
        # if new node is at start, then path to goal is found
        if robots_plan:
            
            if node_new == self.s_bot:
                self.s_bot = node_new
                self.robot_path_to_goal = True
                self.update_path(self.s_bot) # update path to goal for plotting
                
                self.other_goals[self.curr_tree_idx] = self.s_bot
                self.update_multi_paths(self.s_bot, self.curr_tree_idx)
                self.path_to_goal[self.curr_tree_idx] = True
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
                        self.verify_queue(u)

        if robots_plan:
            self.update_path(self.s_bot) # update path to goal for plotting
            self.other_goals[self.curr_tree_idx] = self.s_bot
            self.update_multi_paths(self.s_bot, self.curr_tree_idx)
            
        # else:
        for j, goal_j in enumerate(self.other_goals):
            self.update_multi_paths(goal_j, j)

    def random_node(self, robots_plan=False):
        
        delta = self.utils.delta

        # if path to goal is already found,
        # returns a node located exactly at the robot’s current position (no randomness) with probability of bot_sample_rate
        #
        if robots_plan and (not self.robot_path_to_goal) and (np.random.random() < self.bot_sample_rate):
            return Node((self.s_bot.x, self.s_bot.y))
        
        #------------------------
        
        # uniform random Node inside the env space
        # return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
        #             np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))
        
        #------------------------
        
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
        
        mu_X_free = (self.x_range[1] - self.x_range[0]) * (self.y_range[1] - self.y_range[0]) * 1/3
        for (_, _, r) in self.obs_circle:
            mu_X_free -= np.pi * r ** 2
        for (_, _, w, h) in self.obs_rectangle:
            mu_X_free -= w * h

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
                        
        for t in np.linspace(0,1,3):
            shot = Node((o[0] + t * d[0], o[1] + t * d[1]))
            
            if not self.hjr_fno.is_feasible(shot, self.theta_slice, self.time_slice):
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
        x_start: Tuple[float, float],
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
        
        
        self.iter_max = iter_max
        
        self.env = env.Env(safe_regions=safe_regions)
        self.plotting = plotting.Plotting(x_start, x_goal, safe_regions=safe_regions)
        self.utils = utils.Utils()
        
        #HJR-FNO configs
        self.Tf_reach = 7.5 #must be less than 8s (underapproximation of the training data)
        self.hjr_fno = HJR_FNO(safe_regions=safe_regions, Tf_reach=self.Tf_reach)

        # plotting
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.suptitle('SFF-RRTX')
        self.ax.set_xlim(self.env.x_range[0], self.env.x_range[1]+1)
        self.ax.set_ylim(self.env.y_range[0], self.env.y_range[1]+1)      


        #Define different tree branches rooted at each target location
        self.rrtx_trees = {}

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
                iter_max=self.sub_iter_count,
                safe_regions=safe_regions,
                env=self.env,
                plotting=self.plotting,
                util=self.utils,
                hjr_fno=self.hjr_fno,
                fig= self.fig,
                ax= self.ax
            )
        

        
        # Define distance matrix 
        # D_ij means i-th tree distance to j-th goal; if optimal path is not found, D_ij = np.inf
        n = len(x_goal)
        self.D = np.full((n, n), np.inf)
        np.fill_diagonal(self.D, 0.0)

        
        #Color map
        cmap = plt.get_cmap("hsv")
        self.colorList = [cmap(i) for i in np.linspace(0, 1, len(x_goal), endpoint=False)]
        

    def print_distance_matrix(self, precision=3):
        n = self.D.shape[0]

        # Header
        header = "      " + "".join(f"{j:^10}" for j in range(n))
        print(header)
        print("      " + "-" * (10 * n))

        for i in range(n):
            row_str = f"{i:^4} |"
            for j in range(n):
                if np.isinf(self.D[i, j]):
                    row_str += f"{'∞':^10}"
                else:
                    row_str += f"{self.D[i, j]:^10.{precision}f}"
            print(row_str)
            
    def generate_atsp_file(self, filename, distance_matrix, name="SampleATSP", comment="Generated by Python"):
        """
        Generates a .atsp file from a given distance matrix.

        Args:
            filename (str): The name of the output file (e.g., 'my_problem.atsp').
            distance_matrix (list of lists or numpy array): The N x N distance matrix.
            name (str): The problem name.
            comment (str): A brief description.
        """
        dimension = len(distance_matrix)
        if any(len(row) != dimension for row in distance_matrix):
            raise ValueError("Distance matrix must be square (N x N)")

        with open(filename, 'w') as f:
            # Write the Specification Part
            f.write(f"NAME: {name}\n")
            f.write(f"TYPE: ATSP\n")
            f.write(f"COMMENT: {comment}\n")
            f.write(f"DIMENSION: {dimension}\n")
            f.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\n")
            f.write(f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
            f.write(f"EDGE_WEIGHT_SECTION\n")

            # Write the Data Part (distance matrix)
            for row in distance_matrix:
                # Join elements with a space and add a newline
                f.write(" ".join(map(str, row)) + "\n")
            
            # Write the End of File marker
            f.write("EOF\n")

        
    def planning(self):
        
        iter = 0
        prev_plotting = time.time()
        
        
        for i in range(len(self.rrtx_trees)):
            
            print(f"Initialize the tree {i}")
            
            self.rrtx_trees[i].planning()
            iter += self.sub_iter_count
        
        while iter < self.iter_max:
            
            tree_id, rrtx = random.choice(list(self.rrtx_trees.items()))
            print(f"tree {tree_id}, goal: {(rrtx.s_goal.x, rrtx.s_goal.y)}")
            rrtx.planning()
            
            # only update the plot at 5 Hz
            elapsed_plotting = time.time() - prev_plotting
            if elapsed_plotting >= 0.2:
                prev_plotting = time.time()
                
            
                # ================= PLOTTING =======================
                                            
                # clear axes
                self.ax.clear()
                
                # restore static axis properties
                self.ax.set_xlim(self.env.x_range[0], self.env.x_range[1] + 1)
                self.ax.set_ylim(self.env.y_range[0], self.env.y_range[1] + 1)

                # draw environment
                self.plotting.plot_env(self.ax, colorList=self.colorList)

                for i, tree_i in enumerate(self.rrtx_trees.values()):
                    
                    
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
                    self.plotting.plot_reachable_set(self.ax, self.hjr_fno, tree_i.theta_slice, tree_i.time_slice)

                # force redraw
                plt.pause(0.001)
            
                # ================= END OF PLOTTING =======================
            
            iter += self.sub_iter_count
    
        
        for i, tree_i in self.rrtx_trees.items():
            for local_j, global_j in enumerate(tree_i.other_goals_id):
                if tree_i.path_to_goal[local_j]:
                    self.D[i, global_j] = tree_i.other_goals[local_j].cost_to_goal
                    print(f"index {(i, global_j)}, cost: {tree_i.other_goals[local_j].lmc}")

        self.print_distance_matrix()
        
        # draw environment
        self.plotting.plot_env(self.ax, colorList=self.colorList)
        # plt.show()
        
        
        #TODO Asymmetric TSP solver
        # atsp_file_name = "/home/kmuenpra/git/Asymmetric-Travelling-Salesman-Problem-Optimized-by-Simulated-Annealing/test1.atsp"
        # self.generate_atsp_file(filename=atsp_file_name, distance_matrix=np.round(self.D * 1000).astype(int))
        
        # Simulate Robot's plan
        atsp_sequence = [0, 2, 3, 4, 1, 0]
        
        heading = 0.0
        prev_id = 100
        obs_list = []
        for i, id in enumerate(atsp_sequence):
            
            if i == 0:
                prev_id = id
                continue
            
            # x_start = (self.rrtx_trees[prev_id].s_goal.x, self.rrtx_trees[prev_id].s_goal.y)
            
            print("set robot position")
            self.rrtx_trees[id].reset_robot(start_id=prev_id, heading=heading)
            
            print("Update Obstacles")
            if obs_list:
                print(obs_list)
                print(self.rrtx_trees[id].obs_circle)
                
                #Update orphan node list:
                # self.rrtx_trees[id].update_gamma() # free space volume changed, so gamma must change too
                
                for obs in obs_list:
                    E_O = [(v, u) 
                            for v in self.rrtx_trees[id].tree_nodes
                                for u in v.all_out_neighbors() 
                                    if self.utils.is_intersect_circle(*self.utils.get_ray(v, u), obs[:2], obs[2])]
                    
                    # To preserve graph structure:
                    # instead of removing (v->u) from the neighbor set, make the edge haviing infinite cost
                    for v, u in E_O:
                        v.infinite_dist_nodes.add(u)
                        u.infinite_dist_nodes.add(v)
                        if v.parent and v.parent == u:
                            self.rrtx_trees[id].verify_orphan(v)
                            # should theoretically check if the robot is on this edge now, but we do not
                            # v.parent.children.remove(v) # these two lines are from the Julia code
                            # v.parent = None 
                            
                    heapq.heapify(self.rrtx_trees[id].Q) # reheapify after removing a bunch of elements and ruining queue

                
                self.rrtx_trees[id].propagate_descendants(robots_plan=True)                
                self.rrtx_trees[id].reduce_inconsistency(robots_plan=True)
                
            
            
            print(f"Starting planning from Target#{prev_id} to Target#{id}")
            for _ in range(self.iter_max):
            
                new_obs_flag = self.rrtx_trees[id].planning_with_robot(steps=10) #run 10 steps
                
                # only update the plot at 5 Hz
                elapsed_plotting = time.time() - prev_plotting
                if elapsed_plotting >= 0.2:
                    prev_plotting = time.time()
                
                    # ====continue============= PLOTTING =======================
                                                
                    # clear axes
                    self.ax.clear()

                    # restore static axis properties
                    self.ax.set_xlim(self.env.x_range[0], self.env.x_range[1] + 1)
                    self.ax.set_ylim(self.env.y_range[0], self.env.y_range[1] + 1)

                    # draw environment
                    self.plotting.plot_env(self.ax)

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
                    self.plotting.plot_reachable_set(self.ax, self.hjr_fno, self.rrtx_trees[id].theta_slice, self.rrtx_trees[id].time_slice)
                    
                    
                    if new_obs_flag:
                        
                        print(self.rrtx_trees[id].curr_tree_idx)
                        print(self.rrtx_trees[id].multi_paths[self.rrtx_trees[id].curr_tree_idx])
                        print(self.rrtx_trees[id].path_to_goal[self.rrtx_trees[id].curr_tree_idx])
                        path_col = LineCollection(self.rrtx_trees[id].multi_paths[self.rrtx_trees[id].curr_tree_idx], colors=self.colorList[id], linewidths=2.5)
                        
                        # draw path to goal
                        for j, goal_j in enumerate(self.rrtx_trees[id].other_goals):
                            if self.rrtx_trees[id].path_to_goal[j]:
                                path_col = LineCollection(self.rrtx_trees[id].multi_paths[j], colors=self.colorList[id], linewidths=2.5)
                                self.ax.add_collection(path_col)
                    
                        plt.pause(3)
                    else:

                        # force redraw
                        plt.pause(0.001)
                    
                    # ================= END OF PLOTTING =======================
                    
                #Terminate when reach the goals
                if self.rrtx_trees[id].s_bot.cost_to_goal == 0.0 and self.rrtx_trees[id].s_bot.lmc == 0.0:
                    print("Successfully reach the goal!")
                    break
                
            prev_id = id
            obs_list = self.rrtx_trees[id].obs_circle
            heading = self.rrtx_trees[id].robot_state[2]

def main():
    x_start = (-11, -17, 0)  # Starting node
    x_goal = [(-20, 23),  (-11, -17),  (5, 13), (15, 20), (15,-15)]  # Goal node
    # x_goal = [(-20, 23), (15,-15)]  # Goal node



    sff = SFF_star(
        x_start=x_start, 
        x_goal=x_goal, 
        heading=0.0,
        lidar_range=5.0,
        step_len= 3.0, 
        gamma_FOS = 100.0,
        epsilon=0.05,
        bot_sample_rate=0.10,  
        iter_max=10000,
        safe_regions=[[-5, -1, 2],
                      [6, 8.5, 2],
                      [-10, -14, 2],
                      [18,1,2],
                      [15,15,2],
                      [-8,13,2],
                      [18,-12,2],
                      [-15,23,2]],
        )
    sff.planning()


if __name__ == '__main__':
    main()

'''
TODO:
- Fix the heading during the reset_robot to be the heading toward the first parent (Note this assumption in the code)
- when checking for the is_feasible, make it faster by checking whether the obstacles center in local frame exceeds the grid_space of HJR-FNO
    - Also try to use the official ray, rather than using np.linspace()
- Maybe retrain HJRNO model with the case where there is no obstacles, or simply pre-compute it to be used in the experiment
    >>>> when Nodel.parent is None, and Node.x and Node.y is the goal location >>> simply executed toward the goal.
- control policy (try simple switch for now), and think about condition for contingency behavior
'''