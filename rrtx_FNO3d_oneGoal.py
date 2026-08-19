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

import nmpc_dubins

# =========================
# Local / project imports
# =========================
import env
import plotting
import utils

# Shared fixed constants (eval/config.yaml). The MPPI env reads the same file,
# so dt, the speed / yaw-rate limits, the lidar radius and the goal tolerance
# are identical on both sides by construction rather than by coincidence.
from eval.config import CFG, load_config

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
        # children is keyed by the child's unique id, NOT stored in a set.
        #
        # __hash__ below is the node's POSITION and __eq__ also matches on equal cost
        # keys, so a set cannot hold two distinct nodes that share a position -- and this
        # planner creates those routinely (random_node returns Node((s_bot.x, s_bot.y)),
        # saturate can land a sample exactly on an existing node, reset_robot_position
        # inserts at the robot's exact spot). In a set, `add` was then a silent no-op and
        # `remove` could evict the wrong twin, leaving `children` out of sync with the
        # `parent` pointers. propagate_descendants discovers orphan descendants by walking
        # `children`, so a missing entry meant a whole subtree was never marked
        # unreachable: it kept a finite, self-consistent cost while its parent became an
        # inf-cost evicted orphan. find_parent then preferred those cheap-looking
        # "zombies" and the robot drove at a dead node. Measured: 89 desynced links and
        # 211 / 1470 nodes with a finite lmc but no route to the root.
        self.children = {}   # {child.id: child}
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
        # Detach from the old parent and attach to the new one. Both operations are keyed
        # by self.id, so they address THIS node and can never hit a position twin -- which
        # is why the old set-based version needed a bare `except:` to hide its failures.
        if self.parent is not None:
            self.parent.children.pop(self.id, None)

        self.parent = new_parent
        new_parent.children[self.id] = self

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
        # `u.parent is not self` must be an IDENTITY test: `!=` runs __eq__, which also
        # matches on equal cost keys, so an unrelated node sharing self's key would make a
        # perfectly good candidate look like self's own child and get skipped.
        cands = sorted(
            ((u, self.distance(u) + u.lmc)
             for u in (self.all_out_neighbors() - orphan_nodes) if u.parent and u.parent is not self),
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



# ----------------------------------------------------------------------
# Cost predicates -- the SINGLE definition of "is there a path from here".
#
# RRTx keeps two upper bounds on the cost-to-goal: cost_to_goal (g) and the lookahead
# estimate lmc, with lmc <= g. lmc is written as soon as a parent is found
# (find_parent), whereas g is only reconciled when the node is popped from the priority
# queue -- and nothing pushes a freshly created node. So a node that was just connected
# routinely has a finite lmc together with g = inf.
#
# Reading g in some places and lmc in others is what produced the spurious
# "[WARNING] Robot is isolated" reports and the recovery loop that could never observe
# its own success. Both classes in this module must use these two helpers rather than
# touching either field directly.
# ----------------------------------------------------------------------
def path_cost(v):
    """Best available upper bound on v's cost-to-goal; inf iff no route is known."""
    if v is None:
        return np.inf
    return min(v.cost_to_goal, v.lmc)


def has_path_to_goal(v):
    """True iff a finite-cost route to the goal is known through v."""
    return v is not None and math.isfinite(min(v.cost_to_goal, v.lmc))


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
        # Does the chain from s_bot actually terminate at the root? Maintained by
        # update_path; a stale-finite lmc can make has_path_to_goal(s_bot) True while the
        # chain dead-ends at an evicted orphan, so both are required before the robot moves.
        self._path_reaches_root = False
        # Node flagged by update_path as carrying a stale finite lmc over a dead parent;
        # cleared by repair_broken_chain(). See both for why sampling cannot fix it.
        self._broken_chain_node = None
        # Global prune of unrooted-but-finite-cost nodes (repair_broken_chain). Measured
        # NET-NEGATIVE on env_D/seed 1: it deletes ~370 of ~1000 nodes per call, orphans the
        # robot again each time, and the population regenerates because eviction does not
        # stop a node participating via neighbour sets -- "83.8 m + goal reached" became
        # "107.8 m + budget exhausted". Kept but OFF -- and now redundant: with RRTx Alg 9
        # honoured (orphans stay in V), the invariant it tried to restore holds by
        # construction.
        self.prune_unrooted_enable = False
        
        #State and Sensor
        self.robot_state = [0,0,0]
        # self.robot_position = [self.s_bot.x, self.s_bot.y]
        self.robot_speed = CFG.v_max  # m/s
        # Motion-execution parameters, deliberately matched to the MPPI side so the
        # two planners run the SAME tracker with the SAME discretization. Both
        # sides now read them from eval/config.yaml, so the match is enforced
        # rather than maintained by hand:
        #   motion_dt   == Navigation2DEnv.dynamics' delta_t  == CFG.dt_c
        #   robot_w_max == env.u_max[1] (omega bound)         == CFG.omega_max
        # See Utils.update_robot_position_dubins and the pure-pursuit tracker in
        # mppi_src/guidance.py. One control step per planning_with_robot() cycle.
        self.motion_dt = CFG.dt_c        # s
        self.robot_w_max = CFG.omega_max  # rad/s

        # Rotate in place toward s_bot.parent before resuming MPC tracking after a
        # contingency retreat. Armed in main(), consumed in planning_with_robot().
        self._align_pending = False
        self._align_steps = 0

        # ---- Nonlinear MPC tracker (see mpc_reference / nmpc_dubins.py) -------
        # The tree is GEOMETRIC: bare (x, y) nodes joined by straight edges, so the
        # committed path has corners a bounded-curvature vehicle cannot drive. Rather
        # than pre-smoothing the polyline into Dubins arcs and chasing a lookahead
        # point (dubins_path.py, generate-and-test: ~20% of arcs were rejected
        # wholesale and replaced by straight edges with discontinuous headings), the
        # polyline is now tracked directly by an optimizer that carries the control
        # limits, the domain box and the reachable-set invariance constraint INSIDE
        # the solve -- so it deforms the trajectory instead of discarding candidates.
        self.turn_radius = self.robot_speed / self.robot_w_max   # rho, kept for plots
        self.mpc_enable = True        # False -> fall back to pure pursuit
        self.mpc_horizon = 10         # steps (x motion_dt = 1.0 s lookahead)
        self.mpc_ref_span = 8.0       # how far along the path to build the reference [m]
        # The MPC carries the buffer here and nowhere else (see mpc_slack_fn).
        self.mpc_slack_margin = hjr_fno.feasibility_buffer
        self.mpc_ref = None           # (N+1, 2) reference: tracked + plotted
        self.mpc_pred = None          # (N+1, 3) predicted plan: plotted
        self.mpc_fallbacks = 0        # solves that failed -> pure pursuit (diagnostic)
        self.mpc = nmpc_dubins.DubinsNMPC(
            dt=self.motion_dt,
            horizon=self.mpc_horizon,
            v_min=0.0,
            v_max=self.robot_speed,
            w_max=self.robot_w_max,
            state_box=(self.env.x_range[0], self.env.x_range[1],
                       self.env.y_range[0], self.env.y_range[1]),
            slack_margin=self.mpc_slack_margin,
        )

        # ---- execution-time safety filter (see safe_pure_pursuit_step) --------
        # RRTX certifies its EDGES at insertion time, but the pure-pursuit tracker
        # drives an arc, not the edge, so the executed motion can leave the
        # certified straight line (worst case ~2*v/w_max = 1.2 m, more than the
        # 0.5 m obstacle inflation). This filter re-checks the motion the tracker
        # is about to execute, and is the direct analogue of MPPI's RBR filter
        # (constraint_func=env.points_safe under use_rbr).
        self.exec_filter = True
        self.filter_horizon = 15          # steps of pure pursuit to look ahead
        # speeds to try, largest first; 0.0 = rotate in place (unicycle, v may be 0)
        self.filter_speed_scales = (1.0, 0.5, 0.25, 0.0)
        # diagnostics: how the filter resolved each control step
        self.filter_counts = {"nominal": 0, "slowed": 0, "rotate": 0, "blocked": 0}
        # A purely reactive filter can deadlock: if the pursued waypoint is itself
        # unreachable the robot rotates in place forever. Escalate to the
        # contingency behaviour after this many consecutive zero-progress steps
        # (50 steps x 0.1 s = 5 s) instead of stalling silently.
        self.filter_stall_limit = 50
        self._filter_stall = 0

        # NOTE: for evaluating per-step solve time
        # eval.episode_log.EpisodeRecorder, set by the driver. None -> no logging.
        self.recorder = None
        # False under --no_plot: skips the GUI event pump inside the timed block.
        self.interactive = True
        # Control steps executed. Counted independently of the recorder so
        # --max_steps means the same thing with and without --no_log.
        self.steps_taken = 0

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

                # robots_plan must be passed through: random_node's bias toward the robot
                # (and toward a pending reset target) is gated on it, so omitting it made
                # that branch dead code -- and with a single goal the `candidates` branch
                # below is empty too, leaving nothing biased toward the robot at all.
                v = self.random_node(robots_plan=robots_plan)

            else:

                #randomly sample the goal that hasn't been reached, OR the robot's current position
                candidates = [g for g, reached in zip(self.other_goals, self.path_to_goal) if not reached]

                if candidates:
                    v = random.choice(candidates)

                elif robots_plan and (not self.robot_path_to_goal) and self.s_bot is not None:
                    v = Node((self.s_bot.x, self.s_bot.y))

                else:

                    v = self.random_node(robots_plan=robots_plan)
                
            v_nearest = self.nearest(v)
            v = self.saturate(v_nearest, v)

            # Refuse a sample that coincides with the node it saturated against: saturate()
            # returns the sample unchanged when it already sits within step_len, so repeated
            # samples of the SAME point stack coincident nodes, and a pile of k of them costs
            # k feasibility calls on every later sample landing nearby. Free -- it reuses the
            # distance saturate() just computed.
            if v is None or math.hypot(v.x - v_nearest.x, v.y - v_nearest.y) < 1e-6:
                continue

            if not self.utils.is_collision(v_nearest, v) and self.is_feasible_ray(v_nearest, v):
                    
                self.extend(v, v_nearest, robots_plan=robots_plan)
                
                if v.parent is not None:
                    self.rewire_neighbours(v, robots_plan=robots_plan)
                    self.reduce_inconsistency(robots_plan=robots_plan)

            
            if robots_plan and self._has_validated_route():
                self.robot_path_to_goal = True
            
            
            for j, goal_j in enumerate(self.other_goals):
                if self.has_path_to_goal(goal_j):
                    self.path_to_goal[j] = True
                    
    def repair_broken_chain(self):
        """Enforce the invariant `finite lmc => the parent chain reaches the root`.

        Orphaning just the one node update_path flagged is NOT enough, and that failure is
        instructive: it treated a symptom of evicting orphans from V (a deviation from RRTx
        Alg 9, since removed) rather than the cause.

        So sweep globally: BFS from the root over the child index derived from `parent`
        pointers (authoritative -- a node's own `children` record may disagree), then orphan
        every node that the BFS did not reach but which still advertises a finite lmc. After
        this, any node passing find_parent's filter genuinely reaches the root, so the robot
        can only re-attach to something real.

        One O(len(tree_nodes)) pass, negligible beside the FNO feasibility queries. Call
        from a quiescent point (never mid-rewire). Returns True if anything was pruned.
        """
        self._broken_chain_node = None

        kids = {}
        for nd in self.tree_nodes:
            if nd.parent is not None:
                kids.setdefault(nd.parent.id, []).append(nd)

        rooted = {self.s_goal.id}
        stack = [self.s_goal]
        while stack:
            n = stack.pop()
            for c in kids.get(n.id, ()):
                if c.id not in rooted:
                    rooted.add(c.id)
                    stack.append(c)

        unrooted = [nd for nd in self.tree_nodes
                    if nd.id not in rooted and math.isfinite(nd.lmc)]
        if not unrooted:
            return False

        print(f"[repair] pruning {len(unrooted)} node(s) with a finite cost but no route "
              f"to the root (of {len(self.tree_nodes)})")
        for nd in unrooted:
            self.verify_orphan(nd)
        self.propagate_descendants(robots_plan=True)
        return True

    def _has_validated_route(self):
        """Does the robot have a route we are willing to drive?

        Both halves are required. has_path_to_goal(s_bot) only says min(cost_to_goal, lmc)
        is finite, and an lmc can be stale-finite on a branch whose ancestor was orphaned;
        _path_reaches_root (set by update_path) says the chain actually terminates at the
        root. Driving on the first alone is what sent the robot up dead branches.
        """
        return self.has_path_to_goal(self.s_bot) and self._path_reaches_root

    def _go_pending(self, new_position, heading):
        """Record the robot at `new_position` as not-yet-connected and arm the reconnect.

        The single exit shared by every failure branch of reset_robot_position, so they all
        leave the same well-defined state: the robot's true position recorded, a pending
        target for random_node()/_try_connect_pending to work toward, and the stale s_bot
        orphaned so nothing reads a cost describing a route from where the robot no longer
        is. Always returns False ("not connected").
        """
        self._pending_reset_target = Node(new_position)   # original, unsaturated
        self._pending_reset_heading = heading

        # Robot state is recorded even though it is not in the tree yet.
        self.robot_position = list(new_position)
        self.robot_state = [new_position[0], new_position[1],
                            heading if heading is not None else self.robot_state[2]]
        self.robot_path_to_goal = False
        self.path = []
        self.path_node = []

        if self.s_bot is not None and self.s_bot.parent is not None:
            self.verify_orphan(self.s_bot)
            self.propagate_descendants(robots_plan=True)
        return False

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

        # Orphans stay in the graph (RRTx Alg 9 keeps them in V), so `nearest` may well be
        # one. `lmc < inf` is the test that matters and the only one the paper relies on:
        # an orphan carries lmc = inf and is rejected implicitly, and it becomes a valid
        # snap target again the moment rewireNeighbors gives it a finite cost.
        if dist < SNAP_THRESHOLD and nearest.lmc < np.inf:
            # Reuse an existing well-connected node
            self.s_bot = nearest
            self._pending_reset_target = None

            # The reused node is very likely INCONSISTENT: extend()/find_parent() give a
            # new node its lmc but leave cost_to_goal at the inf default, and nothing
            # queues it, so reduce_inconsistency never reconciles the two unless the node
            # happens to be popped for some other reason. Snapping onto such a node hands
            # the robot lmc < inf together with cost_to_goal = inf -- and every consumer
            # reads cost_to_goal (the ATSP matrix, the isolation check), so a perfectly
            # connected robot is reported as having no path to any goal.
            if self.s_bot.cost_to_goal != self.s_bot.lmc:
                self.verify_queue(self.s_bot)
                self.reduce_inconsistency(robots_plan=True)

        else:
            # Wire a node in AT THE ROBOT'S TRUE POSITION -- deliberately NOT saturated.
            # saturate() returns a point one step_len from `nearest` along the direction
            # of the robot, i.e. NOT where the robot is; Step 2 below then overwrote
            # robot_state with that node's coordinates, teleporting the robot by
            # dist(robot, nearest) - step_len metres. That is how a robot at
            # (-11.00, 10.07) ended up reported at (-13.83, 4.53) right after an
            # obstacle update pruned the nodes near it.
            #
            # For the same reason the old `if not V_near: V_near = [nearest]` fallback is
            # gone: it accepted a parent at ANY distance, so the robot node could be
            # attached by an edge far longer than step_len. If nothing is close enough we
            # go pending instead, and random_node()/_try_connect_pending grow the tree
            # toward the robot in step_len increments until a real connection exists.
            # near() reads self.search_radius, which is only assigned inside the sampling
            # loops of planning() / planning_with_robot() -- so here it is whatever the last
            # loop left behind, and it is still 0.0 if no loop has run yet (the reset during
            # init_tree). A zero radius makes near() return nothing and forces every reset
            # down the pending path. Recompute it for the current tree size.
            self.search_radius = self.shrinking_ball_radius()
            V_near = [u for u in self.near(new_node)
                      if math.hypot(u.x - new_node.x, u.y - new_node.y) <= self.step_len]

            V_near_free = [u for u in V_near if not self.utils.is_collision(u, new_node)]

            if not V_near_free:
                # No collision-free connection yet — go pending; random_node() biases
                # sampling toward the robot until a real connection exists.
                return self._go_pending(new_position, heading)

            # Has collision-free neighbors — wire into tree normally
            self._pending_reset_target = None
            self.find_parent(new_node, V_near_free)

            if new_node.parent is None:
                # find_parent now refuses evicted / inf-cost parents, so this fires whenever
                # every nearby node is itself unreachable. It has to behave exactly like the
                # no-free-neighbours case above: the old `return False` left
                # _pending_reset_target cleared (two lines up) and s_bot on the robot's
                # PREVIOUS position, with nothing scheduled to re-attach it.
                return self._go_pending(new_position, heading)

            self.add_node(new_node)

            for u in V_near_free:
                new_node.N_o_plus.add(u)
                new_node.N_o_minus.add(u)
                u.N_r_plus.add(new_node)
                u.N_r_minus.add(new_node)


            self.s_bot = new_node
            self._pending_reset_target = None
            self.rewire_neighbours(new_node, robots_plan=True)
            self.verify_queue(self.s_bot)              # <-- the robot node itself must be in Q
            self.reduce_inconsistency(robots_plan=True)

            # self.rewire_neighbours(new_node)
            # self.reduce_inconsistency()

            # self.s_bot = new_node
            # self._pending_reset_target = None

        # --- Step 2: Update robot state ---
        # The robot's state is the position it was RESET TO, never s_bot's coordinates.
        # s_bot is only the graph anchor: in the snap branch it may sit up to
        # SNAP_THRESHOLD away, which is the same small lag the executor already tolerates
        # (planning_with_robot advances s_bot to its parent once the robot is within 0.5 m).
        self.robot_position = [new_position[0], new_position[1]]
        self.robot_state = [new_position[0], new_position[1],
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
            
        if self.has_path_to_goal(self.s_bot):
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
                        
                if self.has_path_to_goal(self.s_bot):
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

        if len(self.all_nodes_coor) % 500 == 0:
            print("Total node counts: ", len(self.all_nodes_coor))
                        
        for step_idx in range(steps): #set plotting Hz
            
            # Attempt to connect a pending reset target each iteration
            if self._pending_reset_target is not None:
                # Always update pending target to current robot position
                # before attempting connection — robot may have moved since it was set
                self._pending_reset_target = Node((self.robot_position[0], 
                                                self.robot_position[1]))
                self._try_connect_pending()
                    
            if not self.has_path_to_goal(self.s_bot):
                self.robot_path_to_goal = False
            elif self.s_bot.cost_to_goal != self.s_bot.lmc:
                # finite lmc but a stale g: connected, just not reconciled yet -> queue it
                # rather than declaring the robot path-less (that mislabel is what the
                # isolation checks used to trip on).
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
            
                # RRTX Tree expands for 10 steps (for exploration)
                # Robot itself actually only moves the very last iteration
                if step_idx == steps - 1:

                    applied = False   # NOTE: for evaluating per-step solve time

                    # After a contingency: rotate in place toward s_bot.parent before
                    # resuming translation.
                    if self._align_pending:
                        # align_heading_step costs one control step, but only on the
                        # paths where it actually rotates; compare to detect that.
                        _before = list(self.robot_state)
                        if self.align_heading_step():
                            self._align_pending = False
                            self._align_steps = 0
                            print(f"[align] heading aligned at "
                                  f"{np.round(self.robot_position, 2)}; MPC tracking resumed")
                        applied = list(self.robot_state) != _before
                    else:
                        # Nonlinear MPC
                        target = [self.s_bot.parent.x, self.s_bot.parent.y]
                        mpc_state = self.mpc_step() if self.mpc_enable else None

                        if mpc_state is not None:
                            # MPC solved: apply its first step. Same dynamics as the
                            # pure-pursuit integrator (position with the OLD heading, then
                            # heading), so the two are interchangeable step for step.
                            self.robot_state = mpc_state
                            applied = True
                        else:

                            # MPC could not solve -> call contingency plans
                            if self.HJ_contingency_enable:
                                self.contingency_triggered = True

                            else: #If there is no reachable set, then use Dubins pure pursuit control
                                next_state, tag = self.safe_pure_pursuit_step(target)
                                if next_state is None:
                                    # Even standing still is inadmissible -> the current state
                                    # is already outside the feasible set (the reachable set
                                    # shrank under the robot). Hold and escalate.
                                    print("[exec-filter] BLOCKED: current state inadmissible "
                                        f"at {np.round(self.robot_position, 2)}"
                                        + (" -> contingency" if self.HJ_contingency_enable else ""))
                                    if self.HJ_contingency_enable:
                                        self.contingency_triggered = True
                                else:
                                    if tag != "nominal":
                                        print(f"[exec-filter] {tag}: backing off at "
                                            f"{np.round(self.robot_position, 2)}")
                                    self.robot_state = next_state
                                    applied = True

                                # A reactive filter can rotate in place forever if the
                                # pursued waypoint is itself unreachable. Escalate instead
                                # of stalling silently.
                                if self.filter_stalled():
                                    print(f"[exec-filter] STALLED for {self._filter_stall} steps "
                                        "(no progress) -> contingency")
                                    self._filter_stall = 0
                                    if self.HJ_contingency_enable:
                                        self.contingency_triggered = True




                            # Unfiltered fallback, kept for reference. Enforces NOTHING --
                            # no obstacle check, no reachable-set check.
                            # self.robot_state = self.utils.update_robot_position_dubins(
                            #     self.robot_state,
                            #     target,
                            #     self.motion_dt,
                            #     v=self.robot_speed,
                            #     w_max=self.robot_w_max,
                            #     stop_at=(self.s_goal.x, self.s_goal.y),
                            # )
                    self.robot_position = self.robot_state[:2]

                    # NOTE: for evaluating per-step solve time
                    # A control was applied -> one control step. The branches that
                    # only trigger the contingency move nothing, so they emit none.
                    # Marked before the lidar sweep so any update_obs it triggers is
                    # charged to this same step.
                    if applied:
                        self.steps_taken += 1
                        if self.recorder is not None:
                            self.recorder.mark(*self.robot_state, mode="nominal")

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

                        # ---- orphaned with no valid parent -> run the contingency --------
                        # The update may have destroyed the robot's route ("robot node got
                        # orphaned"). Two in-place repairs have already been attempted by
                        # this point: update_LMC inside reduce_inconsistency (a new parent
                        # among s_bot's out-neighbours), and the explicit reconnect below.
                        # If BOTH fail the robot is standing in space the tree can no longer
                        # reach, and the honest move is to spend the reachable set we have
                        # certified: drive into a safe region and wait there while the tree
                        # regrows toward it, rather than hold position in open space.
                        #
                        # A validated route means BOTH a finite cost AND a chain that
                        # actually terminates at the root -- a stale finite lmc alone is
                        # exactly the zombie branch that used to send the robot backwards.
                        if self.HJ_contingency_enable and not self._has_validated_route():
                            # order matters: clear a stale cost first (otherwise update_LMC
                            # refuses to look for a new parent), then retry the reconnect.
                            if self.prune_unrooted_enable and self.repair_broken_chain():
                                self.reduce_inconsistency(robots_plan=True)
                            if self._pending_reset_target is not None:
                                self._try_connect_pending()
                            if not self._has_validated_route():
                                print("[contingency] robot orphaned and could not be rewired "
                                      "to any valid parent -> retreating into a safe set")
                                self.contingency_triggered = True
                        # -----------------------------------------------------------------

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

            
            if self._has_validated_route():
                self.robot_path_to_goal = True
                
            for j, goal_j in enumerate(self.other_goals):
                if self.has_path_to_goal(goal_j):
                    self.path_to_goal[j] = True
                    
            # Allow matplotlib to process events (including mouse clicks)
            # This is crucial for the click handler to work.
            # Skipped when headless: there is no window to click, and this call
            # sits INSIDE the block the per-step solve time is measured over.
            if step_idx % 10 == 0 and self.interactive:
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
            # IDENTITY, not `==`: the question is "was the edge that just died the exact
            # edge v uses to reach the goal", and only `is` answers it. Node.__eq__ also
            # matches on equal cost keys and on positions within 1e-6, so `v.parent == u`
            # orphaned v whenever u merely RESEMBLED its parent -- routinely, because
            # neighbour sets are never cleaned on eviction, so u is often a dead (inf, inf)
            # node whose key matches any other inf-cost parent. Those are all false
            # positives (a real parent edge still matches via id()), and each one costs v's
            # whole subtree. `is` also removes the need for the `v.parent and` guard, which
            # was a None-check relying on Node.__len__ always returning 2.
            if v.parent is u:
                self.verify_orphan(v)
                # should theoretically check if the robot is on this edge now, but we do not
                # v.parent.children.remove(v) # these two lines are from the Julia code
                # v.parent = None 
                
        heapq.heapify(self.Q) # reheapify after removing a bunch of elements and ruining queue

    def verify_orphan(self, v):
        # Algorithm 10
        # if v is in Q, remove it from Q and add it to orphan_nodes
        self._pop_from_queue(v)   # identity-matched; see _pop_from_queue
        self.orphan_nodes.add(v)

    def propagate_descendants(self, robots_plan=False):
        
        # ------------------
        # NOTE Orphan nodes are all nodes that is disconnected from the goal nodes due to newly-observed obstacles
        # ------------------
        
        # Algorithm 9
        if not self.orphan_nodes:
            return
        # Recursively add the descendants of every orphan, via BFS.
        #
        # The child index is rebuilt here from the `parent` pointers rather than read off
        # each node's own `children` record. Parent pointers are what the route actually
        # follows (update_path walks them), so they are authoritative; deriving the index
        # from them means a desynced `children` record cannot hide a descendant and leave
        # it advertising a finite cost with no route to the goal. One O(len(tree_nodes))
        # pass, negligible beside the FNO feasibility queries.
        kids = {}
        for nd in self.tree_nodes:
            if nd.parent is not None:
                kids.setdefault(nd.parent.id, []).append(nd)

        orphan_queue = deque(list(self.orphan_nodes))
        visited = {nd.id for nd in self.orphan_nodes}
        while orphan_queue:
            node = orphan_queue.pop()
            for child in kids.get(node.id, ()):
                # `visited` also stops a node being re-queued via several routes, which the
                # previous version did unboundedly.
                if child.id not in visited:
                    visited.add(child.id)
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
                v.parent.children.pop(v.id, None)   # identity-keyed; see Node.set_parent
                v.parent = None

        # NOTE deliberately NOT removed from V or from the kd-tree.
        #
        # RRTx Algorithm 9 (propogateDescendants) reads:
        #     7 forall v in V^c_T do
        #     8    V^c_T <- V^c_T \ {v}        <-- removed from the ORPHAN SET only
        #     9    g(v) <- inf
        #    10    lmc(v) <- inf
        #    11    if p+(v) != 0 then
        #    12       C-(p+(v)) <- C-(p+(v)) \ {v}
        #    13       p+(v) <- 0
        #
        # There is no line removing v from V or from the nearest-neighbour structure. An
        # orphan stays a full graph citizen with infinite cost: still in every neighbour
        # set, still returned by near(). That is exactly how it gets back in --
        # rewireNeighbors (Alg 4) adopts it, because lmc(u) = inf satisfies
        # `lmc(u) > d(u,v) + lmc(v)` for any neighbour v with a finite cost. Evicting them
        # here (a deviation this file used to make) broke that cascade two ways: the
        # descendant enumeration below could no longer see them, and nothing could rewire
        # them, so the planner escalated to the contingency policy instead of doing the
        # local repair RRTx is built around.
        #
        # Infinite cost is self-filtering everywhere it matters -- findParent (Alg 6) and
        # updateLMC (Alg 14) both compare `lmc(v) > d + lmc(u)`, which is false for
        # lmc(u) = inf -- so no explicit "is this node still in the tree" test is needed.

        # check if robot node got orphaned
        if robots_plan:
            if any(o is self.s_bot for o in self.orphan_nodes) or not self.has_path_to_goal(self.s_bot):
                print('robot node got orphaned')
                self.robot_path_to_goal = False
                self.path = []

                # Arm the reconnect machinery. Orphaning strips s_bot of its parent AND
                # evicts it from tree_nodes/kd_tree, so nothing can re-attach it: the
                # sampling bias in random_node and _try_connect_pending both key off
                # _pending_reset_target, which used to be left as None here. The robot
                # then stayed isolated until some unrelated caller happened to invoke
                # reset_robot_position -- which is what made the planner sit idle in a
                # safe set after a contingency.
                self._pending_reset_target = Node((self.robot_position[0],
                                                  self.robot_position[1]))
                self._pending_reset_heading = self.robot_state[2]
        
        # else:
        #Check if path between goal got orphaned
        for j, goal_j in enumerate(self.other_goals):
            if any(o is goal_j for o in self.orphan_nodes) or not self.has_path_to_goal(goal_j):
                self.path_to_goal[j] = False
                self.multi_paths[j] = []
                self.multi_path_nodes[j] = []

        self.orphan_nodes = set([]) # reset orphan_nodes to empty set

    def verify_queue(self, v):
        # Algorithm 13
        # this does not do the updating, it is done after all changes are made (in propagate_descendants)
        # if v is in Q, update its cost and position, otherwise just add it
        # (identity-matched removal so a key-twin is not evicted: see _pop_from_queue)
        self._pop_from_queue(v)
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
    
        
        
        # Nodes whose consistency the loop is trying to establish. With a single goal
        # other_goals is EMPTY, so with robots_plan=False this list would be empty too --
        # any([]) is False and the loop would never run, leaving every cost unreconciled
        # during the initial tree growth. Fall back to draining the queue in that case.
        targets = self.other_goals + ([self.s_bot] if (robots_plan and self.s_bot is not None) else [])

        while len(self.Q) > 0 and (
                            not targets
                            or any(
                                self.Q[0][0] < v.get_key()
                                or v.lmc != v.cost_to_goal
                                or np.isinf(v.cost_to_goal)
                                or v in {node for _, node in self.Q}

                                for v in targets
                            )
                            ):
                
        

            try:
                v = heapq.heappop(self.Q)[1]
            except TypeError:
                print('something went wrong with the queue')
        
            # NOTE `v.cost_to_goal - v.lmc` is nan when BOTH are inf -- precisely the state
            # propagate_descendants leaves every orphan in. Under a bare `> epsilon` test
            # that comparison is False, so update_LMC never runs, the orphan never searches
            # its neighbours for a new parent, and the assignment below re-stamps it inf
            # forever. update_obstacles queues the robot node and calls this method
            # expecting exactly that recovery, so an orphaned robot could never self-heal.
            # An infinite lmc is inconsistent by definition -- treat it as such.
            if np.isinf(v.lmc) or v.cost_to_goal - v.lmc > self.epsilon:
                v.update_LMC(self.orphan_nodes, self.search_radius, self.epsilon, self.utils)
                self.rewire_neighbours(v, robots_plan=robots_plan) #find better paths through v

            v.cost_to_goal = v.lmc

            # If v just regained a finite cost it may be an orphan that was evicted from
            # the graph; without this it stays invisible to near()/nearest() forever.

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
            
            # Compare POSITIONS, not `node_new == self.s_bot`: Node.__eq__ also matches on
            # equal cost keys, and get_key() is (min(cost_to_goal, lmc), cost_to_goal), so an
            # orphaned s_bot keyed (inf, inf) compares equal to EVERY new node whose lmc is
            # also inf -- silently re-pointing s_bot at an unrelated node metres away with no
            # parent and infinite cost, which strands the robot.
            if self.s_bot is not None and \
                    math.hypot(node_new.x - self.s_bot.x, node_new.y - self.s_bot.y) < 1e-6:
                self.s_bot = node_new
                # The robot is now IN the tree, which is exactly what the pending reconnect
                # was waiting for -- so disarm it. Leaving it armed kept random_node's 50%
                # pull toward the robot switched on for the whole of growth, and because
                # saturate() returns a sample unchanged once a node is within step_len, every
                # one of those samples became another node at the SAME coordinates. Measured
                # on a 1000-sample init_tree: 479/975 samples drawn at the pending target,
                # 363 coincident nodes on one point, 46% of the tree within 1.5 m of the
                # start, |V_near| up to 363 -- one is_feasible_ray call per pile member per
                # sample, which is what makes init_node_counts scale quadratically.
                self._pending_reset_target = None
                self._pending_reset_heading = None
                # Re-attaching the robot node does NOT by itself prove a route exists --
                # find_parent only set lmc, and the parent chain may still be rooted in an
                # inf-cost subtree. Ask the predicate instead of asserting True blindly.
                self.robot_path_to_goal = self.has_path_to_goal(self.s_bot)
                self.update_path(self.s_bot) # update path to goal for plotting
                
                # self.other_goals[self.curr_tree_idx] = self.s_bot
                # self.update_multi_paths(self.s_bot, self.curr_tree_idx)
                # self.path_to_goal[self.curr_tree_idx] = True
                return
        # else:
            
        for j in range(len(self.other_goals)):
            # Position test for the same reason as above: an unreached goal is also keyed
            # (inf, inf), so `==` let any inf-lmc node replace the goal object outright.
            goal_j = self.other_goals[j]
            if math.hypot(node_new.x - goal_j.x, node_new.y - goal_j.y) < 1e-6:
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
        #
        # Only candidates that are IN the graph with a FINITE lmc are eligible. The old
        # version took np.argmin over the costs, so when every candidate was inf it still
        # attached v to candidate 0 -- handing v a dead parent and lmc = inf. Those
        # attachments are how branches end up terminating at an evicted, inf-cost orphan
        # instead of at the root: update_path then materialises the branch up to the break
        # and the tracker is told to drive along it (measured: 104 / 1644 path builds
        # truncated, the robot steered at a dead branch ending near the START).
        # u.in_tree additionally rejects nodes propagate_descendants has already evicted,
        # which remain reachable through stale neighbour sets.
        #
        # Sorting and taking the first collision-free candidate also replaces the old
        # `del U[min_idx]; find_parent(v, U)` recursion, which mutated the CALLER's list --
        # extend() then skipped wiring neighbour links for whatever the recursion removed.
        # Callers re-check is_collision when they wire neighbours, so this is safe.
        # RRTx Algorithm 6 (findParent) selects on
        #     d(v,u) <= r  and  lmc(v) > d(v,u) + lmc(u)  and  the edge is collision-free
        # An orphan carries lmc(u) = inf, so `lmc(v) > d + inf` is false and it is rejected
        # IMPLICITLY -- no membership test is needed, and adding one would block the very
        # nodes that rewireNeighbors is meant to pull back into the tree.
        cands = sorted(
            ((u, math.hypot(v.x - u.x, v.y - u.y) + u.lmc) for u in U),
            key=lambda z: z[1],
        )
        for u, cost in cands:
            if not math.isfinite(cost):
                break                      # sorted ascending: nothing finite remains
            if not self.utils.is_collision(u, v):
                v.set_parent(u)
                v.lmc = cost   # u.lmc is already included in `cost`
                return
        # No usable parent: leave v.parent = None. Callers treat that as failure (extend
        # returns without adding; reset_robot_position goes pending via _go_pending).
        

    def rewire_neighbours(self, v, robots_plan=False):
        #NOTE remove is_feasible_ray in rewire_neighbor
        
        # Algorithm 4
        # inf/inf -> nan under a bare `> epsilon` test; see reduce_inconsistency.
        if np.isinf(v.lmc) or v.cost_to_goal - v.lmc > self.epsilon:
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
        
        if (robots_plan) and (not self.robot_path_to_goal) and (self.s_bot is not None) \
                and (np.random.random() < self.bot_sample_rate):
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
        # Always use current robot position. The node is placed AT that position and is
        # deliberately NOT saturated: a saturated node sits step_len from `nearest`
        # rather than at the robot, and the assignments at the end of this method would
        # then rewrite robot_state with it -- a silent teleport (same defect as
        # reset_robot_position). Neighbours are likewise capped at step_len instead of
        # falling back to `nearest` at any distance; if nothing is close enough we stay
        # pending and keep growing the tree toward the robot.
        target = Node((self.robot_position[0], self.robot_position[1]))

        # near() reads self.search_radius; refresh it for the current tree size rather than
        # inheriting whatever the last sampling loop left (0.0 before any loop has run).
        self.search_radius = self.shrinking_ball_radius()
        V_near = [u for u in self.near(target)
                  if math.hypot(u.x - target.x, u.y - target.y) <= self.step_len]

        V_near_free = [u for u in V_near if not self.utils.is_collision(u, target)]
        if not V_near_free:
            return  # still blocked, keep biasing

        self.find_parent(target, V_near_free)
        if target.parent is None:
            return

        self.add_node(target)

        for u in V_near_free:
            target.N_o_plus.add(u)
            target.N_o_minus.add(u)
            u.N_r_plus.add(target)
            u.N_r_minus.add(target)

        # Assign s_bot BEFORE the queue work: reduce_inconsistency's loop condition and
        # refresh_paths only consider the robot node when robots_plan=True, and
        # find_parent/add_node set lmc while leaving cost_to_goal at its inf default
        # (rewire_neighbours queues in-neighbours, never `target` itself). Without the
        # verify_queue below, a robot reconnected through THIS path keeps
        # cost_to_goal = inf -- and since the reconnect machinery routes most resets
        # here, that is the dominant source of "isolated" reports for a robot that is
        # demonstrably attached to the tree.
        self.s_bot = target
        self._pending_reset_target = None

        self.rewire_neighbours(target, robots_plan=True)
        self.verify_queue(self.s_bot)
        self.reduce_inconsistency(robots_plan=True)

        # s_bot is AT the robot, so this only fills in the heading.
        heading = self._pending_reset_heading
        self.robot_state = [self.s_bot.x, self.s_bot.y,
                            heading if heading is not None else self.robot_state[2]]
        self.robot_position = [self.s_bot.x, self.s_bot.y]
        self.robot_path_to_goal = self.has_path_to_goal(self.s_bot)
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
        """Materialise the parent chain from `node` to the root as self.path / path_node.

        A chain that does NOT end at the root is discarded. It can happen: a branch whose
        ancestor was orphaned keeps its own stale finite lmc, so the walk runs into an
        evicted, inf-cost, parentless node and stops early. The result looks like a
        perfectly good path -- and used to be handed straight to the tracker, which then
        drove the robot along a dead branch (measured: 104 / 1644 builds truncated, the
        committed path ending near the START instead of at the goal). Publishing an empty
        path instead makes the robot hold and the reconnect machinery take over, which is
        the truthful answer: no route is known from here.

        `seen` is keyed by id, not by node: Node.__eq__ matches positions within 1e-6 and
        also matches equal cost keys, so a value-based set could abort the walk at a
        legitimate duplicate-position node.
        """
        path, path_node = [], []
        seen = set()  # ids; guards against transient parent cycles (mid-rewire)
        while node.parent is not None and id(node) not in seen:
            seen.add(id(node))
            path_node.append(node)
            path.append(np.array([[node.x, node.y], [node.parent.x, node.parent.y]]))
            node = node.parent

        # `node` is now the chain's terminus: the root, if the chain is whole.
        # _path_reaches_root gates motion (see planning / planning_with_robot): without it
        # has_path_to_goal(s_bot) still reports True off the stale lmc and the robot would
        # be cleared to drive at s_bot.parent, i.e. one hop up the dead branch.
        self._path_reaches_root = (node is self.s_goal)
        if not self._path_reaches_root:
            # Record WHERE the chain breaks so it can be repaired at a safe point.
            # path_node[-1] is the child of the dead terminus, i.e. the node still
            # advertising a finite lmc that was computed through a parent which has since
            # become an inf-cost orphan. Sampling alone never clears that stale value, so
            # a recovery loop gated on _path_reaches_root would spin forever (observed:
            # "[Recovery iter 10] cost = 60.666" while the chain still did not reach the
            # root). repair_broken_chain() does the actual orphaning -- NOT here, because
            # update_path is called from inside reduce_inconsistency and re-entering
            # propagate_descendants mid-rewire would corrupt the sweep.
            self._broken_chain_node = path_node[-1] if path_node else None
            self.path = []
            self.path_node = []
            return

        self._broken_chain_node = None

        self.path = path
        self.path_node = path_node

    # ------------------------------------------------------------------
    # Nonlinear-MPC tracking of the (geometric) tree path
    # ------------------------------------------------------------------
    def mpc_reference(self, n_steps, ds):
        """Build the (n_steps+1, 2) position reference the MPC tracks.

        Waypoints are the robot's CURRENT position, then the tree nodes from s_bot
        toward the goal, then the goal itself -- the same chain the old Dubins
        reference used, but left as a raw polyline: the MPC does not need a
        dynamically feasible reference, it needs a direction of travel.

        Sampled at `ds` = v_max * dt arc-length spacing so the reference is
        reachable at full speed. `mpc_ref_span` caps how far along the path we look;
        past the end the final point repeats, which is what lets the robot settle
        onto the goal rather than overshoot it.

        @return (n_steps+1, 2), or None when there is no usable path.
        """
        if not self.path_node or self.s_bot is None:
            return None

        pts = [[self.robot_state[0], self.robot_state[1]]]
        for n in self.path_node[1:]:
            pts.append([n.x, n.y])
        pts.append([self.s_goal.x, self.s_goal.y])
        pts = np.asarray(pts, dtype=float)

        # drop duplicate consecutive waypoints (the robot sits on top of s_bot right
        # after a switch); a zero-length segment has no direction
        keep = [0]
        for i in range(1, len(pts)):
            if np.hypot(*(pts[i] - pts[keep[-1]])) > 1e-6:
                keep.append(i)
        pts = pts[keep]
        if len(pts) < 2:
            return None

        # truncate to the reference span (cost cap; the horizon never reaches past it)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        last = int(np.searchsorted(cum, self.mpc_ref_span)) + 1
        pts = pts[:max(2, min(last, len(pts)))]

        return nmpc_dubins.reference_from_polyline(pts, n_steps, ds)

    def mpc_slack_fn(self):
        """Closure giving the reachable-set invariance slack at arbitrary positions.

        slack(p) = safe_margin[region(p)] - V(p)

        No buffer here: the solver is asked for slack >= mpc_slack_margin
        (== feasibility_buffer), so the buffer is counted once. Out-of-domain V = +inf
        is floored to a large finite violation.

        Returns None when the contingency constraint is disabled, which makes the MPC
        drop the constraint entirely (obstacles are already excluded from
        feasible_region, per the design note in nmpc_dubins).
        """
        if not self.HJ_contingency_enable:
            return None

        hjr = self.hjr_fno

        def slack(points):
            pts = np.atleast_2d(np.asarray(points, dtype=float))
            vals = np.asarray(hjr.feasibility_values(pts), dtype=float)
            region = np.asarray(
                hjr.find_feasible_closest_region(robot_pose=pts)).reshape(-1)
            thr = np.array([hjr.safe_margin[r] for r in region], dtype=float)
            out = thr - vals
            return np.where(np.isfinite(out), out, -1e3)

        return slack

    def mpc_step(self):
        """Solve one MPC step and return the next state, or None to fall back.

        @return [x, y, theta] to apply, or None when the reference is unusable or the
                solve failed / produced an inadmissible first move.
        """
        ds = max(1e-6, self.robot_speed * self.motion_dt)
        ref = self.mpc_reference(self.mpc_horizon, ds)
        self.mpc_ref = ref
        if ref is None:
            return None

        u0, pred, ok = self.mpc.solve(self.robot_state, ref, self.mpc_slack_fn())
        self.mpc_pred = pred
        if not ok or pred is None:
            self.mpc_fallbacks += 1
            return None

        nxt = pred[1]
        return [float(nxt[0]), float(nxt[1]), float(nxt[2])]

    def align_heading_step(self):
        """One rotate-in-place control step toward s_bot.parent (v = 0).

        Same motion_dt / robot_w_max as every other control step, so this costs exactly
        one control step of simulated time and adds nothing to the travelled distance.

        @return True when the alignment phase is over (heading reached, no target to
                align to, step budget spent, or the rotation is inadmissible),
                False while it should continue.
        """
        parent = self.s_bot.parent if self.s_bot is not None else None
        if parent is None:
            return False        # no route yet -- stay armed, do not rotate

        dx, dy = parent.x - self.robot_state[0], parent.y - self.robot_state[1]
        if math.hypot(dx, dy) < 1e-6:
            return True         # coincident parent: heading is undefined

        err = nmpc_dubins.wrap_pi(math.atan2(dy, dx) - self.robot_state[2])
        if abs(err) <= self.robot_w_max * self.motion_dt:
            return True

        omega = float(np.clip(err / self.motion_dt, -self.robot_w_max, self.robot_w_max))
        cand = [self.robot_state[0], self.robot_state[1],
                self.robot_state[2] + omega * self.motion_dt]

        if not self._poses_admissible([cand]):
            print(f"[align] rotation inadmissible at {np.round(self.robot_position, 2)}; "
                  "resuming MPC")
            return True

        self.robot_state = cand
        self._align_steps += 1
        if self._align_steps >= math.ceil(math.pi / (self.robot_w_max * self.motion_dt)) + 5:
            print(f"[align] step budget spent with {abs(err):.3f} rad remaining; "
                  "resuming MPC")
            return True
        return False

    def update_multi_paths(self, node, idx):
        self.multi_paths[idx] = []
        self.multi_path_nodes[idx] = []

        seen = set()  # guard against transient parent cycles (mid-rewire)
        while node.parent and node not in seen:
            seen.add(node)
            self.multi_path_nodes[idx].append(node)
            self.multi_paths[idx].append(np.array([[node.x, node.y], [node.parent.x, node.parent.y]]))
            node = node.parent
    
    # Thin aliases so `self.has_path_to_goal(...)` reads naturally inside the tree code;
    # the definitions live at module level (see path_cost / has_path_to_goal).
    path_cost = staticmethod(path_cost)
    has_path_to_goal = staticmethod(has_path_to_goal)

    def node_in_queue(self, node):
        """Key of `node`'s own entry in Q, or None. Identity-matched -- see
        _pop_from_queue for why value equality must not be used here."""
        for key, n in self.Q:
            if n is node:
                return key
        return None

    def _pop_from_queue(self, node):
        """Remove `node`'s OWN entry from Q (matched by identity) and return its key.

        Node.__eq__ reports equality when two nodes merely share a KEY
        (`self.get_key() == other.get_key()`), and get_key() is
        (min(cost_to_goal, lmc), cost_to_goal). After propagate_descendants every
        orphan shares the key (inf, inf), so the previous `nodes.index(node)` +
        `self.Q.remove((key, node))` pair matched whichever unrelated node sat first in
        Q and silently evicted THAT one instead. The evicted node's pending cost update
        was then lost forever, leaving it at cost_to_goal = inf.

        list.remove also breaks the heap invariant, so re-heapify before returning.
        """
        for i, (key, n) in enumerate(self.Q):
            if n is node:
                self.Q.pop(i)
                heapq.heapify(self.Q)
                return key
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

    # ------------------------------------------------------------------
    # execution-time safety filter
    # ------------------------------------------------------------------
    def _poses_admissible(self, poses) -> bool:
        """Are ALL of these poses admissible? Obstacle-clear against the KNOWN
        obstacles inflated by utils.delta (same rule as Utils.is_inside_obs) AND,
        when the contingency constraint is on, inside the HJ reachable set.

        One batched points_feasible call for the whole set -- that call has a fixed
        overhead that dominates its per-point cost, so checking 15 poses costs
        essentially the same as checking 1."""
        poses = np.atleast_2d(np.asarray(poses, dtype=float))
        pts = poses[:, :2]

        if self.obs_circle:
            obs = np.asarray(self.obs_circle, dtype=float)          # (M, 3)
            d2 = ((pts[:, None, 0] - obs[None, :, 0]) ** 2
                  + (pts[:, None, 1] - obs[None, :, 1]) ** 2)
            if np.any(d2 <= (obs[None, :, 2] + self.utils.delta) ** 2):
                return False

        if not self.HJ_contingency_enable:
            return True

        # B1: theta is only consumed by the theta-dependent "HJR_sets" source
        thetas = poses[:, 2] if self.hjr_fno.feasibility_source == "HJR_sets" else None
        return bool(np.all(self.hjr_fno.points_feasible(
            pts, thetas=thetas, reachable_set_constraint=True)))

    def _pursuit_rollout(self, state, target, n_steps, v):
        """Predict `n_steps` of pure pursuit toward a FIXED `target` at speed `v`.

        The target is held fixed over the rollout; the real tracker re-targets every
        step (the waypoint advances), so this is an approximation -- but a
        conservative one for the purpose it serves: holding the target makes the
        predicted turn tighter and the predicted path hug the current heading error,
        so a rollout that passes is not optimistic about the near term. Only the
        FIRST pose is ever executed; the rest is early warning.
        """
        poses = np.empty((n_steps, 3), dtype=float)
        s = list(state)
        for k in range(n_steps):
            s = self.utils.update_robot_position_dubins(
                s, target, self.motion_dt, v=v, w_max=self.robot_w_max,
                stop_at=(self.s_goal.x, self.s_goal.y),
            )
            poses[k] = s
        return poses

    def safe_pure_pursuit_step(self, target):
        """One control step of pure pursuit toward `target`, filtered for safety.

        Tries progressively gentler commands and executes the FIRST whose predicted
        `filter_horizon`-step rollout is entirely admissible:

            v = robot_speed  ->  0.5*  ->  0.25*  ->  0 (rotate in place)

        Slowing down helps for two reasons: the step is shorter (less distance into
        whatever lies ahead) and, since rho = v/w_max shrinks, the turn is tighter,
        so the robot can round a corner it would otherwise overshoot. v = 0 keeps
        turning while standing still, which is admissible for a unicycle (env
        u_min[0] = 0) and is how the robot recovers from a large heading error.

        @return (next_state, tag) with tag in {"nominal", "slowed", "rotate"}, or
                (None, "blocked") when even standing still is inadmissible -- which
                means the CURRENT state is already infeasible (the reachable set
                shrank under the robot after a lidar reveal). The caller should hold
                position and escalate to the contingency behaviour.
        """
        for scale in self.filter_speed_scales:
            poses = self._pursuit_rollout(
                self.robot_state, target, self.filter_horizon, self.robot_speed * scale)
            if self._poses_admissible(poses):
                tag = "nominal" if scale == 1.0 else ("rotate" if scale == 0.0 else "slowed")
                self.filter_counts[tag] += 1
                # zero-progress steps (v = 0) accumulate toward a stall escalation
                self._filter_stall = 0 if scale > 0.0 else self._filter_stall + 1
                return list(poses[0]), tag

        self.filter_counts["blocked"] += 1
        self._filter_stall += 1
        return None, "blocked"

    def filter_stalled(self) -> bool:
        """True once the filter has produced `filter_stall_limit` consecutive
        zero-progress steps -- the robot is rotating in place or held because the
        pursued waypoint is itself unreachable, and the tree needs to re-route."""
        return self._filter_stall >= self.filter_stall_limit

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
Single-goal RRTX driver.

This module is the one-goal counterpart of rrtx_FNO3d.py: exactly ONE RRTX tree, rooted
at the destination, with the robot starting at x_goal[start_goal_index]. The SFF_star
orchestrator and its Held-Karp ATSP tour are gone -- there is no tour to optimise, no
goal-to-goal distance matrix, and no "other trees" to keep in sync. RRTX itself is
unchanged apart from two single-goal fixes (see reduce_inconsistency / planning).

Everything that used to be SFF_star state is now local to main(); the helpers below take
what they need explicitly.
'''


class GifRecorder:
    """Streams figure frames straight to a gif (save_mode=True), else does nothing.

    Frames are written as they arrive rather than accumulated: the live figure is
    10x10 in at 100 dpi, i.e. ~3 MB per RGB frame, so a 500-step run would otherwise
    need >1 GB of RAM.
    """

    def __init__(self, save_mode=False, save_path=None, save_every=1, save_fps=10):
        self.save_mode = save_mode
        self.save_path = save_path or "video/rrtx_FNO3d_oneGoal.gif"
        self.save_every = max(1, int(save_every))
        self.save_fps = int(save_fps)
        self._writer = None
        self._frame_count = 0
        self._captured = 0
        self._frame_shape = None
        if self.save_mode:
            # headless rendering: no window, and plt.pause()/plt.show() become no-ops
            plt.switch_backend("Agg")

    def capture(self, fig, force: bool = False) -> None:
        """Append `fig` to the gif. Every save_every-th call is written; force=True
        always writes (used for the final summary frame)."""
        if not self.save_mode:
            return
        self._frame_count += 1
        if not force and (self._frame_count % self.save_every) != 0:
            return

        import imageio.v2 as imageio

        if self._writer is None:
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            self._writer = imageio.get_writer(
                self.save_path, mode="I", fps=self.save_fps, loop=0
            )
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frame = frame[..., :3].copy()                     # drop alpha

        # every gif frame must have the same size, but the final summary plot is a
        # fresh 8x8 figure while the live one is 10x10 -- resize to the first frame
        if self._frame_shape is None:
            self._frame_shape = frame.shape[:2]
        elif frame.shape[:2] != self._frame_shape:
            from PIL import Image

            frame = np.asarray(
                Image.fromarray(frame).resize(
                    (self._frame_shape[1], self._frame_shape[0]), Image.BILINEAR
                )
            )

        self._writer.append_data(frame)
        self._captured += 1

    def close(self) -> None:
        """Finalize the gif. Safe to call repeatedly, and when save_mode=False."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            print(f"[save_mode] wrote {self._captured} frames to {self.save_path}")


def detect_initial_obstacles(tree, hjr_fno, start_xy, HJ_contingency_enable=True):
    """Sense the lidar footprint of the robot's INITIAL pose and register what it sees,
    before the tree is grown. Returns the newly detected obstacles.

    Doing this up front rather than on the first execution step means the tree is built
    collision-free and reachability-feasible from the outset: planning() rejects edges via
    utils.is_collision (which reads the shared obstacle store) and via is_feasible_ray
    (which reads hjr_fno's value tubes), so both must know about these obstacles BEFORE the
    first sample. Otherwise the whole growth budget goes into edges through obstacles that
    are already in sensor range, and the first execution step tears all of it down --
    orphaning the robot node and triggering a contingency before the robot has moved.
    """
    x0, y0 = start_xy

    # The tree's Utils and hjr_fno.utils wrap the SAME Env, so the unknown-obstacle list is
    # one shared object and lidar_detected() moves entries out of it in place.
    _, detected_obs = hjr_fno.utils.lidar_detected(robot_position=(x0, y0))

    if not detected_obs:
        print(f"[init] no unknown obstacles within lidar range "
              f"({hjr_fno.utils.sensing_radius}) of the start ({x0:.2f}, {y0:.2f})")
        return []

    print(f"\n[init] {len(detected_obs)} obstacle(s) detected within lidar range of the "
          f"start ({x0:.2f}, {y0:.2f}): {detected_obs}")

    # 1) HJ reachable sets first, so the value tubes and re-certified delta_hat that
    #    is_feasible_ray consults during growth already account for these obstacles.
    if HJ_contingency_enable:
        t0 = time.time()
        hjr_fno.update_obs(detected_obs)
        print(f"[init] reachable sets updated in {time.time() - t0:.2f} s")

    # 2) Register in the shared obstacle store for collision checking + plotting.
    #    NOTE update_obstacles() is deliberately NOT used: it calls verify_queue(s_bot),
    #    and s_bot may still be None here. There is no graph to repair yet either.
    tree.add_new_obstacle(detected_obs, record=True)

    return detected_obs


def init_tree(tree, _plotting, hjr_fno, fig, ax, _env, start_xy,
              init_node_counts=1000, HJ_contingency_enable=True,
              recorder=None, showPlot=True):
    """Grow the initial tree, with the robot already anchored at `start_xy`.

    Order matters here. s_bot is seeded BEFORE growth so the tree can be grown with
    robots_plan=True, which (a) biases sampling toward the robot and (b) keeps
    reduce_inconsistency's queue loop alive -- with a single goal, other_goals is empty, so
    robots_plan=False would leave it with nothing to make consistent.
    """
    detect_initial_obstacles(tree, hjr_fno, start_xy,
                             HJ_contingency_enable=HJ_contingency_enable)

    # Anchor the robot. On a tree holding only the root this normally goes pending (the
    # root is farther than step_len), which is correct: _pending_reset_target is armed and
    # random_node()/_try_connect_pending grow toward the robot from here on.
    tree.reset_robot_position((start_xy[0], start_xy[1]), heading=None)
    if tree.s_bot is None:
        # Pending path taken, so s_bot was never assigned. Seed it at the robot's true
        # position (unwired, inf cost) so every `self.s_bot.` access downstream is safe.
        tree.s_bot = Node((start_xy[0], start_xy[1]))
        tree.robot_position = [start_xy[0], start_xy[1]]

    print(f"Initialize the tree rooted at goal ({tree.s_goal.x:.2f}, {tree.s_goal.y:.2f}) "
          f"with {init_node_counts} samples")
    tree.planning(iter_max=init_node_counts, robots_plan=True)

    if showPlot:
        ax.clear()
        ax.set_xlim(_env.x_range[0], _env.x_range[1] + 1)
        ax.set_ylim(_env.y_range[0], _env.y_range[1] + 1)
        _plotting.plot_env(ax)

        if tree.all_nodes_coor:
            nodes = np.array(tree.all_nodes_coor)
            ax.scatter(nodes[:, 0], nodes[:, 1], s=4, c='gray', alpha=0.5)

        edges = [np.array([[nd.parent.x, nd.parent.y], [nd.x, nd.y]])
                 for nd in tree.tree_nodes if nd.parent]
        if edges:
            ax.add_collection(LineCollection(edges, colors='blue', linewidths=0.5, alpha=0.2))

        if HJ_contingency_enable:
            _plotting.plot_reachable_set(ax, hjr_fno, theta=tree.robot_state[2],
                                         time=tree.Tf_reach)
        plt.pause(0.001)
        if recorder is not None:
            recorder.capture(fig)


def main(scenario: str = None, save_mode: bool = False, save_path: str = None,
         save_every: int = 1, seed: int = None, no_plot: bool = False,
         max_steps: int = None, nodes: int = None, arm: str = None,
         log_dir: str = None, no_log: bool = False, mem_profile: bool = False):
    """
    scenario: optional name of a shared evaluation scenario from eval/scenarios.py
        ("env_A", "env_B", ...) or a path to a scenario .json.
        When given, the safe regions / unknown obstacles / start / goal / lidar all
        come from it, so this run is directly comparable to the MPPI run on the same
        scenario (mppi_src/navigation2d.py --scenario ...). When None, the hard-coded
        configuration below is used.
    save_mode: record the live plot to a gif (headless, Agg backend) instead of
        showing a window. Defaults to video/rrtx_FNO3d_oneGoal_<scenario>.gif; a still of
        the final plot is written alongside it as *_final.png. The gif is finalized even
        if the run errors out or never reaches the goal.
    save_every: keep only every N-th frame (use 2-5 for long runs to shrink the gif).
    seed: seed numpy + random so a run is reproducible. Sampling is otherwise
        unseeded, so two runs of the same scenario grow different trees and a
        failure cannot be re-observed -- pass a seed when debugging.
        It ALSO selects the scenario's obstacle layout (eval/scenarios.py): with
        --seed the obstacles are generated randomly and reproducibly from
        (scenario, seed) -- identical in every arm, so this run is comparable to
        the MPPI runs at the same seed -- and WITHOUT it the scenario's
        hand-authored baseline list is used. Safe regions, start and goal are
        hand-picked either way.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        print(f"[seed] numpy + random seeded with {seed}")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    HJ_contingency_enable = True
    showPlot = not no_plot
    if no_plot and not save_mode:
        # No window at all: Agg makes plt.pause()/plt.show() no-ops, so a headless
        # run cannot block on the final summary figure.
        plt.switch_backend("Agg")

    # Two separate budgets. These used to be one `iter_max` doing both jobs:
    #   init_node_counts -> number of RRT* SAMPLES used to build the initial tree
    #   iter_max         -> max EXECUTION steps; each one runs planning_with_robot(steps=10),
    #                       which takes 10 more samples and moves the robot once
    init_node_counts = nodes or 3000
    # Outer-iteration cap. In nominal operation one outer iteration advances the
    # robot one control step, so a smaller iter_max than --max_steps would end the
    # episode early and report a timeout MPPI never hits -- for a reason that has
    # nothing to do with the planner. --max_steps stays the single authority.
    iter_max = max(1000, max_steps or 0)

    step_len = 1.5      # matches the 'lookahead' in the Dubins pure-pursuit tracker
    gamma_FOS = 20.0
    epsilon = 0.05
    bot_sample_rate = 0.10

    if scenario is not None:
        from eval.scenarios import get_scenario, rrtx_kwargs

        # seed selects the obstacle layout too: random-but-reproducible from
        # (scenario, seed), the same map every other arm gets at that seed;
        # seed=None keeps the hand-authored baseline list.
        sc = get_scenario(scenario, seed)
        kw = rrtx_kwargs(sc)
        print(f"[scenario] {sc.tag}: {len(sc.safe_regions)} safe regions, "
              f"{len(sc.obstacles)} obstacles, domain +/-{sc.domain}"
              + (f", obstacles RANDOM (seed {sc.obs_seed}, digest {sc.obs_digest})"
                 if sc.obs_seed is not None else ", obstacles hand-authored"))

        start_goal_index = kw["start_goal_index"]
        x_goal = kw["x_goal"]
        heading = kw["heading"]
        lidar_range = kw["lidar_range"]
        safe_regions = kw["safe_regions"]
        env_kwargs = kw["env_kwargs"]
        Tf_reach = kw["Tf_reach"]
        # sc.tag, not sc.name: two seeds of the same env are two different maps,
        # so keying the gif on the name alone would silently overwrite.
        save_path = save_path or f"video/rrtx_FNO3d_oneGoal_{sc.tag}.gif"
    else:
        x_goal = [(-12.94, 17.0), (-8.44, -5.77)]     # [start, destination]
        start_goal_index = 0
        heading = 0.0
        # This built-in fallback map is env_C's ancestor, so it takes env_C's
        # lidar radius (8.0) from eval/config.yaml rather than a literal here.
        lidar_range = load_config("env_C").lidar_radius
        safe_regions = [
            [-11.06, 15.48, 2],
            [ -7.31, -5.95, 2],
            [ -5.06, 11.00, 2],
            [ -3.56,  2.75, 2],
            [  2.31,  7.63, 2],
            [  9.19,  1.63, 2],
            [  9.19, 12.13, 2],
            [  9.19, -6.70, 2],
        ]
        env_kwargs = None
        Tf_reach = CFG.Tf_reach

    assert start_goal_index < len(x_goal), "start_goal_index out of range"
    assert len(x_goal) == 2, ("one-goal driver expects x_goal = [start, destination]; "
                              f"got {len(x_goal)} entries")

    # goal_0 is only the robot's starting location; the single tree is rooted at goal_1.
    start_xy = x_goal[start_goal_index]
    goal_idx = 1 - start_goal_index
    goal_xy = x_goal[goal_idx]

    # ------------------------------------------------------------------
    # Environment / plotting / HJR-FNO  (one shared Env, as before)
    # ------------------------------------------------------------------
    from HJR_FNO.HJR_FNO3d import HJR_FNO

    recorder = GifRecorder(save_mode=save_mode, save_path=save_path,
                           save_every=save_every)

    _env = env.Env(safe_regions=safe_regions, **(env_kwargs or {}))
    # Single shared obstacle store: Plotting, the tree and hjr_fno all use this same Env,
    # so obstacles detected during execution are visible everywhere at once.
    # x_goal is passed whole so plotting.plot_env keeps drawing both markers.
    _plotting = plotting.Plotting(start_xy, x_goal, safe_regions=safe_regions, _env=_env)
    hjr_fno = HJR_FNO(env=_env, safe_regions=safe_regions, Tf_reach=Tf_reach)
    # Feasible set = (safe_margin - feasibility_buffer)-sublevel set. Must precede RRTX().
    hjr_fno.feasibility_buffer = 0.2

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle("HJR-FNO Contingency (single goal)")
    ax.set_xlim(_env.x_range[0], _env.x_range[1] + 1)
    ax.set_ylim(_env.y_range[0], _env.y_range[1] + 1)

    current_state = [start_xy[0], start_xy[1], heading]
    prev_heading = heading

    # ------------------------------------------------------------------
    # The one and only tree: rooted at the destination, no other goals
    # ------------------------------------------------------------------
    tree = RRTX(
        x_start=start_xy,
        x_goal=goal_xy,
        goal_id=goal_idx,
        other_goals=[],            # single goal: nothing to route between
        other_goals_id=[],
        heading=heading,
        lidar_range=lidar_range,
        step_len=step_len,
        gamma_FOS=gamma_FOS,
        epsilon=epsilon,
        bot_sample_rate=bot_sample_rate,
        iter_max=init_node_counts,
        safe_regions=safe_regions,
        hjr_fno=hjr_fno,
        HJ_contingency_enable=HJ_contingency_enable,
        fig=fig,
        ax=ax,
        plotting=_plotting,
        environment=_env,
    )
    tree.prob_q = 0.9
    tree.interactive = showPlot   # NOTE: for evaluating per-step solve time

    def on_click(event):
        """Mouse click = adversary detected -> run the contingency policy."""
        if event.inaxes != ax:
            return
        if HJ_contingency_enable:
            print("Adversary detected!")
            tree.contingency_triggered = True
        else:
            print("Contingency constraint disabled. No action taken.")

    # NOTE: for evaluating per-step solve time
    # bound before the try so the finally block can always reach them
    rec = rt = fc = mem = None
    reached_goal = False

    try:
        # --------------------------------------------------------------
        # Build the initial tree (robot anchored at goal_0's location)
        # --------------------------------------------------------------
        init_tree(tree, _plotting, hjr_fno, fig, ax, _env, start_xy,
                  init_node_counts=init_node_counts,
                  HJ_contingency_enable=HJ_contingency_enable,
                  recorder=recorder, showPlot=showPlot)

        tree.update_robot_heading()
        print(f"\nRobot start ({start_xy[0]:.2f}, {start_xy[1]:.2f}) -> "
              f"goal ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}) | "
              f"cost to goal = {path_cost(tree.s_bot):.3f}")

        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        prev_plotting = time.time()

        # --------------------------------------------------------------
        # Execution loop
        # --------------------------------------------------------------
        print("Start Robot's Plan Execution")

        state_history = []

        # NOTE: for evaluating per-step solve time
        # (rec is the metrics logger; `recorder` above is the gif writer)
        if not no_log:
            from eval.episode_log import EpisodeRecorder
            from eval.profiling import FeasCounter, MemoryProbe, ReachTimer

            # baseline AFTER init_tree, so the initial tree counts as planner memory
            # trace_cpu starts tracemalloc, which taxes EVERY allocation -- 3.5x
            # on this workload (Nodes, kd-tree, arrays). It buys only cpu_peak_mb,
            # so it is opt-in; GPU peak and RSS delta cost nothing and stay on.
            mem = MemoryProbe(trace_cpu=mem_profile).baseline()
            rt = ReachTimer(hjr_fno)     # wraps update_obs in place
            fc = FeasCounter(hjr_fno)    # wraps points_feasible in place
            rec = EpisodeRecorder(
                arm or "rrtx_fno", sc.name if scenario else "default",
                seed=seed, knob={"n": init_node_counts}, out_dir=log_dir,
            )
            rec.start(*current_state)
            tree.recorder = rec

        for plan_iter in range(iter_max):

            if max_steps and tree.steps_taken >= max_steps:
                print(f"TIMEOUT: {max_steps} control steps without reaching the goal")
                break

            # ---- advance the robot / grow + rewire the tree ----
            if rec is not None:                     # NOTE: for evaluating per-step solve time
                rt.reset_step()
                fc.reset_step()
            _t0 = time.perf_counter()
            new_obs, new_obs_flag, distance_moved = tree.planning_with_robot(steps=10)
            if rec is not None:                     # NOTE: for evaluating per-step solve time
                ReachTimer.sync()
                # 1 row if the robot moved, else 0 -- the cost carries forward
                rec.charge(t_s=time.perf_counter() - _t0, t_reach_s=rt.step_s,
                           t_predict_s=rt.step_predict_s, t_certify_s=rt.step_certify_s,
                           n_events=rt.step_events, n_feas=fc.step_n)
            current_state = tree.robot_state
            state_history.append(list(current_state) + [goal_idx])

            if not tree.robot_path_to_goal and (plan_iter % 3 == 0):
                print("robot's position", (tree.s_bot.x, tree.s_bot.y))
                print("is feasible?", hjr_fno.is_feasible(v=np.atleast_2d(current_state[:2])))
                print("robot's Path to goal", tree.robot_path_to_goal)
                print("Robot's cost to goal", tree.s_bot.cost_to_goal,
                      "| path_cost =", path_cost(tree.s_bot))
                print("Robot's LMC cost", tree.s_bot.lmc)
                print("Path List", tree.path)
                print("Pending Target", tree._pending_reset_target)

            # ---- contingency ----
            if tree.contingency_triggered and HJ_contingency_enable:

                if rec is not None:                 # NOTE: for evaluating per-step solve time
                    rt.reset_step()
                    fc.reset_step()
                _t0 = time.perf_counter()
                detected_obs_during_contingency, contingency_trajectory, _, _, _, _, _ = \
                    hjr_fno.contingency_policy(current_state, _plotting, fig, ax,
                                               showplot=showPlot)
                tree.steps_taken += max(0, len(contingency_trajectory) - 1)
                if rec is not None:                 # NOTE: for evaluating per-step solve time
                    ReachTimer.sync()
                    # one row per executed dt_c, each with its own measured cost
                    rec.add_rollout(hjr_fno.last_rollout)
                    rec.charge(t_s=time.perf_counter() - _t0, t_reach_s=rt.step_s,
                               t_predict_s=rt.step_predict_s, t_certify_s=rt.step_certify_s,
                               n_events=rt.step_events, n_feas=fc.step_n)

                state_history.extend([list(s) + [goal_idx]
                                      for s in contingency_trajectory.tolist()])

                # keep current_state a plain list: contingency_trajectory[-1] is a numpy row
                current_state = [float(contingency_trajectory[-1][0]),
                                 float(contingency_trajectory[-1][1]),
                                 float(contingency_trajectory[-1][2])]
                print("Position after contingency", current_state)

                if len(detected_obs_during_contingency) > 0:
                    new_obs += detected_obs_during_contingency
                    new_obs_flag = True
                    tree.update_obstacles(detected_obs_during_contingency, robots_plan=True)

                if len(contingency_trajectory) > 1:
                    # Anchor the robot at wherever the rollout left it -- inside a safe
                    # region when the policy succeeded. If it cannot be wired into the tree
                    # there, reset_robot_position goes through _go_pending, which records
                    # this position as _pending_reset_target: the robot then IDLES here
                    # while random_node biases sampling toward it and the tree regrows.
                    connected = tree.reset_robot_position(
                        (current_state[0], current_state[1]), heading=None)
                    tree.update_robot_heading()
                    if not connected:
                        print(f"[contingency] idling in the safe set at "
                              f"({current_state[0]:.2f}, {current_state[1]:.2f}); "
                              f"tree will regrow toward the pending target")
                else:
                    # contingency_policy found no certified region (it returns a 1-row
                    # trajectory then). The robot has not moved, so leave the pending target
                    # armed at its current position and let the tree come to it.
                    print("[contingency] policy returned no trajectory; holding position")
                    if tree._pending_reset_target is None:
                        tree._pending_reset_target = Node((current_state[0], current_state[1]))
                        tree._pending_reset_heading = current_state[2]

                tree.contingency_triggered = False

                # Arm the rotate-in-place alignment. It is consumed inside the
                # robot_path_to_goal branch of planning_with_robot, so it cannot fire
                # until replanning has produced a validated route.
                tree._align_pending = True
                tree._align_steps = 0

                # Re-evaluate reachability AFTER the reset: the rollout may have moved the
                # robot somewhere the tree can reach. Uses the VALIDATED test, so a
                # stale-finite lmc on a broken chain cannot end the recovery early.
                if not tree._has_validated_route():
                    print("\n[INFO] No path to goal after contingency; replanning...")
                    recovery_iter = 0
                    while not tree._has_validated_route() and recovery_iter < 100:
                        # planning_with_robot(), NOT planning(): growing the tree was never
                        # the bottleneck here -- ATTACHING the robot was. planning() has no
                        # way to wire s_bot in (it is evicted, so it is absent from the
                        # kd-tree, never a neighbour of a new node, and never queued, so
                        # update_LMC cannot reach it either); attachment could only happen
                        # by luck, when a sample landed within 1e-6 of the robot and
                        # add_node adopted it. planning_with_robot calls
                        # _try_connect_pending() at every one of its 10 sub-steps, which is
                        # the deliberate reconnect, and it refreshes search_radius.
                        # It also resumes motion by itself once the route is validated.
                        if rec is not None:         # NOTE: for evaluating per-step solve time
                            rt.reset_step()
                            fc.reset_step()
                        _t0 = time.perf_counter()
                        tree.planning_with_robot(steps=10)
                        if rec is not None:         # NOTE: for evaluating per-step solve time
                            ReachTimer.sync()
                            # no motion -> no row; the cost lands on the step that moves
                            rec.charge(t_s=time.perf_counter() - _t0, t_reach_s=rt.step_s,
                                       t_predict_s=rt.step_predict_s, t_certify_s=rt.step_certify_s,
                                       n_events=rt.step_events, n_feas=fc.step_n)
                        recovery_iter += 1

                        # A fresh obstacle seen while recovering can re-trigger the
                        # contingency (see the hook after update_obstacles). Stop here and
                        # let the next main iteration run it rather than fighting it.
                        if tree.contingency_triggered:
                            print("[Recovery] contingency re-triggered; deferring to the "
                                  "next iteration")
                            break

                        if recovery_iter % 10 == 0:
                            print(f"[Recovery iter {recovery_iter}] "
                                  f"cost = {path_cost(tree.s_bot):.3f} "
                                  f"pending={tree._pending_reset_target is not None} "
                                  f"nodes={len(tree.tree_nodes)}")
                    tree.robot_path_to_goal = tree._has_validated_route()
                    print(f"[Recovery complete] path to goal = {tree.robot_path_to_goal} "
                          f"after {recovery_iter} iteration(s)")
                    current_state = tree.robot_state

            # ---- plotting at 5 Hz ----
            if showPlot and (time.time() - prev_plotting) >= 0.2:
                prev_plotting = time.time()

                ax.clear()
                fig.suptitle(f"HJR-FNO Contingency (single goal)\n"
                             f"step {plan_iter} | cost to goal = {path_cost(tree.s_bot):.2f}")
                ax.set_xlim(_env.x_range[0], _env.x_range[1] + 1)
                ax.set_ylim(_env.y_range[0], _env.y_range[1] + 1)

                _plotting.plot_env(ax)

                if tree.all_nodes_coor:
                    nodes = np.array(tree.all_nodes_coor)
                    ax.scatter(nodes[:, 0], nodes[:, 1], s=4, c='gray', alpha=0.5)

                edges = [np.array([[nd.parent.x, nd.parent.y], [nd.x, nd.y]])
                         for nd in tree.tree_nodes if nd.parent]
                if edges:
                    ax.add_collection(LineCollection(edges, colors='blue',
                                                     linewidths=0.3, alpha=0.45))

                if tree.path:
                    ax.add_collection(LineCollection(tree.path, colors='black',
                                                     linewidths=1.5))

                # MPC reference the tracker is following (magenta) and the plan the
                # optimizer actually committed to (cyan). Where the two differ is the
                # optimizer bending the trajectory to stay inside the reachable set.
                # getattr so the overlay degrades quietly when the MPC is disabled.
                _ref = getattr(tree, "mpc_ref", None)
                if _ref is not None and len(_ref) >= 2:
                    ax.plot(_ref[:, 0], _ref[:, 1], color="#FF00FF", lw=1.5,
                            alpha=0.8, ls="--", zorder=7)
                    ax.plot(_ref[-1, 0], _ref[-1, 1], marker="x", color="#FF00FF",
                            markersize=8, mew=2, zorder=8)
                _pred = getattr(tree, "mpc_pred", None)
                if _pred is not None and len(_pred) >= 2:
                    ax.plot(_pred[:, 0], _pred[:, 1], color="#00CED1", lw=2.2,
                            alpha=0.95, zorder=8)

                _plotting.plot_robot(ax, tree.robot_position, tree.lidar_range)

                if HJ_contingency_enable:
                    _plotting.plot_reachable_set(ax, hjr_fno, tree.robot_state[2],
                                                 tree.Tf_reach)
                plt.pause(0.001)
                recorder.capture(fig)

            # ---- arrival: s_bot IS the root node ----
            if tree.s_bot.cost_to_goal == 0.0 and tree.s_bot.lmc == 0.0:
                print("Successfully reach the goal!")
                current_state = tree.robot_state
                reached_goal = True
                break

            prev_heading = current_state[2]

        ###### END OF EXECUTION LOOP ######

        if not reached_goal:
            print(f"[WARNING] execution budget exhausted after {iter_max} steps "
                  f"without reaching the goal")
        # the metrics log is written in the finally block, so a crash keeps it

        fig.canvas.mpl_disconnect(cid)

        if not state_history:
            state_history.append(list(current_state) + [goal_idx])

        # ---- per-region obstacle dump (local frames), as before ----
        for i in range(len(hjr_fno.safe_regions)):
            obs = np.array(hjr_fno.obs_list[i])          # shape (N, 3)
            if obs.ndim != 2 or obs.shape[0] == 0:
                continue
            xs, ys = hjr_fno.safe_regions[i][:2]
            obs_local = obs.copy()
            obs_local[:, 0] -= xs
            obs_local[:, 1] -= ys
            print(obs_local.tolist())

        # Needed by the CSV dump below whether or not anything is drawn.
        data = np.vstack(state_history)

        # --------------------------------------------------------------
        # Final summary plot
        # --------------------------------------------------------------
        if showPlot:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(_env.x_range[0], _env.x_range[1] + 1)
            ax.set_ylim(_env.y_range[0], _env.y_range[1] + 1)
            _plotting.plot_env(ax, colorList=None)

            x_traj, y_traj = data[:, 0], data[:, 1]

            ax.scatter(goal_xy[0], goal_xy[1], marker='*', s=300, c='red',
                    edgecolors='black', linewidths=1.5, zorder=10)
            ax.plot(x_traj, y_traj, color='tab:blue', linewidth=2)
            ax.scatter(x_traj[0], y_traj[0], color='red', s=60, zorder=5)
            ax.scatter(x_traj[-1], y_traj[-1], color='red', s=60, zorder=5)

            if HJ_contingency_enable:
                _plotting.plot_reachable_set(ax, hjr_fno, tree.robot_state[2], tree.Tf_reach)

            if save_mode:
                # hold the summary frame for ~1 s, save a still beside the gif, then finalize
                for _ in range(recorder.save_fps):
                    recorder.capture(fig, force=True)
                still = os.path.splitext(recorder.save_path)[0] + "_final.png"
                fig.savefig(still, dpi=130, bbox_inches="tight")
                print(f"[save_mode] wrote final plot to {still}")

            plt.show()

        # ---- distance + CSV ----
        xy = data[:, :2]
        total_distance = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        print("Total XY distance:", total_distance)

        output_dir = "/home/kmuenpra/git/HJR-FNO-ContingencyPlanning/exp_results"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "state_history_oneGoal.csv")
        np.savetxt(file_path, data, delimiter=",",
                   header="x,y,theta,goal_id", comments="")
        print(f"Saved to: {file_path}")

        return tree, data

    finally:
        # finalize the gif no matter how the run ends (goal reached, exception, Ctrl-C)
        recorder.close()

        # NOTE: for evaluating per-step solve time
        # same policy for the metrics log: a crashed or timed-out episode is
        # exactly the one worth having on disk.
        if rec is not None:
            rec.finish(
                goal_reached=reached_goal,
                final_goal_dist=math.hypot(current_state[0] - goal_xy[0],
                                           current_state[1] - goal_xy[1]),
                mem=mem.measure(), nodes=len(tree.tree_nodes),
            )
            rec.print_summary()
            rec.to_csv()


if __name__ == '__main__':
    # optional: python rrtx_FNO3d.py env_A --save_mode --save_every 2
    import argparse

    _ap = argparse.ArgumentParser(description="RRTX-HJR contingency planner")
    _ap.add_argument("scenario", nargs="?", default=None,
                     help="eval/scenarios.py name or path to a scenario .json")
    _ap.add_argument("--save_mode", action="store_true",
                     help="record the run to a gif (headless) instead of showing a window")
    _ap.add_argument("--save_path", default=None, help="output .gif path")
    _ap.add_argument("--save_every", type=int, default=1, help="keep every N-th frame")
    _ap.add_argument("--seed", type=int, default=None,
                     help="seed numpy + random for a reproducible run (sampling is "
                          "otherwise unseeded, so no two runs grow the same tree)")
    # NOTE: for evaluating per-step solve time
    _ap.add_argument("--no_plot", action="store_true",
                     help="headless: skip the live plot (required for timing runs)")
    _ap.add_argument("--max_steps", type=int, default=None,
                     help="episode timeout in CONTROL STEPS (distinct from iter_max, "
                          "which caps outer iterations)")
    _ap.add_argument("--nodes", type=int, default=None,
                     help="initial tree size (default 3000); the knob for the sweep")
    _ap.add_argument("--arm", default=None, help="label used in the log filename")
    _ap.add_argument("--log_dir", default=None, help="default eval/results/")
    _ap.add_argument("--no_log", action="store_true", help="disable the per-step log")
    _ap.add_argument("--mem_profile", action="store_true",
                     help="also measure cpu_peak_mb via tracemalloc. Costs ~3.5x on "
                          "allocation-heavy code, so it is OFF by default; use it only "
                          "for the runs that feed the memory table")
    _args = _ap.parse_args()
    main(_args.scenario, save_mode=_args.save_mode,
         save_path=_args.save_path, save_every=_args.save_every, seed=_args.seed,
         no_plot=_args.no_plot, max_steps=_args.max_steps, nodes=_args.nodes,
         arm=_args.arm, log_dir=_args.log_dir, no_log=_args.no_log,
         mem_profile=_args.mem_profile)

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