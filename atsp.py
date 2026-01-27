import math, random, time, matplotlib.pyplot as plt
from heapq import heappush, heappop


import math
from itertools import combinations
from typing import List, Tuple, Dict

INF = float("inf")

class SA:
    '''
    https://github.com/aaronyw/Asymmetric-Travelling-Salesman-Problem-Optimized-by-Simulated-Annealing.git
    '''
    def __init__(self, data, infinity=1<<17, initial_t=0, rate=0.999, stopping_t=0.0001, initial_fitness=1, initial_solution=None, iteration_bound=[1 << 17, 1 << 24], regularization_bound=(0.3, 3), learning_plot=False, silent_mode=False):
        '''
        :param data: as full edge distance matrix or single line input (refer to README.md file)
        :param infinity: a large integer number that is larger than the cost of all possible solutions
        :param initial_t: initial temperature
        :param rate: the rate temperature decreases each iteration
        :param stopping_t: the final temperature accepted by algorithm
        :param initial_fitness: increase this number if algorithm failed to descend in early stage
        :param iteration_bound: the (minimal, maximum) iteration number accepted by algorithm; higher minimal iteration will increase the chance to find better solution
        :param regularization_bound: as (lower_bound, upper_bound) to regularize self.regulator
               - lower_bound affects how high each Tempering can go; the algorithm might fail to descend if this number is too small or fail to jump out of local minimal if this number is too large
               - upper_bound affects how flat the learning line can be; the Temper might lose its magic if this number is too large
        :param learning_plot: setting this as True will also output the detail info for each Temper
        :param silent_mode: setting this as True will only output one dot '.' when finished; used in multi-threading mode
        '''
        self.INF = infinity
        self.rate = rate
        if isinstance(data[0], list):
            self.n = len(data)
            self.M = []
            for row in data:
                self.M.append([self.INF if x == infinity else x for x in row])
        else:
            self.n = data.pop(0)
            self.M = [[self.INF] * self.n for _ in range(self.n)]
            while data:
                u = data.pop(0)
                v = data.pop(0)
                weight = data.pop(0)
                if u != v:
                    self.M[u][v] = weight
        self.T_initial = self.T = initial_t if initial_t else math.sqrt(self.n)
        self.T_stopping = stopping_t
        self.regularization_bound = regularization_bound
        self.iteration_bound = iteration_bound
        self.regulator = 1
        self.fitness = initial_fitness
        self.control = self.fitness + 1
        if initial_solution:
            initial_solution = initial_solution[:-1] if initial_solution[0] == initial_solution[-1] else initial_solution
            self.current_solution = initial_solution if len(initial_solution) == self.n else self.initialization()            
        else:
            self.current_solution = self.initialization()
        self.best_solution = list(self.current_solution)
        self.initial_cost, self.best_cost, self.worst_cost, self.current_cost = [self.trip_cost(self.current_solution)]*4
        self.cost_list = [self.current_cost]
        self.reheat_x = []
        self.reheat_y = []
        self.detailed_info = learning_plot
        self.silent_mode = silent_mode

    def trip_cost(self, candidate):
        array = candidate + [candidate[0]]
        res = [self.M[u][v] for u, v in zip(array[:-1], array[1:])]
        return sum(res)

    def initialization(self):
        node = 0
        res = [node]

        array = list(range(1, self.n))

        while array:
            _array = list(self.M[node])
            _h = []
            for idx, val in enumerate(_array):
                heappush(_h, (val, idx))
            node = None
            while node not in array:
                node = heappop(_h)[1]

            array.remove(node)
            res.append(node)

        return res

    def accept(self, candidate, random_acceptance=True):
        res = False
        candidate_cost = self.trip_cost(candidate)
        if candidate_cost < self.current_cost:
            self.current_cost = candidate_cost
            self.current_solution = candidate
            if candidate_cost < self.best_cost:
                self.best_cost = candidate_cost
                self.best_solution = candidate
                res = True
        elif random_acceptance:
            random.seed(time.time())
            if random.random() < math.exp((self.current_cost - candidate_cost)*self.regulator/self.T):  # probability function
                self.current_cost = candidate_cost
                self.current_solution = candidate
                res = True
            else:
                if self.current_cost > self.worst_cost:
                    self.worst_cost = self.current_cost

        self.T *= self.rate
        self.cost_list.append(self.current_cost)
        return res

    def anneal(self):
        def cycle(_idx):
            if _idx < 0:
                return self.n + _idx
            elif _idx < self.n:
                return _idx
            else:
                return _idx - self.n

        def transform(last):
            _shift = list(range(self.n - 1))
            random.shuffle(_shift)
            candidate = list(self.current_solution)
            for _i in _shift:
                if candidate[_i] not in last:
                    pivot = _i
                    X = candidate[pivot]
                    _shift.remove(pivot)
                    break
            a_idx = cycle(pivot - 1)
            b_idx = cycle(pivot + 1)
            A = candidate[a_idx]
            B = candidate[b_idx]
            for _i in _shift:
                c_idx = cycle(pivot + _i)
                C = candidate[c_idx]
                y_idx = cycle(c_idx + 1)
                Y = candidate[y_idx]
                d_idx = cycle(y_idx + 1)
                D = candidate[d_idx]
                if self.M[A][B] + self.M[Y][X] + self.M[X][D] < self.INF:
                    part_a = list(candidate[:d_idx])
                    part_b = list(candidate[d_idx:])
                    if X in part_a:
                        part_a.remove(X)
                    if X in part_b:
                        part_b.remove(X)
                    if self.accept(part_a + [X] + part_b):
                        return [X]
                if _i and self.M[A][Y] + self.M[Y][B] + self.M[C][X] + self.M[X][D] < self.INF:
                    new_c = list(candidate)
                    new_c[pivot], new_c[y_idx] = new_c[y_idx], new_c[pivot]
                    if self.accept(new_c):
                        return [X, Y]

            return []

        nodes = []
        while self.T > self.T_stopping:
            nodes = transform(nodes)
            # if not nodes:
            #     self.T *= self.rate
            #     self.cost_list.append(self.current_cost)

    def solve(self):
        def sort_order(array):
            idx = array.index(0)
            return array[idx:] + array[0:idx + 1]

        def display(percentage, number):
            print('\r', end='')
            bar = [':']
            space = [' ']
            bar_n = math.ceil(percentage * 50)
            space_n = 50 - bar_n
            print(''.join(bar*bar_n + space*space_n) + '%s' % number, flush=True, end='')

        if not self.silent_mode:
            if self.detailed_info:
                print('Initialized: ', self.best_cost, '| Fitness:', self.fitness, '/', self.control)
            else:
                print(''.join(['FITNESS'] + [' ']*42), 'COST')
        last_best = self.current_cost + 1
        while self.current_cost < last_best and len(self.cost_list) < self.iteration_bound[1] and self.fitness >= 0:
            if len(self.cost_list) > 1:
                self.T = self.T_initial
                self.regulator = max(self.regulator/(self.control - self.fitness), self.regularization_bound[0])
                if not self.silent_mode:
                    if self.detailed_info:
                        print('Temper from', self.current_cost, 'at', len(self.cost_list), '| Fitness:', self.fitness, '/', self.control)
                    else:
                        display(self.fitness/self.control, self.current_cost)
                    # print(self.normalization)
                self.reheat_x.append(len(self.cost_list))
                self.reheat_y.append(self.current_cost)
            last_best = self.current_cost
            last_worst = self.worst_cost
            self.anneal()
            if self.current_cost < last_best:
                self.fitness += 1
                self.control += 1
                if self.current_cost == self.best_cost:
                    self.fitness += 1
                    # self.control += 1
                    # alternative:
                    if self.control <= self.fitness:
                        self.control = self.fitness + 1
                self.regulator = min(self.n/(last_best - self.current_cost), self.regularization_bound[1])
            else:
                self.fitness -= 1
                if self.worst_cost > last_worst:
                    self.fitness += 1
                    self.control = self.fitness + 1
                last_best = self.current_cost + 1
                if not self.fitness and len(self.cost_list) < self.iteration_bound[0]:
                    self.fitness = 3
                    if self.control <= 3:
                        self.control = 4
        if self.best_cost > self.INF:
            return sort_order(self.best_solution), 0  # indication that there might be NO solution for the problem
        if self.detailed_info:
            self.plot_learning()
        else:
            if self.silent_mode:
                print(str(self.best_cost) + '.', end='')
            else:
                print()
        return sort_order(self.best_solution), self.best_cost

    def plot_learning(self):
        print(len(self.cost_list), 'iterations from', self.initial_cost, 'to', self.best_cost)
        plt.plot(list(range(len(self.cost_list))), self.cost_list)
        plt.plot(self.reheat_x, self.reheat_y, 'ro')
        plt.ylabel('Trip Cost')
        plt.xlabel('Iteration')
        plt.show()
        
        
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


def held_karp(
    cost: List[List[float]],
    prefix: List[int],  # e.g. [0,3,2]
) -> Tuple[float, List[int]]:

    start = prefix[0]
    current = prefix[-1]

    # cost of forced prefix
    prefix_cost = 0.0
    for i in range(len(prefix) - 1):
        prefix_cost += cost[prefix[i]][prefix[i + 1]]

    n = len(cost)

    visited = set(prefix)
    nodes = [i for i in range(n) if i not in visited]

    # DP[(subset, j)] = min cost to start at `current`, visit subset, end at j
    DP: Dict[Tuple[Tuple[int, ...], int], float] = {}
    parent: Dict[Tuple[Tuple[int, ...], int], int] = {}

    # ------------------
    # Base cases
    # ------------------
    for j in nodes:
        DP[((j,), j)] = cost[current][j]
        parent[((j,), j)] = current

    # ------------------
    # Build DP
    # ------------------
    for size in range(2, len(nodes) + 1):
        for subset in combinations(nodes, size):
            subset = tuple(sorted(subset))
            for j in subset:
                prev_subset = tuple(x for x in subset if x != j)

                best_cost = INF
                best_prev = None

                for k in prev_subset:
                    key = (prev_subset, k)
                    if key not in DP:
                        continue

                    c = DP[key] + cost[k][j]
                    if c < best_cost:
                        best_cost = c
                        best_prev = k

                if best_prev is not None:
                    DP[(subset, j)] = best_cost
                    parent[(subset, j)] = best_prev

    # ------------------
    # Close cycle back to start
    # ------------------
    full_subset = tuple(sorted(nodes))
    min_cost = INF
    last_node = None

    for j in nodes:
        key = (full_subset, j)
        if key not in DP:
            continue

        c = DP[key] + cost[j][start]
        if c < min_cost:
            min_cost = c
            last_node = j

    if last_node is None:
        return INF, []

    # ------------------
    # Reconstruct remainder
    # ------------------
    remainder = []
    subset = full_subset
    j = last_node

    while subset:
        remainder.append(j)
        prev_j = parent[(subset, j)]
        subset = tuple(x for x in subset if x != j)
        j = prev_j

    remainder.reverse()

    full_tour = prefix + remainder + [start]
    total_cost = prefix_cost + min_cost

    return total_cost, full_tour
