from mcts import MonteCarlo
from node import TreeNode
from actions import move_immutable, step_state, check_move_ok
from reward import calculate_reward_and_best_action
from utils import Graph, Node
from itertools import product
import numpy as np
from copy import deepcopy
import math

filename : str = "../graphs/graph1.json"
graph = Graph(filename)
timestamp : int = 0
origin = graph.get_Origin()

'''
State Representation:
[W1, W2, W3, W4, W5, W6, W7, W8, W9, rewards_extracted, budget_used, budget_left, S1, S2, ... , S9]
W_{i} = (x, y, type, is_hired, is_fired, is_extracting, extraction_time_left, timestamp)
S_{i} = (x, y, type, is_sighted, accessed)

Actions:

0 - Move North
1 - Move South
2 - Move East
3 - Move West
4 - Move NE
5 - Move NW
6 - Move SE
7 - Move SW
8 - Hire
9 - Extract
10 - Don’t Move (Idle for those that are not hired)
11 - Fire
'''

# WORKER Accessors:
X = 0
Y = 1
TYPE = 2
IS_HIRED = 3
IS_FIRED_BEFORE = 4
IS_EXTRACTING = 5
EXTRACT_TIME_LEFT = 6
TIMESTAMP = 7

# SITE Accessors
IS_SIGHTED = 3
ACCESSED = 4

MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_EAST = 2
MOVE_WEST = 3
MOVE_NORTH_EAST = 4
MOVE_NORTH_WEST = 5
MOVE_SOUTH_EAST = 6
MOVE_SOUTH_WEST = 7
HIRE = 8
EXTRACT = 9
IDLE = 10
FIRE = 11


INITIAL_BUDGET = 10000

state = list()

# Fill up the initial state
for i in range(1, 10):
    if (1 <= i <= 3):
        type_worker = 1
    elif (i in [4, 5]):
        type_worker = 2
    else:
        type_worker = 3
    state.append([origin.get_x_coordinate(), origin.get_y_coordinate(), type_worker, False, False, False, 0, 1])

state.extend([0, 0, INITIAL_BUDGET])

type_1_sites = graph.retrieve_all_sites_of_type(2)
type_2_sites = graph.retrieve_all_sites_of_type(3)
type_3_sites = graph.retrieve_all_sites_of_type(4)

for site in type_1_sites:
    state.append([site.get_x_coordinate(), site.get_y_coordinate(), 1, False, False])
for site in type_2_sites:
    state.append([site.get_x_coordinate(), site.get_y_coordinate(), 2, False, False])
for site in type_3_sites:
    state.append([site.get_x_coordinate(), site.get_y_coordinate(), 3, False, False])

def print_state(state):
    for i, worker in enumerate(state):
        if (i < 9):
            print(f"Worker {i + 1}'s state : {worker}")
        elif (i == 9):
            print(f"Rewards Extracted So Far : {state[9]}")
        elif (i == 10):
            print(f"Budget Used : {state[10]}")
        elif (i == 11):
            print(f"Budget Left : {state[11]}")
        else:
            print(f"Reward Site : {state[i]}")

print_state(state)

def floyd_warshall(graph: Graph) :
        vertices = graph.get_vertices()
        edges = graph.get_edges()
        ssp = dict()
        for v1 in vertices.keys():
            for v2 in vertices.keys():
                if (v1 not in ssp.keys()):
                    ssp[v1] = dict()
                if (v2 not in ssp.keys()):
                    ssp[v2] = dict()
                if v1 == v2 :
                    ssp[v1][v2] = 0
                    continue
                ssp[v1][v2] = 1 if (v2 in edges[v1]) else np.inf
                ssp[v2][v1] = 1 if (v1 in edges[v2]) else np.inf
        for k in vertices.keys():
            for i in vertices.keys():
                for j in vertices.keys():
                    ssp[i][j] = min(ssp[i][j], ssp[i][k] + ssp[k][j])
        assert(len(vertices.keys()) == len(ssp.keys()))
        # for i in vertices.keys():
        #     print("===========================")
        #     for j in vertices.keys():
        #         print(f"{i} -> {j} : {ssp[i][j]}")
        return ssp

ssp = floyd_warshall(graph=graph)

current_state = TreeNode(state)

montecarlo: MonteCarlo = MonteCarlo(current_state)

# child finder
def child_finder(node : TreeNode, montecarlo : MonteCarlo):
    state : list = node.state
    moves : list = list()
    if state[0][TIMESTAMP] == 20 :
        return
    for worker_idx in range(9):
        w = state[worker_idx]
        if (not w[IS_HIRED]):
            moves.append([HIRE, IDLE])
        else :
            if (w[IS_EXTRACTING] and w[EXTRACT_TIME_LEFT] > 0):
                moves.append([EXTRACT])
            else:
                curr_node:Node = graph.get_Node(w[X], w[Y])
                if (curr_node.get_reward() > 0 and curr_node.get_type() - 1 <= w[TYPE]):
                    # accessible reward note
                    rwd_site = None
                    for reward_idx in range(12, 12 + 9):
                        site = state[reward_idx]
                        if (site[X] == w[X] and site[Y] == w[Y]):
                            rwd_site = site
                            break
                    if (rwd_site[ACCESSED]):
                        continue
                    else :
                        moves.append([EXTRACT])
                else :
                    # Not a reward node
                    adj_nodes: list = graph.get_adjacent_nodes_by_coordinates(w[X], w[Y])
                    movement_actions = []
                    for next_node in adj_nodes:
                        profit, action = calculate_reward_and_best_action(state, (w[X], w[Y]), (next_node[X], next_node[Y]), ssp, w[TYPE])
                        movement_actions.append((action, profit))
                    sorted(movement_actions, reverse=True, key=lambda x : x[1])
                    if (len(movement_actions) == 0):
                        moves.append([FIRE])
                    elif (movement_actions[0][1] < 0):
                        moves.append([FIRE])
                    else :
                        moves.append([movement_actions[0][0], movement_actions[1][0]])
    movement_combinations = list(product(*moves))
    for move_combi in movement_combinations:
        if (check_move_ok(state=state, move_combi=move_combi, graph=graph)):
            node.add_child(step_state(state, move_combi, graph))


# node evaluator
def node_evaluator(node, montecarlo):
    MAX_PROFIT_ESTIMATE = 35000
    MIN_PROFIT_ESTIMATE = -35000
    if (node.state[0][TIMESTAMP] == 20 or node.state[11] <= 0):
        profit = node.state[9] - node.state[10]
        return profit/(MAX_PROFIT_ESTIMATE - MIN_PROFIT_ESTIMATE)



for timestamp in range(1, 21):
    pass