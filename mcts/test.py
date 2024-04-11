from mcts import MonteCarlo
from node import TreeNode
from actions import check_state_ok, step_state
from reward import calculate_reward_and_best_action
from utils import Graph, Node
from utils import LOG_STATES, LOG_DETAILED, LOG_TIME_TAKEN, LOG_PROFIT, LOG_TIMESTAMPS, LOG_ACTIONS
from itertools import product
import numpy as np
from copy import deepcopy
import math
import sys
import random

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

REWARD_EXTRACTED_IDX = 9
BUDGET_USED_IDX = 10
BUDGET_LEFT_IDX = 11
REWARD_START_IDX = 12

INITIAL_BUDGET = 10000

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
            
def get_test_output(graph_idx, timestamp, state, time_taken):
    graph_number = graph_idx + 1
    profit = state[9] - state[10]
    
    output = f'Graph {graph_number}, TS {timestamp+1} - Profit: {profit} [Time: {time_taken}]'
    
    return output

def get_extracting_workers_count(state: list) -> int:
    count = 0
    for i in range(9):
        if (state[i][IS_EXTRACTING]):
            count += 1
    return count

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

def child_finder(node : TreeNode, montecarlo : MonteCarlo):
    state : list = node.state
    moves : list = list()
    if state[0][TIMESTAMP] == 20 or state[11] <= 0:
        return
       
    for worker_idx in range(9):
        w = state[worker_idx]
        if (not w[IS_HIRED]):
            # moves.append([HIRE, IDLE])
            moves.append([HIRE, IDLE])
            print(f'Worker {worker_idx} - Best Actions: HIRE, IDLE') if LOG_ACTIONS else None
            
        else :
            curr_node:Node = graph.get_Node(w[X], w[Y])
            
            # Worker is extracting, continue extracting
            if (w[IS_EXTRACTING] and w[EXTRACT_TIME_LEFT] > 0):
                moves.append([EXTRACT])
                print(f'Worker {worker_idx} - Best Actions: Continue Extracting') if LOG_ACTIONS else None
                
            # Worker has been fired before, IDLE it
            elif (w[IS_FIRED_BEFORE]):
                moves.append([IDLE])
                print(f'Worker {worker_idx} - Best Actions: IDLE') if LOG_ACTIONS else None
                
            else:
                # Current node is a reward node, either extract or move on if already accessed
                if (curr_node.get_reward() > 0 and curr_node.get_type() - 1 <= w[TYPE]):
                    rwd_site = None
                    
                    # Lookup reward_site to check if currently accessed
                    for reward_idx in range(12, 12 + 9):
                        site = state[reward_idx]
                        if (site[X] == w[X] and site[Y] == w[Y]):
                            rwd_site = site
                            break
                        
                    # reward_site is currently being accessed, continue searching
                    if (rwd_site[ACCESSED]):
                        adj_nodes: list = graph.get_adjacent_nodes_by_coordinates(w[X], w[Y])
                        movement_actions = []
                        
                        # search for the best action to take
                        for next_node in adj_nodes:
                            print("next node: ", next_node) if LOG_DETAILED else None
                            profit, action = calculate_reward_and_best_action(state, (w[X], w[Y]), next_node.get_coordinate(), ssp, w[TYPE])
                            movement_actions.append((action, profit))
                        movement_actions = sorted(movement_actions, reverse=True, key=lambda x : x[1])
                        print("movement actions : ", movement_actions) if LOG_DETAILED else None
                        print(f'Worker {worker_idx} at ({w[X]}, {w[Y]}) - Best Actions: {movement_actions[:5]}') if LOG_ACTIONS else None

                        # No valid positive movement actions, FIRE
                        if (len(movement_actions) == 0 or movement_actions[0][1] <= 0):
                            moves.append([FIRE])
                        
                        # appends two best action
                        else:
                            curr_moves = []
                            for action in movement_actions:
                                if action[1] > 0:
                                    curr_moves.append(action[0])
                            moves.append(curr_moves[:2])
                    
                    # reward_site is not being accessed, extract
                    else :
                        print(f'Worker {worker_idx} - Best Actions: Extract {curr_node.get_reward()}') if LOG_ACTIONS else None
                        moves.append([EXTRACT])
                
                # Current node is NOT a reward node, add movement actions
                else :
                    adj_nodes: list = graph.get_adjacent_nodes_by_coordinates(w[X], w[Y])
                    movement_actions = []
                    
                    # search for the best action to take
                    for next_node in adj_nodes:
                        print("next node: ", next_node) if LOG_DETAILED else None
                        profit, action = calculate_reward_and_best_action(state, (w[X], w[Y]), next_node.get_coordinate(), ssp, w[TYPE])
                        movement_actions.append((action, profit))
                    movement_actions = sorted(movement_actions, reverse=True, key=lambda x : x[1])
                    print("movement actions : ", movement_actions) if LOG_DETAILED else None
                    print(f'Worker {worker_idx} at ({w[X]}, {w[Y]}) - Best Actions: {movement_actions[:5]}') if LOG_ACTIONS else None

                    # No valid positive movement actions, FIRE
                    if (len(movement_actions) == 0 or movement_actions[0][1] <= 0):
                        moves.append([FIRE])
                        
                    # appends two best action
                    else:
                        curr_moves = []
                        for action in movement_actions:
                            if action[1] > 0:
                                curr_moves.append(action[0])
                        moves.append(curr_moves[:2])
                        
    movement_combinations = list(product(*moves))
    print(f'Max expansion this round : {len(movement_combinations)} nodes') if LOG_DETAILED else None
    for move_combi in movement_combinations:
        if len(move_combi) != 9:
            print(f'Original Moves: {moves}')
            print(f"Move combin {move_combi}, Length {len(move_combi)}\n")
        assert(len(move_combi) == 9)

        new_state = step_state(state, move_combi, graph)
        node.add_child(TreeNode(new_state))
    
def node_evaluator(node, montecarlo):
    # MAX_PROFIT_ESTIMATE = 35000
    # MIN_PROFIT_ESTIMATE = -35000
    
    # if (not check_state_ok(node.state)):
    #     # print(f'state not ok, {node.state}')
    #     return -1

    # # print(f'state ok, {node.state}')
    # if (node.state[0][TIMESTAMP] == 20 or node.state[11] <= 0):
    #     profit = node.state[9] - node.state[10]
    #     return profit/(10000) if profit >= 0 else -1
    
    MAX_PROFIT_ESTIMATE = 20000
    MIN_PROFIT_ESTIMATE = -20000
    if (node.state[0][TIMESTAMP] == 20 or node.state[11] <= 0 or len(node.children) == 0):
        profit = node.state[9] - node.state[10]
        normalized_profit = profit/(MAX_PROFIT_ESTIMATE - MIN_PROFIT_ESTIMATE)
        
        return normalized_profit if normalized_profit >= 0 else -1
    
    if (not check_state_ok(node.state)):
        return -1

if __name__ == "__main__":

    TEST_GRAPH_COUNT = 10
    WRITE_TO_LOG = False
    
    if WRITE_TO_LOG:
        log_filename = 'test.log.txt'
        log_file = open(log_filename, 'w')
        sys.stdout = log_file
    
    for graph_idx in range(TEST_GRAPH_COUNT):
        filename : str = f"../graphs/graph{graph_idx+1}.json"
        graph = Graph(filename)
        timestamp : int = 0
        origin = graph.get_Origin()
        state = list()

        for i in range(1, 10):
            type_worker = 1 if 1 <= i <= 3 else 2 if i in [4, 5] else 3
            state.append([origin.get_x_coordinate(), origin.get_y_coordinate(), type_worker, False, False, False, 0, 1])

        state.extend([0, 0, INITIAL_BUDGET])
            
        """
        Retrieves the sites of type 1, 2, 3 and appends them to the state
        """
        def append_sites_of_type(site_type):
            for site in graph.retrieve_all_sites_of_type(site_type + 1):
                state.append([site.get_x_coordinate(), site.get_y_coordinate(), site_type, False, False])

        for site_type in range(1, 4):
            append_sites_of_type(site_type)

        print_state(state) if LOG_STATES else None

        ssp = floyd_warshall(graph=graph)
        print(f'{len(state)}\n{state}') if LOG_DETAILED else None
        current_state = TreeNode(state)
        
        """
        Initialization for MCTS
        """
        montecarlo: MonteCarlo = MonteCarlo(current_state)
        montecarlo.child_finder = child_finder
        montecarlo.node_evaluator = node_evaluator

        from time import time
        overall_start = time()

        
        print(' ')
        for timestamp in range(20):
            
            start = time()
            print('\n===============================') if LOG_TIMESTAMPS else None
            print(f"Timestamp {timestamp+1}") if LOG_TIMESTAMPS else None
            montecarlo.simulate(graph_idx, timestamp, 1200)
            print(f"Simulation done for TS {timestamp+1}") if LOG_STATES else None
            new_tree_node : TreeNode = montecarlo.make_choice()
            montecarlo.root_node = new_tree_node
            
            test_output = get_test_output(graph_idx, timestamp, montecarlo.root_node.state, f'{time() - start}')
            print(test_output)
        
        print(f"Overall Time taken : {time() - overall_start}") if LOG_TIME_TAKEN else None
        print("Finished Running MCTS") if LOG_STATES else None
        print_state(montecarlo.root_node.state) if LOG_STATES else None
        
        if LOG_PROFIT:
            print('==============================================================')
            test_output = get_test_output(graph_idx, -1, montecarlo.root_node.state, time() - overall_start)
            print(test_output)
            print('==============================================================')
    
    if WRITE_TO_LOG:
        log_file.close()
        sys.stdout = sys.__stdout__
