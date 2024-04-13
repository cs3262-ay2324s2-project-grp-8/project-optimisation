from utils import Graph, Node
import numpy as np
from copy import deepcopy
import math
from mamcts_agent import MultiAgentController
from node import AgentNode

filename : str = "../graphs/graph1.json"
graph = Graph(filename)
timestamp : int = 0
origin = graph.get_Origin()
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

if __name__ == "__main__":

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

    print_state(state) # sanity check
    
    """
    Initialization for MCTS
    """
    controller : MultiAgentController = MultiAgentController(graph=graph)
    root : AgentNode = AgentNode(parent=None, state=state, idx=-1)

    for i in range(9 * 20):
        next_node: AgentNode = controller.search(root)
        if next_node is None:
            print(f"Final State : {root.state}")
            break
        print(f"NEXT STATE : {next_node.state}")
        root = next_node

    print(f"Final State: {root.state}")
    print(f"Root type : {root.idx}")
    print(f"Profit : {root.state[9] - root.state[10]}")

    # from time import time
    # overall_start = time()

    # for timestamp in range(20):
        
    #     start = time()
        
    #     print('\n===============================')
    #     print(f"Timestamp {timestamp+1}") if LOG_BASIC else None
    #     montecarlo.simulate(50)
    #     print(f"Simulation done for TS {timestamp+1}") if LOG_BASIC else None
    #     new_tree_node : TreeNode = montecarlo.make_choice()
    #     montecarlo.root_node = new_tree_node

    #     print(f"Time taken for TS {timestamp+1} : {time() - start}") if LOG_TIME_TAKEN else None
    
    # print(f"Overall Time taken : {time() - overall_start}") if LOG_TIME_TAKEN else None
    # print("Finished Running MCTS") if LOG_BASIC else None
    # print_state(montecarlo.root_node.state) if LOG_BASIC else None