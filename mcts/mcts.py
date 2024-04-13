import random
import time
from utils import LOG_STATES, LOG_DETAILED

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

class MonteCarlo:

    def __init__(self, root_node):
        self.root_node = root_node
        self.child_finder = None
        self.node_evaluator = lambda child, montecarlo: None

    def make_choice(self):
        best_children = []
        best_score = float('-inf')
        most_visits = float('-inf')
        
        if len(self.root_node.children) == 0:
            return self.root_node
        
        print(f'Number of Potential Children: {len(self.root_node.children)}') if LOG_DETAILED else None

        for child in self.root_node.children:
            if child.visits > most_visits:
                most_visits = child.visits
                best_children = [child]
            elif child.visits == most_visits:
                best_children.append(child)

        return random.choice(best_children)

    def simulate(self, graph_idx, timestamp, expansion_count):
        
        overall_start_time = time.time()
        
        print(' ')
        print(f'[BEFORE] Graph {graph_idx}, TS {timestamp} - Simulation {0}')
        print_state(self.root_node.state)
        print(' ')
        
        for simulate_round in range(expansion_count):
            
            start_time = time.time()
            
            current_node = self.root_node

            while current_node.expanded:
                current_node = current_node.get_preferred_child(self.root_node)

            self.expand(current_node)
            # state = current_node.state
            # print(f'Graph {graph_idx}, TS {timestamp} - Simulation {simulate_round+1}, Time Taken: {time.time() - start_time}')
            # print_state(current_node.state)
            # print(' ')
        print(f'[AFTER] Graph {graph_idx}, TS {timestamp} - Time Taken: {time.time() - overall_start_time}')
        print_state(current_node.state)
        print(' ')

    def expand(self, node):
        self.child_finder(node, self)
        
        if len(node.children) == 0:
            child_win_value = self.node_evaluator(node, self, True)
            node.update_win_value(child_win_value)
        
        for child in node.children:
            # print(f'positive reward') if child.state[9] > 0 else None
            child_win_value = self.node_evaluator(child, self)
            if child_win_value != None:
                # print(child_win_value)
                child.update_win_value(child_win_value)

            if not child.is_scorable():
                print("From child is scoreable call") if LOG_DETAILED else None
                self.random_rollout(child)
                child.children = []

        if len(node.children):
            node.expanded = True

    def random_rollout(self, node):
        self.child_finder(node, self)
        
        if len(node.children) != 0:
            child = random.choice(node.children)
            node.children = []
            node.add_child(child)
            child_win_value = self.node_evaluator(child, self)
        else:
            child_win_value = self.node_evaluator(node, self, True)

        if child_win_value != None:
            node.update_win_value(child_win_value)
        else:
            print("from recursive random rollout") if LOG_DETAILED else None
            self.random_rollout(child)
