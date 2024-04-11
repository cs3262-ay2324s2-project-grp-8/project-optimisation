import random
from utils import LOG_STATES, LOG_DETAILED

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

    def make_exploratory_choice(self):
        children_visits = map(lambda child: child.visits, self.root_node.children)
        children_visit_probabilities = [visit / self.root_node.visits for visit in children_visits]
        random_probability = random.uniform(0, 1)
        probabilities_already_counted = 0.

        for i, probability in enumerate(children_visit_probabilities):
            if probabilities_already_counted + probability >= random_probability:
                return self.root_node.children[i]

            probabilities_already_counted += probability

    def simulate(self, graph_idx, timestamp, expansion_count):
        for simulate_round in range(expansion_count):
            current_node = self.root_node

            while current_node.expanded:
                current_node = current_node.get_preferred_child(self.root_node)

            self.expand(current_node)
            # state = current_node.state
            # print(f'Graph {graph_idx}, TS {timestamp} - Expansion {simulate_round}, State:\n{state}')

    def expand(self, node):
        self.child_finder(node, self)

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
        child = random.choice(node.children)
        node.children = []
        node.add_child(child)
        child_win_value = self.node_evaluator(child, self)

        if child_win_value != None:
            node.update_win_value(child_win_value)
        else:
            print("from recursive random rollout") if LOG_DETAILED else None
            self.random_rollout(child)
