from typing import Literal
from utils import Graph, Node
from node import AgentNode
import numpy as np
from copy import deepcopy
import random
DEBUG = False

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

# Attribute Accessors:
X = 0
Y = 1
TYPE = 2
IS_HIRED = 3
IS_FIRED_BEFORE = 4
IS_EXTRACTING = 5
EXTRACT_TIME_LEFT = 6
TIMESTAMP = 7
REWARD_EXTRACTED = 8

REWARD = 9
BUDGET_USED = 10
BUDGET_LEFT = 11

# SITE Accessors
IS_SIGHTED = 3
ACCESSED = 4

DELTA_TO_ACTIONS: dict[tuple[Literal[0], Literal[1]] | tuple[Literal[0], Literal[-1]] | tuple[Literal[1], Literal[0]] | tuple[Literal[-1], Literal[0]] | tuple[Literal[1], Literal[1]] | tuple[Literal[-1], Literal[1]] | tuple[Literal[1], Literal[-1]] | tuple[Literal[-1], Literal[-1]], int] = {
    (0, 1) : MOVE_NORTH,
    (0, -1) : MOVE_SOUTH,
    (1, 0) : MOVE_EAST,
    (-1, 0) : MOVE_WEST,
    (1, 1) : MOVE_NORTH_EAST,
    (-1, 1): MOVE_NORTH_WEST,
    (1, -1): MOVE_SOUTH_EAST,
    (-1, -1): MOVE_SOUTH_WEST
}

class MultiAgentController:
    def __init__(self, graph : Graph):
        self.graph : Graph = graph
        self.ssp: dict = self.floyd_warshall(graph)
        self.N = 20
        self.nWorkers = 9

    def floyd_warshall(self, graph: Graph) :
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
        return ssp

    
    def ucb(self, n : AgentNode, C=1.4):
        ucb = np.inf if n.N == 0 else n.U / n.N + C * np.sqrt(np.log(n.parent.N) / n.N)
        return ucb
    
    def search(self, root : AgentNode, nSimulations=150, nWorkers=9):
        if (self.is_terminal(root)):
            return None
        for i in range(nSimulations):
            print(f"Simulation {i + 1}") if DEBUG else None
            self.execute_round(root)

        max_state: AgentNode = max(root.children.values(), key=lambda p : p.N)
        print(f"MAX STATE : {max_state}") if DEBUG else None
        return max_state

    def execute_round(self, n : AgentNode):
        print("Executing Round") if DEBUG else None
        node: AgentNode = self.select_node(n) # start with worker index 0 # either its children or itself
        print(f"Selected Node {node}") if DEBUG else None
        child : AgentNode = self.expand(node)
        print("Expanded Child") if DEBUG else None
        reward = self.rollout(child)
        print(f"reward : {reward}") if DEBUG else None
        self.backpropagate(node, reward)

    def select_node(self, n : AgentNode):
        if n.children:
            highest = -np.inf
            candidate = None
            for _, child in n.children.items():
                if (self.ucb(child) > highest):
                    highest = self.ucb(child)
                    candidate = child
            return candidate
        return n
    
    def rollout(self, n : AgentNode):
        cost = 1
        state = n.state # Agent 0's state
        i : int = n.idx # Agent 0
        while not self.is_terminal_state(state):
            i = (i + 1) % 9
            print(f"Simulation : Worker {i}") if DEBUG else None
            moves: list = self.get_possible_moves(state, next_idx=i)
            print(f"Possible Moves for Worker {i} : moves") if DEBUG else None
            action: int = random.choice(list(moves))
            print(f"Action Chosen : {action}") if DEBUG else None
            cost += 1
            state = self.move(action, i, state=state)
            print(f"New state : {state}") if DEBUG else None
        print(f"Rollout completed : state {state} is terminal") if DEBUG else None
        return self.utility(state)
            

    def expand(self, n : AgentNode):
        if (self.is_terminal(n)):
            print(f"n : {n} is terminal!") if DEBUG else None
            return n
        ns_ = n.state
        next_idx:int = 0 if n.idx == -1 else (n.idx + 1) % 9
        # Get possible moves for the current Agent i.e Agent 0
        moves : list = self.get_possible_moves(ns_, next_idx=next_idx) 
        for move in moves:
            # Invoke the move on current Agent i.e Agent 0
            new_state = self.move(move, next_idx, ns_)
            #print(f"Cfm parent : {n}")
            print(f"IDX : {n.idx}") if DEBUG else None
            n.children[move] = AgentNode(parent=n, state=new_state, idx=next_idx)
            # Above: Fix Agent 0 move-state to parent
        print(n.children) if DEBUG else None
        for _, child in n.children.items():
            assert(child.parent is not None)
        return self.select_node(n)
    
    def backpropagate(self, n : AgentNode, reward):
        n.U += reward
        n.N += 1
        if n.parent:
            self.backpropagate(n.parent, reward)
    
    def is_terminal(self, node : AgentNode):
        if node.state[8][TIMESTAMP] >= 20 or node.state[BUDGET_LEFT] < 100:
            return True
        return False

    def is_terminal_state(self, state:list):
        if state[8][TIMESTAMP] >= 20 or state[BUDGET_LEFT] < 100:
            return True
        else:
            return False

    def get_possible_moves(self, state:list, next_idx:int):
        possible_move : list = list()
        w : list = state[next_idx]
        if state[BUDGET_LEFT] < (500 if w[TYPE] == 3 else 100 * w[TYPE]):
            return [FIRE]
        if not w[IS_HIRED] and not w[IS_FIRED_BEFORE]:
            return [IDLE]
        if w[IS_FIRED_BEFORE]:
            return [IDLE]
        if w[IS_EXTRACTING] and w[EXTRACT_TIME_LEFT] > 0:
            return [EXTRACT]
        curr_node: Node = self.graph.get_Node(w[X], w[Y])
        if (curr_node.get_reward() > 0 and curr_node.get_type() - 1 <= w[TYPE]):
            rwd_site = None
            for reward_idx in range(12, 12 + 9):
                site = state[reward_idx]
                if (site[X] == w[X] and site[Y] == w[Y]):
                    rwd_site = site
                    break
            if not rwd_site[ACCESSED]:
                return [EXTRACT]
        adj_nodes : list = self.graph.get_adjacent_nodes_by_coordinates(w[X], w[Y])
        for next_node in adj_nodes:
            new_x , new_y = next_node.get_coordinate()
            new_coords = (new_x, new_y)
            for rwd_idx in range(12, 12 + 9):
                rwd_coords = (state[rwd_idx][X], state[rwd_idx][Y])
                if state[rwd_idx][ACCESSED]: # someone chope liao
                    continue
                if state[rwd_idx][TYPE] > w[TYPE]: # worker too noob liao
                    break
                reward_signal:int = (5000 if state[rwd_idx][TYPE] == 3 else 1000 * state[rwd_idx][TYPE])
                cost_signal : int = (500 if w[TYPE] == 3 else 100 * w[TYPE]) * (state[rwd_idx][TYPE] - 2 + self.ssp[new_coords][rwd_coords])
                profit_signal: int = reward_signal - cost_signal
                print(f"profit : {profit_signal}") if DEBUG else None
                if (profit_signal > 0):
                    possible_move.append(DELTA_TO_ACTIONS[(new_x - w[X], new_y - w[Y])])
        if len(possible_move) == 0:
            return [FIRE]
        else:
            return possible_move


    def utility(self, state:list):
        return state[REWARD] - state[BUDGET_USED]

    def move(self, action : int, worker_idx : int, state: list) -> list:
        new_state : list = deepcopy(state)
        new_state[worker_idx][TIMESTAMP] += 1
        moves_dict:dict[int, tuple[int, int]] = {
            MOVE_NORTH: (0, 1),
            MOVE_SOUTH: (0, -1),
            MOVE_EAST: (1, 0),
            MOVE_WEST: (-1, 0),
            MOVE_NORTH_EAST: (1, 1),
            MOVE_NORTH_WEST: (-1, 1),
            MOVE_SOUTH_EAST: (1, -1),
            MOVE_SOUTH_WEST: (-1, -1)
        }
        if action in moves_dict:
            dx, dy = moves_dict[action]
            new_state[worker_idx][X] += dx
            new_state[worker_idx][Y] += dy
        elif action == IDLE:
            pass
        elif action == HIRE:
            new_state[worker_idx][IS_HIRED] = True
        elif action == FIRE:
            new_state[worker_idx][IS_HIRED] = False
            new_state[worker_idx][IS_FIRED_BEFORE] = True
        elif action == EXTRACT:
            if (new_state[worker_idx][IS_EXTRACTING]):
                new_state[worker_idx][EXTRACT_TIME_LEFT] -= 1
                if (new_state[worker_idx][EXTRACT_TIME_LEFT] == 0):
                    new_state[worker_idx][IS_EXTRACTING] = False
                    node : Node = self.graph.get_Node(new_state[worker_idx][X], new_state[worker_idx][Y])
                    assert(node.get_reward() in [1000, 2000, 5000])
                    new_state[REWARD] += node.get_reward()
            else:
                new_state[worker_idx][IS_EXTRACTING] = True
                node : Node = self.graph.get_Node(new_state[worker_idx][X], new_state[worker_idx][Y])
                new_state[worker_idx][EXTRACT_TIME_LEFT] = node.get_type() - 1 
            nx, ny = new_state[worker_idx][X], new_state[worker_idx][Y]
            for i in range(12, 12 + 9):
                if (new_state[i][X] == nx and new_state[i][Y] == ny):
                    new_state[i][ACCESSED] = True
                    break 
        """
        adds cost of extractin for each worker type
        """
        if action not in [FIRE, HIRE, IDLE]:
            new_state[BUDGET_USED] += 500 if new_state[worker_idx][TYPE] == 3 else 100 * new_state[worker_idx][TYPE]
            new_state[BUDGET_LEFT] -= 500 if new_state[worker_idx][TYPE] == 3 else 100 * new_state[worker_idx][TYPE]

        return new_state
    
    
    
    
    
    