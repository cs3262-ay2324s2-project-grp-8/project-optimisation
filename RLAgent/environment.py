import numpy as np
import os
import random
import argparse
from agent import AgentWorker
from utils import Graph, Worker, Node

'''
1 - Move North
2 - Move South
3 - Move East
4 - Move West
5 - Move NE
6 - Move NW
7 - Move SE
8 - Move SW
9 - Hire
10 - Extract
11 - Don’t Move (Idle for those that are not hired)
'''

MOVE_NORTH = 1
MOVE_SOUTH = 2
MOVE_EAST = 3
MOVE_WEST = 4
MOVE_NORTH_EAST = 5
MOVE_NORTH_WEST = 6
MOVE_SOUTH_EAST = 7
MOVE_SOUTH_WEST = 8
HIRE = 9
EXTRACT = 10
IDLE = 11

CURRENT_BUDGET = -3
COST_INCURRED = -2
REWARDS_EXTRACTED = -1

class Environment(object):

    ACTIONS_TO_DELTA = {
        MOVE_NORTH : (0, 1),
        MOVE_SOUTH : (0, -1),
        MOVE_EAST : (1, 0),
        MOVE_WEST : (-1, 0),
        MOVE_NORTH_EAST : (1, 1),
        MOVE_NORTH_WEST : (-1, 1),
        MOVE_SOUTH_EAST : (1, -1),
        MOVE_SOUTH_WEST : (-1, 1),
        HIRE : (0, 0),
        EXTRACT: (0, 0),
        IDLE: (0, 0)
    }

    def __init__(self, agents, isTrain=True) -> None:
        self.number_of_graphs_to_train = 10000
        self.number_of_workers = 9
        self.max_timestamps = 20
        self.playoff_iterations = 5000
        self.isTrain = isTrain
        self.filling_steps = 4
        self.steps_b_updates = 500
        self.worker_agents = agents # Note: Worker Agents will each have a model, but it will not be reset.

    def obtain_graph_information(self, id : int):
        name = 'graphs/graph' + str(id) + '.json'
        return Graph(name)
    
    def step(self, state, actions, graph, ts):
        done = False
        for worker_idx in range(0, len(self.worker_agents)):
            if (actions[worker_idx] == IDLE or (not self.worker_agents[worker_idx].isHired() and actions[worker_idx] != HIRE)):
                continue
            worker = self.worker_agents[worker_idx]
            w_x, w_y = state[worker_idx], state[worker_idx + 1] 
            if (MOVE_NORTH <= actions[worker_idx] <= MOVE_SOUTH_WEST):
                assert(worker.isHired())
                state[worker_idx] = state[worker_idx] + Environment.ACTIONS_TO_DELTA[actions[worker_idx]][0]
                state[worker_idx + 1] = state[worker_idx + 1] + Environment.ACTIONS_TO_DELTA[actions[worker_idx]][1]
                state[COST_INCURRED] += worker.get_rate()
                state[CURRENT_BUDGET] -= worker.get_rate()
                worker.move_to_coordinates(state[worker_idx], state[worker_idx + 1])
            if (actions[worker_idx] == HIRE):
                worker.hire()
            if (actions[worker_idx] == EXTRACT):
                state[COST_INCURRED] += worker.get_rate()
                state[CURRENT_BUDGET] -= worker.get_rate()
                if (worker.is_extracting()):
                    assert(worker.isHired())
                    worker.decrease_waitTime()
                    if (worker.get_waitTime() == 0):
                        worker.done_extracting()
                        state[REWARDS_EXTRACTED] += worker.reward_at_location(graph, zero_out=True)
                else:
                    assert(worker.isHired())
                    worker.extract(graph)
                    assert(worker.is_extracting())
        if (ts >= self.max_timestamps or state[CURRENT_BUDGET] <= 0):
            done = True
        return state, state[REWARDS_EXTRACTED], done

    def run_for_graph(self, graph: Graph):

        total_step = 0
        max_profit = -np.inf
        origin = graph.get_Origin()
        type1_sites = graph.retrieve_all_sites_of_type(2)
        type2_sites = graph.retrieve_all_sites_of_type(3)
        type3_sites = graph.retrieve_all_sites_of_type(4)
        vertices = graph.get_vertices()
        edges = graph.get_edges()

        for play_off_iters in range(1, self.playoff_iterations + 1):

            '''
            This part is for state reset and worker agents reset position - worker agents reset on location, but not their models
            '''
            profit_history = []
            for agent in self.worker_agents:
                agent.move_back_to_origin(origin) # Does not AND should not reset the model
            state = []
            for p in range(self.number_of_workers):
                state.extend(origin.get_coordinate())
            for type1 in type1_sites:
                state.extend(type1.get_coordinate())
            for type2 in type2_sites:
                state.extend(type2.get_coordinate())
            for type3 in type3_sites:
                state.extend(type3.get_coordinate())
            state.extend([10000, 0 , 0]) # budget ,costs incurred so far, followed by rewards collected so far

            # we will forgo the randomness move from the actual implementation

            state = np.array(state)

            done = False
            reward_all = 0
            time_step = 0

            while not done and time_step <= self.max_timestamps:
                actions = []
                for agent in self.worker_agents:
                    actions.append(agent.greedy_move(state, graph, self.ACTIONS_TO_DELTA))
                next_state, reward, done = self.step(state, actions, graph=graph, ts=time_step)
                next_state = np.array(next_state)

                if self.isTrain :
                    for agent in self.worker_agents:
                        agent.observe((state, actions, reward, next_state, done))
                        if total_step >= self.filling_steps:
                            agent.decay_epsilon()
                            if time_step % self.steps_b_updates == 0:
                                agent.replay()
                            agent.update_target_model()
                total_step += 1
                time_step += 1
                state = next_state
                profit_all = next_state[REWARDS_EXTRACTED] - next_state[COST_INCURRED] # Actually the profit
            profit_history.append(profit_all)

            # print("Graph {p}, Profit {profit}, Final Timestamp {ts}, Done? {done}".format(p=play_off_iters, profit=reward_all, ts=time_step, done=done))

            if self.isTrain:
                if total_step % 100 == 0:
                    if profit_all > max_profit:
                        max_profit = profit_all
                        for agent in self.worker_agents:
                            # print(agent.agent_name)
                            # print(agent.brain)
                            agent.brain.save_model()
        print(f'Graph finished running')

    def train(self, number_of_graphs=5):

        for graph_number in range(0, number_of_graphs):
            graph = Graph(f"./graphs/graph{graph_number + 1}.json")
            # graph = Graph(f"../graphs/training_graphs/g{graph_number + 1}.json")
            self.run_for_graph(graph=graph)

        print(f"Finished Training")




                               

