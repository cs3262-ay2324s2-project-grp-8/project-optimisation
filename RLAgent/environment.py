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
        MOVE_SOUTH_WEST : (-1, -1),
        HIRE : (0, 0),
        EXTRACT: (0, 0),
        IDLE: (0, 0)
    }

    DELTA_TO_ACTIONS = {
        (0, 1) : MOVE_NORTH,
        (0, -1) : MOVE_SOUTH,
        (1, 0) : MOVE_EAST,
        (-1, 0) : MOVE_WEST,
        (1, 1) : MOVE_NORTH_EAST,
        (-1, 1): MOVE_NORTH_WEST,
        (1, -1): MOVE_SOUTH_EAST,
        (-1, -1): MOVE_SOUTH_WEST
    }

    def __init__(self, agents, isTrain=True) -> None:
        self.number_of_graphs_to_train = 10000
        self.number_of_workers = 9
        self.max_timestamps = 20
        self.playoff_iterations = 1 #5000
        self.isTrain = isTrain
        self.filling_steps = 4
        self.steps_b_updates = 500
        self.worker_agents = agents # Note: Worker Agents will each have a model, but it will not be reset.

    def obtain_graph_information(self, id : int):
        name = 'graphs/graph' + str(id) + '.json'
        return Graph(name)
    
    def step(self, state, actions, graph, ts, ssp):
        # Note : ACTIONS is 0-indexed here
        done = False
        reward_signal = 0
        for worker_idx in range(0, len(self.worker_agents)):
            # if worker idle, then dont care. If worker is not hired and action is not hired, dont care
            if (actions[worker_idx] == IDLE - 1 or (not self.worker_agents[worker_idx].isHired() and actions[worker_idx] != HIRE - 1)):
                continue
            worker = self.worker_agents[worker_idx]
            w_x, w_y = state[worker_idx], state[worker_idx + 1] 
            if (MOVE_NORTH <= actions[worker_idx] + 1 <= MOVE_SOUTH_WEST):
                # standard move.
                assert(worker.isHired())
                new_x , new_y = state[worker_idx] + Environment.ACTIONS_TO_DELTA[actions[worker_idx]][0], state[worker_idx + 1] + Environment.ACTIONS_TO_DELTA[actions[worker_idx]][1]
                reward_signal_i, _ = self.calculate_reward(graph, (state[worker_idx], state[worker_idx + 1]), (new_x, new_y), ssp, worker.get_type(), state[CURRENT_BUDGET])
                reward_signal += 0 if reward_signal_i == -np.inf else reward_signal_i
                state[worker_idx] = new_x
                state[worker_idx + 1] = new_y
                state[COST_INCURRED] += worker.get_rate()
                state[CURRENT_BUDGET] -= worker.get_rate()
                worker.move_to_coordinates(state[worker_idx], state[worker_idx + 1])
            if (actions[worker_idx] == HIRE - 1):
                loc = (state[worker_idx], state[worker_idx + 1])
                reward_signal_i, _ = self.calculate_reward(graph, loc, loc, ssp, worker.get_type(), state[CURRENT_BUDGET])
                reward_signal += 0 if reward_signal_i == -np.inf else reward_signal_i
                worker.hire()
            if (actions[worker_idx] == EXTRACT - 1):
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
                    reward_signal += worker.reward_at_location(graph, zero_out=False) - worker.get_type() * worker.get_rate()
        if (ts >= self.max_timestamps or state[CURRENT_BUDGET] <= 0):
            done = True
        return state, reward_signal, done

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
        return ssp
    
    def calculate_reward(self, graph, curr_loc, next_loc, ssp, agent_type, curr_budget):
        max_calculated_reward = -np.inf
        best_action = None
        for i in range(1, agent_type + 1):
            reward_sites = graph.retrieve_all_sites_of_type(i)
            for rs in reward_sites:
                if (graph.workers_cost_rate[agent_type-1] * (i + ssp[next_loc][(rs.get_coordinate())]) > curr_budget):
                    continue
                else:
                    reward = graph.site_type_rewards[i + 1] - graph.workers_cost_rate[agent_type-1] * (i + ssp[next_loc][(rs.get_coordinate())])
                    if (reward > max_calculated_reward):
                        max_calculated_reward = reward
                        if (curr_loc == next_loc):
                            best_action = HIRE
                        else:
                            best_action = Environment.DELTA_TO_ACTIONS[(next_loc[0] - curr_loc[0], next_loc[1] - curr_loc[0])]
        return max_calculated_reward, best_action

    def run_for_graph(self, graph: Graph):

        total_step = 0
        max_profit = -np.inf
        origin = graph.get_Origin()
        type1_sites = graph.retrieve_all_sites_of_type(2)
        type2_sites = graph.retrieve_all_sites_of_type(3)
        type3_sites = graph.retrieve_all_sites_of_type(4)
        vertices = graph.get_vertices()
        edges = graph.get_edges()
        shortest_path = dict()
        print(graph)

        shortest_path = self.floyd_warshall(graph)

        for play_off_iters in range(1, self.playoff_iterations + 1):

            '''
            This part is for state reset and worker agents reset position - worker agents reset on location, but not their models
            '''
            profit_history = []
            for agent in self.worker_agents:
                agent.reset_worker_without_model(origin) # Does not AND should not reset the model
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
                    a = agent.greedy_move(state, graph, self.ACTIONS_TO_DELTA, shortest_path, self.calculate_reward)
                    print("time_step: ", time_step, "0-index action: ", a , " for agent")
                    actions.append(a)
                next_state, reward, done = self.step(state, actions, graph=graph, ts=time_step, ssp=shortest_path)
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

            print("Graph {p}, Profit {profit}, Final Timestamp {ts}, Done? {done}".format(p=play_off_iters, profit=reward_all, ts=time_step, done=done))

            if self.isTrain:
                if total_step % 100 == 0:
                    if profit_all > max_profit:
                        max_profit = profit_all
                        for agent in self.worker_agents:
                            # print(agent.agent_name)
                            # print(agent.brain)
                            agent.brain.save_model()
        print(f'Graph finished running')

    def train(self, number_of_graphs=1):

        for graph_number in range(0, number_of_graphs):
            graph = Graph(f"../graphs/graph{graph_number + 1}.json")
            # graph = Graph(f"../graphs/training_graphs/g{graph_number + 1}.json")
            self.run_for_graph(graph=graph)

        print(f"Finished Training")




                               

