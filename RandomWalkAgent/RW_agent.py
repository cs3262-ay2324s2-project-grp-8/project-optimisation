from copy import deepcopy
from RW_utils import Graph, Worker, Node
from RW_utils import DEBUG_PROFIT_DETAILS, DEBUG_SCHEDULE
import random

class RandomWalkAgent() :

    def __init__(self, graph : Graph) -> None:
        self.graph: Graph = graph
        self.schedule: list = []
        self.BUDGET = 10000.00
        self.REWARD = 0.00
        self.COST = 0.00
        self.actions = ["HIRE_1", "HIRE_2", "HIRE_3", "MOVE", "EXTRACT"]
        self.worker_list: list[Worker] = []

    def hire_worker(self, action, ts):
        schedule = None
        if (self.BUDGET >= self.graph.get_worker_cost(1) and action == "HIRE_1"):
            schedule = (1, self.graph.get_Origin(), ts + 1)
            self.COST += self.graph.get_worker_cost(1)
            self.BUDGET -= self.graph.get_worker_cost(1)
            self.worker_list.append(Worker(1, self.graph.get_Origin(), self.graph.get_worker_cost(1), ts))
        if (self.BUDGET >= self.graph.get_worker_cost(2) and action == "HIRE_2"):
            schedule= (2, self.graph.get_Origin(), ts + 1)
            self.COST += self.graph.get_worker_cost(2)
            self.BUDGET -= self.graph.get_worker_cost(2)
            self.worker_list.append(Worker(2, self.graph.get_Origin(), self.graph.get_worker_cost(2), ts))
        if (self.BUDGET >= self.graph.get_worker_cost(3) and action == "HIRE_3"):
            schedule=(3, self.graph.get_Origin(), ts + 1)
            self.COST += self.graph.get_worker_cost(3)
            self.BUDGET -= self.graph.get_worker_cost(3)
            self.worker_list.append(Worker(3, self.graph.get_Origin(), self.graph.get_worker_cost(3), ts))
        return schedule

    def get_schedule(self):
        for ts in range(20):
            print(ts) if DEBUG_SCHEDULE else None
            schedule = []
            if (len(self.worker_list) == 0): # Case where its empty
                print("Choosing to hire worker first") if DEBUG_SCHEDULE else None
                while(len(self.worker_list) == 0):
                    chosen_action = random.choice(["HIRE_1", "HIRE_2", "HIRE_3"])
                    schedule1 = self.hire_worker(chosen_action, ts)
                    print(schedule1) if DEBUG_SCHEDULE else None
                    schedule.append(schedule1)
            else : 
                print(self.schedule) if DEBUG_SCHEDULE else None
                for worker_state in self.schedule[-1]:
                    worker_type, worker_location, worker_ts = worker_state
                    if (worker_ts > ts):
                        schedule.append(deepcopy(worker_state))
                    else:
                        if (not self.graph.is_reward_site_by_node(node=worker_location)):
                            assert(worker_ts == ts)
                            # deduct cost
                            if (self.BUDGET < self.graph.get_worker_cost(worker_type)):
                                continue
                            self.COST += self.graph.get_worker_cost(worker_type)
                            self.BUDGET -= self.graph.get_worker_cost(worker_type)
                            neighbours = self.graph.get_adjacent_nodes_by_Node(worker_location)
                            next_state: Node = random.choice(neighbours)
                            if (self.graph.is_reward_site_by_node(next_state) and not next_state.is_accessed_before() and worker_type >= next_state.get_type()):
                                next_state.access()
                                schedule.append((worker_type, next_state, ts + next_state.get_acquire_time() + 1))
                            else :
                                schedule.append((worker_type,next_state, ts + 1))
                        else:
                            # worker is in reward site
                            assert(worker_ts == ts)
                            if (self.BUDGET >= self.graph.get_worker_cost(worker_type) * worker_location.get_acquire_time()):
                                self.BUDGET -= self.graph.get_worker_cost(worker_type) * worker_location.get_acquire_time()
                                self.COST += self.graph.get_site_reward(worker_location.get_type() - 1)
                                self.REWARD += worker_location.get_reward()
                                print("Get reward") if DEBUG_PROFIT_DETAILS else None
                                if (self.BUDGET < self.graph.get_worker_cost(worker_type)):
                                    continue
                                neighbours = self.graph.get_adjacent_nodes_by_Node(worker_location)
                                next_state: Node = random.choice(neighbours)
                                if (self.graph.is_reward_site_by_node(next_state) and not next_state.is_accessed_before() and worker_type >= next_state.get_type()):
                                    next_state.access()
                                    schedule.append((worker_type, next_state, ts + next_state.get_acquire_time() + 1))
                                else :
                                    schedule.append((worker_type,next_state, ts + 1))
                            else:
                                # fire the worker
                                continue
                if (random.choice([True, False]) and self.BUDGET >= self.graph.get_worker_cost(1)):
                    while(True):
                        schedule1 = self.hire_worker(random.choice(["HIRE_1", "HIRE_2", "HIRE_3"]), ts)
                        if (schedule1 is None):
                            new_worker = None
                        else:
                            schedule.append(schedule1)
                            break
            self.schedule.append(schedule)
        print(self.schedule) if DEBUG_SCHEDULE else None
        return self.schedule
    
    def get_profit(self):
        print(f'Rewards: {self.REWARD}, COST: {self.COST}, $ LEFT: {self.BUDGET}') if DEBUG_PROFIT_DETAILS else None
        return self.REWARD - self.COST

    def get_worker_list(self):
        
        new_worker_list = []
        for worker in self.worker_list:
            new_worker_list.append(worker.type)
        
        return new_worker_list