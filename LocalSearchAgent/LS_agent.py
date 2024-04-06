from copy import deepcopy
from LS_utils import Graph, Worker

class LocalSearchAgent() :
    def __init__(self, graph : Graph) -> None:
        self.steps = 20
        self.processed = False
        self.graph: Graph = graph
        self.origin = graph.get_Origin().get_coordinate()
        self.BUDGET = 10000.00
        self.PROFIT = 0
        self.worker_list = list()
        self.required_vertices_1 = 0
        self.required_vertices_2 = 0
        self.required_vertices_3 = 0
        self.shortest_path = self.floyd_warshall(graph)
        self.specific_shortest_path = self.preprocess(self.shortest_path)

    def floyd_warshall(self, graph):
        inf = float('inf')
        vertices = list(self.graph.get_vertices().keys())
        num_vertices = len(vertices)
        distance = [[inf for _ in range(num_vertices)] for _ in range(num_vertices)]

        for i in range(num_vertices):
            distance[i][i] = 0

        vertices_index = dict()
        for i in range(num_vertices):
            vertices_index[vertices[i]] = i
        
        edges = self.graph.get_edges()    
        for i in range(num_vertices):
            for edge in edges[vertices[i]]:
                distance[i][vertices_index[edge]] = 1

        for k in range(num_vertices):
            for i in range(num_vertices):
                for j in range(num_vertices):
                    curr_distance = distance[i][k] + distance[k][j]
                    if distance[i][j] > curr_distance:
                        distance[i][j] = curr_distance

        return distance

    def preprocess(self, shortest_path):
        vertices = list(self.graph.get_vertices().keys())
        num_vertices = len(vertices)

        vertices_index = dict()
        shortest_distance = dict()
        vertices_required = list()
        
        for i in range(num_vertices):
            vertices_index[vertices[i]] = i
            
            if self.graph.is_reward_site_by_node(self.graph.get_vertices()[vertices[i]]):
                shortest_distance[vertices[i]] = {1: list(), 2: list(), 3: list()}
                vertices_required.append(i)

        for i in vertices_required:
            temp = shortest_path[i]
            for j in range(num_vertices):
                if j not in vertices_required:
                    continue
                if i == j:
                    continue
                vertex_type = self.graph.get_Node(*vertices[j]).get_type() - 1
                if vertex_type != 0:
                    shortest_distance[vertices[i]][self.graph.get_Node(*vertices[j]).get_type() - 1].append([vertices[j], shortest_path[i][j]])
            for k in [1,2,3]:
                shortest_distance[vertices[i]][k].sort(key = lambda x: x[1])

        for i in vertices_required:
            x = self.graph.get_Node(*vertices[i]).get_type() - 1
            if x == 1:
                self.required_vertices_1 += 1
            elif x == 2:
                self.required_vertices_2 += 1
            elif x == 3:
                self.required_vertices_3 += 1

        return shortest_distance

    def simulate(self, worker_list):
        origin = self.graph.get_Origin()
        curr_schedule = list()
        
        all_schedules = list()
        reward = [0]
        cost = [0]
        
        for i in range(len(worker_list)):
            worker_num = worker_list[i]
            curr_schedule.append(Worker(worker_num, self.graph.get_Origin(), self.graph.get_worker_cost(worker_num), 0))
            cost[0] += self.graph.get_worker_cost(worker_num)

        curr_schedule = [curr_schedule]
        all_schedules.append(curr_schedule)
        if cost[0] > self.BUDGET:
            return None, 0, 0, cost[0]

        visited = list()
        for ts in range(1, self.steps):
            temp_schedule = list()
            temp_cost = cost[-1]
            temp_reward = reward[-1]
            for w in all_schedules[-1][-1]:
                if w.get_timestamp() > ts:
                    temp_schedule.append(Worker(w.get_type(), w.get_location(), w.get_rate(), w.get_timestamp()))
                    continue
                
                if w.get_type() == 1:
                    distances_1 = self.specific_shortest_path[w.get_location().get_coordinate()][1]
                    for i in visited:
                        distances_1.remove(i) if i in distances_1 else 0
                    if len(distances_1) == 0:
                        continue
                        
                    best_1 = distances_1[0]
                    next_node_1 = self.graph.get_Node(*best_1[0])
                    if best_1[1] + next_node_1.get_acquire_time() + ts > self.steps:
                        continue

                    if temp_cost + w.get_rate() * best_1[1] > self.BUDGET:
                        continue
                        
                    temp_schedule.append(Worker(w.get_type(), next_node_1, w.get_rate(), best_1[1] + next_node_1.get_acquire_time() + ts))
                    temp_cost += w.get_rate() * best_1[1]
                    temp_reward += next_node_1.get_reward()
                    visited.append(best_1[0])
                
                elif w.get_type() == 2:
                    distances_1 = self.specific_shortest_path[w.get_location().get_coordinate()][1]
                    distances_2 = self.specific_shortest_path[w.get_location().get_coordinate()][2]
                    for i in visited:
                        distances_1.remove(i) if i in distances_1 else 0
                        distances_2.remove(i) if i in distances_2 else 0
                    if len(distances_1) == 0 and len(distances_2) == 0:
                        continue
                    
                    best_1 = distances_1[0] if len(distances_1) > 0 else None
                    best_2 = distances_2[0] if len(distances_2) > 0 else None

                    next_node_1 = self.graph.get_Node(*best_1[0]) if best_1 is not None else None
                    next_node_2 = self.graph.get_Node(*best_2[0]) if best_2 is not None else None
                    
                    best_1 = best_1 if best_1 is not None and best_1[1] + next_node_1.get_acquire_time() + ts <= self.steps else None
                    best_2 = best_2 if best_2 is not None and best_2[1] + next_node_2.get_acquire_time() + ts <= self.steps else None

                    next_node_1 = next_node_1 if next_node_1 is not None and best_1 is not None and temp_cost + w.get_rate() * best_1[1] <= self.BUDGET else None
                    next_node_2 = next_node_2 if next_node_2 is not None and best_2 is not None and temp_cost + w.get_rate() * best_2[1] <= self.BUDGET else None

                    if next_node_1 is None and next_node_2 is None:
                        continue
                    
                    if next_node_1 is not None and next_node_2 is None:
                        temp_schedule.append(Worker(w.get_type(), next_node_1, w.get_rate(), best_1[1] + next_node_1.get_acquire_time() + ts))
                        temp_cost += w.get_rate() * best_1[1]
                        temp_reward += next_node_1.get_reward()
                        visited.append(best_1[0])

                    elif next_node_1 is None and next_node_2 is not None:
                        temp_schedule.append(Worker(w.get_type(), next_node_2, w.get_rate(), best_2[1] + next_node_2.get_acquire_time() + ts))
                        temp_cost += w.get_rate() * best_2[1]
                        temp_reward += next_node_2.get_reward()
                        visited.append(best_2[0])

                    else:
                        profit_1 = next_node_1.get_reward() - w.get_rate() * best_1[1]
                        profit_2 = next_node_2.get_reward() - w.get_rate() * best_2[1]
                        if profit_1 >= profit_2:
                            temp_schedule.append(Worker(w.get_type(), next_node_1, w.get_rate(), best_1[1] + next_node_1.get_acquire_time() + ts))
                            temp_cost += w.get_rate() * best_1[1]
                            temp_reward += next_node_1.get_reward()
                            visited.append(best_1[0])
                        else:
                            temp_schedule.append(Worker(w.get_type(), next_node_2, w.get_rate(), best_2[1] + next_node_2.get_acquire_time() + ts))
                            temp_cost += w.get_rate() * best_2[1]
                            temp_reward += next_node_2.get_reward()
                            visited.append(best_2[0])
                    
                elif w.get_type() == 3:
                    distances_1 = self.specific_shortest_path[w.get_location().get_coordinate()][1]
                    distances_2 = self.specific_shortest_path[w.get_location().get_coordinate()][2]
                    distances_3 = self.specific_shortest_path[w.get_location().get_coordinate()][3]
                    for i in visited:
                        distances_1.remove(i) if i in distances_1 else 0
                        distances_2.remove(i) if i in distances_2 else 0
                        distances_3.remove(i) if i in distances_3 else 0
                    if len(distances_1) == 0 and len(distances_2) == 0 and len(distances_3) == 0:
                        continue
                    
                    best_1 = distances_1[0] if len(distances_1) > 0 else None
                    best_2 = distances_2[0] if len(distances_2) > 0 else None
                    best_3 = distances_3[0] if len(distances_3) > 0 else None

                    next_node_1 = self.graph.get_Node(*best_1[0]) if best_1 is not None else None
                    next_node_2 = self.graph.get_Node(*best_2[0]) if best_2 is not None else None
                    next_node_3 = self.graph.get_Node(*best_3[0]) if best_3 is not None else None
                    
                    best_1 = best_1 if best_1 is not None and best_1[1] + next_node_1.get_acquire_time() + ts <= self.steps else None
                    best_2 = best_2 if best_2 is not None and best_2[1] + next_node_2.get_acquire_time() + ts <= self.steps else None
                    best_2 = best_3 if best_3 is not None and best_3[1] + next_node_3.get_acquire_time() + ts <= self.steps else None

                    next_node_1 = next_node_1 if next_node_1 is not None and best_1 is not None and temp_cost + w.get_rate() * best_1[1] <= self.BUDGET else None
                    next_node_2 = next_node_2 if next_node_2 is not None and best_2 is not None and temp_cost + w.get_rate() * best_2[1] <= self.BUDGET else None
                    next_node_3 = next_node_3 if next_node_3 is not None and best_3 is not None and temp_cost + w.get_rate() * best_3[1] <= self.BUDGET else None
                    
                    if next_node_1 is None and next_node_2 is None and next_node_3 is None:
                        continue
                    
                    if next_node_1 is not None and next_node_2 is None and next_node_3 is None:
                        temp_schedule.append(Worker(w.get_type(), next_node_1, w.get_rate(), best_1[1] + next_node_1.get_acquire_time() + ts))
                        temp_cost += w.get_rate() * best_1[1]
                        temp_reward += next_node_1.get_reward()
                        visited.append(best_1[0])

                    elif next_node_1 is None and next_node_2 is not None and next_node_3 is None:
                        temp_schedule.append(Worker(w.get_type(), next_node_2, w.get_rate(), best_2[1] + next_node_2.get_acquire_time() + ts))
                        temp_cost += w.get_rate() * best_2[1]
                        temp_reward += next_node_2.get_reward()
                        visited.append(best_2[0])

                    elif next_node_1 is None and next_node_2 is None and next_node_3 is not None:
                        temp_schedule.append(Worker(w.get_type(), next_node_3, w.get_rate(), best_3[1] + next_node_3.get_acquire_time() + ts))
                        temp_cost += w.get_rate() * best_3[1]
                        temp_reward += next_node_3.get_reward()
                        visited.append(best_3[0])

                    else:
                        neg_inf = float('-inf')
                        profit_1 = next_node_1.get_reward() - w.get_rate() * best_1[1] if next_node_1 is not None else neg_inf
                        profit_2 = next_node_2.get_reward() - w.get_rate() * best_2[1] if next_node_2 is not None else neg_inf
                        profit_3 = next_node_3.get_reward() - w.get_rate() * best_3[1] if next_node_3 is not None else neg_inf
                        
                        if profit_1 >= profit_2 and profit_1 >= profit_3 and profit_1 != neg_inf:
                            temp_schedule.append(Worker(w.get_type(), next_node_1, w.get_rate(), best_1[1] + next_node_1.get_acquire_time() + ts))
                            temp_cost += w.get_rate() * best_1[1]
                            temp_reward += next_node_1.get_reward()
                            visited.append(best_1[0])
                        elif profit_2 > profit_1 and profit_2 >= profit_3 and profit_2 != neg_inf:
                            temp_schedule.append(Worker(w.get_type(), next_node_2, w.get_rate(), best_2[1] + next_node_2.get_acquire_time() + ts))
                            temp_cost += w.get_rate() * best_2[1]
                            temp_reward += next_node_2.get_reward()
                            visited.append(best_2[0])
                        elif profit_3 > profit_1 and profit_3 > profit_2 and profit_3 != neg_inf:
                            temp_schedule.append(Worker(w.get_type(), next_node_3, w.get_rate(), best_3[1] + next_node_3.get_acquire_time() + ts))
                            temp_cost += w.get_rate() * best_3[1]
                            temp_reward += next_node_3.get_reward()
                            visited.append(best_3[0])
            new_schedule = deepcopy(all_schedules[-1])
            new_schedule.append(temp_schedule)
            all_schedules.append(new_schedule)
            reward.append(temp_reward)
            cost.append(temp_cost)
        
        profit = list(map(lambda x: x[0] - x[1], zip(reward, cost)))
        best = profit.index(max(profit))
        
        return all_schedules[best], profit[best], reward[best], cost[best]

    def add_actions(self, worker_list):
        new_worker_list = list()
        
        if worker_list.count(1) < self.required_vertices_1:
            # Hire_1
            temp_1 = deepcopy(worker_list)
            temp_1.append(1)
            temp_1.sort()
            new_worker_list.append(temp_1)

        if worker_list.count(2) < self.required_vertices_2:
            # Hire_2
            temp_2 = deepcopy(worker_list)
            temp_2.append(2)
            temp_2.sort()
            new_worker_list.append(temp_2)

        if worker_list.count(3) < self.required_vertices_3:
            # Hire_3
            temp_3 = deepcopy(worker_list)
            temp_3.append(3)
            temp_3.sort()
            new_worker_list.append(temp_3)
        
        if 3 in worker_list:
            # Replace_1
            temp_4 = deepcopy(worker_list)
            temp_4.remove(3)
            temp_4.append(1)
            temp_4.sort()
            new_worker_list.append(temp_4)
            # Replace_2
            temp_5 = deepcopy(worker_list)
            temp_5.remove(3)
            temp_5.append(2)
            temp_5.sort()
            new_worker_list.append(temp_5)

        return new_worker_list
        
    def process(self):
        # print(self.required_vertices_1, self.required_vertices_2, self.required_vertices_3)
        worker_list = [3]

        history_worker_list = [worker_list]
        history_profit = [self.simulate(worker_list)[1]]

        history = dict()
        history_wl = set()
        history_wl.add(tuple(worker_list))
        history[tuple(worker_list)] = history_profit[0]
        
        while True:
            next = self.add_actions(worker_list)
            if len(next) == 0:
                break
            
            profit = list()
            for l in next:
                if tuple(l) in history_wl:
                    profit.append(history[tuple(l)])
                else:
                    s = self.simulate(l)[1]
                    history_wl.add(tuple(l))
                    history[tuple(l)] = s
                    profit.append(s)
            biggest = profit.index(max(profit))
            if profit[biggest] <= 0:
                break
            else:
                history_worker_list.append(next[biggest])
                history_profit.append(profit[biggest])
                worker_list = next[biggest]

        self.processed = True
        best = history_profit.index(max(history_profit))
        self.PROFIT = history_profit[best]
        self.worker_list = history_worker_list[best]

    def get_profit(self):
        if not self.processed:
            self.process()
        return self.PROFIT

    def get_worker_list(self):
        if not self.processed:
            self.process()
        return self.worker_list
