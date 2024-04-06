import json

DEBUG = False
DEBUG_TIME_TAKEN = False
DEBUG_PROFIT_ONLY = False
LOG_FULL = True

BASIC_SITE = 0
ORIGN      = 1
SITE1      = 2
SITE2      = 3
SITE3      = 4

TYPE_ID_TO_STRING_TYPE: dict[int, str] = {
    0 : "BASIC SITE",
    1 : "ORIGIN",
    2 : "REWARD TYPE 1",
    3 : "REWARD TYPE 2",
    4 : "REWARD TYPE 3"
}
class Node:

    def __init__(self, node_info, coordinate):
        self.type = node_info["vertex_type"]
        self.reward = node_info["reward"]
        self.current_reward = node_info["reward"]
        self.active_period = node_info["mult_time_active"]
        self.acquire_time = node_info["time_to_acquire"]
        self.coordinate = coordinate
        self.accessed = False
        self.current_accessors = []
        self.sighted = False
        self.extracted = False
        self.extracted_before = False
        self.extractor : Node | None = None
    
    def __repr__(self) -> str:
        return f'({self.get_x_coordinate(), self.get_y_coordinate()} TYPE : {TYPE_ID_TO_STRING_TYPE[self.type]})'

    def get_type(self):
        '''
        Basic               := 0
        Origin              := 1
        Site1, Site2, Site3 := 2, 3, 4
        '''
        return self.type

    def get_coordinate(self):
        return self.coordinate

    def get_active_period(self) -> list :
        return self.active_period
    
    def someone_eyeing_node(self):
        return self.sighted
    
    def sight(self):
        self.sighted = True
    
    def get_acquire_time(self):
        return self.acquire_time
    
    def enter(self, agent_worker):
        self.current_accessors.append(agent_worker)

    def leave(self, agent_worker):
        self.current_accessors.remove(agent_worker)

    def can_extract(self):
        return not self.extracted_before and not self.extracted and not self.sighted

    def extract(self, agent_worker):
        self.enter(agent_worker=agent_worker)
        self.extractor = agent_worker
        self.extracted = True

    def has_extractor(self):
        return self.extractor is not None
    
    def leave_node_extractor(self):
        self.leave(self.extractor)
        self.extractor = None
        
    def reset_node(self):
        self.current_accessors.clear()
        self.extractor = None
        self.current_reward = self.reward
        self.extracted = False
        self.accessed = False
        self.sighted = False

    def access(self):
        self.accessed = True

    def is_accessed_before(self):
        return self.accessed

    def get_coordinate(self):
        return self.coordinate
    
    def get_x_coordinate(self):
        return self.coordinate[0]
    
    def get_y_coordinate(self):
        return self.coordinate[1]
    
    def get_reward(self):
        return self.reward
    
    def zero_reward(self):
        self.current_reward = 0
        self.accessed = True


class Graph:

    def __init__(self, json_filename):
        self.data = json.load(open(json_filename))
        self.vertices = dict() # key (x1, y1) -> value : Node
        self.edges = dict() # key (x1, y1) -> value : list of int tuples
        self.workers_cost_rate: list[float] = [100.0, 200.0, 500.0]
        self.site_type_rewards: list[float] = [0, 0, 0, 0, 0]
        self.sites_info = dict() # key : site type -> value: list of Node objects
        self.process()
    
    def __str__(self) -> str:
        stuff =  {
            "vertices": self.vertices,
            "edges" : self.edges
        }
        string_construct = "Origin : " + str(self.get_Origin()) + "\n"
        for e, v in stuff["edges"].items():
            string_construct += str(e) + str(v) + "\n"
        string_construct += "Type 1 Reward Sites: \n"
        for k in self.retrieve_all_sites_of_type(SITE1):
            string_construct += str(k) + "\n"
        string_construct += "\n"
        string_construct += "Type 2 Reward Sites: \n"
        for k in self.retrieve_all_sites_of_type(SITE2):
            string_construct += str(k) + "\n"
        string_construct += "\n"
        string_construct += "Type 3 Reward Sites: \n"
        for k in self.retrieve_all_sites_of_type(SITE3):
            string_construct += str(k) + "\n"
        string_construct += "\n"
        return string_construct

    def process(self):

        # Process the vertices
        vertices_data = self.data["vertices"]
        for x_coordinate in vertices_data:
            for y_coordinate in vertices_data[x_coordinate]:
                x_coordinate, y_coordinate = int(x_coordinate), int(y_coordinate)
                node = Node(vertices_data[str(x_coordinate)][str(y_coordinate)], (x_coordinate, y_coordinate))
                self.vertices[(x_coordinate, y_coordinate)] = node
                if (node.get_type() > 1):
                    self.site_type_rewards[node.get_type() - 1] = node.get_reward()
                if (node.get_type() not in self.sites_info.keys()):
                    self.sites_info[node.get_type()] = [node]
                else:
                    self.sites_info[node.get_type()].append(node)

        # Process the edges
        edges_data = self.data["edges"]
        for edge in edges_data:
            x1, y1, x2, y2 = edge
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if ((x1, y1) not in self.edges.keys()):
                self.edges[(x1, y1)] = [(x2, y2)]
            else:
                self.edges[(x1, y1)].append((x2, y2))
            if ((x2, y2) not in self.edges.keys()):
                self.edges[(x2, y2)] = [(x1, y1)]
            else:
                self.edges[(x2, y2)].append((x1, y1))
            
        print("Graph Initialised Successfully!")

    def get_edges(self):
        return self.edges

    def get_vertices(self):
        return self.vertices
    
    def get_Node(self, x : int, y: int):
        return self.vertices[(x, y)]
    
    def get_adjacent_nodes_by_coordinates(self, x : int, y : int):
        node: Node = self.vertices[(x, y)]
        return self.get_adjacent_nodes_by_Node(node)

    def get_adjacent_nodes_by_Node(self, node: Node):
        neighbours = self.edges[node.get_coordinate()]
        return [self.vertices[neighbour] for neighbour in neighbours]
    
    def get_site_reward(self, type: int):
        return self.site_type_rewards[type]
    
    def retrieve_all_sites_of_type(self, type : int):
        return self.sites_info[type]
    
    def get_worker_cost(self, type : int):
        return self.workers_cost_rate[type - 1]
    
    def get_Origin(self):
        return self.retrieve_all_sites_of_type(ORIGN)[0]
    
    def is_reward_site_by_node(self, node : Node):
        return node.get_type() > 0
    
    def is_reward_site(self, x : int, y : int):
        return self.is_reward_site_by_node(self.get_Node(x, y))
    
class Worker:

    def __init__(self, type : int, start, rate, timestamp):
        self.type = type
        self.location = start
        self.rate = 500 if type == 3 else type * 100
        self.ts = timestamp
        self.is_Hired = False
        self.isExtracting = False
        self.fired = False
        self.waitTime = 0

    def get_type(self):
        return self.type

    def get_location(self):
        return self.location
    
    def get_rate(self):
        return self.rate
    
    def move(self, next_node: Node):
        self.location = next_node.get_coordinate()

    def move_to_coordinates(self, x, y):
        self.location = (x, y)

    def get_timestamp(self):
        return self.ts
    
    def hire(self):
        self.is_Hired = True

    def fire(self):
        self.is_Hired = False
        self.fired = True
        
    def fire(self):
        self.is_Hired = False

    def extract(self, node : Node):
        self.isExtracting = True
        self.waitTime = node.get_type() - 1
        node.extract(self)

    def done_extracting(self):
        self.isExtracting = False

    def is_extracting(self):
        return self.isExtracting

    def decrease_waitTime(self):
        self.waitTime -= 1

    def get_waitTime(self):
        return self.waitTime
        
    def isHired(self):
        return self.is_Hired

    def isFiredBefore(self):
        return self.fired
    
    def reward_at_location(self, graph, zero_out):
        node = graph.get_Node(self.location[0], self.location[1])
        reward = node.get_reward()
        if (zero_out):
            node.zero_reward()
        return reward


# if __name__ == "__main__" :
#     graph = Graph("Test_Graph_Slightly_Off_the_Beaten_Path/Test_Graph_Slightly_Off_the_Beaten_Path.json")
#     # print(graph.get_vertices())
#     ORIGIN = 1
#     print(graph.get_adjacent_nodes_by_coordinates(5,5))
#     print(graph.get_edges())
#     print(graph.retrieve_all_sites_of_type(ORIGIN))
#     print(len(graph.get_vertices()))
#     print(len(graph.get_edges()))