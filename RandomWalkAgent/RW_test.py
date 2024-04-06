from RW_agent import RandomWalkAgent
from RW_utils import Graph
import sys

if __name__ == '__main__':

    LOG_FILENAME = 'RW_test.log.txt'
    TEST_GRAPH_COUNT = 5

    log_file = open(LOG_FILENAME, "w")
    sys.stdout = log_file

    for graph_number in range(TEST_GRAPH_COUNT):

        graph = Graph(f"../graphs/graph{graph_number + 1}.json")
        rwa = RandomWalkAgent(graph=graph)
        rwa.get_schedule()
        profit = rwa.get_profit()
        worker_list = rwa.get_worker_list()

        print(f'Graph {graph_number + 1} - Profit: {profit}, Worker List: {worker_list}')