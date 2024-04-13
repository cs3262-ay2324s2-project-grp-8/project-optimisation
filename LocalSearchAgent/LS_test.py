from LocalSearchAgent.agent_LS import LocalSearchAgent
from LocalSearchAgent.utils_LS import Graph
import sys

if __name__ == '__main__':
    
    LOG_FILENAME = 'LS_test.log.txt'
    TEST_GRAPH_COUNT = 5
    
    log_file = open(LOG_FILENAME, "w")
    sys.stdout = log_file
    
    for graph_number in range(TEST_GRAPH_COUNT):
        
        graph = Graph(f"../graphs/graph{graph_number + 1}.json")
        lsa = LocalSearchAgent(graph=graph)
        lsa.process()
        profit = lsa.get_profit()
        worker_list = lsa.get_worker_list()
        
        print(f'Graph {graph_number + 1} - Profit: {profit}, Worker List: {worker_list}')