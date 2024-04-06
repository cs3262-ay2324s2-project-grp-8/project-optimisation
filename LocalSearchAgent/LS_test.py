from LS_agent import LocalSearchAgent
from LS_utils import Graph

if __name__ == '__main__':
    
    TEST_GRAPH_COUNT = 5
    
    for graph_number in range(TEST_GRAPH_COUNT):
        
        graph = Graph(f"../graphs/graph{graph_number + 1}.json")
        lsa = LocalSearchAgent(graph=graph)
        lsa.process()
        profit = lsa.get_profit()
        worker_list = lsa.get_worker_list()
        
        print(f'Graph {graph_number + 1} - Profit: {profit}, Worker List: {worker_list}')