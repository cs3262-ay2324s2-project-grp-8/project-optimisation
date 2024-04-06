from agent import AgentWorker
from brain import Brain
from environment import Environment
import sys

args = {
    'agent_count': 9,
    'learning_rate': 0.1,
    'memory_model': 'UER',
    'memory_capacity': 10000,
    'target_type': 'DQN',
    'target_frequency': 100,
    'maximum_exploration': 1000,
    'batch_size': 32,
    'gamma': 0.95,
    'number_nodes': 256,
    'optimizer': 'Adam',
    'test': True
}

def get_agent_type(agent_index):
    return 1 if agent_index < 3 else 2 if agent_index < 5 else 3

if __name__ == "__main__":
    
    log = True
    log_file = None
    log_filename = 'test.log.txt'
    
    if log:
        log_file = open(log_filename, "w")
        sys.stdout = log_file
    
    agents = []
    
    """
    w1x, w1y, w2x, w2y, w3x, w3y, w4x, w4y, …….. , w9x, w9y
    ,s1x, s1y, s2x, s2y, ….., s9x, s9y
    , CurrentBudget, CostsIncurredSoFar, RewardsExtractedSoFar
    """
    state_size = 39
    action_size = 12
    
    for idx, agent in enumerate(range(args['agent_count'])):
        
        agent_type = get_agent_type(idx)
        agent_name = f'agent_{idx}_type_{agent_type}'
        agent_learning_rate = 0.1
        timestamp = 0
        
        new_agent = AgentWorker(args, type=agent_type, start=None, rate=agent_learning_rate, timestamp=timestamp, agent_index=idx, state_size=state_size, action_size=action_size, agent_name=agent_name)
        
        agents.append(new_agent)
        
    environment = Environment(agents, isTrain=False)
    environment.train(number_of_graphs=1)
    
    if log:
        log_file.close()
        sys.stdout = sys.__stdout__