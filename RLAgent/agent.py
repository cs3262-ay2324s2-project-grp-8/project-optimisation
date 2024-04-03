import numpy as np
import random

from utils import Worker, Node

from prioritized_experience_replay import Memory as PER
from uniform_experience_replay import Memory as UER
from brain import Brain

MAX_EPSILON, MIN_EPSILON = 1.0, 0.01
MAX_BETA, MIN_BETA = 0.4, 1.0

class AgentWorker(Worker):
    
    epsilon = MAX_EPSILON
    beta = MAX_BETA
    
    def __init__(self, arguments, type, start, rate, timestamp, agent_index, state_size, action_size, agent_name):
        super().__init__(type=type, start=start, rate=rate, timestamp=timestamp)

        self.agent_index = agent_index
        self.state_size = state_size
        self.action_size = action_size
        self.brain = Brain(state_size, action_size, agent_name, arguments)
        self.gamma = arguments['gamma']
        self.learning_rate = arguments['learning_rate']
        self.memory_model = arguments['memory_model']
        
        if self.memory_model == 'PER':
            self.memory = PER(arguments['memory_capacity'], arguments['pr_scale'])
        elif self.memory_model == 'UER':
            self.memory = UER(arguments['memory_capacity'])
        
        self.target_type = arguments['target_type']
        self.update_target_frequency = arguments['target_frequency']
        self.max_exploration_step = arguments['maximum_exploration']
        self.batch_size = arguments['batch_size']
        self.step = 0
        self.test = arguments['test']
        if self.test:
            self.epsilon = MIN_EPSILON
    
    def find_targets_per(self, batch):
        batch_len = len(batch)

        states = np.array([o[1][0] for o in batch])
        states_ = np.array([o[1][3] for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)
        pTarget_ = self.brain.predict(states_, target=True)

        x = np.zeros((batch_len, self.state_size))
        y = np.zeros((batch_len, self.action_size))
        errors = np.zeros(batch_len)

        for i in range(batch_len):
            o = batch[i][1]
            s = o[0]
            a = o[1][self.agent_index]
            r = o[2]
            s_ = o[3]
            done = o[4]

            t = p[i]
            old_value = t[a]
            if done:
                t[a] = r
            else:
                if self.target_type == 'DDQN':
                    t[a] = r + self.gamma * pTarget_[i][np.argmax(p_[i])]
                elif self.target_type == 'DQN':
                    t[a] = r + self.gamma * np.amax(pTarget_[i])
                else:
                    print('Invalid type for target network!')

            x[i] = s
            y[i] = t
            errors[i] = np.abs(t[a] - old_value)

        return [x, y, errors]

    def find_targets_uer(self, batch):
        batch_len = len(batch)

        states = np.array([o[0] for o in batch])
        states_ = np.array([o[3] for o in batch])
        
        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)
        pTarget_ = self.brain.predict(states_, target=True)

        x = np.zeros((batch_len, self.state_size))
        y = np.zeros((batch_len, self.action_size))
        errors = np.zeros(batch_len)

        for i in range(batch_len):
            o = batch[i]
            s = o[0]
            a = o[1][self.agent_index]
            r = o[2]
            s_ = o[3]
            done = o[4]

            t = p[i]
            old_value = t[a]
            if done:
                t[a] = r
            else:
                if self.target_type == 'DDQN':
                    t[a] = r + self.gamma * pTarget_[i][np.argmax(p_[i])]
                elif self.target_type == 'DQN':
                    t[a] = r + self.gamma * np.amax(pTarget_[i])
                else:
                    print('Invalid type for target network!')

            x[i] = s
            y[i] = t
            errors[i] = np.abs(t[a] - old_value)

        return [x, y]
    
    def greedy_move(self, state, graph, ACTIONS):
        if np.random.rand() <= self.epsilon:
            move = random.randrange(self.action_size)
            
            curr_location = self.get_location()
            selected_move = ACTIONS[move + 1]
            
            # print(f'Curr Location: {curr_location}, Move: {selected_move}, Move Code: {move + 1}')
            
            new_location = (curr_location[0] + selected_move[0], curr_location[1] + selected_move[1])
            
            adj_nodes = [coordinate.get_coordinate() for coordinate in graph.get_adjacent_nodes_by_coordinates(curr_location[0], curr_location[1])]
            
            is_valid_move = new_location in adj_nodes
            # print(f'Agent: {self.agent_index+1}, Current Location: {curr_location}, New Location: {new_location}, Adjacent Nodes: {adj_nodes}, Is Valid Move: {is_valid_move}')
            
            if (is_valid_move):
                return move
            else:
                return 11
        
        exploration = np.argmax(self.brain.predict_one_sample(state))
        print(exploration)
        return exploration

    
    def observe(self, sample):
        if self.memory_model == 'UER':
            self.memory.remember(sample)
        elif self.memory_model == 'PER':
            _, _, errors = self.find_targets_per([[0, sample]])
            self.memory.remember(sample, errors[0])
    
    def decay_epsilon(self):
        self.step += 1

        if self.test:
            self.epsilon = MIN_EPSILON
            self.beta = MAX_BETA
        else:
            if self.step < self.max_exploration_step:
                self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * (self.max_exploration_step - self.step)/self.max_exploration_step
                self.beta = MAX_BETA + (MIN_BETA - MAX_BETA) * (self.max_exploration_step - self.step)/self.max_exploration_step
            else:
                self.epsilon = MIN_EPSILON
    
    def replay(self):
        if self.memory_model == 'UER':
            batch = self.memory.sample(self.batch_size)
            x, y = self.find_targets_uer(batch)
            self.brain.train_model(x, y)
        
        elif self.memory_model == 'PER':
            [batch, batch_indices, batch_priorities] = self.memory.sample(self.batch_size)
            x, y, errors = self.find_targets_per(batch)

            normalized_batch_priorities = [float(i) / sum(batch_priorities) for i in batch_priorities]
            importance_sampling_weights = [(self.batch_size * i) ** (-1 * self.beta)
                                           for i in normalized_batch_priorities]
            normalized_importance_sampling_weights = [float(i) / max(importance_sampling_weights)
                                                      for i in importance_sampling_weights]
            sample_weights = [errors[i] * normalized_importance_sampling_weights[i] for i in range(len(errors))]

            self.brain.train_model(x, y, np.array(sample_weights))

            self.memory.update(batch_indices, errors)

    def update_target_model(self):
        if self.step % self.update_target_frequency == 0:
            self.brain.update_target_model()
    
    def move_back_to_origin(self, origin: Node):
        self.move(origin)