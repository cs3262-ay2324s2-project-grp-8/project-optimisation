import numpy as np
import random

from utils import Worker, Node

from uniform_experience_replay import Memory as UER
from brain import Brain

MAX_EPSILON, MIN_EPSILON = 1.0, 0.01
MAX_BETA, MIN_BETA = 0.4, 1.0

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
        self.memory = UER(arguments['memory_capacity'])
        self.target_type = arguments['target_type']
        self.update_target_frequency = arguments['target_frequency']
        self.max_exploration_step = arguments['maximum_exploration']
        self.batch_size = arguments['batch_size']
        self.step = 0
        self.test = arguments['test']
        if self.test:
            self.epsilon = MIN_EPSILON

    def reset_worker_without_model(self, origin: Node):
        self.isExtracting = False
        self.is_Hired = False
        self.move_back_to_origin(origin=origin)

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
                if self.target_type == 'DQN':
                    t[a] = r + self.gamma * np.amax(pTarget_[i])
                else:
                    print('Invalid type for target network!')

            x[i] = s
            y[i] = t
            errors[i] = np.abs(t[a] - old_value)

        return [x, y]
    
    def greedy_move(self, state, graph, ACTIONS, ssp, reward_fn):
        # Returns MOVES by zero-index
        print("===================================")
        print(f'Agent state: {"HIRED" if self.is_Hired else "UNEMPLOYED"}')
        if (self.is_extracting()):
            return EXTRACT - 1
        #print("Agent is Hired Already")
        if (not self.is_Hired):
            print("Agent not Hired")
            return HIRE - 1 if np.random.rand() <= 0.5 else IDLE - 1
        rng = np.random.rand()
        curr_location = self.get_location()
        valid_moves_reward_signal_dict = dict() # store moves in 0-index also
        print("===================================")
        for mv in range(1, 12):
            if ((mv == HIRE or mv == IDLE) and self.is_Hired):
                continue
            selected_move = ACTIONS[mv]
            new_location = (curr_location[0] + selected_move[0], curr_location[1] + selected_move[1])
            print("curr loc : ", curr_location, " new loc: ", new_location)
            # print("get edges : \n", graph.get_edges)
            adj_nodes = graph.get_edges()[curr_location]
            print("adj modes: ", adj_nodes)
            is_valid_move = (new_location in adj_nodes) # or (new_location == curr_location)
            if (is_valid_move):
                print("mv ", mv , " is a valid move from ", curr_location, " to ", new_location)
                valid_moves_reward_signal_dict[mv - 1] = 0 # back to 0 index
        print("valid moves: ", valid_moves_reward_signal_dict.keys())
        assert(len(valid_moves_reward_signal_dict.keys()) > 0)
        if rng <= self.epsilon:
            move = random.randrange(self.action_size)
            while (move not in valid_moves_reward_signal_dict.keys()):
                move = random.randrange(self.action_size)
            return move
        elif self.epsilon < rng <= 2 * self.epsilon:
            # greedy approach
            # iterate through all valid moves and estimate reward
            best_move, highest_reward = None, -np.inf
            for vm in valid_moves_reward_signal_dict.keys(): # 0-idex
                selected_move = ACTIONS[vm + 1]
                new_location = (curr_location[0] + selected_move[0], curr_location[1] + selected_move[1])
                valid_moves_reward_signal_dict[vm] = reward_fn(graph, curr_location, new_location, ssp, self.type, state[CURRENT_BUDGET])
                if (valid_moves_reward_signal_dict[vm] > highest_reward):
                    highest_reward = valid_moves_reward_signal_dict[vm]
                    best_move = vm
            return best_move
        else:
            probabilities = self.brain.predict_one_sample(state)
            #print("Probabilities from Agent's brain: ",probabilities)
            move = np.argmax(probabilities)
            # check if move is within the valid moves, 
            # if invalid, then 0 the prob, then get the next highest
            while (int(move) not in valid_moves_reward_signal_dict.keys()):
                probabilities[move] = -np.inf
                move = np.argmax(probabilities)
                #print("Probabilities from Agent's brain: ",probabilities)
            return move


    
    def observe(self, sample):
        self.memory.remember(sample)
    
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
        batch = self.memory.sample(self.batch_size)
        x, y = self.find_targets_uer(batch)
        self.brain.train_model(x, y)
        
    def update_target_model(self):
        if self.step % self.update_target_frequency == 0:
            self.brain.update_target_model()
    
    def move_back_to_origin(self, origin: Node):
        self.move(origin)