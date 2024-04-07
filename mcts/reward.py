from utils import Graph, Node
import numpy as np

# WORKER Accessors:
X = 0
Y = 1
TYPE = 2
IS_HIRED = 3
IS_FIRED_BEFORE = 4
IS_EXTRACTING = 5
EXTRACT_TIME_LEFT = 6
TIMESTAMP = 7

# SITE Accessors
IS_SIGHTED = 3
ACCESSED = 4

# Actions
MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_EAST = 2
MOVE_WEST = 3
MOVE_NORTH_EAST = 4
MOVE_NORTH_WEST = 5
MOVE_SOUTH_EAST = 6
MOVE_SOUTH_WEST = 7

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

def calculate_reward_and_best_action(state, curr_loc, next_loc, ssp, agent_type):
    #print(f"Calculating Reward for : {curr_loc} -> {next_loc}") if DEBUG else None
    max_calculated_reward = -np.inf
    best_move = None
    # moves_profit_signal = []
    for reward_idx in range(12, 12 + 9):
        rwd_site = state[reward_idx]
        if (rwd_site[ACCESSED]):
            continue
        if (rwd_site[TYPE] > agent_type):
            break
        reward_signal = (5000 if rwd_site[TYPE] == 3 else 1000 * rwd_site)
        cost_signal = (500 if agent_type == 3 else 100 * agent_type) * (rwd_site[TYPE] + ssp[next_loc][(rwd_site[X], rwd_site[Y])])
        profit_signal = reward_signal - cost_signal
        if (profit_signal >= state[11]):
            continue # profit signal exceed current budget
        if (profit_signal > max_calculated_reward):
            mv = DELTA_TO_ACTIONS[(next_loc[0] - curr_loc[0], next_loc[1] - curr_loc[1])]
            # moves_profit_signal.append((mv, profit_signal))
            best_move = mv
            max_calculated_reward = max(max_calculated_reward, profit_signal)
        #sorted(moves_profit_signal, reverse=True, key=lambda x : x[1])
    return max_calculated_reward, best_move

            
