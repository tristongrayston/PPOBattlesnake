# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import random
import typing
from PPO import PPO
import numpy as np

# Create a "current run" which calculates how long the snake is, and how many people are on the board. 
cur_playthrough = []

# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "",  # TODO: Your Battlesnake Username
        "color": "#888888",  # TODO: Choose color
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    cur_playthrough = []
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    _, body, observation_vector = observation_to_values(game_state)
    if body == 0:
        reward = -100
    else:
        reward = 100
        
    agent.next_state_append(reward, observation_vector, 1)
    print("GAME OVER\n")

def observation_to_values(observation):
    # Nathan's modified observation helper function
    #init
    board = observation['board']
    health = 100
    n_channels = 5
    state_matrix = np.zeros((n_channels, board["height"], board["width"]))
    #fill
    for _snake in board['snakes']:
        health = np.array(_snake['health'])
        #place head on channel 0
        state_matrix[0, _snake['head']['x'], _snake['head']['y']] = 1
        #place tail on channel 1
        state_matrix[1, _snake['body'][-1]['x'], _snake['body'][-1]['y']] = 1
        #place body on channel 2
        for _body_segment in _snake['body']:
            state_matrix[2, _body_segment['x'], _body_segment['y']] = 1
        
        # Calculate variables nessessary for reward.
        num_snakes = np.count_nonzero(state_matrix[0])
        body_length = np.count_nonzero(state_matrix[2])

    #place food on channel 3
    for _food in board["food"]:
        state_matrix[3,_food['x'], _food['y']] = 1
    #create health channel
    state_matrix[4] = np.full((board["height"], board["width"]), health)
    #flatten
    # state_matrix = state_matrix.reshape(-1,1) # don't flatten if using conv layer

    state_matrix = np.concatenate([state_matrix, health.reshape(1,1)], axis=0)
    
    return num_snakes, body_length, state_matrix # .flatten() #dont flatten if using conv2d layers


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
reward = 1
def move(game_state: typing.Dict) -> typing.Dict:  
    '''
    This is like the step-function in any gym env. The main game loop should theoretically take place in here.
    '''
    num_snakes, body_length, observation_vector = observation_to_values(game_state)

    # Check if we store into next_states or not 
    if len(cur_playthrough) != 0:
        # cur reward
        # Turn our observation state into something
        prev_move = cur_playthrough[-1]
        if prev_move[0] < num_snakes:
            reward += 20
        if prev_move[1] < body_length:
            reward += 15
        cur_playthrough.append((num_snakes, body_length))
        agent.next_state_append(reward, observation_vector, 0)

    reward = 1
    # Default code which prevents the snake from going back into itself. 
    # 0 = left
    # 1 = right
    # 2 = down
    # 3 = up
    is_move_safe = {0: True, 1: True, 2: True, 3: True}

    my_head = game_state["you"]["body"][0]  # Coordinates of your head
    my_neck = game_state["you"]["body"][1]  # Coordinates of your "neck"

    if my_neck["x"] < my_head["x"]:  # Neck is left of head, don't move left
        is_move_safe[0] = False

    elif my_neck["x"] > my_head["x"]:  # Neck is right of head, don't move right
        is_move_safe[1] = False

    elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
        is_move_safe[2] = False

    elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
        is_move_safe[3] = False

    # we now get our action. 
    action = agent.make_action(observation_vector)
    
    # We calculate our rewards in the normal game loop. 
    # We note that we actually have a safeguard for choosing actions that make the snake kill itself. 
    # We can hardcode those rewards without having the snake actually kill itself. 

    if is_move_safe[action] == False:
        reward -= 100
        # If the move we return is wrong, we'll have it output a random safe move. 
        # Are there any safe moves left?
        safe_moves = []
        for move, isSafe in is_move_safe.items():
            if isSafe:
                safe_moves.append(move)

        if len(safe_moves) == 0:
            print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
            return {"move": "down"}

        # Choose a random move from the safe ones
        next_move = random.choice(safe_moves)
    next_move = action

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server
    agent = PPO(512, 4)

    run_server({"info": info, "start": start, "move": move, "end": end})
