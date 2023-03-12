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
import torch

# Create a "current run" which calculates how long the snake is, and how many people are on the board. 
cur_playthrough = []

# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "I AM THE AUTHOR",  # TODO: Your Battlesnake Username
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
    print(game_state)
    agent.batch_rewards[-1] -= 5
    print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:  
    '''
    This is like the step-function in any gym env. The main game loop should theoretically take place in here.
    '''

    # Default code which prevents the snake from going back into itself. 
    move_dict = {0: "left", 1: "right", 2: "down", 3: "up"}
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
    action = agent.make_action(game_state)
    
    # We calculate our rewards in the normal game loop. 
    # We note that we actually have a safeguard for choosing actions that make the snake kill itself. 
    # We can hardcode those rewards without having the snake actually kill itself. 

    if len(agent.batch_states) < 2:
        reward = 0
    else:
        # the following keeps in mind that: 
        # state_vector[-1] = cur_body_length
        # state_vector[-2] = num_snakes_on_board
        prev_state = agent.batch_states[-2]
        prev_state_params = (prev_state[-2].item(), prev_state[-1].item())
        cur_state = agent.batch_states[-1]
        cur_state_params = (cur_state[-2].item(), cur_state[-1].item())
        reward = 0.25 + 2*(cur_state_params[1] - prev_state_params[1]) + 1*(cur_state_params[0] - prev_state_params[0])

    if is_move_safe[action] == False:
        reward -= 5
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

    agent.batch_rewards.append(reward)
    agent.learn()

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": move_dict[next_move]}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server
    agent = PPO(607, 4)

    run_server({"info": info, "start": start, "move": move, "end": end})
