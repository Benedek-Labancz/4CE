'''
This script allows a human to play 4CE against
a custom AI model.
'''


from game import _4CE, _2CE
from agent import Agent
from utils import load_yaml, load_json
from log import load_model
import torch
import argparse
import os

X = 1
O = -1
E = 0
M = 3

str_symbols = {
    O: "O",
    X: "X",
    E: "_",
    M: "M"
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--exp_spec', type=str)
    args = parser.parse_args()

    spec = load_json(args.exp_spec)

    game = _2CE()
    agent = Agent(game, spec["num_cells"], spec["epsilon"])
    if args.model_path is not None:
        agent = load_model(agent, args.model_path)

    players_turn = O

    done = False
    while not done:
        #os.system("cls||clear")
        game.print_state(game.state)
        game.print_score(game.score)
        if game.player == players_turn: # Player's turn
            action = input("Player's turn: ").split(' ')
            action = tuple(int(coord) for coord in action)
            new_state, reward, done, info = game.step(action)
            game.state = new_state
            game.switch_turn()
        else:
            fp_state = game.first_person_state(game.state)
            fp_state = game.flatten_state(fp_state)
            action = agent.act(fp_state)
            new_state, reward, done, info = game.step(action)
            game.state = new_state
            game.switch_turn()