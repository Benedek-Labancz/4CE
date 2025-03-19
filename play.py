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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--exp_spec', type=str)
    parser.add_argument('--first_player', type=bool)
    args = parser.parse_args()

    spec = load_json(args.exp_spec)

    game = _2CE()
    agent = Agent(game, spec["num_cells"], spec["epsilon"])
    if args.model_path is not None:
        agent = load_model(agent, args.model_path)

    if args.first_player:
        players_turn = game.X
    else:
        players_turn = game.O

    done = False
    game.print_state(game.state)
    while not done:
        if game.player == players_turn: # Player's turn
            action = input("Player's turn: ").split(' ')
            action = tuple(int(coord) for coord in action)
            new_state, reward, done, info = game.step(action)
        else:
            fp_state = game.fp(game.state)
            fp_state = game.flat(fp_state)
            action = agent.act(fp_state)
            new_state, reward, done, info = game.step(action)
        os.system("cls||clear")
        game.print_state(game.state)
        game.print_score(game.score)
    winner = game.determine_winner()
    print(f"The winner is {winner}.")
    print(f"Final score: ")
    game.print_score(game.score)