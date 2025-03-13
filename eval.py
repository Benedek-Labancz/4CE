from game import Game
from agent import Agent
from collections import defaultdict
from tqdm import tqdm

def self_play_eval(agent, num_episodes):
    '''
    Evaluates agent performance, i.e. average undiscounted return
    over a number of episodes.
    '''
    game = Game()

    total_reward_log = []
    pbar = tqdm(total=num_episodes)
    for _ in range(num_episodes):
        total_rewards = defaultdict(lambda: 0)
        game.reset()
        done = False
        while not done:
            fp_state = game.first_person_state(game.state)
            action = agent.act(fp_state)
            new_state, reward, done, info = game.step(action)
            total_rewards[game.player] += reward
            game.state = new_state
            game.switch_turn()
        for key in total_rewards:
            total_reward_log.append(total_rewards[key])
        pbar.update()
    return sum(total_reward_log) / (2 * num_episodes)