from gym.wrappers import RecordEpisodeStatistics, RecordVideo
import gym
import torch
import os
import json

def record_gym(env_name, agent, video_folder, name_prefix):
    env = gym.make(env_name, render_mode='rgb_array')
    env = RecordVideo(env, video_folder=video_folder, name_prefix=name_prefix, episode_trigger=lambda x: True)
    env = RecordEpisodeStatistics(env)
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.vanilla_act(torch.tensor(state))
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()

def create_experiment(exp_name, spec):
    '''
    Creates folders ready to welcome experiment logs.
    Saves the specification of the experiment as a JSON file.
    Returns the path to log videos, plots and models to.
    '''
    exp_path = os.path.join(spec["log_dir"], exp_name)
    if exp_name in os.listdir(spec["log_dir"]):
        if len(os.listdir(exp_path)) != 0:
            indices = os.listdir(exp_path)
            indices = [int(i) for i in indices]
            exp_index = max(indices) + 1
        else:
            exp_index = 1
    else:
        os.mkdir(exp_path)
        exp_index = 1
    exp_iter_path = os.path.join(exp_path, str(exp_index))
    os.mkdir(exp_iter_path)
    video_path = os.path.join(exp_iter_path, 'videos')
    plot_path = os.path.join(exp_iter_path, 'plots')
    model_path = os.path.join(exp_iter_path, 'models')
    os.mkdir(video_path)
    os.mkdir(plot_path)
    os.mkdir(model_path)
    spec_json = json.dumps(spec, indent=4)
    with open(os.path.join(exp_iter_path, "spec.json"), "w") as f:
        f.write(spec_json)
    return video_path, plot_path, model_path


def save_model(model, save_folder, name_prefix):
    '''
    Saves the state_dict of a model to the specified path.
    The model is saved as a .pt file.
    '''
    torch.save(model.state_dict(), os.path.join(save_folder, name_prefix + '.pt'))


def load_model(agent, model_path):
    '''
    Load a model into an agent's Q-function.
    '''
    agent.Q.load_state_dict(torch.load(model_path))
    return agent

