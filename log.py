from gym.wrappers import RecordEpisodeStatistics, RecordVideo
import gym
import torch
import os
import json
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

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

def create_experiment(exp_name, info, spec):
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
    with open(os.path.join(exp_iter_path, "info.txt"), "w") as f:
        f.write(info)
    return video_path, plot_path, model_path

def make_plot(values, title, x_label, y_label, ylim=None, path=None):
    values = np.array(values)
    plt.plot(values[:, 0], values[:, 1])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(path)
    plt.close()

def make_scatter(values, title, x_label, y_label, path):
    values = np.array(values)
    plt.scatter(values[:, 0], values[:, 1])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(path)
    plt.close()

def make_multiplot(values, title, x_label, y_label, path):
    plt.plot([log[0] for log in values], [log[1] for log in values])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(path)
    plt.close()

def make_histogram(values, title, x_label, y_label, path):
    plt.hist(values)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(path)
    plt.close()

class Log:
    def __init__(self, key, title, x_label=None, y_label=None, ylim=None, path=None, plot_func=lambda: False):
        self.key = key
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.ylim = ylim
        self.path = path
        self.plot_func = plot_func
        self.data = []

    def reset(self):
        self.data = []

class Logger:
    def __init__(self, log_path):
        self.logs = dict()
        self.log_path = log_path
        sns.set_theme()

    def add_log(self, key, title, x_label=None, y_label=None, ylim=None, path='default.png', plot_func=lambda: False):
        if path is None:
            path = self.log_path
        self.logs[key] = Log(key, title, x_label, y_label, ylim, os.path.join(self.log_path, path), plot_func)

    def log(self, key, values):
        self.logs[key].data.append(values)

    def log_t(self, key, t, values):
        self.logs[key].data.append([t, values])

    def clear_log(self, key):
        self.logs[key].reset()

    def plot_all(self):
        for key in self.logs:
            self.logs[key].plot_func(key)
    
    def make_line_plot(self, key):
        log = self.logs[key]
        make_plot(log.data, log.title, log.x_label, log.y_label, log.ylim, log.path)

    def make_scatter(self, key):
        log = self.logs[key]
        make_scatter(log.data, log.title, log.x_label, log.y_label, log.path)

    def make_multiline_plot(self, key):
        log = self.logs[key]
        make_multiplot(log.data, log.title, log.x_label, log.y_label, log.path)

    def make_histogram(self, key):
        log = self.logs[key]
        make_histogram(log.data, log.title, log.x_label, log.y_label, log.path)