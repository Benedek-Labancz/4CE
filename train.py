import os
import argparse
from utils import load_yaml, make_plot
from game import Game, print_state, print_score
from agent import Agent, ReplayBuffer
from log import record_gym, create_experiment, save_model
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from tqdm import tqdm
import gym

def train(game, agent, spec):
    
    buffer = ReplayBuffer(max_size=spec["max_buffer_size"])
    optimizer = Adam(params=agent.Q.parameters(), lr=spec["lr"])

    t = 0
    T = spec["max_timesteps"]
    while t < T:
        game.reset()
        done = False
        while not done:
            action = agent.explore_act(game.state)
            new_state, reward, done, info = game.step(action)
            # TODO: continue here, you want to collect experience, i.e. first-personify current state and extend replay buffer
            # also set up training arguments in the experiment specification


def vanilla_train_refined(game, agent, spec):
    
    buffer = ReplayBuffer(max_size=spec["max_buffer_size"])
    optimizer = Adam(params=agent.Q.parameters(), lr=spec["lr"])
    loss_module = MSELoss()

    t = 0
    T = spec["max_timesteps"]
    episode = 0
    pbar = tqdm(total=T)
    
    # Initialize all logs; they will be in the form [[t1, v1], ..., [tn, vn]] 
    reward_log = []
    average_reward_log = []
    best_average_reqard = 0 # to track the best model performance
    epsilon_log = []
    loss_log = []
    q_log = []

    # Open a new experiment run
    video_path, plot_path, model_path = create_experiment(args.exp_name, spec)

    # Run initial random steps to collect data before training
    t_init = 0
    while t_init < spec["init_random_steps"]:
        state, _ = game.reset()
        done = False
        while not done:
            action = agent.vanilla_explore_act(torch.tensor(state, dtype=torch.float32))
            new_state, reward, terminated, truncated, info = game.step(action)
            done = terminated or truncated
            buffer.extend(state, action, reward, new_state, done)
            state = new_state
            t_init += 1
            if t_init >= spec["init_random_steps"]:
                break

    # Main training loop
    while t < T:
        total_reward = 0
        state, _ = game.reset()
        done = False
        while not done:
            # Collect data
            action = agent.vanilla_explore_act(torch.tensor(state, dtype=torch.float32))
            new_state, reward, terminated, truncated, info = game.step(action)
            done = terminated or truncated
            total_reward += reward # logging purposes
            # Save experience
            buffer.extend(state, action, reward, new_state, done)
            state = new_state

            # Decay epsilon
            agent.epsilon = max(agent.epsilon * spec["epsilon_decay"], spec["min_epsilon"])
            epsilon_log.append([t, agent.epsilon])

            # Sync target network weights
            if t % spec["sync_frequency"] == 0:
                agent.sync_target()
                # we could mark syncing points on a chart

            # Record current agent performance
            if t % spec["record_frequency"] == 0:
                record_gym(env_name=spec["env_name"], agent=agent, video_folder=video_path, name_prefix=spec["name_prefix"]+ '_'+ str(t))

            # Heart of DQN; Experience Replay
            if t % spec["optim_frequency"] == 0:
                avg_loss_values = []
                avg_q_values = []
                for _ in range(spec["optim_iter"]): # We optimize on several batches before collecting data again
                    # Sample our batch from the replay buffer
                    samples = buffer.sample(spec["batch_size"])
                    samples = buffer.batch_samples(samples)

                    # Compute the target according to the DQN equation
                    target = torch.tensor(samples["reward"]) + torch.tensor((1 - samples["done"])) * spec["gamma"] * torch.max(agent.TargetNet(torch.tensor(samples["new_state"])), axis=1)[0]
                    
                    # Estimate Q-values with the current Q-network
                    current_estimate = agent.Q(torch.tensor(samples["state"]))
                    current_estimate = torch.gather(current_estimate, dim=1, index=torch.tensor(samples['action'], dtype=torch.int64).reshape(-1, 1))
                    current_estimate = current_estimate.type(dtype=torch.float64)
                    current_estimate = current_estimate.squeeze()
                    avg_q_values.append(current_estimate.detach().mean())

                    # Compute loss (target - estimate)^2
                    loss_values = loss_module(current_estimate, target)
                    avg_loss_values.append(loss_values.detach().mean())

                    # Backprop and optimize
                    loss_values.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                loss_log.append([t, np.mean(avg_loss_values)])
                q_log.append([t, np.mean(avg_q_values)])

            pbar.update()
            t += 1

        episode += 1

        # Logging
        reward_log.append([t, total_reward])
        average_reward = np.mean([reward_log[-i][1] for i in range(1, min(10, len(reward_log)))])
        average_reward_log.append([t, average_reward])

        # Save best model
        if average_reward > best_average_reqard:
            name_prefix = f'best_t{t}_r{round(average_reward)}'
            save_model(agent.Q, model_path, name_prefix)
            best_average_reqard = average_reward
            # Record the performance of the best model
            record_gym(spec["env_name"], agent, video_path, name_prefix)

        if episode % spec["plot_frequency"] == 0:
            make_plot(reward_log, "Cartpole", "Timestep", "Reward per episode", os.path.join(plot_path, 'cartpole_rwpe.png'))
            make_plot(average_reward_log, "Cartpole", "Timestep", "Average reward in last 10 episodes", os.path.join(plot_path, 'cartpole_mrw.png'))
            make_plot(epsilon_log, "Cartpole", "Timestep", "epsilon", os.path.join(plot_path, 'cartpole_epsilon.png'))
            make_plot(loss_log, "Cartpole", "Timestep", "Mean loss", os.path.join(plot_path, 'cartpole_loss.png'))
            make_plot(q_log, "Cartpole", "Timestep", "Mean Q-value", os.path.join(plot_path, 'cartpole_q.png'))
    
    return agent

def vanilla_train(game, agent, spec):
    
    buffer = ReplayBuffer(max_size=spec["max_buffer_size"])
    optimizer = Adam(params=agent.Q.parameters(), lr=spec["lr"])
    loss_module = MSELoss()

    t = 0
    T = spec["max_timesteps"]
    episode = 0
    pbar = tqdm(total=T)
    
    # Initialize all logs; they will be in the form [[t1, v1], ..., [tn, vn]] 
    reward_log = []
    average_reward_log = []
    best_average_reqard = 0 # to track the best model performance
    epsilon_log = []
    loss_log = []
    q_log = []

    # Open a new experiment run
    video_path, plot_path, model_path = create_experiment(args.exp_name, spec)

    # Run initial random steps to collect data before training
    t_init = 0
    while t_init < spec["init_random_steps"]:
        state, _ = game.reset()
        done = False
        while not done:
            action = agent.vanilla_explore_act(torch.tensor(state, dtype=torch.float32))
            new_state, reward, terminated, truncated, info = game.step(action)
            done = terminated or truncated
            buffer.extend(state, action, reward, new_state, done)
            state = new_state
            t_init += 1
            if t_init >= spec["init_random_steps"]:
                break

    # Main training loop
    while t < T:
        total_reward = 0
        state, _ = game.reset()
        done = False
        while not done:
            # Collect data
            for _ in range(spec["num_collection_steps"]):
                action = agent.vanilla_explore_act(torch.tensor(state, dtype=torch.float32))
                new_state, reward, terminated, truncated, info = game.step(action)
                done = terminated or truncated
                total_reward += reward # logging purposes
                # Save experience
                buffer.extend(state, action, reward, new_state, done)
                state = new_state

                # Decay epsilon
                agent.epsilon = max(agent.epsilon * spec["epsilon_decay"], spec["min_epsilon"])
                epsilon_log.append([t, agent.epsilon])

                # Sync target network weights
                if t % spec["sync_frequency"] == 0:
                    agent.sync_target()
                    # we could mark syncing points on a chart

                # Record current agent performance
                if t % spec["record_frequency"] == 0:
                    record_gym(env_name=spec["env_name"], agent=agent, video_folder=video_path, name_prefix=spec["name_prefix"]+ '_'+ str(t))

                pbar.update()
                t += 1

                # Break to avoid stepping a terminated environment
                # TODO: note that this algorithm leads to inconsistent number of collection steps
                # Refine the loops to solve the issue
                if done:
                    break

            # Heart of DQN; Experience Replay
            avg_loss_values = []
            avg_q_values = []
            for _ in range(spec["optim_iter"]): # We optimize on several batches before collecting data again
                # Sample our batch from the replay buffer
                samples = buffer.sample(spec["batch_size"])
                samples = buffer.batch_samples(samples)

                # Compute the target according to the DQN equation
                target = torch.tensor(samples["reward"]) + torch.tensor((1 - samples["done"])) * spec["gamma"] * torch.max(agent.TargetNet(torch.tensor(samples["new_state"])), axis=1)[0]
                
                # Estimate Q-values with the current Q-network
                current_estimate = agent.Q(torch.tensor(samples["state"]))
                current_estimate = torch.gather(current_estimate, dim=1, index=torch.tensor(samples['action'], dtype=torch.int64).reshape(-1, 1))
                current_estimate = current_estimate.type(dtype=torch.float64)
                current_estimate = current_estimate.squeeze()
                avg_q_values.append(current_estimate.detach().mean())

                # Compute loss (target - estimate)^2
                loss_values = loss_module(current_estimate, target)
                avg_loss_values.append(loss_values.detach().mean())

                # Backprop and optimize
                loss_values.backward()
                optimizer.step()
                optimizer.zero_grad()

            loss_log.append([t, np.mean(avg_loss_values)])
            q_log.append([t, np.mean(avg_q_values)])

        episode += 1

        # Logging
        reward_log.append([t, total_reward])
        average_reward = np.mean([reward_log[-i][1] for i in range(1, min(10, len(reward_log)))])
        average_reward_log.append([t, average_reward])

        # Save best model
        if average_reward > best_average_reqard:
            name_prefix = f'best_t{t}_r{round(average_reward)}'
            save_model(agent.Q, model_path, name_prefix)
            best_average_reqard = average_reward
            # Record the performance of the best model
            record_gym(spec["env_name"], agent, video_path, name_prefix)

        if episode % spec["plot_frequency"] == 0:
            make_plot(reward_log, "Cartpole", "Timestep", "Reward per episode", os.path.join(plot_path, 'cartpole_rwpe.png'))
            make_plot(average_reward_log, "Cartpole", "Timestep", "Average reward in last 10 episodes", os.path.join(plot_path, 'cartpole_mrw.png'))
            make_plot(epsilon_log, "Cartpole", "Timestep", "epsilon", os.path.join(plot_path, 'cartpole_epsilon.png'))
            make_plot(loss_log, "Cartpole", "Timestep", "Mean loss", os.path.join(plot_path, 'cartpole_loss.png'))
            make_plot(q_log, "Cartpole", "Timestep", "Mean Q-value", os.path.join(plot_path, 'cartpole_q.png'))
    
    return agent

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str)
    parser.add_argument('--exp_name', type=str)
    args = parser.parse_args()

    spec = load_yaml(args.exp_path, args.exp_name)
    game = gym.make("CartPole-v1")
    agent = Agent(game, spec["num_cells"], spec["epsilon"])

    agent = vanilla_train(game, agent, spec)

