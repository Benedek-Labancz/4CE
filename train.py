import os
import argparse
from collections import defaultdict
from tqdm import tqdm

from utils import load_yaml, make_plot, make_histogram, make_multiplot
from game import _4CE, _2CE
from agent import Agent, ReplayBuffer
from log import record_gym, create_experiment, save_model
from eval import self_play_eval
from config import device

import torch
from torch.optim import Adam
from torch.nn import MSELoss
import numpy as np

import gym


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


def base_train(game, agent, spec):
    
    print("Training Started.")
    print("Using device", device)

    buffer = ReplayBuffer(max_size=spec["max_buffer_size"])
    optimizer = Adam(params=agent.Q.parameters(), lr=spec["lr"])
    loss_module = MSELoss()

    t = 0
    T = spec["max_timesteps"]
    episode = 0
    pbar = tqdm(total=T)
    
    # Initialize all logs; they will be in the form [[t1, v1], ..., [tn, vn]] 
    reward_log = []
    win_score_log = []
    average_reward_log = []
    best_avg_q = 0 # to track the best model performance
    epsilon_log = []
    q_log = []
    loss_log = []
    grad_log = []
    eval_log = []
    single_reward_log = []

    # Open a new experiment run
    video_path, plot_path, model_path = create_experiment(args.exp_name, spec)

    # Initial random steps to collect data before training
    '''
    t_init = 0
    while t_init < spec["init_random_steps"]:
        game.reset()
        done = False
        while not done:
            fp_state = game.first_person_state(game.state) # we treat all steps as being done in "first-person" by the agent; thus we use all data and utilize self-play
            action = agent.explore_act(fp_state)
            new_state, reward, done, info = game.step(action)
            fp_state = game.flatten_state(fp_state) # we need to flatten the state to be fed into the agent's networks
            fp_new_state = game.first_person_state(new_state)
            fp_new_state = game.flatten_state(fp_new_state)
            flat_action = game.coords_to_action(action) # we need flat actions to easily acces the right action value later
            buffer.extend(fp_state, flat_action, reward, fp_new_state, done)
            game.state = new_state
            game.switch_turn()
            t_init += 1
            if t_init == spec["init_random_steps"]:
                break
    '''

    def get_response(game, agent, new_state):
        '''
        Function to simulate getting a response to a move done by the agent.
        The opponent is always assumed to play optimally (although we utilize self-play),
        thus we use agent.act() to generate a response.
        '''
        opponent_player = O if game.player == X else X
        fp_new_state = game.first_person_state(new_state, opponent_player)
        fp_new_state = game.flatten_state(fp_new_state)
        response = agent.act(fp_new_state)
        future_state = game.perform_action(new_state, response, opponent_player)
        points_scored = game.evaluate_action(future_state, response)
        opponent_reward = points_scored
        future_done = game.terminal_state(future_state)
        return future_state, opponent_reward, future_done

    # Main collection and training loop
    while t < T:
        game.reset()
        done = False
        total_rewards = defaultdict(lambda: 0)
        while not done:
            # Main data collection

            '''
            We perceive the state from first person.
            Act according to policy or explore.
            '''
            fp_state = game.first_person_state(game.state)
            fp_state = game.flatten_state(fp_state)
            action = agent.act(fp_state)
            flat_action = game.coords_to_action(action)

            '''
            Step the environment so that we receive the
            state following immediately.
            '''
            new_state, reward, done, info = game.step(action)
            flat_new_state = game.flatten_state(new_state)
            '''
            If the new_state is terminal, no response is received anymore.
            Thus we save new_state and the reward we got.
            '''
            if done:
                total_rewards[game.player] += reward # logging
                buffer.extend(fp_state, flat_action, reward, flat_new_state, done)
                single_reward_log.append(reward) # logging
            else:
                '''
                If the new_state is non-terminal, we receive a response from the opponent.
                This response is analogous to the environment dynamics leading to a newer state,
                a state where it's our turn again.
                '''
                # Generate the response from the new_state
                future_state, opponent_reward, future_done = get_response(game, agent, new_state)
                fp_future_state = game.first_person_state(future_state)
                fp_future_state = game.flatten_state(fp_future_state)
                net_reward = reward - opponent_reward
                total_rewards[game.player] += net_reward # logging
                buffer.extend(fp_state, flat_action, net_reward, fp_future_state, future_done)
                single_reward_log.append(net_reward) # logging

            '''
            Set the game state and switch turn
            to prepare for next step.
            '''
            game.state = new_state
            game.switch_turn()

            pbar.update()
            t += 1
            if t == T:
                break

            # Epsilon decay
            agent.epsilon = max(spec["min_epsilon"], agent.epsilon * spec["epsilon_decay"])
            epsilon_log.append([t, agent.epsilon])

            # Sync target network weights
            if t % spec["sync_frequency"] == 0:
                agent.sync_target()

            # Experience replay
            if t % spec["optim_frequency"] == 0:
                avg_q_values = []
                grad_norms = []
                avg_loss_values = []
                for _ in range(spec["optim_iter"]):
                    samples = buffer.sample(spec["batch_size"])
                    samples = buffer.batch_samples(samples)

                    target = samples["reward"]
                    target = target + (1 - samples["done"]) * spec["gamma"] * torch.max(agent.TargetNet(samples["new_state"]))
                    target = target.reshape(1, -1)
                    current_estimate = torch.gather(agent.Q(samples["state"]), dim=1, index=samples["action"])
                    avg_q_values.append(current_estimate.detach().mean())

                    loss_value = loss_module(current_estimate, target)
                    avg_loss_values.append(loss_value.detach().mean())
                    grad_norm = torch.autograd.grad(loss_value, agent.Q.parameters(), retain_graph=True)[0].norm()
                    print(grad_norm)
                    grad_norms.append(grad_norm)
                    loss_value.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                average_q_value = np.array(avg_q_values).mean().item()
                q_log.append([t, average_q_value])
                average_grad_norm = np.array(grad_norms).mean().item()
                grad_log.append([t, average_grad_norm])
                average_loss_value = np.array(avg_loss_values).mean().item()
                loss_log.append([t, average_loss_value])

                if average_q_value > best_avg_q:
                    name_prefix = 'best' # we want to overwrite the best model, as many of that will be produced during training
                    save_model(agent.Q, model_path, name_prefix)
                    best_avg_q = average_q_value
            
            if t % spec["checkpoint_frequency"] == 0:
                name_prefix = f'checkpoint_{t}'
                save_model(agent.Q, model_path, name_prefix)

            '''
            if t % spec["eval_frequency"] == 0:
                performance = self_play_eval(game, agent, spec["eval_num_episodes"])
                eval_log.append([t, performance])
            '''

        episode += 1

        win_score = max(game.score.values())
        win_score_log.append([t, win_score])
        avg_game_reward = sum(total_rewards.values()) / 2
        reward_log.append([t, avg_game_reward])
        average_reward = np.mean([reward_log[-i][1] for i in range(1, min(10, len(reward_log)))])
        average_reward_log.append([t, average_reward])

        if episode % spec["plot_frequency"] == 0:
            make_plot(reward_log, f"Avg Game Reward in {spec['game_name']}", "Timestep", "Mean reward per game", os.path.join(plot_path, 'v0_reward_per_episode.png'))
            make_plot(average_reward_log, "Average performance in last 10 games", "Timestep", "Average mean reward", os.path.join(plot_path, 'v0_mean_reward.png'))
            make_plot(win_score_log, "Winning Score in self-play", "Timestep", "Winning Score", os.path.join(plot_path, 'v0_win.png'))
            make_plot(epsilon_log, f"{spec['game_name']} agent epsilon", "Timestep", "epsilon", os.path.join(plot_path, 'v0_epsilon.png'))
            make_plot(q_log, f"{spec['game_name']} agent's average Q-value", "Timestep", "Mean Q-value", os.path.join(plot_path, 'v0_q.png'))
            make_plot(grad_log, "Gradient of loss", "Timestep", "Average norm of gradient of loss w.r.t. Q-network", os.path.join(plot_path, 'v0_grad.png'))
            make_plot(loss_log, "Loss", "Timestep", "Average loss", os.path.join(plot_path, 'v0_loss.png'))
            if len(eval_log) > 0:
                make_plot(eval_log, f"{spec['game_name']} agent average performance", "Timestep", "Mean Reward", os.path.join(plot_path, 'v0_eval.png'))
    
    make_histogram(single_reward_log, "Reward Frequency", "Reward Value", "Number of occurences", os.path.join(plot_path, 'v0_rhist.png'))


def train(game, agent, spec):
    
    print("Training Started.")
    print("Using device", device)

    buffer = ReplayBuffer(max_size=spec["max_buffer_size"])
    optimizer = Adam(params=agent.Q.parameters(), lr=spec["lr"])
    loss_module = MSELoss()

    t = 0
    T = spec["max_timesteps"]
    episode = 0
    pbar = tqdm(total=T)
    
    # Initialize all logs; they will be in the form [[t1, v1], ..., [tn, vn]] 
    reward_log = []
    win_score_log = []
    average_reward_log = []
    best_avg_q = 0 # to track the best model performance
    epsilon_log = []
    q_log = []
    loss_log = []
    grad_log = []
    eval_log = []
    single_reward_log = []

    state_diff_log = []

    # Open a new experiment run
    video_path, plot_path, model_path = create_experiment(args.exp_name, spec)

    # Initial random steps to collect data before training
    '''
    t_init = 0
    while t_init < spec["init_random_steps"]:
        game.reset()
        done = False
        while not done:
            fp_state = game.first_person_state(game.state) # we treat all steps as being done in "first-person" by the agent; thus we use all data and utilize self-play
            action = agent.explore_act(fp_state)
            new_state, reward, done, info = game.step(action)
            fp_state = game.flatten_state(fp_state) # we need to flatten the state to be fed into the agent's networks
            fp_new_state = game.first_person_state(new_state)
            fp_new_state = game.flatten_state(fp_new_state)
            flat_action = game.coords_to_action(action) # we need flat actions to easily acces the right action value later
            buffer.extend(fp_state, flat_action, reward, fp_new_state, done)
            game.state = new_state
            game.switch_turn()
            t_init += 1
            if t_init == spec["init_random_steps"]:
                break
    '''

    def get_response(game, agent, new_state):
        '''
        Function to simulate getting a response to a move done by the agent.
        The opponent is always assumed to play optimally (although we utilize self-play),
        thus we use agent.act() to generate a response.
        '''
        opponent_player = O if game.player == X else X
        fp_new_state = game.first_person_state(new_state, opponent_player)
        fp_new_state = game.flatten_state(fp_new_state)
        response = agent.act(fp_new_state)
        future_state = game.perform_action(new_state, response, opponent_player)
        points_scored = game.evaluate_action(future_state, response)
        opponent_reward = points_scored
        future_done = game.terminal_state(future_state)
        return future_state, opponent_reward, future_done

    # Main collection and training loop
    while t < T:
        game.reset()
        done = False
        total_rewards = defaultdict(lambda: 0)
        while not done:
            # Main data collection

            '''
            We perceive the state from first person.
            Act according to policy or explore.
            '''
            fp_state = game.first_person_state(game.state)
            fp_state = game.flatten_state(fp_state)

            state_diff_log.append([t, agent.Q(fp_state).detach()]) # logging

            action = agent.explore_act(fp_state)
            flat_action = game.coords_to_action(action)

            '''
            Step the environment so that we receive the
            state following immediately.
            '''
            new_state, reward, done, info = game.step(action)
            flat_new_state = game.flatten_state(new_state)
            '''
            If the new_state is terminal, no response is received anymore.
            Thus we save new_state and the reward we got.
            '''
            if done:
                total_rewards[game.player] += reward # logging
                buffer.extend(fp_state, flat_action, reward, flat_new_state, done)
                single_reward_log.append(reward) # logging
            else:
                '''
                If the new_state is non-terminal, we receive a response from the opponent.
                This response is analogous to the environment dynamics leading to a newer state,
                a state where it's our turn again.
                '''
                # Generate the response from the new_state
                future_state, opponent_reward, future_done = get_response(game, agent, new_state)
                fp_future_state = game.first_person_state(future_state)
                fp_future_state = game.flatten_state(fp_future_state)
                net_reward = reward - opponent_reward
                total_rewards[game.player] += net_reward # logging
                buffer.extend(fp_state, flat_action, net_reward, fp_future_state, future_done)
                single_reward_log.append(net_reward) # logging

            '''
            Set the game state and switch turn
            to prepare for next step.
            '''
            game.state = new_state
            game.switch_turn()

            pbar.update()
            t += 1
            if t == T:
                break

            # Epsilon decay
            agent.epsilon = max(spec["min_epsilon"], agent.epsilon * spec["epsilon_decay"])
            epsilon_log.append([t, agent.epsilon])

            # Sync target network weights
            if t % spec["sync_frequency"] == 0:
                agent.sync_target()

            # Experience replay
            if t % spec["optim_frequency"] == 0:
                avg_q_values = []
                grad_norms = []
                avg_loss_values = []
                for _ in range(spec["optim_iter"]):
                    samples = buffer.sample(spec["batch_size"])
                    samples = buffer.batch_samples(samples)

                    target = samples["reward"]
                    target = target + (1 - samples["done"]) * spec["gamma"] * torch.max(agent.TargetNet(samples["new_state"]))
                    target = target.reshape(1, -1)
                    print("TARGET", target)
                    current_estimate = torch.gather(agent.Q(samples["state"]), dim=1, index=samples["action"])
                    print("ESTIMATE", current_estimate)
                    avg_q_values.append(current_estimate.detach().mean())

                    loss_value = loss_module(current_estimate, target)
                    avg_loss_values.append(loss_value.detach().mean())
                    grad_norm = torch.autograd.grad(loss_value, agent.Q.parameters(), retain_graph=True)[0].norm()
                    grad_norms.append(grad_norm)
                    loss_value.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                average_q_value = np.array(avg_q_values).mean().item()
                q_log.append([t, average_q_value])
                average_grad_norm = np.array(grad_norms).mean().item()
                grad_log.append([t, average_grad_norm])
                average_loss_value = np.array(avg_loss_values).mean().item()
                loss_log.append([t, average_loss_value])

                if average_q_value > best_avg_q:
                    name_prefix = 'best' # we want to overwrite the best model, as many of that will be produced during training
                    save_model(agent.Q, model_path, name_prefix)
                    best_avg_q = average_q_value
            
            if t % spec["checkpoint_frequency"] == 0:
                name_prefix = f'checkpoint_{t}'
                save_model(agent.Q, model_path, name_prefix)

            '''
            if t % spec["eval_frequency"] == 0:
                performance = self_play_eval(game, agent, spec["eval_num_episodes"])
                eval_log.append([t, performance])
            '''

        episode += 1

        win_score = max(game.score.values())
        win_score_log.append([t, win_score])
        avg_game_reward = sum(total_rewards.values()) / 2
        reward_log.append([t, avg_game_reward])
        average_reward = np.mean([reward_log[-i][1] for i in range(1, min(10, len(reward_log)))])
        average_reward_log.append([t, average_reward])

        if episode % spec["plot_frequency"] == 0:
            make_plot(reward_log, f"Avg Game Reward in {spec['game_name']}", "Timestep", "Mean reward per game", os.path.join(plot_path, 'v0_reward_per_episode.png'))
            make_plot(average_reward_log, "Average performance in last 10 games", "Timestep", "Average mean reward", os.path.join(plot_path, 'v0_mean_reward.png'))
            make_plot(win_score_log, "Winning Score in self-play", "Timestep", "Winning Score", os.path.join(plot_path, 'v0_win.png'))
            make_plot(epsilon_log, f"{spec['game_name']} agent epsilon", "Timestep", "epsilon", os.path.join(plot_path, 'v0_epsilon.png'))
            make_plot(q_log, f"{spec['game_name']} agent's average Q-value", "Timestep", "Mean Q-value", os.path.join(plot_path, 'v0_q.png'))
            make_plot(grad_log, "Gradient of loss", "Timestep", "Average norm of gradient of loss w.r.t. Q-network", os.path.join(plot_path, 'v0_grad.png'))
            make_plot(loss_log, "Loss", "Timestep", "Average loss", os.path.join(plot_path, 'v0_loss.png'))
            make_multiplot(state_diff_log, "Difference between states", "Timestep", "Predicted Action Value", os.path.join(plot_path, 'v0_state_diff.png'))
            if len(eval_log) > 0:
                make_plot(eval_log, f"{spec['game_name']} agent average performance", "Timestep", "Mean Reward", os.path.join(plot_path, 'v0_eval.png'))
    
    make_histogram(single_reward_log, "Reward Frequency", "Reward Value", "Number of occurences", os.path.join(plot_path, 'v0_rhist.png'))

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
    best_average_reward = 0 # to track the best model performance
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
        if average_reward > best_average_reward:
            name_prefix = f'best_t{t}_r{round(average_reward)}'
            save_model(agent.Q, model_path, name_prefix)
            best_average_reward = average_reward
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
    game = _2CE()
    agent = Agent(game, spec["num_cells"], spec["epsilon"])

    agent = train(game, agent, spec)

