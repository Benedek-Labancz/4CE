import numpy as np
import torch
from torch.nn import LazyLinear, ReLU, Tanh


class ReplayBuffer:
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size

    def __len__(self):
        return len(self.storage)

    def extend(self, state, action, reward, new_state, done):
        '''
        Extend the buffer with an experience.
        If max_size is reached, the oldest element from the buffer is discarded.
        '''
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "new_state": new_state,
            "done": int(done)
        }
        if self.__len__() == self.max_size:
            self.storage.pop()
        self.storage.insert(0, experience)
    
    def sample(self, n):
        '''
        Sample n number of experiences with replacement from a uniform distribution.
        '''
        samples = np.random.choice(self.storage, size=n, replace=True)
        return samples

    def batch_samples(self, samples):
        '''
        Turn a list of samples into a batched dictionary
        batch = {
            state: [s1, s2, ..., sn],
            action: [a1, a2, ..., an],
            ...
        }
        '''
        batch = {
            'state': np.array([sample['state'] for sample in samples]),
            'action': np.array([sample['action'] for sample in samples]),
            'reward': np.array([sample['reward'] for sample in samples]),
            'new_state': np.array([sample['new_state'] for sample in samples]),
            'done': np.array([sample['done'] for sample in samples])
        }
        return batch


class QNet(torch.nn.Module):
    def __init__(self, num_cells, num_actions):
        super().__init__()
        self.input = LazyLinear(num_cells[0])
        self.hidden = LazyLinear(num_cells[1])
        self.output = LazyLinear(num_actions)
        self.activation = Tanh()

    def forward(self, state): 
        #batch_size = state.shape[0] # could be more efficient
        #state = state.reshape(batch_size, -1)
        h = self.activation(self.input(state))
        h = self.activation(self.hidden(h))
        out = self.output(h)
        return out

class Agent:
    def __init__(self, game, num_cells, epsilon):
        self.game = game
        #self.num_actions = self.game.action_spec["n"]
        self.num_actions = self.game.action_space.n
        self.Q = QNet(num_cells, self.num_actions)
        self.TargetNet = QNet(num_cells, self.num_actions)
        self.sync_target() # init target with same parameters
        self.epsilon = epsilon

    def act(self, state):
        '''
        Greedily choose an action with the highest state-action value.
        '''
        action_values = self.Q(state)
        action_values = action_values * self.game.get_action_mask(state) # TODO: what about negative values? to those, zero is superior
        flat_action = torch.argmax(action_values, dim=0)
        action = self.game.action_to_coords(flat_action)
        return action
    
    def explore_act(self, state):
        '''
        Explore according to an epsilon-greedy approach.
        With probability epsilon, choose a random action,
        and probability (1 - epsilon), act in a greedy fashion.
        '''
        if np.random.random() < self.epsilon:
            action_mask = self.game.get_action_mask(state)
            flat_action = np.random.choice([i for i in range(self.num_actions) if action_mask[i]])
            action = self.game.action_to_coords(flat_action)
            return action
        else:
            return self.act(state)

    def vanilla_act(self, state):
        action_values = self.Q(state)
        action = torch.argmax(action_values, dim=0).item()
        return action
    
    def vanilla_explore_act(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
            return action
        else:
            return self.vanilla_act(state)
    
    def sync_target(self):
        '''
        Sync the parameters of the target network with the Q-network.
        This usually happens every some number of steps during training.
        '''
        self.TargetNet.load_state_dict(self.Q.state_dict())
