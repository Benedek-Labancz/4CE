import yaml
import json
import torch
import os

def load_yaml(filename, param_set):
    with open(filename, 'r') as f:
        all_sets = yaml.safe_load(f)
        return all_sets[param_set]

def load_json(path):
    with open(path, 'r') as f:
        data = f.read()
        data = json.loads(data)
    return data
    
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