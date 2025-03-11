import yaml
from matplotlib import pyplot as plt
import numpy as np

def load_yaml(filename, param_set):
    with open(filename, 'r') as f:
        all_sets = yaml.safe_load(f)
        return all_sets[param_set]
    
def make_plot(values, title, x_label, y_label, path):
    values = np.array(values)
    plt.plot(values[:, 0], values[:, 1])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(path)
    plt.close()