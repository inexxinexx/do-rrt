import numpy as np
import matplotlib.pyplot as plt

def plot_line(data):
    for key, y in data.items():
        x = np.arange(len(y))
        plt.plot(x, y, label=key)
        
    plt.title('loss curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    plt.legend()
    
    plt.show()
    
import json
import itertools
from utils import traj_smooth, plot_traj, traj_prune

def process_and_plot_trajs(file_path, result_path='', save=False):
    with open(file_path, 'r') as f:
        loaded_res = json.load(f)
        
    num_smooth_states = 100
    
    for i, result in enumerate(loaded_res):
        print('i =', i, 'epp=', result['epp'])
        
        ptraj = result['ptraj']
        
        # # ver.1
        # ptraj = traj_prune(ptraj)
        # ptraj = traj_smooth(ptraj, num_smooth_states)
        
        # ver.2: smoother than ver.1
        ptraj = traj_prune(ptraj)
        ptraj = traj_smooth(ptraj, int(num_smooth_states/2))
        ptraj = traj_prune(ptraj)
        ptraj = traj_smooth(ptraj, int(num_smooth_states*12))
        
        save_path = result_path + '/...'
        epp = list(itertools.chain(result['key_points'][0], result['key_points'][-1]))
        plot_traj(save_path, ptraj, end_point_pairs=epp, key_points=result['key_points'], color='r', save=False)
    
if __name__ == '__main__':
    # data = np.load('results/loss_****.npy', allow_pickle=True).item()
    # plot_line(data)
    
    ##################################################
    file_path = 'results\main_eval_res.json'
    result_path = ''
    process_and_plot_trajs(file_path, result_path, save=False)