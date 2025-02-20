import argparse
from short_traj_gen.agent import Actor_agent
from short_traj_gen.dataset_loader import determ_orient, GoalOrientation, TrajectoryDataset
from rrt.rrt_planner import RRTPlanner
from utils import traj_smooth, plot_traj, plot_trajs, traj_prune, set_seed
import json

import numpy as np

class DDRRTSearch():
    def __init__(self, args, kkey_points_dict):
        self.planar = RRTPlanner(**vars(args))
        self.upper_key_points = kkey_points_dict['upper_key_points']
        self.lower_key_points = kkey_points_dict['lower_key_points']
        
        self.init_delta = args.delta
        self.num_smooth_states = args.num_smooth_states
        self.result_path = args.result_path
        
    '''
        RRT search with linear delta decay
    '''
    def _delta_linear_decay(self):
        self.planar.delta = max(0, min(1, self.planar.delta - 0.1)) # with a 0.1 linear delta decay
        self.planar.goal_threshold = self.planar.delta
        
    def rrt_LDD_search(self, key_points):
        planar = self.planar
        
        planar.delta = max(0, self.init_delta)
        planar.goal_threshold = planar.delta
        
        ptraj, ptraj_valid = None, True
        ptraj_pre, ptraj_valid_pre = None, None
        
        i = 0
        while planar.delta >= 0 and ptraj_valid:
            ptraj_pre = ptraj
            ptraj_valid_pre = ptraj_valid
            
            print(f'Current delta: {planar.delta}.')
            ptraj, ptraj_valid = planar.rrt_connect_search(key_points)
                        
            self._delta_linear_decay()
            
            i += 1
            
        delta_pre = planar.delta + 0.2 if not ptraj_valid else 0
        
        print(f'After {i} times of LDD rrt_connect_search, `delta` is {delta_pre}, `ptraj_valid` is {ptraj_valid_pre}.')
        return delta_pre, ptraj_pre, ptraj_valid_pre
    
    '''
        RRT search with exponential delta decay
    '''
    def _set_planar_delta(self, delta):
        self.planar.delta = delta
        self.planar.goal_threshold = delta
    
    def _delta_binary_search(self, delta_min, delta_max, key_points):
        planar = self.planar
        
        delta_min = max(0, delta_min)
        delta_max = max(delta_min, delta_max)
        
        self._set_planar_delta(delta_max)
        
        print(f'Current delta: {planar.delta}.')
        ptraj_b, ptraj_valid_b = planar.rrt_connect_search(key_points)      # currently best
        
        i = 1
        while round(delta_max - delta_min, 1) > 0.1:
            delta_mid = round((delta_min + delta_max) / 2.0, 1)             # consider delta with only one decimal place
            
            self._set_planar_delta(delta_mid)
            
            print(f'Current delta: {planar.delta}.')
            ptraj, ptraj_valid = planar.rrt_connect_search(key_points)
            
            if ptraj_valid:
                ptraj_b = ptraj
                ptraj_valid_b = ptraj_valid
                
                delta_max = delta_mid
            else:
                delta_min = delta_mid
                
            i += 1
            
        print(f'After {i} times of `rrt_connect_search` with binary searched EDD, `delta` is {delta_max}, `ptraj_valid` is {ptraj_valid_b}.')
        return delta_max, ptraj_b, ptraj_valid_b
    
    def rrt_EDD_search(self, key_points):
        delta, ptraj, ptraj_valid = self._delta_binary_search(0, self.init_delta, key_points)
        return delta, ptraj, ptraj_valid
    
    def construct_keyPoints(self, end_point_pair):
        upper_key_points, lower_key_points = self.upper_key_points, self.lower_key_points
        
        start_pos = end_point_pair[:3]
        end_pos = end_point_pair[-3:]
        O_pos = upper_key_points[-1]
        
        '''
            In the dataset, some sequences are only parts of complete trajectories. Specifically,
            for a sequence in the dataset, if both endpoints are either far from or very close to `O_pos`,
            only one endpoint is used for evaluation.
        '''
        
        end_dir = determ_orient(start_pos, end_pos)
        if end_dir == GoalOrientation.O:                                    # In dir
            start_dir = determ_orient(O_pos, start_pos)
            
            key_points = [start_pos]
            if start_dir == GoalOrientation.E:
                key_points += upper_key_points
            elif start_dir == GoalOrientation.W:
                key_points += lower_key_points
            elif start_dir == GoalOrientation.S or start_dir == GoalOrientation.SE:
                key_points += lower_key_points[-3:]
            elif start_dir == GoalOrientation.SW:
                key_points += lower_key_points[-4:]
            elif start_dir == GoalOrientation.NE:
                key_points += upper_key_points[-4:]
            elif start_dir == GoalOrientation.N or start_dir == GoalOrientation.NW:
                key_points += upper_key_points[-3:]
            else:
                raise ValueError('Incorrect start_dir!')
            
        else:                                                               # Out dir
            end_dir = determ_orient(O_pos, end_pos)
            
            upper_key_points, lower_key_points = (
                upper_key_points[-1:-3:-1] + lower_key_points[:2],          # [middle, middle left, lower left, lower]
                lower_key_points[-1:-3:-1] + upper_key_points[:2]           # [middle, middle right, upper right, upper]
            )
            
            if end_dir == GoalOrientation.E:
                key_points = lower_key_points[:2]
            elif end_dir == GoalOrientation.W:
                key_points = upper_key_points[:2]
            elif end_dir == GoalOrientation.N or end_dir == GoalOrientation.NE:
                key_points = lower_key_points[:3]
            elif end_dir == GoalOrientation.NW:
                key_points = lower_key_points
            elif end_dir == GoalOrientation.SE:
                key_points = upper_key_points
            elif end_dir == GoalOrientation.S or end_dir == GoalOrientation.SW:
                key_points = upper_key_points[:3]
            else:
                raise ValueError('Incorrect end_dir!')
            key_points += [end_pos]
            
        return key_points
    
    def plot_ptraj(self, ptraj, key_points=None):
        ptraj = traj_prune(ptraj)
        ptraj = traj_smooth(ptraj, self.num_smooth_states)
        plot_traj(self.result_path, ptraj, key_points=key_points)
    
'''
    util funcs
'''
def load_testing_endPoints(folder_path, temp_path):
    seq_list = TrajectoryDataset(folder_path, temp_path).data
    
    epp_arr = []
    for seq in seq_list:
        end_point_pair = np.concatenate((seq[0, :], seq[-1, :]))
        epp_arr.append(end_point_pair)
        
    return epp_arr                                                          # [numpy([x1, y1, z1, x2, y2, z2]) for 1st seq, ...]

if __name__ == '__main__':
    '''
        Hyperparameter setting
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=999, help='Random seed')
    
    # hyperparameters for DDPG training
    parser.add_argument('--result_path', type=str, default='results', help='Path for storing statistical results, observation plots, and etc.')
    parser.add_argument('--num_epochs', type=int, default=1000, help='num_epochs')
    parser.add_argument('--save_every_epochs', type=int, default=1000, help='Number of epochs between model saves')
    parser.add_argument('--model_para_path', type=str, default='model_ws', help='model_para_path')
    
    # hyperparameters of rrt
    delta_default = 1.0                                     # different from `main.py`
    parser.add_argument('--max_step_size', type=int, default=3, help='max_step_size the robot can move in a single interpolation step')
    parser.add_argument('--delta', type=float, default=delta_default, help='Discontinuity bound')
    parser.add_argument('--goal_threshold', type=float, default=delta_default, help='Reach the ultimate goal region')
    parser.add_argument('--num_shortTrajs', type=int, default=20, help='Number of short trajectories to be generated in one expansion during rrt search')
    parser.add_argument('--size_of_shortTraj', type=int, default=20, help='Length of each short trajectory, i.e., how many states (or positions) will be included in each generated trajectory')
    parser.add_argument('--rrt_max_rounds', type=int, default=100, help='max_rounds for traversing all key_points in rrt_connect_search')

    # hyperparameters of traj smoothing
    parser.add_argument('--num_smooth_states', type=int, default=100, help='Number of evenly spaced state points to be generated for smoothing a trajectory')

    args = parser.parse_args()
    # print(args)
    
    '''
        Environment setting
    '''
    O_pos = [0.82, 0, 0]
    upper_key_points = [
        [3.12, 0.79, 0.4],                                  # upper right
        [0.20, 1.47, 0.4],                                  # upper
        
        [-2.06, 0.40, 0.2],                                 # upper left
        
        [0, 0, 0.2],                                        # middle left
        O_pos                                               # middle
    ]
    lower_key_points = [
        [-1.95, -0.56, 0.4],                                # lower left
        [0.27, -1.31, 0.4],                                 # lower
        [3.15, -0.82, 0.4],                                 # lower right
        [1.44, 0, 0.2],                                     # middle right
        O_pos                                               # middle
    ]
    kkey_points_dict = {
        'upper_key_points': upper_key_points,
        'lower_key_points': lower_key_points
    }
    
    args.state_dim = 3                                      # (x, y, z)
    
    '''
        Procedure steps
    '''
    set_seed(args.seed)
    
    folder_path = 'datasets/test_set'
    temp_path = 'Temp'
    epp_arr = load_testing_endPoints(folder_path, temp_path)
    
    searcher = DDRRTSearch(args, kkey_points_dict)
    
    i = 1
    res = []
    num_success = 0
    for epp in epp_arr:
        print('i =', i)
        print('epp =', epp)
        key_points = searcher.construct_keyPoints(epp)
        # delta, ptraj, ptraj_valid = searcher.rrt_LDD_search(key_points)
        delta, ptraj, ptraj_valid = searcher.rrt_EDD_search(key_points)
        
        res.append({
            'epp': epp.tolist(),
            'key_points': [p.tolist() if isinstance(p, np.ndarray) else p for p in key_points],
            'delta': delta,
            'ptraj': [p.tolist() for p in ptraj],
            'ptraj_valid': ptraj_valid
        })
        
        if ptraj_valid:
            num_success += 1
            
        # searcher.plot_ptraj(ptraj, key_points=key_points)
        
        i += 1
        
    with open(args.result_path + '/main_eval_res.json', 'w') as f:
        json.dump(res, f)
    
    print(f'Success rate of dataset {folder_path} is {num_success / len(epp_arr)}.')