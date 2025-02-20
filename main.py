import argparse
from short_traj_gen.agent import Actor_agent
from short_traj_gen.dataset_loader import determ_orient, GoalOrientation
from rrt.rrt_planner import RRTPlanner
from utils import traj_smooth, plot_traj, plot_trajs, traj_prune, set_seed

import numpy as np
import torch

def __model_test(args, start_point, goal : GoalOrientation, ptraj_sz, direction=RRTPlanner.Direction.FORWARD):
    nn_planer = Actor_agent(mode='test', **vars(args))
    nn_planer.load()
    
    pos = np.array(start_point, dtype=np.float32)
    
    pos_tensor = torch.from_numpy(pos).unsqueeze(0)
    
    num_classes = len(GoalOrientation)
    goal = np.eye(num_classes, dtype=np.float32)[goal.value]
    goal_tensor = torch.from_numpy(goal).unsqueeze(0)
    
    if direction == RRTPlanner.Direction.FORWARD:
        predict = nn_planer.predict
    else:
        predict = nn_planer.predict_b
    
    ptraj = [pos]
    for _ in range(ptraj_sz-1):
        next_pos_tensor = predict(pos_tensor, goal_tensor)
        next_pos = next_pos_tensor.squeeze(0).cpu().detach().numpy()
        ptraj.append(next_pos)
        print('next_pos =', next_pos)
        pos_tensor = next_pos_tensor
            
    return ptraj

def __model_test_2(args, num_ptrajs, ptraj_sz, goal : GoalOrientation, direction=RRTPlanner.Direction.FORWARD, xyz_ranges=None):
    nn_planer = Actor_agent(mode='test', **vars(args))
    nn_planer.load()
    
    if xyz_ranges:
        x = np.random.uniform(xyz_ranges['x'][0], xyz_ranges['x'][1], num_ptrajs).astype(np.float32)
        y = np.random.uniform(xyz_ranges['y'][0], xyz_ranges['y'][1], num_ptrajs).astype(np.float32)
        z = np.random.uniform(xyz_ranges['z'][0], xyz_ranges['z'][1], num_ptrajs).astype(np.float32)
    else:
        x = np.random.uniform(-6, 6, num_ptrajs).astype(np.float32)
        y = np.random.uniform(-6, 6, num_ptrajs).astype(np.float32)
        z = np.random.uniform(0, 1, num_ptrajs).astype(np.float32)
    
    starts = np.column_stack((x, y, z))
    
    goal = goal.value
    num_classes = len(GoalOrientation)
    goal = np.eye(num_classes, dtype=np.float32)[goal]
    goal_tensor = torch.from_numpy(goal).unsqueeze(0)
    
    if direction == RRTPlanner.Direction.FORWARD:
        predict = nn_planer.predict
    else:
        predict = nn_planer.predict_b
    
    ptrajs = []
    for start in starts:
        ptraj = [start]
        pos_tensor = torch.from_numpy(start).unsqueeze(0)
        for _ in range(ptraj_sz-1):
            next_pos_tensor = predict(pos_tensor, goal_tensor)
            next_pos = next_pos_tensor.squeeze(0).cpu().detach().numpy()
            ptraj.append(next_pos)
            pos_tensor = next_pos_tensor
        ptrajs.append(ptraj)
        
    return ptrajs

def model_training(args):
    agent = Actor_agent(**vars(args))
    agent.train()
    
def rrt_search(args, key_points):
    planar = RRTPlanner(**vars(args))
    ptraj, ptraj_valid = planar.rrt_connect_search(key_points)
    print('ptraj_valid:', 'true.' if ptraj_valid else 'false.')
    return ptraj

if __name__ == '__main__':
    '''
        Hyperparameter setting
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=555, help='Random seed')

    # hyperparameters for DDPG training
    parser.add_argument('--lr_a', type=float, default=1e-3, help='Learning rate of actor')
    parser.add_argument('--lr_c', type=float, default=1e-3, help='Learning rate of critic')
    parser.add_argument('--folder_path', type=str, default='datasets/training_set', help='Training set folder path.')
    parser.add_argument('--batch_size', type=int, default=20, help='batch_size')
    # parser.add_argument('--num_epochs', type=int, default=5e6, help='num_epochs')
    parser.add_argument('--num_epochs', type=int, default=1000, help='num_epochs')
    parser.add_argument('--result_path', type=str, default='results', help='Path for storing statistical results, observation plots, and etc.')
    parser.add_argument('--model_para_path', type=str, default='model_ws', help='model_para_path')
    parser.add_argument('--save_every_epochs', type=int, default=1000, help='Number of epochs between model saves')
    parser.add_argument('--terminate_bootstrapping', type=bool, default=True, help='Stops accumulation after terminal states')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor in the Bellman equation')
    parser.add_argument('--tau', type=float, default=1e-3, help='hyperparameter used in the soft update of target networks')

    # hyperparameters of rrt
    delta_default = 0.3
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
    # key_points = [
    #     [0.5, -2.8, 0.6],                                   # from bottom
    #     [3.15, -0.82, 0.4],
    #     [1.44, 0, 0.2],
    #     O_pos
    # ]
    # key_points = [
    #     [-1, -4, 0.6],                                      # from bottom left
    #     [0.27, -1.31, 0.4],
    #     [3.15, -0.82, 0.4],
    #     [1.44, 0, 0.2],
    #     O_pos
    # ]
    # key_points = [
    #     [-1.3, 3.7, 0.6],                                   # from top left
    #     [-2.06, 0.40, 0.2],
    #     [0, 0, 0.2],
    #     O_pos
    # ]
    key_points = [
        [3.8, 3.4, 0.4],                                    # from top right
        [0.20, 1.47, 0.4],
        [-2.06, 0.40, 0.2],
        [0, 0, 0.2],
        O_pos
    ]
    
    args.state_dim = 3                                      # (x, y, z)
    
    '''
        Procedure steps
    '''
    set_seed(args.seed)
    
    # model_training(args)
    
    ptraj = rrt_search(args, key_points)
    
    ptraj = traj_prune(ptraj)
    ptraj = traj_smooth(ptraj, args.num_smooth_states)
    plot_traj(args.result_path, ptraj, key_points=key_points)
    
    
    # '''
    #     Tests are all after `model_training(args)`.
    # '''
    
    # '''
    #     Test: Predict a size of `ptraj_sz` traj
    # '''
    # # ptraj = __model_test(args, key_points[0], GoalOrientation.O, ptraj_sz=20)
    # ptraj = __model_test(args, key_points[1], GoalOrientation.O, ptraj_sz=20, direction=RRTPlanner.Direction.BACKWARD)
    # plot_traj(args.result_path, ptraj)
    
    # '''
    #     Test: Random starts
    # '''
    # # Not specifying xyz_ranges
    # # ptrajs = __model_test_2(args, num_ptrajs=400, ptraj_sz=20, goal=GoalOrientation.O)
    # ptrajs = __model_test_2(args, num_ptrajs=400, ptraj_sz=20, goal=GoalOrientation.O, direction=RRTPlanner.Direction.BACKWARD)
    
    # plot_trajs(args.result_path, ptrajs)
    
    # # Specifying xyz_ranges
    # delta_as_r = delta_default
    # def c2range(c, delta_as_r):
    #     return [c - delta_as_r, c + delta_as_r]
    # xyz_pos = key_points[0]
    # # xyz_pos = key_points[1]
    # # xyz_pos = key_points[2]
    # xyz_ranges = {                                          # Note: these ranges constitute a CUBOID not a ball that we idealize
    #     'x': c2range(xyz_pos[0], delta_as_r),
    #     'y': c2range(xyz_pos[1], delta_as_r),
    #     'z': c2range(xyz_pos[2], delta_as_r)
    # }
    # ptrajs = __model_test_2(args, num_ptrajs=args.num_shortTrajs, ptraj_sz=args.size_of_shortTraj, goal=GoalOrientation.O, xyz_ranges=xyz_ranges)
    # # ptrajs = __model_test_2(args, num_ptrajs=args.num_shortTrajs, ptraj_sz=args.size_of_shortTraj, goal=GoalOrientation.O, direction=RRTPlanner.Direction.BACKWARD, xyz_ranges=xyz_ranges)
        
    # plot_trajs(args.result_path, ptrajs, key_points=key_points)