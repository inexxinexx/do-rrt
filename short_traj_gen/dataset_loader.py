import torch
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import numpy as np
import json
import math
from enum import Enum
import torch.nn.functional as F

class GoalOrientation(Enum):
    O = 0,
    E = 1,
    NE = 2,
    N = 3,
    NW = 4,
    SW = 5,
    S = 6,
    SE = 7,
    W = 8

class TrajectoryDataset(Dataset):
    def __init__(self, folder_path, result_path):
        files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')])
        
        def _read_file(_path, delim='\t'):
            data = []
            with open(_path, 'r') as f:
                for line in f:
                    line = line.strip().split(delim)
                    line = [np.float32(e) for e in line]
                    data.append(line)
            return np.asarray(data)
        
        max_agents_in_file = 0
        seq_list = []
        for path in tqdm(files):
            data = _read_file(path, delim=' ')                              # (frame_id, agent_id, x, y, z, unused wind_x, unused wind_y)
            if len(data) == 0:
                print(f'{path} is empty.')
                continue
            
            curr_seq_data = data
            agents_in_curr_seq = np.unique(curr_seq_data[:,1])
            max_agents_in_file = max(max_agents_in_file, len(agents_in_curr_seq))
            for agent_idx, agent_id in enumerate(agents_in_curr_seq):
                curr_agent_seq = curr_seq_data[curr_seq_data[:,1] == agent_id, :]
                curr_agent_seq = curr_agent_seq[:, 2:5]
                curr_agent_seq[:, -1] = np.vectorize(discretize_altitude)(curr_agent_seq[:, -1])
                
                if len(curr_agent_seq) > 1:
                    seq_list.append(curr_agent_seq)
                
        print('max_agents_in_file =', max_agents_in_file)
        
        # Cal and save the ranges of pos on the x, y, and z axes
        pos_min = np.min([seq.min(axis=0) for seq in seq_list], axis=0)
        pos_max = np.max([seq.max(axis=0) for seq in seq_list], axis=0)
        pos_range = {
            'min': pos_min.tolist(),
            'max': pos_max.tolist()
        }
        with open(os.path.join(result_path, 'pos_range.json'), 'w') as f:
            json.dump(pos_range, f)
            
        print('pos_min =', pos_min, 'pos_max =', pos_max)
        
        # Cal and save the range of r, i.e., adjacent positions' distances
        distances = [np.linalg.norm(seq[1:] - seq[:-1], axis=1) for seq in seq_list]
        r_min = np.min([d.min() for d in distances])
        r_max = np.max([d.max() for d in distances])
        r_range = {'min': float(r_min), 'max': float(r_max)}
        with open(os.path.join(result_path, 'r_range.json'), 'w') as f:
            json.dump(r_range, f)
        
        print('r_min =', r_min, 'r_max =', r_max)
        
        self.data = seq_list                                                # shape(files*agents, not necessarily equal T, 3:xyz)
        
def discretize_altitude(z):
    if z < 0.5 * 0.3048:
        return 0.0
    elif 0.5 * 0.3048 <= z < 1.5 * 0.3048:
        return 0.2                                                          # 1000
    elif 1.5 * 0.3048 <= z < 2.5 * 0.3048:
        return 0.4                                                          # 2000
    elif 2.5 * 0.3048 <= z < 3.5 * 0.3048:
        return 0.6                                                          # 3000
    elif 3.5 * 0.3048 <= z < 4.5 * 0.3048:
        return 0.8                                                          # 4000
    else:
        return 1.0                                                          # 5000
    
def determ_orient(start_pos, end_pos, O_pos=[0.82, 0, 0]):
    start2O_d_square = (start_pos[0] - O_pos[0])**2 + (start_pos[1] - O_pos[1])**2
    end2O_d_square = (end_pos[0] - O_pos[0])**2 + (end_pos[1] - O_pos[1])**2
    
    if start2O_d_square < end2O_d_square:
        angle = math.degrees(math.atan2(end_pos[1] - O_pos[1], end_pos[0] - O_pos[0]))
        
        if -1 * 22.5 <= angle < 1 * 22.5:
            return GoalOrientation.E
        elif 1 * 22.5 <= angle < 3 * 22.5:
            return GoalOrientation.NE
        elif 3 * 22.5 <= angle < 5 * 22.5:
            return GoalOrientation.N
        elif 5 * 22.5 <= angle < 7 * 22.5:
            return GoalOrientation.NW
        elif -7 * 22.5 <= angle < -5 * 22.5:
            return GoalOrientation.SW
        elif -5 * 22.5 <= angle < -3 * 22.5:
            return GoalOrientation.S
        elif -3 * 22.5 <= angle < -1 * 22.5:
            return GoalOrientation.SE
        else:
            return GoalOrientation.W
        
    else:
        return GoalOrientation.O
    
def collate_fn(batch):
    s_s = []
    ns_s = []
    g_s = []
    num_classes = len(GoalOrientation)
    for seq in batch:
        seq_s = seq[:-1]                                                    # seq is a np.ndarray, dtype=np.float32
        seq_ns = seq[1:]
        seq_g = np.full((seq_s.shape[0],), fill_value=determ_orient(seq[0], seq[-1]).value)
        seq_g = np.eye(num_classes, dtype=np.float32)[seq_g]                # goal enum to one-hot float32
                
        s_s.append(seq_s)
        ns_s.append(seq_ns)
        g_s.append(seq_g)
        
    s_s = torch.from_numpy(np.vstack(s_s))
    ns_s = torch.from_numpy(np.vstack(ns_s))
    g_s = torch.from_numpy(np.vstack(g_s))
        
    return s_s, ns_s, g_s                                                   # all to float32 for the later cat
    
def get_traj_DataLoader(folder_path, batch_size, result_path):
    dataset = TrajectoryDataset(folder_path, result_path).data
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        collate_fn=collate_fn,
                        pin_memory=True,
                        drop_last=True,
                        shuffle=True)
    return loader