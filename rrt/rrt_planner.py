from .node import State, Node, ShortTraj
from enum import Enum
import random
from short_traj_gen.agent import Actor_agent
import numpy as np
import torch
from short_traj_gen.dataset_loader import GoalOrientation, determ_orient
from .rrt_solution import RRTSolution

class RRTPlanner():
    class Direction(Enum):
            FORWARD = 1,
            BACKWARD = 2
            
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.nn_planer = Actor_agent(mode='test', **kwargs)
        self.nn_planer.load()
        
    def _nearest_state(self, n, T_n):
        min_d = float('inf')
        neigh = None
        for tn in T_n:
            d = tn.dist(n)
            if d < min_d:
                min_d = d
                neigh = tn
        return neigh
    
    def _interpolate(self, _from : Node, _to : Node, step_sz : float):            
        d_f2t = _from.dist(_to)
        if d_f2t > step_sz:
            dt = step_sz / d_f2t
            interp_pos = _from.state.pos + dt * (_to.state.pos - _from.state.pos)
        else:
            interp_pos = _to.state.pos
        return interp_pos
    
    def _genShortTrajs_fromNN(self, origin_pos : np.ndarray, r : float, goal : GoalOrientation, direction : Direction):
        thetas = np.random.uniform(0, 2 * np.pi, self.num_shortTrajs)
        radii = np.random.uniform(0, r, self.num_shortTrajs)
        starts = origin_pos + radii[:, np.newaxis] * np.column_stack((np.cos(thetas), np.sin(thetas), np.zeros(self.num_shortTrajs)))
        starts = starts.astype(np.float32)
        
        num_classes = len(GoalOrientation)
        goal = np.eye(num_classes, dtype=np.float32)[goal.value]
        goal_tensor = torch.from_numpy(goal).unsqueeze(0)
        
        if direction == RRTPlanner.Direction.FORWARD:
            predict = self.nn_planer.predict
        else:
            predict = self.nn_planer.predict_b
                
        trajs = []
        with torch.no_grad():
            for pos in starts:
                traj = ShortTraj(self.size_of_shortTraj)
                traj.add(State(pos))
                
                pos_tensor = torch.from_numpy(pos).unsqueeze(0)
                
                for _ in range(self.size_of_shortTraj-1):
                    next_pos_tensor = predict(pos_tensor, goal_tensor)
                    
                    next_pos = next_pos_tensor.squeeze(0).cpu().detach().numpy()
                    traj.add(State(next_pos))
                    
                    pos_tensor = next_pos_tensor
                    
                trajs.append(traj)
            
        return trajs
    
    def _recover_planned_traj_from_sol(self, sol):
        ptraj = []                                                          # ptraj is a pos array
        
        for i in range(0, len(sol), 2):
            fwd_n = sol[i]
            bwd_n = sol[i+1]
            
            fwd_pos_arr = []
            bwd_pos_arr = []
            tmp_pos_arr = None
            tmp_n = None
            for direction in [RRTPlanner.Direction.FORWARD, RRTPlanner.Direction.BACKWARD]:
                if direction == RRTPlanner.Direction.FORWARD:
                    tmp_pos_arr = fwd_pos_arr
                    tmp_n = fwd_n
                else:
                    tmp_pos_arr = bwd_pos_arr
                    tmp_n = bwd_n
                    
                while tmp_n.parent is not None:                             # tmp_n is not a key_point and has traj
                    for s in reversed(tmp_n.traj):
                        tmp_pos_arr.append(s.pos)
                    tmp_n = tmp_n.parent
                    
                if direction == RRTPlanner.Direction.FORWARD or i == len(sol) - 2:     # after 'while', if tmp_n is a FORWARD key_point or the last BACKWARD key_point
                    tmp_pos_arr.append(tmp_n.state.pos)
                    
            ptraj += reversed(fwd_pos_arr)
            ptraj += bwd_pos_arr
            
        return ptraj
    
    def rrt_connect_search(self, key_points):
        goal = determ_orient(key_points[0], key_points[-1])
        
        T_n_arr = []
        for kp in key_points:
            T_n_arr.append([Node(State.from_xyz(*kp))])
            
        rrt_sol = RRTSolution(len(T_n_arr)-1)
                
        rand_node = Node()
        
        rrt_round = 0
        while not rrt_sol.finished() and rrt_round < self.rrt_max_rounds:
            for i in range(0, len(T_n_arr)-1):
                if rrt_sol.sol_npairs_found(i):
                    continue
                
                T_n = T_n_arr[i]
                T_nrev = T_n_arr[i+1]                                           # also used in next round, as T_n
                
                for direction in [RRTPlanner.Direction.FORWARD, RRTPlanner.Direction.BACKWARD]:
                    if direction == RRTPlanner.Direction.FORWARD:
                        x_rand = random.choice(T_nrev).state.pos
                    else:
                        x_rand = random.choice(T_n).state.pos
                        
                    rand_node.state.pos = x_rand
                    
                    if direction == RRTPlanner.Direction.FORWARD:
                        near_node = self._nearest_state(rand_node, T_n)
                    else:
                        near_node = self._nearest_state(rand_node, T_nrev)
                        
                    x_target = self._interpolate(near_node, rand_node, self.max_step_size)
                    
                    trajs = self._genShortTrajs_fromNN(near_node.state.pos, self.delta, goal, direction)
                    
                    min_d = float('inf')
                    chosen_traj = None
                    for traj in trajs:
                        d = traj.back().dist(x_target)
                        if d < min_d:
                            min_d = d
                            chosen_traj = traj
                            
                    __expand_end = chosen_traj.back()
                    
                    new_node = Node(__expand_end, near_node, chosen_traj)
                    
                    if direction == RRTPlanner.Direction.FORWARD:
                        tmp_node = self._nearest_state(new_node, T_n)
                    else:
                        tmp_node = self._nearest_state(new_node, T_nrev)
                        
                    if new_node.dist(tmp_node) < self.delta / 2.:
                        del tmp_node
                        continue
                    
                    if direction == RRTPlanner.Direction.FORWARD:
                        T_n.append(new_node)
                        tmp_node = self._nearest_state(new_node, T_nrev)
                    else:
                        T_nrev.append(new_node)
                        tmp_node = self._nearest_state(new_node, T_n)
                        
                    if new_node.dist(tmp_node) < self.goal_threshold:
                        if direction == RRTPlanner.Direction.FORWARD:
                            sol_fwd = new_node
                            sol_bwd = tmp_node
                        else:
                            sol_fwd = tmp_node
                            sol_bwd = new_node
                            
                        rrt_sol.insert(i, sol_fwd, sol_bwd)
                        print('i = {}, sol_fwd = {}, sol_bwd = {}, goal_dist = {}'.format(i, sol_fwd.state.pos, sol_bwd.state.pos, sol_fwd.dist(sol_bwd)))
                        break                                                   # If `rrt_sol.insert(i)` is executed in `Direction.FORWARD`,
                                                                                # there is no need to execute it in `Direction.BACKWARD`.
                        
            rrt_round += 1
            
        sol_n_arr = rrt_sol.get_flattened_sol()
        ptraj = self._recover_planned_traj_from_sol(sol_n_arr)
        return ptraj, rrt_sol.finished()