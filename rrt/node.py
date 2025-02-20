import numpy as np

class State():
    __slots__ = ['pos']
    def __init__(self, pos : np.ndarray):
        if pos.size == 3:
            self.pos = pos.astype(np.float32) if pos.dtype != np.float32 else pos
        else:
            print(f'Error: Wrong pos {pos}!')
            
    @classmethod
    def from_xyz(cls, x, y, z):
        return cls(np.array([x, y, z], dtype=np.float32))
    
    def dist(self, pos : np.ndarray):
        return np.linalg.norm(self.pos - pos)
        
class ShortTraj():
    __slots__ = ['states', 'size']
    def __init__(self, size):
        self.size = size
        self.states = []
        
    def add(self, s : State):
        if isinstance(s, State) and len(self.states) < self.size:
            self.states.append(s)
            
    def front(self):
        return self.states[0]
    
    def back(self):
        return self.states[-1]
    
    def __reversed__(self):
        return reversed(self.states)

class Node():
    __slots__ = ['state', 'parent', 'traj']
    def __init__(self, s: 'State' = None, p: 'Node' = None, traj: 'ShortTraj' = None):
        if s is not None:
            self.state = s
        else:
            self.state = State.from_xyz(0, 0, 0)
        
        self.parent = p
        self.traj = traj
        
    def dist(self, n : 'Node' = None):
        return np.linalg.norm(self.state.pos - n.state.pos)
    
    def dist_pos(self, x : np.ndarray):
        return np.linalg.norm(self.state.pos - x)