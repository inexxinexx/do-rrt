class RRTSolution():
    __slots__ = ['sol_npairs_arr', 'size']
    def __init__(self, sz : int):
        self.sol_npairs_arr = [[] for _ in range(sz)]
        self.size = sz
    
    def finished(self):
        return all(len(sol_npairs) > 0 for sol_npairs in self.sol_npairs_arr)
    
    def sol_npairs_found(self, sol_npairs_i : int):
        return len(self.sol_npairs_arr[sol_npairs_i]) > 0
    
    def insert(self, sol_npairs_i : int, fwd_n, bwd_n):
        if sol_npairs_i < 0 or sol_npairs_i >= self.size or self.sol_npairs_found(sol_npairs_i):
            print(f'Error: Cannot insert sol_npairs_arr at index {sol_npairs_i}!')
        else:
            self.sol_npairs_arr[sol_npairs_i].append(fwd_n)
            self.sol_npairs_arr[sol_npairs_i].append(bwd_n)
            
    # flatten `[[fwd_n, bwd_n], [fwd_n, bwd_n], ...]` to `[fwd_n, bwd_n, fwd_n, bwd_n, ...]`
    def get_flattened_sol(self):
        sol_n_arr = [x for sol_npairs in self.sol_npairs_arr for x in sol_npairs]
        return sol_n_arr