from short_traj_gen.dataset_loader import TrajectoryDataset

import numpy as np
from collections import defaultdict
import pandas as pd
import plotly.graph_objects as go

class TrajectoryGridAnalyzer:
    def __init__(self, folder_path, result_path, grid_size=0.1):
        self.folder_path = folder_path
        self.result_path = result_path
        self.grid_size = grid_size
        
        self.seq_list = self.load_seq_list(folder_path, result_path)
        self.grid_counts = None

    def load_seq_list(self, folder_path, result_path):
        dataset = TrajectoryDataset(folder_path, result_path)
        return dataset.data
    
    def _compute_full_grid_counts(self):
        grid_counts = defaultdict(int)
        
        for traj in self.seq_list:
            # Convert each trajectory point xyz to grid cell index
            traj_grid = np.floor(traj / self.grid_size).astype(int)
            
            # # Remove duplicate grid points in each trajectory
            # traj_grid = np.unique(traj_grid, axis=0)
            
            # Count each unique grid cell that the trajectory passes through
            for grid_point in traj_grid:
                grid_counts[tuple(grid_point)] += 1
                
        self.grid_counts = grid_counts
        return grid_counts
    
    def _filter_grid_counts(self, x_range=None, y_range=None, z_range=None):
        if self.grid_counts is None:
            raise ValueError('Grid counts have not been computed. Call compute_full_grid_counts() first.')
        
        filtered_counts = {}
        
        for grid_point, count in self.grid_counts.items():
            x, y, z = grid_point
            
            # Convert grid index back to real-world coordinates (grid center)
            real_x = (x + 0.5) * self.grid_size
            real_y = (y + 0.5) * self.grid_size
            real_z = (z + 0.5) * self.grid_size
            
            if (x_range is None or (x_range[0] <= real_x <= x_range[1])) and \
               (y_range is None or (y_range[0] <= real_y <= y_range[1])) and \
               (z_range is None or (z_range[0] <= real_z <= z_range[1])):
                filtered_counts[(real_x, real_y, real_z)] = count
        
        sorted_counts = sorted(filtered_counts.items(), key=lambda item: item[1], reverse=True)
        return sorted_counts
    
    def compute_and_filter_counts(self, x_range=None, y_range=None, z_range=None):
        if self.grid_counts is None:
            self._compute_full_grid_counts()
        
        filtered_sorted_counts = self._filter_grid_counts(x_range=x_range, y_range=y_range, z_range=z_range)
        
        return filtered_sorted_counts
    
    def display_filtered_counts(self, counts, top=None):
        df = pd.DataFrame(counts, columns=['Grid Center (X, Y, Z)', 'Count'])
        precision = abs(int(np.log10(self.grid_size)))
        df['Grid Center (X, Y, Z)'] = df['Grid Center (X, Y, Z)'].apply(
            lambda x: f'({x[0]:.{precision}f}, {x[1]:.{precision}f}, {x[2]:.{precision}f})'
        )
        df_sorted = df.sort_values(by='Count', ascending=False)
        if top is not None:
            df_sorted = df_sorted.head(top)
        print(df_sorted.to_string(index=False))
    
    def plot_trajectories_3d(self, fraction=1.0, save=False, filename='trajs_plot.html'):
        fig = go.Figure()
        
        num_trajs = len(self.seq_list)
        num_samples = max(int(fraction * num_trajs), 1)
        sampled_indices = np.random.choice(num_trajs, num_samples, replace=False)
        
        for i in sampled_indices:
            traj = self.seq_list[i]
            x = traj[:, 0]
            y = traj[:, 1]
            z = traj[:, 2]
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name=f'traj {i}'))
            
        fig.update_layout(
            title='3D Trajectories',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'                           # Make the proportions of the x, y, and z axes equal.
            )
        )
        
        if save:
            save_path = os.path.join(self.result_path, filename)
            fig.write_html(save_path)
        else:
            fig.show()
    
import json
from collections import Counter

def analyze_delta_distribution(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    delta_counter = Counter()
    total_cnt = 0
    for entry in data:
        delta = round(entry['delta'], 1)
        delta_counter[delta] += 1
        total_cnt += 1
        
    print('Delta dist:')
    for delta, cnt in sorted(delta_counter.items()):
        print(f'Delta {delta}: {cnt} occurrences, {cnt / total_cnt:.2f}')
        
    # # Filtered Count
    # filtered_delta_counter = Counter()
    # filtered_total_cnt = 0
    # for entry in data:
    #     epp = entry['epp']
    #     if round(epp[2], 1) != 0.2 or round(epp[5], 1) != 0.2:
    #         delta = round(entry['delta'], 1)
    #         filtered_delta_counter[delta] += 1
    #         filtered_total_cnt += 1

    # print('\nFiltered delta dist (both epp z-values not 0.2):')
    # for delta, cnt in sorted(filtered_delta_counter.items()):
    #     print(f'Delta {delta}: {cnt} occurrences, {cnt / filtered_total_cnt:.2f}')
        
    # print(f'\nFiltered count over total count: {filtered_total_cnt / total_cnt:.2f}')
    
if __name__ == '__main__':
    folder_path = 'datasets/training_set'
    result_path = 'Temp'
    # grid_size = 0.1
    grid_size = 0.01
    analyzer = TrajectoryGridAnalyzer(folder_path=folder_path, result_path=result_path, grid_size=grid_size)
    
    # # x_range = None
    # # y_range = None
    # # z_range = None
    # # x_range = (-1.5, 1.5)
    # # y_range = (-1.5, 1.5)
    # # z_range = (0, 0.1)
    # x_range = (-0.2, 2.3)
    # y_range = (0.5, 6)
    # z_range = (0.2, 0.5)
    # all_counts = analyzer.compute_and_filter_counts()
    # filtered_counts = analyzer.compute_and_filter_counts(x_range=x_range, y_range=y_range, z_range=z_range)
    # # print('filtered_counts =\n\t', filtered_counts)
    # analyzer.display_filtered_counts(filtered_counts, top=50)
    
    analyzer.plot_trajectories_3d(fraction=0.05)
    
    ##################################################
    # analyze_delta_distribution('results\main_eval_res.json')