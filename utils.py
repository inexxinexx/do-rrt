import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import interpolate

import torch
import random

def set_seed(seed):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Compute the 2D cross product of vectors OA and OB, where points are projected to the XY plane
# This helps to determine the relative orientation of the points
def _cross_product(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

# Check if point p3 lies on segment (p1, p2) in the XY plane
def _point_on_segment(p1, p2, p3):
    # Check if p3 is collinear with (p1, p2) by verifying the cross product is zero
    cross_prod = _cross_product(p1, p2, p3)
    if abs(cross_prod) > 1e-6:                                              # not collinear
        return False

    # Check if p3 is within the bounding box of (p1, p2)
    return min(p1[0], p2[0]) <= p3[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p3[1] <= max(p1[1], p2[1])

# Check if two line segments (p1, p2) and (p3, p4) intersect in the XY plane
def _segments_intersect(p1, p2, p3, p4):
    d1 = _cross_product(p3, p4, p1)
    d2 = _cross_product(p3, p4, p2)
    d3 = _cross_product(p1, p2, p3)
    d4 = _cross_product(p1, p2, p4)

    # Check if the line segments straddle each other (cross each other)
    # return (d1 * d2 < 0) and (d3 * d4 < 0)
    if (d1 * d2 < 0) and (d3 * d4 < 0):
        return True

    # Check if any point of one segment lies on the other segment
    return (_point_on_segment(p1, p2, p3) or _point_on_segment(p1, p2, p4) or
            _point_on_segment(p3, p4, p1) or _point_on_segment(p3, p4, p2))

# Remove the intersecting segments from a 3D trajectory
def traj_prune(ptraj):
    ptraj2 = []
    ptraj_sz = len(ptraj)
    
    i = 0
    while i < ptraj_sz:
        ptraj2.append(ptraj[i])
        
        for j in range(i + 2, ptraj_sz - 1):
            if _segments_intersect(ptraj[i], ptraj[i+1], ptraj[j], ptraj[j+1]):
                i = j
                break
        else:
            i += 1
            
    return ptraj2

# Compute distances between consecutive points
def _compute_distances(ptraj):
    distances = np.sqrt(np.sum(np.diff(ptraj, axis=0)**2, axis=1))
    return np.concatenate([[0], np.cumsum(distances)])                      # cumulative distance

# Resample trajectory based on cumulative distance
def _resample_traj(ptraj, num_samples=100):
    distances = _compute_distances(ptraj)
    total_length = distances[-1]
    
    # Generate equally spaced distances along the trajectory
    equal_distances = np.linspace(0, total_length, num_samples)
    
    # Interpolate x, y, z coordinates based on the equal distances
    x = np.interp(equal_distances, distances, ptraj[:, 0])
    y = np.interp(equal_distances, distances, ptraj[:, 1])
    z = np.interp(equal_distances, distances, ptraj[:, 2])
    
    return np.vstack((x, y, z)).T

# Due to the lack of action information, we perform trajectory smoothing rather than trajectory optimization.
def traj_smooth(ptraj, num_smooth_states=100):                              # ptraj = [np.array([x,y,z]), ...]
    ptraj = np.asarray(ptraj)
    
    ptraj = _resample_traj(ptraj, num_samples=num_smooth_states)
    
    x = ptraj[:,0]
    y = ptraj[:,1]
    z = ptraj[:,2]
    
    # Manually set parameter values to ensure the endpoints are fixed
    u = np.linspace(0, 1, len(ptraj))
    tck, _ = interpolate.splprep([x, y, z], u=u, s=2, k=3)                  # `s` controls the smoothness; the larger the value, the stronger the smoothing effect.
    
    unew = np.linspace(0, 1, num_smooth_states)                             # generate `num_smooth_states` interpolation points
    x_smooth, y_smooth, z_smooth = interpolate.splev(unew, tck)
    
    ptraj_smooth = np.vstack((x_smooth, y_smooth, z_smooth)).T
    return ptraj_smooth

def plot_traj(result_path, ptraj, end_point_pairs=None, key_points=None, color='r', save=False):
    ptraj = np.asarray(ptraj)
    
    plt.plot(ptraj[:, 0], ptraj[:, 1], color=color)
    
    plt.axis('equal')
    plt.grid(True)
    plt.xlim([-12, 12])
    plt.ylim([-12, 12])
    
    # Mark end_point_pairs
    if end_point_pairs:
        plt.scatter(end_point_pairs[0], end_point_pairs[1], color='red', marker='o')  # start_pos
        plt.scatter(end_point_pairs[3], end_point_pairs[4], color='blue', marker='o') # end_pos
        
    # Mark key_points
    if key_points:
        for pos in key_points:
            plt.scatter(pos[0], pos[1], color='k', marker='x')
            
    if save:
        plt.savefig(os.path.join(result_path, 'rrt.png'))
    else:
        plt.show()
        
def plot_trajs(result_path, ptrajs, save=False, key_points=None):   
    for ptraj in ptrajs:
        ptraj = np.asarray(ptraj)
        
        plt.plot(ptraj[:, 0], ptraj[:, 1], color='gray', marker='|', markersize=10, linestyle='-')
        
        # Highlight the first and last points
        plt.scatter(ptraj[0, 0], ptraj[0, 1], color='g')
        plt.scatter(ptraj[-1, 0], ptraj[-1, 1], color='r')

    plt.axis('equal')
    plt.grid(True)
    plt.xlim([-12, 12])
    plt.ylim([-12, 12])
    
    # Mark key_points
    if key_points:
        for pos in key_points:
            plt.scatter(pos[0], pos[1], color='k', marker='x')

    if save:
        plt.savefig(os.path.join(result_path, '__model_test_2.png'))
    else:
        plt.show()
        
if __name__ == '__main__':
    T = 10
    ptraj = np.random.rand(T, 3) * 10 - 5                                   # range in [-5, 5]^3
    plot_traj('', ptraj)