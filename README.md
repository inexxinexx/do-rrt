# Safety-Compliant Navigation: Navigation Point-Guided Planning with Primitive Trajectories

### File Overview
- stats.py
    - Provides visualization and statistical analysis of the training set (`datasets/training_set`).
    - Can be used to determine navigation points.
- main.py
    - Uncomment `model_training(args)` to train GNext networks, which are saved in `model_ws/`.
    - Supports single trajectory planning with a given sequence of navigation points (`key_points`) and a given discontinuity bound δ.
- main_eval.py
    - Performs batch trajectory planning for the start and goal positions in the test set (`datasets/test_set`).
    - Allows adaptive determination of δ.
    - Results are stored in `results/main_eval_res.json`.
- rrt/
    - DAMP-RRT-Connect algorithm
- short_traj_gen/
    - GNext networks
    - `GenerateTrajs` function for generating primitive trajectories
- plot.py
    - Visualizes batch planning results from `results/main_eval_res.json`.
- utils.py
    - utility functions

### Usage
1. Analyze training data: Use `stats.py` for dataset visualization and statistical analysis, which can help identify navigation points.
2. Train GNext networks: Uncomment and run `model_training(args)` in `main.py`.
3. Single trajectory planning: Run `main.py` with a specified `key_points` and a fixed δ.
4. Batch trajectory planning: Run `main_eval.py`.
5. Visualize batch results: Run `plot.py`.