# read and plot trajectories from rawseeds dataset

# GROUNDTRUTH:
# - Timestamp [seconds.microseconds]
# - X [m]
# - Y [m]
# - Theta [rad]
# - C_xx
# - C_xy
# - C_xt
# - C_yx
# - C_yy
# - C_yt
# - C_tx
# - C_ty
# - C_tt

# The C are the entries of the covariance matrix. 


base_dataset = '/media/arvc/DATOS1/Juanjo/Datasets/rawseeds'
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path





def plot_trajectories(dataset_path, sequences):
    plt.figure(figsize=(10, 8))
    for seq in sequences:
        seq_path = os.path.join(dataset_path, seq)
        # look for the file with extension .csv
        files = os.listdir(seq_path)
        gt_file = [f for f in files if f.endswith('.csv')][0]
        
        gt_file = os.path.join(seq_path, gt_file)
        
        # Load groundtruth data
        # data = pd.read_csv(gt_file, delim_whitespace=True, header=None,
        #                    names=['timestamp', 'x', 'y', 'theta', 
        #                           'C_xx', 'C_xy', 'C_xt', 
        #                           'C_yx', 'C_yy', 'C_yt', 
        #                           'C_tx', 'C_ty', 'C_tt'])

        data = pd.read_csv(gt_file)
        # add a header
        data.columns = ['timestamp', 'x', 'y', 'theta', 
                        'C_xx', 'C_xy', 'C_xt', 
                        'C_yx', 'C_yy', 'C_yt', 
                        'C_tx', 'C_ty', 'C_tt']
        
        # Plot trajectory
        plt.plot(data['x'], data['y'], label=f'Trajectory_{seq}', color=np.random.rand(3,))
        plt.scatter(data['x'].iloc[0], data['y'].iloc[0], color='green', marker='o', label='Start')
        plt.scatter(data['x'].iloc[-1], data['y'].iloc[-1], color='red', marker='x', label='End')

    plt.title(f'Trajectory for Sequence {seq}')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    sequences = os.listdir(base_dataset)
    # filter only directories
    sequences = [seq for seq in sequences if os.path.isdir(os.path.join(base_dataset, seq))]
    plot_trajectories(base_dataset, sequences)