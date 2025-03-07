# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config import PARAMS
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import tqdm


from datasets.base_datasets import TrainingTuple


def construct_query_dict(df_centroids, base_path, filename, ind_nn_r, ind_r_r=50):
    # ind_nn_r: threshold for positive examples
    # ind_r_r: threshold for negative examples
    # Baseline dataset parameters in the original PointNetVLAD code: ind_nn_r=10, ind_r=50
    # Refined dataset parameters in the original PointNetVLAD code: ind_nn_r=12.5, ind_r=50
    tree = KDTree(df_centroids[['x', 'y']])
    ind_nn = tree.query_radius(df_centroids[['x', 'y']], r=ind_nn_r)
    ind_r = tree.query_radius(df_centroids[['x', 'y']], r=ind_r_r)
    queries = {}
    for anchor_ndx in range(len(ind_nn)):
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['x', 'y']])
        query = df_centroids.iloc[anchor_ndx]["file"]
        timestamp = df_centroids.iloc[anchor_ndx]["timestamp"]

        positives = ind_nn[anchor_ndx]
        non_negatives = ind_r[anchor_ndx]

        positives = positives[positives != anchor_ndx]
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                            positives=positives, non_negatives=non_negatives, position=anchor_pos)

    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


def get_pointcloud_positions(folder_dir, folders):
    # given a folder, return the positions of the pointclouds
    # each pointcloud file name is the timestamp, the 'x', 'y' and 'a' orientation, for example file_pathname = t1152904768.768371_x-8.640943_y2.861793_a-0.209387.ply
    timestamps = []
    x_positions = []
    y_positions = []
    orientations = []
    files_names = []
    for folder in folders:
        room_dir = os.path.join(folder_dir, folder)
        # check if the folder is a directory    
        if not os.path.isdir(room_dir):
            continue
        for file in os.listdir(room_dir):
            if file.endswith(".ply"):        
                files_names.append(folder + '/' +file)
                # quitar la extension del archivo
                file = file[:-4]
                timestamp_index = file.index('t')
                x_index = file.index('_x')
                y_index = file.index('_y')
                a_index = file.index('_a')
                timestamp = file[timestamp_index+1:x_index]         


                x = file[x_index+2:y_index]
                y = file[y_index+2:a_index]
                a = file[a_index+2:]
                # x, y, a are strings, parse them to float
                x = float(x)
                y = float(y)
                a = float(a)
                timestamp = float(timestamp)

                timestamps.append(timestamp)
                x_positions.append(x)
                y_positions.append(y)
                orientations.append(a)
    df_locations = pd.DataFrame({ 'file': files_names, 'timestamp': timestamps, 'x': x_positions, 'y': y_positions, 'orientation': orientations})
    return df_locations


def generate_pickle(run_folder, pickle_filename):
    print('Dataset root: {}'.format(PARAMS.dataset_folder))
    base_path = PARAMS.dataset_folder

    all_room_folders = sorted(os.listdir(os.path.join(base_path, run_folder)))
   
    folder_dir = os.path.join(base_path, run_folder)
    df_locations = get_pointcloud_positions(folder_dir, all_room_folders)


    print("Number of pointclouds: " + str(len(df_locations['file'])))

    construct_query_dict(df_locations, base_path + run_folder, pickle_filename, ind_nn_r=PARAMS.positive_distance, ind_r_r=PARAMS.negative_distance)

if __name__ == '__main__':
    PARAMS.train_folder = 'TrainingBaseline2/'
    TRAIN_FOLDER = PARAMS.train_folder
    VAL_FOLDER = PARAMS.val_folder
    generate_pickle(TRAIN_FOLDER, "training_queries_baseline2.pickle")
    generate_pickle(VAL_FOLDER, "validation_queries_baseline.pickle")




