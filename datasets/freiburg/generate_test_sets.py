# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import os
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config import PARAMS


RUNS_FOLDERS = ['TestCloudy', 'TestNight', 'TestSunny']


def output_to_file(output, base_path, filename):
    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
                files_names.append(room_dir + '/' +file)
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

def construct_query_and_database_sets(base_path, df_test, df_database, output_name):    

    database_tree = KDTree(np.array(df_database[['x', 'y']]))

    database = {}
    test = {}

    for index, row in df_database.iterrows():
        database[len(database.keys())] = {'query': row['file'], 'x': row['x'],
                                            'y': row['y']}
        
    for index, row in df_test.iterrows():
        test[len(test.keys())] = {'query': row['file'], 'x': row['x'],
                                            'y': row['y']}
    #distances = []
    for key in range(len(test.keys())):
        coor = np.array([[test[key]["x"], test[key]["y"]]])
        # query the database tree for the nearest point cloud
        index = database_tree.query_radius(coor, r=0.5)
        #distances.append(index[0])    
        # change the shape from (n, ) to (1, n) 
        index = np.array([index[0]])   
        
        # indices of the positive matches in database i of each query (key) in test set j
        test[key][0] = index.tolist()
    
    #print("Mean of the minimum distance: ", np.mean(distances))
    output_to_file(database, base_path, output_name + '_evaluation_database.pickle')
    output_to_file(test, base_path, output_name + '_evaluation_query.pickle')

def generate_test_pickle(base_path):
    database_dir = os.path.join(base_path, 'Train')
    cloudy_dir = os.path.join(base_path, 'TestCloudy')
    night_dir = os.path.join(base_path, 'TestNight')
    sunny_dir = os.path.join(base_path, 'TestSunny')
    all_room_folders = sorted(os.listdir(os.path.join(base_path, 'Train')))
    database_locations = get_pointcloud_positions(database_dir, all_room_folders)
    cloudy_locations = get_pointcloud_positions(cloudy_dir, all_room_folders)

    print("Number of testing pointclouds for cloudy " +  str(len(cloudy_locations['file'])))
    construct_query_and_database_sets(base_path,  cloudy_locations, database_locations, output_name='cloudy')
    if os.path.exists(night_dir):
        night_locations = get_pointcloud_positions(night_dir, all_room_folders)
        print("Number of testing pointclouds for night " +  str(len(night_locations['file'])))        
        construct_query_and_database_sets(base_path,  night_locations, database_locations, output_name='night')

    if os.path.exists(sunny_dir):
        sunny_locations = get_pointcloud_positions(sunny_dir, all_room_folders)
        print("Number of testing pointclouds for sunny " +  str(len(sunny_locations['file'])))
        construct_query_and_database_sets(base_path,  sunny_locations, database_locations, output_name='sunny')

if __name__ == '__main__':
    PARAMS.dataset_folder = '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_LARGE/SAARBRUCKEN_A/'
    database_folder = 'Train' 
    print('Dataset root: {}'.format(PARAMS.dataset_folder))
    base_path = PARAMS.dataset_folder
    database_dir = os.path.join(base_path, database_folder)
    cloudy_dir = os.path.join(base_path, 'TestCloudy')
    night_dir = os.path.join(base_path, 'TestNight')
    sunny_dir = os.path.join(base_path, 'TestSunny')
    all_room_folders = sorted(os.listdir(os.path.join(base_path, database_folder)))
    database_locations = get_pointcloud_positions(database_dir, all_room_folders)
    cloudy_locations = get_pointcloud_positions(cloudy_dir, all_room_folders)
    night_locations = get_pointcloud_positions(night_dir, all_room_folders)
    sunny_locations = get_pointcloud_positions(sunny_dir, all_room_folders)


    print("Number of testing pointclouds for cloudy " +  str(len(cloudy_locations['file'])))
    construct_query_and_database_sets(base_path,  cloudy_locations, database_locations, output_name='cloudy')

    print("Number of testing pointclouds for night " +  str(len(night_locations['file'])))
    construct_query_and_database_sets(base_path,  night_locations, database_locations, output_name='night')

    print("Number of testing pointclouds for sunny " +  str(len(sunny_locations['file'])))
    construct_query_and_database_sets(base_path,  sunny_locations, database_locations, output_name='sunny')


