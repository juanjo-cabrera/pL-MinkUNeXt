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


envs_360loc = ["atrium", "concourse", "hall", "piatrium"]
trainSeq_360loc = ["daytime_360_0", "daytime_360_1", "daytime_360_0", "daytime_360_2"]
condIlum_360loc = [["daytime_360_1", "daytime_360_2", "nighttime_360_1", "nighttime_360_2"],
                     ["daytime_360_0", "daytime_360_2", "nighttime_360_0"],
                     ["daytime_360_1", "daytime_360_2", "nighttime_360_1", "nighttime_360_2"],
                    ["daytime_360_0", "daytime_360_1", "nighttime_360_0"]]


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

def construct_query_and_database_sets(base_path, df_test, df_database, output_name, distance_threshold=10.0):    

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
        index, _ = database_tree.query_radius(coor, r=distance_threshold, sort_results=True, return_distance=True)
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



def read_coords_txt(env="atrium", ilum="daytime_360_0", imgSet="database"):
    
    if imgSet == "database":
        txtName = f'{PARAMS.dataset_folder }{env}/pose/360_mapping_gt.txt'
    else:
        txtName = f'{PARAMS.dataset_folder }{env}/pose/query_gt_360_{ilum.split("time")[0]}.txt'

    with open(txtName, 'r') as file:
        lines = file.readlines()
    coords = {}
    for line in lines:
        line = line.split()
        imgPath, seqIdx = line[0], line[0].split("/")[-3].split("_")[-1]
        if seqIdx != ilum[-1] and imgSet != "database":
            continue
        coordX, coordY, coordZ = float(line[1]), float(line[2]), float(line[3])
        coords[imgPath] = (coordX, coordY, coordZ)
    df_locations = pd.DataFrame({ 'file': list(coords.keys()), 'x': [coords[k][0] for k in coords], 'y': [coords[k][1] for k in coords], 'z': [coords[k][2] for k in coords]})
    return df_locations


def build_header_results_csv(params):
    header = params + ["R@1 atrium day", "R@1 atrium night", "R@1 atrium avg",
                       "R@1 concourse day", "R@1 concourse night", "R@1 concourse avg",
                       "R@1 hall day", "R@1 hall night", "R@1 hall avg",
                       "R@1 piatrium day", "R@1 piatrium night", "R@1 piatrium avg",
                       "R@1 Global Avg.",
                       "R@N atrium day", "R@N atrium night", "R@N atrium avg",
                       "R@N concourse day", "R@N concourse night", "R@N concourse avg",
                       "R@N hall day", "R@N hall night", "R@N hall avg",
                       "R@N piatrium day", "R@N piatrium night", "R@N piatrium avg",
                       "R@N Global Avg."]
    return header

def build_row_results_csv(params, r1_list, rn_list):
    row = params + [f"{x:.2f}" for x in r1_list] + [f"{x:.2f}" for x in rn_list]
    return row




if __name__ == '__main__':
    PARAMS.dataset_folder = '/media/arvc/DATOS1/Marcos/DATASETS/360LOC/'
    pcd_folder = 'PCD_DEPTH_ANY_PANORAMAS_LARGE/'
    for env in envs_360loc:
        database_locations = read_coords_txt(env=env, ilum=trainSeq_360loc[envs_360loc.index(env)], imgSet="database")
        # replace root path from database_locations file paths 'image' to 'PCD_DISTILL_ANY_DEPTH_LARGE'
        database_locations['file'] = PARAMS.dataset_folder + env + '/' + database_locations['file'].str.replace('image', pcd_folder)
        # change the extension from .jpg to .ply
        database_locations['file'] = database_locations['file'].str.replace('.jpg', '.ply')

        for ilum in condIlum_360loc[envs_360loc.index(env)]:
            query_locations = read_coords_txt(env=env, ilum=ilum, imgSet="query")
            # replace root path from query_locations file paths 'image' to 'PCD_DISTILL_ANY_DEPTH_LARGE'
            query_locations['file'] = PARAMS.dataset_folder + env + '/' + query_locations['file'].str.replace('image', pcd_folder)
            # change the extension from .jpg to .ply
            query_locations['file'] = query_locations['file'].str.replace('.jpg', '.ply')
            print(f"Environment: {env} - Illumination: {ilum} - Database size: {len(database_locations)} - Query size: {len(query_locations)}")
            if env == "concourse":
                d = 5.0
            else:
                d = 10.0
            construct_query_and_database_sets(PARAMS.dataset_folder + f'/{env}/', query_locations, database_locations, output_name=f'{env}_{ilum}_360loc', distance_threshold=d)



    # database_folder = 'Train' 
    # print('Dataset root: {}'.format(PARAMS.dataset_folder))
    # base_path = PARAMS.dataset_folder
    # database_dir = os.path.join(base_path, database_folder)
    # cloudy_dir = os.path.join(base_path, 'TestCloudy')
    # night_dir = os.path.join(base_path, 'TestNight')
    # sunny_dir = os.path.join(base_path, 'TestSunny')
    # all_room_folders = sorted(os.listdir(os.path.join(base_path, database_folder)))
    # database_locations = get_pointcloud_positions(database_dir, all_room_folders)
    # cloudy_locations = get_pointcloud_positions(cloudy_dir, all_room_folders)
    # night_locations = get_pointcloud_positions(night_dir, all_room_folders)
    # sunny_locations = get_pointcloud_positions(sunny_dir, all_room_folders)


    # print("Number of testing pointclouds for cloudy " +  str(len(cloudy_locations['file'])))
    # construct_query_and_database_sets(base_path,  cloudy_locations, database_locations, output_name='cloudy')

    # print("Number of testing pointclouds for night " +  str(len(night_locations['file'])))
    # construct_query_and_database_sets(base_path,  night_locations, database_locations, output_name='night')

    # print("Number of testing pointclouds for sunny " +  str(len(sunny_locations['file'])))
    # construct_query_and_database_sets(base_path,  sunny_locations, database_locations, output_name='sunny')


