"""
THIS SCRIPT CONTAINS FUNCTIONS THAT CREATE FIGURES THAT GRAPHICALLY REPRESENT THE TEST RESULTS OF A LOCALIZATION STAGE
Among these graphics, we can find:
- Maps with the network predictions (hierarchical and global localization)
Test script will call these functions
"""


# read csv file with pandas 
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config.config import PARAMS


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


def get_axes_limits(coordX, coordY, xmax, xmin, ymax, ymin):
    if coordX < xmin:
        xmin = coordX
    if coordX > xmax:
        xmax = coordX
    if coordY < ymin:
        ymin = coordY
    if coordY > ymax:
        ymax = coordY
    return xmax, xmin, ymax, ymin



# The maps represent the network prediction for every test image:
# Blue points represent the capture points of the images from the visual model
# The rest of points represent the coordinates of the test images
# Test images will be colored:
# Green: if they are retrieved among k=1
# Yellow/Orange: if they are retrieved among k~5-15
# Red: if they are not retrieved among k=20
def display_coord_map(df, df_database):
    # df header 'query_image', 'query_x', 'query_y', 'retrieved_database_image', 'retrieved_database_x', 'retrieved_database_y', 'real_database_image', 'real_database_x', 'real_database_y', 'recall@1', 'recall@1%'
    # df_database header 'file', 'timestamp', 'x', 'y', 'orientation'
    # plt tkagg
    plt.switch_backend('tkagg')    
    

    
    xmin, xmax, ymin, ymax = 1000, -1000, 1000, -1000
    plt.figure(figsize=(9, 6), dpi=120, edgecolor='black')

    firstk1, firstErrork, firstErrorRoom = True, True, True
    # get the coordinates of the visual model
    mapVM = df_database[['x', 'y']].to_numpy()
    plt.scatter(mapVM[:, 0], mapVM[:, 1], color='blue', label="Visual Model")
    xmax, xmin, ymax, ymin = get_axes_limits(mapVM[0][0], mapVM[0][1], xmax, xmin, ymax, ymin)

    # get the coordinates of the test images
    mapTest = df[['query_x', 'query_y', 'retrieved_database_x', 'retrieved_database_y', 'recall@1', 'recall@1%']].to_numpy()
    # get the coordinates of the real database images
    mapReal = df[['real_database_x', 'real_database_y']].to_numpy()
    



    for t in range(len(mapTest)):
        # si el recall@1 es 1, el color es verde
        if mapTest[t][4] == 1:
            if firstk1:
                plt.scatter(mapTest[t][0], mapTest[t][1], color='green', label='Recall@1 prediction')
                firstk1 = False
            else:
                plt.scatter(mapTest[t][0], mapTest[t][1], color='green')
                plt.plot([mapTest[t][0], mapTest[t][2]], [mapTest[t][1], mapTest[t][3]], color='green')
            xmax, xmin, ymax, ymin = get_axes_limits(mapTest[t][2], mapTest[t][3], xmax, xmin, ymax, ymin)
        # si el recall@1 es 0 y el recall@1% es 1, el color es amarillo
        elif mapTest[t][4] == 0 and mapTest[t][5] == 1:
            if firstErrork:
                plt.scatter(mapTest[t][0], mapTest[t][1], color='orange', label='Recall@1% prediction')
                firstErrork = False
            else:
                plt.scatter(mapTest[t][0], mapTest[t][1], color='orange')
                plt.plot([mapTest[t][0], mapTest[t][2]], [mapTest[t][1], mapTest[t][3]], color='orange')
            xmax, xmin, ymax, ymin = get_axes_limits(mapTest[t][2], mapTest[t][3], xmax, xmin, ymax, ymin)
        # si el recall@1 es 0 y el recall@1% es 0, el color es rojo
        elif mapTest[t][4] == 0 and mapTest[t][5] == 0:
            if firstErrorRoom:
                plt.scatter(mapTest[t][0], mapTest[t][1], color='red', label='Predictions not among Recall@1 and Recall@1%')
                firstErrorRoom = False
            else:
                plt.scatter(mapTest[t][0], mapTest[t][1], color='red')
                plt.plot([mapTest[t][0], mapTest[t][2]], [mapTest[t][1], mapTest[t][3]], color='red')
            xmax, xmin, ymax, ymin = get_axes_limits(mapTest[t][2], mapTest[t][3], xmax, xmin, ymax, ymin)



    # plt.axis([xmin-0.5, xmax+0.5, ymin-0.25, ymax+0.25])
    # plt.ylabel('y (m)', fontsize=18)
    # plt.xlabel('x (m)', fontsize=18)
    # plt.suptitle('Hierarchical localization', fontsize=24)
    # plt.title(f'Loss function: {loss}, Illumination: {ilum}', fontsize=20)
    # plt.legend(fontsize=14)
    # plt.grid()
    # plt.savefig(os.path.join(direc, "map_" + sl + "_" + ilum + ".png"), dpi=400)
    # plt.close()

    plt.axis([xmin-0.5, xmax+0.5, ymin-0.25, ymax+0.25])
    plt.ylabel('y (m)', fontsize=18)
    plt.xlabel('x (m)', fontsize=18)
    plt.title('Pseudo-LiDAR PR', fontsize=24)
    plt.legend(fontsize=14)
    plt.grid()
    # save the figure in the same folder as the csv file
    plt.show()
    print('Figure saved in: {}'.format(os.path.join(os.path.dirname(df['query_image'][0]), 'map.png')))




if __name__ == "__main__":

    dataset_path  = '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_DISTILL_ANY_DEPTH_LARGE/FRIBURGO_A/'
    # dataset_path  = '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_DISTILL_ANY_DEPTH_LARGE/FRIBURGO_B/'
    # dataset_path  = '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_DISTILL_ANY_DEPTH_LARGE/SAARBRUCKEN_A/'
    # dataset_path = '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_DISTILL_ANY_DEPTH_LARGE/SAARBRUCKEN_B/'
   
    cloudy_query_path = dataset_path + 'TestCloudy/'
    night_query_path = dataset_path + 'TestNight/'
    sunny_query_path = dataset_path + 'TestSunny/'
    if not os.path.exists(cloudy_query_path):
        # elude cloudy information if it does not exist
        cloudy_query_path = None
    if not os.path.exists(night_query_path):
        # elude night information if it does not exist
        night_query_path = None
    if not os.path.exists(sunny_query_path):
        # elude sunny information if it does not exist
        sunny_query_path = None
    
    if cloudy_query_path is not None:
        print('Cloudy query path: {}'.format(cloudy_query_path))
        save_csv_path_cloudy = os.path.join(cloudy_query_path, 'results.csv')
    if night_query_path is not None:
        print('Night query path: {}'.format(night_query_path))
        save_csv_path_night = os.path.join(night_query_path, 'results.csv')
    if sunny_query_path is not None:
        print('Sunny query path: {}'.format(sunny_query_path))
        save_csv_path_sunny = os.path.join(sunny_query_path, 'results.csv')

  
    database_folder = 'Train' 
    base_path = dataset_path
    database_dir = os.path.join(base_path, database_folder)
    all_room_folders = sorted(os.listdir(os.path.join(base_path, database_folder)))

    df_database = get_pointcloud_positions(database_dir, all_room_folders)
   
    # read csv file with pandas    
    # header 'query_image', 'query_x', 'query_y', 'retrieved_database_image', 'retrieved_database_x', 'retrieved_database_y', 'real_database_image', 'real_database_x', 'real_database_y', 'recall@1', 'recall@1%'
    df_cloudy = pd.read_csv(save_csv_path_cloudy)
    df_night = pd.read_csv(save_csv_path_night)
    df_sunny = pd.read_csv(save_csv_path_sunny)

    display_coord_map(df_cloudy, df_database)
    display_coord_map(df_night, df_database)
    display_coord_map(df_sunny, df_database)