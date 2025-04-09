
import os
import numpy as np
import open3d as o3d
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from config import PARAMS

class PointCloud():
    def __init__(self, points, colors):
        self.points = points
        self.colors = colors

# quiero una función que lea los pcds, los procese, asocie los features y los guarde en un .ply
depth_estimator = 'PCD_DISTILL_ANY_DEPTH_LARGE'
dataset_path = f'/media/arvc/DATOS/Juanjo/Datasets/COLD/{depth_estimator}/FRIBURGO_A/'



magnitude_path = dataset_path.replace(depth_estimator, 'MAGNITUDE')


# list all the files in the features_path
magnitude_folders = os.listdir(magnitude_path)

# list the files of each folder
magnitude_files = []
angles_files = []
pcds_large_files = []


for folder in magnitude_folders:
    folder_path = magnitude_path + folder
    files = os.listdir(folder_path)
    files = [folder_path + '/' + file for file in files]
    magnitude_files.extend(files)
    angle_files = [file.replace('MAGNITUDE', 'ANGLE') for file in files]
    angles_files.extend(angle_files)
    pcds_large_files.extend([file.replace('MAGNITUDE', 'PCD_DISTILL_ANY_DEPTH_LARGE').replace('.npy', '.ply') for file in files])
    
# load the features
magnitudes = [np.load(file) for file in magnitude_files]
angles = [np.load(file) for file in angles_files]
pcds_large = []

for i in range(len(magnitudes)):
    pcd_large = o3d.io.read_point_cloud(pcds_large_files[i])
    
    magnitude = magnitudes[i].reshape(-1, 1)
    angle = angles[i].reshape(-1, 1)
    # global normalize the magnitude
    magnitude = (magnitude / PARAMS.max_magnitude) + 0.5
    rad_angle = np.deg2rad(angle)
    angle = (angle / 360.0) + 0.5

    x_ang = np.cos(rad_angle)/2 + 0.5
    y_ang = np.sin(rad_angle)/2 + 0.5
    xmag = np.cos(rad_angle) * magnitude
    ymag = np.sin(rad_angle) * magnitude
    
    gray = np.mean(np.asarray(pcd_large.colors), axis=1).reshape(-1, 1)
    rgb = np.asarray(pcd_large.colors)
    r = rgb[:, 0].reshape(-1, 1)
    g = rgb[:, 1].reshape(-1, 1)
    b = rgb[:, 2].reshape(-1, 1)
    hue = self.rgb_to_hue(np.asarray(pcd_large.colors))

    features = np.column_stack((r, g, b, gray, hue, magnitude, angle, x_ang, y_ang, xmag, ymag))


    pcd_large = PointCloud(points=np.asarray(pcd_large.points), colors=features)
 
 
    pcd_large = self.filter_by_height(pcd_large, height=PARAMS.height)   
         
    pcd_large.points, pcd_large.colors = self.voxel_downsample_with_features(pcd_large.points, pcd_large.colors, voxel_size=PARAMS.voxel_size)
       

    points_large, features_large = self.global_normalize(pcd_large, max_distance=PARAMS.max_distance)

    # guarda un fichero por nube con x, y, z, r, g, b, gray, hue, magnitude, angle, x_ang, y_ang, xmag, ymag
    # save the point cloud
    save_path = dataset_path.replace(depth_estimator, 'PROCESSED_PCDS') + f'{folder}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_file = save_path + f'{i}.ply'
    pcd_large = o3d.geometry.PointCloud()
    pcd_large.points = o3d.utility.Vector3dVector(points_large)
    pcd_large.colors = o3d.utility.Vector3dVector(features_large)