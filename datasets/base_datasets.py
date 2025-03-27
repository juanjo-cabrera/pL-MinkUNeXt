# Base dataset classes, inherited by dataset-specific classes
import os
import pickle
from typing import List
from typing import Dict
import torch
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from scipy.spatial import cKDTree
from config import PARAMS

class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        assert position.shape == (2,)

        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position


class EvaluationTuple:
    # Tuple describing an evaluation set element
    def __init__(self, timestamp: int, rel_scan_filepath: str, position: np.array):
        # position: x, y position in meters
        assert position.shape == (2,)
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.position = position

    def to_tuple(self):
        return self.timestamp, self.rel_scan_filepath, self.position

class PointCloud():
    def __init__(self, points, colors):
        self.points = points
        self.colors = colors

class TrainingDataset(Dataset):
    def __init__(self, dataset_path, query_filename, transform=None, set_transform=None):
        # remove_zero_points: remove points with all zero coords
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        self.queries: Dict[int, TrainingTuple] = pickle.load(open(self.query_filepath, 'rb'))
        print('{} queries in the dataset'.format(len(self)))
 
        # pc_loader must be set in the inheriting class
        self.pc_loader: PointCloudLoader = None
        if PARAMS.use_depth_features:
            features_path = dataset_path.replace('PCD_non_metric_Friburgo', 'Friburgo_A_DepthAnything_large_output_conv1_features')
            # list all the files in the features_path
            features_folders = os.listdir(features_path)
            # list the files of each folder
            features_files = []
            for folder in features_folders:
                folder_path = features_path + folder
                files = os.listdir(folder_path)
                files = [folder_path + '/' + file for file in files]
                features_files.extend(files)
        
            # load the features
            self.features = {}
            self.features['files'] = features_files
            self.features['values'] = [np.load(file) for file in features_files]
            print('Features loaded: ', len(self.features))

        self.processed_pcds = {}
        dataset2_path = dataset_path.replace('Train_extended', 'fr_seq2_cloudy1')
        # magnitude_path = dataset_path.replace('Validation', 'Train_extended')
        magnitude_path1 = dataset_path.replace('PCD_LARGE', 'MAGNITUDE')
        magnitude_path2 = dataset2_path.replace('PCD_LARGE', 'MAGNITUDE')
     
        # list all the files in the features_path
        magnitude_folders1 = os.listdir(magnitude_path1)
        magnitude_folders2 = os.listdir(magnitude_path2)

        # list the files of each folder
        magnitude_files = []
        angles_files = []
        pcds_large_files = []
        for folder in magnitude_folders1:
            folder_path = magnitude_path1 + folder
            files = os.listdir(folder_path)
            files = [folder_path + '/' + file for file in files]
            magnitude_files.extend(files)
            angle_files = [file.replace('MAGNITUDE', 'ANGLE') for file in files]
            angles_files.extend(angle_files)
            pcds_large_files.extend([file.replace('MAGNITUDE', 'PCD_LARGE').replace('.npy', '.ply') for file in files])
            pcds_base_files = [file.replace('PCD_LARGE', 'PCD_BASE') for file in pcds_large_files]
            pcds_small_files = [file.replace('PCD_LARGE', 'PCD_SMALL') for file in pcds_large_files]
        if not 'Validation' in dataset_path:
            for folder in magnitude_folders2:
                folder_path = magnitude_path2 + folder
                files = os.listdir(folder_path)
                files = [folder_path + '/' + file for file in files]
                magnitude_files.extend(files)
                angle_files = [file.replace('MAGNITUDE', 'ANGLE') for file in files]
                angles_files.extend(angle_files)
                pcds_large_files.extend([file.replace('MAGNITUDE', 'PCD_LARGE').replace('.npy', '.ply') for file in files])
                pcds_base_files = [file.replace('PCD_LARGE', 'PCD_BASE') for file in pcds_large_files]
                pcds_small_files = [file.replace('PCD_LARGE', 'PCD_SMALL') for file in pcds_large_files]
        # load the features
        magnitudes = [np.load(file) for file in magnitude_files]
        if not 'Validation' in dataset_path:
            self.max_magnitude = np.max(magnitudes)
            PARAMS.max_magnitude = self.max_magnitude
        print('Max magnitude computed: ', PARAMS.max_magnitude)
        angles = [np.load(file) for file in angles_files]
        pcds_large = []
        pcds_base = []
        pcds_small = []
        for i in range(len(magnitudes)):
            pcd_large = o3d.io.read_point_cloud(pcds_large_files[i])
            if not 'Validation' in dataset_path:
                pcd_base = o3d.io.read_point_cloud(pcds_base_files[i])
                pcd_small = o3d.io.read_point_cloud(pcds_small_files[i])
            if PARAMS.use_gradients:
                magnitude = magnitudes[i].reshape(-1, 1)
                angle = angles[i].reshape(-1, 1)
                # global normalize the magnitude
                magnitude = (magnitude / PARAMS.max_magnitude) + 0.5
                rad_angle = np.deg2rad(angle)
            
                x = np.cos(rad_angle)/2 + 0.5
                y = np.sin(rad_angle)/2 + 0.5
                
                if PARAMS.use_magnitude:
                    features = magnitude
                elif PARAMS.use_angle:
                    angle = (angle / 360.0)
                    features = angle + 0.5
                elif PARAMS.use_anglexy:
                    features = np.column_stack((x, y))
                elif PARAMS.use_hue:
                    hue = self.rgb_to_hue(np.asarray(pcd_large.colors))
                    features = hue
                elif PARAMS.use_magnitude_hue:
                    hue = self.rgb_to_hue(np.asarray(pcd_large.colors))
                    features = np.column_stack((magnitude, hue))
                elif PARAMS.use_magnitude_ones:
                    ones = np.ones((magnitude.shape[0], 1))
                    features = np.column_stack((magnitude, ones))
                elif PARAMS.use_anglexy_hue:
                    hue = self.rgb_to_hue(np.asarray(pcd_large.colors))
                    features = np.column_stack((x, y, hue))
                elif PARAMS.use_anglexy_ones:
                    ones = np.ones((magnitude.shape[0], 1))
                    features = np.column_stack((x, y, ones))
                elif PARAMS.use_magnitude_anglexy:
                    features = np.column_stack((magnitude, x, y))
                elif PARAMS.use_magnitude_anglexy_hue:
                    hue = self.rgb_to_hue(np.asarray(pcd_large.colors))
                    features = np.column_stack((magnitude, x, y, hue))
                    
                elif PARAMS.use_magnitude_angle_hue:
                    angle = (angle / 360.0) + 0.5
                    hue = self.rgb_to_hue(np.asarray(pcd_large.colors))
                    features = np.column_stack((magnitude, angle, hue))
                elif PARAMS.use_magnitude_anglexy_hue_grey:
                    hue = self.rgb_to_hue(np.asarray(pcd_large.colors))
                    grey = np.mean(np.asarray(pcd_large.colors), axis=1).reshape(-1, 1)
                    features = np.column_stack((magnitude, x, y, hue, grey))
                elif PARAMS.use_magnitude_angle_hue_grey:
                    angle = (angle / 360.0) + 0.5
                    hue = self.rgb_to_hue(np.asarray(pcd_large.colors))
                    grey = np.mean(np.asarray(pcd_large.colors), axis=1).reshape(-1, 1)
                    features = np.column_stack((magnitude, angle, hue, grey))
                elif PARAMS.use_magnitude_anglexy_hue_rgb:
                    hue = self.rgb_to_hue(np.asarray(pcd_large.colors))
                    features = np.column_stack((magnitude, x, y, hue, np.asarray(pcd_large.colors)))
                elif PARAMS.use_magnitude_angle_hue_rgb:
                    angle = (angle / 360.0) + 0.5
                    hue = self.rgb_to_hue(np.asarray(pcd_large.colors))
                    features = np.column_stack((magnitude, angle, hue, np.asarray(pcd_large.colors)))
                elif PARAMS.use_magnitude_anglexy_hue_ones:
                    hue = self.rgb_to_hue(np.asarray(pcd_large.colors))
                    ones = np.ones((magnitude.shape[0], 1))
                    features = np.column_stack((magnitude, x, y, hue, ones))     
            else: 
                features = np.ones((np.asarray(pcd_large.points).shape[0], 1))   

            pcd_large = PointCloud(points=np.asarray(pcd_large.points), colors=features)
            if not 'Validation' in dataset_path:
                pcd_base = PointCloud(points=np.asarray(pcd_base.points), colors=features)
                pcd_small = PointCloud(points=np.asarray(pcd_small.points), colors=features)

            # filter the points by height
            if PARAMS.height is not None:
                pcd_large = self.filter_by_height(pcd_large, height=PARAMS.height)   
                if not 'Validation' in dataset_path:
                    pcd_base = self.filter_by_height(pcd_base, height=PARAMS.height)
                    pcd_small = self.filter_by_height(pcd_small, height=PARAMS.height)    
            # show pointcloud
            #o3d.visualization.draw_geometries([pcd])
            if PARAMS.voxel_size is not None:            
                pcd_large.points, pcd_large.colors = self.voxel_downsample_with_features(pcd_large.points, pcd_large.colors, voxel_size=PARAMS.voxel_size)
                if not 'Validation' in dataset_path:
                    pcd_base.points, pcd_base.colors = self.voxel_downsample_with_features(pcd_base.points, pcd_base.colors, voxel_size=PARAMS.voxel_size)
                    pcd_small.points, pcd_small.colors = self.voxel_downsample_with_features(pcd_small.points, pcd_small.colors, voxel_size=PARAMS.voxel_size)

            points_large, colors_large = self.global_normalize(pcd_large, max_distance=PARAMS.max_distance)
            if not 'Validation' in dataset_path:
                points_base, colors_base = self.global_normalize(pcd_base, max_distance=PARAMS.max_distance)
                points_small, colors_small = self.global_normalize(pcd_small, max_distance=PARAMS.max_distance)
            
            pc_large = {}
            pc_large['points'] = points_large
            pc_large['colors'] = colors_large
            pcds_large.append(pc_large)
            if not 'Validation' in dataset_path:
                pc_base = {}
                pc_base['points'] = points_base
                pc_base['colors'] = colors_base
                pc_small = {}
                pc_small['points'] = points_small
                pc_small['colors'] = colors_small            
                pcds_base.append(pc_base)
                pcds_small.append(pc_small)
        
        self.processed_pcds['large'] = pcds_large
        print('Point clouds loaded: ', len(pcds_large))
        if not 'Validation' in dataset_path:
            self.processed_pcds['base'] = pcds_base
            self.processed_pcds['small'] = pcds_small
        self.processed_pcds['files'] = pcds_large_files
        print('Files loaded: ', len(pcds_large_files))

    def global_normalize(self, pcd, max_distance=15.0):
            import copy
            """
            Normalize a pointcloud to achieve mean zero, scaled between [-1, 1] and with a fixed number of points
            """
            pcd = copy.deepcopy(pcd)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

            [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            z_mean = np.mean(z)

            x = x - x_mean
            y = y - y_mean
            z = z - z_mean

            x = x / max_distance
            y = y / max_distance
            z = z / max_distance

            points[:, 0] = x
            points[:, 1] = y
            points[:, 2] = z

            # pointcloud_normalized = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            return points, colors

    def voxel_downsample_with_features(self, points, features, voxel_size):
        """
        Realiza un voxel downsampling en una nube de puntos y mantiene las características asociadas.
        
        Args:
            points (np.ndarray): Array de Nx3 con las coordenadas de los puntos.
            features (np.ndarray): Array de NxM con las características asociadas a cada punto.
            voxel_size (float): Tamaño del voxel para el downsampling.

        Returns:
            downsampled_points (np.ndarray): Puntos downsampled (Mx3).
            downsampled_features (np.ndarray): Features promediadas o combinadas correspondientes (MxM).
        """
        # Crear la nube de puntos en Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Realizar el voxel downsampling usando Open3D
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        downsampled_points = np.asarray(downsampled_pcd.points)

        # Obten el índice del punto original más cercanos a los puntos downsampled
        kdtree = cKDTree(points)
        _, indices = kdtree.query(downsampled_points)
        # Obten la feature correspondiente para los puntos en cada voxel
        downsampled_features = features[indices]      

        return downsampled_points, downsampled_features
           
    def filter_by_height(self, pcd=None, height=0.5):
        # filter the points by height
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        idx = points[:, 2] > height
        # now select the final pointclouds    
        pcd_non_plane = PointCloud(points=points[idx], colors=colors[idx])        
        return pcd_non_plane
    
    def rgb_to_hue(self, rgb):
        # Separar los canales RGB
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

        # Calcular el numerador y denominador para la fórmula del arcoseno
        numerator = (r - g) + (r - b)
        denominator = 2 * np.sqrt((r - g) ** 2 + (r - b) * (g - b))

        # Evitar divisiones por cero: establecer denominadores cercanos a cero en un valor pequeño
        denominator = np.where(denominator == 0, 1e-10, denominator)

        # Calcular theta usando la fórmula del arcoseno
        theta = np.arccos(numerator / denominator)

        # Convertir theta de radianes a grados manualmente
        theta_degrees = theta * (180.0 / np.pi)

        # Ajustar el valor del Hue según el valor de B y G
        hue = np.where(b <= g, theta_degrees, 360 - theta_degrees)
        # Pasar a formato [N, 1]
        hue = hue.reshape(-1, 1)
        # normalizar a [0.5, 1.5]
        hue = (hue / 360.0) + 0.5
        return hue

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        # file_pathname = os.path.join(self.dataset_path, self.queries[ndx].rel_scan_filepath)
        # si el path está completo, no se le añade el dataset_path
        if self.queries[ndx].rel_scan_filepath.startswith('/home/'):
            file_pathname = self.queries[ndx].rel_scan_filepath
        else:
            file_pathname = os.path.join(self.dataset_path, self.queries[ndx].rel_scan_filepath)

        if self.transform is not None:
            if PARAMS.use_video:
                file_pathname = file_pathname.replace('PCD_non_metric_Friburgo', 'PCD_Depth_Anything_Video_Small')
                if np.random.rand() < 0.5:
                    file_pathname = file_pathname.replace('PCD_non_metric_Friburgo', 'PCD_Depth_Anything_Video_Small')

            else:
                if PARAMS.aug_mode == '3depths0.7':
                    # replace file_pathname '.../PCD_non_metric_Friburgo/...' by '.../PCD_non_metric_Friburgo_base/...'
                    # en una probabilidad del 40% se cambia el path
                    if np.random.rand() < 0.7:
                        if np.random.rand() < 0.5:
                            file_pathname = file_pathname.replace('PCD_LARGE', 'PCD_BASE')
                        else:
                            file_pathname = file_pathname.replace('PCD_LARGE', 'PCD_SMALL')
                elif PARAMS.aug_mode == '3depths0.8':
                    # replace file_pathname '.../PCD_non_metric_Friburgo/...' by '.../PCD_non_metric_Friburgo_base/...'
                    # en una probabilidad del 40% se cambia el path
                    if np.random.rand() < 0.8:
                        if np.random.rand() < 0.5:
                            file_pathname = file_pathname.replace('PCD_LARGE', 'PCD_BASE')
                        else:
                            file_pathname = file_pathname.replace('PCD_LARGE', 'PCD_SMALL')
                elif PARAMS.aug_mode == '3depths0.9':
                    # replace file_pathname '.../PCD_non_metric_Friburgo/...' by '.../PCD_non_metric_Friburgo_base/...'
                    # en una probabilidad del 40% se cambia el path
                    if np.random.rand() < 0.9:
                        if np.random.rand() < 0.5:
                            file_pathname = file_pathname.replace('PCD_LARGE', 'PCD_BASE')
                        else:
                            file_pathname = file_pathname.replace('PCD_LARGE', 'PCD_SMALL')
                elif PARAMS.aug_mode == 'only_best_effects':
                    # replace file_pathname '.../PCD_non_metric_Friburgo/...' by '.../PCD_non_metric_Friburgo_base/...'
                    # en una probabilidad del 40% se cambia el path
                    if np.random.rand() < PARAMS.p_depth:
                        if np.random.rand() < 0.5:
                            file_pathname = file_pathname.replace('PCD_LARGE', 'PCD_BASE')
                        else:
                            file_pathname = file_pathname.replace('PCD_LARGE', 'PCD_SMALL')
                elif PARAMS.aug_mode == 'only_best_effects0.5':
                    # replace file_pathname '.../PCD_non_metric_Friburgo/...' by '.../PCD_non_metric_Friburgo_base/...'
                    # en una probabilidad del 50% se cambia el path
                    if np.random.rand() < 0.5:
                        if np.random.rand() < 0.5:
                            file_pathname = file_pathname.replace('PCD_LARGE', 'PCD_BASE')
                        else:
                            file_pathname = file_pathname.replace('PCD_LARGE', 'PCD_SMALL')

        # if PARAMS.use_depth_features:
        #     original_file_pathname = os.path.join(self.dataset_path, self.queries[ndx].rel_scan_filepath)
        #     both_file_pathnames = [file_pathname, original_file_pathname]
        #     query = self.pc_loader.read_pc(both_file_pathnames)
        # else:
        # import time 
        # start = time.time()
        if PARAMS.use_depth_features:
            # get index of the file_pathname in the features['files']
            features_pathname = file_pathname.replace('_small', '')
            features_pathname = features_pathname.replace('_base', '')
            features_pathname = features_pathname.replace('PCD_non_metric_Friburgo', 'Friburgo_A_DepthAnything_large_output_conv1_features')
            features_pathname = features_pathname.replace('.ply', '.npy')
            index = self.features['files'].index(features_pathname)
            features = self.features['values'][index]
     
            query = self.pc_loader.read_pc_features(file_pathname, features)
        else:
            # query = self.pc_loader.read_pc(file_pathname, self.max_gradient)
            if 'SMALL' in file_pathname:
                file_pathname = file_pathname.replace('SMALL', 'LARGE')
                index = self.processed_pcds['files'].index(file_pathname)
                query = self.processed_pcds['small'][index]
            elif 'BASE' in file_pathname:
                file_pathname = file_pathname.replace('BASE', 'LARGE')
                index = self.processed_pcds['files'].index(file_pathname)
                query = self.processed_pcds['base'][index]
            else:
                index = self.processed_pcds['files'].index(file_pathname)
                query = self.processed_pcds['large'][index]
        # end = time.time()
        # print("Time to read point cloud: ", end - start)

        query_points = torch.tensor(query['points'], dtype=torch.float)      
        query_color = torch.tensor(query['colors'], dtype=torch.float)
        # normalize the color values /255.0
        # query_color = query_color / 255.0 No need, it is already normalized

        if self.transform is not None:
            query_points = self.transform(query_points)
            if PARAMS.add_noise and np.random.rand() < PARAMS.noise_prob:
                # add noise to the color values for each channel individually
                query_color = query_color + torch.randn_like(query_color) * PARAMS.sigma   

        query_pc = {}
        query_pc['points'] = query_points
        query_pc['colors'] = query_color

        return query_pc, ndx

    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives


class EvaluationSet:
    # Evaluation set consisting of map and query elements
    def __init__(self, query_set: List[EvaluationTuple] = None, map_set: List[EvaluationTuple] = None):
        self.query_set = query_set
        self.map_set = map_set

    def save(self, pickle_filepath: str):
        # Pickle the evaluation set

        # Convert data to tuples and save as tuples
        query_l = []
        for e in self.query_set:
            query_l.append(e.to_tuple())

        map_l = []
        for e in self.map_set:
            map_l.append(e.to_tuple())
        pickle.dump([query_l, map_l], open(pickle_filepath, 'wb'))

    def load(self, pickle_filepath: str):
        # Load evaluation set from the pickle
        query_l, map_l = pickle.load(open(pickle_filepath, 'rb'))

        self.query_set = []
        for e in query_l:
            self.query_set.append(EvaluationTuple(e[0], e[1], e[2]))

        self.map_set = []
        for e in map_l:
            self.map_set.append(EvaluationTuple(e[0], e[1], e[2]))

    def get_map_positions(self):
        # Get map positions as (N, 2) array
        positions = np.zeros((len(self.map_set), 2), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            positions[ndx] = pos.position
        return positions

    def get_query_positions(self):
        # Get query positions as (N, 2) array
        positions = np.zeros((len(self.query_set), 2), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            positions[ndx] = pos.position
        return positions


class PointCloudLoader:
    def __init__(self):
        # remove_zero_points: remove points with all zero coordinates
        # remove_ground_plane: remove points on ground plane level and below
        # ground_plane_level: ground plane level
        self.remove_zero_points = True
        self.remove_ground_plane = True
        self.ground_plane_level = None
        self.set_properties()

    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level. Must be defined in inherited classes.
        raise NotImplementedError('set_properties must be defined in inherited classes')

    def __call__(self, file_pathname):
        # Reads the point cloud from a disk and preprocess (optional removal of zero points and points on the ground
        # plane and below
        # file_pathname: relative file path
        assert os.path.exists(file_pathname), f"Cannot open point cloud: {file_pathname}"
        pc = self.read_pc(file_pathname)
        assert pc.shape[1] == 3

        if self.remove_zero_points:
            mask = np.all(np.isclose(pc, 0), axis=1)
            pc = pc[~mask]

        if self.remove_ground_plane:
            mask = pc[:, 2] > self.ground_plane_level
            pc = pc[mask]

        return pc

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        raise NotImplementedError("read_pc must be overloaded in an inheriting class")
    
    def read_pc_features(self, file_pathname: str, features) -> np.ndarray:
        # Reads the point cloud without pre-processing
        raise NotImplementedError("read_pc must be overloaded in an inheriting class")


