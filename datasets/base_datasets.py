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
        self.max_gradient = None
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
        if PARAMS.use_gradients:
            gradients_path = dataset_path.replace('Validation', 'Train_extended')
            gradients_path = gradients_path.replace('PCD_LARGE', 'MAGNITUDE')
            # list all the files in the features_path
            gradients_folders = os.listdir(gradients_path)
            # list the files of each folder
            gradients_files = []
            for folder in gradients_folders:
                folder_path = gradients_path + folder
                files = os.listdir(folder_path)
                files = [folder_path + '/' + file for file in files]
                gradients_files.extend(files)
        
            # load the features
            gradients = [np.load(file) for file in gradients_files]
            # compute the maximum gradient value
            self.max_gradient = np.max(gradients)
            gradients = []
           

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
                            file_pathname = file_pathname.replace('PCD_non_metric_Friburgo', 'PCD_non_metric_Friburgo_base')
                        else:
                            file_pathname = file_pathname.replace('PCD_non_metric_Friburgo', 'PCD_non_metric_Friburgo_small')
                elif PARAMS.aug_mode == '3depths0.8':
                    # replace file_pathname '.../PCD_non_metric_Friburgo/...' by '.../PCD_non_metric_Friburgo_base/...'
                    # en una probabilidad del 40% se cambia el path
                    if np.random.rand() < 0.8:
                        if np.random.rand() < 0.5:
                            file_pathname = file_pathname.replace('PCD_non_metric_Friburgo', 'PCD_non_metric_Friburgo_base')
                        else:
                            file_pathname = file_pathname.replace('PCD_non_metric_Friburgo', 'PCD_non_metric_Friburgo_small')
                elif PARAMS.aug_mode == '3depths0.9':
                    # replace file_pathname '.../PCD_non_metric_Friburgo/...' by '.../PCD_non_metric_Friburgo_base/...'
                    # en una probabilidad del 40% se cambia el path
                    if np.random.rand() < 0.9:
                        if np.random.rand() < 0.5:
                            file_pathname = file_pathname.replace('PCD_non_metric_Friburgo', 'PCD_non_metric_Friburgo_base')
                        else:
                            file_pathname = file_pathname.replace('PCD_non_metric_Friburgo', 'PCD_non_metric_Friburgo_small')
                elif PARAMS.aug_mode == 'only_best_effects':
                    # replace file_pathname '.../PCD_non_metric_Friburgo/...' by '.../PCD_non_metric_Friburgo_base/...'
                    # en una probabilidad del 40% se cambia el path
                    if np.random.rand() < PARAMS.p_depth:
                        if np.random.rand() < 0.5:
                            file_pathname = file_pathname.replace('PCD_non_metric_Friburgo', 'PCD_non_metric_Friburgo_base')
                        else:
                            file_pathname = file_pathname.replace('PCD_non_metric_Friburgo', 'PCD_non_metric_Friburgo_small')
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
            query = self.pc_loader.read_pc(file_pathname, self.max_gradient)
        # end = time.time()
        # print("Time to read point cloud: ", end - start)

        query_points = torch.tensor(query['points'], dtype=torch.float)      
        query_color = torch.tensor(query['colors'], dtype=torch.float)
        # normalize the color values /255.0
        # query_color = query_color / 255.0 No need, it is already normalized

        if self.transform is not None:
            query_points = self.transform(query_points)

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


