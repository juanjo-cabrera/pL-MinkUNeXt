# Code taken from MinkLoc3Dv2 repo: https://github.com/jac99/MinkLoc3Dv2.git

import numpy as np
from typing import List
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sklearn.neighbors import KDTree
import sys
import os
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config import PARAMS 
from datasets.quantization import quantizer
from datasets.base_datasets import EvaluationTuple, TrainingDataset
from datasets.augmentation import TrainSetTransform
from datasets.freiburg.pnv_train import PNVTrainingDataset
from datasets.freiburg.pnv_train import TrainTransform as PNVTrainTransform
from datasets.samplers import BatchSampler
from datasets.base_datasets import PointCloudLoader
from datasets.freiburg.pnv_raw import PNVPointCloudLoader
import math

def get_pointcloud_loader(dataset_type) -> PointCloudLoader:
    return PNVPointCloudLoader()


def make_datasets(validation: bool = True):
    # Create training and validation datasets
    datasets = {}
    train_set_transform = TrainSetTransform(PARAMS.aug_mode)
    # PoinNetVLAD datasets (RobotCar and Inhouse)
    # PNV datasets have their own transform
    train_transform = PNVTrainTransform(PARAMS.aug_mode)
    datasets['train'] = PNVTrainingDataset(PARAMS.dataset_folder + PARAMS.train_folder, PARAMS.train_file,
                                           transform=train_transform, set_transform=train_set_transform)
    if validation:
        datasets['val'] = PNVTrainingDataset(PARAMS.dataset_folder + PARAMS.val_folder, PARAMS.val_file)

    return datasets


def rgb_to_hue_pytorch(rgb_tensor):
    # Normalizar los valores RGB al rango [0, 1]
    # rgb_tensor = rgb_tensor / 255.0 ya está normalizado

    # Separar los canales RGB
    r = rgb_tensor[:, 0]
    g = rgb_tensor[:, 1]
    b = rgb_tensor[:, 2]

    # Calcular el numerador y denominador para la fórmula del arcoseno
    numerator = (r - g) + (r - b)
    denominator = 2 * torch.sqrt((r - g) ** 2 + (r - b) * (g - b))

    # Evitar divisiones por cero: establecer denominadores cercanos a cero en un valor pequeño
    denominator = torch.where(denominator == 0, torch.tensor(1e-10, device=denominator.device), denominator)

    # Calcular theta usando la fórmula del arcoseno
    theta = torch.acos(numerator / denominator)

    # Convertir theta de radianes a grados manualmente
    theta_degrees = theta * (180.0 / torch.pi)

    # Ajustar el valor del Hue según el valor de B y G
    hue = torch.where(b <= g, theta_degrees, 360 - theta_degrees)
    # Pasar a formato [N, 1]
    hue = hue.unsqueeze(1)
    # normalizar a [0.5, 1.5]
    hue = (hue / 360.0) + 0.5
    return hue


def make_collate_fn(dataset: TrainingDataset, quantizer, batch_split_size=None):
    # quantizer: converts to polar (when polar coords are used) and quantizes
    # batch_split_size: if not None, splits the batch into a list of multiple mini-batches with batch_split_size elems
    def collate_fn(data_list):
        # Constructs a batch object
        clouds = [e[0] for e in data_list]
        labels = [e[1] for e in data_list]


        points = [e['points'] for e in clouds]
        colors = [e['colors'] for e in clouds]

        # lengths of each cloud
        lens = [len(cloud) for cloud in points]
        points = torch.cat(points, dim=0)       # Produces (batch_size, n_points, 3) tensor
        colors = torch.cat(colors, dim=0)
        if dataset.set_transform is not None:
            # Apply the same transformation on all dataset elements
            points = dataset.set_transform(points)
        points = points.split(lens)
        colors = colors.split(lens)

        # Compute positives and negatives mask
        # dataset.queries[label]['positives'] is bitarray
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        coords = []
        feats = []
        for point, color in zip(points, colors):
            quatizated = ME.utils.sparse_quantize(coordinates=point, features=color, quantization_size=PARAMS.quantization_size)
            coords.append(quatizated[0])
            feats.append(quatizated[1])

        if batch_split_size is None or batch_split_size == 0:
            coords = ME.utils.batched_coordinates(coords)
            if PARAMS.use_rgb or PARAMS.use_dino_features or PARAMS.use_gradients:
                feats = torch.cat(feats, dim=0)
            elif PARAMS.use_depth_features:
                intermediate_feats = torch.cat(feats, dim=0)
                initial_feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
                feats = {}             
                feats['initial'] = initial_feats
                feats['intermediate'] = intermediate_feats

            elif PARAMS.use_gray:
                feats = torch.cat(feats, dim=0)
                feats = torch.mean(feats, dim=1, keepdim=True)
            # elif PARAMS.use_hue:
            #     feats = torch.cat(feats, dim=0)
            #     feats = rgb_to_hue_pytorch(feats)
            else:
                feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            # Assign a dummy feature equal to 1 to each point
            # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation          
            pointcloud_batch = {'coords': coords, 'features': feats}
        else:
             # Split the batch into chunks
            pointcloud_batch = []
            for i in range(0, len(coords), batch_split_size):
                temp_coords = coords[i:i + batch_split_size]
                temp_feats = feats[i:i + batch_split_size]
                c = ME.utils.batched_coordinates(temp_coords)
                # if PARAMS.use_rgb or PARAMS.use_hue or PARAMS.use_dino_features or PARAMS.use_gradients or PARAMS.use_magnitude or PARAMS.use_magnitude_hue or PARAMS.use_magnitude_ones or PARAMS.use_angle or PARAMS.use_anglexy or PARAMS.use_anglexy_hue or PARAMS.use_anglexy_ones or PARAMS.use_magnitude_anglexy_hue or PARAMS.use_magnitude_anglexy_hue_ones:
                #     f = torch.cat(temp_feats, dim=0)
                # elif PARAMS.use_depth_features:
                #     intermediate_feats = torch.cat(temp_feats, dim=0)
                #     initial_feats = torch.ones((c.shape[0], 1), dtype=torch.float32)
                #     # f = {'initial': initial_feats, 'intermediate': f}
                # elif PARAMS.use_gray:
                #     f = torch.cat(temp_feats, dim=0)
                #     f = torch.mean(f, dim=1, keepdim=True)
                # # elif PARAMS.use_hue:
                # #     f = torch.cat(temp_feats, dim=0)
                # #     f = rgb_to_hue_pytorch(f)
                # else:
                #     f = torch.ones((c.shape[0], 1), dtype=torch.float32)
                if not PARAMS.use_depth_features:
                    f = torch.cat(temp_feats, dim=0)
                    pointcloud_minibatch = {'coords': c, 'features': f}
                else:
                    intermediate_feats = torch.cat(temp_feats, dim=0)
                    initial_feats = torch.ones((c.shape[0], 1), dtype=torch.float32)
                    pointcloud_minibatch = {'coords': c, 'initial_features': initial_feats, 'intermediate_features': intermediate_feats}
                pointcloud_batch.append(pointcloud_minibatch)

        return pointcloud_batch, positives_mask, negatives_mask

    return collate_fn

class CartesianQuantizer():
    def __init__(self, quant_step: float):
        self.quant_step = quant_step

    def __call__(self, pc):
        # Converts to polar coordinates and quantizes with different step size for each coordinate
        # pc: (N, 3) point cloud with Cartesian coordinates (X, Y, Z)
        assert pc.shape[1] == 3
        quantized_pc, ndx = ME.utils.sparse_quantize(pc, quantization_size=self.quant_step, return_index=True)
        # Return quantized coordinates and index of selected elements
        return quantized_pc, ndx

def make_dataloaders(validation=True):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(validation=validation)
    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], batch_size=PARAMS.batch_size,
                                 batch_size_limit=PARAMS.batch_size_limit,
                                 batch_expansion_rate=PARAMS.batch_expansion_rate)

    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_collate_fn(datasets['train'],  quantizer, PARAMS.batch_split_size)
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler,
                                     collate_fn=train_collate_fn, num_workers=PARAMS.num_workers,
                                     pin_memory=True)
    if validation and 'val' in datasets:
        val_collate_fn = make_collate_fn(datasets['val'], quantizer, PARAMS.batch_split_size)
        val_sampler = BatchSampler(datasets['val'], batch_size=PARAMS.val_batch_size)
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=PARAMS.num_workers, pin_memory=True)

    return dataloders


def filter_query_elements(query_set: List[EvaluationTuple], map_set: List[EvaluationTuple],
                          dist_threshold: float) -> List[EvaluationTuple]:
    # Function used in evaluation dataset generation
    # Filters out query elements without a corresponding map element within dist_threshold threshold
    map_pos = np.zeros((len(map_set), 2), dtype=np.float32)
    for ndx, e in enumerate(map_set):
        map_pos[ndx] = e.position

    # Build a kdtree
    kdtree = KDTree(map_pos)

    filtered_query_set = []
    count_ignored = 0
    for ndx, e in enumerate(query_set):
        position = e.position.reshape(1, -1)
        nn = kdtree.query_radius(position, dist_threshold, count_only=True)[0]
        if nn > 0:
            filtered_query_set.append(e)
        else:
            count_ignored += 1

    print(f"{count_ignored} query elements ignored - not having corresponding map element within {dist_threshold} [m] "
          f"radius")
    return filtered_query_set


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e

