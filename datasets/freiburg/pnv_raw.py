import os
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config import PARAMS 
import numpy as np

import open3d as o3d
from datasets.base_datasets import PointCloudLoader
from scipy.spatial import cKDTree


class PNVPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Point clouds are already preprocessed with a ground plane removed
        self.remove_zero_points = False
        self.remove_ground_plane = False
        self.ground_plane_level = None

    

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
    
    def global_normalize_without_color(self, points, max_distance=15.0):
            import copy
            """
            Normalize a pointcloud to achieve mean zero, scaled between [-1, 1] and with a fixed number of points
            """

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
            return points
    
    
    def filter_by_height(self, pcd=None, height=0.5):
        # filter the points by height
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        idx = points[:, 2] > height
        # now select the final pointclouds
        pcd_non_plane = o3d.geometry.PointCloud()
        pcd_non_plane.points = o3d.utility.Vector3dVector(points[idx])
        pcd_non_plane.colors = o3d.utility.Vector3dVector(colors[idx])

        # show pointcloud
        #o3d.visualization.draw_geometries([pcd_non_plane])
        
        return pcd_non_plane
    
    def filter_by_height_features(self, pcd, features, height=0.5):
        # filter the points by height
        points = np.asarray(pcd.points)
        idx = points[:, 2] > height

        return points[idx], features[idx]
    
    def read_pointcloud_with_features(self, path):
        """
        Lee una nube de puntos y sus características desde un archivo PLY.

        Args:
            path (str): Ruta del archivo PLY.

        Returns:
            pcd (open3d.geometry.PointCloud): Nube de puntos leída.
            features (np.ndarray): Array con las características asociadas a cada punto.
        """
        # Abrir el archivo PLY y leer el contenido
        with open(path, 'r') as ply_file:
            lines = ply_file.readlines()

        # Encontrar el header y la posición de los datos
        header_end_index = lines.index("end_header\n")
        header = lines[:header_end_index + 1]
        
        # Contar la cantidad de propiedades (coordenadas + features)
        properties = [line for line in header if line.startswith("property")]
        num_features = len(properties) - 3  # Restar las 3 coordenadas (x, y, z)

        # Leer los datos de los puntos
        point_data = np.loadtxt(lines[header_end_index + 1:])

        # Separar las coordenadas (x, y, z) y los features
        points = point_data[:, :3]  # Coordenadas (x, y, z)
        features = point_data[:, 3:]  # Features adicionales

        # Crear la nube de puntos con Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        return pcd, features
    def voxel_downsample_with_features_copy(self, points, features, voxel_size):
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

        # Crear un KD-Tree para asociar los puntos downsampled con los originales
        kdtree = cKDTree(points)

        # Buscar los índices de los puntos originales más cercanos a los puntos downsampled
        indices = kdtree.query_ball_point(downsampled_points, voxel_size)

        # Calcular las features combinadas (promedio) para los puntos en cada voxel
        downsampled_features = np.array([features[idx].mean(axis=0) for idx in indices if len(idx) > 0])

        return downsampled_points, downsampled_features
    
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
    
    def normalize_color(self, color, is_color_in_range_0_255=False):
        r"""
        Convert color in range [0, 1] to [0.5, 1.5]. If the color is in range [0,
        255], use the argument `is_color_in_range_0_255=True`.

        `color` (torch.Tensor): Nx3 color feature matrix
        `is_color_in_range_0_255` (bool): If the color is in range [0, 255] not [0, 1], normalize the color to [0, 1].
        """
        if is_color_in_range_0_255:
            color /= 255
        color = color + 0.5
        return color

    def read_pc(self, file_pathname: str, max_gradient=903.84625)-> np.ndarray:
        # Reads the point cloud without pre-processing
        # Returns Nx3 ndarray
        file_path = os.path.join(file_pathname)
        if not PARAMS.use_dino_features and not PARAMS.use_depth_features:            
            pcd = o3d.io.read_point_cloud(file_path)
            if PARAMS.use_gradients:
                magnitude_pathname = file_pathname.replace('PCD_SMALL', 'MAGNITUDE')
                magnitude_pathname = magnitude_pathname.replace('PCD_BASE', 'MAGNITUDE')
                magnitude_pathname = magnitude_pathname.replace('PCD_LARGE', 'MAGNITUDE')
                angle_pathname = file_pathname.replace('PCD_SMALL', 'ANGLE')
                angle_pathname = angle_pathname.replace('PCD_BASE', 'ANGLE')
                angle_pathname = angle_pathname.replace('PCD_LARGE', 'ANGLE')
                # replace the extension .ply by .npy
                magnitude_pathname = magnitude_pathname.replace('.ply', '.npy')
                angle_pathname = angle_pathname.replace('.ply', '.npy')
                magnitude = np.load(magnitude_pathname)
                angle = np.load(angle_pathname)
                magnitude = magnitude.reshape(-1, 1)
                angle = angle.reshape(-1, 1)
                # normalize the magnitude and angle
                magnitude = (magnitude / max_gradient) + 0.5
                angle = (angle / 360.0) + 0.5
                ones = np.ones((magnitude.shape[0], 1))
                features = np.concatenate((magnitude, angle, ones), axis=1)

                pcd.colors = o3d.utility.Vector3dVector(features)
            # filter the points by height
            if PARAMS.height is not None:
                pcd = self.filter_by_height(pcd, height=PARAMS.height)       
            # show pointcloud
            #o3d.visualization.draw_geometries([pcd])
            if PARAMS.voxel_size is not None:
                pcd = pcd.voxel_down_sample(voxel_size=PARAMS.voxel_size)

            points, colors = self.global_normalize(pcd, max_distance=PARAMS.max_distance)
            # normalize the color
            # colors = self.normalize_color(colors)
            pc = {}
            pc['points'] = points
            pc['colors'] = colors

        elif PARAMS.use_depth_features:   
            pcd = o3d.io.read_point_cloud(file_path)     
            # check ig filepathname contains the word '_small' or '_base' remove it
            features_pathname = file_pathname.replace('_small', '')
            features_pathname = features_pathname.replace('_base', '')
            features_pathname = features_pathname.replace('PCD_non_metric_Friburgo', 'Friburgo_A_DepthAnything_large_output_conv1_features')
            features_pathname = features_pathname.replace('.ply', '.npy')
            features = np.load(features_pathname, mmap_mode='r')
            features = features.reshape(-1, features.shape[0])
            # filter the points by height
            # import time 
            # start = time.time()
            if PARAMS.height is not None:
                points, features = self.filter_by_height_features(pcd, features, height=PARAMS.height)    
            # end = time.time()
            # print("Time to filter by height: ", end-start)   
            # show pointcloud
            #o3d.visualization.draw_geometries([pcd])
            if PARAMS.voxel_size is not None:
                # start = time.time()
                points, features = self.voxel_downsample_with_features(points, features, voxel_size=PARAMS.voxel_size)        
                # end = time.time()
                # print("Time to voxel downsample: ", end-start)
                points = self.global_normalize_without_color(points, max_distance=PARAMS.max_distance)
                pc = {}
                pc['points'] = points
                pc['colors'] = features
             
        else:
            pcd, features = self.read_pointcloud_with_features(file_pathname)
    
            # filter the points by height
            if PARAMS.height is not None:
                points, features = self.filter_by_height_features(pcd, features, height=PARAMS.height)       
            # show pointcloud
            #o3d.visualization.draw_geometries([pcd])
            if PARAMS.voxel_size is not None:
                points, features = self.voxel_downsample_with_features(points, features, voxel_size=PARAMS.voxel_size)        
                points = self.global_normalize_without_color(points, max_distance=PARAMS.max_distance)
                pc = {}
                pc['points'] = points
                pc['colors'] = features
    
        return pc
    
    def read_pc_features(self, file_pathname: str, features) -> np.ndarray:
        # Reads the point cloud without pre-processing
        # Returns Nx3 ndarray
        file_path = os.path.join(file_pathname)
    
        pcd = o3d.io.read_point_cloud(file_path)     
        # check ig filepathname contains the word '_small' or '_base' remove it
        # features_pathname = file_pathname.replace('_small', '')
        # features_pathname = features_pathname.replace('_base', '')
        # features_pathname = features_pathname.replace('PCD_non_metric_Friburgo', 'Friburgo_A_DepthAnything_large_output_conv1_features')
        # features_pathname = features_pathname.replace('.ply', '.npy')
        # features = np.load(features_pathname, mmap_mode='r')
        features = features.reshape(-1, features.shape[0])
        # filter the points by height
        # import time 
        # start = time.time()
        if PARAMS.height is not None:
            points, features = self.filter_by_height_features(pcd, features, height=PARAMS.height)  
        # end = time.time()
        # print("Time to filter by height: ", end-start)     
        # show pointcloud
        #o3d.visualization.draw_geometries([pcd])
        if PARAMS.voxel_size is not None:
            # start = time.time()
            points, features = self.voxel_downsample_with_features(points, features, voxel_size=PARAMS.voxel_size)  
            # end = time.time()
            # print("Time to voxel downsample: ", end-start)      
            points = self.global_normalize_without_color(points, max_distance=PARAMS.max_distance)
        
        pc = {}
        pc['points'] = points
        pc['colors'] = features    
    
        return pc
