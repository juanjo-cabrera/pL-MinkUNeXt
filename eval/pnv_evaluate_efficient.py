# Evaluation using PointNetVLAD evaluation protocol and test sets
# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad
import sys
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import torch
import MinkowskiEngine as ME
import tqdm
from datasets.freiburg.pnv_raw import PNVPointCloudLoader
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PARAMS 
from datasets.quantization import quantizer
from model.minkunext import model
from datasets.dataset_utils import rgb_to_hue_pytorch
from datasets.base_datasets import *

class EvaluationDataset():
    def __init__(self, dataset_path):
        # remove_zero_points: remove points with all zero coords
        database_path = dataset_path + 'Train/'
        cloudy_query_path = dataset_path + 'TestCloudy/'
        night_query_path = dataset_path + 'TestNight/'
        sunny_query_path = dataset_path + 'TestSunny/'
        assert os.path.exists(database_path), 'Cannot access database path: {}'.format(database_path)
        assert os.path.exists(cloudy_query_path), 'Cannot access cloudy query path: {}'.format(cloudy_query_path)
        assert os.path.exists(night_query_path), 'Cannot access night query path: {}'.format(night_query_path)
        assert os.path.exists(sunny_query_path), 'Cannot access sunny query path: {}'.format(sunny_query_path)
        self.database_path = database_path
        self.cloudy_query_path = cloudy_query_path
        self.night_query_path = night_query_path
        self.sunny_query_path = sunny_query_path

        self.max_gradient = None
        # pc_loader must be set in the inheriting class
        self.pc_loader: PointCloudLoader = None

        self.processed_pcds = {}

        database_magnitude = database_path.replace('PCD_LARGE', 'MAGNITUDE')
        query_cloudy_magnitude = cloudy_query_path.replace('PCD_LARGE', 'MAGNITUDE')
        query_night_magnitude = night_query_path.replace('PCD_LARGE', 'MAGNITUDE')
        query_sunny_magnitude = sunny_query_path.replace('PCD_LARGE', 'MAGNITUDE')
     
        # list all the files in the features_path
        folders = os.listdir(database_magnitude)
        # list the files of each folder
        database_magnitude_files = []
        database_angles_files = []
        database_pcds_files = []

        query_cloudy_magnitude_files = []
        query_cloudy_angles_files = []
        query_cloudy_pcds_files = []

        query_night_magnitude_files = []
        query_night_angles_files = []
        query_night_pcds_files = []

        query_sunny_magnitude_files = []
        query_sunny_angles_files = []
        query_sunny_pcds_files = []

        for folder in folders:
            database_folder_path = database_magnitude + folder
            query_cloudy_folder_path = query_cloudy_magnitude + folder
            query_night_folder_path = query_night_magnitude + folder
            query_sunny_folder_path = query_sunny_magnitude + folder
            database_files = os.listdir(database_folder_path)
            query_cloudy_files = os.listdir(query_cloudy_folder_path)
            query_night_files = os.listdir(query_night_folder_path)
            query_sunny_files = os.listdir(query_sunny_folder_path)
            database_files = [database_folder_path + '/' + file for file in database_files]
            query_cloudy_files = [query_cloudy_folder_path + '/' + file for file in query_cloudy_files]
            query_night_files = [query_night_folder_path + '/' + file for file in query_night_files]
            query_sunny_files = [query_sunny_folder_path + '/' + file for file in query_sunny_files]

            database_magnitude_files.extend(database_files)
            database_angles_files.extend([file.replace('MAGNITUDE', 'ANGLE') for file in database_files])
            database_pcds_files.extend([file.replace('MAGNITUDE', 'PCD_LARGE').replace('.npy', '.ply') for file in database_files])

            query_cloudy_magnitude_files.extend(query_cloudy_files)
            query_cloudy_angles_files.extend([file.replace('MAGNITUDE', 'ANGLE') for file in query_cloudy_files])
            query_cloudy_pcds_files.extend([file.replace('MAGNITUDE', 'PCD_LARGE').replace('.npy', '.ply') for file in query_cloudy_files])

            query_night_magnitude_files.extend(query_night_files)
            query_night_angles_files.extend([file.replace('MAGNITUDE', 'ANGLE') for file in query_night_files])
            query_night_pcds_files.extend([file.replace('MAGNITUDE', 'PCD_LARGE').replace('.npy', '.ply') for file in query_night_files])

            query_sunny_magnitude_files.extend(query_sunny_files)
            query_sunny_angles_files.extend([file.replace('MAGNITUDE', 'ANGLE') for file in query_sunny_files])
            query_sunny_pcds_files.extend([file.replace('MAGNITUDE', 'PCD_LARGE').replace('.npy', '.ply') for file in query_sunny_files])

        self.max_magnitude = PARAMS.max_magnitude
        print('Max magnitude from training data', PARAMS.max_magnitude)
       
        database_magnitudes = [np.load(file) for file in database_magnitude_files]
        database_angles = [np.load(file) for file in database_angles_files]        
        query_cloudy_magnitudes = [np.load(file) for file in query_cloudy_magnitude_files]
        query_cloudy_angles = [np.load(file) for file in query_cloudy_angles_files]
        query_night_magnitudes = [np.load(file) for file in query_night_magnitude_files]
        query_night_angles = [np.load(file) for file in query_night_angles_files]
        query_sunny_magnitudes = [np.load(file) for file in query_sunny_magnitude_files]
        query_sunny_angles = [np.load(file) for file in query_sunny_angles_files]

        self.database_pcds = self.process_pcds(database_magnitudes, database_angles, database_pcds_files)
        self.query_cloudy_pcds = self.process_pcds(query_cloudy_magnitudes, query_cloudy_angles, query_cloudy_pcds_files)
        self.query_night_pcds = self.process_pcds(query_night_magnitudes, query_night_angles, query_night_pcds_files)
        self.query_sunny_pcds = self.process_pcds(query_sunny_magnitudes, query_sunny_angles, query_sunny_pcds_files)

    
    def process_pcds(self, magnitudes, angles, pcds_files):
        pcds = []
        for i in range(len(magnitudes)):
            pcd = o3d.io.read_point_cloud(pcds_files[i])            
            if PARAMS.use_gradients:
                magnitude = magnitudes[i].reshape(-1, 1)
                angle = angles[i].reshape(-1, 1)
                # global normalize the magnitude
                magnitude = (magnitude / self.max_magnitude) + 0.5
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
                    hue = self.rgb_to_hue(np.asarray(pcd.colors))
                    features = hue
                elif PARAMS.use_magnitude_hue:
                    hue = self.rgb_to_hue(np.asarray(pcd.colors))
                    features = np.column_stack((magnitude, hue))
                elif PARAMS.use_magnitude_ones:
                    ones = np.ones((magnitude.shape[0], 1))
                    features = np.column_stack((magnitude, ones))
                elif PARAMS.use_anglexy_hue:
                    hue = self.rgb_to_hue(np.asarray(pcd.colors))
                    features = np.column_stack((x, y, hue))
                elif PARAMS.use_anglexy_ones:
                    ones = np.ones((magnitude.shape[0], 1))
                    features = np.column_stack((x, y, ones))
                elif PARAMS.use_magnitude_anglexy:
                    features = np.column_stack((magnitude, x, y))
                elif PARAMS.use_magnitude_anglexy_hue:
                    hue = self.rgb_to_hue(np.asarray(pcd.colors))
                    features = np.column_stack((magnitude, x, y, hue))
                elif PARAMS.use_magnitude_anglexy_hue_ones:
                    hue = self.rgb_to_hue(np.asarray(pcd.colors))
                    ones = np.ones((magnitude.shape[0], 1))
                    features = np.column_stack((magnitude, x, y, hue, ones))    
                else:
                    features = np.ones((magnitude.shape[0], 1))        

            pcd = PointCloud(points=np.asarray(pcd.points), colors=features)  

            # filter the points by height
            if PARAMS.height is not None:
                pcd = self.filter_by_height(pcd, height=PARAMS.height)   
              
            if PARAMS.voxel_size is not None:            
                pcd.points, pcd.colors = self.voxel_downsample_with_features(pcd.points, pcd.colors, voxel_size=PARAMS.voxel_size)
              
            points, colors = self.global_normalize(pcd, max_distance=PARAMS.max_distance)

            pc = {}
            pc['points'] = torch.tensor(points, dtype=torch.float)  
            pc['colors'] = torch.tensor(colors, dtype=torch.float)  
            pcds.append(pc)

            pcds_dict = {}
            pcds_dict['pcds'] = pcds
            pcds_dict['files'] = pcds_files
        return pcds_dict

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
    
    def evaluate(self, model, device, log: bool = False, show_progress: bool = False):
        if PARAMS.use_gradients:
            print('REMINDER: using max_gradient=903.84625 for train_extended')
        # Run evaluation on all eval datasets

        eval_database_files = ['cloudy_evaluation_database.pickle', 'night_evaluation_database.pickle', 'sunny_evaluation_database.pickle']
        eval_query_files = ['cloudy_evaluation_query.pickle', 'night_evaluation_query.pickle', 'sunny_evaluation_query.pickle']

        assert len(eval_database_files) == len(eval_query_files)

        stats = {}
        database_embeddings = []
        database_file = eval_database_files[0]
        p = os.path.join(PARAMS.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)
        #for set in tqdm.tqdm(database_sets, disable=not show_progress, desc='Computing database embeddings'):
        database_embeddings.append(self.get_latent_vectors(model, database_sets, device))

        for query_file in eval_query_files:
            location_name = query_file.split('_')[0]
            p = os.path.join(PARAMS.dataset_folder, query_file)
            with open(p, 'rb') as f:
                query_sets = pickle.load(f)

            temp = self.evaluate_dataset(model, device, database_sets, database_embeddings, query_sets, log=log, show_progress=show_progress)
            stats[location_name] = temp

        return stats



    def evaluate_dataset(self, model, device, database_sets, database_embeddings, query_sets, log: bool = False,
                        show_progress: bool = False):
        # Run evaluation on a single dataset
        recall = np.zeros(25)
        one_percent_recall = []

        query_embeddings = []

        model.eval()   

        #for set in tqdm.tqdm(query_sets, disable=not show_progress, desc='Computing query embeddings'):
        query_embeddings.append(self.get_latent_vectors(model, query_sets, device))
        i = 0
        j = 0
        recall, one_percent_recall, mean_error = self.get_recall(i, j, database_embeddings, query_embeddings, query_sets,
                                            database_sets, log=log)
    
        stats = {'ave_one_percent_recall': one_percent_recall, 'ave_recall': recall, 'mean_error': mean_error}
        return stats



    def get_latent_vectors(self, model, set, device):
        # Adapted from original PointNetVLAD code

        # if PARAMS.debug:
        #     embeddings = np.random.rand(len(set), 256)
        #     return embeddings

        # pc_loader = PNVPointCloudLoader()

        model.eval()
        embeddings = None
        for i, elem_ndx in enumerate(set):
            #pc_file_path = os.path.join(PARAMS.dataset_folder, set[elem_ndx]["query"])
            #query = pc_loader.read_pc(set[elem_ndx]["query"])
            query_filename = set[elem_ndx]["query"]
            if 'Train' in query_filename:
                index = self.database_pcds['files'].index(query_filename)
                query_pc = self.database_pcds['pcds'][index]
            elif 'TestCloudy' in query_filename:
                index = self.query_cloudy_pcds['files'].index(query_filename)
                query_pc = self.query_cloudy_pcds['pcds'][index]
            elif 'TestNight' in query_filename:
                index = self.query_night_pcds['files'].index(query_filename)
                query_pc = self.query_night_pcds['pcds'][index]
            elif 'TestSunny' in query_filename:
                index = self.query_sunny_pcds['files'].index(query_filename)
                query_pc = self.query_sunny_pcds['pcds'][index]

            # query_points = torch.tensor(query['points'], dtype=torch.float)      
            # query_color = torch.tensor(query['colors'], dtype=torch.float)

            # query_pc = {}
            # query_pc['points'] = query_points
            # query_pc['colors'] = query_color
        

            embedding = self.compute_embedding(model, query_pc, device)
            if embeddings is None:
                embeddings = np.zeros((len(set), embedding.shape[1]), dtype=embedding.dtype)
            embeddings[i] = embedding

        return embeddings


    def compute_embedding(self, model, pc, device):
        with torch.no_grad():
            points = pc['points']
            color = pc['colors']
            coords, feats = ME.utils.sparse_quantize(coordinates=points, features=color, quantization_size=PARAMS.quantization_size)

            bcoords = ME.utils.batched_coordinates([coords])
            
            # if PARAMS.use_rgb or PARAMS.use_hue or PARAMS.use_dino_features or PARAMS.use_gradients or PARAMS.use_magnitude or PARAMS.use_magnitude_hue or PARAMS.use_magnitude_ones or PARAMS.use_angle or PARAMS.use_anglexy or PARAMS.use_anglexy_hue or PARAMS.use_anglexy_ones or PARAMS.use_magnitude_anglexy_hue or PARAMS.use_magnitude_anglexy_hue_ones:
            #     feats = feats.to(device)
            # elif PARAMS.use_depth_features:
            #         intermediate_feats = feats.to(device)
            #         initial_feats = torch.ones((coords.shape[0], 1), dtype=torch.float32).to(device)
            # elif PARAMS.use_gray:
            #     feats = torch.mean(feats, dim=1, keepdim=True)
            #     feats = feats.to(device)
            # # elif PARAMS.use_hue:
            # #     feats = rgb_to_hue_pytorch(feats)
            # #     feats = feats.to(device)
            # else:
            #     print('Cuidado, no se ha seleccionado ninguna característica, empleando unos')
            #     feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            #     feats = feats.to(device)

            if not PARAMS.use_depth_features:
                feats = feats.to(device)
                batch = {'coords': bcoords.to(device), 'features': feats}
            else:
                intermediate_feats = feats.to(device)
                initial_feats = torch.ones((coords.shape[0], 1), dtype=torch.float32).to(device)
                batch = {'coords': bcoords.to(device), 'initial_features': initial_feats, 'intermediate_features': intermediate_feats}
            
            #coords, _ = quantizer(pc)
            #with torch.no_grad():
            
            #bcoords = ME.utils.batched_coordinates([coords])
            #feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
            #batch = {'coords': bcoords.to(device), 'features': feats.to(device)}

            # Compute global descriptor
            y = model(batch)
            embedding = y['global'].detach().cpu().numpy()

        return embedding


    def get_recall(self, m, n, database_vectors, query_vectors, query_sets, database_sets, log=False):
        """
        database_positions = []
        for i, elem_ndx in enumerate(database_sets):
            #pc_file_path = os.path.join(PARAMS.dataset_folder, set[elem_ndx]["query"])
            position = np.array([database_sets[elem_ndx]["x"], database_sets[elem_ndx]["y"]])
            database_positions.append(position)
        
        database_positions = np.array(database_positions)
        database_positions_tree = KDTree(database_positions)
        """
        # Original PointNetVLAD code
        database_output = database_vectors[m]
        queries_output = query_vectors[n]

        # When embeddings are normalized, using Euclidean distance gives the same
        # nearest neighbour search results as using cosine distance
        database_nbrs = KDTree(database_output)

        num_neighbors = 25
        recall = [0] * num_neighbors

        one_percent_retrieved = 0
        threshold = max(int(round(len(database_output)/100.0)), 1)

        num_evaluated = 0
        errors = []
        for i in range(len(queries_output)):
            # i is query element ndx
            query_details = query_sets[i]    # {'query': path, 'x': , 'y': }
            true_neighbor = query_details[0][0]
            #database_details = database_sets[true_neighbor]
            query_position = query_details['x'], query_details['y']
            # numpy array of position
            query_position = np.array([query_position])
            # check if index is correct
            #distance_position, index = database_positions_tree.query(query_position, k=1)
            #groundtruth_position = database_details['x'], database_details['y']
            # numpy array of position 
            
            if len(true_neighbor) == 0:
                continue
            num_evaluated += 1

            # Find nearest neightbours
            distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
            estimated_position = database_sets[indices[0][0]]['x'], database_sets[indices[0][0]]['y']
            estimated_position = np.array([estimated_position])
            #compute euclidean error between current_position and true_position

            metric_error = np.linalg.norm(estimated_position - query_position)
            errors.append(metric_error)


            for j in range(len(indices[0])):
                if indices[0][j] in true_neighbor:
                    recall[j] += 1
                    break

            if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbor)))) > 0:
                one_percent_retrieved += 1

        one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
        recall = (np.cumsum(recall)/float(num_evaluated))*100
        mean_error = np.mean(errors)
        return recall, one_percent_recall, mean_error


def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall']))
        print(stats[database_name]['ave_recall'])
        print('Mean error: {:.2f}'.format(stats[database_name]['mean_error']))


def pnv_write_eval_stats(file_name, prefix, stats):
    s = prefix
    ave_1p_recall_l = []
    ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        for ds in stats:
            ave_1p_recall = stats[ds]['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = stats[ds]['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        mean_1p_recall = np.mean(ave_1p_recall_l)
        mean_recall = np.mean(ave_recall_l)
        s += ", {:0.2f}, {:0.2f}\n".format(mean_1p_recall, mean_recall)
        f.write(s)


if __name__ == "__main__":
  
    if torch.cuda.is_available():
        device = PARAMS.cuda_device
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    # set cuda device
    torch.cuda.set_device(device)
    
    evaluation_set = EvaluationDataset(PARAMS.dataset_folder)
    PARAMS.weights_path = '/home/arvc/Juanjo/develop/IndoorMinkUnext/weights/Indoor_MinkUNeXt_rot270_pos0.7neg0.7voxel_size0.05height-0.25_20240906_1622_best.pth'
    
    model.load_state_dict(torch.load(PARAMS.weights_path, map_location=device))

    model.to(device)

    stats = evaluation_set.evaluate(model, device, log=False, show_progress=True)
    print_eval_stats(stats)

    # Save results to the text file

    model_name = os.path.split(PARAMS.weights_path)[1]
    model_name = os.path.splitext(model_name)[0]
    prefix = "{}, {}".format(PARAMS.protocol, model_name)
    pnv_write_eval_stats("/home/arvc/Juanjo/develop/IndoorMinkUnext/training/results.txt", prefix, stats)


