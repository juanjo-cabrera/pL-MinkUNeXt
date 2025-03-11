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
from model.minkunext import MinkUNeXt
from datasets.dataset_utils import rgb_to_hue_pytorch


def evaluate_sa(model, device, log: bool = False, show_progress: bool = False):
    # Run evaluation on all eval datasets

    eval_database_files = ['cloudy_evaluation_database.pickle', 'night_evaluation_database.pickle']
    eval_query_files = ['cloudy_evaluation_query.pickle', 'night_evaluation_query.pickle']

    assert len(eval_database_files) == len(eval_query_files)

    stats = {}
    database_embeddings = []
    database_file = eval_database_files[0]
    p = os.path.join(PARAMS.test_folder, database_file)
    with open(p, 'rb') as f:
        database_sets = pickle.load(f)
    #for set in tqdm.tqdm(database_sets, disable=not show_progress, desc='Computing database embeddings'):
    database_embeddings.append(get_latent_vectors(model, database_sets, device))

    for query_file in eval_query_files:
        location_name = query_file.split('_')[0]
        p = os.path.join(PARAMS.test_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        temp = evaluate_dataset(model, device, database_sets, database_embeddings, query_sets, log=log, show_progress=show_progress)
        stats[location_name] = temp

    return stats


def evaluate_dataset(model, device, database_sets, database_embeddings, query_sets, log: bool = False,
                     show_progress: bool = False):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    one_percent_recall = []

    query_embeddings = []

    model.eval()   

    #for set in tqdm.tqdm(query_sets, disable=not show_progress, desc='Computing query embeddings'):
    query_embeddings.append(get_latent_vectors(model, query_sets, device))
    i = 0
    j = 0
    recall, one_percent_recall, mean_error = get_recall(i, j, database_embeddings, query_embeddings, query_sets,
                                        database_sets, log=log)
 
    stats = {'ave_one_percent_recall': one_percent_recall, 'ave_recall': recall, 'mean_error': mean_error}
    return stats


def get_latent_vectors(model, set, device):
    # Adapted from original PointNetVLAD code

    if PARAMS.debug:
        embeddings = np.random.rand(len(set), 256)
        return embeddings

    pc_loader = PNVPointCloudLoader()

    model.eval()
    embeddings = None
    for i, elem_ndx in enumerate(set):
        #pc_file_path = os.path.join(PARAMS.dataset_folder, set[elem_ndx]["query"])
        query = pc_loader.read_pc(set[elem_ndx]["query"])
        query_points = torch.tensor(query['points'], dtype=torch.float)      
        query_color = torch.tensor(query['colors'], dtype=torch.float)

        query_pc = {}
        query_pc['points'] = query_points
        query_pc['colors'] = query_color
       

        embedding = compute_embedding(model, query_pc, device)
        if embeddings is None:
            embeddings = np.zeros((len(set), embedding.shape[1]), dtype=embedding.dtype)
        embeddings[i] = embedding

    return embeddings


def compute_embedding(model, pc, device):
    with torch.no_grad():
        points = pc['points']
        color = pc['colors']
        coords, feats = ME.utils.sparse_quantize(coordinates=points, features=color, quantization_size=PARAMS.quantization_size)

        bcoords = ME.utils.batched_coordinates([coords])
        """
        if not PARAMS.use_rgb:
            feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
        if PARAMS.use_gray:
            feats = torch.mean(feats, dim=1, keepdim=True)
        """ 
        if PARAMS.use_rgb or PARAMS.use_dino_features or PARAMS.use_gradients or PARAMS.use_magnitude or PARAMS.use_magnitude_hue or PARAMS.use_magnitude_ones or PARAMS.use_angle or PARAMS.use_anglexy or PARAMS.use_anglexy_hue or PARAMS.use_anglexy_ones or PARAMS.use_magnitude_anglexy_hue or PARAMS.use_magnitude_anglexy_hue_ones:
            feats = feats.to(device)
        elif PARAMS.use_gray:
            feats = torch.mean(feats, dim=1, keepdim=True)
            feats = feats.to(device)
        elif PARAMS.use_hue:
            feats = rgb_to_hue_pytorch(feats)
            feats = feats.to(device)
        else:
            feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            feats = feats.to(device)
    
        batch = {'coords': bcoords.to(device), 'features': feats}
        #coords, _ = quantizer(pc)
        #with torch.no_grad():
        
        #bcoords = ME.utils.batched_coordinates([coords])
        #feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
        #batch = {'coords': bcoords.to(device), 'features': feats.to(device)}

        # Compute global descriptor
        y = model(batch)
        embedding = y['global'].detach().cpu().numpy()

    return embedding


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets, log=False):
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

    PARAMS.cuda_device = 'cuda:0'
    PARAMS.test_folder = '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_LARGE/SAARBRUCKEN_A/'
    if torch.cuda.is_available():
        device = PARAMS.cuda_device
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    # set cuda device
    torch.cuda.set_device(device)
    
    
    # PARAMS.weights_path = '/media/arvc/DATOS/Juanjo/weights/DepthMinkunext/aiai_weights/Indoor_MinkUNeXt_truncated_augonly_best_effects0.5pos0.7neg0.7voxel_size0.05height-0.25_20250217_0953_best.pth'
    # PARAMS.weights_path = '/media/arvc/DATOS/Juanjo/weights/DepthMinkunext/aiai_weights/Indoor_MinkUNeXt_truncated_hue_augonly_best_effects0.5pos0.7neg0.7voxel_size0.05height-0.25_20250220_1604_best_test.pth'
    # PARAMS.weights_path = '/media/arvc/DATOS/Juanjo/weights/DepthMinkunext/aiai_weights/Indoor_MinkUNeXt_pos_per_query12batch_size256_truncated_augonly_best_effects0.5pos0.7neg0.7voxel_size0.05height-0.25_20250301_0106_best.pth'
    # PARAMS.weights_path = '/media/arvc/DATOS/Juanjo/weights/DepthMinkunext/aiai_weights/Indoor_MinkUNeXt_pos_per_query12batch_size512_truncated_augonly_best_effects0.5pos0.7neg0.7voxel_size0.05height-0.25_20250302_0339_best_test.pth'
    # PARAMS.weights_path = '/media/arvc/DATOS/Juanjo/weights/DepthMinkunext/aiai_weights/Indoor_MinkUNeXt_pos_per_query16batch_size512_truncated_augonly_best_effects0.5pos0.7neg0.7voxel_size0.05height-0.25_20250302_1046_best_test.pth'
    # PARAMS.weights_path = '/media/arvc/DATOS/Juanjo/weights/DepthMinkunext/aiai_weights/Indoor_MinkUNeXt_pos_per_query20batch_size512_truncated_augonly_best_effects0.5pos0.7neg0.7voxel_size0.05height-0.25_20250302_1757_best_test.pth'
    # PARAMS.weights_path = '/media/arvc/DATOS/Juanjo/weights/DepthMinkunext/aiai_weights/Indoor_MinkUNeXt_pos_per_query12batch_size256_truncated_augonly_best_effects0.5pos0.7neg0.7voxel_size0.05height-0.25_20250301_0106_best_test.pth'
    # PARAMS.weights_path = '/media/arvc/DATOS/Juanjo/weights/DepthMinkunext/aiai_weights/Indoor_MinkUNeXt_pos_per_query24batch_size512_truncated_augonly_best_effects0.5pos0.7neg0.7voxel_size0.05height-0.25_20250303_0112_best_test.pth'
    PARAMS.weights_path = '/media/arvc/DATOS/Juanjo/weights/DepthMinkunext/aiai_weights/Indoor_MinkUNeXt_gradients_pos_per_query12batch_size512_truncated_augonly_best_effects0.5pos0.7neg0.7voxel_size0.05height-0.25_20250308_2326_best_test.pth'
    PARAMS.use_magnitude =  True
    # Load a pretrained model                           
              
    if PARAMS.use_magnitude_hue or PARAMS.use_magnitude_ones or PARAMS.use_anglexy:
        model = MinkUNeXt(in_channels=2, out_channels=512, D=3)       
    elif PARAMS.use_rgb or PARAMS.use_gradients or PARAMS.use_anglexy_hue or PARAMS.use_anglexy_ones:
        model = MinkUNeXt(in_channels=3, out_channels=512, D=3)
    elif PARAMS.use_magnitude_anglexy_hue:
        model = MinkUNeXt(in_channels=4, out_channels=512, D=3)
    elif PARAMS.use_magnitude_anglexy_hue_ones:
        model = MinkUNeXt(in_channels=5, out_channels=512, D=3)
    else: 
        model = MinkUNeXt(in_channels=1, out_channels=512, D=3)

    model.load_state_dict(torch.load(PARAMS.weights_path, map_location=device))

    model.to(device)

    stats = evaluate_sa(model, device, log=False, show_progress=True)
    print_eval_stats(stats)

    # Save results to the text file

    model_name = os.path.split(PARAMS.weights_path)[1]
    model_name = os.path.splitext(model_name)[0]
    prefix = "{}, {}".format(PARAMS.protocol, model_name)
    pnv_write_eval_stats("/home/arvc/Juanjo/develop/DepthMinkUNeXt/training/results_saarbrucken_a.txt", prefix, stats)


