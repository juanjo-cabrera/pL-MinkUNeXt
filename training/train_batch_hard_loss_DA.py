import torch
import random
import numpy as np
#import PARAMS obejct from config.py in config file
import sys
import os
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config import PARAMS 
from trainer import *
from model.minkunext import MinkUNeXt
from losses.triplet_loss import make_loss
import time
from eval.pnv_evaluate import *
from datasets.freiburg.generate_training_tuples_baseline import *


def bh_print_stats(stats, phase):
    if 'num_pairs' in stats:
        # For batch hard contrastive loss
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   Pairs per batch (all/non-zero pos/non-zero neg): {:.1f}/{:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pairs'],
                       stats['pos_pairs_above_threshold'], stats['neg_pairs_above_threshold']))
    elif 'num_triplets' in stats:
        # For triplet loss
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   Triplets per batch (all/non-zero): {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_triplets'],
                       stats['num_non_zero_triplets']))
    elif 'num_pos' in stats:
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   #positives/negatives: {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pos'], stats['num_neg']))

    s = ''
    l = []
    if 'mean_pos_pair_dist' in stats:
        s += 'Pos dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}   Neg dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}'
        l += [stats['min_pos_pair_dist'], stats['mean_pos_pair_dist'], stats['max_pos_pair_dist'],
              stats['min_neg_pair_dist'], stats['mean_neg_pair_dist'], stats['max_neg_pair_dist']]
    if 'pos_loss' in stats:
        if len(s) > 0:
            s += '   '
        s += 'Pos loss: {:.4f}  Neg loss: {:.4f}'
        l += [stats['pos_loss'], stats['neg_loss']]
    if len(l) > 0:
        print(s.format(*l))

def get_datetime():
    return time.strftime("%Y%m%d_%H%M")

def do_train(model):
    # Create model class
    loss_fn = make_loss(params=PARAMS)

    
    
    s = get_datetime()
    if PARAMS.dataset_folder == '/media/arvc/DATOS/Juanjo/Datasets/PCD_non_metric_Friburgo_base/':
        model_name = 'Indoor_MinkUNeXt_base_batchhard_' + 'pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + 'voxel_size' + str(PARAMS.voxel_size) + 'height' + str(PARAMS.height) + '_' + s
    elif PARAMS.dataset_folder == '/media/arvc/DATOS/Juanjo/Datasets/PCD_non_metric_Friburgo_small/':
        model_name = 'Indoor_MinkUNeXt_small_batchhard_' + 'pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + 'voxel_size' + str(PARAMS.voxel_size) + 'height' + str(PARAMS.height) + '_' + s
    else:
        model_name = 'Indoor_MinkUNeXt_batchhard_aug' + str(PARAMS.aug_mode) + 'pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + 'voxel_size' + str(PARAMS.voxel_size) + 'height' + str(PARAMS.height) + '_' + s
    weights_path = create_weights_folder()
    model_pathname = os.path.join(weights_path, model_name)
    
    if PARAMS.print_model_info:
        print(model)
        n_params = sum([param.nelement() for param in model.parameters()])
        print('Number of model parameters: {}'.format(n_params))

    # Move the model to the proper device before configuring the optimizer
    if torch.cuda.is_available():
        device = PARAMS.cuda_device
    else:
        device = "cpu"
    # set cuda device
    torch.cuda.set_device(device)
    model.to(device)
    print('Model device: {}'.format(device))
    
    # set up dataloaders
    dataloaders = make_dataloaders()

    # Training elements
    print('OPTIMIZER: ', PARAMS.optimizer)
    if PARAMS.optimizer == 'Adam':
        optimizer_fn = torch.optim.Adam
    elif PARAMS.optimizer == 'AdamW':
        optimizer_fn = torch.optim.AdamW
    else:
        raise NotImplementedError(f"Unsupported optimizer: {PARAMS.optimizer}")

    if PARAMS.weight_decay is None or PARAMS.weight_decay == 0:
        optimizer = optimizer_fn(model.parameters(), lr=PARAMS.initial_lr)
    else:
        optimizer = optimizer_fn(model.parameters(), lr=PARAMS.initial_lr, weight_decay=PARAMS.weight_decay)

    if PARAMS.scheduler is None:
        scheduler = None
    else:
        if PARAMS.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PARAMS.epochs+1,
                                                                   eta_min=PARAMS.initial_lr)
        elif PARAMS.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, PARAMS.scheduler_milestones, gamma=0.1)
        else:
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(PARAMS.scheduler))

    if PARAMS.batch_split_size is None or PARAMS.batch_split_size == 0:
        train_step_fn = training_step
    else:
        # Multi-staged training approach with large batch split into multiple smaller chunks with batch_split_size elems
        train_step_fn = multistaged_training_step

    ###########################################################################
    # Initialize Weights&Biases logging service
    ###########################################################################

    # Create a dictionary with the parameters
    params_dict = vars(PARAMS)
    # Initialize wandb
    if PARAMS.dataset_folder == '/media/arvc/DATOS/Juanjo/Datasets/PCD_non_metric_Friburgo_base/':
        wandb.init(project='IndoorMinkUNeXt_base_experiment_batchhard', config=params_dict, name='pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + 'voxel_size' + str(PARAMS.voxel_size) + 'height' + str(PARAMS.height))
    elif PARAMS.dataset_folder == '/media/arvc/DATOS/Juanjo/Datasets/PCD_non_metric_Friburgo_small/':
        wandb.init(project='IndoorMinkUNeXt_small_experiment_batchhard', config=params_dict, name='pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + 'voxel_size' + str(PARAMS.voxel_size) + 'height' + str(PARAMS.height))
    else:
        wandb.init(project='IndoorMinkUNeXt_experiment_batchhard', config=params_dict, name= 'aug' + str(PARAMS.aug_mode) +'pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + 'voxel_size' + str(PARAMS.voxel_size) + 'height' + str(PARAMS.height))

    ###########################################################################
    #
    ###########################################################################

    is_validation_set = 'val' in dataloaders
    if is_validation_set:
        phases = ['train', 'val']
    else:
        phases = ['train']

    # Training statistics
    stats = {'train': [], 'val': [], 'eval': []}

    for epoch in tqdm.tqdm(range(1, PARAMS.epochs + 1)):
        metrics = {'train': {}, 'val': {}}      # Metrics for wandb reporting
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_stats = []  # running stats for the current epoch

            count_batches = 0
            for batch, positives_mask, negatives_mask in dataloaders[phase]:
                # batch is (batch_size, n_points, 3) tensor
                # labels is list with indexes of elements forming a batch
                count_batches += 1
                batch_stats = {}


                batch = {e: batch[e].to(device) for e in batch}

                n_positives = torch.sum(positives_mask).item()
                n_negatives = torch.sum(negatives_mask).item()
                if n_positives == 0 or n_negatives == 0:
                    # Skip a batch without positives or negatives
                    print('WARNING: Skipping batch without positive or negative examples')
                    continue

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Compute embeddings of all elements
                    embeddings = model(batch)
                    loss, temp_stats, _ = loss_fn(embeddings['global'], positives_mask, negatives_mask)

                    temp_stats = tensors_to_numbers(temp_stats)
                    batch_stats.update(temp_stats)
                    batch_stats['loss'] = loss.item()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_stats.append(batch_stats)
                torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

            # Compute mean stats for the phase
            epoch_stats = {}
            for key in running_stats[0].keys():
                temp = [e[key] for e in running_stats]
                epoch_stats[key] = np.mean(temp)

            stats[phase].append(epoch_stats)
            bh_print_stats(epoch_stats, phase)

        # ******* EPOCH END *******

        if scheduler is not None:
            scheduler.step()

        loss_metrics = {'train': stats['train'][-1]['loss']}

        if 'val' in phases:
            #if epoch <= 40:
            if epoch % 10 == 0:
                # write results to a .txt withou deleting previous results
                file_name = '/home/arvc/Juanjo/develop/DepthMinkUNeXt/training/experiment_batchhard_results_v4.txt'
                model.eval()
                model.to(device)
                print('Model evaluation epoch: {}'.format(epoch))
                
                # Evaluate the model
                stats_validation = evaluate(model, device, log=False, show_progress=True)
                mae_cloudy, mae_night, mae_sunny = stats_validation['cloudy']['mean_error'], stats_validation['night']['mean_error'], stats_validation['sunny']['mean_error']
                recall_cloudy, recall_night, recall_sunny = stats_validation['cloudy']['ave_recall'][0], stats_validation['night']['ave_recall'][0], stats_validation['sunny']['ave_recall'][0]
                mean_mae = (mae_cloudy + mae_night + mae_sunny) / 3
                
        
                # write results to a .txt withou deleting previous results
                mean_mae = (mae_cloudy + mae_night + mae_sunny) / 3
                with open(file_name, "a") as f:
                    f.write(f'{model_name} + epoch_ + {epoch}, {mae_cloudy}, {mae_night}, {mae_sunny}, {mean_mae}, {recall_cloudy}, {recall_night}, {recall_sunny}\n')
                    print('Results saved to: ', file_name)
                    # Log mae_cloudy, mae_night, mae_sunny, mean_mae for wandb
                metrics['val']['mae_cloudy'] = mae_cloudy
                metrics['val']['mae_night'] = mae_night
                metrics['val']['mae_sunny'] = mae_sunny
                metrics['val']['mean_mae'] = mean_mae
                metrics['val']['recall_cloudy'] = recall_cloudy
                metrics['val']['recall_night'] = recall_night
                metrics['val']['recall_sunny'] = recall_sunny
                print_eval_stats(stats_validation)
                wandb.log(metrics['val'])
                torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

                model.train(True)
    

        # Log metrics for wandb
        if 'train' in phases:
            if 'num_triplets' in stats['train'][-1]:
                metrics['train']['num_triplets'] = stats['train'][-1]['num_triplets']
                metrics['train']['num_non_zero_triplets'] = stats['train'][-1]['num_non_zero_triplets']
            elif 'num_pairs' in stats['train'][-1]:
                metrics['train']['num_pairs'] = stats['train'][-1]['num_pairs']
                metrics['train']['pos_pairs_above_threshold'] = stats['train'][-1]['pos_pairs_above_threshold']
                metrics['train']['neg_pairs_above_threshold'] = stats['train'][-1]['neg_pairs_above_threshold']
            metrics['train']['loss'] = stats['train'][-1]['loss']
        
        if 'val' in phases:
            if 'num_triplets' in stats['val'][-1]:
                metrics['val']['num_triplets'] = stats['val'][-1]['num_triplets']
                metrics['val']['num_non_zero_triplets'] = stats['val'][-1]['num_non_zero_triplets']
            elif 'num_pairs' in stats['val'][-1]:
                metrics['val']['num_pairs'] = stats['val'][-1]['num_pairs']
                metrics['val']['pos_pairs_above_threshold'] = stats['val'][-1]['pos_pairs_above_threshold']
                metrics['val']['neg_pairs_above_threshold'] = stats['val'][-1]['neg_pairs_above_threshold']

        

 

        if PARAMS.batch_expansion_th is not None:
            # Dynamic batch expansion
            epoch_train_stats = stats['train'][-1]
            if 'num_non_zero_triplets' not in epoch_train_stats:
                print('WARNING: Batch size expansion is enabled, but the loss function is not supported')
            else:
                # Ratio of non-zero triplets
                rnz = epoch_train_stats['num_non_zero_triplets'] / epoch_train_stats['num_triplets']
                if rnz < PARAMS.batch_expansion_th:
                    dataloaders['train'].batch_sampler.expand_batch()
        
        # Log metrics for wandb
        wandb.log(metrics['train'])
        
    print('')
    

    
    ###########################################################################
    # Save final model weights
    final_model_path = model_pathname + '_final.pth'
    print(f"Saving weights: {final_model_path}")
    torch.save(model.state_dict(), final_model_path)
    # Finaliza la sesión de wandb
    wandb.finish()

       
    



if __name__ == '__main__': 


    # Establecer la semilla para PyTorch
    torch.manual_seed(42)

    # Si estás usando un GPU, también necesitas establecer la semilla para el backend CUDA
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # Si estás usando múltiples GPUs

    # Establecer la semilla para otros paquetes que uses, como numpy y random
 
    np.random.seed(42)
    random.seed(42)
 

    # Si estás usando ciertos algoritmos de PyTorch que involucran operaciones en paralelo, 
    # podrías necesitar lo siguiente para reproducibilidad
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    voxel_sizes = [0.05]
    heights = [-0.25]

    # distances from 0.3 up to 2.0
    positive_distance = [0.5]
    negative_distance = [1.0]
    dataset_folders = ['/media/arvc/DATOS/Juanjo/Datasets/PCD_non_metric_Friburgo/']
    PARAMS.epochs = 80
    TRAIN_FOLDER = "TrainingBaseline/"
    VAL_FOLDER = "Validation/"
    PARAMS.loss = 'BatchHardTripletMarginLoss'
    PARAMS.normalize_embeddings = False
    PARAMS.margin = 0.2
    PARAMS.num_workers = 12
    PARAMS.batch_size = 16
    PARAMS.batch_size_limit = 54
    PARAMS.batch_expansion_rate = 1.4
    PARAMS.batch_expansion_th = 0.7
    PARAMS.cuda_device = 'cuda:1'
    PARAMS.batch_split_size = None
    PARAMS.optimizer = 'Adam'
    # del 32 al 38
    #aug_modes = [25]
    #aug_modes = [25]
    #aug_modes = ['remove_block', 'jitter', 'remove_points', 'translation', 'move_block', 'scale', 'all_effects1']
    aug_modes = ['remove_block', 'scale_xy2']
    
    for aug_mode in aug_modes:
        PARAMS.aug_mode = aug_mode
        print('Augmentation mode: ', PARAMS.aug_mode)
        for voxel_size in voxel_sizes:
            for height in heights:
                for dataset_folder in dataset_folders:
                    PARAMS.dataset_folder = dataset_folder
                    base_path = PARAMS.dataset_folder
                    PARAMS.height = height
                    PARAMS.voxel_size = voxel_size
                    print('Voxel size: ', PARAMS.voxel_size)
                    print('Height: ', PARAMS.height)
                    for i in range(len(positive_distance)):
                        # Restablecer semillas aleatorias
                        torch.manual_seed(42)
                        torch.cuda.manual_seed(42)
                        np.random.seed(42)
                        random.seed(42)
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = False
                        # Load a pretrained model
                        model = MinkUNeXt(in_channels=1, out_channels=512, D=3)
                        if PARAMS.weights_path is not None:
                            model.load_state_dict(torch.load(PARAMS.weights_path))
                            print('Model loaded from: {}'.format(PARAMS.weights_path))
                        PARAMS.positive_distance = positive_distance[i]            
                        PARAMS.negative_distance = negative_distance[i]
                        print('Positive distance: ', PARAMS.positive_distance)
                        print('Negative distance: ', PARAMS.negative_distance)
                        train_pickle = 'training_queries_baseline_pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + '.pickle'
                        val_pickle = 'validation_queries_baseline_pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + '.pickle'
                        # check if the pickle files exist
                        if not os.path.exists(base_path + TRAIN_FOLDER + train_pickle) or not os.path.exists(base_path + VAL_FOLDER + val_pickle):
                            generate_pickle(TRAIN_FOLDER, train_pickle)
                            generate_pickle(VAL_FOLDER, val_pickle)
                        PARAMS.train_file = train_pickle
                        PARAMS.val_file = val_pickle
                        do_train(model)

                        # empty cache
                        torch.cuda.empty_cache()
            
        
