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
from losses.truncated_smoothap import TruncatedSmoothAP
import time
from eval.pnv_evaluate import *
from datasets.freiburg.generate_training_tuples_baseline import *
from datasets.freiburg.generate_training_tuples_baseline_3depths import *

def get_datetime():
    return time.strftime("%Y%m%d_%H%M")

def do_train(model):
    # Create model class
    loss_fn = TruncatedSmoothAP(tau1=PARAMS.tau1, similarity=PARAMS.similarity,
                                    positives_per_query=PARAMS.positives_per_query)

    
    
    s = get_datetime()
    if PARAMS.dataset_folder == '/media/arvc/DATOS/Juanjo/Datasets/PCD_non_metric_Friburgo_base/':
        model_name = 'Indoor_MinkUNeXt_base_truncated_' + 'pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + 'voxel_size' + str(PARAMS.voxel_size) + 'height' + str(PARAMS.height) + '_' + s
    elif PARAMS.dataset_folder == '/media/arvc/DATOS/Juanjo/Datasets/PCD_non_metric_Friburgo_small/':
        model_name = 'Indoor_MinkUNeXt_small_truncated_' + 'pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + 'voxel_size' + str(PARAMS.voxel_size) + 'height' + str(PARAMS.height) + '_' + s
    else:
        model_name = 'Indoor_MinkUNeXt_truncated_aug' + str(PARAMS.aug_mode) + 'pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + 'voxel_size' + str(PARAMS.voxel_size) + 'height' + str(PARAMS.height) + '_' + s
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
        wandb.init(project='IndoorMinkUNeXt_base_experiment_truncated', config=params_dict, name='pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + 'voxel_size' + str(PARAMS.voxel_size) + 'height' + str(PARAMS.height))
    elif PARAMS.dataset_folder == '/media/arvc/DATOS/Juanjo/Datasets/PCD_non_metric_Friburgo_small/':
        wandb.init(project='IndoorMinkUNeXt_small_experiment_truncated', config=params_dict, name='pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + 'voxel_size' + str(PARAMS.voxel_size) + 'height' + str(PARAMS.height))
    else:
        wandb.init(project='IndoorMinkUNeXt_experiment_truncated', config=params_dict, name= 'aug' + str(PARAMS.aug_mode) +'pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + 'voxel_size' + str(PARAMS.voxel_size) + 'height' + str(PARAMS.height))

    ###########################################################################
    #
    ###########################################################################

    # Training statistics
    stats = {'train': [], 'eval': []}
    stats['best_val_recall'] = 0

    if 'val' in dataloaders:
        # Validation phase
        phases = ['train', 'val']
        stats['val'] = []
    else:
        phases = ['train']

    for epoch in tqdm.tqdm(range(1, PARAMS.epochs + 1)):
        metrics = {'train': {}, 'val': {}}      # Metrics for wandb reporting

        for phase in phases:
            running_stats = []  # running stats for the current epoch and phase
            count_batches = 0

            if phase == 'train':
                global_iter = iter(dataloaders['train'])
            else:
                global_iter = None if dataloaders['val'] is None else iter(dataloaders['val'])

            while True:
                count_batches += 1
                batch_stats = {}
                if PARAMS.debug and count_batches > 2:
                    break

                try:
                    temp_stats = train_step_fn(global_iter, model, phase, device, optimizer, loss_fn)
                    batch_stats['global'] = temp_stats

                except StopIteration:
                    # Terminate the epoch when one of dataloders is exhausted
                    break

                running_stats.append(batch_stats)

            # Compute mean stats for the phase
            epoch_stats = {}
            for substep in running_stats[0]:
                epoch_stats[substep] = {}
                for key in running_stats[0][substep]:
                    temp = [e[substep][key] for e in running_stats]
                    if type(temp[0]) is dict:
                        epoch_stats[substep][key] = {key: np.mean([e[key] for e in temp]) for key in temp[0]}
                    elif type(temp[0]) is np.ndarray:
                        # Mean value per vector element
                        epoch_stats[substep][key] = np.mean(np.stack(temp), axis=0)
                    else:
                        epoch_stats[substep][key] = np.mean(temp)

            stats[phase].append(epoch_stats)
            print_stats(phase, epoch_stats)

            # Log metrics for wandb
            metrics[phase]['loss1'] = epoch_stats['global']['loss']
            if 'num_non_zero_triplets' in epoch_stats['global']:
                metrics[phase]['active_triplets1'] = epoch_stats['global']['num_non_zero_triplets']

            if 'positive_ranking' in epoch_stats['global']:
                metrics[phase]['positive_ranking'] = epoch_stats['global']['positive_ranking']

            if 'recall' in epoch_stats['global']:
                metrics[phase]['recall@1'] = epoch_stats['global']['recall'][1]

            if 'ap' in epoch_stats['global']:
                metrics[phase]['AP'] = epoch_stats['global']['ap']


            # save the model weights with the best validation recall@1
            if phase == 'val' and PARAMS.save_best and epoch_stats['global']['recall'][1] > stats['best_val_recall']:
                stats['best_val_recall'] = epoch_stats['global']['recall'][1]
                best_model_path = model_pathname + '_best.pth'
                print(f"Saving best weights: {best_model_path}")
                torch.save(model.state_dict(), best_model_path)


        # ******* FINALIZE THE EPOCH *******

        # evaluate the model 
        if 'val' in phases:
            #if epoch >= 50:
            if epoch % 10 == 0:
                # write results to a .txt withou deleting previous results
                file_name = '/home/arvc/Juanjo/develop/DepthMinkUNeXt/training/experiment_truncated_results_v4.txt'
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

                model.train(True)



        wandb.log(metrics)

        if scheduler is not None:
            scheduler.step()

        #if params.save_freq > 0 and epoch % params.save_freq == 0:
        #    torch.save(model.state_dict(), model_pathname + "_" + str(epoch) + ".pth")

        #if params.batch_expansion_th is not None:
            # Dynamic batch size expansion based on number of non-zero triplets
            # Ratio of non-zero triplets
        #    le_train_stats = stats['train'][-1]  # Last epoch training stats
        #    rnz = le_train_stats['global']['num_non_zero_triplets'] / le_train_stats['global']['num_triplets']
        #    if rnz < params.batch_expansion_th:
        #        dataloaders['train'].batch_sampler.expand_batch()
    
    print('Training completed.')

    # Finaliza la sesión de wandb
    wandb.finish()
    # Save final model weights
    final_model_path = model_pathname + '_final.pth'
    print(f"Saving weights: {final_model_path}")
    torch.save(model.state_dict(), final_model_path)
       
    



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
    positive_distance = [0.7]
    negative_distance = [0.7]
    dataset_folders = ['/media/arvc/DATOS/Juanjo/Datasets/PCD_non_metric_Friburgo/']
    PARAMS.epochs = 80
    TRAIN_FOLDER = "TrainingBaseline/"
    VAL_FOLDER = "Validation/"
    aug_modes = [0]
    PARAMS.cuda_device = 'cuda:0'
    
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
                        train_pickle = 'training_queries_baseline_pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + '_3depths.pickle'
                        val_pickle = 'validation_queries_baseline_pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + '.pickle'
                        # check if the pickle files exist
                        if not os.path.exists(base_path + TRAIN_FOLDER + train_pickle) or not os.path.exists(base_path + VAL_FOLDER + val_pickle):
                            generate_pickle_3depths(TRAIN_FOLDER, train_pickle)
                            generate_pickle(VAL_FOLDER, val_pickle)
                        PARAMS.train_file = train_pickle
                        PARAMS.val_file = val_pickle
                        do_train(model)

                        # empty cache
                        torch.cuda.empty_cache()
            
        
