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
from eval.pnv_evaluate_efficient import *
from datasets.freiburg.generate_training_tuples_baseline import *
from datasets.freiburg.generate_test_sets import *

def get_datetime():
    return time.strftime("%Y%m%d_%H%M")

def do_train(model, dataloaders, evaluation_set_fa, evaluation_set_fb, evaluation_set_sa, evaluation_set_sb):
    
    # Create model class
    loss_fn = TruncatedSmoothAP(tau1=PARAMS.tau1, similarity=PARAMS.similarity,
                                    positives_per_query=PARAMS.positives_per_query)
    
    s = get_datetime()
    model_name = 'Indoor_MinkUNeXt_pos_per_query'+ str(PARAMS.positives_per_query) + 'batch_size' + str(PARAMS.batch_size) + '_truncated_aug' + str(PARAMS.aug_mode) + 'pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + 'voxel_size' + str(PARAMS.voxel_size) + 'height' + str(PARAMS.height) + '_' + s
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
    best_test_recall = 0

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
                # print(f"Epoch {epoch}, {phase} phase, batch {count_batches}")
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
            if epoch % 10 == 0 or epoch ==1:
                # write results to a .txt withou deleting previous results
                file_name = '/home/arvc/Juanjo/develop/DepthMinkUNeXt/training/experiment_image_features2.txt'
                model.eval()
                model.to(device)
                print('Model evaluation epoch: {}'.format(epoch))
                
                # Evaluate the model
                stats_validation = evaluation_set_fa.evaluate(model, device, log=False, show_progress=True)
                mae_cloudy, mae_night, mae_sunny = stats_validation['cloudy']['mean_error'], stats_validation['night']['mean_error'], stats_validation['sunny']['mean_error']
                fa_recall_cloudy, fa_recall_night, fa_recall_sunny = stats_validation['cloudy']['ave_recall'][0], stats_validation['night']['ave_recall'][0], stats_validation['sunny']['ave_recall'][0]
                fa_one_percent_recall_cloudy, fa_one_percent_recall_night, fa_one_percent_recall_sunny = stats_validation['cloudy']['ave_one_percent_recall'], stats_validation['night']['ave_one_percent_recall'], stats_validation['sunny']['ave_one_percent_recall']
                
                fb_stats = evaluation_set_fb.evaluate(model, device, log=False, show_progress=True)
                sa_stats = evaluation_set_sa.evaluate(model, device, log=False, show_progress=True)
                sb_stats = evaluation_set_sb.evaluate(model, device, log=False, show_progress=True)
                print_eval_stats(stats_validation)
                print_eval_stats(fb_stats)
                print_eval_stats(sa_stats)
                print_eval_stats(sb_stats)

                fb_recall_cloudy, fb_recall_sunny = fb_stats['cloudy']['ave_recall'][0], fb_stats['sunny']['ave_recall'][0]
                fb_one_percent_recall_cloudy, fb_one_percent_recall_sunny = fb_stats['cloudy']['ave_one_percent_recall'], fb_stats['sunny']['ave_one_percent_recall']
                sa_recall_cloudy, sa_recall_night = sa_stats['cloudy']['ave_recall'][0], sa_stats['night']['ave_recall'][0]
                sa_one_percent_recall_cloudy, sa_one_percent_recall_night = sa_stats['cloudy']['ave_one_percent_recall'], sa_stats['night']['ave_one_percent_recall']
                sb_recall_cloudy, sb_recall_night, sb_recall_sunny = sb_stats['cloudy']['ave_recall'][0], sb_stats['night']['ave_recall'][0], sb_stats['sunny']['ave_recall'][0]
                sb_one_percent_recall_cloudy, sb_one_percent_recall_night, sb_one_percent_recall_sunny = sb_stats['cloudy']['ave_one_percent_recall'], sb_stats['night']['ave_one_percent_recall'], sb_stats['sunny']['ave_one_percent_recall']

                # mean recall
                mean_recall = (fa_recall_cloudy + fa_recall_night + fa_recall_sunny + fb_recall_cloudy + fb_recall_sunny + sa_recall_cloudy + sa_recall_night + sb_recall_cloudy + sb_recall_night + sb_recall_sunny) / 10
                # mean one percent recall
                mean_one_percent_recall = (fa_one_percent_recall_cloudy + fa_one_percent_recall_night + fa_one_percent_recall_sunny + fb_one_percent_recall_cloudy + fb_one_percent_recall_sunny + sa_one_percent_recall_cloudy + sa_one_percent_recall_night + sb_one_percent_recall_cloudy + sb_one_percent_recall_night + sb_one_percent_recall_sunny) / 10

                if mean_recall > best_test_recall:
                    best_test_recall = mean_recall
                    best_test_model_path = model_pathname + '_best_test.pth'
                    print(f"Saving best test weights: {best_test_model_path}")
                    torch.save(model.state_dict(), best_test_model_path)
                
        
                # write results to a .txt withou deleting previous results
                mean_mae = (mae_cloudy + mae_night + mae_sunny) / 3
                with open(file_name, "a") as f:
                    if PARAMS.use_magnitude:
                        f.write('Feature: Magnitude\n')
                    if PARAMS.use_hue:
                        f.write('Feature: Hue\n')
                    if PARAMS.use_magnitude_hue:
                        f.write('Feature: Magnitude + Hue\n')
                    if PARAMS.use_magnitude_ones:
                        f.write('Feature: Magnitude + Ones\n')
                    if PARAMS.use_angle:
                        f.write('Feature: Angle\n')
                    if PARAMS.use_anglexy:
                        f.write('Feature: AngleXY\n')
                    if PARAMS.use_anglexy_hue:
                        f.write('Feature: AngleXY + Hue\n')
                    if PARAMS.use_anglexy_ones:
                        f.write('Feature: AngleXY + Ones\n')
                    if PARAMS.use_magnitude_anglexy_hue:
                        f.write('Feature: Magnitude + AngleXY + Hue\n')
                    if PARAMS.use_magnitude_anglexy_hue_ones:
                        f.write('Feature: Magnitude + AngleXY + Hue + Ones\n')
                    if PARAMS.use_magnitude_angle_hue:
                        f.write('Feature: Magnitude + Angle + Hue\n')
                    if PARAMS.use_magnitude_anglexy_hue_grey:
                        f.write('Feature: Magnitude + AngleXY + Hue + Grey\n')
                    if PARAMS.use_magnitude_angle_hue_grey:
                        f.write('Feature: Magnitude + Angle + Hue + Grey\n')
                    if PARAMS.use_magnitude_anglexy_hue_rgb:
                        f.write('Feature: Magnitude + AngleXY + Hue + RGB\n')
                    if PARAMS.use_magnitude_angle_hue_rgb:
                        f.write('Feature: Magnitude + Angle + Hue + RGB\n')
                    f.write(f'{model_name} + epoch_ + {epoch}, {fa_recall_cloudy}, {fa_recall_night}, {fa_recall_sunny}, {fb_recall_cloudy}, {fb_recall_sunny}, {sa_recall_cloudy}, {sa_recall_night}, {sb_recall_cloudy}, {sb_recall_night}, {sb_recall_sunny}, {mean_recall}, {fa_one_percent_recall_cloudy}, {fa_one_percent_recall_night}, {fa_one_percent_recall_sunny}, {fb_one_percent_recall_cloudy}, {fb_one_percent_recall_sunny}, {sa_one_percent_recall_cloudy}, {sa_one_percent_recall_night}, {sb_one_percent_recall_cloudy}, {sb_one_percent_recall_night}, {sb_one_percent_recall_sunny}, {mean_one_percent_recall}\n')
                    print('Results saved to: ', file_name)
                    # Log mae_cloudy, mae_night, mae_sunny, mean_mae for wandb
                metrics['val']['mae_cloudy'] = mae_cloudy
                metrics['val']['mae_night'] = mae_night
                metrics['val']['mae_sunny'] = mae_sunny
                metrics['val']['mean_mae'] = mean_mae
                metrics['val']['recall_cloudy'] = fa_recall_cloudy
                metrics['val']['recall_night'] = fa_recall_night
                metrics['val']['recall_sunny'] = fa_recall_sunny
                print_eval_stats(stats_validation)
                wandb.log(metrics['val'])

                model.train(True)



        wandb.log(metrics)

        if scheduler is not None:
            scheduler.step()

    
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



    PARAMS.height = -0.25
    PARAMS.voxel_size = 0.05
    print('Voxel size: ', PARAMS.voxel_size)
    print('Height: ', PARAMS.height)
                            
    # PARAMS.epochs = 50
    # PARAMS.scheduler_milestones = [20, 30]
    PARAMS.epochs = 200
    PARAMS.scheduler_milestones = [150, 180]

    PARAMS.TRAIN_FOLDER = "Train_extended/"
    PARAMS.VAL_FOLDER = "Validation/"

    PARAMS.cuda_device = 'cuda:1'
    PARAMS.use_rgb = False
    PARAMS.use_gray = False
    PARAMS.use_video = False
    
   
    PARAMS.batch_size = 512
    print('Batch size: ', PARAMS.batch_size)
            
    PARAMS.positives_per_query = 16
    print('Positives per query: ', PARAMS.positives_per_query)                    

    PARAMS.batch_split_size = 16
    PARAMS.val_batch_size = 16
    PARAMS.add_noise = False
    PARAMS.use_gradients = True
    PARAMS.positive_distance = 0.4
    PARAMS.negative_distance = 0.4
     
    #dataset_folders = ['/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_SMALL/FRIBURGO_A/', '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_BASE/FRIBURGO_A/', '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_DISTILL_ANY_DEPTH_LARGE_TEACHER/FRIBURGO_A/', '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_DISTILL_ANY_DEPTH_SMALL/FRIBURGO_A/', '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_DISTILL_ANY_DEPTH_BASE/FRIBURGO_A/']
    dataset_folder = '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_DISTILL_ANY_DEPTH_LARGE/FRIBURGO_A/'
  
    PARAMS.dataset_folder = dataset_folder
    print('Positive distance: ', PARAMS.positive_distance)
    print('Negative distance: ', PARAMS.negative_distance)
    train_pickle = 'training_queries_2seq_pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + '.pickle'
    val_pickle = 'validation_queries_2seq_pos' + str(PARAMS.positive_distance) + 'neg' + str(PARAMS.negative_distance) + '.pickle'
    generate_pickle_two_sequences(PARAMS.TRAIN_FOLDER, train_pickle)
    generate_pickle(PARAMS.VAL_FOLDER, val_pickle)
    PARAMS.train_file = train_pickle
    PARAMS.val_file = val_pickle
    PARAMS.use_image_features = True

    PARAMS.aug_mode = '6depths0.4'
    # PARAMS.aug_mode = 'only_best_effects0.5'
    use_features = ['use_magnitude_anglexy_gray', 'use_magnitude_anglexy_hue', 'use_magnitude_anglexy_hue_grey']
    #use_features = ['use_magnitude_anglexy_hue']
    for feature in use_features:
        if feature == 'use_ones':
            PARAMS.use_gradients = False
        else:
            PARAMS.use_gradients = True
            setattr(PARAMS, feature, True)
            print('Feature: ', feature)
            # set the other features to False
            for f in use_features:
                if f != feature:
                    setattr(PARAMS, f, False)
                
        # set up dataloaders
        dataloaders = make_dataloaders()
        print('Loading evaluation dataset...')
        generate_test_pickle(PARAMS.dataset_folder)
        evaluation_set_fa = EvaluationDataset(PARAMS.dataset_folder)
        # replace FRIBURGO_A with FRIBURGO_B
        PARAMS.test_folder = dataset_folder.replace('FRIBURGO_A', 'FRIBURGO_B')
        generate_test_pickle(PARAMS.test_folder)
        evaluation_set_fb = EvaluationDataset(PARAMS.test_folder)
        PARAMS.test_folder = dataset_folder.replace('FRIBURGO_A', 'SAARBRUCKEN_A')
        generate_test_pickle(PARAMS.test_folder)
        evaluation_set_sa = EvaluationDataset(PARAMS.test_folder)
        PARAMS.test_folder = dataset_folder.replace('FRIBURGO_A', 'SAARBRUCKEN_B')
        generate_test_pickle(PARAMS.test_folder)
        evaluation_set_sb = EvaluationDataset(PARAMS.test_folder)
        print('Evaluation dataset loaded.')       
        
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
        if PARAMS.use_magnitude_hue or PARAMS.use_magnitude_ones or PARAMS.use_anglexy or PARAMS.use_xymag:
            model.conv0p1s1 = ME.MinkowskiConvolution(
                2, 32, kernel_size=5, dimension=3)
        elif PARAMS.use_rgb or PARAMS.use_anglexy_hue or PARAMS.use_anglexy_ones or PARAMS.use_magnitude_anglexy or PARAMS.use_magnitude_angle_hue:
            model.conv0p1s1 = ME.MinkowskiConvolution(    
                3, 32, kernel_size=5, dimension=3)
        elif PARAMS.use_magnitude_anglexy_hue or PARAMS.use_magnitude_angle_hue_grey or PARAMS.use_magnitude_anglexy_gray:
            model.conv0p1s1 = ME.MinkowskiConvolution(
                4, 32, kernel_size=5, dimension=3)
        elif PARAMS.use_magnitude_anglexy_hue_ones or PARAMS.use_magnitude_anglexy_hue_grey:
            model.conv0p1s1 = ME.MinkowskiConvolution(
                5, 32, kernel_size=5, dimension=3)
        elif PARAMS.use_magnitude_angle_hue_rgb:
            model.conv0p1s1 = ME.MinkowskiConvolution(
                6, 32, kernel_size=5, dimension=3)
        elif PARAMS.use_magnitude_anglexy_hue_rgb:
            model.conv0p1s1 = ME.MinkowskiConvolution(
                7, 32, kernel_size=5, dimension=3)
        
        do_train(model, dataloaders, evaluation_set_fa, evaluation_set_fb, evaluation_set_sa, evaluation_set_sb)

        # empty cache
        torch.cuda.empty_cache()
                            
                        
