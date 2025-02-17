# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# For information on dataset see: https://github.com/mikacuy/pointnetvlad


import torchvision.transforms as transforms
import numpy as np
import random

from datasets.augmentation import *
from datasets.base_datasets import TrainingDataset
from datasets.freiburg.pnv_raw import PNVPointCloudLoader


class PNVTrainingDataset(TrainingDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_loader = PNVPointCloudLoader()


class TrainTransform:
    # Augmentations specific for PointNetVLAD datasets (RobotCar and Inhouse)
    def __init__(self, aug_mode):
        self.aug_mode = aug_mode
        if self.aug_mode == 1 or self.aug_mode == 2:
            # Augmentations without random rotation around z-axis
            
            t = [JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                 RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4)]            
            """
          
            t = [JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                 RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4), RandomRotation_mod(max_theta=20, p=0.5, axis=np.array([0, 0, 1]))]
            """
          
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 0: # No augmentation
            self.transform = None
        elif self.aug_mode == '3depths0.7' or self.aug_mode == '3depths0.8' or self.aug_mode == '3depths0.9':
            self.transform = None
        elif self.aug_mode == 3:
            t = [JitterPoints(sigma=0.001, clip=0.002)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 4:
            t = [RemoveRandomPoints(r=(0.0, 0.1))]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 5:
            t = [RandomTranslation(max_delta=0.01)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 6:
            # Augmentations with random rotation around z-axis
            t = [RemoveRandomBlock(p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 7:
            self.transform = None
        elif self.aug_mode == 8:
            self.transform = None
        elif self.aug_mode == 9:
            t = [RandomScale(min=0.9, max=1.1)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 10:
            t = [RandomShear()]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 11:
            t = [RandomFlip([0.25, 0.25, 0.])]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 12:
            t = [MoveRandomBlock(p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 13:
            t = [ElasticDistortion()]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 14:
            t = [RandomOcclusion(p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 15:
            t = [AddRandomNoise()]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 16:
            t = [RandomDropout()]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 17:
            t = [RandomTranslation(max_delta=0.1)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 18:
            t = [RandomTranslation(max_delta=0.5)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 19:
            # Augmentations with random rotation around z-axis
            t = [RemoveRandomBlock(p=1.1)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 20:
            t = [MoveRandomBlock(p=1.1)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 21:
            t = [RandomScale(min=0.5, max=1.5)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 22:
            t = [RemoveRandomBlock(p=1.1, scale=(0.02, 0.5), ratio=(0.3, 3.3))]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 23:
            t = [RandomTranslation(max_delta=0.25)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 24:
            t = [RemoveRandomPoints(r=(0.0, 0.5))]
            self.transform = transforms.Compose(t)

        elif self.aug_mode == 25:
            t = [RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3))]
            self.transform = transforms.Compose(t)

        elif self.aug_mode == 'jitter':
            t = [JitterPoints(sigma=0.01, clip=0.02, probability=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'remove_points':
            t = [RemoveRandomPoints(r=(0.0, 0.4), p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'remove_points2':
            t = [RemoveRandomPoints(r=(0.4, 0.6), p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'translation':
            t = [RandomTranslation(max_delta=0.05, p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'translation_xy':
            t = [RandomTranslationB(max_delta=0.05, p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'remove_block':
            t = [RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3))]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'move_block':
            t = [MoveRandomBlock(p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'scale':
            t = [RandomScale(min=0.9, max=1.1, p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'scale_xy':
            t = [RandomScaleXY(min=0.9, max=1.1, p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'scale_xy2':
            t = [RandomScaleXY(min=0.8, max=1.2, p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'scale_xy3':
            t = [RandomScaleXY(min=0.7, max=1.3, p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'move_block2':
            t = [MoveRandomBlock2(p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'random_rotation':
            t = [RandomRotation_mod(max_theta=180, p=0.4, axis=np.array([0, 0, 1]))]
            self.transform = transforms.Compose(t)

        elif self.aug_mode == 'all_effects1':
            t = [JitterPoints(sigma=0.001, clip=0.002, probability=0.1), RemoveRandomPoints(r=(0.0, 0.5), p=0.1),
                 RandomTranslation(max_delta=0.01, p=0.1), RemoveRandomBlock(p=0.1, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock(p=0.1), RandomScale(min=0.9, max=1.1, p=0.1)]
            self.transform = transforms.Compose(t)

        # best all effects (remove random points, random translation xy, remove random block, move random block 2, random scale xy 2)
        elif self.aug_mode == 'only_best_effects' or self.aug_mode == 'only_best_effects0.5':
            t = [RemoveRandomPoints(r=(0.0, 0.4), p=0.1), RandomTranslationB(max_delta=0.05, p=0.1), RemoveRandomBlock(p=0.1, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock2(p=0.15), RandomScaleXY(min=0.8, max=1.2, p=0.1), RandomRotation_mod(max_theta=180, p=0.05, axis=np.array([0, 0, 1]))]
            self.transform = transforms.Compose(t)

        elif self.aug_mode == 'best_effects1': # remove random block, move random block 2, random scale xy 2
            t = [RemoveRandomBlock(p=0.13, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock2(p=0.13), RandomScaleXY(min=0.8, max=1.2, p=0.13)]
            self.transform = transforms.Compose(t)

        elif self.aug_mode == 'best_effects2': # remove random block, move random block 2, random scale xy 2
            t = [RemoveRandomBlock(p=0.15, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock2(p=0.1), RandomScaleXY(min=0.8, max=1.2, p=0.15)]
            self.transform = transforms.Compose(t)

        elif self.aug_mode == 'best_effects3': # remove random block, move random block 2, random scale xy 2
            t = [RemoveRandomBlock(p=0.2, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock2(p=0.2), RandomScaleXY(min=0.8, max=1.2, p=0.2)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'best_effects4': # remove random block, move random block 2, random scale xy 2
            t = [RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock2(p=0.4), RandomScaleXY(min=0.8, max=1.2, p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'best_effects5': # remove random block, move random block 2, random scale xy 2
            t = [RemoveRandomBlock(p=0.25, scale=(0.02, 0.5), ratio=(0.3, 3.3)), RandomScaleXY(min=0.8, max=1.2, p=0.25)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'best_effects6': # remove random block, move random block 2, random scale xy 2
            t = [RemoveRandomBlock(p=0.5, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock2(p=0.2), RandomScaleXY(min=0.8, max=1.2, p=0.5)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'best_effects7': # remove random block, move random block 2, random scale xy 2
            t = [RemoveRandomBlock(p=0.3, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock2(p=0.3), RandomScaleXY(min=0.8, max=1.2, p=0.3)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'best_effects8': # remove random block, move random block 2, random scale xy 2
            t = [RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3)), RandomScaleXY(min=0.8, max=1.2, p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'da2best1': # remove random block, move random block 2, random scale xy 2
            t = [DA2best(p_both=0.4, p_each = 0.5, scale=(0.02, 0.5), ratio=(0.3, 3.3), min=0.8, max=1.2)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'da2best2': # remove random block, move random block 2, random scale xy 2
            t = [DA2best(p_both=0.4, p_each = 0.6, scale=(0.02, 0.5), ratio=(0.3, 3.3), min=0.8, max=1.2)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'da2best3': # remove random block, move random block 2, random scale xy 2
            t = [DA2best(p_both=0.4, p_each = 0.7, scale=(0.02, 0.5), ratio=(0.3, 3.3), min=0.8, max=1.2)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'da2best4': # remove random block, move random block 2, random scale xy 2
            t = [DA2best(p_both=0.4, p_each = 0.8, scale=(0.02, 0.5), ratio=(0.3, 3.3), min=0.8, max=1.2)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'da2best5': # remove random block, move random block 2, random scale xy 2
            t = [DA2best(p_both=0.4, p_each = 0.9, scale=(0.02, 0.5), ratio=(0.3, 3.3), min=0.8, max=1.2)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'da2best6': # remove random block, move random block 2, random scale xy 2
            t = [DA2best(p_both=0.4, p_each = 0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3), min=0.8, max=1.2)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'da2best7': # remove random block, move random block 2, random scale xy 2
            t = [DA2best(p_both=0.4, p_each = 0.3, scale=(0.02, 0.5), ratio=(0.3, 3.3), min=0.8, max=1.2)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'da2best8': # remove random block, move random block 2, random scale xy 2
            t = [DA2best(p_both=0.4, p_each = 0.2, scale=(0.02, 0.5), ratio=(0.3, 3.3), min=0.8, max=1.2)]
            self.transform = transforms.Compose(t)
            
        elif self.aug_mode == 'da2best9': # remove random block, move random block 2, random scale xy 2
            t = [DA2best(p_both=0.5, p_each = 0.6, scale=(0.02, 0.5), ratio=(0.3, 3.3), min=0.8, max=1.2)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'da2best10': # remove random block, move random block 2, random scale xy 2
            t = [DA2best(p_both=0.6, p_each = 0.6, scale=(0.02, 0.5), ratio=(0.3, 3.3), min=0.8, max=1.2)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'da2best11': # remove random block, move random block 2, random scale xy 2
            t = [DA2best(p_both=0.7, p_each = 0.6, scale=(0.02, 0.5), ratio=(0.3, 3.3), min=0.8, max=1.2)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'da2best12': # remove random block, move random block 2, random scale xy 2
            t = [DA2best(p_both=0.8, p_each = 0.6, scale=(0.02, 0.5), ratio=(0.3, 3.3), min=0.8, max=1.2)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'da2best13': # remove random block, move random block 2, random scale xy 2
            t = [DA2best(p_both=0.3, p_each = 0.6, scale=(0.02, 0.5), ratio=(0.3, 3.3), min=0.8, max=1.2)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 'da2best14': # remove random block, move random block 2, random scale xy 2
            t = [DA2best(p_both=0.2, p_each = 0.6, scale=(0.02, 0.5), ratio=(0.3, 3.3), min=0.8, max=1.2)]
            self.transform = transforms.Compose(t)

            
            
            
        
        elif self.aug_mode == 26:
            t = [MoveRandomBlock(p=0.6)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 27:
            t = [MoveRandomBlock(p=0.6), RemoveRandomBlock(p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 28:
            t = [MoveRandomBlock(p=0.6), RemoveRandomBlock(p=0.4), RemoveRandomPoints(r=(0.0, 0.5))]      
          
            self.transform = transforms.Compose(t)

        elif self.aug_mode == 29:
            t = [RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock(p=0.8)]
            self.transform = transforms.Compose(t)  
        elif self.aug_mode == 30:
            t = [MoveRandomBlock(p=0.8)]          
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 31:
            t = [RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3))]
            self.transform = transforms.Compose(t)


        elif self.aug_mode == 32:
            # Individually apply each augmentation with probability 0.5 to RemoveRandomBlock and the rest with probability 1.0
            t = [JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.5)),
                 RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4, scale=(0.02, 0.5))]       
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 33:
            t = [JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                 RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4, scale=(0.02, 0.5))]    
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 34:
            # todos los efectos
            t = [JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.5)),
                 RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock(p=0.4), RandomScale(min=0.9, max=1.1)]    
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 35:
            # todos los efectos con p=0.4
            t = [JitterPoints(sigma=0.001, clip=0.002, probability=0.4), RemoveRandomPoints(r=(0.0, 0.5), p=0.4),
                 RandomTranslation(max_delta=0.01, p=0.4), RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock(p=0.4), RandomScale(min=0.9, max=1.1, p=0.4)]    
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 36:
            t = [RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock(p=1.1)]
            self.transform = transforms.Compose(t)  
        elif self.aug_mode == 37:
            # todos los efectos
            t = [RemoveRandomPoints(r=(0.0, 0.5)),
                 RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock(p=0.4), RandomScale(min=0.9, max=1.1)]    
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 38:
            # todos los efectos con p=0.4
            t = [RemoveRandomPoints(r=(0.0, 0.5), p=0.4),
                 RandomTranslation(max_delta=0.01, p=0.4), RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock(p=0.4), RandomScale(min=0.9, max=1.1, p=0.4)]    
            self.transform = transforms.Compose(t)
        
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


