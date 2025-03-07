import yaml
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
class Config():
    def __init__(self, yaml_file=os.path.join(current_directory, 'general_parameters.yaml')):
        # print(current_directory)
        # print(os.path.join(current_directory, 'general_parameters.yaml'))
        # # print(os.path.dirname(os.path.realpath(__file__)))
        # print(os.getcwd())
        with open(yaml_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            print(config)

            self.dataset_folder= config.get('dataset_folder')
            self.map_dir = self.dataset_folder + 'TrainingBaseline/'
            self.testing_cloudy_dir = self.dataset_folder + 'TestCloudy/'
            self.testing_night_dir = self.dataset_folder + 'TestNight/'
            self.testing_sunny_dir = self.dataset_folder + 'TestSunny/'
            self.cuda_device = config.get('cuda_device')

            self.positive_distance = config.get('positive_distance')
            self.negative_distance = config.get('negative_distance')
            self.voxel_size = config.get('voxel_size')
            self.max_distance = config.get('max_distance')
            self.height = config.get('height')
            self.use_rgb = config.get('use_rgb')
            self.use_gray = config.get('use_gray')
            self.use_dino_features = config.get('use_dino_features')
            self.use_depth_features = config.get('use_depth_features')
            self.use_hue = config.get('use_hue')
            self.use_video = config.get('use_video')
            self.use_gradients = config.get('use_gradients')

            self.save_best = config.get('save_best')
            self.quantization_size = config.get('quantization_size')
            self.num_workers = config.get('num_workers')
            self.batch_size = config.get('batch_size')
            self.batch_size_limit = config.get('batch_size_limit')
            self.batch_expansion_rate = config.get('batch_expansion_rate')
            self.batch_expansion_th = config.get('batch_expansion_th')
            self.batch_split_size = config.get('batch_split_size')
            self.val_batch_size = config.get('val_batch_size')
            

            self.optimizer = config.get('optimizer')
            self.initial_lr = config.get('initial_lr')
            self.scheduler = config.get('scheduler')
            self.aug_mode = config.get('aug_mode')
            self.weight_decay = config.get('weight_decay')
            self.loss = config.get('loss')
            self.margin = config.get('margin')
            self.tau1 = config.get('tau1')
            self.positives_per_query = config.get('positives_per_query')
            self.similarity = config.get('similarity')
            self.normalize_embeddings = config.get('normalize_embeddings')

            self.protocol = config.get('protocol')

            if self.protocol == 'baseline':
                self.epochs = config.get('baseline').get('epochs')
                self.scheduler_milestones = config.get('baseline').get('scheduler_milestones')
                self.train_file = config.get('baseline').get('train_file')
                self.val_file = config.get('baseline').get('val_file')
                self.train_folder = config.get('baseline').get('train_folder')
                self.val_folder = config.get('baseline').get('val_folder')
            
            self.p_depth = config.get('data_augmentation').get('p_depth')
            self.p_others = config.get('data_augmentation').get('p_others')

            self.print_model_info = config.get('print').get('model_info')
            self.print_model_parameters = config.get('print').get('number_of_parameters')
            self.debug = config.get('print').get('debug')
            self.weights_path = config.get('evaluate').get('weights_path')

PARAMS = Config()


