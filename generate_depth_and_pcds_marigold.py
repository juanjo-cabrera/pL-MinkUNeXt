import sys
import torch
import os
from PIL import Image
import shutil
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import yaml
# Add the parent directory to sys.path
sys.path.append('/home/arvc/Juanjo/develop/others_work/lama')
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint, make_training_model
# from saicinpainting.training.trainers
from saicinpainting.training.data.datasets import get_transforms
from torchvision import transforms
from torch.utils.data._utils.collate import default_collate

# Add the parent directory to sys.path
sys.path.append('/home/arvc/Juanjo/develop/others_work/Depth-Anything-V2')
# Add the parent directory to sys.path
sys.path.append('/home/arvc/Juanjo/develop/others_work/depth-fm')
sys.path.append('/home/arvc/Juanjo/develop/others_work/Marigold')
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config.config import PARAMS

# Add the parent directory to sys.path

def get_depthFM(raw_img):
    from depthfm import DepthFM
    if torch.cuda.is_available():
        device = PARAMS.device
    else:
        device = "cpu"

    model = DepthFM('/home/arvc/Juanjo/develop/others_work/depth-fm/checkpoints/depthfm-v1.ckpt')
    model.eval()
    model.to(device)
   
    raw_img = raw_img.to(device)

    # Generate depth
    model.model.dtype = torch.float16
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        depth = model.predict_depth(raw_img, num_steps=2, ensemble_size=4)
    depth = depth.squeeze(0).squeeze(0).cpu().numpy()   
    return depth

def get_depth_marigold(raw_img, pipe):
    device = PARAMS.device
    from marigold import MarigoldPipeline
    #dtype = torch.float32
    #variant = None
    #checkpoint_path = '/home/arvc/Juanjo/develop/others_work/Marigold/checkpoint/marigold-v1-0'
    #pipe = MarigoldPipeline.from_pretrained(
    #    checkpoint_path, variant=variant, torch_dtype=dtype
    #)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass

    pipe = pipe.to(device)


    # -------------------- Inference and saving --------------------
    with torch.no_grad():     
        # Read input image
        input_image = Image.fromarray(raw_img)

        # Predict depth
        """
        pipe_out = pipe(
            input_image,
            denoising_steps=4,
            ensemble_size=5,
            processing_res=0,
            match_input_res=True,
            batch_size=0,
            color_map=None,
            show_progress_bar=False,
            resample_method="bilinear",
        )
        """
        pipe_out = pipe(input_image)      

        depth = pipe_out.depth_np
        return depth


def get_depth_v2(raw_img, encoder='vits'):
    from depth_anything_v2.dpt import DepthAnythingV2
    if torch.cuda.is_available():
        device = PARAMS.device
    else:
        device = "cpu"
    # set cuda device
    torch.set_default_device(device)


    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}}


    #encoder = 'vits' # or 'vits', 'vitb', 'vitg'
    model = DepthAnythingV2(encoder=encoder, features=model_configs[encoder]['features'], out_channels=model_configs[encoder]['out_channels'])
    model.load_state_dict(torch.load(f'/home/arvc/Juanjo/develop/others_work/Depth-Anything-V2/depth_anything_v2_{encoder}.pth', map_location=device))
    model.eval()
    model.to(device)
    depth = model.infer_image(raw_img)  # HxW raw depth map
    return depth


def get_metric_depth_v2(raw_img):
    from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
    if torch.cuda.is_available():
        device = PARAMS.device
    else:
        device = "cpu"

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    encoder = 'vitl'  # or 'vits', 'vitb'
    dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20  # 20 for indoor model, 80 for outdoor model

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(
        torch.load(f'/home/arvc/Juanjo/develop/others_work/Depth-Anything-V2/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    model.eval()
    model.to(device)
    depth = model.infer_image(raw_img)  # HxW raw depth map
    return depth



def get_pointcloud_cylindrical(color_image, depth, scale_y=0.015, depth_min=1, depth_max=6.0):
    width, height = color_image.size

    # Resize depth prediction to match the original image size
    resized_pred = Image.fromarray(depth).resize((width, height), Image.NEAREST)

    # Convert depth from 0-255 to actual depth values in meters
    z = np.array(resized_pred)
    z = depth_min + (z / 255.0) * (depth_max - depth_min)

    # Map each pixel (u, v) in the panorama to cylindrical coordinates
    # u -> azimuth (theta), v -> height (y)
    theta = (np.arange(width) / width) * 2 * np.pi  # azimuthal angle (theta)
    y = (np.arange(height) - height / 2) * scale_y  # convert pixel height to meters

    theta, y = np.meshgrid(theta, y)

    # Convert cylindrical coordinates to Cartesian coordinates
    x = z * np.sin(theta)
    z = z * np.cos(theta)

    # Stack x, y, z into a list of 3D points
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    # transform points to the world coordinates
    # apply a rotation matrix of 90 degrees around the x-axis
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    points = np.dot(points, np.linalg.inv(rotation_matrix).T)

    # Normalize color image and flatten
    colors = np.array(color_image).reshape(-1, 3) / 255.0

    # Create the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd



def load_lama_model(config_path, checkpoint_path):
    from omegaconf import OmegaConf
    with open(config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'


    # model = make_training_model(train_config)
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    # model = torch.load(checkpoint_path['state_dict'], map_location='cpu')
    # model.load_state_dict(checkpoint_path['state_dict'])
    model.freeze()
    if torch.cuda.is_available():
        device = PARAMS.device
    else:
        device = "cpu"
    model.to(device)
    return model

def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod
def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')

def inpaint_with_lama(model, image, mask, device='cuda'):
    if torch.cuda.is_available():
        device = PARAMS.device
    else:
        device = "cpu"
    torch.set_default_device(device)

    # Asegúrate de que la imagen tenga tres canales (RGB)
    if image.ndim == 2:  # Si la imagen es de un solo canal (grises)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Pre-process image and mask
    image = np.transpose(image, (2, 0, 1))
    image = image.astype('float32') / 255
    mask = mask.astype('float32') / 255

    result = dict(image=image, mask=mask[None, ...])
    result['unpad_to_size'] = result['image'].shape[1:]
    result['image'] = pad_img_to_modulo(result['image'], mod=8)
    result['mask'] = pad_img_to_modulo(result['mask'], mod=8)
    batch = default_collate([result])

    with torch.no_grad():
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1
        batch = model(batch)
        cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
        unpad_to_size = batch.get('unpad_to_size', None)
        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]
        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
    return cur_res


def process_image_and_pcd(image_path, src_color_dir, depth_dir, pcd_dir, pipe):
    #plt.switch_backend('TkAgg')  # Specify the backend
    """
    Procesa la imagen y la guarda en la misma ubicación
    con '_depth' agregado al nombre del archivo.
    """
    # check if already processed

    try:
        new_image_path = image_path.replace(src_color_dir, depth_dir)     
        # Si ya existe, pasa a la siguiente imagen
        if os.path.exists(new_image_path):
            print(f"La imagen {new_image_path} ya existe.")
            # read the image
            image_depth = cv2.imread(new_image_path, cv2.IMREAD_UNCHANGED)
            image_color = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            #plt.imshow(image_depth, cmap='gray')
            #plt.show()
        else:
            image_color = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            # Aquí podrías aplicar algún filtro de procesamiento a la imagen
            #image_depth = get_depth_v2(image_color, encoder='vitb')
            image_depth = get_depth_marigold(image_color, pipe)
            image_depth = (image_depth - image_depth.min()) / (image_depth.max() - image_depth.min()) * 255.0
            image_depth = image_depth.astype(np.uint8)
            #processed_image_vitb = get_depth_v2(image_color, encoder='vitb')
            #processed_image_vitl = get_depth_v2(image_color, encoder='vitl')
            #image_depth = get_depth_marigold(image_color, pipe)
            # Muestra las 3 imágenes procesadas en una ventana
            """
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(processed_image_vits, cmap='gray')
            plt.title('VITS')
            plt.subplot(1, 3, 2)
            plt.imshow(processed_image_vitb, cmap='gray')
            plt.title('VITB')
            plt.subplot(1, 3, 3)
            plt.imshow(processed_image_vitl, cmap='gray')
            plt.title('VITL')
            plt.show()
            """
            #image_depth = processed_image_vitl.copy()

            #cv2.imshow('Processed Image', image_depth)    
            
                
            cv2.imwrite(new_image_path, image_depth)

            print(f"Imagen procesada y guardada en {new_image_path}")
    except Exception as e:
        print(f"Error al procesar la imagen {image_path}: {e}")

    new_pcd_path = image_path.replace(src_color_dir, pcd_dir) 
    new_pcd_path = os.path.splitext(new_pcd_path)[0] + ".ply"
    if os.path.exists(new_pcd_path):
        print(f"La point cloud {new_pcd_path} ya existe.")
        return
    try:
        mask = np.zeros([128, 512], dtype=bool)
        mask[:, 370:400] = True
        mask[90:128, 340:430] = True
        # plt.imshow(mask, cmap=plt.get_cmap('gray'))
        # plt.show()

        # Realizar el inpainting
        lama_config = '/home/arvc/Juanjo/develop/others_work/lama/big-lama/config.yaml'
        lama_checkpoint = '/home/arvc/Juanjo/develop/others_work/lama/big-lama/models/best.ckpt'
        lama_model = load_lama_model(lama_config, lama_checkpoint)
        # from one channel to three channels of image_depth
        #image_depth = cv2.cvtColor(image_depth, cv2.COLOR_GRAY2BGR)
        # or just replicate the channel to three channels
        image_depth = (image_depth - image_depth.min()) / (image_depth.max() - image_depth.min()) * 255.0
        image_depth = np.stack((image_depth,) * 3, axis=-1)
        #image_depth = cv2.imread('/media/arvc/DATOS/Juanjo/Datasets/Depth_Friburgo_small/TestCloudy/1PO-A/t1152902971.192228_x-8.615742_y2.881353_a2.980192.jpeg', cv2.IMREAD_UNCHANGED)
        image_depth = inpaint_with_lama(lama_model, image_depth.astype(np.uint8), mask.astype(np.uint8))
        image_color = inpaint_with_lama(lama_model, image_color.astype(np.uint8), mask.astype(np.uint8))
        # show the both inpainted images
        """
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image_color)
        plt.title('Inpainted Color Image')
        plt.subplot(1, 2, 2)
        plt.imshow(image_depth, cmap='gray')
        plt.title('Inpainted Depth Image')
        plt.show()
     
       """
        
        image_depth = (image_depth - image_depth.min()) / (image_depth.max() - image_depth.min()) * 255.0
        image_depth = image_depth.astype(np.uint8)
        image_depth = 255.0 - image_depth
        #quedate con un solo canal al ser en escala de grises
        image_depth = image_depth[:,:,0]

        # convert img_cloudy_rgb to RGB format
        # img_cloudy_rgb = cv2.cvtColor(img_cloudy_rgb, cv2.COLOR_BGR2RGB)
        pcd = get_pointcloud_cylindrical(Image.fromarray(image_color, 'RGB'), image_depth.astype(np.uint8))
        # show the point cloud
        #o3d.visualization.draw_geometries([pcd])

        # Guardar la nube de puntos en formato PLY
        new_pcd_path = image_path.replace(src_color_dir, pcd_dir) 
        new_pcd_path = os.path.splitext(new_pcd_path)[0] + ".ply"
        o3d.io.write_point_cloud(new_pcd_path, pcd)
        # pcd.save(new_pcd_path)

        print(f"Imagen procesada y guardada en {new_pcd_path}")
    except Exception as e:
        print(f"Error al procesar la imagen {image_path}: {e}")

def global_normalize(pcd):
    import copy
    """
    Normalize a pointcloud to achieve mean zero, scaled between [-1, 1] and with a fixed number of points
    """
    pcd = copy.deepcopy(pcd)
    # points = np.asarray(self.pointcloud_normalized.points)
    points = np.asarray(pcd.points)

    [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)

    x = x - x_mean
    y = y - y_mean
    z = z - z_mean

    x = x / 14 * 50
    y = y / 14 * 50
    z = z / 14 * 50

    points[:, 0] = x
    points[:, 1] = y
    points[:, 2] = z

    pointcloud_normalized = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return pointcloud_normalized

def process_image(image_path, src_dir, dst_dir):
    """
    Procesa la imagen y la guarda en la misma ubicación
    con '_depth' agregado al nombre del archivo.
    """
    try:
        new_image_path = image_path.replace(src_dir, dst_dir)     
        # Si ya existe, pasa a la siguiente imagen
        if os.path.exists(new_image_path):
            print(f"La imagen {new_image_path} ya existe.")
            return
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # Aquí podrías aplicar algún filtro de procesamiento a la imagen
        processed_image = get_depth_v2(image)
        # Muestra la imagen procesada
        #cv2.imshow('Processed Image', processed_image)    
        
           
        cv2.imwrite(new_image_path, processed_image)

        print(f"Imagen procesada y guardada en {new_image_path}")
    except Exception as e:
        print(f"Error al procesar la imagen {image_path}: {e}")



def copy_structure_and_process_images(src_color_dir, depth_dir, pcd_dir, exclude_dir_names, depth='depth'):
    """
    Copia la estructura de carpetas de src_dir a dst_dir y procesa las imágenes,
    excluyendo las carpetas con el nombre dado.

    Primero comprueba si ya existen las carpetas de destino, si no, las crea.
    """
    from marigold import MarigoldPipeline
    dtype = torch.float32
    variant = None
    checkpoint_path = '/home/arvc/Juanjo/develop/others_work/Marigold/checkpoint/marigold-lcm-v1-0'
    pipe = MarigoldPipeline.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype)
    for root, dirs, files in os.walk(src_color_dir):
        # Excluir la carpetas especificadas
        #dirs[:] = [d for d in dirs if d != exclude_dir_name]
        for exclude_dir_name in exclude_dir_names:
            if exclude_dir_name in dirs:
                dirs.remove(exclude_dir_name)

        # Crea las carpetas en el directorio de destino
        for dir in dirs:
            # Si la carpeta no existe, la crea    
            os.makedirs(os.path.join(pcd_dir, os.path.relpath(os.path.join(root, dir), src_color_dir)), exist_ok=True)
            os.makedirs(os.path.join(depth_dir, os.path.relpath(os.path.join(root, dir), src_color_dir)), exist_ok=True)

        for file in files:
            src_file_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_file_path, src_color_dir)
            dst_file_path = os.path.join(pcd_dir, rel_path)

            # Procesa las imágenes y guarda en el directorio destino
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                process_image_and_pcd(src_file_path, src_color_dir, depth_dir, pcd_dir, pipe)
          
            else:
                # Copia los archivos que no son imágenes sin procesar
                os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
                shutil.copy(src_file_path, dst_file_path)

if __name__ == "__main__":

    environments = ['FRIBURGO_A', 'FRIBURGO_B', 'SAARBRUCKEN_A', 'SAARBRUCKEN_B']
    # environments = ['FRIBURGO_A']
    for environment in environments:
        # Directorio fuente y destino
        src_color_directory  = '/media/arvc/DATOS/Marcos/DATASETS/COLD/' + environment + '/'
        pcd_directory = '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_MARIGOLD/' + environment + '/'
        depth_directory = '/media/arvc/DATOS/Juanjo/Datasets/COLD/DEPTH_MARIGOLD/' + environment + '/'
        exclude_directory_names = ['RepImages', 'RepresentativeImages', 'fr_seq2_cloudy3', 'Train2']

        copy_structure_and_process_images(src_color_directory, depth_directory, pcd_directory, exclude_directory_names)