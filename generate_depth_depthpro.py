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

# Add the parent directory to sys.path
sys.path.append('/home/arvc/Juanjo/develop/others_work/Depth-Anything-V2')

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config.config import PARAMS

# Add the parent directory to sys.path



def get_depth_v2(raw_img):
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


    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'
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


def process_image_and_pcd(model, image_processor, image_path, src_color_dir, depth_dir, pcd_dir, device):
    # plt.switch_backend('TkAgg')  # Specify the backend
    """
    Procesa la imagen y la guarda en la misma ubicación
    con '_depth' agregado al nombre del archivo.
    """
    # check if already processed

    
    new_image_path = image_path.replace(src_color_dir, depth_dir)     
    # Si ya existe, pasa a la siguiente imagen
    # if os.path.exists(new_image_path):
    #     print(f"La imagen {new_image_path} ya existe.")
    #     return
    image_color = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = Image.open(image_path) 
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    post_processed_output = image_processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)],
    )
    image_depth = post_processed_output[0]["predicted_depth"]
    image_depth = 255 - image_depth 
    image_depth = (image_depth - image_depth.min()) / (image_depth.max() - image_depth.min()) * 255.0    
    image_depth = image_depth.cpu().numpy().astype(np.uint8) 
    # Muestra la imagen procesada
    # cv2.imshow('Depth Image', image_depth)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(new_image_path, image_depth)

    print(f"Imagen procesada y guardada en {new_image_path}")
  


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
    from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
    # from depth_anything_v2.dpt import DepthAnythingV2
    if torch.cuda.is_available():
        device = PARAMS.cuda_device
    else:
        device = "cpu"
    # set cuda device
    torch.set_default_device(device)


    # model_configs = {
    # 'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    # 'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    # 'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    # 'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}}


    # #encoder = 'vits' # or 'vits', 'vitb', 'vitg'
    # encoder = 'vitl'  # or 'vits', 'vitb'
    # model = DepthAnythingV2(encoder=encoder, features=model_configs[encoder]['features'], out_channels=model_configs[encoder]['out_channels'])
    # print(model)

    # model.load_state_dict(torch.load(f'/home/arvc/Juanjo/develop/others_work/Depth-Anything-V2/depth_anything_v2_{encoder}.pth', map_location=device))
    # model.to(device).eval()

    image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)
    """
    Copia la estructura de carpetas de src_dir a dst_dir y procesa las imágenes,
    excluyendo las carpetas con el nombre dado.

    Primero comprueba si ya existen las carpetas de destino, si no, las crea.
    """
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
                process_image_and_pcd(model, image_processor, src_file_path, src_color_dir, depth_dir, pcd_dir, device)
          
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
        pcd_directory = '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_DEPTH_PRO/' + environment + '/'
        depth_directory = '/media/arvc/DATOS/Juanjo/Datasets/COLD/DEPTH_DEPTH_PRO/' + environment + '/'
        exclude_directory_names = ['RepImages', 'RepresentativeImages', 'fr_seq2_cloudy3', 'Train2']

        copy_structure_and_process_images(src_color_directory, depth_directory, pcd_directory, exclude_directory_names)
