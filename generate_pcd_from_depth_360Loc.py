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

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config.config import PARAMS


# Add the parent directory to sys.path

def load_model_by_name(arch_name, checkpoint_path, device):
    model_kwargs = dict(
        vits=dict(
            encoder='vits', 
            features=64,
            out_channels=[48, 96, 192, 384]
        ),
        vitb=dict(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768],
        ),
        vitl=dict(
            encoder="vitl", 
            features=256, 
            out_channels=[256, 512, 1024, 1024], 
            use_bn=False, 
            use_clstoken=False, 
            max_depth=150.0, 
            mode='disparity',
            pretrain_type='dinov2',
            del_mask_token=False
        )
    )

    # Load model
    if arch_name == 'depthanything-large':
        model = DepthAnything(**model_kwargs['vitl']).to(device)
        # checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"large/model.safetensors", repo_type="model")

    elif arch_name == 'depthanything-base':
        model = DepthAnythingV2(**model_kwargs['vitb']).to(device)
        # checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"base/model.safetensors", repo_type="model")
    elif arch_name == 'depthanything-small':
        model = DepthAnythingV2(**model_kwargs['vits']).to(device)
        # checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"base/model.safetensors", repo_type="model")

    else:
        raise NotImplementedError(f"Unknown architecture: {arch_name}")
    
    # safetensors 
    model_weights = load_file(checkpoint_path)
    model.load_state_dict(model_weights)
    del model_weights
    torch.cuda.empty_cache()
    return model

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



# def get_pointcloud_cylindrical_v2(color_image, depth, mask, scale_y=0.015, depth_min=1, depth_max=40.0):
#     width, height = color_image.size
#     # Apply mask to depth    
#     # Convertir depth a float32 para poder asignar NaN
#     depth = depth.astype(np.float32)
    
#     # Apply mask to depth    
#     depth[mask] = np.nan
#     # normalize depth to 0-255
#     depth = (depth - np.nanmin(depth)) / (np.nanmax(depth) - np.nanmin(depth)) * 255.0
#     #invert depth
#     depth = 255.0 - depth
#     depth = depth.astype(np.uint8)
#     depth = depth[:,:,0]
    

#     # Resize depth prediction to match the original image size
#     resized_pred = Image.fromarray(depth).resize((width, height), Image.NEAREST)

#     # Convert depth from 0-255 to actual depth values in meters
#     z = np.array(resized_pred)
#     z = depth_min + (z / 255.0) * (depth_max - depth_min)

#     # Map each pixel (u, v) in the panorama to cylindrical coordinates
#     # u -> azimuth (theta), v -> height (y)
#     theta = (np.arange(width) / width) * 2 * np.pi  # azimuthal angle (theta)
#     y = (np.arange(height) - height / 2) * scale_y  # convert pixel height to meters

#     theta, y = np.meshgrid(theta, y)

#     # Convert cylindrical coordinates to Cartesian coordinates
#     x = z * np.sin(theta)
#     z = z * np.cos(theta)

#     # Stack x, y, z into a list of 3D points
#     points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
#     # transform points to the world coordinates
#     # apply a rotation matrix of 90 degrees around the x-axis
#     rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
#     points = np.dot(points, np.linalg.inv(rotation_matrix).T)

#     # Normalize color image and flatten
#     colors = np.array(color_image).reshape(-1, 3) / 255.0

#     # Create the point cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     o3d.visualization.draw_geometries([pcd])


#     return pcd


def get_pointcloud_cylindrical_v2(color_image, depth, mask, scale_y=0.015, depth_min=1, depth_max=40.0):
    width, height = color_image.size
    
    # Convertir depth a float32 para poder asignar NaN
    depth = depth.astype(np.float32)

    #dilate mask
    mask = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Apply mask to depth    
    depth[mask] = np.nan
    
    # normalize depth to 0-255
    depth = (depth - np.nanmin(depth)) / (np.nanmax(depth) - np.nanmin(depth)) * 255.0
    
    # invert depth
    depth = 255.0 - depth
    depth = depth.astype(np.uint8)
    depth = depth[:,:,0]
    
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
    
    # Resize mask to match the resized depth
    mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize((width, height), Image.NEAREST)
    mask_resized = np.array(mask_resized) > 127  # Convertir de vuelta a booleano
    
    # Filtrar puntos que corresponden a la máscara
    mask_flat = mask_resized.reshape(-1)
    points = points[~mask_flat]  # Mantener solo los puntos donde mask es False
    
    # transform points to the world coordinates
    # apply a rotation matrix of 90 degrees around the x-axis
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    points = np.dot(points, np.linalg.inv(rotation_matrix).T)

    # Normalize color image and flatten
    colors = np.array(color_image).reshape(-1, 3) / 255.0
    
    # Filtrar colores correspondientes a la máscara
    colors = colors[~mask_flat]

    # Create the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Mostrar solo si hay puntos válidos
    if len(points) > 0:
        o3d.visualization.draw_geometries([pcd])

    return pcd


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

    # o3d.visualization.draw_geometries([pcd])

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
        device = PARAMS.cuda_device
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
        device = PARAMS.cuda_device
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



def process_pcd(src_color_dir, depth_dir, pcd_dir, device):
    plt.switch_backend('TkAgg')  # Specify the backend
    """
    Procesa la imagen y la guarda en la misma ubicación
    con '_depth' agregado al nombre del archivo.
    """
    # check if already processed


    if os.path.exists(pcd_dir):
        print(f"La point cloud {pcd_dir} ya existe.")
        return

    
    image_color = cv2.imread(src_color_dir, cv2.IMREAD_UNCHANGED) 
    image_depth = cv2.imread(depth_dir, cv2.IMREAD_UNCHANGED)  
    img_shape = image_color.shape
    # image_color = image_color[0:img_shape[0]-500, :]
    # image_depth = image_depth[0:img_shape[0]-500, :]
    # image color is in BGR format, convert to RGB
    image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB) 

    #resize image to 192x384 (/16)
    image_color = cv2.resize(image_color, (384, 192), interpolation=cv2.INTER_AREA)
    image_depth = cv2.resize(image_depth, (384, 192), interpolation=cv2.INTER_AREA)




    # image_depth = image_depth[0:img_shape[0]-500, :]
    # 1. Cargar imagen
    # img = cv2.imread(depth_dir, cv2.IMREAD_GRAYSCALE)
    # crop the image to remove bottom part (500 pixels)
    # img_shape = img.shape
    # img = img[0:img_shape[0]-500, :]

    # # 2. Preparar los datos para K-Means
    # # Convertimos la imagen (matriz 2D) en un vector columna (1D) de tipo float32
    # pixel_values = img.reshape((-1, 1))
    # pixel_values = np.float32(pixel_values)

    # # 3. Definir criterios de parada del algoritmo (para que no tarde infinito)
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    # # 4. Aplicar K-Means
    # # K = 3 (intentamos separar: Fondo Negro, Objetos Grises, Plataforma Blanca)
    # k = 3
    # _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # centers = np.uint8(centers)
    # indice_cluster_cercano = np.argmax(centers) 

    # print(f"Centros encontrados (Intensidades): {centers.flatten()}")
    # print(f"El cluster a eliminar es el índice: {indice_cluster_cercano} (Valor: {centers[indice_cluster_cercano]})")

    # # 6. Crear la máscara
    # # Reconstruimos la imagen de etiquetas para que tenga forma visual
    # labels_img = labels.reshape(img.shape)  
    # mascara = np.where(labels_img == indice_cluster_cercano, 0, 255).astype('uint8')
    # # mascara negada:
    # mascara_negada = np.bitwise_not(mascara)
    # # pasar a booleano
    # mascara_negada_bool = mascara_negada.astype(bool)
    # pcd = get_pointcloud_cylindrical_v2(Image.fromarray(image_color, 'RGB'), image_depth.astype(np.uint8), mascara_negada_bool)

    image_depth = image_depth.astype(np.uint8)
    image_depth = 255.0 - image_depth
    #quedate con un solo canal al ser en escala de grises
    image_depth = image_depth[:,:,0]

    # equalize depth image histogram
    image_depth = cv2.equalizeHist(image_depth.astype(np.uint8))


    # show images

    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 3, 1)
    # plt.title("Imagen Color")
    # plt.imshow(image_color)
    # plt.subplot(1, 3, 2)
    # plt.title("Imagen Depth")
    # plt.imshow(image_depth, cmap='gray')
    # plt.subplot(1, 3, 3)
    # plt.title("Histograma Depth")
    # plt.hist(image_depth.ravel(), bins=256, range=(0, 256))
    # plt.show()

    pcd = get_pointcloud_cylindrical(Image.fromarray(image_color, 'RGB'), image_depth.astype(np.uint8))
    
    
    
    
    # # dilata la mascara negada
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # mascara_negada = cv2.dilate(mascara_negada, kernel, iterations=4)
    # image_depth = inpaint_with_lama(lama_model, img.astype(np.uint8), mascara_negada.astype(np.uint8))
 

    # mascara_2 = np.where(labels_img == segundo_indice, 0, 255).astype('uint8')

    # # 7. Aplicar la máscara a la imagen original
    # img_filtrada = cv2.bitwise_and(img, img, mask=mascara)

    # # --- VISUALIZACIÓN ---
    # plt.figure(figsize=(12, 6))

    # plt.subplot(1, 3, 1)
    # plt.title("Mapa Original")
    # plt.imshow(img, cmap='gray')

    # plt.subplot(1, 3, 2)
    # plt.title("Cluster más cercano (Rojo)")
    # # Truco visual: mostramos la máscara sobre la imagen
    # vis_mask = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # vis_mask[labels_img == indice_cluster_cercano] = [255, 0, 0] # Pintar de rojo lo detectado
    # # vis_mask[labels_img == indice_cluster_cercano] = [np.nan, np.nan, np.nan] # Pintar de rojo lo detectado
    # # normalizar vis_mask para que se vea mejor
    # # vis_mask = (vis_mask - np.nanmin(vis_mask)) / (np.nanmax(vis_mask) - np.nanmin(vis_mask))
    # plt.imshow(vis_mask)

    # plt.subplot(1, 3, 3)
    # plt.title("Segundo indice (rojo)")
    # vis_mask2 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # vis_mask2[labels_img == segundo_indice] = [255, 0, 0] # Pintar de rojo lo detectado
    # plt.imshow(image_depth, cmap='gray')

    # plt.show()

    # mask = np.zeros([128, 512], dtype=bool)
    # mask[:, 370:400] = True
    # mask[90:128, 340:430] = True

    # image_depth = inpaint_with_lama(lama_model, image_depth.astype(np.uint8), mask.astype(np.uint8))
    # image_color = inpaint_with_lama(lama_model, image_color.astype(np.uint8), mask.astype(np.uint8))
  
    # image_depth = image_depth.astype(np.uint8)
    # image_depth = 255.0 - image_depth
    # #quedate con un solo canal al ser en escala de grises
    # image_depth = image_depth[:,:,0]
  
    # pcd = get_pointcloud_cylindrical(Image.fromarray(image_color, 'RGB'), image_depth.astype(np.uint8))
    #o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(pcd_dir, pcd)
    print(f"Point cloud processed and saved in {pcd_dir}")


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
   
 
    if torch.cuda.is_available():
        device = PARAMS.cuda_device
    else:
        device = "cpu"

    lama_config = '/home/arvc/Juanjo/develop/others_work/lama/big-lama/config.yaml'
    lama_checkpoint = '/home/arvc/Juanjo/develop/others_work/lama/big-lama/models/best.ckpt'
    lama_model = load_lama_model(lama_config, lama_checkpoint)


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
                process_pcd(lama_model, src_file_path, src_color_dir, depth_dir, pcd_dir, device)
          
            else:
                # Copia los archivos que no son imágenes sin procesar
                os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
                shutil.copy(src_file_path, dst_file_path)

if __name__ == "__main__":
    PARAMS.cuda_device = 'cuda:1'

    if torch.cuda.is_available():
        device = PARAMS.cuda_device
    else:
        device = "cpu"

    # lama_config = '/home/arvc/Juanjo/develop/others_work/lama/big-lama/config.yaml'
    # lama_checkpoint = '/home/arvc/Juanjo/develop/others_work/lama/big-lama/models/best.ckpt'
    # lama_model = load_lama_model(lama_config, lama_checkpoint)
    
    base_dir = '/media/arvc/DATOS1/Marcos/DATASETS/360LOC/'
    environments = os.listdir(base_dir)
    for environment in environments:
        map_dir = os.path.join(base_dir, environment, 'mapping')
        queries_dir = os.path.join(base_dir, environment, 'query_360')
        map_sequences = os.listdir(map_dir)
        # join 'image' in the root of map_sequences
        map_sequences = [os.path.join(map_dir, seq, 'image') for seq in map_sequences]
        query_sequences = os.listdir(queries_dir)
        query_sequences = [os.path.join(queries_dir, seq, 'image') for seq in query_sequences]
        
        # create a folder at the same level of 'image' called 'depth' and 'pcd'
        # exclude_directory_names = ['depth', 'pcd']
        import tqdm
        for seq in tqdm.tqdm(map_sequences + query_sequences):
            # if 'daytime_360_2' in seq:
            # for seq in map_sequences + query_sequences:
            os.makedirs(os.path.join(os.path.dirname(seq), 'DEPTH_DISTILL_ANY_DEPTH_LARGE'), exist_ok=True)
            os.makedirs(os.path.join(os.path.dirname(seq), 'PCD_DISTILL_ANY_DEPTH_LARGE'), exist_ok=True)
            # for file in os.listdir(seq):
            for file in tqdm.tqdm(os.listdir(seq)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    src_image_file = os.path.join(seq, file)
                    src_depth_file = os.path.join(os.path.dirname(seq), 'DEPTH_DISTILL_ANY_DEPTH_LARGE', file)
                    # src_depth_file = os.path.join(os.path.dirname(seq), 'depth', file)
                    dst_pcd_file = os.path.join(os.path.dirname(seq), 'PCD_DISTILL_ANY_DEPTH_LARGE', os.path.splitext(file)[0] + '.ply')
                    process_pcd(src_image_file, src_depth_file, dst_pcd_file, device)
                



 
