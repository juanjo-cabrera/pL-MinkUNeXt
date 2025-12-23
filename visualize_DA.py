# read csv file with pandas 
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import sys
import matplotlib.image as mpimg
import open3d as o3d
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config.config import PARAMS
from datasets.augmentation import RemoveRandomPoints, RandomRotation_mod, RemoveRandomBlock, MoveRandomBlock2, RandomScaleXY, ElasticDistortion
 # Importamos PyVista solo si es necesario
import pyvista as pv
import cv2
import torch

def get_pointcloud_positions(folder_dir, folders):
    # given a folder, return the positions of the pointclouds
    # each pointcloud file name is the timestamp, the 'x', 'y' and 'a' orientation, for example file_pathname = t1152904768.768371_x-8.640943_y2.861793_a-0.209387.ply
    timestamps = []
    x_positions = []
    y_positions = []
    orientations = []
    files_names = []
    for folder in folders:
        room_dir = os.path.join(folder_dir, folder)
        # check if the folder is a directory    
        if not os.path.isdir(room_dir):
            continue
        for file in os.listdir(room_dir):
            if file.endswith(".ply"):        
                files_names.append(room_dir + '/' +file)
                # quitar la extension del archivo
                file = file[:-4]
                timestamp_index = file.index('t')
                x_index = file.index('_x')
                y_index = file.index('_y')
                a_index = file.index('_a')
                timestamp = file[timestamp_index+1:x_index]         


                x = file[x_index+2:y_index]
                y = file[y_index+2:a_index]
                a = file[a_index+2:]
                # x, y, a are strings, parse them to float
                x = float(x)
                y = float(y)
                a = float(a)
                timestamp = float(timestamp)

                timestamps.append(timestamp)
                x_positions.append(x)
                y_positions.append(y)
                orientations.append(a)
    df_locations = pd.DataFrame({ 'file': files_names, 'timestamp': timestamps, 'x': x_positions, 'y': y_positions, 'orientation': orientations})
    return df_locations


def get_axes_limits(coordX, coordY, xmax, xmin, ymax, ymin):
    if coordX < xmin:
        xmin = coordX
    if coordX > xmax:
        xmax = coordX
    if coordY < ymin:
        ymin = coordY
    if coordY > ymax:
        ymax = coordY
    return xmax, xmin, ymax, ymin



def display_coord_map(df, df_database):
    # df header 'query_image', 'query_x', 'query_y', 'retrieved_database_image', 'retrieved_database_x', 'retrieved_database_y', 'real_database_image', 'real_database_x', 'real_database_y', 'recall@1', 'recall@1%'
    # df_database header 'file', 'timestamp', 'x', 'y', 'orientation'
    # plt tkagg
    plt.switch_backend('tkagg')    
    

    
    xmin, xmax, ymin, ymax = 1000, -1000, 1000, -1000
    plt.figure(figsize=(9, 6), dpi=120, edgecolor='black')

    firstk1, firstErrork, firstErrorRoom = True, True, True
    # get the coordinates of the visual model
    mapVM = df_database[['x', 'y']].to_numpy()
    plt.scatter(mapVM[:, 0], mapVM[:, 1], color='blue', label="Visual Model")
    xmax, xmin, ymax, ymin = get_axes_limits(mapVM[0][0], mapVM[0][1], xmax, xmin, ymax, ymin)

    # get the coordinates of the test images
    mapTest = df[['query_x', 'query_y', 'retrieved_database_x', 'retrieved_database_y', 'recall@1', 'recall@1%']].to_numpy()
    # get the coordinates of the real database images
    mapReal = df[['real_database_x', 'real_database_y']].to_numpy()
    



    for t in range(len(mapTest)):
        # si el recall@1 es 1, el color es verde
        if mapTest[t][4] == 1:
            if firstk1:
                plt.scatter(mapTest[t][0], mapTest[t][1], color='green', label='Recall@1 prediction')
                firstk1 = False
            else:
                plt.scatter(mapTest[t][0], mapTest[t][1], color='green')
                plt.plot([mapTest[t][0], mapTest[t][2]], [mapTest[t][1], mapTest[t][3]], color='green')
            xmax, xmin, ymax, ymin = get_axes_limits(mapTest[t][2], mapTest[t][3], xmax, xmin, ymax, ymin)
        # si el recall@1 es 0 y el recall@1% es 1, el color es amarillo
        elif mapTest[t][4] == 0 and mapTest[t][5] == 1:
            if firstErrork:
                plt.scatter(mapTest[t][0], mapTest[t][1], color='orange', label='Recall@1% prediction')
                firstErrork = False
            else:
                plt.scatter(mapTest[t][0], mapTest[t][1], color='orange')
                plt.plot([mapTest[t][0], mapTest[t][2]], [mapTest[t][1], mapTest[t][3]], color='orange')
            xmax, xmin, ymax, ymin = get_axes_limits(mapTest[t][2], mapTest[t][3], xmax, xmin, ymax, ymin)
        # si el recall@1 es 0 y el recall@1% es 0, el color es rojo
        elif mapTest[t][4] == 0 and mapTest[t][5] == 0:
            if firstErrorRoom:
                plt.scatter(mapTest[t][0], mapTest[t][1], color='red', label='Predictions not among Recall@1 and Recall@1%')
                firstErrorRoom = False
            else:
                plt.scatter(mapTest[t][0], mapTest[t][1], color='red')
                plt.plot([mapTest[t][0], mapTest[t][2]], [mapTest[t][1], mapTest[t][3]], color='red')
            xmax, xmin, ymax, ymin = get_axes_limits(mapTest[t][2], mapTest[t][3], xmax, xmin, ymax, ymin)

    plt.axis([xmin-0.5, xmax+0.5, ymin-0.25, ymax+0.25])
    plt.ylabel('y (m)', fontsize=18)
    plt.xlabel('x (m)', fontsize=18)
    plt.title('Pseudo-LiDAR PR', fontsize=24)
    plt.legend(fontsize=14)
    plt.grid()
    # save the figure in the same folder as the csv file
    plt.show()
    print('Figure saved in: {}'.format(os.path.join(os.path.dirname(df['query_image'][0]), 'map.png')))

def get_pointcloud_and_apply_DA_image(pcd_file_path, dst_file_path, DA_to_apply):
    # Configuramos PyVista para renderizado sin pantalla
    pv.OFF_SCREEN = True
    pv.start_xvfb(wait=0.1)  # Inicia un servidor X virtual
    
    # Cargamos la nube de puntos con Open3D
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    points = np.asarray(pcd.points)
    points = torch.from_numpy(points).float()

    
    
    # Aplicar el efecto seleccionado
    if DA_to_apply == 'remove_random_points':
        transform = RemoveRandomPoints(r=(0.0, 0.4), p=1.0)
        points = transform(points)
        
    elif DA_to_apply == 'random_rotation':
        transform = RandomRotation_mod(p=1.0, axis=np.array([0, 0, 1]), max_theta=180)
        points = transform(points)
        
    elif DA_to_apply == 'remove_random_block':
        transform = RemoveRandomBlock(p=1.0, scale=(0.02, 0.5), ratio=(0.3, 3.3))
        points = transform(points)
        
    elif DA_to_apply == 'MoveRandomBlock':
        transform = MoveRandomBlock2(p=1.0, scale=(0.02, 0.5), ratio=(0.3, 3.3), max_move=0.7)
        points = transform(points)
        
    elif DA_to_apply == 'RandomScaleXY':
        transform = RandomScaleXY(min=1.15, max=1.6, p=1.0)
        points = transform(points)
        
    elif DA_to_apply == 'ElasticDistortion':
        transform = ElasticDistortion(p=1.0)
        points = transform(points)

    # Convertir puntos transformados de vuelta a Open3D
    points = points.numpy()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Rotar la nube de puntos 90 grados alrededor del eje Z (sentido horario)
    R = pcd.get_rotation_matrix_from_xyz([0, 0, -np.pi/2])
    pcd.rotate(R, center=pcd.get_center())
    
    # Convertir a formato PyVista
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    # Crear escena PyVista
    plotter = pv.Plotter(off_screen=True)
    point_cloud = pv.PolyData(points)
    if colors is not None:
        point_cloud['colors'] = colors
        plotter.add_points(point_cloud, render_points_as_spheres=False, point_size=5, rgb=True)
    else:
        plotter.add_points(point_cloud, render_points_as_spheres=True, point_size=3)
    
    zoom = 1.5
    plotter.camera.zoom(zoom)
    
    # Crear directorio y guardar imagen
    parent_dir = os.path.dirname(dst_file_path)
    os.makedirs(parent_dir, exist_ok=True)
    plotter.screenshot(dst_file_path, window_size=(1280, 820))
    
    print(f"Imagen guardada en: {dst_file_path}")
    pcd_image = mpimg.imread(dst_file_path)
    
    # Recortar imagen
    height, width, _ = pcd_image.shape
    crop_top = int(height * 0.1)
    crop_bottom = int(height * 0.05)
    crop_sides = int(width * 0.05)
    pcd_image = pcd_image[crop_top:(height-crop_bottom), crop_sides:(width-crop_sides), :]
    
    plotter.close()
    return pcd_image

def images2gif(folder, output, fps=2):
    """
    Convierte una secuencia de imágenes a un archivo GIF.
    
    Args:
        folder: Carpeta con las imágenes
        output: Ruta de salida del GIF
        fps: Frames por segundo del GIF
    """
    # Usar explícitamente imageio.v2 para evitar la advertencia de deprecación
    import imageio.v2 as imageio
    
    # Convertir fps a duración en segundos
    duration = 1.0 / fps

    # get all files in folder and subfolders
    files = []
    all_filenames = []
    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            # Solo incluir archivos de imagen
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                files.append(os.path.join(root, filename))
                all_filenames.append(filename)

    # Ordenar archivos por nombre
    files = [x for _, x in sorted(zip(all_filenames, files))]
    
    # Leer todas las imágenes
    images = []
    for file in files:
        images.append(imageio.imread(file))
    
    # Guardar como GIF
    imageio.mimsave(output, images, duration=duration, loop=0)  # 0 = bucle infinito
    print(f"GIF creado en: {output} a {fps} FPS")

def frames2video(folder, output, fps=30):
    """
    Convierte los frames en una carpeta a un video MP4.
    
    Args:
        folder: Carpeta con los frames
        output: Ruta de salida del video
        fps: Frames por segundo del video
    """
    # get all files in folder and subfolders
    files = []
    all_filenames = []
    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            # Sino terminan en .png o .jpg o .jpeg, no los añadimos
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                files.append(os.path.join(root, filename))
                all_filenames.append(filename)

    # sort files based on the number in the filename
    files = [x for _, x in sorted(zip(all_filenames, files))]

    # get first image to get size
    img = cv2.imread(files[0])
    height, width, _ = img.shape

    # create video writer with MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V codec for MP4
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    # write images to video
    for file in files:
        img = cv2.imread(file)
        out.write(img)

    out.release()


def video2gif_ffmpeg(video_path, gif_path, fps=6):
    """
    Convierte un video MP4 a GIF usando FFmpeg.
    """
    import subprocess
    
    # Comando FFmpeg para convertir video a GIF
    command = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', (f'fps={fps},scale=640:-1:flags=lanczos,'
                'split[s0][s1];[s0]palettegen=max_colors=256:reserve_transparent=0[p];'
                '[s1][p]paletteuse=dither=none'),
        '-loop', '0',
        gif_path
    ]
    
    # Ejecutar comando
    try:
        subprocess.run(command, check=True)
        print(f"GIF creado con FFmpeg en: {gif_path} a {fps} FPS")
    except subprocess.CalledProcessError as e:
        print(f"Error al crear el GIF con FFmpeg: {e}")
    except FileNotFoundError:
        print("FFmpeg no está instalado. Instálalo o asegúrate de que está en el PATH.")
        print("Alternativa: pip install moviepy")


def plot_pcds(df_database, dataset_path, output_path):
    """
    Plots the database PCD from all the distilled models y crea GIF directamente
    """
    depth_estimator = 'PCD_DISTILL_ANY_DEPTH_LARGE'
    i = 0
    row = df_database.iloc[i]
    images = []

    # check if the output figure already exists
    DA_options = ['remove_random_points', 'random_rotation', 'remove_random_block', 'MoveRandomBlock', 'RandomScaleXY', 'ElasticDistortion']
    for DA_to_apply in DA_options:
        output_file_path = os.path.join(output_path, f'{i}_{depth_estimator}_{DA_to_apply}.png')
        print(f"Processing index: {i} with model: {depth_estimator}")
        # Load PCD files
        pcd_path = row['file'].replace('PCD_DISTILL_ANY_DEPTH_LARGE', depth_estimator)
        
        image = get_pointcloud_and_apply_DA_image(pcd_path, output_file_path, DA_to_apply)

            


if __name__ == "__main__":

    dataset_path_base  = '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_DISTILL_ANY_DEPTH_LARGE/'
    output_path_base = '/media/arvc/DATOS/Juanjo/Datasets/COLD/VISUAL_RESULTS_DA/'
    environments = ['FRIBURGO_A']
    for environment in environments:
        # Directorio fuente y destino
        dataset_path = dataset_path_base + environment + '/'
        output_path = output_path_base + environment + '/'
        # create the output path if it does not exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
   

        
        database_folder = 'Train' 
        base_path = dataset_path
        database_dir = os.path.join(base_path, database_folder)
        all_room_folders = sorted(os.listdir(os.path.join(base_path, database_folder)))

        df_database = get_pointcloud_positions(database_dir, all_room_folders)
    

  
        plot_pcds(df_database, dataset_path, output_path)





