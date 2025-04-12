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


# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config.config import PARAMS

# def process_pcd(pcd_file_path, dst_dir, dst_file_path, vis):
#     """
#     Procesa un archivo PCD y guarda la imagen en el directorio de destino.
#     """
#     # Cargar el archivo PCD
#     pcd = o3d.io.read_point_cloud(pcd_file_path)

#     # o3d.visualization.draw_geometries([pcd],
#     #                               zoom=0.55999999999999983,
#     #                               front=[-0.37967999380788348, 0.47925146570274385, 0.79130344048526513],
#     #                               lookat=[0.0856678808010439, 0.12434512282477835, -0.06113851773026123],
#     #                               up=[0.49515957698681812, -0.61724131521917425, 0.61141651278383014])
#     vis.add_geometry(pcd)
#     # Configurar los parámetros de la vista
#     view_control = vis.get_view_control()
#     view_control.set_zoom(0.55999999999999983)
#     view_control.set_front([-0.37967999380788348, 0.47925146570274385, 0.79130344048526513])
#     view_control.set_lookat([0.0856678808010439, 0.12434512282477835, -0.06113851773026123])
#     view_control.set_up([0.49515957698681812, -0.61724131521917425, 0.61141651278383014])
#     # Renderizar la nube de puntos
#     vis.poll_events()
#     vis.update_renderer()

#     # Crear el directorio de destino si no existe
#     os.makedirs(dst_dir, exist_ok=True)

#     # Guardar la captura de pantalla
#     output_image_path = dst_file_path.replace('.ply', '.jpeg')
#     vis.capture_screen_image(output_image_path)
#     # img = vis.capture_screen_float_buffer(True)
#     vis.remove_geometry(pcd)
#     # img = np.asarray(img)
#     # img = np.uint8(img * 255)

#     # Convertir la imagen de RGB a BGR para OpenCV
#     # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     # Guardar la imagen
#     # cv2.imwrite(output_image_path, img)
#     print(f"Imagen guardada en: {output_image_path}")


def process_pcd(pcd_file_path, dst_dir, dst_file_path, vis=None):
    """
    Procesa un archivo PCD y guarda la imagen en el directorio de destino 
    utilizando PyVista para renderizado headless.
    """
    # Importamos PyVista solo si es necesario
    import pyvista as pv
    from pyvistaqt import BackgroundPlotter
    
    # Configuramos PyVista para renderizado sin pantalla
    pv.OFF_SCREEN = True
    
    # Cargamos la nube de puntos con Open3D
    pcd = o3d.io.read_point_cloud(pcd_file_path)
     # Rotar la nube de puntos 90 grados alrededor del eje Z (sentido horario)
    R = pcd.get_rotation_matrix_from_xyz([0, 0, -np.pi/2])  # Rotación de -90 grados en radianes alrededor de Z
    pcd.rotate(R, center=pcd.get_center())
 
    
    # Convertimos la nube de Open3D a un formato que PyVista pueda usar
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    # Creamos la escena PyVista
    plotter = pv.Plotter(off_screen=True)
    
    # Añadimos los puntos
    point_cloud = pv.PolyData(points)
    if colors is not None:
        point_cloud['colors'] = colors
        plotter.add_points(point_cloud, render_points_as_spheres=False, point_size=5, rgb=True)
    else:
        plotter.add_points(point_cloud, render_points_as_spheres=True, point_size=3)
    
#    # Extraer los parámetros de Open3D
#     front = [-0.37967999380788348, 0.47925146570274385, 0.79130344048526513]
#     lookat = [0.0856678808010439, 0.12434512282477835, -0.06113851773026123]
#     up = [0.49515957698681812, -0.61724131521917425, 0.61141651278383014]
    zoom = 1.5
#     # Calcular posición de cámara usando front y lookat
#     # En Open3D, front es la dirección de vista
#     # En PyVista, necesitamos la posición de la cámara
#     distance = 0.25  # Distancia desde lookat en la dirección front
#     position = [
#         lookat[0] - front[0] * distance,
#         lookat[1] - front[1] * distance,
#         lookat[2] - front[2] * distance
#     ]
    
#     # Configurar la cámara en PyVista
#     plotter.camera_position = [position, lookat, up]
    
#     # Ajustar el zoom
    plotter.camera.zoom(zoom)
    # Creamos el directorio de destino
    os.makedirs(dst_dir, exist_ok=True)
    
    # Guardamos la imagen
    output_image_path = dst_file_path.replace('.ply', '.jpeg')
    plotter.screenshot(output_image_path, window_size=(1280, 820))
    
    print(f"Imagen guardada en: {output_image_path}")

def copy_structure_and_process_images(src_dir, dst_dir, exclude_dir_names, depth='depth'):
   
    # Crear el visualizador
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(visible=False)
    vis = None

    """
    Copia la estructura de carpetas de src_dir a dst_dir y procesa las imágenes,
    excluyendo las carpetas con el nombre dado.

    Primero comprueba si ya existen las carpetas de destino, si no, las crea.
    """
    for root, dirs, files in os.walk(src_dir):
        # Excluir la carpetas especificadas
        #dirs[:] = [d for d in dirs if d != exclude_dir_name]
        for exclude_dir_name in exclude_dir_names:
            if exclude_dir_name in dirs:
                dirs.remove(exclude_dir_name)

        # Crea las carpetas en el directorio de destino
        for dir in dirs:
            # Si la carpeta no existe, la crea    
            os.makedirs(os.path.join(dst_dir, os.path.relpath(os.path.join(root, dir), src_dir)), exist_ok=True)

        for file in files:
            src_file_path = os.path.join(root, file)
            dst_file_path = os.path.join(dst_dir, os.path.relpath(src_file_path, src_dir))
            # Procesa las imágenes y guarda en el directorio destino
            if file.lower().endswith(('.ply')):
                process_pcd(src_file_path, dst_dir, dst_file_path, vis)
    
    
    # Cierra el visualizador
    # vis.destroy_window()
          
     

if __name__ == "__main__":
    PARAMS.cuda_device = 'cuda:0'
    environments = ['FRIBURGO_A', 'FRIBURGO_B', 'SAARBRUCKEN_A', 'SAARBRUCKEN_B']
    #environments = ['FRIBURGO_A']
    for environment in environments:
        # Directorio fuente y destino
        src_directory  = '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_DISTILL_ANY_DEPTH_LARGE/' + environment + '/'
        dst_directory = '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_DISTILL_ANY_DEPTH_LARGE_IMAGES/' + environment + '/'
        exclude_directory_names = ['RepImages', 'RepresentativeImages', 'fr_seq2_cloudy3', 'Train2', 'fr_seq2_cloudy1', 'Train_extended', 'Validation']

        copy_structure_and_process_images(src_directory, dst_directory, exclude_directory_names)

