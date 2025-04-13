import os
import cv2
import numpy as np
import open3d as o3d
import pyvista as pv

def get_pointcloud_image(pcd_file_path):
    dst_file_path = pcd_file_path.replace('PCD_DISTILL_ANY_DEPTH_LARGE', 'PCD_DISTILL_ANY_DEPTH_LARGE_IMAGES').replace('.ply', '.jpeg')
    # Verificamos si el archivo ya existe
    if os.path.exists(dst_file_path):
        print(f"Imagen ya existe: {dst_file_path}")
        return cv2.imread(dst_file_path)
    
    # Configuramos PyVista para renderizado sin pantalla
    pv.OFF_SCREEN = True
    pv.start_xvfb(wait=0.1)  # Inicia un servidor X virtual
    
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
    
    zoom = 1.5

    plotter.camera.zoom(zoom)
    
    # get the parent directory of the file
    parent_dir = os.path.dirname(dst_file_path)
    # Creamos el directorio de destino
    os.makedirs(parent_dir, exist_ok=True)
    
    # Guardamos la imagen
    plotter.screenshot(dst_file_path, window_size=(1280, 820))
    
    print(f"Imagen guardada en: {dst_file_path}")
    pcd_image = cv2.imread(dst_file_path)
    # Liberar recursos
    plotter.close()
    
    return pcd_image

def pcds2video(folder, output, fps=30):
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
            # check if ends in .ply
            if filename.endswith('.ply'):               
                files.append(os.path.join(root, filename))
                all_filenames.append(filename)

    # sort files based on the number in the filename
    files = [x for _, x in sorted(zip(all_filenames, files))]

    # get first image to get size
    img = get_pointcloud_image(files[0])
    height, width, _ = img.shape

    # create video writer with MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V codec for MP4
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    # write images to video
    for file in files:
        img = get_pointcloud_image(file)
        out.write(img)

    out.release()


if __name__ == '__main__':
    dataset_path_base  = '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_DISTILL_ANY_DEPTH_LARGE/'
    input_path_base = '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_DISTILL_ANY_DEPTH_LARGE/'
    output_path_base = '/media/arvc/DATOS/Juanjo/Datasets/COLD/VIDEOS_RESULTS/'
    environments = ['FRIBURGO_A', 'FRIBURGO_B', 'SAARBRUCKEN_A', 'SAARBRUCKEN_B']
    for environment in environments:
        # Directorio fuente y destino
        dataset_path = dataset_path_base + environment + '/'
        output_path = output_path_base + environment + '/'
        input_path = input_path_base + environment + '/'      
   
        cloudy_query_path = dataset_path + 'TestCloudy/'
        night_query_path = dataset_path + 'TestNight/'
        sunny_query_path = dataset_path + 'TestSunny/'
        cloudy_input_path = input_path + 'TestCloudy/'
        night_input_path = input_path + 'TestNight/'
        sunny_input_path = input_path + 'TestSunny/'
        cloudy_output_path = output_path + 'TestCloudy/'
        night_output_path = output_path + 'TestNight/'
        sunny_output_path = output_path + 'TestSunny/'

        if not os.path.exists(cloudy_query_path):
            # elude cloudy information if it does not exist
            cloudy_query_path = None
        if not os.path.exists(night_query_path):
            # elude night information if it does not exist
            night_query_path = None
        if not os.path.exists(sunny_query_path):
            # elude sunny information if it does not exist
            sunny_query_path = None        

        if cloudy_query_path is not None:
            if not os.path.exists(cloudy_output_path):
                os.makedirs(cloudy_output_path)
            pcds2video(cloudy_input_path, cloudy_output_path + f'cloudy_pcds_{environment}.mp4', fps=60)
            
        if night_query_path is not None:
            if not os.path.exists(night_output_path):
                os.makedirs(night_output_path)
            pcds2video(night_input_path, night_output_path + f'night_pcds_{environment}.mp4', fps=60)

        if sunny_query_path is not None:
            if not os.path.exists(sunny_output_path):
                os.makedirs(sunny_output_path)
            pcds2video(sunny_input_path, sunny_output_path + f'sunny_pcds_{environment}.mp4', fps=60)
        print('Finished converting frames to video for environment:', environment)
    print('All videos converted successfully!')
        
        
        

