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
 # Importamos PyVista solo si es necesario
import pyvista as pv


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

def get_pointcloud_image(pcd_file_path, dst_file_path):

    
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
    pcd_image = mpimg.imread(dst_file_path)
    
    # Opcionalmente, recortar la imagen para centrar en la parte relevante
    # (ajustar estos valores según sea necesario)
    height, width, _ = pcd_image.shape
    crop_top = int(height * 0.1)     # Recortar 18% desde arriba
    crop_bottom = int(height * 0.05)   # Recortar 10% desde abajo
    crop_sides = int(width * 0.05)    # Recortar 5% de los lados
    
    # Aplicar recorte
    pcd_image = pcd_image[crop_top:(height-crop_bottom), crop_sides:(width-crop_sides), :]
    # Liberar recursos
    plotter.close()
    
    return pcd_image

def plot_images_and_positions(df, df_database, dataset_path):
    """
    Plots the query image, retrieved database image, real database image, and their positions on the map.
    """
    for index, row in df.iterrows():
        # Load images
        query_image_path = os.path.join(dataset_path, row['query_image'])
        retrieved_image_path = os.path.join(dataset_path, row['retrieved_database_image'])
        real_image_path = os.path.join(dataset_path, row['real_database_image'])
        # replace PCD, by DEPTH and .ply by .jpeg
        query_image_path = query_image_path.replace('PCD', 'DEPTH').replace('.ply', '.jpeg')
        retrieved_image_path = retrieved_image_path.replace('PCD', 'DEPTH').replace('.ply', '.jpeg')
        real_image_path = real_image_path.replace('PCD', 'DEPTH').replace('.ply', '.jpeg')
        query_image = mpimg.imread(query_image_path)
        retrieved_image = mpimg.imread(retrieved_image_path)
        real_image = mpimg.imread(real_image_path)

        # Create a figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Query vs Database Images (Index: {index})", fontsize=20)
        
        # Plot images
        axes[0, 0].imshow(query_image)
        axes[0, 0].set_title("Query Image", fontsize=20)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(retrieved_image)
        axes[0, 1].set_title("Retrieved Database Image", fontsize=20)
        axes[0, 1].axis('off')

        axes[1, 0].imshow(real_image)
        axes[1, 0].set_title("Real Database Image", fontsize=20)
        axes[1, 0].axis('off')

        # Plot positions on the map
        axes[1, 1].scatter(df_database['x'], df_database['y'], color='blue', label="Database Positions", s=1)
        # axes[1, 1].scatter(row['query_x'], row['query_y'], color='red', label="Query Position")
        # dibuja una cruz en la posicion de la query
        axes[1, 1].scatter(row['query_x'], row['query_y'], color='red', label="Query Position", marker='x', s=200)
        axes[1, 1].scatter(row['retrieved_database_x'], row['retrieved_database_y'], color='orange', label="Retrieved Position", s=200)
        axes[1, 1].scatter(row['real_database_x'], row['real_database_y'], color='green', label="Real Position", s=200)
        axes[1, 1].set_title("Positions on Map", fontsize=20)
        axes[1, 1].set_xlabel("x (m)", fontsize=20)
        axes[1, 1].set_ylabel("y (m)", fontsize=20)
        axes[1, 1].legend(fontsize=20)
        axes[1, 1].grid()

        # Adjust layout and show/save the figure
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def plot_pcds_and_positions(df, df_database, dataset_path, output_path):
    """
    Plots the query PCD, retrieved database PCD, real database PCD, and their positions on the map.
    """
    for index, row in df.iterrows():
        if index % 50 == 0:
            # check if the output figure already exists
            output_file_path = os.path.join(output_path, f'{index}.png')
            if os.path.exists(output_file_path):
                print(f"Figure already exists for index: {index}, skipping...")
                continue
            # Print the index being processed
            print(f"Processing index: {index}")
            # Load PCD files
            query_pcd_path = os.path.join(dataset_path, row['query_image'])
            retrieved_pcd_path = os.path.join(dataset_path, row['retrieved_database_image'])
            real_pcd_path = os.path.join(dataset_path, row['real_database_image'])

            query_dst_file_path = query_pcd_path.replace('PCD_DISTILL_ANY_DEPTH_LARGE', 'PCD_DISTILL_ANY_DEPTH_LARGE_IMAGES').replace('.ply', '.jpeg')
            retrieved_dst_file_path = retrieved_pcd_path.replace('PCD_DISTILL_ANY_DEPTH_LARGE', 'PCD_DISTILL_ANY_DEPTH_LARGE_IMAGES').replace('.ply', '.jpeg')
            real_dst_file_path = real_pcd_path.replace('PCD_DISTILL_ANY_DEPTH_LARGE', 'PCD_DISTILL_ANY_DEPTH_LARGE_IMAGES').replace('.ply', '.jpeg')
            # load pcds
            query_image = get_pointcloud_image(query_pcd_path, query_dst_file_path)
            retrieved_image = get_pointcloud_image(retrieved_pcd_path, retrieved_dst_file_path)
            real_image = get_pointcloud_image(real_pcd_path, real_dst_file_path)
            # Create a figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Query vs Database Point Clouds (Index: {index})", fontsize=24)

            # Plot images
            axes[0, 0].imshow(query_image)
            axes[0, 0].set_title("Query Point Cloud", fontsize=20)
            axes[0, 0].axis('off')

            axes[0, 1].imshow(retrieved_image)
            axes[0, 1].set_title("Retrieved Database Point Cloud", fontsize=20)
            axes[0, 1].axis('off')

            axes[1, 0].imshow(real_image)
            axes[1, 0].set_title("Nearest Database Point Cloud", fontsize=20)
            axes[1, 0].axis('off')

            # Plot positions on the map
            axes[1, 1].scatter(df_database['x'], df_database['y'], color='blue', label="Database Positions", s=4)
            # axes[1, 1].scatter(row['query_x'], row['query_y'], color='red', label="Query Position")
            # dibuja una cruz en la posicion de la query
            axes[1, 1].scatter(row['query_x'], row['query_y'], color='red', label="Query Position", marker='x', s=225, linewidths=4)
            axes[1, 1].scatter(row['retrieved_database_x'], row['retrieved_database_y'], color='orange', 
                  marker='o', s=150, label="Retrieved Database Position")
            axes[1, 1].scatter(row['real_database_x'], row['real_database_y'], color='green', 
                  marker='o', s=150, facecolors='none', edgecolors='green', linewidths=4, label="Nearest Database Position")
            axes[1, 1].set_title("Positions on Map", fontsize=20)
            axes[1, 1].set_xlabel("x (m)", fontsize=16)
            axes[1, 1].set_ylabel("y (m)", fontsize=16)
            axes[1, 1].legend(fontsize=12.5)
            axes[1, 1].grid()

            # Reducir espacio entre subplots y ajustar layout
            plt.subplots_adjust(wspace=0.05, hspace=0.1)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Guardar la figura con mayor calidad
            plt.savefig(os.path.join(output_path, f'{index}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            


if __name__ == "__main__":

    dataset_path_base  = '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_DISTILL_ANY_DEPTH_LARGE/'
    output_path_base = '/media/arvc/DATOS/Juanjo/Datasets/COLD/VISUAL_RESULTS/'
    environments = ['FRIBURGO_A', 'FRIBURGO_B', 'SAARBRUCKEN_A', 'SAARBRUCKEN_B']
    for environment in environments:
        # Directorio fuente y destino
        dataset_path = dataset_path_base + environment + '/'
        output_path = output_path_base + environment + '/'
        # create the output path if it does not exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
   
        cloudy_query_path = dataset_path + 'TestCloudy/'
        night_query_path = dataset_path + 'TestNight/'
        sunny_query_path = dataset_path + 'TestSunny/'
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
        
        
        database_folder = 'Train' 
        base_path = dataset_path
        database_dir = os.path.join(base_path, database_folder)
        all_room_folders = sorted(os.listdir(os.path.join(base_path, database_folder)))

        df_database = get_pointcloud_positions(database_dir, all_room_folders)
    
        # read csv file with pandas    
        # header 'query_image', 'query_x', 'query_y', 'retrieved_database_image', 'retrieved_database_x', 'retrieved_database_y', 'real_database_image', 'real_database_x', 'real_database_y', 'recall@1', 'recall@1%'
        # display_coord_map(df_cloudy, df_database)
        # display_coord_map(df_night, df_database)
        # display_coord_map(df_sunny, df_database)

        if cloudy_query_path is not None:
            if not os.path.exists(cloudy_output_path):
                os.makedirs(cloudy_output_path)
            print('Cloudy query path: {}'.format(cloudy_query_path))
            save_csv_path_cloudy = os.path.join(cloudy_query_path, 'results.csv')
            df_cloudy = pd.read_csv(save_csv_path_cloudy)
            plot_pcds_and_positions(df_cloudy, df_database, dataset_path, cloudy_output_path)
        if night_query_path is not None:
            if not os.path.exists(night_output_path):
                os.makedirs(night_output_path)
            print('Night query path: {}'.format(night_query_path))
            save_csv_path_night = os.path.join(night_query_path, 'results.csv')
            df_night = pd.read_csv(save_csv_path_night)
            plot_pcds_and_positions(df_night, df_database, dataset_path, night_output_path)
        if sunny_query_path is not None:
            if not os.path.exists(sunny_output_path):
                os.makedirs(sunny_output_path)
            print('Sunny query path: {}'.format(sunny_query_path))
            save_csv_path_sunny = os.path.join(sunny_query_path, 'results.csv')
            df_sunny = pd.read_csv(save_csv_path_sunny)
            plot_pcds_and_positions(df_sunny, df_database, dataset_path, sunny_output_path)
            
        
        
        

