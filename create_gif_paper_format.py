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

    # check if the output image already exists
    if os.path.exists(dst_file_path):
        print(f"Image already exists for {pcd_file_path}, skipping...")
        pcd_image = mpimg.imread(dst_file_path)
        height, width, _ = pcd_image.shape
        crop_top = int(height * 0.1)     # Recortar 18% desde arriba
        crop_bottom = int(height * 0.05)   # Recortar 10% desde abajo
        crop_sides = int(width * 0.05)    # Recortar 5% de los lados
        # Aplicar recorte
        pcd_image = pcd_image[crop_top:(height-crop_bottom), crop_sides:(width-crop_sides), :]
        # # show the image
        # plt.imshow(pcd_image)
        # plt.axis('off')
        # plt.show()
        return pcd_image

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


def plot_pcds_and_positions(df, df_database, dataset_path, output_path):
    """
    Plots the query PCD, retrieved database PCD, real database PCD, and their positions on the map.
    """
    k = 10
    if 'SAARBRUCKEN_B' in dataset_path:
        k = 5
    i = 0
    for index, row in df.iterrows():
        if i % k == 0:
            # check if the output figure already exists
            output_file_path = os.path.join(output_path, f'{i}.png')
            # if os.path.exists(output_file_path):
            #     print(f"Figure already exists for index: {index}, skipping...")
            #     continue
            # Print the index being processed
            print(f"Processing index: {i}")
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
            # Crear un mapa como una imagen separada
            fig_map = plt.figure(figsize=(6, 4))
            ax_map = fig_map.add_subplot(111)
            ax_map.scatter(df_database['x'], df_database['y'], color='blue', label="Database", s=4)
            ax_map.scatter(row['query_x'], row['query_y'], color='red', label="Query", marker='x', s=225, linewidths=4)
            ax_map.scatter(row['retrieved_database_x'], row['retrieved_database_y'], color='orange',
                marker='o', s=150, label="Retrieved Database")
            ax_map.scatter(row['real_database_x'], row['real_database_y'], color='green',
                marker='o', s=150, facecolors='none', edgecolors='green', linewidths=4, label="Nearest Database")
            ax_map.set_xlabel("x (m)", fontsize=16)
            ax_map.set_ylabel("y (m)", fontsize=16)
            
            # Ajustar el mapa según el dataset
            if 'FRIBURGO_A' in dataset_path:
                ax_map.legend(fontsize=20)
            elif 'FRIBURGO_B' in dataset_path:
                ax_map.legend(fontsize=20)
                ax_map.legend().get_frame().set_linewidth(1)
                ax_map.set_xlim(df_database['x'].min() - 4, df_database['x'].max() + 0.5)
                ax_map.set_ylim(df_database['y'].min() - 0.5, df_database['y'].max() + 0.5)
            elif 'SAARBRUCKEN_A' in dataset_path:
                ax_map.legend(fontsize=20)
                ax_map.legend().get_frame().set_linewidth(1)
                ax_map.set_xlim(df_database['x'].min() - 5, df_database['x'].max() + 0.5)
                ax_map.set_ylim(df_database['y'].min() - 0.5, df_database['y'].max() + 1.0)
            elif 'SAARBRUCKEN_B' in dataset_path:
                ax_map.legend(fontsize=20)
                ax_map.legend().get_frame().set_linewidth(1)
                ax_map.set_xlim(df_database['x'].min() - 0.5, df_database['x'].max() + 0.5)
                ax_map.set_ylim(df_database['y'].min() - 0.5, df_database['y'].max() + 2.0)
            plt.legend(fontsize=13)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            # add grid
            ax_map.grid()

            # Convertir el mapa directamente a un array NumPy sin guardarlo como archivo
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            canvas = FigureCanvas(fig_map)
            canvas.draw()
            map_image = np.array(canvas.renderer.buffer_rgba())

            # Convertir la imagen del mapa de RGBA a RGB
            if map_image.shape[2] == 4:  # Si tiene 4 canales (RGBA)
                # Crear un fondo blanco
                background = np.ones((map_image.shape[0], map_image.shape[1], 3), dtype=np.uint8) * 255
                
                # Extraer el canal alfa
                alpha = map_image[:, :, 3:4] / 255.0
                
                # Combinar el fondo con la imagen según el canal alfa
                rgb = map_image[:, :, :3]
                map_image = (rgb * alpha + background * (1 - alpha)).astype(np.uint8)

            plt.close(fig_map)

            # Asegúrate de que todas las imágenes tengan las mismas dimensiones
            # En lugar de usar padding (que puede dejar espacios blancos), redimensionalas
            from skimage.transform import resize

            # Determina el tamaño objetivo (altura máxima, ancho proporcional)
            target_height = max(query_image.shape[0], retrieved_image.shape[0], real_image.shape[0], map_image.shape[0])

            def resize_image_keep_aspect(img, target_height):
                aspect = img.shape[1] / img.shape[0]
                target_width = int(target_height * aspect)
                return resize(img, (target_height, target_width), preserve_range=True).astype(np.uint8)

            # Redimensionar imágenes
            query_resized = resize_image_keep_aspect(query_image, target_height)
            retrieved_resized = resize_image_keep_aspect(retrieved_image, target_height)
            real_resized = resize_image_keep_aspect(real_image, target_height)
            map_resized = resize_image_keep_aspect(map_image, target_height)

            # Verificar que todas las imágenes tengan el mismo número de canales
            print(f"Canales - query: {query_resized.shape[2]}, retrieved: {retrieved_resized.shape[2]}, real: {real_resized.shape[2]}, map: {map_resized.shape[2]}")

            # Concatenar las imágenes (sin espacio entre ellas)
            final_image = np.hstack((query_resized, retrieved_resized, real_resized, map_resized))

            # Crear figura para mostrar la imagen concatenada sin bordes
            fig, ax = plt.subplots(figsize=(12, 6), frameon=False)
            ax.imshow(final_image)
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Elimina los márgenes

            # Guardar la figura sin bordes
            plt.savefig(os.path.join(output_path, f'{i}.jpeg'), dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
        i += 1



if __name__ == "__main__":

    dataset_path_base  = '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_DISTILL_ANY_DEPTH_LARGE/'
    output_path_base = '/media/arvc/DATOS/Juanjo/Datasets/COLD/VISUAL_RESULTS_PAPER/'
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


        if cloudy_query_path is not None:
            if not os.path.exists(cloudy_output_path):
                os.makedirs(cloudy_output_path)
            print('Cloudy query path: {}'.format(cloudy_query_path))
            save_csv_path_cloudy = os.path.join(cloudy_query_path, 'results.csv')
            df_cloudy = pd.read_csv(save_csv_path_cloudy)
            timestamps = df_cloudy['query_image']
            # está entre '/t' y '_x'
            timestamps = [float(x.split('/t')[1].split('_x')[0]) for x in timestamps]
            # add the timestamps to the dataframe
            df_cloudy['timestamp'] = timestamps
            # sort the dataframe by the timestamp
            df_cloudy = df_cloudy.sort_values(by=['timestamp'])
            # get the pointcloud positions
            plot_pcds_and_positions(df_cloudy, df_database, dataset_path, cloudy_output_path)
        if night_query_path is not None:
            if not os.path.exists(night_output_path):
                os.makedirs(night_output_path)
            print('Night query path: {}'.format(night_query_path))
            save_csv_path_night = os.path.join(night_query_path, 'results.csv')
            df_night = pd.read_csv(save_csv_path_night)
            timestamps = df_night['query_image']
            # está entre '/t' y '_x'
            timestamps = [float(x.split('/t')[1].split('_x')[0]) for x in timestamps]
            # add the timestamps to the dataframe
            df_night['timestamp'] = timestamps
            # sort the dataframe by the timestamp
            df_night = df_night.sort_values(by=['timestamp'])
            # get the pointcloud positions
            plot_pcds_and_positions(df_night, df_database, dataset_path, night_output_path)
        if sunny_query_path is not None:
            if not os.path.exists(sunny_output_path):
                os.makedirs(sunny_output_path)
            print('Sunny query path: {}'.format(sunny_query_path))
            save_csv_path_sunny = os.path.join(sunny_query_path, 'results.csv')
            df_sunny = pd.read_csv(save_csv_path_sunny)
            timestamps = df_sunny['query_image']
            # está entre '/t' y '_x'
            timestamps = [float(x.split('/t')[1].split('_x')[0]) for x in timestamps]
            # add the timestamps to the dataframe
            df_sunny['timestamp'] = timestamps
            # sort the dataframe by the timestamp
            df_sunny = df_sunny.sort_values(by=['timestamp'])
            # get the pointcloud positions
            plot_pcds_and_positions(df_sunny, df_database, dataset_path, sunny_output_path)





