import os
import cv2
import numpy as np


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


if __name__ == '__main__':
    dataset_path_base  = '/media/arvc/DATOS/Juanjo/Datasets/COLD/PCD_DISTILL_ANY_DEPTH_LARGE/'
    input_path_base = '/media/arvc/DATOS/Marcos/DATASETS/COLD/'
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
            frames2video(cloudy_input_path, cloudy_output_path + f'cloudy_pano_{environment}.mp4', fps=60)
            
        if night_query_path is not None:
            if not os.path.exists(night_output_path):
                os.makedirs(night_output_path)
            frames2video(night_input_path, night_output_path + f'night_pano_{environment}.mp4', fps=60)

        if sunny_query_path is not None:
            if not os.path.exists(sunny_output_path):
                os.makedirs(sunny_output_path)
            frames2video(sunny_input_path, sunny_output_path + f'sunny_pano_{environment}.mp4', fps=60)
        print('Finished converting frames to video for environment:', environment)
    print('All videos converted successfully!')
        
        
        

