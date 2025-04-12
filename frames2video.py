import os
import cv2
import numpy as np


def frames2video(folder, output, fps=30):
    # get all files in folder and subfolders
    files = []
    all_filenames = []
    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            files.append(os.path.join(root, filename))
            all_filenames.append(filename)

    # sort files
    #all_filenames.sort()
    # sort files based on the number in the filename
    files = [x for _, x in sorted(zip(all_filenames, files))]
    #files.sort()

    # get first image to get size
    img = cv2.imread(files[0])
    height, width, _ = img.shape

    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    # write images to video
    for file in files:
        img = cv2.imread(file)
        out.write(img)

    out.release()


if __name__ == '__main__':
    folder = '/media/arvc/Extreme SSD/Depth_Friburgo/TestCloudy'
    output = 'cloudy_depth.avi'
    frames2video(folder, output)