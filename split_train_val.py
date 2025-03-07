import os


src_directory  = '/media/arvc/DATOS/Marcos/DATASETS/COLD/FRIBURGO_A/fr_seq2_cloudy3/'
val_directory = '/media/arvc/DATOS/Marcos/DATASETS/COLD/FRIBURGO_A/Validation/'
dst_directory = '/media/arvc/DATOS/Marcos/DATASETS/COLD/FRIBURGO_A/Train_extended/'

# Create the destination directory
if not os.path.exists(dst_directory):
    os.makedirs(dst_directory)

val_images = []
all_room_folders = sorted(os.listdir(src_directory))
for folder in all_room_folders:
    room_dir = os.path.join(val_directory, folder)
    # check if the folder is a directory    
    if not os.path.isdir(room_dir):
        continue
    for file in os.listdir(room_dir):
        if file.endswith(".jpeg"):        
            val_images.append(file)

# copy the images to the destination directory from the source directory except the validation images
for folder in all_room_folders:
    room_dir = os.path.join(src_directory, folder)
    # check if the folder is a directory    
    if not os.path.isdir(room_dir):
        continue
    for file in os.listdir(room_dir):
        if file.endswith(".jpeg") and file not in val_images:        
            src = os.path.join(room_dir, file)
            dst = os.path.join(dst_directory, folder, file)
            # copy the source file to the destination directory
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.system('cp ' + src + ' ' + dst)
            print('cp ' + src + ' ' + dst)