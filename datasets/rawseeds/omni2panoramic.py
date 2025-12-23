import numpy as np
import os
import cv2 as cv
import time
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import time

def cargar_parametros_calibracion(ruta_calibracion):
    """
    Carga los parámetros de calibración desde un archivo .mat de MATLAB.
    
    Args:
        ruta_calibracion: Ruta al archivo Calibration_04-Intrinsics_OMNI.mat
    
    Returns:
        salida: Lista con [centro_x, centro_y, radio, xi] extraída del archivo .mat
    """
    try:
                # Cargar el archivo .mat
        mat_data = sio.loadmat(ruta_calibracion)
        
        # Extraer parámetros según la estructura del archivo
        # cc: centro de la imagen (2, 1) -> [cc_x, cc_y]
        # gammac: parámetros de escala (2, 1) -> focal lengths
        # xi: parámetro de distorsión omnidireccional (1, 1)
        
        cc = mat_data['cc'].flatten()  # Centro de la imagen [cx, cy]
        centro_x = float(cc[0])
        centro_y = float(cc[1])
        
        # gammac contiene los factores de escala (focal lengths)
        gammac = mat_data['gammac'].flatten()
        radio = float(gammac[0])  # Usar el primer factor de escala como radio
        
        # xi es el parámetro de distorsión omnidireccional
        xi = float(mat_data['xi'][0, 0])
        
        salida = [centro_x, centro_y, radio, xi]
        
        print(f"Parámetros de calibración cargados exitosamente:")
        print(f"  - Centro X: {centro_x}")
        print(f"  - Centro Y: {centro_y}")
        print(f"  - Radio (focal length): {radio}")
        print(f"  - Xi (distorsión omnidireccional): {xi}")
        return salida



        # # Cargar el archivo .mat
        # mat_data = sio.loadmat(ruta_calibracion)
        
        # # Inspeccionar el contenido del archivo
        # print("Claves disponibles en el archivo .mat:")
        # for key in mat_data.keys():
        #     if not key.startswith('__'):
        #         print(f"  - {key}: {mat_data[key].shape}")
        
        # # Extraer parámetros (ajusta los nombres según tu archivo)
        # # Opciones comunes en archivos de calibración de cámaras omnidireccionales:
        
        # # Intenta diferentes nombres posibles
        # if 'Intrinsics' in mat_data:
        #     intrinsics = mat_data['Intrinsics']
        # elif 'intrinsics' in mat_data:
        #     intrinsics = mat_data['intrinsics']
        # elif 'calibration' in mat_data:
        #     intrinsics = mat_data['calibration']
        # else:
        #     # Si no encuentra los nombres estándar, mostrar todas las claves
        #     raise KeyError(f"No se encontraron parámetros de calibración. Claves disponibles: {[k for k in mat_data.keys() if not k.startswith('__')]}")
        
        # # Extraer valores individuales (ajusta según la estructura de tu archivo)
        # # Típicamente para cámaras omnidireccionales:
        # centro_x = float(intrinsics[0, 0]) if intrinsics.ndim > 1 else float(intrinsics[0])
        # centro_y = float(intrinsics[0, 1]) if intrinsics.ndim > 1 else float(intrinsics[1])
        # radio = float(intrinsics[0, 2]) if intrinsics.ndim > 1 else float(intrinsics[2])
        # xi = float(intrinsics[0, 3]) if intrinsics.ndim > 1 else float(intrinsics[3])
        
        # salida = [centro_x, centro_y, radio, xi]
        
        # print(f"Parámetros de calibración cargados exitosamente: {salida}")
        # return salida
    
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en {ruta_calibracion}")
        print("Usando valores por defecto: [320, 320, 350, 15.5]")
        return [320, 320, 350, 15.5]
    
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        print("Usando valores por defecto: [320, 320, 350, 15.5]")
        return [320, 320, 350, 15.5]
    

def create_path(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


def o2p(omni_img, r_int, r_ext, ilum):
    # r_int: internal radius (radius of the black point in the centre)
    # r_ext: external radius (edge of the omnidirectional image)

    dimensions = omni_img.shape
    [Cx, Cy] = [dimensions[0], dimensions[1]]

    # if ilum == "Cloudy":
    #     omni_img = omni_img[160:1080, 64:1440]
    #     [Cx, Cy] = [920, 1376]
    # elif ilum == "Sunny":
    #     omni_img = omni_img[44:1080, 76:1440]
    #     [Cx, Cy] = [1036, 1364]
    # else:
    #     omni_img = omni_img[128:1080, 68:1440]
    #     [Cx, Cy] = [952, 1372]

    # cv.imwrite(os.path.join(panoDir, "Prueba1.png"), omni_img)

    # Centre of the image
    Cx = Cx/2
    Cy = Cy/2

    r1 = r_ext
    r2 = r_int

    i = 0
    j = 0

    pi = np.pi
    w = int(np.floor((r1 + r2) * pi))
    l1 = int(len(range(r1, r2 - 1, -1)))
    l2 = int(len(range(w-1, -1, -1)))

    panoramic_matrix = np.zeros([l1, l2, 3])
    for r in range(r1, r2-1, -1):
        for var in range(w, -1, -1):
            coorx1 = int(np.floor(r * np.sin(var * 2 * pi / w) + Cx))
            valorx1 = (r * np.sin(var * 2 * pi / w) + Cx) - np.floor(r * np.sin(var * 2 * pi / w) + Cx)

            coory1 = int(np.floor(r * np.cos(var * 2 * pi / w) + Cy))
            valory1 = (r * np.cos(var * 2 * pi / w) + Cy) - np.floor(r * np.cos(var * 2 * pi / w) + Cy)

            A = ((2 - valorx1 - valory1) * omni_img[coorx1, coory1, 0:])
            B = ((1 - valorx1 + valory1) * omni_img[coorx1, coory1 + 1, 0:])
            C = ((1 + valorx1 - valory1) * omni_img[coorx1 + 1, coory1, 0:])
            D = ((valorx1 + valory1) * omni_img[coorx1 + 1, coory1 + 1, 0:])

            [red, green, blue] = (A + B + C + D) / 4
            # panoramic_matrix[i, j, 0:] = int((A + B + C + D)/ 4)
            panoramic_matrix[i, j:-1, 0] = int(red)
            panoramic_matrix[i, j:-1, 1] = int(green)
            panoramic_matrix[i, j:-1, 2] = int(blue)

            j += 1
        j = 0
        i += 1

    return panoramic_matrix.astype(np.uint8)


def pasar_panoram(imagen, salida):
    """
    Convierte una imagen omnidireccional a una panorámica.

    Args:
        imagen: Imagen de entrada (en formato NumPy array).

    Returns:
        panoramica: Imagen panorámica.
        t: Tiempo de ejecución.
    """
    start_time = time.time()
    # salida = [240, 331, 233, 15.5]
    # salida = [320, 320, 300, 15.5]

    perimetro = 2 * np.pi * ((salida[2] + salida[3]) / 2)
    perim = round(perimetro)

    A = imagen.astype(float)
    panoramica = []

    i = 0

    for r in range(int(salida[2]) - 5, int(salida[3]) + 5, -1):
        row = []
        for var in range(perim, -1, -1):
            coorx1 = int(np.floor(r * np.sin(var * 2 * np.pi / (perim + 1)) + salida[0]))
            valorx1 = (r * np.sin(var * 2 * np.pi / (perim + 1)) + salida[0]) - np.floor(
                r * np.sin(var * 2 * np.pi / (perim + 1)) + salida[0])

            coory1 = int(np.floor(r * np.cos(var * 2 * np.pi / (perim + 1)) + salida[1]))
            valory1 = (r * np.cos(var * 2 * np.pi / (perim + 1)) + salida[1]) - np.floor(
                r * np.cos(var * 2 * np.pi / (perim + 1)) + salida[1])

            if coorx1 + 1 < A.shape[0] and coory1 + 1 < A.shape[1]:
                interpolated_value = (
                                             ((2 - valorx1 - valory1) * A[coorx1, coory1]) +
                                             ((1 - valorx1 + valory1) * A[coorx1, coory1 + 1]) +
                                             ((1 + valorx1 - valory1) * A[coorx1 + 1, coory1]) +
                                             ((valorx1 + valory1) * A[coorx1 + 1, coory1 + 1])
                                     ) / 4
                row.append(interpolated_value)
            else:
                row.append(np.zeros(A.shape[2]))

        panoramica.append(row)
        i += 1

    panoramica = np.array(panoramica).astype(np.uint8)
    # recorta la imagen panoramica para para eliminar 90 pixeles inferiores
    panoramica = panoramica[0:panoramica.shape[0]-90, 0:panoramica.shape[1], :]
    panoramica = cv.resize(panoramica, (1024, 256), interpolation=cv.INTER_CUBIC)
    t = time.time() - start_time

    return panoramica


if __name__ == "__main__":
    base_dataset = '/media/arvc/DATOS1/Juanjo/Datasets/rawseeds'
    ruta_calibracion = '/media/arvc/DATOS1/Juanjo/Datasets/rawseeds/Calibration_04-Intrinsics_OMNI.mat'
    salida = cargar_parametros_calibracion(ruta_calibracion)
    salida[2] = salida[2] + 30
    salida[3] = salida[3] + 5
    sequences = os.listdir(base_dataset)
    # filter only directories
    sequences = [seq for seq in sequences if os.path.isdir(os.path.join(base_dataset, seq))]
    for seq in sequences:
        seq_path = os.path.join(base_dataset, seq)
        omniDir = os.path.join(seq_path, "OMNI")
        panoDir = create_path(os.path.join(seq_path, "PANO"))
        for img_file in os.listdir(omniDir):
            if img_file.endswith('.png') or img_file.endswith('.jpg'):
                img_path = os.path.join(omniDir, img_file)
                imgOmni = cv.imread(img_path)
                # show imgOmni
                # plt.imshow(cv.cvtColor(imgOmni, cv.COLOR_BGR2RGB))
                # plt.axis('off')
                # plt.show()
                imgPanoramica = pasar_panoram(imgOmni, salida)
                destinationDir = os.path.join(panoDir, img_file)
                # change extension from .png to .jpeg
                destinationDir = os.path.splitext(destinationDir)[0] + '.jpeg'
                # show imgPanoramica
                # plt.imshow(cv.cvtColor(imgPanoramica, cv.COLOR_BGR2RGB))
                # plt.axis('off')
                # plt.show()
                cv.imwrite(destinationDir, imgPanoramica)


    # buildings = os.listdir(omniDir)

    # for building in buildings:
    #     buildingDir = os.path.join(omniDir, building)
    #     buildPanoDir = create_path(os.path.join(panoDir, building))

    #     sequences = os.listdir(buildingDir)

    #     for seq in sequences:
    #         sequenceDir = os.path.join(buildingDir, seq)
    #         rooms = os.listdir(sequenceDir)

    #         for room in rooms:
    #             roomDir = os.path.join(sequenceDir, room)
    #             images = os.listdir(roomDir)
    #             roomPanoDir = create_path(os.path.join(buildPanoDir, room))

    #             for img in images:
    #                 imgOmni = cv.imread(os.path.join(roomDir, img))
    #                 imgPanoramica = pasar_panoram(imgOmni)

    #                 destinationDir = os.path.join(roomPanoDir, img)
    #                 cv.imwrite(destinationDir, imgPanoramica)




