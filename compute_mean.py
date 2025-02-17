import numpy as np
# Ruta al archivo .txt
file_path = '/home/arvc/Juanjo/develop/DepthMinkUNeXt/training/experiment_truncated_results_v4.txt'

# Leer el archivo línea por línea
lines = []
means = []
with open(file_path, 'r') as file:
    for line in file:
        # Dividir la línea por comas
        datos = line.strip().split(', ')
        if len(datos) < 8:
            continue
        
        # Convertir los valores de las tres últimas columnas a float y agregarlos a las listas
        means.append(np.mean([float(datos[-3]), float(datos[-2]), float(datos[-1])]))
        lines.append(line)

# Calcular la media de cada fila

# Crea un nuevo archivo con los valores originales y afdemás las medias de las tres últimas columnas
with open('/home/arvc/Juanjo/develop/DepthMinkUNeXt/training/experiment_truncated_results_v4_mean.txt', 'w') as file:
    
    for line, mean in zip(lines, means):
        file.write(f'{line.strip()}, {mean}\n')

print('Archivo creado con éxito')