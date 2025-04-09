#!/bin/bash

# Ejecutar el script de prueba
python3 training/train_DA_ablation.py

# Verificar si el segundo script se ejecutó correctamente
if [ $? -ne 0 ]; then
  echo "Error al ejecutar training/train_DA_ablation.py"
  exit 1
fi

# Ejecutar el script de prueba
python3 training/train_depth_estimators.py

# Verificar si el segundo script se ejecutó correctamente
if [ $? -ne 0 ]; then
  echo "Error al ejecutar training/train_depth_estimators.py"
  exit 1
fi

