#!/bin/bash

# Ejecutar el script de prueba
python3 eval/pnv_evaluate_friburgo_b.py

# Verificar si el segundo script se ejecutó correctamente
if [ $? -ne 0 ]; then
  echo "Error al ejecutar eval/pnv_evaluate_friburgo_b.py"
  exit 1
fi

# Ejecutar el script de prueba
python3 eval/pnv_evaluate_saarbrucken_a.py

# Verificar si el segundo script se ejecutó correctamente
if [ $? -ne 0 ]; then
  echo "Error al ejecutar eval/pnv_evaluate_saarbrucken_a.py"
  exit 1
fi

# Ejecutar el script de prueba
python3 eval/pnv_evaluate_saarbrucken_b.py

# Verificar si el segundo script se ejecutó correctamente
if [ $? -ne 0 ]; then
  echo "Error al ejecutar eval/pnv_evaluate_saarbrucken_b.py"
  exit 1
fi