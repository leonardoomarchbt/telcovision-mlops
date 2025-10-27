#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import yaml
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# --- Configuración de rutas y parámetros ---
def load_params(param_file="params.yaml"):
    try:
        with open(param_file, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Archivo de parámetros '{param_file}' no encontrado.")
        sys.exit(1)

# --- inicio ---
params = load_params()

raw_data_path = 'data/raw/telco_churn.csv'

# Ruta de salida para los datos procesados
output_path = 'data/processed/telco_churn_ok.csv'

# Crear la carpeta de salida si no existe
os.makedirs(os.path.dirname(output_path), exist_ok=True)

data = pd.read_csv(raw_data_path)

# Análisis Exploratorio de Datos
print(f"Filas y columnas iniciales: {data.shape}")
print(f"Celdas nulas: \n{data.isnull().sum()}")

data.head()

data.gender.unique()

genero_mapping = {
    'Female': 1, # 1 Femenino
    'Male': 2    # 2 Masculino
}

data['gender'] = data['gender'].map(genero_mapping)

data.hist(figsize = (20,20), color='green')

data.info()

correlacion = data.corr(numeric_only=True)  

correlacion

sns.heatmap(correlacion, annot = True)

# rellenamos nulos con un valor 0.
data.fillna(0, inplace=True) 

# guardo el dataset procesado
data.to_csv(output_path, index=False)

print(f"Datos procesados guardados en: {output_path})")