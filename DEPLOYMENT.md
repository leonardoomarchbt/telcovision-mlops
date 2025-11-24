1. Prerrequisitos y Artefactos

Esta sección le dice al ingeniero cómo obtener el modelo y qué necesita instalar.


# 🚀 GUÍA DE DESPLIEGUE - MODELO TELCO CHURN FINAL

## 1. Configuración del Ambiente

Para ejecutar el servicio, se requiere el modelo final y las librerías de Python.

### A. Obtención del Modelo (DVC Pull)

El modelo es un archivo binario rastreado por DVC. Debe ser descargado antes de iniciar el servicio.

1.  Clonar el repositorio y moverse a la rama master.
2.  Autenticar DVC (usando el PAT).
3.  Descargar el modelo final:
    ```bash
    dvc pull models/LogisticRegression_pipeline.pkl
    ```

### B. Dependencias del Servicio

El entorno de servicio necesita Python, la librería para cargar modelos, y el servidor web.

```bash
# requirements_service.txt (Archivo sugerido para el despliegue)
scikit-learn==1.3.2
pandas==2.1.2
pyyaml
uvicorn
fastapi

2. Código del Servicio de Inferencia (API)

Este código muestra cómo cargar el pipeline de sklearn y exponer una ruta /predict que toma datos JSON de un nuevo cliente y devuelve la predicción.

Archivo sugerido: api/main.py

# api/main.py
import pickle
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

# --- 1. Definición del Contrato de Datos (Input) ---
# Usamos Pydantic para validar los datos de entrada de la API.
class CustomerData(BaseModel):
    age: int
    gender: str
    region: str
    contract_type: str
    tenure_months: int
    monthly_charges: float
    total_charges: float
    internet_service: str
    phone_service: str
    multiple_lines: str
    payment_method: str
    
    # Ejemplo de datos para la documentación de la API (útil para Swagger/OpenAPI)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 35, "gender": "Male", "region": "North", 
                    "contract_type": "Month-to-Month", "tenure_months": 12, 
                    "monthly_charges": 65.5, "total_charges": 786.0, 
                    "internet_service": "DSL", "phone_service": "Yes", 
                    "multiple_lines": "No", "payment_method": "Electronic check"
                }
            ]
        }
    }


# --- 2. Carga del Modelo ---
MODEL_PATH = Path("models") / "LogisticRegression_pipeline.pkl"
app = FastAPI(title="Servicio de Predicción de Churn de Telco")

# Cargar el pipeline fuera de la función de endpoint (solo una vez al iniciar la API)
try:
    with open(MODEL_PATH, 'rb') as f:
        MODEL_PIPELINE = pickle.load(f)
    print("Modelo LogisticRegression cargado con éxito.")
except FileNotFoundError:
    print(f"ERROR: Modelo no encontrado en {MODEL_PATH}. Ejecute 'dvc pull'.")
    MODEL_PIPELINE = None


# --- 3. Endpoint de Predicción ---
@app.post("/predict")
def predict_churn(data: CustomerData):
    if MODEL_PIPELINE is None:
        return {"error": "El modelo no está cargado. Verifique la ruta del artefacto."}
        
    # Convertir el objeto Pydantic a un DataFrame de Pandas
    input_df = pd.DataFrame([data.model_dump()])
    
    # El pipeline gestiona automáticamente el preprocesamiento (escalado, OHE)
    probability = MODEL_PIPELINE.predict_proba(input_df)[0][1]
    prediction = int(MODEL_PIPELINE.predict(input_df)[0])
    
    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "probability_churn": float(f"{probability:.4f}"),
        "model_version_source": "DagsHub/DVC"
    }

# --- 4. Instrucciones para Ejecutar el Servicio ---
# Comando: uvicorn main:app --reload --port 8000