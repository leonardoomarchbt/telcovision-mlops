import pickle
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import json
import yaml

# --- Configuración ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "LogisticRegression" 
MODEL_PATH = PROJECT_ROOT / "models" / f"{MODEL_NAME}_pipeline.pkl"

# Rutas de salida para la Etapa 7
EVAL_METRICS_PATH = PROJECT_ROOT / "models" / "evaluation_metrics.json"
ROC_PLOT_PATH = PROJECT_ROOT / "models" / "roc_curve.png"
REPORT_PATH = PROJECT_ROOT / "models" / "classification_report.txt"

# --- 1. Cargar Datos y Modelo ---
df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "telco_churn_ok.csv")
y = df["churn"].values
X = df.drop(columns=["churn"])

# Si deseas evaluar solo sobre el conjunto de prueba (más riguroso):
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# X_eval, y_eval = X_test, y_test
X_eval, y_eval = X, y # Evaluar sobre el dataset completo para una métrica robusta

try:
    with open(MODEL_PATH, 'rb') as f:
        model_pipeline = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Modelo no encontrado en {MODEL_PATH}. ¿Corrió dvc pull?")
    exit(1)


# --- 2. Evaluación y Generación de Artefactos ---
y_proba = model_pipeline.predict_proba(X_eval)[:, 1]
y_pred = model_pipeline.predict(X_eval)

# A. Calcular Métricas de Producción y Guardar JSON
roc_auc = auc(*roc_curve(y_eval, y_proba)[:2])
report = classification_report(y_eval, y_pred, output_dict=True)

eval_metrics = {
    "final_roc_auc": float(roc_auc),
    "final_f1_score": report['1']['f1-score'], # F1 de la clase CHURN (1)
    "model_used": MODEL_NAME,
    "params_used": model_pipeline.named_steps['clf'].get_params()
}

with open(EVAL_METRICS_PATH, "w") as f:
    json.dump(eval_metrics, f, indent=4)

# B. Generar Curva ROC (PNG)
fpr, tpr, _ = roc_curve(y_eval, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.legend(loc="lower right")
plt.title('Curva ROC Final de Producción')
plt.savefig(ROC_PLOT_PATH)
plt.close()

# C. Generar Reporte de Clasificación (TXT)
with open(REPORT_PATH, "w") as f:
    f.write(classification_report(y_eval, y_pred))

print(f"[OK] Evaluación finalizada. ROC AUC: {roc_auc:.4f}")