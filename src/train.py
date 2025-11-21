\
import os
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import pickle
import yaml


# ---------- CONFIGURACIÓN DE MLFLOW ----------
# Local por defecto: file:///<proyecto>/mlruns
PROJECT_ROOT = Path(__file__).resolve().parents[1]
#local_tracking_uri = f"file:///{PROJECT_ROOT.as_posix()}/mlruns"
#mlflow.set_tracking_uri(local_tracking_uri)

# --- OPCIONAL: DagsHub MLflow Tracking ---
# Descomenta estas líneas si deseas registrar en DagsHub (y comenta la URI local de arriba).
# Requiere setear variables de entorno:
# # set MLFLOW_TRACKING_USERNAME='leonardo.quiroga'
# # set MLFLOW_TRACKING_PASSWORD="0cef0d7198d17681bdb0a6dd6da2f3a073e88847"
   # PowerShell:     $env:MLFLOW_TRACKING_USERNAME="<USER>"; $env:MLFLOW_TRACKING_PASSWORD="<TOKEN>"
   # Bash:           export MLFLOW_TRACKING_USERNAME=<USER>; export MLFLOW_TRACKING_PASSWORD=<TOKEN>
mlflow.set_tracking_uri("https://dagshub.com/leonardo.quiroga/telcovision-mlops.mlflow")

mlflow.set_experiment("telcovision_Experimentos")

print(f"[INFO] MLflow tracking URI: {mlflow.get_tracking_uri()}")

# ---------- NUEVA SECCIÓN: CARGA DE PARÁMETROS DE params.yaml ----------
params_path = PROJECT_ROOT / "params.yaml"

if not params_path.exists():
    raise FileNotFoundError(f"No se encontró {params_path}. Asegúrate de crear el archivo.")

with open(params_path, 'r') as f:
    config = yaml.safe_load(f)

# Extraer parámetros de entrenamiento y general settings
try:
    lr_params = config['train']['lr_params']
    rf_params = config['train']['rf_params']
    
except KeyError as e:
    raise KeyError(f"Error al leer la estructura de 'params.yaml'. Falta la clave: {e}")
# ----------------- FIN DE CARGA DE PARÁMETROS --------------------------

# ---------- CARGA Y PREPROCESAMIENTO DE DATOS ----------
data_path = PROJECT_ROOT / "data" / "processed" / "telco_churn_ok.csv"
if not data_path.exists():
    raise FileNotFoundError(
        f"No se encontró {data_path}. Descarga el dataset de UCI y guárdalo con ese nombre."
    )

df = pd.read_csv(data_path)

# Asegurar que la columna objetivo churn sea binaria 0/1
if df["churn"].dtype != np.int64 and df["churn"].dtype != np.int32:
    # churn es booleana True/False en el dataset original
    df["churn"] = df["churn"].astype(int)

y = df["churn"].values
X = df.drop(columns=["churn"])

# Identificar columnas categóricas vs numéricas
cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

# Preprocesamiento
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

# Conjunto train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- EXPERIMENTO PRINCIPAL CON RUNS ANIDADOS ----------
with mlflow.start_run(run_name="Comparativa_Modelos") as parent_run:

    # Helper para entrenar, evaluar y loggear
    def train_evaluate_log(model_name, model_obj, params: dict):
        with mlflow.start_run(run_name=model_name, nested=True) as child_run:
            # Pipeline completo
            pipe = Pipeline(steps=[("prep", preprocessor), ("clf", model_obj)])

            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            # Algunas métricas (roc_auc necesita proba o decision_function)
            if hasattr(pipe.named_steps["clf"], "predict_proba"):
                y_proba = pipe.predict_proba(X_test)[:, 1]
            else:
                # Fallback para modelos sin predict_proba
                try:
                    y_proba = pipe.decision_function(X_test)
                except Exception:
                    y_proba = None

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred)),
            }
            if y_proba is not None:
                try:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
                except Exception:
                    pass

            # Log params + metrics + modelo
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            # mlflow.sklearn.log_model(pipe, artifact_path="model")
            model_artifact_path = Path("models") / f"{model_name}_pipeline.pkl"
            model_artifact_path.parent.mkdir(exist_ok=True, parents=True)

            with open(model_artifact_path, "wb") as f:
                pickle.dump(pipe, f)

            # Log el modelo serializado como artefacto
            mlflow.log_artifact(str(model_artifact_path), artifact_path="model")

            # Artefactos: matriz de confusión simple
            cm = confusion_matrix(y_test, y_pred)
            plt.figure()
            plt.imshow(cm, interpolation="nearest")
            plt.title(f"Matriz de confusión - {model_name}")
            plt.colorbar()
            plt.xlabel("Predicho")
            plt.ylabel("Real")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i, j], ha="center", va="center")
            artifact_dir = PROJECT_ROOT / "models"
            artifact_dir.mkdir(exist_ok=True, parents=True)
            fig_path = artifact_dir / f"{model_name}_confusion_matrix.png"
            plt.savefig(fig_path)
            plt.close()
            mlflow.log_artifact(str(fig_path))

            print(f"[INFO] {model_name} metrics: {metrics}")
            return metrics

    # 1) RandomForest
    # rf_params = {"n_estimators": 200, "max_depth": 10, "random_state": 42}
    train_evaluate_log("RandomForest", 
                       model_obj=__import__("sklearn.ensemble").ensemble.RandomForestClassifier(**rf_params),
                       params=rf_params)

    # 2) LogisticRegression
    # lr_params = {"C": 1.0, "max_iter": 500, "solver": "lbfgs"}
    train_evaluate_log("LogisticRegression", 
                       model_obj=__import__("sklearn.linear_model").linear_model.LogisticRegression(**lr_params),
                       params=lr_params)

print("[OK] Ejecución finalizada. Revisa la UI de MLflow.")
