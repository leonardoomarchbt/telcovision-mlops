
# Laboratorio de Mineria de Datos

# Proyecto Integrador: MLflow + DVC + Git + DagsHub (End-to-End)

## Alumno: Leonardo Omar Quiroga

## Objetivo
Construir, versionar y publicar un proyecto completo de Machine Learning, registrando y comparando experimentos.

---

## Requisitos previos
- Anaconda
- Notepad ++
- Jupyter Lab
- Git
- DagsHub


---

## 1) Estructura del proyecto
```
tecnovision-mlops/
				 ├── .github/workflows
				 ├── data/raw/
				 │           └── telco_churn.csv
				 ├── models/
				 ├── src/
				 ├── requirements.txt
				 ├── .gitignore
				 ├── params.yaml
				 └── README.md
```

---

## 2) EDA
Abrir y ejecutar data_prep.py para preparar datos.

---

## 3) Entrenamiento de modelo
Abrir y ejecutar train.py para entrenar modelo

---

## 4) Experimentos
modificando hiperparametros (en params.yaml) durante 4 corridas suscesivas, se determino que los mejores valores fueron, para el modelo de regresion:
  lr_params:
    C: 0.1 
    max_iter: 200 
    class_weight: 'balanced'  

que obtuvo las siguientes metricas:
	accuracy = 0.6585
	roc_auc = 0.7265
	f1 = 0.5994
	
---

## 5) CI/CD con GitHub Actions
Se configuro ci.yaml

se instalaron dependencias

dvc pull

dvc repro

metricas en log

---

## 6) Iteracion colaborativa
se crea rama feat-propuesta-final-v2.
se realiza pull-requests.
y merge del mejor experimento a master.
se realiza validacion automatica con CI/CD.

--



---




