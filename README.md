# Análisis y Predicción de Enfermedades Cardíacas

Trabajo Fin de Máster (TFM) de **Guillermo Gil Ochoa**. Este proyecto analiza los factores de riesgo asociados a las enfermedades cardíacas y compara distintos modelos de *machine learning* para predecir si una persona padece una enfermedad cardíaca a partir de indicadores de salud y hábitos de vida.

## Dataset

`heart_2020_cleaned.csv` contiene **319.795 registros** de la encuesta *Personal Key Indicators of Heart Disease* (CDC, BRFSS 2020), con 18 variables:

| Tipo | Variables |
|------|-----------|
| Objetivo | `HeartDisease` (Yes / No) |
| Numéricas | `BMI`, `PhysicalHealth`, `MentalHealth`, `SleepTime` |
| Categóricas | `Smoking`, `AlcoholDrinking`, `Stroke`, `DiffWalking`, `Sex`, `AgeCategory`, `Race`, `Diabetic`, `PhysicalActivity`, `GenHealth`, `Asthma`, `KidneyDisease`, `SkinCancer` |

El dataset está muy desbalanceado: solo ~9 % de los registros corresponden a personas con enfermedad cardíaca.

## Contenido del notebook

`TFM_GuillermoGilOchoa_HeartDisease.ipynb` está organizado en:

1. **Importación de librerías**
2. **Carga del dataset**
3. **Análisis exploratorio (EDA)**: distribución por edad, sexo y raza, IMC, salud física y mental, horas de sueño, tabaco, alcohol, actividad física, estado general de salud y correlaciones entre variables numéricas.
4. **Preparación de datos**
   - Creación de la variable `BMI_Category` (bajo peso, normal, sobrepeso, obesidad I–III).
   - Codificación ordinal y *one-hot* de variables categóricas, y escalado de las numéricas.
   - División train/test 80/20.
   - Rebalanceo de clases con **SMOTEENN**.
5. **Entrenamiento de modelos**: Regresión Logística, Random Forest, Gradient Boosting, XGBoost, LightGBM, Árbol de Decisión, KNN, Naive Bayes, SVM, Red Neuronal (ANN), Red Recurrente (RNN), K-Means, PCA + regresión y Modelos Ocultos de Markov (HMM).
6. **Comparativa de resultados**

## Resultados principales

Resultados sobre el conjunto de test (63.959 registros):

| Modelo | Accuracy | Recall (clase enfermo) | F1 (clase enfermo) |
|--------|:--------:|:----------------------:|:------------------:|
| Regresión Logística | 0.74 | **0.76** | 0.34 |
| Gradient Boosting | 0.81 | 0.61 | 0.36 |
| Naive Bayes | 0.84 | 0.47 | 0.34 |
| Random Forest | 0.86 | 0.45 | 0.35 |
| Árbol de Decisión | 0.84 | 0.39 | 0.30 |
| LightGBM | 0.88 | 0.41 | **0.37** |
| XGBoost | 0.88 | 0.40 | **0.37** |
| SVM | **0.91** | 0.11 | 0.19 |

Debido al desbalanceo de clases, la *accuracy* por sí sola es engañosa: SVM obtiene la mayor *accuracy* pero apenas detecta casos positivos. Para un problema de cribado médico, la **Regresión Logística** ofrece el mejor *recall* (detecta el 76 % de los enfermos), mientras que **XGBoost** y **LightGBM** ofrecen el mejor equilibrio global.

## Tecnologías

Python · pandas · NumPy · Matplotlib · Seaborn · Plotly · scikit-learn · imbalanced-learn · XGBoost · LightGBM · TensorFlow/Keras · hmmlearn

## Cómo ejecutarlo

```bash
git clone https://github.com/Ggil22/Analisis_y_Prediccion_Enfermedades_Cardiacas.git
cd Analisis_y_Prediccion_Enfermedades_Cardiacas
pip install -r requirements.txt
jupyter notebook TFM_GuillermoGilOchoa_HeartDisease.ipynb
```

> **Nota:** la celda de carga del dataset usa una ruta absoluta local. Cámbiala por `pd.read_csv("heart_2020_cleaned.csv")` antes de ejecutar el notebook. Las versiones de `requirements.txt` (p. ej. TensorFlow 2.4) requieren una versión de Python antigua (3.8); con versiones más recientes puede ser necesario actualizar las dependencias.

## Estructura

```
├── TFM_GuillermoGilOchoa_HeartDisease.ipynb   # Notebook con el análisis y los modelos
├── heart_2020_cleaned.csv                     # Dataset
└── requirements.txt                           # Dependencias
```

## Autor

**Guillermo Gil Ochoa** — [GitHub](https://github.com/Ggil22)
