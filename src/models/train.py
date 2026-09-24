import os
import json
import time
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.dummy import DummyClassifier

print("=== FASE 3: ENTRENAMIENTO Y VALIDACIÓN TEMPORAL (CATBOOST) ===")

RUTA_DATASET = "data/processed/dataset_maestro.parquet"
DIR_MODELS = "models"
RUTA_MODELO_CBM = os.path.join(DIR_MODELS, "catboost_model.cbm")
RUTA_METRICAS = os.path.join(DIR_MODELS, "metrics_report.json")

os.makedirs(DIR_MODELS, exist_ok=True)

# 1. Cargar el Dataset Maestro
if not os.path.exists(RUTA_DATASET):
    print(f"[ERROR] No se encontró el dataset en {RUTA_DATASET}. Ejecuta build_features.py primero.")
    exit(1)

df = pd.read_parquet(RUTA_DATASET)
print(f"Dataset cargado con éxito: {len(df):,} filas y {len(df.columns)} columnas.")

# 2. Definición de Variables (X) y Objetivo (y)
# Solo variables disponibles ANTES de la elección (cero fuga de datos)
FEATURES_NUMERICAS = [
    "part_lag_inmediata",       # Participación en t-1
    "part_lag_homologa",        # Participación en t-2
    "delta_historica_previa",   # Tendencia previa
    "pct_crecimiento_ln",       # Crecimiento/despoblamiento de la lista nominal
    "pct_jovenes_19_29",        # % de jóvenes de 19 a 29 años
    "pct_18",                   # % de votantes de 18 años
    "pct_mayores_65",           # % de adultos mayores
    "ratio_mujeres",            # % de mujeres en el padrón
    "ln_total",                 # Tamaño de la casilla
    "es_eleccion_presidencial"  # 1: Presidencial, 0: Intermedia
]

FEATURES_CATEGORICAS = [
    "EDOCVE",                   # Clave Entidad
    "MPIOCVE",                  # Clave Municipio
    "TIPOSEC",                  # Urbana (U), Rural (R), Mixta (M)
    "DEF"                       # Distrito Electoral Federal
]

COLUMNAS_X = FEATURES_NUMERICAS + FEATURES_CATEGORICAS
COLUMNA_Y = "riesgo_nivel" # 0: Bajo, 1: Medio, 2: Crítico

# Limpieza y tipos para CatBoost
for cat in FEATURES_CATEGORICAS:
    df[cat] = df[cat].astype(str).fillna("DESCONOCIDO")

for num in FEATURES_NUMERICAS:
    df[num] = pd.to_numeric(df[num], errors="coerce").fillna(0.0)

# 3. División Temporal (Time-Series Split)
# Entrenamiento: 2018 y 2021 | Prueba Ciega: 2024
train_mask = df["AELEC"].isin([2018, 2021])
test_mask = df["AELEC"] == 2024

X_train, y_train = df.loc[train_mask, COLUMNAS_X], df.loc[train_mask, COLUMNA_Y]
X_test, y_test = df.loc[test_mask, COLUMNAS_X], df.loc[test_mask, COLUMNA_Y]

print(f"\nConjunto de Entrenamiento (2018-2021): {len(X_train):,} secciones.")
print(f"Conjunto de Evaluación Ciega (2024):  {len(X_test):,} secciones.")

# 4. Entrenar Modelo Base (Baseline) para Comparativa
print("\n--- Entrenando Modelo Base de Referencia (Dummy) ---")
baseline = DummyClassifier(strategy="stratified", random_state=42)
baseline.fit(X_train[FEATURES_NUMERICAS], y_train)
y_pred_base = baseline.predict(X_test[FEATURES_NUMERICAS])
f1_base_macro = f1_score(y_test, y_pred_base, average="macro")
print(f"F1-Score Macro del Baseline: {f1_base_macro:.4f}")

# 5. Entrenar CatBoostClassifier
print("\n--- Entrenando CatBoostClassifier con Balanceo de Clases ---")
pool_train = Pool(X_train, y_train, cat_features=FEATURES_CATEGORICAS)
pool_test = Pool(X_test, y_test, cat_features=FEATURES_CATEGORICAS)

modelo_catboost = CatBoostClassifier(
    iterations=800,
    learning_rate=0.06,
    depth=6,
    loss_function="MultiClass",
    auto_class_weights="Balanced", # Da mayor peso a las picadas críticas
    random_seed=42,
    verbose=100
)

t0 = time.time()
modelo_catboost.fit(pool_train, eval_set=pool_test, early_stopping_rounds=50, verbose=100)
duracion_train = time.time() - t0
print(f"\nEntrenamiento completado en {duracion_train:.1f} segundos.")

# 6. Evaluación Rigurosa sobre 2024
y_pred = modelo_catboost.predict(pool_test).flatten()
y_proba = modelo_catboost.predict_proba(pool_test)

reporte_dict = classification_report(
    y_test, y_pred, 
    target_names=["Bajo (Verde)", "Medio (Amarillo)", "Critico (Rojo)"],
    output_dict=True
)

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")

print("\n" + "="*65)
print("REPORTE DE EVALUACIÓN SOBRE LA ELECCIÓN FEDERAL 2024 (TEST SET):")
print("="*65)
print(classification_report(y_test, y_pred, target_names=["Bajo (Verde)", "Medio (Amarillo)", "Critico (Rojo)"]))
print(f"Accuracy Global: {acc*100:.2f}%")
print(f"F1-Score Macro:  {f1_macro:.4f} (vs {f1_base_macro:.4f} del Baseline)")

print("\nMatriz de Confusión:")
matriz = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(matriz, 
                   index=["Real Bajo", "Real Medio", "Real Critico"], 
                   columns=["Pred Bajo", "Pred Medio", "Pred Critico"]))

# 7. Guardar Artefactos del Modelo
modelo_catboost.save_model(RUTA_MODELO_CBM)
print(f"\n[OK] Modelo CatBoost serializado guardado en: {RUTA_MODELO_CBM}")

metricas_exportar = {
    "fecha_entrenamiento": time.strftime("%Y-%m-%d %H:%M:%S"),
    "duracion_segundos": round(duracion_train, 2),
    "secciones_entrenamiento": len(X_train),
    "secciones_prueba_2024": len(X_test),
    "accuracy": round(acc, 4),
    "f1_macro_catboost": round(f1_macro, 4),
    "f1_macro_baseline": round(f1_base_macro, 4),
    "reporte_clasificacion": reporte_dict,
    "features_utilizadas": COLUMNAS_X
}

with open(RUTA_METRICAS, "w", encoding="utf-8") as f:
    json.dump(metricas_exportar, f, indent=4, ensure_ascii=False)

print(f"[OK] Reporte de métricas JSON guardado en: {RUTA_METRICAS}")
print("="*65)