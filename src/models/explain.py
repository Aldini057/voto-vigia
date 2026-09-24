import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

print("=== FASE 3.2: EXPLICABILIDAD E INTERPRETABILIDAD NATIVA CATBOOST ===")

RUTA_DATASET = "data/processed/dataset_maestro.parquet"
RUTA_MODELO_CBM = "models/catboost_model.cbm"
DIR_DOCS = "docs"
os.makedirs(DIR_DOCS, exist_ok=True)

# 1. Cargar el modelo entrenado y los datos de prueba (2024)
if not os.path.exists(RUTA_MODELO_CBM):
    print(f"[ERROR] No se encontró el modelo en {RUTA_MODELO_CBM}. Ejecuta train.py primero.")
    exit(1)

modelo = CatBoostClassifier()
modelo.load_model(RUTA_MODELO_CBM)
print("Modelo CatBoost cargado correctamente.")

df = pd.read_parquet(RUTA_DATASET)
df_2024 = df[df["AELEC"] == 2024].copy()

# Variables predictivas
FEATURES_NUMERICAS = [
    "part_lag_inmediata", "part_lag_homologa", "delta_historica_previa",
    "pct_crecimiento_ln", "pct_jovenes_19_29", "pct_18", "pct_mayores_65",
    "ratio_mujeres", "ln_total", "es_eleccion_presidencial"
]
FEATURES_CATEGORICAS = ["EDOCVE", "MPIOCVE", "TIPOSEC", "DEF"]
COLUMNAS_X = FEATURES_NUMERICAS + FEATURES_CATEGORICAS

for cat in FEATURES_CATEGORICAS:
    df_2024[cat] = df_2024[cat].astype(str).fillna("DESCONOCIDO")
for num in FEATURES_NUMERICAS:
    df_2024[num] = pd.to_numeric(df_2024[num], errors="coerce").fillna(0.0)

X_2024 = df_2024[COLUMNAS_X]
pool_2024 = Pool(X_2024, cat_features=FEATURES_CATEGORICAS)

# 2. Importancia Global de Variables (Feature Importance Nativo)
print("\n--- FACTORES QUE MÁS DETONAN EL RIESGO DE ABSTENCIÓN (TOP 10) ---")
importancias = modelo.get_feature_importance(pool_2024, type="FeatureImportance")
df_importancia = pd.DataFrame({
    "Variable": COLUMNAS_X,
    "Importancia (%)": importancias
}).sort_values(by="Importancia (%)", ascending=False).reset_index(drop=True)

for idx, fila in df_importancia.head(10).iterrows():
    print(f"  {idx+1:02d}. {fila['Variable']:<25} : {fila['Importancia (%)']:.2f}%")

# Guardar ranking en CSV para el reporte del INE
df_importancia.to_csv(os.path.join(DIR_DOCS, "feature_importance.csv"), index=False)
print(f"\n[OK] Ranking de importancia guardado en docs/feature_importance.csv")

# 3. Diagnóstico de Casillas en Riesgo Crítico (Focos Rojos)
print("\n--- CASILLAS EN RIESGO CRÍTICO DETECTADAS EN 2024 ---")
predicciones = modelo.predict(pool_2024).flatten()
df_2024["riesgo_predicho"] = predicciones

secciones_criticas = df_2024[df_2024["riesgo_predicho"] == 2]
print(f"Total de casillas en Riesgo Crítico predichas: {len(secciones_criticas):,} ({len(secciones_criticas)/len(df_2024)*100:.1f}% del país)")

if len(secciones_criticas) > 0:
    ejemplo = secciones_criticas.iloc[0]
    print(f"\nEjemplo de Ficha Técnica Institucional:")
    print(f"  Entidad / Municipio: {ejemplo['EDONOM']} - {ejemplo['MPIONOM']}")
    print(f"  Sección Electoral:   {ejemplo['SECCION']} (Tipo: {ejemplo['TIPOSEC']})")
    print(f"  Participación Real:  {ejemplo['participacion']*100:.1f}%")
    print(f"  % Jóvenes (19-29):   {ejemplo['pct_jovenes_19_29']*100:.1f}%")
    print(f"  Participación t-1:   {ejemplo['part_lag_inmediata']*100:.1f}%")
    print(f"  -> Diagnóstico: Foco rojo anticipado por inercia histórica y concentración etaria.")

print("\n" + "="*60)
print("¡EXPLICABILIDAD COMPLETADA CON ÉXITO!")
print("="*60)