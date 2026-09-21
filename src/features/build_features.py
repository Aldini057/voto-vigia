import os
import pandas as pd
from target import calcular_riesgo_seccional

print("=== INICIANDO CONSTRUCCIÓN DEL DATASET MAESTRO (FEATURE ENGINEERING) ===")

ANIOS = [2009, 2012, 2015, 2018, 2021, 2024]
DIR_PROCESSED = "data/processed"
RUTA_DATASET_MAESTRO = os.path.join(DIR_PROCESSED, "dataset_maestro.parquet")

# 1. Cargar todos los años procesados
datos_anuales = {}
for anio in ANIOS:
    ruta_p = os.path.join(DIR_PROCESSED, f"secciones_{anio}.parquet")
    if os.path.exists(ruta_p):
        df_a = pd.read_parquet(ruta_p)
        df_a["AELEC"] = anio
        datos_anuales[anio] = df_a
        print(f"  [Cargado] Año {anio}: {len(df_a):,} secciones.")
    else:
        print(f"  [ERROR] Falta {ruta_p}")

# 2. Construir matriz temporal de entrenamiento
# Usaremos como objetivos los años 2018, 2021 y 2024 con sus respectivos rezagos
ANIOS_TARGET = [2018, 2021, 2024]
filas_entrenamiento = []

for anio in ANIOS_TARGET:
    df_actual = datos_anuales[anio].copy()
    
    # Elección inmediata anterior (t-1) y elección homóloga previa (t-2)
    idx_actual = ANIOS.index(anio)
    anio_lag1 = ANIOS[idx_actual - 1] # t-1
    anio_lag2 = ANIOS[idx_actual - 2] # t-2 (homóloga)

    df_lag1 = datos_anuales[anio_lag1][["id_seccion", "participacion", "ln_total", "part_jovenes_19_29"]].rename(
        columns={
            "participacion": f"part_lag_{anio_lag1}",
            "ln_total": f"ln_lag_{anio_lag1}",
            "part_jovenes_19_29": f"part_joven_lag_{anio_lag1}"
        }
    )
    
    df_lag2 = datos_anuales[anio_lag2][["id_seccion", "participacion"]].rename(
        columns={"participacion": f"part_lag_homologa_{anio_lag2}"}
    )

    # Cruzar datos actuales con la historia previa de cada sección
    df_feat = df_actual.merge(df_lag1, on="id_seccion", how="inner")
    df_feat = df_feat.merge(df_lag2, on="id_seccion", how="inner")

    # Rezagos homologados (nombres estándar para el modelo)
    df_feat["part_lag_inmediata"] = df_feat[f"part_lag_{anio_lag1}"]
    df_feat["part_lag_homologa"] = df_feat[f"part_lag_homologa_{anio_lag2}"]
    df_feat["delta_historica_previa"] = df_feat["part_lag_inmediata"] - df_feat["part_lag_homologa"]
    
    # Crecimiento o despoblamiento de la lista nominal
    df_feat["pct_crecimiento_ln"] = (
        (df_feat["ln_total"] - df_feat[f"ln_lag_{anio_lag1}"]) / df_feat[f"ln_lag_{anio_lag1}"].replace(0, 1)
    )

    # Bandera institucional
    df_feat["es_eleccion_presidencial"] = 1 if anio in [2012, 2018, 2024] else 0

    # Calcular el Target y Semáforo de Riesgo para este año
    df_feat = calcular_riesgo_seccional(df_feat, col_participacion="participacion", col_entidad="EDOCVE")

    # Limpiar columnas auxiliares específicas de año
    df_feat.drop(columns=[f"part_lag_{anio_lag1}", f"ln_lag_{anio_lag1}", f"part_joven_lag_{anio_lag1}", f"part_lag_homologa_{anio_lag2}"], inplace=True)

    filas_entrenamiento.append(df_feat)
    print(f"  [Listo] Año objetivo {anio} procesado con historia previa: {len(df_feat):,} secciones.")

# 3. Consolidar el dataset maestro
df_maestro = pd.concat(filas_entrenamiento, ignore_index=True)

# Guardar en disco
df_maestro.to_parquet(RUTA_DATASET_MAESTRO, index=False)

print("\n" + "="*60)
print(f"¡DATASET MAESTRO GENERADO EXITOSAMENTE!")
print(f"Ruta: {RUTA_DATASET_MAESTRO}")
print(f"Dimensiones totales: {df_maestro.shape[0]:,} filas x {df_maestro.shape} columnas.")
print("Distribución del Semáforo de Riesgo:")
print(df_maestro["riesgo_etiqueta"].value_counts(normalize=True).mul(100).round(2).astype(str) + " %")
print("="*60)