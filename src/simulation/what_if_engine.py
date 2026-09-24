import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

print("=== FASE 4: MOTOR DE SIMULACIÓN CONTRAFACTUAL (WHAT-IF ENGINE) ===")

RUTA_MODELO = "models/catboost_model.cbm"
RUTA_DATASET = "data/processed/dataset_maestro.parquet"

FEATURES_NUMERICAS = [
    "part_lag_inmediata", "part_lag_homologa", "delta_historica_previa",
    "pct_crecimiento_ln", "pct_jovenes_19_29", "pct_18", "pct_mayores_65",
    "ratio_mujeres", "ln_total", "es_eleccion_presidencial"
]
FEATURES_CATEGORICAS = ["EDOCVE", "MPIOCVE", "TIPOSEC", "DEF"]
COLUMNAS_X = FEATURES_NUMERICAS + FEATURES_CATEGORICAS

class WhatIfEngine:
    def __init__(self, ruta_modelo: str = RUTA_MODELO):
        if not os.path.exists(ruta_modelo):
            raise FileNotFoundError(f"No se encontró el modelo en: {ruta_modelo}")
        
        self.modelo = CatBoostClassifier()
        self.modelo.load_model(ruta_modelo)
        print("Motor de simulación inicializado con CatBoost.")

    def simular_intervencion(
        self,
        df_base: pd.DataFrame,
        delta_participacion_previa: float = 0.0,
        delta_pct_jovenes: float = 0.0,
        es_presidencial: int = None
    ) -> dict:
        df_sim = df_base.copy()

        # 1. Preparar datos base
        for cat in FEATURES_CATEGORICAS:
            df_sim[cat] = df_sim[cat].astype(str).fillna("DESCONOCIDO")
        for num in FEATURES_NUMERICAS:
            df_sim[num] = pd.to_numeric(df_sim[num], errors="coerce").fillna(0.0)

        pool_base = Pool(df_sim[COLUMNAS_X], cat_features=FEATURES_CATEGORICAS)
        pred_base = self.modelo.predict(pool_base).flatten()
        prob_base = self.modelo.predict_proba(pool_base)

        # Extraer columna de Riesgo Crítico (última columna) como vector 1D
        p_critico_base = prob_base.take(-1, axis=1)

        # 2. Aplicar perturbaciones contrafactuales
        df_sim["part_lag_inmediata"] = np.clip(df_sim["part_lag_inmediata"] + delta_participacion_previa, 0.0, 1.0)
        df_sim["pct_jovenes_19_29"] = np.clip(df_sim["pct_jovenes_19_29"] + delta_pct_jovenes, 0.0, 1.0)
        
        if es_presidencial is not None:
            df_sim["es_eleccion_presidencial"] = es_presidencial

        pool_sim = Pool(df_sim[COLUMNAS_X], cat_features=FEATURES_CATEGORICAS)
        pred_sim = self.modelo.predict(pool_sim).flatten()
        prob_sim = self.modelo.predict_proba(pool_sim)

        # Extraer columna simulada de Riesgo Crítico como vector 1D
        p_critico_sim = prob_sim.take(-1, axis=1)

        # 3. Métricas de impacto
        criticos_base = int((pred_base == 2).sum())
        criticos_sim = int((pred_sim == 2).sum())
        casillas_rescatadas = criticos_base - criticos_sim

        # Cálculo de votos ganados garantizando dimensiones 1D
        delta_p_crit = np.maximum(0.0, p_critico_base - p_critico_sim)
        ln_vector = df_sim["ln_total"].values.astype(float)
        votos_proyectados_ganados = int((delta_p_crit * ln_vector * 0.15).sum())

        return {
            "total_secciones_simuladas": len(df_base),
            "criticos_escenario_base": criticos_base,
            "criticos_escenario_simulado": criticos_sim,
            "casillas_rescatadas_de_alerta_roja": casillas_rescatadas,
            "reduccion_riesgo_critico_pct": round((casillas_rescatadas / criticos_base * 100) if criticos_base > 0 else 0, 2),
            "votos_adicionales_estimados": max(0, votos_proyectados_ganados),
            "df_resultado": df_sim.assign(
                riesgo_base=pred_base,
                riesgo_simulado=pred_sim,
                prob_critico_base=p_critico_base,
                prob_critico_sim=p_critico_sim
            )
        }

if __name__ == "__main__":
    df_maestro = pd.read_parquet(RUTA_DATASET)
    df_2024 = df_maestro[df_maestro["AELEC"] == 2024].copy()

    engine = WhatIfEngine()

    print("\n" + "="*65)
    print("EJECUTANDO SIMULACIÓN DE PRUEBA A NIVEL NACIONAL:")
    print("Escenario: 'Campaña focalizada del INE que incrementa +5% la participación'")
    print("="*65)

    resultado = engine.simular_intervencion(
        df_base=df_2024,
        delta_participacion_previa=0.05,
        es_presidencial=1
    )

    print(f"Total de secciones analizadas:        {resultado['total_secciones_simuladas']:,}")
    print(f"Casillas en Riesgo Crítico (Base):     {resultado['criticos_escenario_base']:,}")
    print(f"Casillas en Riesgo Crítico (Simulado): {resultado['criticos_escenario_simulado']:,}")
    print(f"-> Casillas rescatadas del semáforo rojo: {resultado['casillas_rescatadas_de_alerta_roja']:,} ({resultado['reduccion_riesgo_critico_pct']}%)")
    print(f"-> Votos adicionales proyectados:      +{resultado['votos_adicionales_estimados']:,} votos")
    print("="*65)