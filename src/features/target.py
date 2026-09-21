import numpy as np
import pandas as pd

def calcular_riesgo_seccional(
    df: pd.DataFrame, 
    col_participacion: str = "participacion",
    col_entidad: str = "EDOCVE"
) -> pd.DataFrame:
    """
    Calcula el nivel de riesgo de abstención anómala por sección electoral
    normalizando la participación respecto a la media y desviación estándar de su entidad.
    
    Regla del Semáforo:
    - Riesgo Crítico (Rojo, 2): Z-Score < -1.25 (o caída neta severa)
    - Riesgo Medio   (Amarillo, 1): -1.25 <= Z-Score < -0.50
    - Riesgo Bajo    (Verde, 0): Z-Score >= -0.50
    """
    df = df.copy()

    # Calcular media y desviación estándar por entidad para ese año
    stats_estatales = df.groupby(col_entidad)[col_participacion].agg(["mean", "std"]).reset_index()
    stats_estatales.rename(columns={"mean": "media_estatal", "std": "std_estatal"}, inplace=True)
    stats_estatales["std_estatal"] = stats_estatales["std_estatal"].replace(0, 0.01)

    df = df.merge(stats_estatales, on=col_entidad, how="left")

    # Z-Score: cuántas desviaciones estándar está la casilla respecto a su estado
    df["z_score_estatal"] = (df[col_participacion] - df["media_estatal"]) / df["std_estatal"]
    df["brecha_estatal_pct"] = (df[col_participacion] - df["media_estatal"]) * 100

    # Clasificación del semáforo con asignación directa
    df["riesgo_nivel"] = 0  # Por defecto: Riesgo Bajo (Verde)
    df.loc[df["z_score_estatal"].between(-1.25, -0.50), "riesgo_nivel"] = 1  # Riesgo Medio (Amarillo)
    df.loc[df["z_score_estatal"] < -1.25, "riesgo_nivel"] = 2  # Riesgo Crítico (Rojo)

    mapeo_etiquetas = {0: "Bajo", 1: "Medio", 2: "Critico"}
    df["riesgo_etiqueta"] = df["riesgo_nivel"].map(mapeo_etiquetas)

    return df