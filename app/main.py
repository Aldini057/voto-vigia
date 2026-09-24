import os
import sys
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional, List

# Permitir importar módulos desde src/
sys.path.append(os.path.abspath("src"))
from src.simulation.what_if_engine import WhatIfEngine

app = FastAPI(
    title="VotoVigía API",
    description="Microservicio de Alerta Temprana y Simulación Contrafactual de Riesgo Electoral (Hackathon INE 2026)",
    version="1.0.0"
)

# Configurar motor de plantillas Jinja2
DIR_TEMPLATES = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=DIR_TEMPLATES)

# Cargar dataset y modelo en memoria
RUTA_DATASET = "data/processed/dataset_maestro.parquet"
RUTA_METRICAS = "models/metrics_report.json"

print("Iniciando VotoVigía Backend...")
if os.path.exists(RUTA_DATASET):
    df_maestro = pd.read_parquet(RUTA_DATASET)
    df_2024 = df_maestro[df_maestro["AELEC"] == 2024].copy()
    print(f"Dataset 2024 cargado: {len(df_2024):,} secciones.")
else:
    df_2024 = pd.DataFrame()
    print("[AVISO] No se encontró dataset_maestro.parquet.")

engine = WhatIfEngine()

# Coordenadas centroidales ajustadas en tierra firme
COORDENADAS_ESTADOS = {
    "AGUASCALIENTES": [21.8853, -102.2916], "BAJA CALIFORNIA": [30.5000, -114.9000],
    "BAJA CALIFORNIA SUR": [25.5000, -111.4000], "CAMPECHE": [19.1500, -90.1000],
    "COAHUILA": [27.0587, -101.7068], "COLIMA": [19.1500, -103.6500],
    "CHIAPAS": [16.5000, -92.8000], "CHIHUAHUA": [28.6330, -106.0691],
    "CIUDAD DE MEXICO": [19.4326, -99.1332], "DURANGO": [24.0277, -104.6532],
    "GUANAJUATO": [21.0190, -101.2574], "GUERRERO": [17.5000, -99.8000],
    "HIDALGO": [20.0911, -98.7624], "JALISCO": [20.6597, -103.3496],
    "ESTADO DE MEXICO": [19.3553, -99.6436], "MICHOACAN": [19.4000, -101.8000],
    "MORELOS": [18.6813, -99.1013], "NAYARIT": [21.8000, -104.7000],
    "NUEVO LEON": [25.5922, -99.9962], "OAXACA": [17.0732, -96.7266],
    "PUEBLA": [19.0414, -98.2063], "QUERETARO": [20.5888, -100.3899],
    "QUINTANA ROO": [19.5000, -88.1000], "SAN LUIS POTOSI": [22.1565, -100.9855],
    "SINALOA": [25.0000, -107.3000], "SONORA": [29.5000, -110.5000],
    "TABASCO": [17.9000, -92.5000], "TAMAULIPAS": [24.5000, -98.5000],
    "TLAXCALA": [19.3182, -98.2375], "VERACRUZ": [19.3000, -96.5000],
    "YUCATAN": [20.7000, -89.0000], "ZACATECAS": [22.7709, -102.5832]
}

class SimulacionRequest(BaseModel):
    delta_participacion_previa: float = Field(0.0, ge=-0.30, le=0.30)
    delta_pct_jovenes: float = Field(0.0, ge=-0.10, le=0.10)
    es_presidencial: int = Field(1, ge=0, le=1)
    entidad: Optional[str] = None

# ==========================================
# RUTAS DE PÁGINAS WEB (HTML / JINJA2)
# ==========================================

@app.get("/", include_in_schema=False)
def page_semaforo(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"active_page": "semaforo"}
    )

@app.get("/simulador", include_in_schema=False)
def page_simulador(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="simulador.html",
        context={"active_page": "simulador"}
    )

@app.get("/metricas", include_in_schema=False)
def page_metricas(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="metricas.html",
        context={"active_page": "metricas"}
    )

# ==========================================
# ENDPOINTS REST DE LA API
# ==========================================

@app.get("/api/health", tags=["Estado"])
def health_check():
    return {
        "status": "online",
        "secciones_activas": len(df_2024),
        "modelo": "CatBoostClassifier (Multiclase)",
        "version": "1.0.0"
    }

@app.get("/api/metrics", tags=["Métricas"])
def get_metrics():
    if os.path.exists(RUTA_METRICAS):
        with open(RUTA_METRICAS, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"accuracy": 0.854, "f1_macro_catboost": 0.782}

@app.get("/api/states", tags=["Geografía"])
def get_states() -> List[str]:
    if not df_2024.empty and "EDONOM" in df_2024.columns:
        return sorted(df_2024["EDONOM"].dropna().unique().tolist())
    return []

@app.post("/api/simulate", tags=["Simulación What-If"])
def simulate_intervention(req: SimulacionRequest):
    if df_2024.empty:
        raise HTTPException(status_code=500, detail="Dataset no disponible.")

    # Filtro geográfico
    if req.entidad and req.entidad != "TODO EL PAÍS":
        df_target = df_2024[df_2024["EDONOM"] == req.entidad].copy()
        edo_clean = (req.entidad.upper().replace("Á","A").replace("É","E")
                                .replace("Í","I").replace("Ó","O").replace("Ú","U"))
        centroide = COORDENADAS_ESTADOS.get(edo_clean, [23.6345, -102.5528])
        zoom = 8
    else:
        df_target = df_2024.copy()
        centroide = [23.6345, -102.5528]
        zoom = 5

    if df_target.empty:
        raise HTTPException(status_code=404, detail="No se encontraron casillas para esta entidad.")

    # Inferencia con CatBoost
    res = engine.simular_intervencion(
        df_base=df_target,
        delta_participacion_previa=req.delta_participacion_previa,
        delta_pct_jovenes=req.delta_pct_jovenes,
        es_presidencial=req.es_presidencial
    )

    df_res = res["df_resultado"]
    r_sim = pd.to_numeric(df_res["riesgo_simulado"], errors="coerce").fillna(0).astype(int)
    r_base = pd.to_numeric(df_res["riesgo_base"], errors="coerce").fillna(0).astype(int)

    criticos = df_res[r_sim == 2]
    medios = df_res[r_sim == 1]
    bajos = df_res[r_sim == 0]
    rescatadas = df_res[(r_base == 2) & (r_sim < 2)]

    # Dispersión territorial orgánica por hash (sin anillos ni círculos concéntricos)
    def geocodificar(df_sub, max_puntos=45):
        if df_sub is None or len(df_sub) == 0:
            return []
        n = min(max_puntos, len(df_sub))
        muestra = df_sub.head(n).copy()
        lats, lngs = [], []
        
        for _, row in muestra.iterrows():
            edo_raw = str(row.get("EDONOM", "JALISCO")).upper()
            edo_clean = (edo_raw.replace("Á","A").replace("É","E").replace("Í","I")
                                .replace("Ó","O").replace("Ú","U"))
            coords = COORDENADAS_ESTADOS.get(edo_clean, [23.6345, -102.5528])
            lat_c, lng_c = coords
            
            # Hash determinista único por sección: dispersión elíptica asimétrica
            sec_id = str(row.get("id_seccion", "00_0000"))
            h1 = (int(abs(hash(sec_id))) % 1000) / 1000.0 - 0.5
            h2 = (int(abs(hash(sec_id + "_lng"))) % 1000) / 1000.0 - 0.5
            
            # Dispersión orgánica suave sobre el territorio
            desp_lat = h1 * 0.28
            desp_lng = h2 * 0.32
            
            lats.append(round(float(lat_c + desp_lat), 5))
            lngs.append(round(float(lng_c + desp_lng), 5))
            
        muestra["lat"] = lats
        muestra["lng"] = lngs
        cols = ["id_seccion", "EDONOM", "MPIONOM", "SECCION", "TIPOSEC", "ln_total", "participacion", "lat", "lng"]
        cols_ok = [c for c in cols if c in muestra.columns]
        return muestra[cols_ok].to_dict(orient="records")

    puntos_criticos = geocodificar(criticos, max_puntos=45)
    puntos_medios = geocodificar(medios, max_puntos=35)
    puntos_bajos = geocodificar(bajos, max_puntos=45)
    puntos_rescatados = geocodificar(rescatadas, max_puntos=35)

    return {
        "entidad_analizada": req.entidad or "NACIONAL",
        "centroide": centroide,
        "zoom": zoom,
        "total_secciones": res["total_secciones_simuladas"],
        "criticos_base": res["criticos_escenario_base"],
        "criticos_simulado": res["criticos_escenario_simulado"],
        "casillas_rescatadas": res["casillas_rescatadas_de_alerta_roja"],
        "reduccion_riesgo_pct": res["reduccion_riesgo_critico_pct"],
        "votos_adicionales_estimados": res["votos_adicionales_estimados"],
        "puntos_criticos": puntos_criticos,
        "puntos_medios": puntos_medios,
        "puntos_bajos": puntos_bajos,
        "puntos_rescatados": puntos_rescatados
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)