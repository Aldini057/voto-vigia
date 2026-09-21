import os
import glob
import time
import pandas as pd

print("=== PROCESANDO EXCLUSIVAMENTE AÑO 2009 ===")

DIR_2009 = "data/raw/DatosAbiertos_DECEyEC_ConteosCensales2009"
SALIDA_2009 = "data/processed/secciones_2009.parquet"

# 1. Verificar existencia de la carpeta y archivos
archivos = glob.glob(os.path.join(DIR_2009, "*.csv"))
if not archivos:
    print(f"[ERROR] No se encontraron archivos CSV en: {DIR_2009}")
    exit(1)

print(f"Se encontraron {len(archivos)} archivos CSV en 2009.")

# 2. Inspeccionar columnas del primer archivo
muestra = pd.read_csv(archivos[0], nrows=5, low_memory=False)
muestra.columns = [c.upper().strip() for c in muestra.columns]
print(f"Columnas detectadas en 2009: {muestra.columns.tolist()}\n")

def mapear_columnas_2009(df: pd.DataFrame) -> pd.DataFrame:
    """Homologa columnas de 2009 con los estándares de 2024."""
    df.columns = [c.upper().strip() for c in df.columns]
    
    mapeo = {
        "ENTIDAD": "EDOCVE",
        "EDO": "EDOCVE",
        "ID_ESTADO": "EDOCVE",
        "CLAVE_ENTIDAD": "EDOCVE",
        "MUNICIPIO": "MPIOCVE",
        "MPIO": "MPIOCVE",
        "CLAVE_MUNICIPIO": "MPIOCVE",
        "SECC": "SECCION",
        "DTO_FED": "DEF",
        "DISTRITO": "DEF",
        "DISTRITO_FEDERAL": "DEF",
        "DTO_LOC": "DEL",
        "DISTRITO_LOCAL": "DEL",
        "TIPO": "TIPOSEC",
        "TIPO_SECCION": "TIPOSEC",
        "LISTA_NOMINAL": "LN",
        "VOTOS": "SV",
        "VOTO": "SV",
        "SI_VOTO": "SV",
        "NO_VOTO": "NV"
    }
    
    df.rename(columns={k: v for k, v in mapeo.items() if k in df.columns and v not in df.columns}, inplace=True)
    
    # Rellenar columnas obligatorias si faltan en 2009
    if "TIPOSEC" not in df.columns:
        df["TIPOSEC"] = "U"
    if "NV" not in df.columns and "LN" in df.columns and "SV" in df.columns:
        df["NV"] = df["LN"] - df["SV"]
        
    return df

def procesar_estado_2009(ruta_csv: str) -> pd.DataFrame:
    df = pd.read_csv(ruta_csv, low_memory=False)
    df = mapear_columnas_2009(df)

    # Conversión numérica limpia
    df["EDAD"] = pd.to_numeric(df.get("EDAD", -1), errors="coerce").fillna(-1)
    df["LN"] = pd.to_numeric(df.get("LN", 0), errors="coerce").fillna(0)
    df["SV"] = pd.to_numeric(df.get("SV", 0), errors="coerce").fillna(0)
    df["NV"] = pd.to_numeric(df.get("NV", 0), errors="coerce").fillna(0)

    # Identificar género
    es_mujer = df["SEXO"].astype(str).str.upper().isin(["1", "M", "MUJER", "2"])

    # Banderas vectorizadas
    df["ln_joven"] = df["LN"].where(df["EDAD"].between(19, 29), 0)
    df["sv_joven"] = df["SV"].where(df["EDAD"].between(19, 29), 0)
    df["ln_18"] = df["LN"].where(df["EDAD"] == 18, 0)
    df["ln_65"] = df["LN"].where(df["EDAD"] >= 65, 0)
    df["ln_mujeres"] = df["LN"].where(es_mujer, 0)

    # Columnas de agrupación que existan en el DataFrame
    posibles_grupos = ["AELEC", "EDOCVE", "EDONOM", "MPIOCVE", "MPIONOM", "SECCION", "TIPOSEC", "DEF", "DEL"]
    cols_existentes = [c for c in posibles_grupos if c in df.columns]

    sec = df.groupby(cols_existentes, as_index=False).agg({
        "LN": "sum",
        "SV": "sum",
        "NV": "sum",
        "ln_joven": "sum",
        "sv_joven": "sum",
        "ln_18": "sum",
        "ln_65": "sum",
        "ln_mujeres": "sum"
    })

    sec.rename(columns={"LN": "ln_total", "SV": "sv_total", "NV": "nv_total"}, inplace=True)
    
    sec["participacion"] = sec["sv_total"] / sec["ln_total"].replace(0, 1)
    sec["pct_jovenes_19_29"] = sec["ln_joven"] / sec["ln_total"].replace(0, 1)
    sec["part_jovenes_19_29"] = sec["sv_joven"] / sec["ln_joven"].replace(0, 1)
    sec["pct_18"] = sec["ln_18"] / sec["ln_total"].replace(0, 1)
    sec["pct_mayores_65"] = sec["ln_65"] / sec["ln_total"].replace(0, 1)
    sec["ratio_mujeres"] = sec["ln_mujeres"] / sec["ln_total"].replace(0, 1)

    sec.drop(columns=["ln_joven", "sv_joven", "ln_18", "ln_65", "ln_mujeres"], inplace=True)
    return sec

# 3. Procesar y consolidar
t_inicio = time.time()
dfs = []

for i, archivo in enumerate(archivos, start=1):
    nombre = os.path.basename(archivo).split("_")[-1].replace(".csv", "").upper()
    print(f"  [{i:02d}/{len(archivos):02d}] Procesando {nombre}...", end="\r")
    try:
        df_e = procesar_estado_2009(archivo)
        dfs.append(df_e)
    except Exception as e:
        print(f"\n    [AVISO] Error en {archivo}: {e}")

df_nacional_2009 = pd.concat(dfs, ignore_index=True)
df_nacional_2009["AELEC"] = 2009

# Estandarizar columnas de texto y evitar conflictos de tipos en PyArrow
for col in ["EDOCVE", "EDONOM", "MPIOCVE", "MPIONOM", "SECCION", "TIPOSEC", "DEF", "DEL"]:
    if col in df_nacional_2009.columns:
        df_nacional_2009[col] = df_nacional_2009[col].astype(str).str.strip()

# Clave seccional unificada
df_nacional_2009["id_seccion"] = (
    df_nacional_2009["EDOCVE"].str.zfill(2) + "_" + 
    df_nacional_2009["SECCION"].str.zfill(4)
)

os.makedirs(os.path.dirname(SALIDA_2009), exist_ok=True)
df_nacional_2009.to_parquet(SALIDA_2009, index=False)

duracion = time.time() - t_inicio
print(f"\n\n[OK] ¡2009 CONSOLIDADO CON ÉXITO en {duracion:.1f} segundos!")
print(f"Total secciones guardadas: {len(df_nacional_2009):,}")
print(f"Archivo generado: {SALIDA_2009}")