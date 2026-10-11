## 🚀 Despliegue Rápido (Reproducibilidad Garantizada)

### Opción 1: Con Docker (Recomendada)
Requisitos: Tener instalado Docker Desktop.

```bash
# 1. Clonar el repositorio
git clone [https://github.com/Aldini057/voto-vigia.git](https://github.com/Aldini057/voto-vigia.git)
cd voto-vigia

# 2. Levantar el contenedor
docker compose up --build

Abrir en el navegador: http://localhost:8000

Opción 2: En Entorno Local (Python)
Requisitos: Python 3.10 o superior.

# 1. Crear y activar entorno virtual
python -m venv venv

# En Windows:
venv\Scripts\activate
# En Linux / Mac:
source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar el servidor
uvicorn app.main:app --reload
Abrir en el navegador: http://localhost:8000

---

### Verificación final en GitHub
Una vez hecho el push, entra al repositorio [Aldini057/voto-vigia](https://github.com/Aldini057/voto-vigia) y comprueba:
1. Que la carpeta **`app/static/geojson/`** exista y contenga `mexico_estados.geojson`.
2. Que tus 3 plantillas en **`app/templates/`** muestren el commit reciente.
3. Que no aparezca ningún archivo `.env` con contraseñas.

Con esta estructura, cualquier computadora que descargue el repositorio levantará el sistema con un solo comando y sin fallos.
