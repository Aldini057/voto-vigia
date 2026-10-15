import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Pruebas unitarias para el endpoint GET /api/health."""

    def test_health_check_success(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "online"
        assert "secciones" in data
        assert isinstance(data["secciones"], int)
        assert data["secciones"] > 0


class TestStatesEndpoint:
    """Pruebas unitarias para el endpoint GET /api/states."""

    def test_get_states_success(self):
        response = client.get("/api/states")
        assert response.status_code == 200
        states = response.json()
        assert isinstance(states, list)
        assert len(states) > 0
        assert "JALISCO" in states
        # Verificar que la lista esté ordenada alfabéticamente
        assert states == sorted(states)


class TestSimulateEndpoint:
    """Pruebas unitarias para el endpoint POST /api/simulate."""

    def test_simulate_nacional_default(self):
        payload = {
            "delta_participacion_previa": 0.0,
            "delta_pct_jovenes": 0.0,
            "es_presidencial": 1,
            "entidad": None,
            "estrategia_focalizada": False,
        }
        response = client.post("/api/simulate", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["entidad_analizada"] == "NACIONAL"
        assert data["zoom"] == 5
        assert isinstance(data["total_secciones"], int)
        assert data["total_secciones"] > 0

        # Verificación de estructura de KPIs y métricas clave
        for key in ["criticos_base", "criticos_simulado", "casillas_rescatadas", "reduccion_riesgo_pct", "kpis"]:
            assert key in data

        kpis = data["kpis"]
        assert "votos_recuperados" in kpis
        assert "focos_rojos_iniciales" in kpis
        assert "tasa_mitigacion_pct" in kpis
        assert "participacion_simulada_pct" in kpis

        # Verificación de puntos geocodificados y estados
        assert isinstance(data["puntos_criticos"], list)
        assert isinstance(data["puntos_medios"], list)
        assert isinstance(data["puntos_bajos"], list)
        assert isinstance(data["estados_metricas"], dict)
        assert len(data["estados_metricas"]) > 0

    def test_simulate_nacional_explicit_string(self):
        payload = {
            "delta_participacion_previa": 0.05,
            "delta_pct_jovenes": 0.02,
            "es_presidencial": 1,
            "entidad": "TODO EL PAÍS",
        }
        response = client.post("/api/simulate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["entidad_analizada"] == "TODO EL PAÍS"
        assert data["zoom"] == 5
        assert data["total_secciones"] > 0

    def test_simulate_estatal_jalisco(self):
        payload = {
            "delta_participacion_previa": 0.10,
            "delta_pct_jovenes": 0.05,
            "es_presidencial": 1,
            "entidad": "JALISCO",
            "estrategia_focalizada": True,
        }
        response = client.post("/api/simulate", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["entidad_analizada"] == "JALISCO"
        assert data["zoom"] == 8
        assert data["centroide"] == [20.6597, -103.3496]
        assert data["total_secciones"] > 0

        # Validar top municipios en simulación focalizada de Jalisco
        assert isinstance(data["top_municipios"], list)
        if len(data["top_municipios"]) > 0:
            municipio_item = data["top_municipios"][0]
            assert "municipio" in municipio_item
            assert "votos_ganados" in municipio_item
            assert "focos_apagados" in municipio_item

    def test_simulate_estado_inexistente_returns_404(self):
        payload = {
            "delta_participacion_previa": 0.0,
            "delta_pct_jovenes": 0.0,
            "es_presidencial": 1,
            "entidad": "ESTADO_INEXISTENTE_XYZ_999",
        }
        response = client.post("/api/simulate", json=payload)
        assert response.status_code == 404
        assert response.json()["detail"] == "No se encontraron casillas."

    def test_simulate_invalid_payload_validation_error(self):
        # delta_participacion_previa excede el rango permitido [-0.30, 0.30]
        payload = {
            "delta_participacion_previa": 0.85,
            "delta_pct_jovenes": 0.0,
            "es_presidencial": 1,
        }
        response = client.post("/api/simulate", json=payload)
        assert response.status_code == 422

