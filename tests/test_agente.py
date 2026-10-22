import json

import pytest
from fastapi.testclient import TestClient

from agente.graph import build_agent, revisar
from app.main import app, montar_agente
from fraude.historial import Historial
from fraude.scoring import Scorer

SOSPECHOSA = {
    "step": 400, "type": "TRANSFER", "amount": 250000.0,
    "nameOrig": "C1", "oldbalanceOrg": 250000.0, "newbalanceOrig": 0.0,
    "nameDest": "C_MULA", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
}
LIMPIA = {
    "step": 14, "type": "CASH_OUT", "amount": 150.0,
    "nameOrig": "C3", "oldbalanceOrg": 8000.0, "newbalanceOrig": 7850.0,
    "nameDest": "C4", "oldbalanceDest": 1200.0, "newbalanceDest": 1350.0,
}


class FakeLLM:
    """LLM de mentira: devuelve lo que le digamos y guarda los prompts."""
    nombre = "fake"

    def __init__(self, respuesta: str | Exception):
        self.respuesta = respuesta
        self.prompts = []

    def generar(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.respuesta, Exception):
            raise self.respuesta
        return self.respuesta


def informe_json(recomendacion="REVIEW"):
    return json.dumps({
        "resumen": "Transferencia que vacia la cuenta hacia una cuenta con fraude previo.",
        "senales": ["Vacia la cuenta"],
        "que_revisar": ["Llamar al titular"],
        "recomendacion": recomendacion,
        "justificacion": "Patron tipico de toma de control de cuenta.",
    })


@pytest.fixture(scope="module")
def scorer(trained_dir):
    return Scorer.load(trained_dir)


@pytest.fixture
def historial():
    import pandas as pd
    df = pd.DataFrame([
        {"step": 10, "type": "TRANSFER", "amount": 5000.0, "nameDest": "C_MULA", "isFraud": 1},
        {"step": 20, "type": "CASH_OUT", "amount": 7000.0, "nameDest": "C_MULA", "isFraud": 1},
        {"step": 500, "type": "TRANSFER", "amount": 1.0, "nameDest": "C_MULA", "isFraud": 1},  # futuro
        {"step": 5, "type": "PAYMENT", "amount": 10.0, "nameDest": "C_MULA", "isFraud": 0},     # tipo sin riesgo
    ])
    return Historial.desde_dataframe(df)


def test_historial_solo_mira_el_pasado(historial):
    h = historial.cuenta_destino("C_MULA", antes_de_step=400)
    assert h == {"disponible": True, "operaciones_recibidas": 2,
                 "importe_recibido": 12000.0, "fraudes_confirmados": 2}


def test_historial_no_disponible():
    assert Historial(None).cuenta_destino("X", 1) == {"disponible": False}


def test_flujo_completo_con_llm(scorer, historial):
    llm = FakeLLM(informe_json("BLOCK"))
    r = revisar(build_agent(scorer, historial, llm), SOSPECHOSA)
    assert r["fuente_informe"] == "llm:fake"
    assert r["recomendacion_final"] == "BLOCK"
    assert any("fraude confirmado" in x for x in r["reglas"])
    assert any("centimo" in x for x in r["reglas"])
    assert any("cuatro ojos" in x for x in r["reglas"])
    # el prompt lleva los datos del caso
    assert "C_MULA" in llm.prompts[0] and "fraudes_confirmados" in llm.prompts[0]


def test_guardarrail_corrige_al_llm(scorer, historial):
    """Aunque el LLM diga APPROVE, si la politica exige BLOCK, gana la politica."""
    r = revisar(build_agent(scorer, historial, FakeLLM(informe_json("APPROVE"))), SOSPECHOSA)
    assert r["informe"]["recomendacion"] == "APPROVE"
    assert r["recomendacion_final"] == "BLOCK"
    assert any("corregido" in n for n in r["notas"])


def test_operacion_limpia_no_llama_al_llm(scorer, historial):
    llm = FakeLLM(informe_json())
    r = revisar(build_agent(scorer, historial, llm), LIMPIA)
    assert r["recomendacion_final"] == "APPROVE"
    assert r["fuente_informe"] == "sin_llm"
    assert llm.prompts == []


def test_sin_clave_usa_plantilla(scorer, historial):
    r = revisar(build_agent(scorer, historial, None), SOSPECHOSA)
    assert r["fuente_informe"] == "plantilla"
    assert r["recomendacion_final"] == "BLOCK"


@pytest.mark.parametrize("respuesta", [
    "esto no es json",
    json.dumps({"resumen": "falta todo lo demas"}),
    json.dumps({**json.loads(informe_json()), "recomendacion": "QUIZAS"}),
    ConnectionError("sin red"),
])
def test_llm_falla_o_responde_mal(scorer, historial, respuesta):
    r = revisar(build_agent(scorer, historial, FakeLLM(respuesta)), SOSPECHOSA)
    assert r["fuente_informe"] == "plantilla"
    assert r["recomendacion_final"] in ("REVIEW", "BLOCK")
    assert r["notas"]


def test_llm_con_bloque_markdown(scorer, historial):
    r = revisar(build_agent(scorer, historial, FakeLLM("```json\n" + informe_json() + "\n```")), SOSPECHOSA)
    assert r["fuente_informe"] == "llm:fake"


def test_endpoint_agent_review(trained_dir, historial):
    with TestClient(app) as c:
        app.state.scorer = Scorer.load(trained_dir)
        app.state.historial = historial
        app.state.llm = FakeLLM(informe_json("REVIEW"))
        montar_agente(app)
        r = c.post("/agent/review", json=SOSPECHOSA)
        assert r.status_code == 200
        body = r.json()
        assert body["recomendacion_final"] == "BLOCK"
        assert body["informe"]["resumen"]
        h = c.get("/health").json()
        assert h["llm"] == "fake" and h["history_loaded"] is True


def test_endpoint_sin_modelo():
    with TestClient(app) as c:
        app.state.scorer = None
        montar_agente(app)
        assert c.post("/agent/review", json=LIMPIA).status_code == 503
