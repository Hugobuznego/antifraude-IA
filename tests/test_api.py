import pytest
from fastapi.testclient import TestClient

from app.main import app
from fraude.scoring import APPROVE, BLOCK, REVIEW, Scorer, decide

FRAUD_LIKE = {
    "step": 3, "type": "TRANSFER", "amount": 181000.0,
    "nameOrig": "C1", "oldbalanceOrg": 181000.0, "newbalanceOrig": 0.0,
    "nameDest": "C2", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
}
NORMAL = {
    "step": 14, "type": "CASH_OUT", "amount": 150.0,
    "nameOrig": "C3", "oldbalanceOrg": 8000.0, "newbalanceOrig": 7850.0,
    "nameDest": "C4", "oldbalanceDest": 1200.0, "newbalanceDest": 1350.0,
}
PAYMENT = {**NORMAL, "type": "PAYMENT", "nameDest": "M5", "oldbalanceDest": 0.0, "newbalanceDest": 0.0}


@pytest.fixture
def client(trained_dir):
    with TestClient(app) as c:
        app.state.scorer = Scorer.load(trained_dir)
        yield c


@pytest.fixture
def client_sin_modelo():
    with TestClient(app) as c:
        app.state.scorer = None
        yield c


def test_decide():
    assert decide(0.1, review_t=0.3, block_t=0.9) == APPROVE
    assert decide(0.5, review_t=0.3, block_t=0.9) == REVIEW
    assert decide(0.95, review_t=0.3, block_t=0.9) == BLOCK


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    r = client.get("/model/info").json()
    assert "thresholds" in r and "test_metrics" in r


def test_fraude_puntua_mas_que_normal(client):
    p_fraud = client.post("/score", json=FRAUD_LIKE).json()
    p_norm = client.post("/score", json=NORMAL).json()
    assert p_fraud["fraud_probability"] > p_norm["fraud_probability"]
    assert p_fraud["decision"] in (REVIEW, BLOCK)
    assert p_norm["decision"] == APPROVE
    assert any("todo el saldo" in r for r in p_fraud["reasons"])


def test_tipo_sin_riesgo_se_aprueba(client):
    r = client.post("/score", json=PAYMENT).json()
    assert r["decision"] == APPROVE and r["fraud_probability"] == 0.0


def test_batch(client):
    r = client.post("/score/batch", json={"transactions": [FRAUD_LIKE, NORMAL, PAYMENT]})
    assert r.status_code == 200
    assert len(r.json()) == 3


@pytest.mark.parametrize("cambio", [
    {"amount": -5},
    {"amount": 0},
    {"type": "BITCOIN"},
    {"oldbalanceOrg": -1},
    {"step": 0},
])
def test_validacion(client, cambio):
    assert client.post("/score", json={**NORMAL, **cambio}).status_code == 422


def test_batch_vacio(client):
    assert client.post("/score/batch", json={"transactions": []}).status_code == 422


def test_sin_modelo_devuelve_503(client_sin_modelo):
    assert client_sin_modelo.get("/health").json()["model_loaded"] is False
    assert client_sin_modelo.post("/score", json=NORMAL).status_code == 503
