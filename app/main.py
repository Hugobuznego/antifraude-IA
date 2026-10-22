import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from agente.graph import build_agent, revisar
from agente.llm import llm_desde_entorno
from fraude.historial import Historial
from fraude.scoring import Scorer


def cargar_env(path: str = ".env") -> None:
    """Lee un .env sencillo (CLAVE=valor). Asi la clave de Gemini no va en el codigo."""
    f = Path(path)
    if not f.exists():
        return
    for linea in f.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            clave, valor = linea.split("=", 1)
            os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


cargar_env()
MODEL_DIR = os.getenv("MODEL_DIR", "artifacts")
HISTORY_DATA = os.getenv("HISTORY_DATA", "data/PS_20174392719_1491204439457_log.csv")


def montar_agente(app: FastAPI) -> None:
    if app.state.scorer is None:
        app.state.agente = None
        return
    app.state.agente = build_agent(app.state.scorer, app.state.historial, app.state.llm)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Si no hay modelo entrenado, la API arranca igual pero /score devuelve 503
    try:
        app.state.scorer = Scorer.load(MODEL_DIR)
    except FileNotFoundError:
        app.state.scorer = None
    # El historial es opcional: si no esta el CSV, el agente lo indica en el informe
    if Path(HISTORY_DATA).exists():
        print(f"Cargando historial desde {HISTORY_DATA} (puede tardar un poco)...")
        app.state.historial = Historial.desde_csv(HISTORY_DATA)
    else:
        app.state.historial = Historial(None)
    app.state.llm = llm_desde_entorno()
    montar_agente(app)
    yield


app = FastAPI(
    title="API antifraude",
    description="Puntua transacciones y decide si aprobar, revisar o bloquear.",
    version="0.1.0",
    lifespan=lifespan,
)


class Transaction(BaseModel):
    step: int = Field(..., ge=1, description="Hora de la simulacion (1 = primera hora)")
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(..., gt=0)
    nameOrig: str
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    nameDest: str
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "step": 3, "type": "TRANSFER", "amount": 181000.0,
                "nameOrig": "C1305486145", "oldbalanceOrg": 181000.0, "newbalanceOrig": 0.0,
                "nameDest": "C553264065", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
            }
        }
    }


class ScoreResult(BaseModel):
    fraud_probability: float
    decision: Literal["APPROVE", "REVIEW", "BLOCK"]
    reasons: list[str]


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=1000)


def get_scorer(request: Request) -> Scorer:
    scorer = request.app.state.scorer
    if scorer is None:
        raise HTTPException(503, "No hay modelo cargado. Ejecuta antes: python -m fraude.train")
    return scorer


@app.get("/health")
def health(request: Request):
    st = request.app.state
    return {
        "status": "ok",
        "model_loaded": st.scorer is not None,
        "history_loaded": st.historial.disponible,
        "llm": st.llm.nombre if st.llm else None,
    }


@app.get("/model/info")
def model_info(request: Request):
    meta = get_scorer(request).metadata
    keys = ("trained_at", "data_source", "feature_set", "thresholds", "assumptions", "test_metrics")
    return {k: meta.get(k) for k in keys}


@app.post("/score", response_model=ScoreResult)
def score(tx: Transaction, request: Request):
    return get_scorer(request).score(tx.model_dump())


@app.post("/score/batch", response_model=list[ScoreResult])
def score_batch(req: BatchRequest, request: Request):
    return get_scorer(request).score_many([t.model_dump() for t in req.transactions])


class AgentReview(BaseModel):
    recomendacion_final: Literal["APPROVE", "REVIEW", "BLOCK"]
    decision_minima_politica: Literal["APPROVE", "REVIEW", "BLOCK"]
    puntuacion: ScoreResult
    historial_destino: dict
    reglas: list[str]
    informe: dict
    fuente_informe: str
    notas: list[str]


@app.post("/agent/review", response_model=AgentReview)
def agent_review(tx: Transaction, request: Request):
    """El agente revisa la operacion y redacta un informe para el analista."""
    get_scorer(request)
    return revisar(request.app.state.agente, tx.model_dump())
