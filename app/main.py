import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from fraude.scoring import Scorer

MODEL_DIR = os.getenv("MODEL_DIR", "artifacts")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Si no hay modelo entrenado, la API arranca igual pero /score devuelve 503
    try:
        app.state.scorer = Scorer.load(MODEL_DIR)
    except FileNotFoundError:
        app.state.scorer = None
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
    return {"status": "ok", "model_loaded": request.app.state.scorer is not None}


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
