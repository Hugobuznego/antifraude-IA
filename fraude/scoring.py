"""Puntuacion de transacciones con el modelo entrenado.

Devuelve probabilidad, decision (APPROVE / REVIEW / BLOCK) y motivos legibles.
Los motivos son reglas sencillas sobre las variables; seran una de las fuentes
del agente que ayude al analista en la siguiente fase.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from fraude.features import build_features

APPROVE, REVIEW, BLOCK = "APPROVE", "REVIEW", "BLOCK"


def decide(proba: float, review_t: float, block_t: float) -> str:
    if proba >= block_t:
        return BLOCK
    if proba >= review_t:
        return REVIEW
    return APPROVE


def explain(feat: pd.Series) -> list[str]:
    """Motivos legibles, solo con informacion disponible antes de autorizar."""
    reasons = []
    if feat["takes_all_balance"]:
        reasons.append("La operacion se lleva todo el saldo de la cuenta de origen")
    elif feat["amount_to_balance"] >= 0.8 and not feat["orig_zero_balance"]:
        reasons.append("El importe es casi todo el saldo disponible")
    if feat["dest_new_account"] and feat["is_transfer"]:
        reasons.append("Transferencia a una cuenta destino sin saldo previo")
    if feat["amount"] >= 200_000:
        reasons.append("Importe alto (>= 200.000)")
    if feat["hour"] < 6:
        reasons.append("Operacion de madrugada")
    return reasons


@dataclass
class Scorer:
    model: object
    metadata: dict

    @classmethod
    def load(cls, model_dir: str | Path) -> "Scorer":
        model_dir = Path(model_dir)
        model = joblib.load(model_dir / "model.joblib")
        metadata = json.loads((model_dir / "metadata.json").read_text())
        return cls(model, metadata)

    @property
    def thresholds(self) -> dict:
        return self.metadata["thresholds"]

    def score_many(self, transactions: list[dict]) -> list[dict]:
        df = pd.DataFrame(transactions)
        feats = build_features(df)
        risky = df["type"].isin(self.metadata["risky_types"]).to_numpy()
        proba = self.model.predict_proba(feats[self.metadata["features"]])[:, 1]

        results = []
        for i in range(len(df)):
            if not risky[i]:
                # Sin fraude historico en este tipo: no se puntua con el modelo
                results.append({
                    "fraud_probability": 0.0,
                    "decision": APPROVE,
                    "reasons": [f"Tipo {df['type'].iloc[i]} sin fraude historico"],
                })
                continue
            p = float(proba[i])
            results.append({
                "fraud_probability": round(p, 4),
                "decision": decide(p, self.thresholds["review"], self.thresholds["block"]),
                "reasons": explain(feats.iloc[i]),
            })
        return results

    def score(self, transaction: dict) -> dict:
        return self.score_many([transaction])[0]
