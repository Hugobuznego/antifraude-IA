"""Entrenamiento y evaluacion.

Uso:
    python -m fraude.train --data data/PS_20174392719_1491204439457_log.csv
    python -m fraude.train --synthetic 200000      # sin descargar nada

Decisiones clave (las que hay que saber contar en una entrevista):
1. Solo se modelan TRANSFER y CASH_OUT: en los datos no hay fraude en el resto.
2. Split TEMPORAL (train -> validacion -> test por `step`), no aleatorio. En produccion
   el modelo siempre predice el futuro; un split aleatorio infla las metricas.
3. La metrica principal es PR-AUC, no accuracy ni ROC-AUC: con <1% de fraude, un modelo
   que dice "nunca es fraude" tiene un 99% de accuracy y no sirve para nada.
4. Los umbrales se eligen por COSTE en euros sobre validacion, respetando la capacidad
   del equipo de revision, y se reportan en test.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraude.data import RISKY_TYPES, generate_synthetic, load_paysim
from fraude.features import FEATURES, build_features

# Supuestos de coste (editables, y hay que explicarlos en el README)
REVIEW_COST = 5.0            # coste de que un analista revise una alerta (EUR)
MIN_BLOCK_PRECISION = 0.95   # solo bloqueamos automaticamente si acertamos >= 95%
MAX_REVIEW_RATE = 0.02       # el equipo puede revisar como mucho el 2% de las transacciones


def temporal_split(df: pd.DataFrame, val_q: float = 0.6, test_q: float = 0.8):
    """Corta por cuantiles de `step`: lo antiguo entrena, lo reciente evalua."""
    s1, s2 = df["step"].quantile([val_q, test_q]).astype(int)
    train = df[df["step"] <= s1]
    val = df[(df["step"] > s1) & (df["step"] <= s2)]
    test = df[df["step"] > s2]
    return train, val, test


def expected_cost(y: np.ndarray, amount: np.ndarray, flagged: np.ndarray) -> float:
    """Dinero perdido por fraude no detectado + coste de revisar todas las alertas."""
    missed = (y == 1) & ~flagged
    return float(amount[missed].sum() + REVIEW_COST * flagged.sum())


def choose_review_threshold(y, proba, amount, max_alerts: int | None = None, grid=None) -> float:
    """Umbral que minimiza el coste total en validacion.

    `max_alerts`: capacidad del equipo de analistas en el periodo. Sin este limite, como un
    fraude medio vale mucho mas que revisar una alerta, el optimo es revisar casi todo,
    cosa que ningun banco puede hacer.
    """
    grid = np.unique(np.concatenate([np.logspace(-4, -1, 40), np.linspace(0.1, 0.99, 90)])) if grid is None else grid
    best_t, best_cost = None, np.inf
    for t in grid:
        flagged = proba >= t
        if max_alerts is not None and flagged.sum() > max_alerts:
            continue
        cost = expected_cost(y, amount, flagged)
        if cost < best_cost:
            best_t, best_cost = t, cost
    return float(best_t if best_t is not None else grid[-1])


def choose_block_threshold(y, proba, min_precision=MIN_BLOCK_PRECISION, grid=None) -> float:
    """Umbral mas bajo con precision suficiente para bloquear sin revision humana.
    Si ninguno llega, devuelve 1.01 (nunca se bloquea automaticamente)."""
    grid = np.linspace(0.5, 0.999, 200) if grid is None else grid
    for t in grid:
        pred = proba >= t
        if pred.sum() >= 5 and precision_score(y, pred, zero_division=0) >= min_precision:
            return float(t)
    return 1.01


def evaluate(y, proba, amount, review_t, block_t) -> dict:
    flagged = proba >= review_t
    blocked = proba >= block_t
    fraud_amount = float(amount[y == 1].sum())
    caught_amount = float(amount[(y == 1) & flagged].sum())
    return {
        "n_transactions": int(len(y)),
        "n_fraud": int(y.sum()),
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "recall_at_review": round(float(recall_score(y, flagged, zero_division=0)), 4),
        "precision_at_review": round(float(precision_score(y, flagged, zero_division=0)), 4),
        "n_alerts": int(flagged.sum()),
        "n_auto_blocked": int(blocked.sum()),
        "precision_at_block": round(float(precision_score(y, blocked, zero_division=0)), 4),
        "fraud_amount_total": round(fraud_amount, 2),
        "fraud_amount_caught_pct": round(caught_amount / fraud_amount, 4) if fraud_amount else None,
        "cost_with_model": round(expected_cost(y, amount, flagged), 2),
        "cost_without_model": round(fraud_amount, 2),
    }


def train(
    df: pd.DataFrame,
    out_dir: str | Path = "artifacts",
    source: str = "unknown",
    max_review_rate: float | None = MAX_REVIEW_RATE,
) -> dict:
    """`max_review_rate`: fraccion maxima de transacciones que el equipo puede revisar
    (None = sin limite)."""
    df = df[df["type"].astype(str).isin(RISKY_TYPES)].copy()
    train_df, val_df, test_df = temporal_split(df)
    if train_df["isFraud"].sum() < 10:
        raise ValueError("Muy pocos fraudes en train para entrenar algo con sentido")

    X_tr, y_tr = build_features(train_df), train_df["isFraud"].to_numpy()
    X_va, y_va = build_features(val_df), val_df["isFraud"].to_numpy()
    X_te, y_te = build_features(test_df), test_df["isFraud"].to_numpy()

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, random_state=42,
    )
    model.fit(X_tr, y_tr)

    # Baseline 1: modelo lineal sencillo. Si el boosting no le gana claro, sobra complejidad.
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    baseline.fit(X_tr, y_tr)

    p_va = model.predict_proba(X_va)[:, 1]
    max_alerts = int(max_review_rate * len(val_df)) if max_review_rate else None
    review_t = choose_review_threshold(y_va, p_va, val_df["amount"].to_numpy(), max_alerts)
    block_t = max(choose_block_threshold(y_va, p_va), review_t)

    p_te = model.predict_proba(X_te)[:, 1]
    amount_te = test_df["amount"].to_numpy()
    metrics = evaluate(y_te, p_te, amount_te, review_t, block_t)
    test_days = max((test_df["step"].max() - test_df["step"].min() + 1) / 24, 1)
    metrics["alerts_per_day"] = round(metrics["n_alerts"] / test_days, 1)

    # Baseline 2: la regla que ya traia el sistema (isFlaggedFraud)
    rule = test_df["isFlaggedFraud"].to_numpy() == 1
    metrics["baseline_rule_recall"] = round(float(recall_score(y_te, rule, zero_division=0)), 4)
    metrics["baseline_logreg_pr_auc"] = round(
        float(average_precision_score(y_te, baseline.predict_proba(X_te)[:, 1])), 4
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": source,
        "features": FEATURES,
        "risky_types": RISKY_TYPES,
        "thresholds": {"review": round(review_t, 4), "block": round(block_t, 4)},
        "assumptions": {
            "review_cost_eur": REVIEW_COST,
            "min_block_precision": MIN_BLOCK_PRECISION,
            "max_review_rate": max_review_rate,
        },
        "split_steps": {
            "train": [int(train_df.step.min()), int(train_df.step.max())],
            "val": [int(val_df.step.min()), int(val_df.step.max())],
            "test": [int(test_df.step.min()), int(test_df.step.max())],
        },
        "test_metrics": metrics,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo antifraude")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="Ruta al CSV de PaySim")
    src.add_argument("--synthetic", type=int, metavar="N", help="Usar N filas sinteticas")
    parser.add_argument("--nrows", type=int, default=None, help="Leer solo N filas del CSV")
    parser.add_argument("--max-review-rate", type=float, default=MAX_REVIEW_RATE,
                        help="Fraccion maxima de transacciones a revisar (capacidad del equipo). 0 = sin limite")
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()

    if args.data:
        df, source = load_paysim(args.data, nrows=args.nrows), f"paysim:{Path(args.data).name}"
    else:
        df, source = generate_synthetic(args.synthetic), f"synthetic:{args.synthetic}"

    meta = train(df, args.out, source, max_review_rate=args.max_review_rate or None)
    print(json.dumps({"thresholds": meta["thresholds"], **meta["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
