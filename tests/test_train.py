import json

import numpy as np
import pandas as pd
import pytest

from fraude.train import (
    choose_block_threshold,
    choose_review_threshold,
    expected_cost,
    temporal_split,
)


def test_split_temporal_sin_solape():
    df = pd.DataFrame({"step": np.arange(1, 1001)})
    tr, va, te = temporal_split(df)
    assert tr.step.max() < va.step.min()
    assert va.step.max() < te.step.min()
    assert len(tr) + len(va) + len(te) == len(df)


def test_coste_esperado():
    y = np.array([1, 1, 0, 0])
    amount = np.array([1000.0, 500.0, 50.0, 50.0])
    flagged = np.array([True, False, True, False])
    # se escapa el fraude de 500 + 2 alertas revisadas a 5 EUR
    assert expected_cost(y, amount, flagged) == 510.0


def test_umbral_revision_respeta_capacidad():
    rng = np.random.default_rng(0)
    y = (rng.random(10_000) < 0.01).astype(int)
    proba = np.clip(y * 0.5 + rng.random(10_000) * 0.5, 0, 1)
    amount = np.full(10_000, 100_000.0)
    t_libre = choose_review_threshold(y, proba, amount)
    t_limitado = choose_review_threshold(y, proba, amount, max_alerts=200)
    assert t_limitado >= t_libre
    assert (proba >= t_limitado).sum() <= 200


def test_umbral_bloqueo_exige_precision():
    y = np.array([0] * 90 + [1] * 10)
    proba = np.concatenate([np.linspace(0, 0.6, 90), np.linspace(0.9, 1.0, 10)])
    t = choose_block_threshold(y, proba, min_precision=0.95)
    pred = proba >= t
    assert (y[pred] == 1).mean() >= 0.95


def test_umbral_bloqueo_imposible_devuelve_nunca():
    y = np.array([0, 1] * 50)
    proba = np.full(100, 0.9)  # el modelo no separa nada
    assert choose_block_threshold(y, proba) > 1


def test_entrenamiento_genera_artefactos(trained_dir):
    assert (trained_dir / "model.joblib").exists()
    meta = json.loads((trained_dir / "metadata.json").read_text())
    t = meta["thresholds"]
    assert 0 < t["review"] <= t["block"]
    assert meta["split_steps"]["train"][1] < meta["split_steps"]["test"][0]
    assert meta["feature_set"] == "realista"
    assert "newbalanceOrig" not in meta["features"]


def test_modelo_supera_baselines(trained_dir):
    m = json.loads((trained_dir / "metadata.json").read_text())["test_metrics"]
    tasa_fraude = m["n_fraud"] / m["n_transactions"]
    # PR-AUC muy por encima del azar (que seria la tasa de fraude)
    assert m["pr_auc"] > 10 * tasa_fraude
    assert m["pr_auc"] >= m["baseline_logreg_pr_auc"]
    assert m["recall_at_review"] > m["baseline_rule_recall"]
    assert m["cost_with_model"] < m["cost_without_model"]


def test_entrena_set_completo(tmp_path, synthetic_df):
    from fraude.train import train
    meta = train(synthetic_df, tmp_path, feature_set="completo")
    assert meta["feature_set"] == "completo"
    assert "newbalanceDest" in meta["features"]


def test_set_desconocido_falla(tmp_path, synthetic_df):
    from fraude.train import train
    with pytest.raises(ValueError):
        train(synthetic_df, tmp_path, feature_set="inventado")


def test_pocos_fraudes_falla(tmp_path):
    from fraude.data import generate_synthetic
    from fraude.train import train
    df = generate_synthetic(2000, fraud_rate=0.0)
    with pytest.raises(ValueError):
        train(df, tmp_path / "artifacts")
