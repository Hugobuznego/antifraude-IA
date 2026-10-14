import pandas as pd

from fraude.data import COLUMNS, RISKY_TYPES, generate_synthetic, load_paysim
from fraude.features import FEATURES, build_features


def test_esquema_paysim(synthetic_df):
    assert list(synthetic_df.columns) == COLUMNS
    assert synthetic_df["step"].between(1, 744).all()
    assert (synthetic_df["amount"] > 0).all()


def test_fraude_solo_en_tipos_de_riesgo(synthetic_df):
    no_risky = synthetic_df[~synthetic_df["type"].isin(RISKY_TYPES)]
    assert no_risky["isFraud"].sum() == 0
    assert synthetic_df["isFraud"].sum() > 0


def test_generador_reproducible():
    a = generate_synthetic(1000, seed=1)
    b = generate_synthetic(1000, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_load_paysim_desde_csv(tmp_path):
    path = tmp_path / "mini.csv"
    generate_synthetic(500).to_csv(path, index=False)
    df = load_paysim(path)
    assert len(df) == 500 and list(df.columns) == COLUMNS


def test_load_paysim_columnas_faltantes(tmp_path):
    path = tmp_path / "malo.csv"
    pd.DataFrame({"step": [1], "amount": [10.0]}).to_csv(path, index=False)
    try:
        load_paysim(path)
        raise AssertionError("Deberia fallar")
    except ValueError as e:
        assert "Faltan columnas" in str(e)


def test_features_sin_nulos(synthetic_df):
    X = build_features(synthetic_df)
    assert list(X.columns) == FEATURES
    assert not X.isna().any().any()


def test_features_caso_tipico_fraude():
    tx = pd.DataFrame([{
        "step": 27, "type": "TRANSFER", "amount": 1000.0,
        "oldbalanceOrg": 1000.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
    }])
    f = build_features(tx).iloc[0]
    assert f["empties_account"] == 1
    assert f["dest_zero_balances"] == 1
    assert f["error_orig"] == 0
    assert f["error_dest"] == 1000
    assert f["hour"] == 3
    assert f["is_transfer"] == 1


def test_features_transaccion_normal():
    tx = pd.DataFrame([{
        "step": 14, "type": "CASH_OUT", "amount": 200.0,
        "oldbalanceOrg": 5000.0, "newbalanceOrig": 4800.0,
        "oldbalanceDest": 100.0, "newbalanceDest": 300.0,
    }])
    f = build_features(tx).iloc[0]
    assert f["empties_account"] == 0
    assert f["dest_zero_balances"] == 0
    assert f["error_orig"] == 0 and f["error_dest"] == 0
