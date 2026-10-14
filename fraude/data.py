"""Carga de datos.

Dos fuentes con el MISMO esquema:
- PaySim real (CSV de Kaggle), para el analisis de verdad.
- Generador sintetico que imita PaySim, para tests y CI (GitHub no puede descargar de Kaggle).
"""

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "step", "type", "amount",
    "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud",
]

TX_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# En PaySim el fraude solo aparece en estos dos tipos.
RISKY_TYPES = ["TRANSFER", "CASH_OUT"]

DTYPES = {
    "step": "int32", "type": "category", "amount": "float64",
    "nameOrig": "string", "oldbalanceOrg": "float64", "newbalanceOrig": "float64",
    "nameDest": "string", "oldbalanceDest": "float64", "newbalanceDest": "float64",
    "isFraud": "int8", "isFlaggedFraud": "int8",
}


def load_paysim(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Lee el CSV de PaySim con tipos ajustados (ahorra mucha memoria: son 6,3M filas)."""
    df = pd.read_csv(path, dtype=DTYPES, nrows=nrows)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {sorted(missing)}")
    return df[COLUMNS]


def generate_synthetic(n: int = 50_000, fraud_rate: float = 0.01, seed: int = 42) -> pd.DataFrame:
    """Genera transacciones con el esquema de PaySim.

    Incluye ruido a proposito (fraudes que no vacian la cuenta, legitimas que si,
    saldos sin actualizar) para que el modelo no lo tenga trivial.
    `fraud_rate` es la tasa de fraude dentro de TRANSFER y CASH_OUT.
    """
    rng = np.random.default_rng(seed)

    step = rng.integers(1, 745, n)  # 744 horas = 31 dias, como PaySim
    tx_type = rng.choice(TX_TYPES, n, p=[0.22, 0.35, 0.01, 0.34, 0.08])
    risky = np.isin(tx_type, RISKY_TYPES)
    is_fraud = risky & (rng.random(n) < fraud_rate)

    old_orig = np.round(rng.lognormal(10, 1.6, n), 2)
    old_orig[rng.random(n) < 0.3] = 0.0  # muchas cuentas origen a 0, como en PaySim
    amount = np.round(rng.lognormal(10.5, 1.3, n), 2)

    # Fraude: casi siempre vacia la cuenta origen
    empties = is_fraud & (rng.random(n) < 0.85)
    old_orig[is_fraud & (old_orig == 0)] = np.round(rng.lognormal(12, 1.2, (is_fraud & (old_orig == 0)).sum()), 2)
    amount[empties] = old_orig[empties]
    # Ruido: algunas legitimas de riesgo tambien vacian la cuenta
    legit_empties = risky & ~is_fraud & (rng.random(n) < 0.05)
    amount[legit_empties] = np.maximum(old_orig[legit_empties], 1.0)

    amount = np.minimum(amount, 10_000_000)  # tope por transaccion en PaySim

    new_orig = np.maximum(old_orig - amount, 0.0)
    new_orig[tx_type == "CASH_IN"] = old_orig[tx_type == "CASH_IN"] + amount[tx_type == "CASH_IN"]
    not_updated = ~is_fraud & (rng.random(n) < 0.25)
    new_orig[not_updated] = old_orig[not_updated]

    old_dest = np.round(rng.lognormal(11, 2, n), 2)
    old_dest[rng.random(n) < 0.35] = 0.0
    new_dest = old_dest + amount
    # Anomalia tipica de fraude: la cuenta destino no refleja el dinero recibido
    dest_anomaly = (is_fraud & (rng.random(n) < 0.6)) | (~is_fraud & risky & (rng.random(n) < 0.08))
    old_dest[dest_anomaly] = 0.0
    new_dest[dest_anomaly] = 0.0

    is_payment = tx_type == "PAYMENT"
    old_dest[is_payment] = 0.0  # en PaySim los comercios no tienen saldo
    new_dest[is_payment] = 0.0

    name_orig = np.char.add("C", rng.integers(1e8, 2e9, n).astype(str))
    name_dest = np.where(is_payment, "M", "C")
    name_dest = np.char.add(name_dest, rng.integers(1e8, 2e9, n).astype(str))

    flagged = (tx_type == "TRANSFER") & (amount > 200_000) & is_fraud & (rng.random(n) < 0.05)

    df = pd.DataFrame({
        "step": step, "type": tx_type, "amount": np.round(amount, 2),
        "nameOrig": name_orig, "oldbalanceOrg": old_orig, "newbalanceOrig": np.round(new_orig, 2),
        "nameDest": name_dest, "oldbalanceDest": old_dest, "newbalanceDest": np.round(new_dest, 2),
        "isFraud": is_fraud.astype("int8"), "isFlaggedFraud": flagged.astype("int8"),
    })
    return df.sort_values("step", kind="stable").reset_index(drop=True).astype(DTYPES)
