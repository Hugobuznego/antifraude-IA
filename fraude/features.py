"""Variables del modelo.

Cada variable responde a una hipotesis de negocio, no a probar por probar:
- error_orig / error_dest: el saldo antes y despues no cuadra con el importe.
- empties_account: el defraudador suele llevarse todo lo que hay.
- dest_zero_balances: el dinero "desaparece" en destino (cuenta mula que vacia al instante).
- hour: el fraude se concentra a horas raras.
"""

import numpy as np
import pandas as pd

FEATURES = [
    "amount", "log_amount",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "error_orig", "error_dest", "amount_to_balance",
    "empties_account", "dest_zero_balances", "orig_zero_balance",
    "is_transfer", "hour",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    amount = df["amount"].astype(float)
    old_o = df["oldbalanceOrg"].astype(float)
    new_o = df["newbalanceOrig"].astype(float)
    old_d = df["oldbalanceDest"].astype(float)
    new_d = df["newbalanceDest"].astype(float)

    out = pd.DataFrame(index=df.index)
    out["amount"] = amount
    out["log_amount"] = np.log1p(amount)
    out["oldbalanceOrg"] = old_o
    out["newbalanceOrig"] = new_o
    out["oldbalanceDest"] = old_d
    out["newbalanceDest"] = new_d
    out["error_orig"] = new_o + amount - old_o
    out["error_dest"] = old_d + amount - new_d
    out["amount_to_balance"] = amount / (old_o + 1.0)
    out["empties_account"] = ((old_o > 0) & (amount >= old_o) & (new_o == 0)).astype(int)
    out["dest_zero_balances"] = ((old_d == 0) & (new_d == 0) & (amount > 0)).astype(int)
    out["orig_zero_balance"] = (old_o == 0).astype(int)
    out["is_transfer"] = (df["type"].astype(str) == "TRANSFER").astype(int)
    out["hour"] = df["step"].astype(int) % 24
    return out[FEATURES]
