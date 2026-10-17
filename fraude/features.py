"""Variables del modelo.

Hay DOS conjuntos de variables, y la diferencia entre ellos es la leccion principal del proyecto:

- "realista": solo informacion disponible ANTES de autorizar la operacion
  (importe, saldos previos, tipo, hora). Es lo que tendria un banco en tiempo real.
- "completo": anade los saldos DESPUES de la operacion (newbalance*) y los descuadres
  derivados (error_*). En PaySim dan un modelo casi perfecto, pero es una fuga de
  informacion: cuando hay que decidir, esos saldos todavia no existen.

Hipotesis de negocio detras de cada variable:
- takes_all_balance: el defraudador suele llevarse todo lo que hay en la cuenta.
- dest_new_account: el dinero va a una cuenta sin saldo previo (posible cuenta mula).
- amount_to_balance: importe desproporcionado respecto al saldo.
- hour: el fraude no sigue el horario normal de la gente.
"""

import numpy as np
import pandas as pd

FEATURES_REALISTA = [
    "amount", "log_amount",
    "oldbalanceOrg", "oldbalanceDest",
    "amount_to_balance", "takes_all_balance",
    "orig_zero_balance", "dest_new_account",
    "is_transfer", "hour",
]

# Solo se pueden calcular cuando la operacion ya se ha ejecutado
POST_TRANSACTION = [
    "newbalanceOrig", "newbalanceDest",
    "error_orig", "error_dest",
    "empties_account", "dest_zero_balances",
]

FEATURE_SETS = {
    "realista": FEATURES_REALISTA,
    "completo": FEATURES_REALISTA + POST_TRANSACTION,
}

DEFAULT_FEATURE_SET = "realista"
ALL_FEATURES = FEATURE_SETS["completo"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todas las variables. Cada modelo luego elige su subconjunto."""
    amount = df["amount"].astype(float)
    old_o = df["oldbalanceOrg"].astype(float)
    new_o = df["newbalanceOrig"].astype(float)
    old_d = df["oldbalanceDest"].astype(float)
    new_d = df["newbalanceDest"].astype(float)

    out = pd.DataFrame(index=df.index)
    # Antes de la operacion
    out["amount"] = amount
    out["log_amount"] = np.log1p(amount)
    out["oldbalanceOrg"] = old_o
    out["oldbalanceDest"] = old_d
    out["amount_to_balance"] = amount / (old_o + 1.0)
    out["takes_all_balance"] = ((old_o > 0) & (amount >= old_o)).astype(int)
    out["orig_zero_balance"] = (old_o == 0).astype(int)
    out["dest_new_account"] = (old_d == 0).astype(int)
    out["is_transfer"] = (df["type"].astype(str) == "TRANSFER").astype(int)
    out["hour"] = df["step"].astype(int) % 24
    # Despues de la operacion
    out["newbalanceOrig"] = new_o
    out["newbalanceDest"] = new_d
    out["error_orig"] = new_o + amount - old_o
    out["error_dest"] = old_d + amount - new_d
    out["empties_account"] = ((old_o > 0) & (amount >= old_o) & (new_o == 0)).astype(int)
    out["dest_zero_balances"] = ((old_d == 0) & (new_d == 0) & (amount > 0)).astype(int)
    return out[ALL_FEATURES]
