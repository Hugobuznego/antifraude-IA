"""Historial de cuentas destino.

En fraude, la cuenta que RECIBE el dinero dice mucho: las cuentas mula reciben
de muchas victimas distintas. Esto es lo que consultaria un analista.

Nota: en un banco real las etiquetas de fraude llegan con retraso (dias o semanas,
cuando el cliente reclama). Aqui usamos las etiquetas de operaciones ANTERIORES a la
que se revisa, que es la aproximacion razonable.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fraude.data import RISKY_TYPES


@dataclass
class Historial:
    df: pd.DataFrame | None  # columnas: step, nameDest, amount, isFraud

    @classmethod
    def desde_csv(cls, path: str | Path) -> "Historial":
        df = pd.read_csv(
            path,
            usecols=["step", "type", "amount", "nameDest", "isFraud"],
            dtype={"step": "int32", "type": "category", "amount": "float64",
                   "nameDest": "string", "isFraud": "int8"},
        )
        df = df[df["type"].isin(RISKY_TYPES)].drop(columns="type").reset_index(drop=True)
        return cls(df)

    @classmethod
    def desde_dataframe(cls, df: pd.DataFrame) -> "Historial":
        cols = ["step", "nameDest", "amount", "isFraud"]
        return cls(df[df["type"].astype(str).isin(RISKY_TYPES)][cols].reset_index(drop=True))

    @property
    def disponible(self) -> bool:
        return self.df is not None

    def cuenta_destino(self, nombre: str, antes_de_step: int) -> dict:
        if not self.disponible:
            return {"disponible": False}
        previas = self.df[(self.df["nameDest"] == nombre) & (self.df["step"] < antes_de_step)]
        return {
            "disponible": True,
            "operaciones_recibidas": int(len(previas)),
            "importe_recibido": round(float(previas["amount"].sum()), 2),
            "fraudes_confirmados": int(previas["isFraud"].sum()),
        }
