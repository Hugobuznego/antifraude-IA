"""Diagnostico: por que el modelo realista sale casi perfecto en PaySim?

Uso:
    python -m fraude.diagnostico --data data/PS_20174392719_1491204439457_log.csv

Comprueba dos hipotesis sobre el simulador:
1. El fraude se lleva EXACTAMENTE todo el saldo de la cuenta origen (al centimo).
2. El volumen de operaciones legitimas cae en las ultimas horas, pero el fraude no,
   asi que el periodo de test tiene una proporcion de fraude distinta.
"""

import argparse

import pandas as pd

from fraude.data import RISKY_TYPES, load_paysim, generate_synthetic
from fraude.train import temporal_split


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def regla(nombre: str, mask: pd.Series, y: pd.Series) -> None:
    tp = int((mask & (y == 1)).sum())
    n_fraud = int(y.sum())
    n_flag = int(mask.sum())
    precision = tp / n_flag if n_flag else 0.0
    recall = tp / n_fraud if n_fraud else 0.0
    print(f"  {nombre}")
    print(f"    marca {n_flag:,} operaciones | precision {pct(precision)} | recall {pct(recall)}")


def diagnostico(df: pd.DataFrame) -> None:
    df = df[df["type"].astype(str).isin(RISKY_TYPES)].copy()
    y = df["isFraud"]
    exacto = (df["oldbalanceOrg"] > 0) & ((df["amount"] - df["oldbalanceOrg"]).abs() < 0.01)
    dest_vacio = df["oldbalanceDest"] == 0
    transfer = df["type"].astype(str) == "TRANSFER"

    print("\n=== Hipotesis 1: el fraude vacia la cuenta al centimo ===")
    print(f"  Fraudes con importe == saldo origen:   {pct(exacto[y == 1].mean())}")
    print(f"  Legitimas con importe == saldo origen: {pct(exacto[y == 0].mean())}")
    print("\n  Reglas de una linea, sin modelo:")
    regla("importe == saldo origen", exacto, y)
    regla("importe == saldo origen Y destino sin saldo", exacto & dest_vacio, y)
    regla("TRANSFER a destino sin saldo", transfer & dest_vacio, y)

    print("\n=== Hipotesis 2: el periodo de test es distinto ===")
    tr, va, te = temporal_split(df)
    for nombre, parte in (("train", tr), ("validacion", va), ("test", te)):
        horas = parte["step"].nunique()
        print(f"  {nombre:<11} horas {parte.step.min()}-{parte.step.max()} | "
              f"{len(parte) / horas:>8,.0f} operaciones/hora | fraude {pct(parte.isFraud.mean())}")

    print("\nComo leerlo:")
    print("  - Si una regla de una linea ya tiene precision y recall altisimos, el modelo")
    print("    no aporta mucho: el patron lo ha 'escrito' el simulador.")
    print("  - Si el % de fraude en test es muy distinto al de train, las metricas de test")
    print("    no se pueden comparar con lo que pasaria en un mes normal.")


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data")
    src.add_argument("--synthetic", type=int)
    args = parser.parse_args()
    df = load_paysim(args.data) if args.data else generate_synthetic(args.synthetic)
    diagnostico(df)


if __name__ == "__main__":
    main()
