# Antifraude en transacciones bancarias

Este proyecto detecta fraude en transferencias y retiradas de efectivo, decide si cada operación se **aprueba, se revisa o se bloquea** y lo sirve con una API.

Datos: [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1). Son 6,3 millones de transacciones de dinero móvil simuladas a partir de logs reales de un operador africano.

## El problema de negocio

El fraude es menos del 0,2% de las transacciones, pero cada caso cuesta de media mucho dinero. El equipo de fraude no puede revisarlo todo, así que el modelo tiene que decidir **dónde poner la atención de los analistas**.

Por eso el proyecto no se queda en "un modelo con buena métrica". Responde a tres preguntas:

1. ¿Cuánto fraude detectamos revisando como mucho el 2% de las transacciones?
2. ¿Cuánto dinero nos ahorramos frente a no tener modelo y frente a la regla actual?
3. ¿Cuándo podemos bloquear automáticamente sin que lo mire un humano?

## Decisiones de diseño

| Decisión | Por qué |
|---|---|
| Solo se modelan `TRANSFER` y `CASH_OUT` | En los datos no hay ningún fraude en los otros tipos |
| Split **temporal** (train, validación, test por hora) | En producción siempre predices el futuro. Un split aleatorio infla las métricas |
| Métrica principal: **PR-AUC** | Con menos del 1% de fraude, decir siempre "no es fraude" da un 99% de accuracy |
| Umbral de revisión elegido por **coste en euros** | Coste = fraude no detectado + 5 € por alerta revisada, con un límite del 2% de transacciones (capacidad del equipo) |
| Umbral de bloqueo con **precisión ≥ 95%** | Bloquear a un cliente legítimo es caro en reputación, así que solo se hace cuando es casi seguro |
| Dos baselines: regla actual (`isFlaggedFraud`) y regresión logística | Si el boosting no gana claro, sobra complejidad |

Todos los supuestos (coste por alerta, capacidad, precisión de bloqueo) están en `fraude/train.py` y se guardan con el modelo.

## Estructura

```
fraude/
  data.py       carga de PaySim + generador sintético con el mismo esquema
  features.py   variables con hipótesis de negocio
  train.py      entrenamiento, umbrales por coste, evaluación
  scoring.py    puntuación + decisión + motivos legibles
app/main.py     API FastAPI
tests/          29 tests (datos, variables, umbrales, entrenamiento, API)
.github/        CI: tests + entrenamiento de prueba en cada push
```

## Cómo usarlo

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

**Entrenar** con los datos reales (ver `data/README.md` para descargarlos):

```bash
python -m fraude.train --data data/PS_20174392719_1491204439457_log.csv
```

O sin descargar nada, con datos sintéticos:

```bash
python -m fraude.train --synthetic 300000
```

**Arrancar la API:**

```bash
uvicorn app.main:app --reload
```

Abre http://localhost:8000/docs. Ahí `POST /score` ya trae un ejemplo de transacción sospechosa listo para probar.

| Endpoint | Qué hace |
|---|---|
| `GET /health` | Estado y si hay modelo cargado |
| `GET /model/info` | Umbrales, supuestos y métricas de test |
| `POST /score` | Puntúa una transacción |
| `POST /score/batch` | Puntúa hasta 1000 transacciones |

Ejemplo de respuesta:

```json
{
  "fraud_probability": 0.8153,
  "decision": "REVIEW",
  "reasons": [
    "La operacion vacia por completo la cuenta de origen",
    "La cuenta destino no refleja el dinero recibido (saldo 0 antes y despues)",
    "Operacion de madrugada"
  ]
}
```

**Tests:**

```bash
pytest -v
```

## Resultados

> Pendiente de rellenar con PaySim real.

Referencia con 300.000 transacciones **sintéticas** (solo para comprobar que el pipeline funciona; no son resultados del proyecto):

| Métrica (test) | Modelo | Regla actual | Regresión logística |
|---|---|---|---|
| PR-AUC | 0,59 | | 0,48 |
| Recall en revisión | 71% | 0,7% | |
| % del importe defraudado detectado | 92% | | |
| Alertas al día | 84 | | |

**Ojo con PaySim real:** es un simulador, y las variables de descuadre de saldo (`error_orig`, `error_dest`) suelen separar el fraude casi a la perfección. Si salen métricas cercanas a 1, hay que contarlo como limitación del dataset y no como éxito del modelo, y probar qué pasa sin esas variables.

## Próximos pasos

- [ ] EDA en notebook con los datos reales
- [ ] Agente LangGraph que ayuda al analista a resolver alertas
- [ ] Dashboard en Power BI con alertas y KPIs
- [ ] Explicabilidad con SHAP
