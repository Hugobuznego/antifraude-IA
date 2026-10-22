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
| Por defecto, **solo variables disponibles antes de autorizar** | Los saldos posteriores a la operación no existen cuando hay que decidir (ver "La fuga de información") |

Todos los supuestos (coste por alerta, capacidad, precisión de bloqueo) están en `fraude/train.py` y se guardan con el modelo.

## Estructura

```
fraude/
  data.py       carga de PaySim + generador sintético con el mismo esquema
  features.py   variables con hipótesis de negocio (sets "realista" y "completo")
  train.py      entrenamiento, umbrales por coste, evaluación
  scoring.py    puntuación + decisión + motivos legibles
  historial.py  historial de cuentas destino
  diagnostico.py  diagnóstico de artefactos del simulador
agente/
  graph.py      agente LangGraph (puntuar, historial, política, redactar, guardarraíl)
  llm.py        conexión con Gemini, intercambiable
app/main.py     API FastAPI
tests/          47 tests (datos, variables, fuga, umbrales, entrenamiento, API, agente)
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

Para reproducir la fuga de información (modelo con saldos posteriores):

```bash
python -m fraude.train --data data/PS_20174392719_1491204439457_log.csv --features completo --out artifacts_completo
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
| `POST /agent/review` | El agente revisa la operación y redacta el informe para el analista |

Ejemplo de respuesta:

```json
{
  "fraud_probability": 0.8153,
  "decision": "REVIEW",
  "reasons": [
    "La operacion se lleva todo el saldo de la cuenta de origen",
    "Transferencia a una cuenta destino sin saldo previo",
    "Operacion de madrugada"
  ]
}
```

**Tests:**

```bash
pytest -v
```

## La fuga de información

El primer modelo con PaySim real, usando **todas** las variables, dio esto en test (552.504 transacciones, 4.258 fraudes):

| Métrica | Valor |
|---|---|
| PR-AUC | 1,00 |
| Recall y precisión | 100% y 100% |
| Fraude detectado (importe) | 100% |

Un modelo perfecto en fraude no es una buena noticia: es una señal de alarma. El modelo usaba `newbalanceOrig` y `newbalanceDest`, que son los saldos **después** de la operación, y sus descuadres. Un banco tiene que aprobar o bloquear **antes** de mover el dinero, así que en producción esa información no existe.

La solución es separar las variables en dos conjuntos (`fraude/features.py`):
- **realista** (por defecto): importe, saldos previos, tipo y hora.
- **completo**: añade los saldos posteriores. Solo se usa para mostrar la fuga.

Hay un test (`test_set_realista_no_usa_saldos_posteriores`) que garantiza que el modelo por defecto no puede volver a usarlos.

## Resultados en PaySim (y por qué no bastan)

| Métrica (test) | Sin saldos | Realista | Completo (fuga) |
|---|---|---|---|
| PR-AUC | 0,40 | 0,9998 | 1,00 |
| Recall en revisión | 45% | 99,8% | 100% |
| % del importe defraudado detectado | 77% | 99,99% | 100% |
| Alertas al día | 417 | 262 | 263 |
| Regresión logística (PR-AUC) | 0,09 | 0,76 | 0,93 |
| Regla actual `isFlaggedFraud` (recall) | 0,3% | 0,3% | 0,3% |

Quitar la fuga no bastó: el modelo realista seguía casi perfecto. `python -m fraude.diagnostico` encontró por qué:

**1. El simulador escribe el fraude con una regla.** El 97,6% de los fraudes mueve *exactamente* el saldo de la cuenta origen, y ninguna operación legítima lo hace (0,00%). La regla de una línea `importe == saldo origen`, sin modelo, tiene un 100% de precisión y un 97,6% de recall.

**2. El periodo de test no se parece al de entrenamiento.**

| Periodo | Operaciones/hora | % fraude |
|---|---|---|
| Train (h 1-281) | 5.953 | 0,19% |
| Validación (h 282-354) | 7.466 | 0,14% |
| Test (h 355-743) | 1.420 | 0,77% |

A partir de la hora ~355 la actividad legítima cae a una cuarta parte pero el fraude sigue igual, así que el % de fraude en test se multiplica por 4. Incluso el modelo "sin saldos" se aprovecha de esto (la hora del día separa mejor cuando apenas hay clientes legítimos).

**Conclusión:** PaySim sirve para construir y probar el sistema (API, umbrales por coste, tests), pero no para medir un modelo de fraude. Cualquier métrica sobre él refleja cómo se programó el simulador.

## Agente para el analista (LangGraph + Gemini)

Un modelo solo dice "probabilidad 0,82". El analista que recibe la alerta necesita contexto para decidir rápido. El agente se lo prepara:

```
START → puntuar → historial → política ─┬─ (algo que revisar) → redactar (Gemini) ─┬→ guardarraíl → END
                                         └─ (operación limpia) → informe breve ─────┘
```

| Paso | Qué hace |
|---|---|
| puntuar | Probabilidad, decisión del modelo y motivos legibles |
| historial | Qué ha recibido antes la cuenta destino y cuántos fraudes confirmados tiene (solo operaciones anteriores) |
| política | Reglas de negocio deterministas que fijan la decisión **mínima**: cuenta destino con fraude previo → BLOCK; importe igual al saldo al céntimo → REVIEW; importes altos → cuatro ojos |
| redactar | Gemini escribe el informe del caso: resumen, señales, qué comprobar y recomendación, en JSON validado con Pydantic |
| guardarraíl | El LLM **nunca** puede recomendar algo menos estricto que la política. Si lo intenta, se corrige y queda anotado |

Decisiones de diseño:
- **El LLM redacta, no decide.** Las decisiones duras las ponen el modelo validado y las reglas, y la última palabra es del analista.
- **Operaciones limpias no llaman al LLM**: ahorra cuota y latencia.
- **Si no hay clave, se agota la cuota o el LLM responde mal**, el agente usa una plantilla y lo anota en `notas`. La API nunca se cae por el LLM.
- **Proveedor intercambiable**: el agente solo necesita un objeto con `generar(prompt)`. Gemini está en `agente/llm.py`, y cambiar a otro LLM es una clase de 10 líneas.
- **Tests sin clave ni internet**: usan un LLM falso que simula respuestas buenas, malas, inválidas y caídas de red.

### Activar Gemini (gratis)

1. Entra en https://aistudio.google.com/apikey con tu cuenta de Google y pulsa **Create API key**.
2. Copia `.env.example` como `.env` y pega la clave. El `.env` está en `.gitignore` y no se sube a GitHub.
3. Arranca la API y prueba `POST /agent/review` en http://localhost:8000/docs.

Sin clave también funciona, pero el informe sale de una plantilla.

## Próximos pasos

- [ ] EDA en notebook con los datos reales
- [ ] Revisar posibles artefactos del simulador (p. ej. volumen de transacciones por hora en el periodo de test)
- [ ] Dashboard en Power BI con alertas y KPIs
- [ ] Explicabilidad con SHAP
