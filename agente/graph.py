"""Agente que ayuda al analista de fraude a resolver una alerta.

Flujo (LangGraph):

    START -> puntuar -> historial -> politica --(hay algo que revisar?)--> redactar -> guardarrail -> END
                                              \\--(operacion limpia)--> informe_breve ---------------/

- puntuar:     probabilidad y decision del modelo + motivos legibles.
- historial:   que sabemos de la cuenta que recibe el dinero.
- politica:    reglas de negocio deterministas. Fijan la decision MINIMA permitida.
- redactar:    el LLM escribe el informe del caso para el analista.
- guardarrail: el LLM nunca puede recomendar algo MENOS estricto que la politica.
               Si lo intenta, se corrige y queda anotado.

Principio de diseno: el LLM redacta y prioriza, pero no decide solo. Las decisiones
duras las ponen el modelo validado y las reglas, y la ultima palabra es del analista.
"""

import json
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from agente.llm import LLM
from fraude.historial import Historial
from fraude.scoring import APPROVE, BLOCK, REVIEW, Scorer

SEVERIDAD = {APPROVE: 0, REVIEW: 1, BLOCK: 2}
IMPORTE_CUATRO_OJOS = 200_000  # politica de ejemplo: importes altos los revisan dos personas


class Informe(BaseModel):
    resumen: str
    senales: list[str]
    que_revisar: list[str]
    recomendacion: Literal["APPROVE", "REVIEW", "BLOCK"]
    justificacion: str


class Estado(TypedDict, total=False):
    transaccion: dict
    puntuacion: dict
    historial: dict
    reglas: list[str]
    decision_minima: str
    informe: dict
    fuente_informe: str
    notas: list[str]
    recomendacion_final: str


def mas_estricta(a: str, b: str) -> str:
    return a if SEVERIDAD[a] >= SEVERIDAD[b] else b


def construir_prompt(estado: Estado) -> str:
    tx = estado["transaccion"]
    datos = {
        "transaccion": {k: tx[k] for k in ("type", "amount", "oldbalanceOrg", "oldbalanceDest", "nameDest", "step")},
        "modelo": estado["puntuacion"],
        "historial_cuenta_destino": estado["historial"],
        "reglas_activadas": estado["reglas"],
        "decision_minima_por_politica": estado["decision_minima"],
    }
    return f"""Eres un asistente de un equipo de prevencion de fraude de un banco.
Redactas el informe de una alerta para que un analista humano decida rapido.

Reglas:
- Usa SOLO los datos de abajo. No inventes datos del cliente ni de la operacion.
- Espanol claro y directo, frases cortas. Importes en euros con separador de miles.
- "recomendacion" no puede ser menos estricta que "decision_minima_por_politica"
  (orden: APPROVE < REVIEW < BLOCK).
- Si un dato no esta disponible, dilo en vez de suponerlo.

Datos del caso:
{json.dumps(datos, ensure_ascii=False, indent=2)}

Responde SOLO con un JSON con esta forma exacta:
{{
  "resumen": "2-3 frases sobre que ha pasado y por que es o no sospechoso",
  "senales": ["senal de riesgo 1", "..."],
  "que_revisar": ["comprobacion concreta que debe hacer el analista", "..."],
  "recomendacion": "APPROVE | REVIEW | BLOCK",
  "justificacion": "1-2 frases"
}}"""


def informe_plantilla(estado: Estado) -> dict:
    """Informe sin LLM: se usa si no hay clave o si el LLM falla."""
    p = estado["puntuacion"]
    return Informe(
        resumen=(f"El modelo asigna una probabilidad de fraude de {p['fraud_probability']:.1%} "
                 f"y propone {p['decision']}."),
        senales=p["reasons"] + estado["reglas"],
        que_revisar=["Contactar con el titular para confirmar la operacion",
                     "Revisar otras operaciones recientes hacia la misma cuenta destino"],
        recomendacion=estado["decision_minima"],
        justificacion="Informe generado sin LLM a partir de las reglas y del modelo.",
    ).model_dump()


def build_agent(scorer: Scorer, historial: Historial, llm: LLM | None):

    def puntuar(estado: Estado) -> Estado:
        return {"puntuacion": scorer.score(estado["transaccion"]), "notas": []}

    def consultar_historial(estado: Estado) -> Estado:
        tx = estado["transaccion"]
        return {"historial": historial.cuenta_destino(tx["nameDest"], tx["step"])}

    def politica(estado: Estado) -> Estado:
        tx, p, h = estado["transaccion"], estado["puntuacion"], estado["historial"]
        reglas, minima = [], p["decision"]
        if h.get("fraudes_confirmados", 0) > 0:
            reglas.append(f"La cuenta destino ya recibio {h['fraudes_confirmados']} operaciones de fraude confirmado")
            minima = mas_estricta(minima, BLOCK)
        if tx["oldbalanceOrg"] > 0 and abs(tx["amount"] - tx["oldbalanceOrg"]) < 0.01:
            reglas.append("El importe coincide al centimo con el saldo de la cuenta origen")
            minima = mas_estricta(minima, REVIEW)
        if tx["amount"] >= IMPORTE_CUATRO_OJOS and minima != APPROVE:
            reglas.append("Importe alto: la resolucion necesita un segundo revisor (cuatro ojos)")
        return {"reglas": reglas, "decision_minima": minima}

    def hay_que_revisar(estado: Estado) -> Literal["redactar", "informe_breve"]:
        limpio = estado["decision_minima"] == APPROVE and not estado["reglas"]
        return "informe_breve" if limpio else "redactar"

    def informe_breve(estado: Estado) -> Estado:
        # Operacion limpia: no gastamos llamadas al LLM (cuota gratuita limitada)
        return {"informe": informe_plantilla(estado), "fuente_informe": "sin_llm"}

    def redactar(estado: Estado) -> Estado:
        if llm is None:
            return {"informe": informe_plantilla(estado), "fuente_informe": "plantilla",
                    "notas": estado["notas"] + ["No hay GEMINI_API_KEY: informe generado con plantilla"]}
        try:
            bruto = llm.generar(construir_prompt(estado))
            texto = bruto.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
            informe = Informe.model_validate_json(texto).model_dump()
            return {"informe": informe, "fuente_informe": f"llm:{llm.nombre}"}
        except (ValidationError, ValueError) as e:
            nota = f"Respuesta del LLM no valida ({type(e).__name__}): se usa plantilla"
        except Exception as e:  # red, cuota agotada, clave incorrecta...
            nota = f"Error llamando al LLM ({type(e).__name__}): se usa plantilla"
        return {"informe": informe_plantilla(estado), "fuente_informe": "plantilla",
                "notas": estado["notas"] + [nota]}

    def guardarrail(estado: Estado) -> Estado:
        propuesta = estado["informe"]["recomendacion"]
        final = mas_estricta(propuesta, estado["decision_minima"])
        notas = list(estado["notas"])
        if final != propuesta:
            notas.append(f"El LLM recomendo {propuesta}; corregido a {final} por politica")
        return {"recomendacion_final": final, "notas": notas}

    g = StateGraph(Estado)
    g.add_node("puntuar", puntuar)
    g.add_node("historial", consultar_historial)
    g.add_node("politica", politica)
    g.add_node("redactar", redactar)
    g.add_node("informe_breve", informe_breve)
    g.add_node("guardarrail", guardarrail)

    g.add_edge(START, "puntuar")
    g.add_edge("puntuar", "historial")
    g.add_edge("historial", "politica")
    g.add_conditional_edges("politica", hay_que_revisar)
    g.add_edge("redactar", "guardarrail")
    g.add_edge("informe_breve", "guardarrail")
    g.add_edge("guardarrail", END)
    return g.compile()


def revisar(agente, transaccion: dict) -> dict:
    estado = agente.invoke({"transaccion": transaccion})
    return {
        "recomendacion_final": estado["recomendacion_final"],
        "decision_minima_politica": estado["decision_minima"],
        "puntuacion": estado["puntuacion"],
        "historial_destino": estado["historial"],
        "reglas": estado["reglas"],
        "informe": estado["informe"],
        "fuente_informe": estado["fuente_informe"],
        "notas": estado["notas"],
    }
