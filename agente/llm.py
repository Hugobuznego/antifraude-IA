"""Conexion con el LLM.

El agente solo necesita algo con un metodo `generar(prompt) -> str`. Asi:
- en produccion usamos Gemini (nivel gratuito de Google AI Studio),
- en los tests usamos un LLM falso y no hace falta clave ni internet,
- cambiar de proveedor es escribir otra clase de 10 lineas.
"""

import os
from typing import Protocol

DEFAULT_MODEL = "gemini-3.5-flash-lite"


class LLM(Protocol):
    nombre: str

    def generar(self, prompt: str) -> str: ...


class GeminiLLM:
    def __init__(self, api_key: str, modelo: str = DEFAULT_MODEL):
        from google import genai  # import aqui: solo hace falta si se usa Gemini

        self.client = genai.Client(api_key=api_key)
        self.nombre = modelo

    def generar(self, prompt: str) -> str:
        from google.genai import types

        resp = self.client.models.generate_content(
            model=self.nombre,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,  # queremos informes consistentes, no creativos
                response_mime_type="application/json",
            ),
        )
        return resp.text


def llm_desde_entorno() -> LLM | None:
    """Devuelve Gemini si hay GEMINI_API_KEY; si no, None (el agente usara plantilla)."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    return GeminiLLM(key, os.getenv("GEMINI_MODEL", DEFAULT_MODEL))
