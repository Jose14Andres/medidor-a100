"""Backends de inferencia (DESARROLLO.md §3.3). Fase 1: Ollama; fase 2: vLLM."""

from medidor.backends.base import Backend, ErrorBackend, Fragmento, Resultado, Solicitud, medir

__all__ = ["Backend", "ErrorBackend", "Fragmento", "Resultado", "Solicitud", "medir"]
