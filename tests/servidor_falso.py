"""Servidor HTTP local que imita vLLM y Ollama con latencias fijas (DESARROLLO.md §9).

Sirve en el mismo puerto las rutas de ambos backends:

- vLLM:   GET /health, GET /v1/models, POST /v1/completions (stream SSE)
- Ollama: GET /api/version, /api/tags, /api/ps, POST /api/generate (stream NDJSON)

Tokenizador falso: un token = una palabra separada por espacios. Cada fragmento de salida es un
token ("tok "), salvo que se configure `tokens_por_fragmento` > 1.

Uso en tests: `with arrancar(Ajustes(...)) as url: ...`.
Uso manual:   `python tests/servidor_falso.py --puerto 8100`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import socketserver
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

NS = 1_000_000_000


@dataclass
class Ajustes:
    modelo: str = "modelo-falso"
    ttft_s: float = 0.05            # espera antes del primer fragmento
    itl_s: float = 0.01             # espera entre fragmentos consecutivos
    tokens_por_fragmento: int = 1
    max_model_len: int = 4096       # vLLM: /v1/models
    num_ctx_defecto: int = 2048     # Ollama: si la solicitud no fija num_ctx
    size_vram: int = 20 * 1024**3   # Ollama /api/ps; 0 simula "cayó a CPU"
    load_duration_s: float = 0.0
    version_ollama: str = "0.24.0"


def _contar_tokens(texto: str) -> int:
    return len(texto.split())


def _fragmentos(n_salida: int, por_fragmento: int) -> list[int]:
    """Reparte n_salida tokens en fragmentos de `por_fragmento` (el último puede ser menor)."""
    completos, resto = divmod(n_salida, por_fragmento)
    return [por_fragmento] * completos + ([resto] if resto else [])


class _Manejador(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    ajustes: Ajustes  # se inyecta en una subclase por servidor

    def log_message(self, *args) -> None:  # silencio en los tests
        pass

    # --- utilidades ---------------------------------------------------------------------------

    def _json(self, codigo: int, cuerpo: dict) -> None:
        datos = json.dumps(cuerpo).encode()
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)

    def _empezar_stream(self, tipo: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", tipo)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _trozo(self, datos: bytes) -> None:
        self.wfile.write(f"{len(datos):X}\r\n".encode() + datos + b"\r\n")
        self.wfile.flush()

    def _terminar_stream(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _leer_json(self) -> dict | None:
        n = int(self.headers.get("Content-Length", 0))
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return None

    # --- rutas ------------------------------------------------------------------------------

    def do_GET(self) -> None:
        a = self.ajustes
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": a.modelo, "object": "model", "max_model_len": a.max_model_len}]})
        elif self.path == "/api/version":
            self._json(200, {"version": a.version_ollama})
        elif self.path == "/api/tags":
            self._json(200, {"models": [{"name": a.modelo, "model": a.modelo}]})
        elif self.path == "/api/ps":
            self._json(200, {"models": [{"name": a.modelo, "model": a.modelo, "size_vram": a.size_vram}]})
        else:
            self._json(404, {"error": f"ruta desconocida: {self.path}"})

    def do_POST(self) -> None:
        cuerpo = self._leer_json()
        if cuerpo is None:
            self._json(400, {"error": "JSON inválido"})
        elif self.path == "/v1/completions":
            self._completions(cuerpo)
        elif self.path == "/api/generate":
            self._generate(cuerpo)
        else:
            self._json(404, {"error": f"ruta desconocida: {self.path}"})

    def _completions(self, cuerpo: dict) -> None:
        a = self.ajustes
        if cuerpo.get("model") != a.modelo:
            self._json(404, {"error": f"modelo desconocido: {cuerpo.get('model')}"})
            return
        if not cuerpo.get("stream"):
            self._json(400, {"error": "el servidor falso solo implementa stream=true"})
            return
        n_entrada = _contar_tokens(cuerpo.get("prompt", ""))
        n_salida = int(cuerpo.get("max_tokens", 16))
        if n_entrada + n_salida > a.max_model_len:
            self._json(400, {"error": "entrada + salida supera max_model_len"})
            return

        self._empezar_stream("text/event-stream")
        trozos = _fragmentos(n_salida, a.tokens_por_fragmento)
        time.sleep(a.ttft_s)
        for i, k in enumerate(trozos):
            if i:
                time.sleep(a.itl_s)
            fin = "length" if i == len(trozos) - 1 else None
            evento = {"object": "text_completion", "model": a.modelo,
                      "choices": [{"index": 0, "text": "tok " * k, "finish_reason": fin}]}
            self._trozo(f"data: {json.dumps(evento)}\n\n".encode())
        if (cuerpo.get("stream_options") or {}).get("include_usage"):
            uso = {"prompt_tokens": n_entrada, "completion_tokens": n_salida,
                   "total_tokens": n_entrada + n_salida}
            self._trozo(f"data: {json.dumps({'choices': [], 'usage': uso})}\n\n".encode())
        self._trozo(b"data: [DONE]\n\n")
        self._terminar_stream()

    def _generate(self, cuerpo: dict) -> None:
        a = self.ajustes
        if cuerpo.get("model") != a.modelo:
            self._json(404, {"error": f"model '{cuerpo.get('model')}' not found"})
            return
        opciones = cuerpo.get("options") or {}
        num_ctx = int(opciones.get("num_ctx", a.num_ctx_defecto))
        # Como Ollama 0.24: si el prompt no cabe en num_ctx, se trunca en silencio.
        n_entrada = min(_contar_tokens(cuerpo.get("prompt", "")), num_ctx)
        n_salida = int(opciones.get("num_predict", 16))
        trozos = _fragmentos(n_salida, a.tokens_por_fragmento)

        if cuerpo.get("stream", True) is False:
            self._json(200, self._final_ollama(n_entrada, n_salida, len(trozos), "tok " * n_salida))
            return

        self._empezar_stream("application/x-ndjson")
        time.sleep(a.ttft_s)
        for i, k in enumerate(trozos):
            if i:
                time.sleep(a.itl_s)
            linea = {"model": a.modelo, "response": "tok " * k, "done": False}
            self._trozo((json.dumps(linea) + "\n").encode())
        self._trozo((json.dumps(self._final_ollama(n_entrada, n_salida, len(trozos), "")) + "\n").encode())
        self._terminar_stream()

    def _final_ollama(self, n_entrada: int, n_salida: int, n_trozos: int, texto: str) -> dict:
        a = self.ajustes
        prompt_ns = int(a.ttft_s * NS)
        eval_ns = int(a.itl_s * max(n_trozos - 1, 0) * NS)
        return {
            "model": a.modelo, "response": texto, "done": True, "done_reason": "length",
            "prompt_eval_count": n_entrada, "prompt_eval_duration": prompt_ns,
            "eval_count": n_salida, "eval_duration": eval_ns,
            "load_duration": int(a.load_duration_s * NS),
            "total_duration": prompt_ns + eval_ns + int(a.load_duration_s * NS),
        }


class _Servidor(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self) -> None:
        # HTTPServer.server_bind llama a socket.getfqdn(), que tarda segundos en Windows.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


@contextlib.contextmanager
def arrancar(ajustes: Ajustes | None = None, puerto: int = 0) -> Iterator[str]:
    """Arranca el servidor en 127.0.0.1 (puerto libre si 0) y devuelve su URL base."""
    manejador = type("Manejador", (_Manejador,), {"ajustes": ajustes or Ajustes()})
    servidor = _Servidor(("127.0.0.1", puerto), manejador)
    hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
    hilo.start()
    try:
        yield f"http://127.0.0.1:{servidor.server_address[1]}"
    finally:
        servidor.shutdown()
        servidor.server_close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Servidor falso vLLM + Ollama")
    ap.add_argument("--puerto", type=int, default=8100)
    ap.add_argument("--modelo", default="modelo-falso")
    ap.add_argument("--ttft", type=float, default=0.05)
    ap.add_argument("--itl", type=float, default=0.01)
    a = ap.parse_args()
    with arrancar(Ajustes(modelo=a.modelo, ttft_s=a.ttft, itl_s=a.itl), a.puerto) as url:
        print(f"servidor falso en {url} (Ctrl+C para parar)")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
