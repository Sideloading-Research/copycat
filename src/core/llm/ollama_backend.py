"""Ollama LLM backend — talks to a local Ollama server."""

import ollama
from src.config import cfg


class OllamaLLM:
    """LLM via Ollama's HTTP API.

    Usage::

        llm = OllamaLLM()
        reply = llm.generate("What is your name?")
    """

    def __init__(self, model: str | None = None, threads: int | None = None):
        self.model = model or cfg.llm_model
        self.threads = threads or cfg.llm_threads

    def generate(self, prompt: str, **kwargs) -> str:
        resp = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"num_thread": self.threads, **kwargs},
        )
        return resp["response"].strip()
