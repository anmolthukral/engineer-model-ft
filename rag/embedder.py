"""Embedding generation for RAG pipeline.

Supports Ollama (local nomic-embed-text) and OpenAI (text-embedding-3-small/large).
"""
import os
from abc import ABC, abstractmethod
from typing import Optional

import requests


class Embedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate embedding vector for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class OllamaEmbedder(Embedder):
    """Ollama local embeddings (nomic-embed-text, mxbai-embed-large, etc.)."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dimension: Optional[int] = None

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        if embeddings and self._dimension is None:
            self._dimension = len(embeddings[0])
        return embeddings

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Probe with a dummy embedding
            test = self.embed("test")
            self._dimension = len(test)
        return self._dimension


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings (text-embedding-3-small, text-embedding-3-large)."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self._dimension = 1536 if "3-small" in model else 3072

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedder() -> Embedder:
    """Factory: prefer OpenAI if key available, else Ollama."""
    if os.environ.get("OPENAI_API_KEY"):
        model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        return OpenAIEmbedder(model=model)
    model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    return OllamaEmbedder(model=model, base_url=base_url)