from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import trimesh

from ..mesh_descriptor import mesh_descriptor_text


@dataclass
class GeminiEmbeddingBackend:
    api_key: str
    model: str = "models/embedding-001"
    descriptor_version: int = 2

    name: str = "gemini"

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        try:
            import google.generativeai as genai
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "google-generativeai is required for Gemini embeddings. "
                "Install it with `pip install google-generativeai`."
            ) from exc

        self._genai = genai
        self._genai.configure(api_key=self.api_key)

    def _embed_content(self, content: str) -> np.ndarray:
        response = self._genai.embed_content(model=self.model, content=content)
        embedding = None
        if isinstance(response, dict):
            embedding = response.get("embedding")
            if embedding is None and "embeddings" in response:
                embedding = response["embeddings"][0].get("embedding")
        else:
            embedding = getattr(response, "embedding", None)

        if embedding is None:
            raise ValueError("Gemini embedding response did not include an embedding")
        return np.array(embedding, dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        return self._embed_content(text)

    def embed_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        text = mesh_descriptor_text(mesh)
        return self._embed_content(text)

    def embed_mesh_with_context(self, mesh: trimesh.Trimesh, context: str) -> np.ndarray:
        text = mesh_descriptor_text(mesh)
        if context:
            text = f"{text} Product/catalog context: {context}."
        return self._embed_content(text)

    def signature(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "model": self.model,
            "descriptor_version": self.descriptor_version,
        }
