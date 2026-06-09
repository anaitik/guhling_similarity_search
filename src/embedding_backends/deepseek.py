from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import trimesh

from ..mesh_descriptor import mesh_descriptor_text


VECTOR_DIM = 48


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.nan_to_num(vector.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32)


def _parse_vector(text: str) -> List[float]:
    text = text.strip()
    if not text:
        raise ValueError("DeepSeek response was empty")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text)
        if not match:
            raise ValueError("DeepSeek response did not include a JSON vector")
        data = json.loads(match.group(0))

    if isinstance(data, dict):
        data = data.get("vector")
    if not isinstance(data, list):
        raise ValueError("DeepSeek response must include a JSON vector array")
    if len(data) != VECTOR_DIM:
        raise ValueError(f"DeepSeek vector must have exactly {VECTOR_DIM} numbers")
    return [float(value) for value in data]


@dataclass
class DeepSeekEmbeddingBackend:
    api_key: str
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    descriptor_version: int = 2
    vector_dim: int = VECTOR_DIM

    name: str = "deepseek"

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("DeepSeek API key is required")
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "openai is required for DeepSeek API access. "
                "Install it with `pip install openai`."
            ) from exc

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _embed_content(self, content: str) -> np.ndarray:
        prompt = (
            "Convert the following 3D mesh/product description into a deterministic "
            f"{self.vector_dim}-number semantic similarity vector. Return only a JSON "
            'object in this exact shape: {"vector": [0.1, 0.2, ...]}. The vector '
            f"array must contain exactly {self.vector_dim} numbers from 0 to 1. "
            "Use the same dimensions every time, covering "
            "shape proportions, flatness, elongation, roundness, hollowness, holes, "
            "mounting/bracket cues, handle/shaft cues, container/shell cues, mechanical "
            "complexity, symmetry, compactness, surface/detail complexity, and likely "
            "functional affordances.\n\n"
            f"{content}"
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You create stable numeric vectors for 3D mesh similarity search. "
                        "Return only valid JSON, with no prose or markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=512,
            extra_body={"thinking": {"type": "disabled"}},
            stream=False,
        )
        text = response.choices[0].message.content or ""
        return _unit_vector(np.array(_parse_vector(text), dtype=np.float32))

    def embed_text(self, text: str) -> np.ndarray:
        return self._embed_content(text)

    def embed_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        return self._embed_content(mesh_descriptor_text(mesh))

    def embed_mesh_with_context(self, mesh: trimesh.Trimesh, context: str) -> np.ndarray:
        text = mesh_descriptor_text(mesh)
        if context:
            text = f"{text} Product/catalog context: {context}."
        return self._embed_content(text)

    def signature(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "descriptor_version": self.descriptor_version,
            "vector_dim": self.vector_dim,
        }
