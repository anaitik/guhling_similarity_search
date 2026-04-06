from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import trimesh

from ..mesh_utils import normalize_mesh


@dataclass
class GeminiEmbeddingBackend:
    api_key: str
    model: str = "models/embedding-001"
    descriptor_version: int = 1

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

    def embed_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        text = self._mesh_descriptor_text(mesh)
        response = self._genai.embed_content(model=self.model, content=text)
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

    def signature(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "model": self.model,
            "descriptor_version": self.descriptor_version,
        }

    def _mesh_descriptor_text(self, mesh: trimesh.Trimesh) -> str:
        mesh_n = normalize_mesh(mesh)
        extents = mesh_n.bounding_box.extents.astype(float)
        ext_sorted = np.sort(extents)
        x, y, z = ext_sorted

        ratio_xy = y / x if x > 1e-8 else 0.0
        ratio_yz = z / y if y > 1e-8 else 0.0
        elongation = z / x if x > 1e-8 else 0.0

        area = float(mesh_n.area)
        volume = float(abs(mesh_n.volume)) if mesh_n.is_watertight else float(mesh_n.volume)

        sphericity = 0.0
        if area > 1e-8 and volume > 0:
            sphericity = (np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0)) / area

        faces = int(mesh_n.faces.shape[0]) if mesh_n.faces is not None else 0
        verts = int(mesh_n.vertices.shape[0])

        tags: List[str] = []
        if elongation > 3.0:
            tags.append("elongated")
        if x / z < 0.25:
            tags.append("flat")
        if abs(ratio_xy - 1.0) < 0.2 and abs(ratio_yz - 1.0) < 0.2:
            tags.append("compact")
        if sphericity > 0.7:
            tags.append("roundish")
        if not tags:
            tags.append("irregular")

        return (
            "3D mesh descriptor. "
            f"Surface area {area:.4f}. "
            f"Volume {volume:.4f}. "
            f"Extents {extents[0]:.4f}, {extents[1]:.4f}, {extents[2]:.4f}. "
            f"Aspect ratios {ratio_xy:.3f}, {ratio_yz:.3f}. "
            f"Sphericity {sphericity:.3f}. "
            f"Vertices {verts}, faces {faces}. "
            f"Tags: {', '.join(tags)}."
        )
