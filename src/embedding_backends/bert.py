from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import trimesh

from ..mesh_descriptor import mesh_descriptor_text


@dataclass
class BertEmbeddingBackend:
    model_name: str = "google/bert_uncased_L-2_H-128_A-2"
    max_length: int = 128
    device: str = "cpu"
    descriptor_version: int = 2
    sample_points: int = 1024

    name: str = "bert"

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModel, BertTokenizer
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "BERT embeddings require torch and transformers. "
                "Install them with `pip install torch transformers`."
            ) from exc

        self._torch = torch
        self._tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def embed_text(self, text: str) -> np.ndarray:
        tokens = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}

        with self._torch.no_grad():
            output = self._model(**tokens)
            hidden = output.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts

        vector = pooled.squeeze(0).detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return vector

    def embed_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        text = mesh_descriptor_text(mesh, sample_points=self.sample_points)
        return self.embed_text(text)

    def embed_mesh_with_context(self, mesh: trimesh.Trimesh, context: str) -> np.ndarray:
        text = mesh_descriptor_text(mesh, sample_points=self.sample_points)
        if context:
            text = f"{text} Product catalog context: {context}."
        return self.embed_text(text)

    def signature(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "descriptor_version": self.descriptor_version,
            "sample_points": self.sample_points,
        }
