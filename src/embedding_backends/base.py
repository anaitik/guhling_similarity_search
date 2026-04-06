from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import numpy as np
import trimesh


class EmbeddingBackend(Protocol):
    name: str

    def embed_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        ...

    def signature(self) -> Dict[str, object]:
        ...
