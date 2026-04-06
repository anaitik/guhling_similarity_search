from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import trimesh

from ..features import compute_shape_features


@dataclass
class LocalEmbeddingBackend:
    radial_bins: int = 32
    d2_bins: int = 32
    points: int = 2048
    d2_pairs: int = 4096
    seed: int = 123

    name: str = "local"

    def embed_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        return compute_shape_features(
            mesh,
            radial_bins=self.radial_bins,
            d2_bins=self.d2_bins,
            points=self.points,
            d2_pairs=self.d2_pairs,
            seed=self.seed,
        )

    def signature(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "radial_bins": self.radial_bins,
            "d2_bins": self.d2_bins,
            "points": self.points,
            "d2_pairs": self.d2_pairs,
            "seed": self.seed,
        }
