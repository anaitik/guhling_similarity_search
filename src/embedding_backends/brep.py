from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import trimesh

from ..brep_features import brep_features_from_step


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.nan_to_num(vector.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32)


def _projection_matrix(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(
        0.0,
        1.0 / np.sqrt(max(input_dim, 1)),
        size=(input_dim, output_dim),
    ).astype(np.float32)


def _text_brep_vector(text: str, dim: int) -> np.ndarray:
    lowered = text.lower()
    groups = [
        ("face planar flat plate surface", ["face", "planar", "plane", "flat", "plate", "surface"]),
        ("edge line sharp prismatic", ["edge", "line", "sharp", "prismatic", "rectangular"]),
        ("vertex corner block box", ["vertex", "corner", "block", "box", "cube"]),
        ("cylinder circular round hole shaft tube", ["cylinder", "cylindrical", "circle", "round", "hole", "shaft", "tube"]),
        ("spline curved organic freeform", ["spline", "curved", "organic", "freeform", "smooth"]),
        ("shell housing cover enclosure", ["shell", "housing", "cover", "case", "enclosure"]),
        ("solid manifold closed", ["solid", "manifold", "closed", "watertight"]),
        ("open thin sheet", ["open", "thin", "sheet"]),
        ("bracket mounting support fixture", ["bracket", "mount", "mounting", "support", "fixture"]),
        ("fastener screw bolt slot", ["fastener", "screw", "bolt", "slot"]),
    ]
    values = []
    for _, words in groups:
        values.append(float(sum(lowered.count(word) for word in words)))
    values.extend(
        [
            float(len(lowered.split())),
            float(lowered.count(",")),
            float(lowered.count(".")),
        ]
    )
    base = np.log1p(np.array(values, dtype=np.float32))
    if len(base) < dim:
        base = np.pad(base, (0, dim - len(base)))
    return _unit_vector(base[:dim])


@dataclass
class BRepMAEEmbeddingBackend:
    latent_dim: int = 128
    mask_ratio: float = 0.35
    views: int = 8
    seed: int = 2026

    name: str = "brep_mae"

    def _embed_features(self, features: np.ndarray) -> np.ndarray:
        pieces = []
        rng = np.random.default_rng(self.seed)
        for view in range(self.views):
            keep = rng.random(len(features)) >= self.mask_ratio
            if not np.any(keep):
                keep[rng.integers(0, len(features))] = True
            masked = features.copy()
            masked[~keep] = 0.0
            projection = _projection_matrix(len(masked), self.latent_dim, self.seed + view)
            pieces.append(np.tanh(masked @ projection))
        return _unit_vector(np.mean(np.vstack(pieces), axis=0))

    def embed_path(self, path: Path) -> np.ndarray:
        return self._embed_features(brep_features_from_step(path).vector)

    def embed_text(self, text: str) -> np.ndarray:
        features = _text_brep_vector(text, 47)
        return self._embed_features(features)

    def embed_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        raise NotImplementedError("BRepMAE backend indexes STEP/STP B-rep files, not STL meshes.")

    def signature(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "latent_dim": self.latent_dim,
            "mask_ratio": self.mask_ratio,
            "views": self.views,
            "seed": self.seed,
            "note": "BRepMAE-style masked B-rep descriptor baseline for STEP retrieval",
        }


@dataclass
class CADGCLEmbeddingBackend:
    latent_dim: int = 128
    augmentations: int = 6
    drop_ratio: float = 0.2
    seed: int = 2025

    name: str = "cadgcl"

    def _embed_features(self, features: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        views = []
        for aug in range(self.augmentations):
            dropped = features.copy()
            mask = rng.random(len(features)) < self.drop_ratio
            dropped[mask] *= 0.25
            noise = rng.normal(0.0, 0.015, size=len(features)).astype(np.float32)
            augmented = _unit_vector(dropped + noise)
            projection = _projection_matrix(len(augmented), self.latent_dim, self.seed + aug)
            views.append(np.tanh(augmented @ projection))

        positive = np.mean(np.vstack(views), axis=0)
        graph_stats = np.array(
            [
                features[0],
                features[1],
                features[2],
                features[7],
                features[8],
                features[13],
                features[14],
                features[16],
            ],
            dtype=np.float32,
        )
        graph_projection = _projection_matrix(len(graph_stats), self.latent_dim, self.seed + 100)
        return _unit_vector(positive + np.tanh(graph_stats @ graph_projection))

    def embed_path(self, path: Path) -> np.ndarray:
        return self._embed_features(brep_features_from_step(path).vector)

    def embed_text(self, text: str) -> np.ndarray:
        features = _text_brep_vector(text, 47)
        return self._embed_features(features)

    def embed_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        raise NotImplementedError("CADGCL backend indexes STEP/STP B-rep files, not STL meshes.")

    def signature(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "latent_dim": self.latent_dim,
            "augmentations": self.augmentations,
            "drop_ratio": self.drop_ratio,
            "seed": self.seed,
            "note": "CADGCL-style contrastive B-rep graph descriptor baseline for STEP retrieval",
        }
