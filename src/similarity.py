from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def build_weight_vector(
    radial_bins: int,
    d2_bins: int,
    include_scale: bool,
    radial_weight: float = 1.0,
    d2_weight: float = 1.0,
    size_weight: float = 1.0,
    extent_weight: float = 1.0,
    topo_weight: float = 1.0,
    scale_weight: float = 1.0,
) -> np.ndarray:
    total = radial_bins + d2_bins + 3 + 3 + 2 + (1 if include_scale else 0)
    weights = np.ones(total, dtype=np.float32)
    idx = 0
    weights[idx : idx + radial_bins] *= radial_weight
    idx += radial_bins
    weights[idx : idx + d2_bins] *= d2_weight
    idx += d2_bins
    weights[idx : idx + 3] *= size_weight
    idx += 3
    weights[idx : idx + 3] *= extent_weight
    idx += 3
    weights[idx : idx + 2] *= topo_weight
    idx += 2
    if include_scale:
        weights[idx] *= scale_weight
    return weights


def _prepare_vectors(
    vectors: np.ndarray,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    weight_vec: Optional[np.ndarray],
    standardize: bool,
) -> np.ndarray:
    out = vectors.astype(np.float32, copy=False)
    if standardize and mean is not None and std is not None:
        out = (out - mean) / std
    if weight_vec is not None:
        out = out * weight_vec
    return out


def _prepare_vector(
    vector: np.ndarray,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    weight_vec: Optional[np.ndarray],
    standardize: bool,
) -> np.ndarray:
    out = vector.astype(np.float32, copy=False)
    if standardize and mean is not None and std is not None:
        out = (out - mean) / std
    if weight_vec is not None:
        out = out * weight_vec
    return out


def top_k_cosine(
    query: np.ndarray, embeddings: np.ndarray, top_k: int
) -> List[Tuple[int, float]]:
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    emb_norm = normalize_embeddings(embeddings)
    scores = emb_norm @ query_norm
    top_k = min(top_k, len(scores))
    idx = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in idx]


def top_k_l2(
    query: np.ndarray, embeddings: np.ndarray, top_k: int
) -> List[Tuple[int, float]]:
    distances = np.linalg.norm(embeddings - query, axis=1)
    top_k = min(top_k, len(distances))
    idx = np.argsort(distances)[:top_k]
    return [(int(i), float(distances[i])) for i in idx]


def prepare_for_search(
    embeddings: np.ndarray,
    query: np.ndarray,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    weight_vec: Optional[np.ndarray],
    standardize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    emb_p = _prepare_vectors(embeddings, mean, std, weight_vec, standardize)
    query_p = _prepare_vector(query, mean, std, weight_vec, standardize)
    return emb_p, query_p
