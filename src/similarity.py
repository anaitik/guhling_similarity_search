from __future__ import annotations

from typing import List, Tuple

import numpy as np


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def top_k_cosine(
    query: np.ndarray, embeddings_norm: np.ndarray, top_k: int
) -> List[Tuple[int, float]]:
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    scores = embeddings_norm @ query_norm
    top_k = min(top_k, len(scores))
    idx = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in idx]
