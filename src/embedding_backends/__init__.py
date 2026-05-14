from __future__ import annotations

from .base import EmbeddingBackend
from .bert import BertEmbeddingBackend
from .brep import BRepMAEEmbeddingBackend, CADGCLEmbeddingBackend
from .experimental import (
    AutoencoderLatentEmbeddingBackend,
    GraphSpectralEmbeddingBackend,
    HybridEmbeddingBackend,
    MultiViewProjectionEmbeddingBackend,
    PartFeatureEmbeddingBackend,
    PointCloudEmbeddingBackend,
    SemanticProfileEmbeddingBackend,
    VoxelEmbeddingBackend,
)
from .gemini import GeminiEmbeddingBackend
from .local import LocalEmbeddingBackend

__all__ = [
    "AutoencoderLatentEmbeddingBackend",
    "BertEmbeddingBackend",
    "BRepMAEEmbeddingBackend",
    "CADGCLEmbeddingBackend",
    "EmbeddingBackend",
    "GeminiEmbeddingBackend",
    "GraphSpectralEmbeddingBackend",
    "HybridEmbeddingBackend",
    "LocalEmbeddingBackend",
    "MultiViewProjectionEmbeddingBackend",
    "PartFeatureEmbeddingBackend",
    "PointCloudEmbeddingBackend",
    "SemanticProfileEmbeddingBackend",
    "VoxelEmbeddingBackend",
]
