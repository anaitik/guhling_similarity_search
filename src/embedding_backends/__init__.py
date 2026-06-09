from __future__ import annotations

from .base import EmbeddingBackend
from .bert import BertEmbeddingBackend
from .brep import BRepMAEEmbeddingBackend, CADGCLEmbeddingBackend
from .deepseek import DeepSeekEmbeddingBackend
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
from .local import LocalEmbeddingBackend

__all__ = [
    "AutoencoderLatentEmbeddingBackend",
    "BertEmbeddingBackend",
    "BRepMAEEmbeddingBackend",
    "CADGCLEmbeddingBackend",
    "DeepSeekEmbeddingBackend",
    "EmbeddingBackend",
    "GraphSpectralEmbeddingBackend",
    "HybridEmbeddingBackend",
    "LocalEmbeddingBackend",
    "MultiViewProjectionEmbeddingBackend",
    "PartFeatureEmbeddingBackend",
    "PointCloudEmbeddingBackend",
    "SemanticProfileEmbeddingBackend",
    "VoxelEmbeddingBackend",
]
