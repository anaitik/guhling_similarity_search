from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

DEFAULT_BACKEND = "local"
DEFAULT_BERT_MODEL = os.getenv("BERT_MODEL", "google/bert_uncased_L-2_H-128_A-2")
BERT_MAX_LENGTH = int(os.getenv("BERT_MAX_LENGTH", "128"))
BERT_SAMPLE_POINTS = int(os.getenv("BERT_SAMPLE_POINTS", "1024"))
BERT_DEVICE = os.getenv("BERT_DEVICE", "cpu")

RADIAL_BINS = int(os.getenv("RADIAL_BINS", "32"))
D2_BINS = int(os.getenv("D2_BINS", "32"))
POINTS = int(os.getenv("POINTS", "2048"))
D2_PAIRS = int(os.getenv("D2_PAIRS", "4096"))

PREVIEW_POINTS = int(os.getenv("PREVIEW_POINTS", "1024"))

def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

RADIAL_WEIGHT = float(os.getenv("RADIAL_WEIGHT", "1.0"))
D2_WEIGHT = float(os.getenv("D2_WEIGHT", "1.0"))
SIZE_WEIGHT = float(os.getenv("SIZE_WEIGHT", "1.0"))
EXTENT_WEIGHT = float(os.getenv("EXTENT_WEIGHT", "1.0"))
TOPO_WEIGHT = float(os.getenv("TOPO_WEIGHT", "1.0"))
SCALE_WEIGHT = float(os.getenv("SCALE_WEIGHT", "1.0"))
INCLUDE_SCALE = _bool_env("INCLUDE_SCALE", False)
LOG_FEATURES = _bool_env("LOG_FEATURES", False)

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL_ENV = "GEMINI_EMBEDDING_MODEL"
DEFAULT_GEMINI_MODEL = os.getenv(GEMINI_MODEL_ENV, "models/embedding-001")
