from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

DEFAULT_BACKEND = "local"

RADIAL_BINS = int(os.getenv("RADIAL_BINS", "32"))
D2_BINS = int(os.getenv("D2_BINS", "32"))
POINTS = int(os.getenv("POINTS", "2048"))
D2_PAIRS = int(os.getenv("D2_PAIRS", "4096"))

PREVIEW_POINTS = int(os.getenv("PREVIEW_POINTS", "1024"))

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL_ENV = "GEMINI_EMBEDDING_MODEL"
DEFAULT_GEMINI_MODEL = os.getenv(GEMINI_MODEL_ENV, "models/embedding-001")
