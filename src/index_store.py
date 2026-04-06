from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .data_loader import load_mesh


def _index_files(index_dir: Path, backend_name: str) -> Dict[str, Path]:
    return {
        "embeddings": index_dir / f"{backend_name}_embeddings.npy",
        "paths": index_dir / f"{backend_name}_paths.json",
        "meta": index_dir / f"{backend_name}_meta.json",
        "mean": index_dir / f"{backend_name}_mean.npy",
        "std": index_dir / f"{backend_name}_std.npy",
    }


def load_index(
    index_dir: Path,
    backend,
    data_paths: List[Path],
) -> Tuple[
    Optional[np.ndarray],
    Optional[List[str]],
    Dict[str, object],
    bool,
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    files = _index_files(index_dir, backend.name)
    if not files["embeddings"].exists() or not files["paths"].exists():
        return None, None, {"status": "missing"}, True, None, None

    meta = {}
    if files["meta"].exists():
        meta = json.loads(files["meta"].read_text(encoding="utf-8"))

    signature_ok = meta.get("signature") == backend.signature()
    if not signature_ok:
        return None, None, {"status": "signature_mismatch", "meta": meta}, True, None, None

    stored_paths = json.loads(files["paths"].read_text(encoding="utf-8"))
    current_paths = [str(p) for p in data_paths]
    current_set = set(current_paths)
    stored_set = set(stored_paths)
    if not stored_set.issubset(current_set):
        return None, None, {"status": "paths_mismatch", "meta": meta}, True, None, None

    if not files["mean"].exists() or not files["std"].exists():
        return None, None, {"status": "stats_missing", "meta": meta}, True, None, None

    embeddings = np.load(files["embeddings"])
    mean = np.load(files["mean"])
    std = np.load(files["std"])
    return embeddings, stored_paths, meta, False, mean, std


def build_index(
    index_dir: Path,
    backend,
    data_paths: List[Path],
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, object]:
    index_dir.mkdir(parents=True, exist_ok=True)
    files = _index_files(index_dir, backend.name)

    embeddings: List[np.ndarray] = []
    stored_paths: List[str] = []
    errors: List[Dict[str, str]] = []

    total = len(data_paths)
    for i, path in enumerate(data_paths, start=1):
        try:
            mesh = load_mesh(path)
            emb = backend.embed_mesh(mesh)
            embeddings.append(emb)
            stored_paths.append(str(path))
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
        if progress_cb:
            progress_cb(i, total)

    if not embeddings:
        raise RuntimeError("No embeddings were created. Check data and dependencies.")

    embeddings_arr = np.vstack(embeddings).astype(np.float32)
    mean = embeddings_arr.mean(axis=0).astype(np.float32)
    std = embeddings_arr.std(axis=0).astype(np.float32)
    std = np.where(std == 0, 1.0, std).astype(np.float32)

    np.save(files["embeddings"], embeddings_arr)
    np.save(files["mean"], mean)
    np.save(files["std"], std)
    files["paths"].write_text(json.dumps(stored_paths, indent=2), encoding="utf-8")

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "count": len(stored_paths),
        "dim": int(embeddings_arr.shape[1]),
        "signature": backend.signature(),
        "errors": errors[:50],
    }
    files["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
