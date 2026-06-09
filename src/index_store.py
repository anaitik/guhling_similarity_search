from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import MAX_INDEX_MESH_BYTES, MAX_INDEX_MESH_MB
from .data_loader import load_mesh, parse_mesh_filename


def _index_files(index_dir: Path, backend_name: str) -> Dict[str, Path]:
    return {
        "embeddings": index_dir / f"{backend_name}_embeddings.npy",
        "paths": index_dir / f"{backend_name}_paths.json",
        "meta": index_dir / f"{backend_name}_meta.json",
        "mean": index_dir / f"{backend_name}_mean.npy",
        "std": index_dir / f"{backend_name}_std.npy",
    }


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.nan_to_num(vector.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32)


def _write_index_files(
    index_dir: Path,
    backend,
    embeddings: List[np.ndarray],
    stored_paths: List[str],
    errors: List[Dict[str, str]],
    extra_meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    if not embeddings:
        details = "; ".join(
            f"{Path(str(error.get('path', 'unknown'))).name}: {error.get('error', 'unknown error')}"
            for error in errors[:3]
        )
        message = "No embeddings were created."
        if details:
            message += f" First errors: {details}"
        else:
            message += " Check data and dependencies."
        raise RuntimeError(message)

    files = _index_files(index_dir, backend.name)
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
    if extra_meta:
        meta.update(extra_meta)
    files["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _oversized_stl_error(path: Path) -> Optional[str]:
    size = path.stat().st_size
    if size <= MAX_INDEX_MESH_BYTES:
        return None
    return (
        f"Skipped oversized STL ({size / 1024 / 1024:.1f} MB; "
        f"limit {MAX_INDEX_MESH_MB:g} MB). Set MAX_INDEX_MESH_MB "
        "higher to include it."
    )


def _path_context(path: Path) -> str:
    meta = parse_mesh_filename(path)
    parts = []
    if meta.get("part_number"):
        parts.append(f"part number {meta['part_number']}")
    if meta.get("value"):
        parts.append(f"value {meta['value']}")

    skipped_folder_names = {"data", "deepseek_data", "step_data"}
    for item in [path.parent.name, path.stem]:
        if item == path.anchor or item.lower() in skipped_folder_names:
            continue
        cleaned = (
            item.replace("_", " ")
            .replace("-", " ")
            .strip()
        )
        if cleaned:
            parts.append(cleaned)
    return " ".join(parts)


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

    error_paths = {
        str(error.get("path"))
        for error in meta.get("errors", [])
        if isinstance(error, dict) and error.get("path")
    }
    missing_paths = current_set - stored_set
    new_paths = missing_paths - error_paths
    if new_paths:
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

    embeddings: List[np.ndarray] = []
    stored_paths: List[str] = []
    errors: List[Dict[str, str]] = []

    total = len(data_paths)
    for i, path in enumerate(data_paths, start=1):
        try:
            if hasattr(backend, "embed_path"):
                emb = backend.embed_path(path)
            else:
                oversized_error = _oversized_stl_error(path)
                if oversized_error:
                    raise ValueError(oversized_error)
                mesh = load_mesh(path)
                if hasattr(backend, "embed_mesh_with_context"):
                    emb = backend.embed_mesh_with_context(mesh, _path_context(path))
                else:
                    emb = backend.embed_mesh(mesh)
            embeddings.append(emb)
            stored_paths.append(str(path))
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
        if progress_cb:
            progress_cb(i, total)

    return _write_index_files(index_dir, backend, embeddings, stored_paths, errors)


def build_hybrid_index_from_component_indexes(
    index_dir: Path,
    backend,
    data_paths: List[Path],
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Optional[Dict[str, object]]:
    if getattr(backend, "name", "") != "hybrid":
        return None

    component_indexes = []
    for component in backend.backends:
        embeddings, stored_paths, meta, stale, _, _ = load_index(
            index_dir, component, data_paths
        )
        if stale or embeddings is None or stored_paths is None:
            return None
        component_indexes.append(
            {
                "backend": component,
                "embeddings": {
                    path: embeddings[idx]
                    for idx, path in enumerate(stored_paths)
                },
                "dim": int(embeddings.shape[1]),
                "meta": meta,
            }
        )

    embeddings: List[np.ndarray] = []
    stored_paths: List[str] = []
    errors: List[Dict[str, str]] = []

    total = len(data_paths)
    for i, path in enumerate(data_paths, start=1):
        path_text = str(path)
        try:
            oversized_error = _oversized_stl_error(path)
            if oversized_error:
                raise ValueError(oversized_error)

            pieces = []
            available = 0
            for component_index, weight in zip(component_indexes, backend.weights):
                vector = component_index["embeddings"].get(path_text)
                if vector is None:
                    vector = np.zeros(component_index["dim"], dtype=np.float32)
                else:
                    available += 1
                pieces.append(_unit_vector(vector) * np.sqrt(float(weight)))

            if available == 0:
                raise ValueError("No cached component embeddings available for this mesh")

            embeddings.append(_unit_vector(np.concatenate(pieces)))
            stored_paths.append(path_text)
        except Exception as exc:
            errors.append({"path": path_text, "error": str(exc)})
        if progress_cb:
            progress_cb(i, total)

    return _write_index_files(
        index_dir,
        backend,
        embeddings,
        stored_paths,
        errors,
        extra_meta={
            "source": "component_indexes",
            "component_counts": {
                getattr(item["backend"], "name", item["backend"].__class__.__name__): item[
                    "meta"
                ].get("count")
                for item in component_indexes
            },
        },
    )
