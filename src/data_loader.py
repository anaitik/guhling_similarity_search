from __future__ import annotations

from pathlib import Path
from typing import List

import io

import trimesh


def list_mesh_files(data_dir: Path) -> List[Path]:
    return sorted(Path(data_dir).rglob("*.stl"))


def list_step_files(data_dir: Path) -> List[Path]:
    extensions = {".step", ".stp"}
    return sorted(p for p in Path(data_dir).rglob("*") if p.suffix.lower() in extensions)


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    if mesh.is_empty or mesh.vertices.size == 0:
        raise ValueError(f"Empty mesh: {path}")
    return mesh


def load_mesh_from_bytes(data: bytes) -> trimesh.Trimesh:
    file_obj = io.BytesIO(data)
    mesh = trimesh.load(file_obj, file_type="stl", force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    if mesh.is_empty or mesh.vertices.size == 0:
        raise ValueError("Empty mesh from upload")
    return mesh
