from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import io

import numpy as np
import trimesh


_PART_VALUE_RE = re.compile(
    r"^\s*(?P<part_number>\d+)[\s_-]+(?P<value>\d+(?:\.\d+)?)\s*$"
)


def parse_mesh_filename(path: Path) -> Dict[str, Optional[str]]:
    """Parse filenames like '1001 10.005.stl' into searchable metadata."""
    stem = Path(path).stem
    match = _PART_VALUE_RE.match(stem)
    if not match:
        return {
            "part_number": None,
            "value": None,
            "display": stem,
        }

    part_number = match.group("part_number")
    value = match.group("value")
    return {
        "part_number": part_number,
        "value": value,
        "display": f"Part {part_number} | Value {value}",
    }


def mesh_sort_key(path: Path) -> Tuple[int, str, float, str]:
    meta = parse_mesh_filename(path)
    part_number = meta.get("part_number")
    value = meta.get("value")
    if part_number is not None and value is not None:
        return (0, f"{int(part_number):012d}", float(value), path.name.lower())
    return (1, "", 0.0, str(path).lower())


def mesh_display_name(path: Path) -> str:
    meta = parse_mesh_filename(path)
    display = meta["display"] or Path(path).stem
    return f"{display} ({Path(path).name})"


def list_mesh_files(data_dir: Path) -> List[Path]:
    return sorted(Path(data_dir).rglob("*.stl"), key=mesh_sort_key)


def list_step_files(data_dir: Path) -> List[Path]:
    extensions = {".step", ".stp"}
    return sorted(
        (p for p in Path(data_dir).rglob("*") if p.suffix.lower() in extensions),
        key=mesh_sort_key,
    )


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    if mesh.is_empty or mesh.vertices.size == 0:
        raise ValueError(f"Empty mesh: {path}")
    return mesh


def _vertex_to_tuple(vertex) -> Tuple[float, float, float]:
    if all(hasattr(vertex, attr) for attr in ("x", "y", "z")):
        return (float(vertex.x), float(vertex.y), float(vertex.z))
    return (float(vertex[0]), float(vertex[1]), float(vertex[2]))


def load_step_mesh(
    path: Path,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.2,
) -> trimesh.Trimesh:
    try:
        import cadquery as cq
    except ImportError as exc:
        raise RuntimeError(
            "STEP 3D preview requires CadQuery. Install it with `pip install cadquery`."
        ) from exc

    model = cq.importers.importStep(str(path))
    shape = model.val() if hasattr(model, "val") else model
    vertices, faces = shape.tessellate(linear_deflection, angular_deflection)

    mesh = trimesh.Trimesh(
        vertices=np.array([_vertex_to_tuple(v) for v in vertices], dtype=np.float64),
        faces=np.array(faces, dtype=np.int64),
        process=False,
    )
    if mesh.is_empty or mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"Empty STEP mesh preview: {path}")
    return mesh


def load_mesh_from_bytes(data: bytes) -> trimesh.Trimesh:
    file_obj = io.BytesIO(data)
    mesh = trimesh.load(file_obj, file_type="stl", force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    if mesh.is_empty or mesh.vertices.size == 0:
        raise ValueError("Empty mesh from upload")
    return mesh
