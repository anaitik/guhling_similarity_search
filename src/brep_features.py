from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


ENTITY_PATTERN = re.compile(r"=\s*([A-Z0-9_]+)\s*\(", re.IGNORECASE)

SURFACE_TYPES = [
    "PLANE",
    "CYLINDRICAL_SURFACE",
    "CONICAL_SURFACE",
    "SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE",
    "B_SPLINE_SURFACE",
    "BEZIER_SURFACE",
    "SURFACE_OF_REVOLUTION",
    "SURFACE_OF_LINEAR_EXTRUSION",
]

CURVE_TYPES = [
    "LINE",
    "CIRCLE",
    "ELLIPSE",
    "B_SPLINE_CURVE",
    "BEZIER_CURVE",
    "POLYLINE",
]

TOPOLOGY_TYPES = [
    "ADVANCED_FACE",
    "FACE_BOUND",
    "FACE_OUTER_BOUND",
    "EDGE_CURVE",
    "ORIENTED_EDGE",
    "VERTEX_POINT",
    "CLOSED_SHELL",
    "OPEN_SHELL",
    "MANIFOLD_SOLID_BREP",
    "SHELL_BASED_SURFACE_MODEL",
]


@dataclass(frozen=True)
class BRepFeatures:
    vector: np.ndarray
    counts: Dict[str, int]
    text: str


def _read_step_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="ignore")


def _entity_counts(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for match in ENTITY_PATTERN.finditer(text):
        name = match.group(1).upper()
        counts[name] = counts.get(name, 0) + 1
    return counts


def _count_refs(text: str) -> int:
    return len(re.findall(r"#\d+", text))


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _log_counts(counts: Dict[str, int], names: Iterable[str]) -> List[float]:
    return [float(np.log1p(counts.get(name, 0))) for name in names]


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.nan_to_num(vector.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32)


def brep_features_from_step(path: Path) -> BRepFeatures:
    text = _read_step_text(path)
    counts = _entity_counts(text)

    faces = counts.get("ADVANCED_FACE", 0)
    edges = counts.get("EDGE_CURVE", 0)
    oriented_edges = counts.get("ORIENTED_EDGE", 0)
    vertices = counts.get("VERTEX_POINT", 0)
    bounds = counts.get("FACE_BOUND", 0) + counts.get("FACE_OUTER_BOUND", 0)
    shells = counts.get("CLOSED_SHELL", 0) + counts.get("OPEN_SHELL", 0)
    refs = _count_refs(text)

    surface_total = sum(counts.get(name, 0) for name in SURFACE_TYPES)
    curve_total = sum(counts.get(name, 0) for name in CURVE_TYPES)
    plane = counts.get("PLANE", 0)
    cylindrical = counts.get("CYLINDRICAL_SURFACE", 0)
    spline_surface = counts.get("B_SPLINE_SURFACE", 0) + counts.get("BEZIER_SURFACE", 0)
    circle = counts.get("CIRCLE", 0)
    line = counts.get("LINE", 0)
    spline_curve = counts.get("B_SPLINE_CURVE", 0) + counts.get("BEZIER_CURVE", 0)

    scalar = np.array(
        [
            np.log1p(faces),
            np.log1p(edges),
            np.log1p(oriented_edges),
            np.log1p(vertices),
            np.log1p(bounds),
            np.log1p(shells),
            np.log1p(refs),
            _ratio(edges, faces),
            _ratio(vertices, edges),
            _ratio(bounds, faces),
            _ratio(oriented_edges, edges),
            _ratio(surface_total, faces),
            _ratio(curve_total, edges),
            _ratio(plane, surface_total),
            _ratio(cylindrical, surface_total),
            _ratio(spline_surface, surface_total),
            _ratio(circle, curve_total),
            _ratio(line, curve_total),
            _ratio(spline_curve, curve_total),
            float(counts.get("MANIFOLD_SOLID_BREP", 0) > 0),
            float(counts.get("OPEN_SHELL", 0) > 0),
            float(counts.get("CLOSED_SHELL", 0) > 0),
        ],
        dtype=np.float32,
    )

    vector = np.concatenate(
        [
            scalar,
            np.array(_log_counts(counts, SURFACE_TYPES), dtype=np.float32),
            np.array(_log_counts(counts, CURVE_TYPES), dtype=np.float32),
            np.array(_log_counts(counts, TOPOLOGY_TYPES), dtype=np.float32),
        ]
    )
    vector = _unit_vector(vector)

    summary = (
        "B-rep STEP descriptor. "
        f"Faces {faces}. Edges {edges}. Vertices {vertices}. Bounds {bounds}. "
        f"Shells {shells}. Planar faces {plane}. Cylindrical surfaces {cylindrical}. "
        f"Spline surfaces {spline_surface}. Lines {line}. Circles {circle}. "
        f"Spline curves {spline_curve}. Manifold solid {counts.get('MANIFOLD_SOLID_BREP', 0) > 0}. "
        f"Open shell {counts.get('OPEN_SHELL', 0) > 0}."
    )
    return BRepFeatures(vector=vector, counts=counts, text=summary)
