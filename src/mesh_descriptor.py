from __future__ import annotations

from typing import List

import numpy as np
import trimesh

from .mesh_utils import normalize_mesh


def mesh_descriptor_text(mesh: trimesh.Trimesh, sample_points: int = 1024, seed: int = 123) -> str:
    """Build a stable geometric text description for language-model embedders."""
    mesh_n = normalize_mesh(mesh)
    extents = mesh_n.bounding_box.extents.astype(float)
    ext_sorted = np.sort(extents)
    x, y, z = ext_sorted

    ratio_xy = y / x if x > 1e-8 else 0.0
    ratio_yz = z / y if y > 1e-8 else 0.0
    elongation = z / x if x > 1e-8 else 0.0

    area = float(mesh_n.area)
    volume = float(abs(mesh_n.volume)) if mesh_n.is_watertight else float(mesh_n.volume)

    sphericity = 0.0
    if area > 1e-8 and volume > 0:
        sphericity = (np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0)) / area

    faces = int(mesh_n.faces.shape[0]) if mesh_n.faces is not None else 0
    verts = int(mesh_n.vertices.shape[0])

    try:
        points = mesh_n.sample(sample_points)
    except Exception:
        points = mesh_n.vertices

    radial_quantiles = np.zeros(5, dtype=float)
    if points.size:
        if len(points) > sample_points:
            rng = np.random.default_rng(seed)
            points = points[rng.choice(len(points), size=sample_points, replace=False)]
        radial = np.linalg.norm(points, axis=1)
        radial_quantiles = np.quantile(radial, [0.1, 0.25, 0.5, 0.75, 0.9])

    tags: List[str] = []
    if elongation > 3.0:
        tags.append("elongated")
    if x / z < 0.25:
        tags.append("flat")
    if abs(ratio_xy - 1.0) < 0.2 and abs(ratio_yz - 1.0) < 0.2:
        tags.append("compact")
    if sphericity > 0.7:
        tags.append("roundish")
    if not mesh_n.is_watertight:
        tags.append("open")
    if not tags:
        tags.append("irregular")

    radial_text = ", ".join(f"{value:.4f}" for value in radial_quantiles)
    return (
        "3D mesh geometry descriptor. "
        f"Surface area {area:.4f}. "
        f"Volume {volume:.4f}. "
        f"Extents x {extents[0]:.4f}, y {extents[1]:.4f}, z {extents[2]:.4f}. "
        f"Sorted aspect ratios short-to-middle {ratio_xy:.3f}, middle-to-long {ratio_yz:.3f}. "
        f"Elongation {elongation:.3f}. "
        f"Sphericity {sphericity:.3f}. "
        f"Radial distribution quantiles {radial_text}. "
        f"Vertices {verts}, faces {faces}. "
        f"Watertight {mesh_n.is_watertight}. "
        f"Shape tags: {', '.join(tags)}."
    )
