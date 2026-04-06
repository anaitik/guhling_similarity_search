from __future__ import annotations

import numpy as np
import trimesh

from .mesh_utils import normalize_mesh


def compute_shape_features(
    mesh: trimesh.Trimesh,
    radial_bins: int = 32,
    d2_bins: int = 32,
    points: int = 2048,
    d2_pairs: int = 4096,
    seed: int = 123,
) -> np.ndarray:
    mesh_n = normalize_mesh(mesh)

    try:
        pts = mesh_n.sample(points)
    except Exception:
        pts = mesh_n.vertices

    if pts.size == 0:
        raise ValueError("Could not sample points from mesh")

    if len(pts) > points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), size=points, replace=False)
        pts = pts[idx]

    radial = np.linalg.norm(pts, axis=1)
    radial = np.clip(radial, 0.0, 1.0)
    radial_hist, _ = np.histogram(
        radial, bins=radial_bins, range=(0.0, 1.0), density=True
    )

    rng = np.random.default_rng(seed)
    idx1 = rng.integers(0, len(pts), size=d2_pairs)
    idx2 = rng.integers(0, len(pts), size=d2_pairs)
    d2 = np.linalg.norm(pts[idx1] - pts[idx2], axis=1)
    d2 = np.clip(d2, 0.0, 2.0)
    d2_hist, _ = np.histogram(d2, bins=d2_bins, range=(0.0, 2.0), density=True)

    area = float(mesh_n.area)
    volume = float(abs(mesh_n.volume)) if mesh_n.is_watertight else float(mesh_n.volume)
    extents = mesh_n.bounding_box.extents.astype(float)

    sphericity = 0.0
    if area > 1e-8 and volume > 0:
        sphericity = (np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0)) / area

    faces = float(mesh_n.faces.shape[0]) if mesh_n.faces is not None else 0.0
    verts = float(mesh_n.vertices.shape[0])

    feature = np.concatenate(
        [
            radial_hist,
            d2_hist,
            np.array([area, volume, sphericity], dtype=float),
            extents,
            np.array([faces, verts], dtype=float),
        ]
    )
    feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
    return feature.astype(np.float32)
