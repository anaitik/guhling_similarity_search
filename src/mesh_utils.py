from __future__ import annotations

import numpy as np
import trimesh


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    if mesh.vertices.size == 0:
        raise ValueError("Mesh has no vertices")

    mesh.apply_translation(-mesh.centroid)

    try:
        mesh.apply_transform(mesh.principal_inertia_transform)
    except Exception:
        # If inertia alignment fails, keep original orientation.
        pass

    max_radius = np.linalg.norm(mesh.vertices, axis=1).max()
    if max_radius > 0:
        mesh.apply_scale(1.0 / max_radius)
    return mesh
