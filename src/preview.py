from __future__ import annotations

import io
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from .mesh_utils import normalize_mesh


def mesh_preview_png(
    mesh: trimesh.Trimesh, points: int = 1024, size: int = 256
) -> bytes:
    mesh_n = normalize_mesh(mesh)
    try:
        pts = mesh_n.sample(points)
    except Exception:
        pts = mesh_n.vertices

    if len(pts) > points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), size=points, replace=False)
        pts = pts[idx]

    fig = plt.figure(figsize=(size / 100.0, size / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.8)
    ax.view_init(elev=20, azim=35)
    ax.set_axis_off()
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()
