from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import trimesh

from ..mesh_utils import normalize_mesh


def _sample_points(mesh: trimesh.Trimesh, points: int, seed: int) -> np.ndarray:
    mesh_n = normalize_mesh(mesh)
    try:
        pts = mesh_n.sample(points)
    except Exception:
        pts = mesh_n.vertices
    if pts.size == 0:
        raise ValueError("Could not sample points from mesh")
    if len(pts) > points:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), size=points, replace=False)]
    return np.asarray(pts, dtype=np.float32)


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.nan_to_num(vector.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32)


def _hist(values: np.ndarray, bins: int, value_range: Tuple[float, float]) -> np.ndarray:
    out, _ = np.histogram(values, bins=bins, range=value_range, density=False)
    total = float(out.sum())
    if total > 0:
        out = out / total
    return out.astype(np.float32)


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _voxel_occupancy(mesh: trimesh.Trimesh, resolution: int, points: int, seed: int) -> np.ndarray:
    pts = _sample_points(mesh, points, seed)
    coords = np.clip(((pts + 1.0) * 0.5 * resolution).astype(int), 0, resolution - 1)
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    grid[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    return grid.reshape(-1)


@dataclass
class GraphSpectralEmbeddingBackend:
    eigen_count: int = 64
    max_vertices: int = 700

    name: str = "graph_spectral"

    def embed_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        mesh_n = normalize_mesh(mesh)
        vertices = mesh_n.vertices
        faces = mesh_n.faces
        if vertices.size == 0 or faces is None or faces.size == 0:
            return np.zeros(self.eigen_count, dtype=np.float32)

        if len(vertices) > self.max_vertices:
            chosen = np.linspace(0, len(vertices) - 1, self.max_vertices).astype(int)
            remap = {int(old): i for i, old in enumerate(chosen)}
            face_mask = np.isin(faces, chosen).all(axis=1)
            faces = np.array(
                [[remap[int(v)] for v in face] for face in faces[face_mask]],
                dtype=np.int64,
            )
            n = len(chosen)
        else:
            n = len(vertices)

        if n == 0 or faces.size == 0:
            return np.zeros(self.eigen_count, dtype=np.float32)

        adjacency = np.zeros((n, n), dtype=np.float32)
        for a, b, c in faces:
            if a < n and b < n and c < n:
                adjacency[a, b] = adjacency[b, a] = 1.0
                adjacency[b, c] = adjacency[c, b] = 1.0
                adjacency[c, a] = adjacency[a, c] = 1.0

        degree = adjacency.sum(axis=1)
        inv_sqrt = np.zeros_like(degree)
        mask = degree > 0
        inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
        laplacian = np.eye(n, dtype=np.float32) - (
            inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
        )
        eigenvalues = np.linalg.eigvalsh(laplacian)[: self.eigen_count]
        out = np.zeros(self.eigen_count, dtype=np.float32)
        out[: len(eigenvalues)] = eigenvalues.astype(np.float32)
        return _unit_vector(out)

    def signature(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "eigen_count": self.eigen_count,
            "max_vertices": self.max_vertices,
        }


@dataclass
class VoxelEmbeddingBackend:
    resolution: int = 16
    points: int = 4096
    seed: int = 123

    name: str = "voxel"

    def embed_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        return _unit_vector(_voxel_occupancy(mesh, self.resolution, self.points, self.seed))

    def signature(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "resolution": self.resolution,
            "points": self.points,
            "seed": self.seed,
        }


@dataclass
class PointCloudEmbeddingBackend:
    points: int = 2048
    bins: int = 32
    seed: int = 123

    name: str = "point_cloud"

    def embed_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        pts = _sample_points(mesh, self.points, self.seed)
        centered = pts - pts.mean(axis=0, keepdims=True)
        covariance = np.cov(centered.T).astype(np.float32)
        eigvals = np.sort(np.linalg.eigvalsh(covariance)).astype(np.float32)
        radial = np.linalg.norm(pts, axis=1)

        features = [
            _hist(pts[:, axis], self.bins, (-1.0, 1.0)) for axis in range(3)
        ]
        features.extend(
            [
                _hist(radial, self.bins, (0.0, 1.0)),
                eigvals,
                pts.mean(axis=0).astype(np.float32),
                pts.std(axis=0).astype(np.float32),
            ]
        )
        return _unit_vector(np.concatenate(features))

    def signature(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "points": self.points,
            "bins": self.bins,
            "seed": self.seed,
        }


@dataclass
class MultiViewProjectionEmbeddingBackend:
    points: int = 4096
    image_bins: int = 24
    seed: int = 123

    name: str = "multi_view_projection"

    def embed_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        pts = _sample_points(mesh, self.points, self.seed)
        views: Iterable[Tuple[int, int]] = ((0, 1), (0, 2), (1, 2))
        hists = []
        for a, b in views:
            hist, _, _ = np.histogram2d(
                pts[:, a],
                pts[:, b],
                bins=self.image_bins,
                range=[[-1.0, 1.0], [-1.0, 1.0]],
            )
            hist = hist.astype(np.float32)
            total = float(hist.sum())
            hists.append((hist / total if total > 0 else hist).reshape(-1))
        return _unit_vector(np.concatenate(hists))

    def signature(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "points": self.points,
            "image_bins": self.image_bins,
            "seed": self.seed,
        }


@dataclass
class PartFeatureEmbeddingBackend:
    normal_bins: int = 12

    name: str = "part_features"

    def embed_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        mesh_n = normalize_mesh(mesh)
        extents = mesh_n.bounding_box.extents.astype(np.float32)
        ext_sorted = np.sort(extents)
        short, middle, long = ext_sorted
        area = float(mesh_n.area)
        volume = float(abs(mesh_n.volume)) if mesh_n.is_watertight else float(mesh_n.volume)
        bbox_volume = float(np.prod(extents))
        fill_ratio = volume / bbox_volume if bbox_volume > 1e-8 else 0.0
        sphericity = 0.0
        if area > 1e-8 and volume > 0:
            sphericity = (np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0)) / area

        normals = np.asarray(mesh_n.face_normals, dtype=np.float32)
        normal_features = []
        if normals.size:
            for axis in range(3):
                normal_features.append(_hist(normals[:, axis], self.normal_bins, (-1.0, 1.0)))
        else:
            normal_features = [np.zeros(self.normal_bins, dtype=np.float32) for _ in range(3)]

        scalar = np.array(
            [
                short,
                middle,
                long,
                middle / short if short > 1e-8 else 0.0,
                long / middle if middle > 1e-8 else 0.0,
                long / short if short > 1e-8 else 0.0,
                area,
                volume,
                fill_ratio,
                sphericity,
                float(mesh_n.is_watertight),
                float(mesh_n.euler_number),
                float(len(mesh_n.vertices)),
                float(len(mesh_n.faces)) if mesh_n.faces is not None else 0.0,
            ],
            dtype=np.float32,
        )
        scalar = np.log1p(np.maximum(scalar, 0.0))
        return _unit_vector(np.concatenate([scalar, *normal_features]))

    def signature(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "normal_bins": self.normal_bins,
        }


@dataclass
class AutoencoderLatentEmbeddingBackend:
    resolution: int = 16
    latent_dim: int = 128
    points: int = 4096
    seed: int = 123

    name: str = "autoencoder_latent"

    def embed_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        voxels = _voxel_occupancy(mesh, self.resolution, self.points, self.seed)
        rng = np.random.default_rng(self.seed)
        projection = rng.normal(
            0.0,
            1.0 / np.sqrt(len(voxels)),
            size=(len(voxels), self.latent_dim),
        ).astype(np.float32)
        latent = np.tanh(voxels @ projection)
        return _unit_vector(latent.astype(np.float32))

    def signature(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "resolution": self.resolution,
            "latent_dim": self.latent_dim,
            "points": self.points,
            "seed": self.seed,
            "note": "deterministic random-projection latent baseline",
        }


@dataclass
class SemanticProfileEmbeddingBackend:
    points: int = 2048
    seed: int = 123

    name: str = "semantic_profile"

    def embed_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        mesh_n = normalize_mesh(mesh)
        pts = _sample_points(mesh, self.points, self.seed)
        extents = mesh_n.bounding_box.extents.astype(np.float32)
        short, middle, long = np.sort(extents)

        area = float(mesh_n.area)
        volume = float(abs(mesh_n.volume)) if mesh_n.is_watertight else float(mesh_n.volume)
        bbox_volume = float(np.prod(extents))
        fill_ratio = volume / bbox_volume if bbox_volume > 1e-8 else 0.0
        sphericity = 0.0
        if area > 1e-8 and volume > 0:
            sphericity = (np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0)) / area

        normals = np.asarray(mesh_n.face_normals, dtype=np.float32)
        z_aligned = 0.0
        axis_aligned = 0.0
        if normals.size:
            abs_normals = np.abs(normals)
            z_aligned = float(np.mean(abs_normals[:, 2] > 0.9))
            axis_aligned = float(np.mean(np.max(abs_normals, axis=1) > 0.9))

        radial = np.linalg.norm(pts, axis=1)
        radial_spread = float(np.std(radial))
        low_z = pts[:, 2] < np.quantile(pts[:, 2], 0.15)
        base_coverage = 0.0
        if np.any(low_z):
            base_xy = pts[low_z][:, :2]
            base_coverage = float(np.prod(base_xy.max(axis=0) - base_xy.min(axis=0)) / 4.0)

        flatness = _clamp01(1.0 - (short / (long + 1e-8)) * 4.0)
        elongation = _clamp01((long / (short + 1e-8) - 1.0) / 6.0)
        compactness = _clamp01(sphericity)
        hollowness = _clamp01(1.0 - fill_ratio)
        topology_complexity = _clamp01(abs(float(mesh_n.euler_number)) / 20.0)
        face_complexity = _clamp01(np.log1p(len(mesh_n.faces)) / 12.0 if mesh_n.faces is not None else 0.0)

        affordances = np.array(
            [
                flatness,  # plate-like
                elongation,  # rod/handle-like
                compactness,  # compact solid
                hollowness,  # cavity/open shell likelihood
                topology_complexity,  # holes/loops proxy
                _clamp01(z_aligned),  # flat/manufactured faces
                _clamp01(axis_aligned),  # rectilinear/mechanical
                _clamp01(base_coverage * 2.0),  # stand/support footprint
                _clamp01(flatness * axis_aligned),  # mounting plate
                _clamp01(topology_complexity * axis_aligned),  # bracket/connector
                _clamp01(hollowness * (1.0 - flatness)),  # container-like
                _clamp01(elongation * (1.0 - flatness)),  # handle/shaft-like
                _clamp01(compactness * (1.0 - axis_aligned)),  # round/organic
                _clamp01(axis_aligned * face_complexity),  # manufactured/mechanical
                _clamp01(flatness * topology_complexity),  # perforated plate
                _clamp01(elongation * topology_complexity),  # loop/clip-like
                _clamp01((1.0 - fill_ratio) * topology_complexity),  # through-hole likelihood
                _clamp01(base_coverage * axis_aligned),  # stable base
                _clamp01(radial_spread * 3.0),  # protrusions
                _clamp01(face_complexity),  # detail richness
            ],
            dtype=np.float32,
        )

        scalar = np.array(
            [
                short,
                middle,
                long,
                middle / short if short > 1e-8 else 0.0,
                long / middle if middle > 1e-8 else 0.0,
                long / short if short > 1e-8 else 0.0,
                area,
                volume,
                fill_ratio,
                sphericity,
                radial_spread,
                float(mesh_n.is_watertight),
                float(mesh_n.euler_number),
                float(len(mesh_n.vertices)),
                float(len(mesh_n.faces)) if mesh_n.faces is not None else 0.0,
            ],
            dtype=np.float32,
        )
        scalar = np.log1p(np.maximum(scalar, 0.0))
        return _unit_vector(np.concatenate([affordances, scalar]))

    def signature(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "points": self.points,
            "seed": self.seed,
            "profile_version": 1,
        }


@dataclass
class HybridEmbeddingBackend:
    backends: Sequence[object]
    weights: Sequence[float]
    name: str = "hybrid"
    component_names: List[str] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.backends) != len(self.weights):
            raise ValueError("Hybrid backend requires one weight per backend")
        active = [(backend, float(weight)) for backend, weight in zip(self.backends, self.weights) if weight > 0]
        if not active:
            raise ValueError("At least one hybrid component must have a positive weight")
        self.backends = [backend for backend, _ in active]
        self.weights = [weight for _, weight in active]
        self.component_names = [getattr(backend, "name", backend.__class__.__name__) for backend in self.backends]

    def embed_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        pieces = []
        for backend, weight in zip(self.backends, self.weights):
            try:
                vector = backend.embed_mesh(mesh)
            except Exception:
                vector = np.zeros(self._component_dim(backend), dtype=np.float32)
            pieces.append(_unit_vector(vector) * np.sqrt(weight))
        return _unit_vector(np.concatenate(pieces))

    def _component_dim(self, backend: object) -> int:
        if isinstance(backend, GraphSpectralEmbeddingBackend):
            return backend.eigen_count
        if isinstance(backend, VoxelEmbeddingBackend):
            return backend.resolution ** 3
        if isinstance(backend, PointCloudEmbeddingBackend):
            return (backend.bins * 4) + 9
        if isinstance(backend, MultiViewProjectionEmbeddingBackend):
            return 3 * (backend.image_bins ** 2)
        if isinstance(backend, PartFeatureEmbeddingBackend):
            return 14 + (3 * backend.normal_bins)
        if isinstance(backend, AutoencoderLatentEmbeddingBackend):
            return backend.latent_dim
        if isinstance(backend, SemanticProfileEmbeddingBackend):
            return 35
        model = getattr(backend, "_model", None)
        config = getattr(model, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
        raise ValueError(f"Cannot infer fallback dimension for {backend!r}")

    def signature(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "components": [
                {
                    "name": name,
                    "weight": weight,
                    "signature": backend.signature(),
                }
                for name, weight, backend in zip(self.component_names, self.weights, self.backends)
            ],
        }
