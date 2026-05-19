from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
try:
    import plotly.graph_objects as go
except Exception:
    go = None

from src.config import (
    BERT_DEVICE,
    BERT_MAX_LENGTH,
    BERT_SAMPLE_POINTS,
    DATA_DIR,
    DEFAULT_BERT_MODEL,
    DEFAULT_GEMINI_MODEL,
    D2_BINS,
    D2_PAIRS,
    D2_WEIGHT,
    EXTENT_WEIGHT,
    GEMINI_API_KEY_ENV,
    INCLUDE_SCALE,
    INDEX_DIR,
    LOG_FEATURES,
    POINTS,
    PREVIEW_POINTS,
    RADIAL_BINS,
    RADIAL_WEIGHT,
    SCALE_WEIGHT,
    SIZE_WEIGHT,
    STEP_DATA_DIR,
    TOPO_WEIGHT,
)
from src.data_loader import (
    list_mesh_files,
    list_step_files,
    load_mesh,
    load_mesh_from_bytes,
    load_step_mesh,
    mesh_display_name,
    parse_mesh_filename,
)
from src.embedding_backends import (
    AutoencoderLatentEmbeddingBackend,
    BertEmbeddingBackend,
    BRepMAEEmbeddingBackend,
    CADGCLEmbeddingBackend,
    GeminiEmbeddingBackend,
    GraphSpectralEmbeddingBackend,
    HybridEmbeddingBackend,
    LocalEmbeddingBackend,
    MultiViewProjectionEmbeddingBackend,
    PartFeatureEmbeddingBackend,
    PointCloudEmbeddingBackend,
    SemanticProfileEmbeddingBackend,
    VoxelEmbeddingBackend,
)
from src.index_store import build_index, load_index
from src.preview import mesh_preview_png
from src.similarity import (
    build_weight_vector,
    prepare_for_search,
    top_k_cosine,
    top_k_l2,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


BACKEND_DESCRIPTIONS = {
    "Hybrid: combine methods": (
        "Builds several local embeddings for each mesh, normalizes each one, applies "
        "your weights, concatenates them, and searches the combined vector."
    ),
    "Semantic: functional profile": (
        "Infers object-like affordances such as mounting, support, handle/shaft, "
        "container, ring/clip, perforated plate, and mechanical complexity, then "
        "searches by that semantic profile."
    ),
    "BERT (local transformer)": (
        "Converts each STL into a geometric text description, embeds that text with a "
        "local BERT model, then searches by vector similarity."
    ),
    "BRepMAE (STEP B-rep)": (
        "Indexes STEP/STP files with a masked B-rep descriptor encoder inspired by "
        "BRepMAE. Put STEP files in the step_data/ folder and rebuild this index."
    ),
    "CADGCL (STEP B-rep)": (
        "Indexes STEP/STP files with graph-style B-rep descriptors and deterministic "
        "contrastive augmentations inspired by CADGCL."
    ),
    "Experimental: graph spectral": (
        "Treats the mesh as a vertex-edge graph, computes Laplacian eigenvalues, and "
        "compares those topology/structure signatures."
    ),
    "Experimental: voxel grid": (
        "Samples the mesh surface into a 3D occupancy grid, then compares filled voxel "
        "patterns like a low-resolution 3D image."
    ),
    "Experimental: point cloud": (
        "Samples points from the mesh surface and builds coordinate, radial, and "
        "covariance features to compare overall 3D point distributions."
    ),
    "Experimental: multi-view projection": (
        "Projects sampled 3D points onto multiple 2D planes, builds view histograms, "
        "then compares those visual silhouettes."
    ),
    "Experimental: part features": (
        "Extracts compact geometric cues such as extents, aspect ratios, normals, "
        "surface area, volume, watertightness, and face/vertex counts."
    ),
    "Experimental: autoencoder latent": (
        "Builds a voxel grid and compresses it into a deterministic latent vector, "
        "similar to an autoencoder-style compressed representation."
    ),
    "Local shape features (fast, offline)": (
        "Uses handcrafted radial and point-pair distance histograms plus size, extent, "
        "and topology features. Fast, explainable, and fully offline."
    ),
    "Gemini (API, slower/costs)": (
        "Converts mesh geometry into text and sends it to Gemini embeddings, then "
        "searches using the returned API embedding vectors."
    ),
}


def _make_backend():
    backend_choice = st.sidebar.selectbox(
        "Embedding backend",
        [
            "Hybrid: combine methods",
            "Semantic: functional profile",
            "BERT (local transformer)",
            "BRepMAE (STEP B-rep)",
            "CADGCL (STEP B-rep)",
            "Experimental: graph spectral",
            "Experimental: voxel grid",
            "Experimental: point cloud",
            "Experimental: multi-view projection",
            "Experimental: part features",
            "Experimental: autoencoder latent",
            "Local shape features (fast, offline)",
            "Gemini (API, slower/costs)",
        ],
    )
    st.sidebar.info(BACKEND_DESCRIPTIONS[backend_choice])

    if backend_choice.startswith("Hybrid"):
        with st.sidebar.expander("Hybrid method weights", expanded=True):
            st.info("Weights of 0 disable a method. Changing weights requires rebuilding the index.")
            graph_weight = st.slider("Graph spectral weight", 0.0, 3.0, 1.0, step=0.1)
            voxel_weight = st.slider("Voxel grid weight", 0.0, 3.0, 1.0, step=0.1)
            point_weight = st.slider("Point cloud weight", 0.0, 3.0, 1.0, step=0.1)
            multiview_weight = st.slider("Multi-view weight", 0.0, 3.0, 1.0, step=0.1)
            part_weight = st.slider("Part features weight", 0.0, 3.0, 1.0, step=0.1)
            semantic_weight = st.slider("Semantic profile weight", 0.0, 3.0, 1.0, step=0.1)
            latent_weight = st.slider("Autoencoder latent weight", 0.0, 3.0, 0.5, step=0.1)
            include_bert = st.checkbox(
                "Include BERT",
                value=False,
                help="Adds the local BERT descriptor embedding. Slower and loads the BERT model.",
            )
            bert_weight = 0.0
            if include_bert:
                bert_weight = st.slider("BERT weight", 0.0, 3.0, 0.5, step=0.1)

        backends = [
            GraphSpectralEmbeddingBackend(),
            VoxelEmbeddingBackend(),
            PointCloudEmbeddingBackend(),
            MultiViewProjectionEmbeddingBackend(),
            PartFeatureEmbeddingBackend(),
            SemanticProfileEmbeddingBackend(),
            AutoencoderLatentEmbeddingBackend(),
        ]
        weights = [
            graph_weight,
            voxel_weight,
            point_weight,
            multiview_weight,
            part_weight,
            semantic_weight,
            latent_weight,
        ]
        if include_bert:
            try:
                backends.append(BertEmbeddingBackend())
                weights.append(bert_weight)
            except Exception as exc:
                st.sidebar.error(str(exc))
                st.stop()

        try:
            backend = HybridEmbeddingBackend(backends=backends, weights=weights)
        except ValueError as exc:
            st.sidebar.error(str(exc))
            st.stop()

        distance_metric = st.sidebar.selectbox(
            "Distance metric",
            [
                "Cosine",
                "L2 (standardized)",
                "L2",
            ],
        )
        return backend, None, distance_metric

    if backend_choice.startswith("Semantic"):
        with st.sidebar.expander("Semantic profile settings", expanded=True):
            st.info("Changing semantic profile settings requires rebuilding the index.")
            points = st.slider("Sample points", 512, 8192, 2048, step=256)
        backend = SemanticProfileEmbeddingBackend(points=points)
        distance_metric = st.sidebar.selectbox(
            "Distance metric",
            [
                "Cosine",
                "L2 (standardized)",
                "L2",
            ],
        )
        return backend, None, distance_metric

    if backend_choice.startswith("Experimental"):
        with st.sidebar.expander("Experimental backend settings", expanded=True):
            st.info("Changing these settings requires rebuilding the index.")

            if "graph spectral" in backend_choice:
                eigen_count = st.slider("Eigenvalue count", 16, 128, 64, step=8)
                max_vertices = st.slider("Max graph vertices", 200, 1500, 700, step=100)
                backend = GraphSpectralEmbeddingBackend(
                    eigen_count=eigen_count,
                    max_vertices=max_vertices,
                )
            elif "voxel grid" in backend_choice:
                resolution = st.slider("Voxel resolution", 8, 32, 16, step=4)
                points = st.slider("Sample points", 1024, 12000, 4096, step=512)
                backend = VoxelEmbeddingBackend(resolution=resolution, points=points)
            elif "point cloud" in backend_choice:
                points = st.slider("Sample points", 512, 8192, 2048, step=256)
                bins = st.slider("Histogram bins", 8, 64, 32, step=4)
                backend = PointCloudEmbeddingBackend(points=points, bins=bins)
            elif "multi-view projection" in backend_choice:
                points = st.slider("Sample points", 1024, 12000, 4096, step=512)
                image_bins = st.slider("Projection bins", 12, 48, 24, step=4)
                backend = MultiViewProjectionEmbeddingBackend(
                    points=points,
                    image_bins=image_bins,
                )
            elif "part features" in backend_choice:
                normal_bins = st.slider("Normal histogram bins", 6, 36, 12, step=3)
                backend = PartFeatureEmbeddingBackend(normal_bins=normal_bins)
            else:
                resolution = st.slider("Voxel resolution", 8, 32, 16, step=4)
                latent_dim = st.slider("Latent dimensions", 32, 512, 128, step=32)
                points = st.slider("Sample points", 1024, 12000, 4096, step=512)
                backend = AutoencoderLatentEmbeddingBackend(
                    resolution=resolution,
                    latent_dim=latent_dim,
                    points=points,
                )

        distance_metric = st.sidebar.selectbox(
            "Distance metric",
            [
                "Cosine",
                "L2 (standardized)",
                "L2",
            ],
        )
        return backend, None, distance_metric

    if backend_choice.startswith("BERT"):
        with st.sidebar.expander("BERT settings", expanded=True):
            st.caption("BERT embeds a generated geometric description for each mesh.")
            st.info("Changing BERT settings requires rebuilding the index.")
            model_name = st.text_input("BERT model", value=DEFAULT_BERT_MODEL)
            max_length = st.slider(
                "Max token length",
                min_value=64,
                max_value=512,
                value=BERT_MAX_LENGTH,
                step=32,
            )
            sample_points = st.slider(
                "Descriptor sample points",
                min_value=256,
                max_value=4096,
                value=BERT_SAMPLE_POINTS,
                step=256,
            )
            device = st.selectbox(
                "Device",
                ["cpu", "cuda"],
                index=1 if BERT_DEVICE == "cuda" else 0,
                help="Use cuda only if PyTorch can see a CUDA GPU.",
            )
        try:
            backend = BertEmbeddingBackend(
                model_name=model_name,
                max_length=max_length,
                sample_points=sample_points,
                device=device,
            )
        except Exception as exc:
            st.sidebar.error(str(exc))
            st.stop()
        distance_metric = st.sidebar.selectbox(
            "Distance metric",
            [
                "Cosine",
                "L2 (standardized)",
                "L2",
            ],
        )
        return backend, None, distance_metric

    if backend_choice.startswith("BRepMAE"):
        with st.sidebar.expander("BRepMAE settings", expanded=True):
            st.info("This backend indexes `.step` / `.stp` files from `step_data/`.")
            latent_dim = st.slider("Latent dimensions", 32, 512, 128, step=32)
            mask_ratio = st.slider("Mask ratio", 0.05, 0.8, 0.35, step=0.05)
            views = st.slider("Masked views", 2, 16, 8, step=1)
        backend = BRepMAEEmbeddingBackend(
            latent_dim=latent_dim,
            mask_ratio=mask_ratio,
            views=views,
        )
        distance_metric = st.sidebar.selectbox(
            "Distance metric",
            [
                "Cosine",
                "L2 (standardized)",
                "L2",
            ],
        )
        return backend, None, distance_metric

    if backend_choice.startswith("CADGCL"):
        with st.sidebar.expander("CADGCL settings", expanded=True):
            st.info("This backend indexes `.step` / `.stp` files from `step_data/`.")
            latent_dim = st.slider("Latent dimensions", 32, 512, 128, step=32)
            augmentations = st.slider("Contrastive augmentations", 2, 16, 6, step=1)
            drop_ratio = st.slider("Feature drop ratio", 0.0, 0.7, 0.2, step=0.05)
        backend = CADGCLEmbeddingBackend(
            latent_dim=latent_dim,
            augmentations=augmentations,
            drop_ratio=drop_ratio,
        )
        distance_metric = st.sidebar.selectbox(
            "Distance metric",
            [
                "Cosine",
                "L2 (standardized)",
                "L2",
            ],
        )
        return backend, None, distance_metric

    if backend_choice.startswith("Gemini"):
        st.sidebar.warning("Gemini backend calls the API for each mesh (may cost).")
        api_key = os.getenv(GEMINI_API_KEY_ENV, "")
        if not api_key:
            api_key = st.sidebar.text_input("Gemini API key", type="password")
        model = st.sidebar.text_input("Gemini embedding model", value=DEFAULT_GEMINI_MODEL)
        if not api_key:
            st.sidebar.error("Gemini key required to build embeddings.")
            st.stop()
        try:
            backend = GeminiEmbeddingBackend(api_key=api_key, model=model)
        except ImportError as exc:
            st.sidebar.error(str(exc))
            st.stop()
        distance_metric = st.sidebar.selectbox(
            "Distance metric",
            [
                "Cosine",
                "L2",
                "L2 (standardized)",
            ],
        )
        return backend, None, distance_metric

    with st.sidebar.expander("Local accuracy tuning"):
        st.caption("Higher values improve accuracy but increase build time.")
        st.info("Changing index parameters requires rebuilding the index.")
        st.markdown("Index parameters")
        radial_bins = st.slider(
            "Radial histogram bins", min_value=8, max_value=128, value=RADIAL_BINS, step=4
        )
        d2_bins = st.slider(
            "D2 histogram bins", min_value=8, max_value=128, value=D2_BINS, step=4
        )
        points = st.slider(
            "Sample points", min_value=512, max_value=8192, value=POINTS, step=256
        )
        d2_pairs = st.slider(
            "D2 pairs", min_value=1024, max_value=20000, value=D2_PAIRS, step=512
        )
        include_scale = st.checkbox(
            "Include scale feature",
            value=INCLUDE_SCALE,
            help="Adds overall size as a feature (useful if size matters).",
        )
        log_features = st.checkbox(
            "Log-scale size features",
            value=LOG_FEATURES,
            help="Compresses large ranges to balance feature influence.",
        )

        st.markdown("Search weights (no rebuild required)")
        radial_weight = st.slider(
            "Radial weight", min_value=0.0, max_value=3.0, value=RADIAL_WEIGHT, step=0.1
        )
        d2_weight = st.slider(
            "D2 weight", min_value=0.0, max_value=3.0, value=D2_WEIGHT, step=0.1
        )
        size_weight = st.slider(
            "Size/shape weight",
            min_value=0.0,
            max_value=3.0,
            value=SIZE_WEIGHT,
            step=0.1,
        )
        extent_weight = st.slider(
            "Extent weight",
            min_value=0.0,
            max_value=3.0,
            value=EXTENT_WEIGHT,
            step=0.1,
        )
        topo_weight = st.slider(
            "Topology weight",
            min_value=0.0,
            max_value=3.0,
            value=TOPO_WEIGHT,
            step=0.1,
        )
        scale_weight = SCALE_WEIGHT
        if include_scale:
            scale_weight = st.slider(
                "Scale weight",
                min_value=0.0,
                max_value=3.0,
                value=SCALE_WEIGHT,
                step=0.1,
            )

    distance_metric = st.sidebar.selectbox(
        "Distance metric",
        [
            "Cosine (weighted)",
            "L2 (weighted)",
            "L2 (standardized + weighted)",
        ],
    )

    backend = LocalEmbeddingBackend(
        radial_bins=radial_bins,
        d2_bins=d2_bins,
        points=points,
        d2_pairs=d2_pairs,
        include_scale=include_scale,
        log_features=log_features,
    )
    search_settings = {
        "radial_weight": radial_weight,
        "d2_weight": d2_weight,
        "size_weight": size_weight,
        "extent_weight": extent_weight,
        "topo_weight": topo_weight,
        "scale_weight": scale_weight,
    }
    return backend, search_settings, distance_metric


def _load_mesh_list() -> Tuple[List[Path], List[str]]:
    paths = list_mesh_files(DATA_DIR)
    rel_paths = [str(p.relative_to(DATA_DIR)) for p in paths]
    return paths, rel_paths


def _backend_data_dir(backend) -> Path:
    if hasattr(backend, "embed_path"):
        return STEP_DATA_DIR
    return DATA_DIR


def _load_data_list(backend) -> Tuple[List[Path], List[str], Path, str]:
    data_dir = _backend_data_dir(backend)
    if hasattr(backend, "embed_path"):
        paths = list_step_files(data_dir)
        label = "STEP/STP"
    else:
        paths = list_mesh_files(data_dir)
        label = "STL"
    rel_paths = [str(p.relative_to(data_dir)) for p in paths]
    return paths, rel_paths, data_dir, label


def _format_data_option(data_dir: Path, rel_path: str, label: str) -> str:
    path = data_dir / rel_path
    if label in {"STL", "STEP/STP"}:
        return mesh_display_name(path)
    return rel_path


def _result_title(path: Path, data_dir: Path) -> str:
    rel = path.relative_to(data_dir)
    if path.suffix.lower() in {".stl", ".step", ".stp"}:
        return mesh_display_name(path)
    return str(rel)


def _render_preview(mesh, caption: str):
    img = mesh_preview_png(mesh, points=PREVIEW_POINTS)
    st.image(img, caption=caption, use_column_width=True)


def _viewer_settings():
    with st.expander("3D viewer settings", expanded=False):
        cols = st.columns(3)
        with cols[0]:
            color_mode = st.selectbox(
                "Surface",
                ["Solid", "Height gradient"],
                index=0,
            )
            color = st.color_picker("Solid color", "#7da0c4")
        with cols[1]:
            view = st.selectbox(
                "Camera",
                ["Isometric", "Front", "Top", "Right"],
                index=0,
            )
            height = st.slider("Viewer height", 260, 720, 420, step=20)
        with cols[2]:
            show_axes = st.checkbox("Axes", value=False)
            show_bounds = st.checkbox("Bounds", value=True)
            flatshading = st.checkbox("Flat shading", value=True)
    return {
        "color_mode": color_mode,
        "color": color,
        "view": view,
        "height": height,
        "show_axes": show_axes,
        "show_bounds": show_bounds,
        "flatshading": flatshading,
    }


def _camera_for_view(view: str):
    cameras = {
        "Front": dict(x=0.0, y=-2.2, z=0.25),
        "Top": dict(x=0.0, y=0.0, z=2.4),
        "Right": dict(x=2.2, y=0.0, z=0.25),
        "Isometric": dict(x=1.55, y=-1.75, z=1.25),
    }
    return dict(eye=cameras.get(view, cameras["Isometric"]))


def _bbox_trace(vertices: np.ndarray):
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    corners = np.array(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
        ]
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    x, y, z = [], [], []
    for start, end in edges:
        x.extend([corners[start, 0], corners[end, 0], None])
        y.extend([corners[start, 1], corners[end, 1], None])
        z.extend([corners[start, 2], corners[end, 2], None])
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(color="rgba(40, 45, 55, 0.45)", width=3),
        hoverinfo="skip",
        showlegend=False,
    )


def _axis_traces(vertices: np.ndarray):
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins)) * 0.58 or 1.0
    axes = [
        ("X", np.array([radius, 0.0, 0.0]), "#d1495b"),
        ("Y", np.array([0.0, radius, 0.0]), "#2a9d8f"),
        ("Z", np.array([0.0, 0.0, radius]), "#277da1"),
    ]
    traces = []
    for name, vector, color in axes:
        end = center + vector
        traces.append(
            go.Scatter3d(
                x=[center[0], end[0]],
                y=[center[1], end[1]],
                z=[center[2], end[2]],
                mode="lines+text",
                text=["", name],
                textposition="top center",
                line=dict(color=color, width=5),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def _mesh_stats(mesh) -> str:
    extents = mesh.extents if mesh.extents is not None else np.zeros(3)
    return (
        f"{len(mesh.vertices):,} vertices | {len(mesh.faces):,} faces | "
        f"size {extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f}"
    )


def _mesh_to_plotly(mesh, title: str, viewer_settings=None):
    if go is None:
        raise RuntimeError("Plotly is not installed.")
    if mesh.faces is None or mesh.faces.size == 0:
        raise ValueError("Mesh has no faces.")

    viewer_settings = viewer_settings or {}
    vertices = mesh.vertices
    faces = mesh.faces

    max_faces = 50000
    if len(faces) > max_faces:
        stride = max(1, int(len(faces) / max_faces))
        faces = faces[::stride]

    x, y, z = vertices.T
    i, j, k = faces.T

    mesh_kwargs = {
        "x": x,
        "y": y,
        "z": z,
        "i": i,
        "j": j,
        "k": k,
        "opacity": 1.0,
        "flatshading": viewer_settings.get("flatshading", True),
        "lighting": dict(
            ambient=0.34,
            diffuse=0.78,
            specular=0.28,
            roughness=0.72,
            fresnel=0.12,
        ),
        "lightposition": dict(x=100, y=-140, z=180),
        "hovertemplate": "x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>",
    }
    if viewer_settings.get("color_mode") == "Height gradient":
        mesh_kwargs.update(
            intensity=z,
            colorscale="Viridis",
            showscale=False,
        )
    else:
        mesh_kwargs["color"] = viewer_settings.get("color", "#7da0c4")

    data = [go.Mesh3d(**mesh_kwargs)]
    if viewer_settings.get("show_bounds", True):
        data.append(_bbox_trace(vertices))
    if viewer_settings.get("show_axes", False):
        data.extend(_axis_traces(vertices))

    fig = go.Figure(
        data=data
    )
    fig.update_layout(
        title=title,
        height=viewer_settings.get("height", 420),
        scene_aspectmode="data",
        scene=dict(
            camera=_camera_for_view(viewer_settings.get("view", "Isometric")),
            xaxis=dict(
                visible=viewer_settings.get("show_axes", False),
                showbackground=False,
                showgrid=True,
                zeroline=False,
                title="X",
            ),
            yaxis=dict(
                visible=viewer_settings.get("show_axes", False),
                showbackground=False,
                showgrid=True,
                zeroline=False,
                title="Y",
            ),
            zaxis=dict(
                visible=viewer_settings.get("show_axes", False),
                showbackground=False,
                showgrid=True,
                zeroline=False,
                title="Z",
            ),
        ),
        margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        uirevision=title,
    )
    return fig

def _render_3d(mesh, title: str, viewer_settings=None):
    fig = _mesh_to_plotly(mesh, title, viewer_settings)
    st.caption(_mesh_stats(mesh))
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
            "modeBarButtonsToAdd": ["resetCameraDefault3d"],
        },
    )


@st.cache_data(show_spinner=False)
def _load_step_preview_mesh(path_text: str):
    return load_step_mesh(Path(path_text))


def _load_preview_mesh(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".stl":
        return load_mesh(path)
    if suffix in {".step", ".stp"}:
        return _load_step_preview_mesh(str(path))
    return None


def _store_uploaded_query_file(upload, allowed_extensions) -> Path:
    suffix = Path(upload.name).suffix.lower()
    if suffix not in allowed_extensions:
        expected = ", ".join(sorted(allowed_extensions))
        raise ValueError(f"Expected one of: {expected}")

    data = upload.getvalue()
    digest = hashlib.sha256(data).hexdigest()[:16]
    safe_stem = "".join(
        ch if ch.isalnum() or ch in {" ", "_", "-"} else "_"
        for ch in Path(upload.name).stem
    ).strip() or "uploaded_query"
    upload_dir = INDEX_DIR / "uploaded_queries"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{safe_stem}_{digest}{suffix}"
    path.write_bytes(data)
    return path


def _build_index_with_progress(backend, mesh_paths: List[Path]):
    progress = st.progress(0.0)
    status = st.empty()

    def _cb(i: int, total: int):
        progress.progress(min(i / total, 1.0))
        status.text(f"Indexing {i}/{total}")

    meta = build_index(INDEX_DIR, backend, mesh_paths, progress_cb=_cb)
    status.text("Index ready")
    return meta


def _read_uploaded_text(files) -> str:
    chunks = []
    for file in files or []:
        raw = file.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="ignore")
        text = text.strip()
        if text:
            chunks.append(f"File: {file.name}\n{text}")
    return "\n\n".join(chunks)


def _build_rag_query(user_text: str, file_context: str) -> str:
    combined = f"{user_text}\n{file_context}".lower()
    vocabulary = {
        "mounting or bracket": ["bracket", "mount", "holder", "support", "fixture", "base"],
        "holes or fasteners": ["hole", "screw", "bolt", "fastener", "slot", "perforated"],
        "cylindrical or round": ["cylinder", "round", "tube", "pipe", "shaft", "ring", "wheel"],
        "flat or plate-like": ["flat", "plate", "panel", "thin", "rectangular"],
        "container or shell": ["cup", "bowl", "container", "case", "cover", "shell", "housing"],
        "handle or grip": ["handle", "grip", "lever", "knob"],
        "long or elongated": ["long", "rod", "bar", "rail", "elongated"],
        "compact or blocky": ["compact", "block", "cube", "box"],
        "mechanical detail": ["gear", "tooth", "teeth", "hinge", "joint", "thread", "connector"],
    }
    inferred_terms = [
        label
        for label, words in vocabulary.items()
        if any(word in combined for word in words)
    ]

    parts = [
        "3D mesh geometry descriptor search query.",
        "Match indexed STL models by shape, function, proportions, topology, and visible product affordances.",
        "Important geometry words: flat, elongated, compact, roundish, open, watertight, holes, base, shaft, ring, shell, bracket, handle, connector, mechanical.",
    ]
    if inferred_terms:
        parts.append("Inferred product features: " + ", ".join(inferred_terms) + ".")
    if file_context:
        parts.append("Uploaded product notes:\n" + file_context[:30000])
    if user_text:
        parts.append("User request:\n" + user_text.strip())
    parts.append(
        "Return closest 3D STL matches even when exact product names differ; prefer geometry and functional affordance similarity."
    )
    return "\n\n".join(parts)


def _search_index(
    query_emb,
    embeddings,
    mean,
    std,
    backend,
    search_settings,
    distance_metric: str,
    top_k: int,
):
    weight_vec = None
    if backend.name == "local" and search_settings is not None:
        weight_vec = build_weight_vector(
            backend.radial_bins,
            backend.d2_bins,
            backend.include_scale,
            radial_weight=search_settings["radial_weight"],
            d2_weight=search_settings["d2_weight"],
            size_weight=search_settings["size_weight"],
            extent_weight=search_settings["extent_weight"],
            topo_weight=search_settings["topo_weight"],
            scale_weight=search_settings["scale_weight"],
        )

    standardize = distance_metric in {"L2 (standardized)", "L2 (standardized + weighted)"}
    embeddings_p, query_p = prepare_for_search(
        embeddings, query_emb, mean, std, weight_vec, standardize
    )

    if distance_metric.startswith("Cosine"):
        return top_k_cosine(query_p, embeddings_p, top_k)
    return top_k_l2(query_p, embeddings_p, top_k)


def _render_results(
    results,
    stored_paths,
    data_dir: Path,
    distance_metric: str,
    show_3d: bool,
    viewer_settings=None,
):
    st.subheader("Results")
    for idx, score in results:
        path = Path(stored_paths[idx])
        rel = path.relative_to(data_dir)
        title = _result_title(path, data_dir)
        try:
            mesh = _load_preview_mesh(path)
        except Exception as exc:
            mesh = None
            preview_error = str(exc)
        else:
            preview_error = ""

        if distance_metric.startswith("Cosine"):
            score_display = min(max(score, -1.0), 1.0)
            distance = 1.0 - score_display

        cols = st.columns([2, 3] if show_3d else [1, 3])
        with cols[0]:
            if mesh is not None:
                if show_3d:
                    try:
                        _render_3d(mesh, title, viewer_settings)
                    except Exception:
                        img = mesh_preview_png(mesh, points=PREVIEW_POINTS)
                        st.image(img, use_column_width=True)
                else:
                    img = mesh_preview_png(mesh, points=PREVIEW_POINTS)
                    st.image(img, use_column_width=True)
            else:
                st.write("Preview unavailable")
                if preview_error:
                    st.caption(preview_error)
        with cols[1]:
            st.markdown(f"**{title}**")
            if path.suffix.lower() in {".stl", ".step", ".stp"}:
                meta = parse_mesh_filename(path)
                if meta.get("part_number") and meta.get("value"):
                    st.write(f"Part number: `{meta['part_number']}`")
                    st.write(f"Value: `{meta['value']}`")
            st.caption(str(rel))
            if distance_metric.startswith("Cosine"):
                st.write(f"Similarity (cosine): {score_display:.6f}")
                st.write(f"Cosine distance: {distance:.6f}")
            elif distance_metric.startswith("L2 (standardized"):
                st.write(f"Distance (standardized L2): {score:.6f}")
            else:
                st.write(f"Distance (L2): {score:.6f}")


def main():
    st.set_page_config(
        page_title="3D Mesh Similarity Search",
        layout="wide",
    )
    st.title("3D Mesh Similarity Search")
    st.caption(f"STL directory: {DATA_DIR} | STEP directory: {STEP_DATA_DIR}")
    with st.expander("Quick start", expanded=True):
        st.markdown(
            """
            1. Put your `.stl` files inside `data/`, or `.step` / `.stp` files inside `step_data/`.
            2. Choose an embedding backend in the sidebar.
            3. Build the index once, then run searches.
            """
        )

    backend, search_settings, distance_metric = _make_backend()
    data_paths, rel_paths, active_data_dir, active_file_label = _load_data_list(backend)

    if not data_paths:
        st.error(f"No `{active_file_label}` files found in {active_data_dir}.")
        st.markdown(
            "Create the folder at the project root and place files inside it."
        )
        return

    st.sidebar.header("Project Status")
    st.sidebar.write(f"{active_file_label} files found: {len(data_paths)}")
    embeddings, stored_paths, meta, stale, mean, std = load_index(
        INDEX_DIR, backend, data_paths
    )

    if stale:
        st.sidebar.warning("Index missing or out of date.")
        if st.sidebar.button("Build / Rebuild index"):
            with st.spinner("Building index..."):
                meta = _build_index_with_progress(backend, data_paths)
            embeddings, stored_paths, meta, stale, mean, std = load_index(
                INDEX_DIR, backend, data_paths
            )
    else:
        if meta and "count" in meta:
            st.sidebar.write(
                f"Index ready: {meta.get('count')} items, dim {meta.get('dim')}"
            )
            if meta.get("count", 0) < len(data_paths):
                st.sidebar.warning("Index is partial (some meshes failed to embed).")

    if embeddings is None or stale:
        st.info("Build the index to enable search.")
        return

    st.subheader("Search")
    st.markdown(
        "Choose a query file, upload a new part, or use the RAG product agent to search from text/files."
    )
    show_3d = st.checkbox("Show 3D previews", value=False)
    if show_3d and go is None:
        st.warning("3D previews require Plotly. Install with `pip install plotly`.")
        show_3d = False
    viewer_settings = _viewer_settings() if show_3d else None
    upload_mode = "Upload STEP/STP" if hasattr(backend, "embed_path") else "Upload STL"
    query_modes = ["Choose from dataset", upload_mode, "RAG product agent"]
    query_mode = st.radio("Query type", query_modes)
    query_mesh = None
    query_path = None
    query_upload_path = None
    text_query = ""
    uploaded_text = ""

    if query_mode == "Choose from dataset":
        selection = st.selectbox(
            f"Select a {active_file_label} file",
            rel_paths,
            format_func=lambda rel: _format_data_option(
                active_data_dir, rel, active_file_label
            ),
        )
        query_path = active_data_dir / selection
        if hasattr(backend, "embed_path"):
            st.write(f"Query file: `{selection}`")
            try:
                query_mesh = _load_preview_mesh(query_path)
                if query_mesh is not None:
                    if show_3d:
                        _render_3d(
                            query_mesh,
                            f"Query: {_format_data_option(active_data_dir, selection, active_file_label)}",
                            viewer_settings,
                        )
                    else:
                        _render_preview(
                            query_mesh,
                            f"Query: {_format_data_option(active_data_dir, selection, active_file_label)}",
                        )
            except Exception as exc:
                st.warning(f"STEP preview unavailable: {exc}")
        else:
            try:
                query_mesh = load_mesh(query_path)
                if show_3d:
                    _render_3d(
                        query_mesh,
                        f"Query: {_format_data_option(active_data_dir, selection, active_file_label)}",
                        viewer_settings,
                    )
                else:
                    _render_preview(
                        query_mesh,
                        f"Query: {_format_data_option(active_data_dir, selection, active_file_label)}",
                    )
            except Exception as exc:
                st.error(f"Failed to load query mesh: {exc}")
                return
    elif query_mode == upload_mode:
        st.info("Upload mode always returns the top 5 most similar results.")
        upload_types = ["step", "stp"] if hasattr(backend, "embed_path") else ["stl"]
        upload = st.file_uploader(f"Upload {active_file_label}", type=upload_types)
        if upload is not None:
            if hasattr(backend, "embed_path"):
                try:
                    query_upload_path = _store_uploaded_query_file(
                        upload, {".step", ".stp"}
                    )
                except Exception as exc:
                    st.error(f"Failed to save uploaded STEP file: {exc}")
                    return

                try:
                    query_mesh = _load_preview_mesh(query_upload_path)
                    if show_3d:
                        _render_3d(
                            query_mesh,
                            f"Query (uploaded): {upload.name}",
                            viewer_settings,
                        )
                    else:
                        _render_preview(query_mesh, f"Query (uploaded): {upload.name}")
                except Exception as exc:
                    st.warning(f"STEP preview unavailable: {exc}")
            else:
                try:
                    query_mesh = load_mesh_from_bytes(upload.getvalue())
                    if show_3d:
                        _render_3d(
                            query_mesh,
                            f"Query (uploaded): {upload.name}",
                            viewer_settings,
                        )
                    else:
                        _render_preview(query_mesh, f"Query (uploaded): {upload.name}")
                except Exception as exc:
                    st.error(f"Failed to load uploaded STL file: {exc}")
                    return
    else:
        if not hasattr(backend, "embed_text"):
            st.warning(
                "The RAG product agent needs a text-capable backend. Use Gemini, BERT, BRepMAE, or CADGCL, "
                "then build the index for that backend."
            )
        st.markdown("Upload product notes/specs and ask for the kind of item you want to find.")
        rag_files = st.file_uploader(
            "Upload product file",
            type=["txt", "md", "csv", "json", "log"],
            accept_multiple_files=True,
        )
        uploaded_text = _read_uploaded_text(rag_files)
        text_query = st.text_area(
            "Describe the product or ask the agent",
            placeholder=(
                "Example: Find a compact mounting bracket with a flat base, two screw holes, "
                "and a raised cylindrical support."
            ),
            height=140,
        )
        if uploaded_text:
            with st.expander("Uploaded text preview"):
                st.text(uploaded_text[:5000])

    top_k = None
    if query_mode == upload_mode:
        top_k = 5
        st.info("Upload mode returns the top 5 most similar results.")
    else:
        top_k = st.slider("Top K results", min_value=1, max_value=30, value=10)
    run = st.button("Search")

    if run:
        if query_mode == "RAG product agent":
            if not hasattr(backend, "embed_text"):
                st.error("Switch to Gemini or BERT to search from text/files.")
                return
            if not text_query.strip() and not uploaded_text:
                st.error("Describe the product or upload a product file first.")
                return
            with st.spinner("Reading context, embedding text query, and searching..."):
                rag_query = _build_rag_query(text_query, uploaded_text)
                query_emb = backend.embed_text(rag_query)
                results = _search_index(
                    query_emb,
                    embeddings,
                    mean,
                    std,
                    backend,
                    search_settings,
                    distance_metric,
                    top_k,
                )
            _render_results(
                results,
                stored_paths,
                active_data_dir,
                distance_metric,
                show_3d,
                viewer_settings,
            )
            return

        if query_mesh is None and query_path is None and query_upload_path is None:
            st.error("Provide a query file first.")
            return

        with st.spinner("Embedding query and searching..."):
            if hasattr(backend, "embed_path") and (
                query_path is not None or query_upload_path is not None
            ):
                query_emb = backend.embed_path(query_upload_path or query_path)
            else:
                query_emb = backend.embed_mesh(query_mesh)
            results = _search_index(
                query_emb,
                embeddings,
                mean,
                std,
                backend,
                search_settings,
                distance_metric,
                top_k + 1,
            )

        if query_path is not None:
            filtered = []
            for idx, score in results:
                if stored_paths[idx] == str(query_path):
                    continue
                filtered.append((idx, score))
            results = filtered[:top_k]
        else:
            results = results[:top_k]

        _render_results(
            results,
            stored_paths,
            active_data_dir,
            distance_metric,
            show_3d,
            viewer_settings,
        )


if __name__ == "__main__":
    main()
