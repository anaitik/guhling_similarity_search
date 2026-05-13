from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

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
    TOPO_WEIGHT,
)
from src.data_loader import list_mesh_files, load_mesh, load_mesh_from_bytes
from src.embedding_backends import (
    AutoencoderLatentEmbeddingBackend,
    BertEmbeddingBackend,
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


def _render_preview(mesh, caption: str):
    img = mesh_preview_png(mesh, points=PREVIEW_POINTS)
    st.image(img, caption=caption, use_column_width=True)


def _mesh_to_plotly(mesh, title: str):
    if go is None:
        raise RuntimeError("Plotly is not installed.")
    if mesh.faces is None or mesh.faces.size == 0:
        raise ValueError("Mesh has no faces.")

    vertices = mesh.vertices
    faces = mesh.faces

    max_faces = 50000
    if len(faces) > max_faces:
        stride = max(1, int(len(faces) / max_faces))
        faces = faces[::stride]

    x, y, z = vertices.T
    i, j, k = faces.T

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color="#7da0c4",
                opacity=1.0,
                flatshading=True,
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene_aspectmode="data",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig

def _render_3d(mesh, title: str):
    fig = _mesh_to_plotly(mesh, title)
    st.plotly_chart(fig, use_container_width=True)


def _build_index_with_progress(backend, mesh_paths: List[Path]):
    progress = st.progress(0.0)
    status = st.empty()

    def _cb(i: int, total: int):
        progress.progress(min(i / total, 1.0))
        status.text(f"Indexing {i}/{total}")

    meta = build_index(INDEX_DIR, backend, mesh_paths, progress_cb=_cb)
    status.text("Index ready")
    return meta


def main():
    st.set_page_config(
        page_title="3D Mesh Similarity Search",
        layout="wide",
    )
    st.title("3D Mesh Similarity Search")
    st.caption(f"Data directory: {DATA_DIR}")
    with st.expander("Quick start", expanded=True):
        st.markdown(
            """
            1. Put your `.stl` files inside the `data/` folder at the project root.
            2. Choose an embedding backend in the sidebar.
            3. Build the index once, then run searches.
            """
        )

    backend, search_settings, distance_metric = _make_backend()
    mesh_paths, rel_paths = _load_mesh_list()

    if not mesh_paths:
        st.error("No `.stl` files found in the data directory.")
        st.markdown(
            "Create `data/` at the project root and place STL files inside it."
        )
        return

    st.sidebar.header("Project Status")
    st.sidebar.write(f"Meshes found: {len(mesh_paths)}")
    embeddings, stored_paths, meta, stale, mean, std = load_index(
        INDEX_DIR, backend, mesh_paths
    )

    if stale:
        st.sidebar.warning("Index missing or out of date.")
        if st.sidebar.button("Build / Rebuild index"):
            with st.spinner("Building index..."):
                meta = _build_index_with_progress(backend, mesh_paths)
            embeddings, stored_paths, meta, stale, mean, std = load_index(
                INDEX_DIR, backend, mesh_paths
            )
    else:
        if meta and "count" in meta:
            st.sidebar.write(
                f"Index ready: {meta.get('count')} items, dim {meta.get('dim')}"
            )
            if meta.get("count", 0) < len(mesh_paths):
                st.sidebar.warning("Index is partial (some meshes failed to embed).")

    if embeddings is None or stale:
        st.info("Build the index to enable search.")
        return

    st.subheader("Search")
    st.markdown(
        "Choose a query mesh from your dataset or upload a new STL to find similar items."
    )
    show_3d = st.checkbox("Show 3D previews", value=False)
    if show_3d and go is None:
        st.warning("3D previews require Plotly. Install with `pip install plotly`.")
        show_3d = False
    query_mode = st.radio("Query type", ["Choose from dataset", "Upload STL"])
    query_mesh = None
    query_path = None

    if query_mode == "Choose from dataset":
        selection = st.selectbox("Select a mesh", rel_paths)
        query_path = DATA_DIR / selection
        try:
            query_mesh = load_mesh(query_path)
            if show_3d:
                _render_3d(query_mesh, f"Query: {selection}")
            else:
                _render_preview(query_mesh, f"Query: {selection}")
        except Exception as exc:
            st.error(f"Failed to load query mesh: {exc}")
            return
    else:
        st.info("Upload mode always returns the top 5 most similar results.")
        upload = st.file_uploader("Upload STL", type=["stl"])
        if upload is not None:
            try:
                query_mesh = load_mesh_from_bytes(upload.read())
                if show_3d:
                    _render_3d(query_mesh, "Query (uploaded)")
                else:
                    _render_preview(query_mesh, "Query (uploaded)")
            except Exception as exc:
                st.error(f"Failed to load uploaded mesh: {exc}")
                return

    top_k = None
    if query_mode == "Upload STL":
        top_k = 5
        st.info("Upload mode returns the top 5 most similar results.")
    else:
        top_k = st.slider("Top K results", min_value=1, max_value=30, value=10)
    run = st.button("Search")

    if run:
        if query_mesh is None:
            st.error("Provide a query mesh first.")
            return

        with st.spinner("Embedding query and searching..."):
            query_emb = backend.embed_mesh(query_mesh)

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
                results = top_k_cosine(query_p, embeddings_p, top_k + 1)
            else:
                results = top_k_l2(query_p, embeddings_p, top_k + 1)

        if query_path is not None:
            filtered = []
            for idx, score in results:
                if stored_paths[idx] == str(query_path):
                    continue
                filtered.append((idx, score))
            results = filtered[:top_k]
        else:
            results = results[:top_k]

        st.subheader("Results")
        for idx, score in results:
            path = Path(stored_paths[idx])
            rel = path.relative_to(DATA_DIR)
            try:
                mesh = load_mesh(path)
            except Exception:
                mesh = None

            if distance_metric.startswith("Cosine"):
                score_display = min(max(score, -1.0), 1.0)
                distance = 1.0 - score_display

            cols = st.columns([1, 3])
            with cols[0]:
                if mesh is not None:
                    if show_3d:
                        try:
                            _render_3d(mesh, f"{rel}")
                        except Exception:
                            img = mesh_preview_png(mesh, points=PREVIEW_POINTS)
                            st.image(img, use_column_width=True)
                    else:
                        img = mesh_preview_png(mesh, points=PREVIEW_POINTS)
                        st.image(img, use_column_width=True)
                else:
                    st.write("Preview unavailable")
            with cols[1]:
                st.markdown(f"**{rel}**")
                if distance_metric.startswith("Cosine"):
                    st.write(f"Similarity (cosine): {score_display:.6f}")
                    st.write(f"Cosine distance: {distance:.6f}")
                elif distance_metric.startswith("L2 (standardized"):
                    st.write(f"Distance (standardized L2): {score:.6f}")
                else:
                    st.write(f"Distance (L2): {score:.6f}")


if __name__ == "__main__":
    main()
