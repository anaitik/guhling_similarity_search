from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

from src.config import (
    DATA_DIR,
    DEFAULT_GEMINI_MODEL,
    D2_BINS,
    D2_PAIRS,
    GEMINI_API_KEY_ENV,
    INDEX_DIR,
    POINTS,
    PREVIEW_POINTS,
    RADIAL_BINS,
)
from src.data_loader import list_mesh_files, load_mesh, load_mesh_from_bytes
from src.embedding_backends import GeminiEmbeddingBackend, LocalEmbeddingBackend
from src.index_store import build_index, load_index
from src.preview import mesh_preview_png
from src.similarity import top_k_cosine

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _make_backend():
    backend_choice = st.sidebar.selectbox(
        "Embedding backend",
        ["Local (fast, offline)", "Gemini (API, slower/costs)"],
    )
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
            return GeminiEmbeddingBackend(api_key=api_key, model=model)
        except ImportError as exc:
            st.sidebar.error(str(exc))
            st.stop()

    return LocalEmbeddingBackend(
        radial_bins=RADIAL_BINS, d2_bins=D2_BINS, points=POINTS, d2_pairs=D2_PAIRS
    )


def _load_mesh_list() -> Tuple[List[Path], List[str]]:
    paths = list_mesh_files(DATA_DIR)
    rel_paths = [str(p.relative_to(DATA_DIR)) for p in paths]
    return paths, rel_paths


def _render_preview(mesh, caption: str):
    img = mesh_preview_png(mesh, points=PREVIEW_POINTS)
    st.image(img, caption=caption, use_column_width=True)


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

    backend = _make_backend()
    mesh_paths, rel_paths = _load_mesh_list()

    if not mesh_paths:
        st.error("No `.stl` files found in the data directory.")
        st.markdown(
            "Create `data/` at the project root and place STL files inside it."
        )
        return

    st.sidebar.header("Project Status")
    st.sidebar.write(f"Meshes found: {len(mesh_paths)}")
    embeddings_norm, stored_paths, meta, stale = load_index(
        INDEX_DIR, backend, mesh_paths
    )

    if stale:
        st.sidebar.warning("Index missing or out of date.")
        if st.sidebar.button("Build / Rebuild index"):
            with st.spinner("Building index..."):
                meta = _build_index_with_progress(backend, mesh_paths)
            embeddings_norm, stored_paths, meta, stale = load_index(
                INDEX_DIR, backend, mesh_paths
            )
    else:
        if meta and "count" in meta:
            st.sidebar.write(
                f"Index ready: {meta.get('count')} items, dim {meta.get('dim')}"
            )
            if meta.get("count", 0) < len(mesh_paths):
                st.sidebar.warning("Index is partial (some meshes failed to embed).")

    if embeddings_norm is None or stale:
        st.info("Build the index to enable search.")
        return

    st.subheader("Search")
    st.markdown(
        "Choose a query mesh from your dataset or upload a new STL to find similar items."
    )
    query_mode = st.radio("Query type", ["Choose from dataset", "Upload STL"])
    query_mesh = None
    query_path = None

    if query_mode == "Choose from dataset":
        selection = st.selectbox("Select a mesh", rel_paths)
        query_path = DATA_DIR / selection
        try:
            query_mesh = load_mesh(query_path)
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
            results = top_k_cosine(query_emb, embeddings_norm, top_k + 1)

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

            cols = st.columns([1, 3])
            with cols[0]:
                if mesh is not None:
                    img = mesh_preview_png(mesh, points=PREVIEW_POINTS)
                    st.image(img, use_column_width=True)
                else:
                    st.write("Preview unavailable")
            with cols[1]:
                st.markdown(f"**{rel}**")
                st.write(f"Similarity score: {score:.4f}")


if __name__ == "__main__":
    main()
