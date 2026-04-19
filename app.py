from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st
import numpy as np
try:
    import plotly.graph_objects as go
except Exception:
    go = None

from src.config import (
    DATA_DIR,
    DEFAULT_GEMINI_MODEL,
    D2_BINS,
    D2_PAIRS,
    D2_WEIGHT,
    EXTENT_WEIGHT,
    GEMINI_API_KEY_ENV,
    INCLUDE_SCALE,
    INDEX_DIR,
    LOG_
)

# Security configuration
SAFE_EXTENSIONS = {'.obj', '.ply', '.stl', '.off', '.gltf', '.glb'}
MAX_VERTICES = 1000000
MAX_COORDINATE = 1e6
MIN_COORDINATE = -1e6


def validate_mesh_geometry(vertices: np.ndarray, faces: np.ndarray) -> Tuple[bool, str]:
    """Validate 3D mesh geometry for safety and sanity."""
    if vertices is None or faces is None:
        return False, "Mesh data cannot be None"
    
    # Check vertex count
    if len(vertices) > MAX_VERTICES:
        return False, f"Vertex count ({len(vertices)}) exceeds maximum allowed ({MAX_VERTICES})"
    
    # Check coordinate bounds
    if len(vertices) > 0:
        if np.any(vertices > MAX_COORDINATE):
            return False, f"Coordinates exceed maximum value ({MAX_COORDINATE})"
        if np.any(vertices < MIN_COORDINATE):
            return False, f"Coordinates below minimum value ({MIN_COORDINATE})"
        
        # Check for NaN or infinite values
        if np.any(np.isnan(vertices)):
            return False, "Mesh contains NaN values"
        if np.any(np.isinf(vertices)):
            return False, "Mesh contains infinite values"
    
    # Check face indices
    if len(faces) > 0:
        max_index = np.max(faces)
        if max_index >= len(vertices):
            return False, f"Face index {max_index} exceeds vertex count {len(vertices)}"
        if np.any(faces < 0):
            return False, "Face indices cannot be negative"
    
    return True, "Mesh validation passed"


def sanitize_file_path(file_path: str, allowed_dirs: List[Path]) -> Optional[Path]:
    """Sanitize file path and restrict to allowed directories."""
    try:
        # Convert to absolute path
        abs_path = Path(file_path).resolve()
        
        # Check file extension
        if abs_path.suffix.lower() not in SAFE_EXTENSIONS:
            return None
        
        # Check if path is within allowed directories
        is_allowed = False
        for allowed_dir in allowed_dirs:
            allowed_dir = allowed_dir.resolve()
            try:
                if abs_path.is_relative_to(allowed_dir):
                    is_allowed = True
                    break
            except ValueError:
                continue
        
        if not is_allowed:
            return None
        
        # Check if file exists and is a file
        if not abs_path.exists() or not abs_path.is_file():
            return None
        
        return abs_path
    except Exception:
        return None


def get_gemini_api_key() -> Optional[str]:
    """Safely retrieve Gemini API key from server-side environment."""
    # Never expose API key in client-side code
    api_key = os.environ.get(GEMINI_API_KEY_ENV)
    
    if not api_key:
        st.error("Gemini API key not configured. Please set it in server environment variables.")
        return None
    
    # Basic validation of API key format
    if not re.match(r'^AIza[0-9A-Za-z-_]{35}$', api_key):
        st.error("Invalid Gemini API key format.")
        return None
    
    return api_key


# Example usage in the app
if __name__ == "__main__":
    st.title("3D Mesh Similarity Search")
    
    # Get API key securely
    api_key = get_gemini_api_key()
    if not api_key:
        st.stop()
    
    # File upload with validation
    uploaded_file = st.file_uploader("Upload 3D mesh", type=[ext.lstrip('.') for ext in SAFE_EXTENSIONS])
    
    if uploaded_file:
        # Save to temporary location in allowed directory
        temp_dir = DATA_DIR / "uploads"
        temp_dir.mkdir(exist_ok=True)
        
        temp_path = temp_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Sanitize and validate file path
        sanitized_path = sanitize_file_path(str(temp_path), [DATA_DIR, temp_dir])
        
        if not sanitized_path:
            st.error("Invalid file path or type. Please upload a valid 3D mesh file.")
            st.stop()
        
        # Load and validate mesh
        try:
            import trimesh
            mesh = trimesh.load(str(sanitized_path))
            
            # Validate geometry
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.faces)
            
            is_valid, message = validate_mesh_geometry(vertices, faces)
            
            if not is_valid:
                st.error(f"Mesh validation failed: {message}")
                st.stop()
            
            st.success(f"Mesh loaded successfully: {len(vertices)} vertices, {len(faces)} faces")
            
            # Process the validated mesh...
            
        except Exception as e:
            st.error(f"Error loading mesh: {str(e)}")
            
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
