# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
import numpy as np

def ensure_vector_dimensions(vector: np.ndarray, target_dim: int) -> np.ndarray:
    """Ensure a vector has the target dimension using zero-padding or truncation"""
    current_dim = vector.shape[0]
    
    if current_dim == target_dim:
        return vector
    elif current_dim < target_dim:
        # Pad with zeros
        return np.pad(vector, (0, target_dim - current_dim))
    else:
        # Truncate to target dimension
        return vector[:target_dim]

def standardize_vector(vector: np.ndarray, target_dim: int = 128) -> np.ndarray:
    """Standardize vector to target dimension with normalization"""
    vector = ensure_vector_dimensions(vector, target_dim)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def standardize_vector_min_max(vector: np.ndarray, target_dim: int = 128) -> np.ndarray:
    """Standardize vector to target dimension using min-max scaling"""
    vector = ensure_vector_dimensions(vector, target_dim)
    min_val = np.min(vector)
    max_val = np.max(vector)
    return (vector - min_val) / (max_val - min_val) if max_val != min_val else vector

def standardize_vector_log(vector: np.ndarray, target_dim: int = 128) -> np.ndarray:
    """Standardize vector to target dimension using log scaling"""
    vector = ensure_vector_dimensions(vector, target_dim)
    return np.log(vector + 1)

def standardize_vector_sqrt(vector: np.ndarray, target_dim: int = 128) -> np.ndarray:
    """Standardize vector to target dimension using sqrt scaling"""
    vector = ensure_vector_dimensions(vector, target_dim)
    return np.sqrt(vector)

def standardize_vector_tanh(vector: np.ndarray, target_dim: int = 128) -> np.ndarray:
    """Standardize vector to target dimension using tanh scaling"""
    vector = ensure_vector_dimensions(vector, target_dim)
    return np.tanh(vector)

def standardize_vector_l1(vector: np.ndarray, target_dim: int = 128) -> np.ndarray:
    """Standardize vector to target dimension using L1 normalization"""
    vector = ensure_vector_dimensions(vector, target_dim)
    norm = np.sum(np.abs(vector))
    return vector / norm if norm > 0 else vector

def standardize_vector_l2(vector: np.ndarray, target_dim: int = 128) -> np.ndarray:
    """Standardize vector to target dimension using L2 normalization"""
    vector = ensure_vector_dimensions(vector, target_dim)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def standardize_vector(vector: np.ndarray, target_dim: int = 128) -> np.ndarray:
    """Ensure consistent vector dimensions"""
    if vector.shape[0] < target_dim:
        return np.pad(vector, (0, target_dim - vector.shape[0]))
    elif vector.shape[0] > target_dim:
        return vector[:target_dim]
    return vector

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """L2 normalization"""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def expand_vector(base_vector: np.ndarray, target_dim: int = 128) -> np.ndarray:
    """Expand a low-dimensional vector to target dimension"""
    if base_vector.shape[0] >= target_dim:
        return base_vector[:target_dim]
    
    expanded = np.zeros(target_dim)
    pattern = np.tile(base_vector, target_dim // base_vector.shape[0] + 1)
    expanded[:target_dim] = pattern[:target_dim]
    return expanded
