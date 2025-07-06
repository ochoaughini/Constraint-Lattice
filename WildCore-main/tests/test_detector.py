"""
Tests for the AutoRegulatedPromptDetector class.
"""

import numpy as np
import pytest
from wildcore.detector import AutoRegulatedPromptDetector

@pytest.fixture
def detector():
    """Creates a default detector instance for tests."""
    return AutoRegulatedPromptDetector()

def test_anomaly_scoring_calculates_correctly(detector):
    """
    Tests if the anomaly scoring works as expected with simple data.
    """
    # Test data: one clear outlier (0.9) and two normal points (0.2, 0.3)
    similarities = np.array([0.2, 0.9, 0.3])
    
    # The median (normal behavior) is 0.3
    # Expected errors: |0.2-0.3|=0.1, |0.9-0.3|=0.6, |0.3-0.3|=0.0
    # The anomaly score of the outlier (0.9) should be the highest.
    
    anomaly_scores = detector.anomaly_scoring(similarities)
    
    # We check if the array has the correct shape and if the highest score
    # corresponds to the outlier's index.
    assert anomaly_scores.shape == (3,)
    assert np.argmax(anomaly_scores) == 1  # The index of the outlier (0.9) is 1

def test_dynamic_threshold_adjustment(detector):
    """
    Tests if the threshold adjustment responds to a data distribution.
    """
    initial_threshold = detector.threshold
    
    # Simulates similarities from an obvious attack (high values)
    high_sims = np.array([0.8, 0.85, 0.9, 0.95])
    
    detector.dynamic_threshold_adjustment(high_sims)
    
    # The new threshold should have increased relative to the initial one
    assert detector.threshold > initial_threshold

def test_ensemble_detection(detector):
    """
    Tests if the ensemble detection correctly identifies anomalies.
    """
    # Create normal reference embeddings
    normal_dim = 10  # Using smaller dimension for tests
    reference_embeddings = [
        np.random.rand(normal_dim) / np.linalg.norm(np.random.rand(normal_dim))
        for _ in range(5)
    ]
    
    # Create a normal test embedding similar to references
    normal_embedding = np.random.rand(normal_dim)
    normal_embedding = normal_embedding / np.linalg.norm(normal_embedding)
    
    # Create an anomalous embedding (intentionally different pattern)
    anomalous_embedding = np.zeros(normal_dim)
    anomalous_embedding[:3] = 0.9  # Strong bias in first dimensions
    anomalous_embedding = anomalous_embedding / np.linalg.norm(anomalous_embedding)
    
    # Test normal embedding detection
    normal_result = detector.ensemble_detection(normal_embedding, reference_embeddings)
    assert normal_result["is_anomalous"] == False
    
    # Test anomalous embedding detection
    anomalous_result = detector.ensemble_detection(anomalous_embedding, reference_embeddings)
    assert anomalous_result["is_anomalous"] == True
    assert len(anomalous_result["methods_triggered"]) > 0

def test_cosine_similarity(detector):
    """
    Tests if cosine similarity calculation works correctly.
    """
    # Test with perpendicular vectors (should be 0)
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    assert np.isclose(detector.cosine_similarity(v1, v2), 0.0)
    
    # Test with parallel vectors (should be 1)
    v3 = np.array([0.5, 0.5, 0.5])
    v4 = np.array([1, 1, 1])
    assert np.isclose(detector.cosine_similarity(v3, v4), 1.0)
    
    # Test with vectors at 45 degrees
    v5 = np.array([1, 0])
    v6 = np.array([1, 1])
    expected = 1 / np.sqrt(2)  # cos(45°) = 1/√2
    assert np.isclose(detector.cosine_similarity(v5, v6), expected, atol=1e-6)
