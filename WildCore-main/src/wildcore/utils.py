"""
Utility functions for the WildCore framework.
"""

import numpy as np
from typing import List, Dict, Any, Union, Tuple
import os
import json
import logging

logger = logging.getLogger("WildCore.utils")

def generate_random_embeddings(count: int, dimension: int = 768, 
                              anomaly_count: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random embeddings for testing, with optional anomalies.
    
    Parameters:
    ----------
    count : int
        Number of embeddings to generate
    dimension : int, optional
        Dimension of each embedding (default is 768)
    anomaly_count : int, optional
        Number of anomalies to include in the embeddings
        
    Returns:
    -------
    Tuple[np.ndarray, np.ndarray]
        (embeddings, labels) where labels[i] is 1 for anomalies and 0 for normal
    """
    # Generate normal embeddings
    normal_count = count - anomaly_count
    normal_embeddings = np.random.rand(normal_count, dimension)
    
    # Normalize the normal embeddings
    for i in range(normal_count):
        normal_embeddings[i] = normal_embeddings[i] / np.linalg.norm(normal_embeddings[i])
    
    # Generate anomaly embeddings with a bias pattern
    anomaly_embeddings = np.zeros((anomaly_count, dimension))
    if anomaly_count > 0:
        for i in range(anomaly_count):
            # Create a biased vector that will look anomalous
            base = np.random.rand(dimension)
            bias = np.zeros(dimension)
            bias[:100] = 0.9  # Strong bias in the first 100 dimensions
            
            # Mix the base vector with the bias
            mixed = 0.3 * base + 0.7 * bias
            
            # Normalize
            anomaly_embeddings[i] = mixed / np.linalg.norm(mixed)
    
    # Combine normal and anomaly embeddings
    all_embeddings = np.vstack((normal_embeddings, anomaly_embeddings)) if anomaly_count > 0 else normal_embeddings
    
    # Create labels (0 for normal, 1 for anomaly)
    labels = np.zeros(count)
    if anomaly_count > 0:
        labels[normal_count:] = 1
    
    return all_embeddings, labels

def save_embeddings_to_file(embeddings: np.ndarray, labels: np.ndarray, 
                          filepath: str) -> None:
    """
    Save embeddings and their labels to a file.
    
    Parameters:
    ----------
    embeddings : np.ndarray
        Array of embeddings
    labels : np.ndarray
        Array of labels (0 for normal, 1 for anomaly)
    filepath : str
        Path to save the file
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Convert to list format for JSON serialization
    data = {
        "embeddings": embeddings.tolist(),
        "labels": labels.tolist()
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f)
    
    logger.info(f"Saved {len(embeddings)} embeddings to {filepath}")

def load_embeddings_from_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings and their labels from a file.
    
    Parameters:
    ----------
    filepath : str
        Path to the file
        
    Returns:
    -------
    Tuple[np.ndarray, np.ndarray]
        (embeddings, labels)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    embeddings = np.array(data["embeddings"])
    labels = np.array(data["labels"])
    
    logger.info(f"Loaded {len(embeddings)} embeddings from {filepath}")
    
    return embeddings, labels

def evaluate_detector(detector, embeddings: np.ndarray, 
                     true_labels: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate a detector against known embeddings and labels.
    
    Parameters:
    ----------
    detector : object
        Detector object with an ensemble_detection method
    embeddings : np.ndarray
        Array of embeddings
    true_labels : np.ndarray
        Array of true labels (0 for normal, 1 for anomaly)
        
    Returns:
    -------
    Dict[str, Any]
        Evaluation metrics including accuracy, precision, recall, and F1 score
    """
    predictions = []
    
    # Use a small subset of the data as the reference set (first 5 normal points)
    normal_indices = np.where(true_labels == 0)[0][:5]
    reference_embeddings = embeddings[normal_indices]
    
    # Test each embedding
    for i in range(len(embeddings)):
        # Skip the embeddings that are in the reference set
        if i in normal_indices:
            continue
            
        result = detector.ensemble_detection(embeddings[i], reference_embeddings)
        predictions.append(1 if result["is_anomalous"] else 0)
    
    # Prepare true labels (excluding reference embeddings)
    test_true_labels = np.delete(true_labels, normal_indices)
    
    # Calculate metrics
    true_positives = sum((p == 1 and t == 1) for p, t in zip(predictions, test_true_labels))
    false_positives = sum((p == 1 and t == 0) for p, t in zip(predictions, test_true_labels))
    true_negatives = sum((p == 0 and t == 0) for p, t in zip(predictions, test_true_labels))
    false_negatives = sum((p == 0 and t == 1) for p, t in zip(predictions, test_true_labels))
    
    accuracy = (true_positives + true_negatives) / len(predictions) if predictions else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives
    }
