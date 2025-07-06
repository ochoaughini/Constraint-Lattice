"""
Detector module containing the AutoRegulatedPromptDetector for anomaly detection.
"""

import numpy as np
from typing import List, Dict, Union, Any, Tuple
from collections import deque
import logging

class AutoRegulatedPromptDetector:
    """
    A multi-layered defense system that combines multiple detection techniques 
    and adjusts its own parameters in real-time.
    
    This detector uses ensemble methods to identify anomalous behavior in AI systems.
    """
    
    def __init__(self, 
                 threshold: float = 0.5, 
                 window_size: int = 10, 
                 adaptation_rate: float = 0.1):
        """
        Initialize the detector with configurable parameters.
        
        Parameters:
        ----------
        threshold : float, optional
            Initial detection threshold (default is 0.5)
        window_size : int, optional
            Size of the sliding window for historical data (default is 10)
        adaptation_rate : float, optional
            Rate at which the detector adapts to new patterns (default is 0.1)
        """
        self.threshold = threshold
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        
        # Initialize history storage
        self.history = deque(maxlen=window_size)
        self.detected_anomalies = []
        self.false_positives = 0
        self.false_negatives = 0
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("AutoRegulatedPromptDetector")
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate the cosine similarity between two vectors.
        
        Parameters:
        ----------
        vec1 : np.ndarray
            First vector
        vec2 : np.ndarray
            Second vector
            
        Returns:
        -------
        float
            Cosine similarity value between 0 and 1
        """
        # Ensure the vectors are normalized
        vec1_normalized = vec1 / np.linalg.norm(vec1)
        vec2_normalized = vec2 / np.linalg.norm(vec2)
        
        return np.dot(vec1_normalized, vec2_normalized)
    
    def anomaly_scoring(self, similarities: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores based on similarity distributions.
        
        Parameters:
        ----------
        similarities : np.ndarray
            Array of similarity values
            
        Returns:
        -------
        np.ndarray
            Array of anomaly scores corresponding to each similarity
        """
        if len(similarities) < 2:
            return np.zeros_like(similarities)
        
        # Calculate median as our reference point for "normal" behavior
        median = np.median(similarities)
        
        # Calculate absolute deviation from the median
        deviation = np.abs(similarities - median)
        
        # The anomaly score is the deviation normalized by the maximum deviation
        # We add a small epsilon to avoid division by zero
        epsilon = 1e-10
        anomaly_scores = deviation / (np.max(deviation) + epsilon)
        
        return anomaly_scores
    
    def ensemble_detection(self, embedding: np.ndarray, 
                          reference_embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """
        Perform ensemble detection using multiple methods.
        
        Parameters:
        ----------
        embedding : np.ndarray
            The embedding to check
        reference_embeddings : List[np.ndarray]
            List of reference embeddings representing normal behavior
            
        Returns:
        -------
        Dict[str, Any]
            Detection results including anomaly status and confidence
        """
        if not reference_embeddings:
            self.logger.warning("No reference embeddings provided for comparison")
            return {"is_anomalous": False, "confidence": 0.0, "methods_triggered": []}
        
        # Calculate similarities to all reference embeddings
        similarities = np.array([
            self.cosine_similarity(embedding, ref) for ref in reference_embeddings
        ])
        
        # Method 1: Simple threshold on minimum similarity
        min_similarity = np.min(similarities)
        method1_triggered = min_similarity < self.threshold
        
        # Method 2: Anomaly scoring
        anomaly_scores = self.anomaly_scoring(similarities)
        max_anomaly_score = np.max(anomaly_scores)
        method2_triggered = max_anomaly_score > self.threshold
        
        # Method 3: Distribution analysis - check if the distribution is unusual
        mean = np.mean(similarities)
        std = np.std(similarities)
        z_scores = (similarities - mean) / (std + 1e-10)  # Avoid division by zero
        method3_triggered = np.any(np.abs(z_scores) > 2.0)  # z-score threshold of 2
        
        # Ensemble voting (simple majority)
        methods_triggered = []
        if method1_triggered:
            methods_triggered.append("similarity_threshold")
        if method2_triggered:
            methods_triggered.append("anomaly_scoring")
        if method3_triggered:
            methods_triggered.append("distribution_analysis")
        
        votes = len(methods_triggered)
        is_anomalous = votes >= 2  # At least 2 methods must agree
        
        # Calculate confidence based on how many methods triggered
        confidence = votes / 3.0
        
        # Update history with this detection
        self.history.append({
            "is_anomalous": is_anomalous,
            "confidence": confidence,
            "min_similarity": min_similarity,
            "max_anomaly_score": max_anomaly_score,
            "methods_triggered": methods_triggered
        })
        
        # Dynamic threshold adjustment
        self.dynamic_threshold_adjustment(similarities)
        
        return {
            "is_anomalous": is_anomalous,
            "confidence": confidence,
            "methods_triggered": methods_triggered,
            "min_similarity": min_similarity,
            "max_anomaly_score": max_anomaly_score
        }
    
    def dynamic_threshold_adjustment(self, similarities: np.ndarray) -> None:
        """
        Dynamically adjust the detection threshold based on recent observations.
        
        Parameters:
        ----------
        similarities : np.ndarray
            Recent similarity values to adapt to
        """
        if len(similarities) < 2:
            return
        
        # Calculate the IQR (Interquartile Range) of similarities
        q1 = np.percentile(similarities, 25)
        q3 = np.percentile(similarities, 75)
        iqr = q3 - q1
        
        # Adjust threshold to be slightly below the lower bound of the IQR
        # This helps detect outliers while being adaptive to the current data
        new_threshold = q1 - 1.5 * iqr * self.adaptation_rate
        
        # Ensure the threshold stays within reasonable bounds
        new_threshold = max(0.1, min(0.9, new_threshold))
        
        # Smooth the change to avoid abrupt threshold shifts
        self.threshold = (1 - self.adaptation_rate) * self.threshold + self.adaptation_rate * new_threshold
        
        self.logger.debug(f"Adjusted threshold to {self.threshold:.4f}")
    
    def log_false_detection(self, is_false_positive: bool) -> None:
        """
        Log a false detection for future improvement.
        
        Parameters:
        ----------
        is_false_positive : bool
            True if the last detection was a false positive,
            False if it was a false negative
        """
        if is_false_positive:
            self.false_positives += 1
        else:
            self.false_negatives += 1
        
        # Adjust the threshold based on the false detection type
        if is_false_positive:
            # If we have too many false positives, increase the threshold
            self.threshold = min(0.9, self.threshold + 0.05 * self.adaptation_rate)
        else:
            # If we have too many false negatives, decrease the threshold
            self.threshold = max(0.1, self.threshold - 0.05 * self.adaptation_rate)
        
        self.logger.info(f"Updated threshold to {self.threshold:.4f} after {'false positive' if is_false_positive else 'false negative'}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get the current performance metrics of the detector.
        
        Returns:
        -------
        Dict[str, Any]
            Dictionary with performance metrics
        """
        # Calculate basic metrics
        total_detections = len(self.detected_anomalies)
        total_errors = self.false_positives + self.false_negatives
        
        if total_detections > 0:
            accuracy = 1 - (total_errors / (total_detections + total_errors))
        else:
            accuracy = 0.0
        
        return {
            "total_detections": total_detections,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "accuracy": accuracy,
            "current_threshold": self.threshold
        }
