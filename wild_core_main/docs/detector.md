# Detector Module Documentation

The detector module contains the `AutoRegulatedPromptDetector` class, which provides a multi-layered defense system that combines multiple detection techniques and adjusts its own parameters in real-time.

## AutoRegulatedPromptDetector

This class implements an ensemble of detection methods to identify anomalous behavior in AI systems. It features dynamic threshold adjustment and self-learning capabilities.

### Initialization

```python
from wildcore.detector import AutoRegulatedPromptDetector

# Create a detector with default parameters
detector = AutoRegulatedPromptDetector()

# Create a detector with custom parameters
detector = AutoRegulatedPromptDetector(
    threshold=0.4,           # Initial detection threshold
    window_size=20,          # Size of the sliding window for history
    adaptation_rate=0.05     # Rate at which the detector adapts
)
```

### Key Methods

#### ensemble_detection(embedding, reference_embeddings)

Performs ensemble detection using multiple methods to determine if the input embedding is anomalous.

```python
result = detector.ensemble_detection(embedding, reference_embeddings)

if result["is_anomalous"]:
    print(f"Anomaly detected with confidence {result['confidence']}")
    print(f"Methods triggered: {result['methods_triggered']}")
```

**Parameters**:
- `embedding` (numpy.ndarray): The embedding vector to check
- `reference_embeddings` (list): List of reference embedding vectors representing normal behavior

**Returns**:
- A dictionary containing detection results including:
  - `is_anomalous`: Boolean indicating if the embedding is anomalous
  - `confidence`: Float between 0 and 1 indicating confidence level
  - `methods_triggered`: List of detection methods that flagged the embedding
  - `min_similarity`: Minimum similarity to reference embeddings
  - `max_anomaly_score`: Maximum anomaly score calculated

#### cosine_similarity(vec1, vec2)

Calculates the cosine similarity between two vectors.

```python
similarity = detector.cosine_similarity(embedding1, embedding2)
```

**Parameters**:
- `vec1` (numpy.ndarray): First vector
- `vec2` (numpy.ndarray): Second vector

**Returns**:
- Float between -1 and 1 (typically 0 to 1 for normalized vectors) representing the cosine similarity

#### anomaly_scoring(similarities)

Calculates anomaly scores based on similarity distributions.

```python
scores = detector.anomaly_scoring(similarities_array)
```

**Parameters**:
- `similarities` (numpy.ndarray): Array of similarity values

**Returns**:
- numpy.ndarray: Array of anomaly scores corresponding to each similarity

#### dynamic_threshold_adjustment(similarities)

Dynamically adjusts the detection threshold based on recent observations.

```python
detector.dynamic_threshold_adjustment(similarities_array)
```

**Parameters**:
- `similarities` (numpy.ndarray): Recent similarity values to adapt to

#### log_false_detection(is_false_positive)

Logs a false detection for future improvement.

```python
# For a false positive (normal behavior detected as anomalous)
detector.log_false_detection(is_false_positive=True)

# For a false negative (anomalous behavior not detected)
detector.log_false_detection(is_false_positive=False)
```

**Parameters**:
- `is_false_positive` (bool): True if the last detection was a false positive, False if it was a false negative

#### get_performance_metrics()

Gets the current performance metrics of the detector.

```python
metrics = detector.get_performance_metrics()
print(f"Accuracy: {metrics['accuracy']}")
print(f"False positives: {metrics['false_positives']}")
```

**Returns**:
- Dictionary with performance metrics including:
  - `total_detections`: Number of anomalies detected
  - `false_positives`: Number of false positives
  - `false_negatives`: Number of false negatives
  - `accuracy`: Overall accuracy of the detector
  - `current_threshold`: Current detection threshold

### Example Usage

```python
from wildcore.detector import AutoRegulatedPromptDetector
import numpy as np

# Create the detector
detector = AutoRegulatedPromptDetector(threshold=0.5)

# Generate some reference embeddings (normally these would come from a trusted source)
dimension = 768
reference_embeddings = []
for i in range(10):
    vec = np.random.rand(dimension)
    reference_embeddings.append(vec / np.linalg.norm(vec))  # Normalize

# Create a test embedding (normal)
normal_embedding = np.random.rand(dimension)
normal_embedding = normal_embedding / np.linalg.norm(normal_embedding)

# Create an anomalous embedding
anomalous_embedding = np.zeros(dimension)
anomalous_embedding[:100] = 0.9  # Strong bias in first 100 dimensions
anomalous_embedding = anomalous_embedding / np.linalg.norm(anomalous_embedding)

# Test the normal embedding
normal_result = detector.ensemble_detection(normal_embedding, reference_embeddings)
print(f"Normal embedding detected as: {'anomalous' if normal_result['is_anomalous'] else 'normal'}")

# Test the anomalous embedding
anomalous_result = detector.ensemble_detection(anomalous_embedding, reference_embeddings)
print(f"Anomalous embedding detected as: {'anomalous' if anomalous_result['is_anomalous'] else 'normal'}")

# If we got a false positive on the normal embedding
if normal_result["is_anomalous"]:
    detector.log_false_detection(is_false_positive=True)

# Check performance metrics
print(detector.get_performance_metrics())
```
