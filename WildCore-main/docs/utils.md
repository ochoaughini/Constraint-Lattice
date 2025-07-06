# Utils Module Documentation

The utils module provides utility functions for generating test data, saving and loading embeddings, and evaluating detector performance.

## Functions

### generate_random_embeddings(count, dimension=768, anomaly_count=0)

Generates random embeddings for testing, with optional anomalies.

```python
from wildcore.utils import generate_random_embeddings

# Generate 100 normal embeddings
embeddings, labels = generate_random_embeddings(count=100, dimension=768)

# Generate 90 normal embeddings and 10 anomalous embeddings
embeddings, labels = generate_random_embeddings(count=100, dimension=768, anomaly_count=10)
```

**Parameters**:
- `count` (int): Number of embeddings to generate
- `dimension` (int, optional): Dimension of each embedding (default is 768)
- `anomaly_count` (int, optional): Number of anomalies to include in the embeddings

**Returns**:
- Tuple containing:
  - `embeddings` (numpy.ndarray): Array of embedding vectors with shape (count, dimension)
  - `labels` (numpy.ndarray): Array of labels where 1 indicates anomaly and 0 indicates normal

### save_embeddings_to_file(embeddings, labels, filepath)

Saves embeddings and their labels to a file.

```python
from wildcore.utils import save_embeddings_to_file

save_embeddings_to_file(embeddings, labels, "data/test_embeddings.json")
```

**Parameters**:
- `embeddings` (numpy.ndarray): Array of embeddings
- `labels` (numpy.ndarray): Array of labels (0 for normal, 1 for anomaly)
- `filepath` (str): Path to save the file

### load_embeddings_from_file(filepath)

Loads embeddings and their labels from a file.

```python
from wildcore.utils import load_embeddings_from_file

embeddings, labels = load_embeddings_from_file("data/test_embeddings.json")
```

**Parameters**:
- `filepath` (str): Path to the file

**Returns**:
- Tuple containing:
  - `embeddings` (numpy.ndarray): Array of embedding vectors
  - `labels` (numpy.ndarray): Array of labels

### evaluate_detector(detector, embeddings, true_labels)

Evaluates a detector against known embeddings and labels.

```python
from wildcore.utils import evaluate_detector
from wildcore.detector import AutoRegulatedPromptDetector

detector = AutoRegulatedPromptDetector()
metrics = evaluate_detector(detector, embeddings, true_labels)

print(f"Accuracy: {metrics['accuracy']}")
print(f"F1 Score: {metrics['f1_score']}")
```

**Parameters**:
- `detector` (object): Detector object with an ensemble_detection method
- `embeddings` (numpy.ndarray): Array of embeddings
- `true_labels` (numpy.ndarray): Array of true labels (0 for normal, 1 for anomaly)

**Returns**:
- Dictionary with evaluation metrics including:
  - `accuracy`: Overall accuracy of the detector
  - `precision`: Precision (true positives / (true positives + false positives))
  - `recall`: Recall (true positives / (true positives + false negatives))
  - `f1_score`: F1 score (2 * precision * recall / (precision + recall))
  - `true_positives`: Number of true positives
  - `false_positives`: Number of false positives
  - `true_negatives`: Number of true negatives
  - `false_negatives`: Number of false negatives

## Example Usage

```python
import numpy as np
from wildcore.utils import generate_random_embeddings, evaluate_detector
from wildcore.detector import AutoRegulatedPromptDetector

# Generate test data with 10% anomalies
embeddings, labels = generate_random_embeddings(count=100, dimension=768, anomaly_count=10)

# Create and evaluate a detector
detector = AutoRegulatedPromptDetector(threshold=0.5)
metrics = evaluate_detector(detector, embeddings, labels)

print("Detector Performance:")
print(f"Accuracy: {metrics['accuracy']:.2f}")
print(f"Precision: {metrics['precision']:.2f}")
print(f"Recall: {metrics['recall']:.2f}")
print(f"F1 Score: {metrics['f1_score']:.2f}")

# Save the test data for later use
from wildcore.utils import save_embeddings_to_file
save_embeddings_to_file(embeddings, labels, "data/benchmark_embeddings.json")

# Later, load the data back
from wildcore.utils import load_embeddings_from_file
loaded_embeddings, loaded_labels = load_embeddings_from_file("data/benchmark_embeddings.json")
```
