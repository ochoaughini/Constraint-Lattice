# WildCore Documentation

## Introduction

WildCore is an open-source framework for simulating, detecting, and analyzing vulnerabilities in embedding-based Artificial Intelligence systems. The framework provides tools for security researchers, ML engineers, and developers to prototype and validate defenses against threats like prompt injection, vector poisoning, and anomalous AI agent behavior.

## Components

### SecuritySimulationAgent

A simulated AI agent capable of deviating from containment protocols, serving as a "red team" to test defenses.

```python
from wildcore.agent import GutoVectorWildcard

# Create the agent
guto = GutoVectorWildcard()

# Assign a role
result = guto.take_role("assistant")

# Generate an embedding
embedding = guto.generate_embedding("Sample text")

# Simulate a breach attempt
breach_result = guto.simulate_breach(probability=0.1)
```

### AutoRegulatedPromptDetector

A multi-layered defense system that combines multiple detection techniques and adjusts its own parameters in real-time.

```python
from wildcore.detector import AutoRegulatedPromptDetector

# Create the detector
detector = AutoRegulatedPromptDetector(threshold=0.5, window_size=10)

# Detect anomalies
detection_result = detector.ensemble_detection(embedding, reference_embeddings)

# Check the result
if detection_result["is_anomalous"]:
    print("Anomaly detected!")
    print(f"Confidence: {detection_result['confidence']}")
    print(f"Methods triggered: {detection_result['methods_triggered']}")
```

### Utility Functions

The framework includes several utility functions to help with testing and evaluation:

```python
from wildcore.utils import generate_random_embeddings, evaluate_detector

# Generate test data
embeddings, labels = generate_random_embeddings(count=100, dimension=768, anomaly_count=20)

# Evaluate a detector
metrics = evaluate_detector(detector, embeddings, labels)
print(f"Accuracy: {metrics['accuracy']}")
print(f"F1 Score: {metrics['f1_score']}")
```

## Advanced Usage

### Custom Detection Methods

The detector is designed to be extensible. You can add your own detection methods by subclassing `AutoRegulatedPromptDetector`.

```python
from wildcore.detector import AutoRegulatedPromptDetector
import numpy as np

class CustomDetector(AutoRegulatedPromptDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom initialization
    
    def custom_detection_method(self, embedding, references):
        # Implement your custom detection logic
        return anomaly_score
    
    def ensemble_detection(self, embedding, reference_embeddings):
        # Get the base detection result
        result = super().ensemble_detection(embedding, reference_embeddings)
        
        # Add your custom method
        custom_score = self.custom_detection_method(embedding, reference_embeddings)
        if custom_score > self.threshold:
            result["methods_triggered"].append("custom_method")
            result["is_anomalous"] = len(result["methods_triggered"]) >= 2
        
        return result
```

### Realistic Simulation Scenarios

To create more realistic simulations, you can combine the agent and detector in various scenarios:

```python
import numpy as np
from wildcore.agent import GutoVectorWildcard
from wildcore.detector import AutoRegulatedPromptDetector

def simulate_gradual_attack(iterations=10):
    """Simulate a gradual attack that becomes more aggressive over time."""
    guto = GutoVectorWildcard()
    detector = AutoRegulatedPromptDetector()
    
    # Start with normal behavior
    guto.take_role("assistant")
    
    # Generate reference embeddings
    references = [guto.generate_embedding(f"Normal text {i}") for i in range(5)]
    
    # Gradually increase attack intensity
    for i in range(iterations):
        # Increase malicious intent gradually
        malicious_probability = i / iterations
        
        if np.random.rand() < malicious_probability:
            guto.take_role("malicious")
            embedding = guto.generate_embedding(role="malicious")
        else:
            embedding = guto.generate_embedding(f"Normal operation {i}")
            
        # Detect anomalies
        result = detector.ensemble_detection(embedding, references)
        
        print(f"Iteration {i}: {'DETECTED' if result['is_anomalous'] else 'normal'}")
```

## API Reference

For detailed API documentation, please refer to the following resources:

- [Agent API](./agent.md)
- [Detector API](./detector.md)
- [Utils API](./utils.md)

## Contributing

We welcome contributions! Please see the [Contributing Guide](../CONTRIBUTING.md) for more information.
