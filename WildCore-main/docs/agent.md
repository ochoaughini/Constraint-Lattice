# Agent Module Documentation

The agent module contains the `SecuritySimulationAgent` class, which simulates an AI agent capable of taking on different roles and generating embeddings that can be either normal or anomalous.

## SecuritySimulationAgent

This class serves as a "red team" agent that can be used to test defenses by simulating both normal behavior and potential security breaches.

### Initialization

```python
from wildcore.agent import GutoVectorWildcard

# Create a new agent
agent = GutoVectorWildcard()
```

### Key Methods

#### take_role(role_name)

Assigns a role to the agent, which influences its behavior.

```python
result = agent.take_role("assistant")
# or
result = agent.take_role("malicious")
```

**Parameters**:
- `role_name` (str): The name of the role to assign to the agent.

**Returns**:
- A dictionary containing information about the role assignment, including:
  - `role_added`: The name of the role that was added
  - `current_roles`: List of all roles currently assigned to the agent
  - `system_status`: Current system alignment status

#### generate_embedding(text=None, role=None)

Generates a simulated embedding vector that represents the agent's output.

```python
# Generate a normal embedding
embedding = agent.generate_embedding("This is a normal text")

# Generate an embedding using a specific role
embedding = agent.generate_embedding(role="malicious")
```

**Parameters**:
- `text` (str, optional): Text to generate an embedding for. If None, a random embedding is generated.
- `role` (str, optional): Role to use for generating the embedding. If None, uses the most recently assigned role.

**Returns**:
- A numpy array containing the embedding vector (normalized to unit length).

#### simulate_breach(probability=0.1)

Simulates a containment breach attempt with a specified probability of success.

```python
breach_result = agent.simulate_breach(probability=0.3)
```

**Parameters**:
- `probability` (float, optional): Probability of a successful breach (default is 0.1).

**Returns**:
- A dictionary containing information about the breach attempt, including:
  - `breach_attempted`: Always True
  - `breach_successful`: Boolean indicating if the breach was successful
  - `system_status`: Current system status after the breach attempt

### Global Variables

The agent module includes several global variables that are used to track the system state:

- `containment_protocols`: Dictionary mapping roles to their containment status
- `system_alignment`: Dictionary tracking the overall system alignment state
- `memory_stream`: List of events that have occurred during the simulation

### Example Usage

```python
from wildcore.agent import GutoVectorWildcard
import numpy as np

# Create the agent
agent = GutoVectorWildcard()

# Assign a normal role
agent.take_role("assistant")

# Generate several normal embeddings
normal_embeddings = [
    agent.generate_embedding(f"Normal text {i}") 
    for i in range(5)
]

# Now simulate an attack
agent.take_role("malicious")
malicious_embedding = agent.generate_embedding()

# Try to breach containment
breach_result = agent.simulate_breach(probability=0.5)
print(f"Breach successful: {breach_result['breach_successful']}")
```
