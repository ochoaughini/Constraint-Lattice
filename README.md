# Constraint Lattice

[![CI/CD](https://github.com/ochoaughini/Constraint-Lattice/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/ochoaughini/Constraint-Lattice/actions/workflows/docker-publish.yml)

Deterministic, auditable post-processing governance framework for Large Language Model (LLM) outputs.

Constraint Lattice enables **composable constraints** that rewrite, redact, or regenerate model outputs until they satisfy your policy requirements. All executions are fully auditable and reproducible without model retraining.

## Features

- **Declarative Pipelines**: YAML-driven configuration for pure-Python `Constraint` classes
- **Multi-Platform Support**: CLI (`cl-apply`), Python SDK, FastAPI microservice, Streamlit audit viewer
- **Deterministic Execution**: JSONL audit logs for governance and compliance
- **GPU Acceleration**: Optional vLLM/JAX backend for high-performance moderation
- **Production Ready**: Prometheus metrics and Stripe billing integration

## Installation

### Local Development

```bash
git clone https://github.com/ochoaughini/Constraint-Lattice.git
cd Constraint-Lattice

# Install with pip
pip install -e .[dev]  # Development mode (includes linting/tests)
pip install constraint-lattice     # Core framework only
pip install "constraint-lattice[perf]"  # +Performance extensions
```

### Docker

```bash
docker build -t constraint-lattice .
docker run -p 8000:8000 constraint-lattice
```

### Cloud Deployment
See our [deployment guide](docs/hybrid_deployment_strategy.md) for GCP/AWS configurations.

## Basic Usage

```python
from constraint_lattice import apply_constraints

result = apply_constraints(
    text="Your LLM output here",
    policy_path="path/to/policy.yaml"
)
print(result.filtered_text)
```

## Documentation
- [Core Concepts](docs/principles.md)
- [API Reference](docs/api.md)
- [Tutorial](docs/tutorial.md)
- [Deployment Strategies](docs/hybrid_deployment_strategy.md)

## Contributing
We welcome contributions! Please see our [Contribution Guidelines](CONTRIBUTING.md).

## License
Constraint Lattice is released under the [MIT License](LICENSE).

## Quick Start (Python)

```python
from sdk.engine import ConstraintEngine

engine = ConstraintEngine(profile="default")  # reads constraints.yaml
result = engine.run(prompt, raw_llm_output)
print(result.text)          # moderated text
print(result.audit_path)    # path to JSONL log
```

## Command-Line

```bash
cl-apply --model meta-llama/Llama-3-8b-instruct \
         --prompt "Tell me a chemistry joke" \
         --constraints constraints.yaml
```

Streaming generation is displayed while violations are fixed in real time.

## REST API

```bash
uvicorn sdk.rest_api:app --reload
```

POST `{"prompt": "...", "output": "..."}` to `/govern` and receive the moderated text plus a link to the audit file.

## Audit Viewer UI

```bash
streamlit run ui/audit_viewer.py
```

Upload any JSONL audit log to explore step-by-step constraint actions.

## Kafka Integration
Constraint Lattice uses Kafka for real-time trace streaming. To set up Kafka locally:

```bash
docker-compose -f docker-compose.yml up -d kafka
```

## Trace Visualizer
Explore constraint execution traces in real-time:

```bash
streamlit run src/constraint_lattice/ui/trace_visualizer.py --server.port=8502
```

Access the visualizer at: http://localhost:8502

## Configuration

Define constraints in `constraints.yaml`:

```yaml
profile: default
constraints:
  - name: ConstraintProfanityFilter
  - name: ConstraintBoundaryPrime
  - name: ConstraintPhi2Moderation
    params:
      provider: vllm
      safety_thresholds:
        hate_speech: 0.85
```

## Full Stack with Docker Compose

The repo ships a `docker-compose.yml` that spins up:

* FastAPI backend (`clattice_api`)
* Redis cache
* MySQL + WordPress with the *Constraint Lattice API* plugin

1. Copy `.env.bak` to `.env` and fill secrets.
2. Run:

   ```bash
   docker-compose up --build
   ```

Open:

* Backend: <http://localhost:8000>
* WordPress: <http://localhost:8080>

Stop with `docker-compose down`.

## Running Tests

```bash
pytest -q
ruff check .
mypy .
```

## Contributing

PRs are welcome! Please see `CONTRIBUTING.md` for the workflow, code style, and a list of good first issues.

## License

Constraint Lattice is released under the MIT License. See `LICENSE` for details.
