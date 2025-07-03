# Constraint Lattice

Deterministic, auditable post-processing governance framework for Large Language Model (LLM) outputs.

Constraint Lattice lets you **compose pluggable constraints** that rewrite, redact, or regenerate model output until it satisfies your policy. Each execution is fully auditable and can be replayed bit-for-bit—all without retraining the model.

## Features
- Declarative YAML pipelines driving pure-Python `Constraint` classes
- Multiple integration options: CLI (`cl-apply`), Python SDK, FastAPI micro-service, Streamlit audit viewer, WordPress plugin, and SaaS starter kit
- Deterministic execution with JSONL audit logs suitable for governance and red-team review
- Optional GPU-accelerated moderation via vLLM or JAX
- Prometheus metrics and Stripe billing hooks for production SaaS deployments

## Install

```bash
pip install constraint-lattice           # Core framework
pip install "constraint-lattice[perf]"   # +Prometheus & vLLM
pip install "constraint-lattice[api]"    # +FastAPI micro-service
```

Or clone the repo for development:

```bash
git clone https://github.com/ochoaughini/Constraint-Lattice.git
cd Constraint-Lattice
pip install -e .[dev]    # Lint, type-check, tests
```

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

## Documentation

- [Tutorial](docs/tutorial.md)
- [Principles and Theory](docs/principles.md)
- [Formal Language Approach](docs/formal_language_approach.md)
- [Hybrid Deployment Strategy](docs/hybrid_deployment_strategy.md)
- [Phi-2 Integration Proposal](docs/phi2_integration_proposal.md)
- [Example: Transformers Gemma](docs/transformers_gemma_example.md)


## Contributing

PRs are welcome! Please see `CONTRIBUTING.md` for the workflow, code style, and a list of good first issues.

## License

Constraint Lattice is released under the MIT License. See `LICENSE` for details.
