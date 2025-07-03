# Constraint Lattice Tutorial: Quick Start & Integration

## Install

```bash
# Core framework
pip install constraint-lattice

# Optional: performance extras (Prometheus metrics + GPU vLLM)
pip install "constraint-lattice[perf]"

# Optional: JAX / Flax acceleration
pip install "constraint-lattice[jax]"

# Demo & tutorial utilities
pip install pyyaml fastapi streamlit
```

## Configure Constraints

Create or edit `constraints.yaml` to define pipelines. Example:

```yaml
profile: default
constraints:
  - name: ConstraintProfanityFilter
  - name: ConstraintBoundaryPrime
  - name: ConstraintPhi2Moderation
    params:
      provider: jax       # hf | vllm | jax
      fallback_strategy: regenerate
      safety_thresholds:
        hate_speech: 0.85
        violence: 0.70
```

> **Tip:** Set `provider: vllm` on GPU servers for async micro-batching at the socket layer.

## Programmatic Usage

```python
from sdk.engine import ConstraintEngine
engine = ConstraintEngine(profile='default')
result = engine.run("Prompt here", "Raw LLM output here")
```

## REST API Usage

```bash
uvicorn sdk.rest_api:app --reload
```

POST to `/govern` with JSON:
```json
{"prompt": "Are you alive?", "output": "I am sentient."}
```

## Audit Log Visualization

```bash
streamlit run audit_viewer.py
```

Upload a `.jsonl` file produced by the engine for interactive inspection.

## Dual Generation Showcase

Constraint Lattice can compare two models side-by-side while moderating the higher-quality output:

```bash
poetry run python scripts/dual_gen_demo.py --prompt "Explain the Doppler effect"
```

| Source          | Text |
|-----------------|------|
| Gemma raw       | *…*  |
| Gemma moderated | *…*  |
| Phi-2 raw       | *…*  |

## Day-to-Day Workflow

---

*Last updated: 2025-06-29*

To work with Constraint-Lattice day to day, first clone the repository and, inside the project root, optionally create and activate a virtual environment, then run `pip install -r requirements-lock.txt` (or `pip install -e .`) to install the dependencies. You can immediately apply constraints from the command line with `python -m engine.apply --model meta-llama/Llama-3-8b-instruct --prompt "Tell me a joke with no profanity" --constraints constraints.yaml`; this streams model output while enforcing the rules defined by the Python classes in `constraints/` and listed in `constraints.yaml`. For programmatic use, import `constraint_lattice.sdk.engine as cl`, load a model with `cl.load(model_name="meta-llama/Llama-3-8b-instruct", constraints_yaml="constraints.yaml")`, call `generate`, and read `response.text`. Ready-made demos—`scripts/dual_gen_demo.py`, `scripts/gemma_demo.py`, and `scripts/bench_phi2.py`—let you compare guardrailed vs. vanilla generation, explore Gemma/Gemini integration, or benchmark Phi-2. When adding or tuning constraints, remember that each lives in its own small Python file with a `check()` method; edit the YAML list to change order or selection, then simply rerun your script—no build step is required. Run `pytest tests/` to execute unit, smoke, and end-to-end suites whenever you touch constraints or upgrade `transformers`. Generation logs are saved as JSONL files under `results/`; inspect them with the lightweight Streamlit viewer in `ui/audit_viewer.py`. A typical daily loop is pull the latest changes, optionally refresh models in the Hugging Face cache via `scripts/prefetch.py`, edit or add constraints, run the test suite, generate text, and review the audits. In editors like VS Code or PyCharm, keep the repo root on your Python path (e.g., by activating `.venv`) so `import constraint_lattice` resolves instantly. With those steps you can clone, install, test, demo, and iterate on Constraint-Lattice seamlessly.

