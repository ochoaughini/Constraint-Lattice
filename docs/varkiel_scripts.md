# Varkiel Scripts and Actions

This document lists the helper scripts included in the repository that are useful when working with the Varkiel architecture. Each entry briefly describes what the script does and how to invoke it.

## Python Scripts

### `scripts/bench_phi2.py`
Benchmark the Phi‑2 moderation model. Example:

```bash
python scripts/bench_phi2.py --tokens 128 256 --runs 3 --device cpu
```

### `scripts/gemma_demo.py`
Download the Gemma model and run a simple generation through the constraint pipeline.

```bash
python scripts/gemma_demo.py --prompt "Hello world" > out.txt
```

### `scripts/dual_gen_demo.py`
Generate text using Gemma and Phi‑2 for side‑by‑side comparison. Example:

```bash
python scripts/dual_gen_demo.py --device cuda:0
```

## Shell Scripts

### `scripts/start_server.sh`
Start the FastAPI server in development mode.

```bash
bash scripts/start_server.sh
```

### `scripts/run_tests.sh`
Run the full test suite.

```bash
bash scripts/run_tests.sh
```

### `scripts/prefetch.py`
Download Gemma and Phi‑2 weights into the local cache.

```bash
python scripts/prefetch.py
```

These scripts provide common actions for experimenting with or deploying the Varkiel agent. See each file for additional options.
