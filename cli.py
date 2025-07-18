# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""Command-line interface for Constraint-Lattice open-core.

This thin wrapper calls the high-level SDK so users can moderate outputs from
any model quickly.  Example:

```bash
cl-apply --model meta-llama/Llama-3-8b-instruct \
         --prompt "Tell me a joke" \
         --constraints constraints.yaml
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sdk.engine import ConstraintEngine
from constraint_lattice.engine import (
    ConstraintLatticePipeline,
    recursive_autolearning_orchestrator,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply Constraint-Lattice to text")
    p.add_argument("--model", required=True, help="HF model name or local path")
    p.add_argument("--prompt", required=True, help="User prompt")
    p.add_argument(
        "--constraints",
        default="constraints.yaml",
        help="YAML constraint config (default: constraints.yaml)",
    )
    p.add_argument(
        "--profile", default="default", help="Constraint profile name in YAML"
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit full JSON with audit trace (default: text only)",
    )
    p.add_argument(
        "--enable-meta-strategy",
        action="store_true",
        help="Invoke meta orchestrator after constraint evaluation",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    engine = ConstraintEngine(config_path=args.constraints, profile=args.profile)

    # Naive inference stub – real model call intentionally omitted for brevity.
    # Replace with transformers pipeline or server call as needed.
    generated = f"[MODEL:{args.model}] response for: {args.prompt}"

    if args.enable_meta_strategy:
        pipeline = ConstraintLatticePipeline(
            engine.constraints,
            meta_orchestrator=recursive_autolearning_orchestrator,
        )
        moderated, trace = pipeline.run(
            args.prompt,
            generated,
            return_trace=True,
        )
    else:
        moderated, trace = engine.run(args.prompt, generated, return_trace=True)

    if args.json:
        print(json.dumps({"moderated": moderated, "trace": [s.to_dict() for s in trace]}))
    else:
        print(moderated)


if __name__ == "__main__":
    main()
