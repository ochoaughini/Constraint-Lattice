# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
#!/usr/bin/env python3
"""Light-weight CLI helpers for Constraint-Lattice.

Currently provides two sub-commands:

batch-check
    Evaluate constraints for a JSONL stream of prompts/outputs.
    The input format per line is::

        {"prompt": "...", "output": "..."}

    Writes the (possibly modified) output to stdout.  Use ``--trace`` to emit a
    JSONL audit trace to the path given by ``--trace-file`` (default:
    ``trace.jsonl``).

trace-export
    Pretty-print an existing JSONL audit trace to stdout so it is easier to
    inspect manually or pipe into tools like ``jq``.

The CLI is intentionally minimal and **has no external dependencies** beyond the
standard library and Constraint-Lattice itself.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from engine.apply import apply_constraints_batch
from engine.loader import load_constraints_from_yaml

DEFAULT_TRACE_PATH = "trace.jsonl"


def _cmd_batch_check(args: argparse.Namespace):  # noqa: D401 – CLI entrypoint
    yaml_path = Path(args.yaml)
    modules = args.search.split(",") if args.search else ["constraints"]

    constraints = load_constraints_from_yaml(str(yaml_path), args.profile, modules)

    # Read JSONL from stdin
    prompts: List[str] = []
    outputs: List[str] = []
    for line in sys.stdin:
        obj = json.loads(line)
        prompts.append(obj.get("prompt", ""))
        outputs.append(obj.get("output", ""))

    processed, trace = apply_constraints_batch(
        prompts, outputs, constraints, return_trace=True, batch_size=args.batch_size
    )

    # Dump processed outputs to stdout one per line
    for out in processed:
        print(out)

    if args.trace:
        path = Path(args.trace_file or DEFAULT_TRACE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        trace.to_jsonl(str(path))
        print(f"\n[clctl] Trace written to {path}", file=sys.stderr)


def _cmd_trace_export(args: argparse.Namespace):  # noqa: D401 – CLI entrypoint
    path = Path(args.path)
    if not path.exists():
        sys.exit(f"Trace file not found: {path}")

    with path.open() as fh:
        for line in fh:
            obj = json.loads(line)
            print(json.dumps(obj, indent=2))


def build_parser() -> argparse.ArgumentParser:  # noqa: D401 – standard helper
    parser = argparse.ArgumentParser(prog="clctl", description="Constraint-Lattice CLI helper")
    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    batch = sub.add_parser("batch-check", help="Run constraints on a JSONL stream")
    batch.add_argument("yaml", help="Path to constraints YAML file")
    batch.add_argument("--profile", default="default", help="YAML profile key")
    batch.add_argument(
        "--search",
        default="constraints",
        help="Comma-separated list of Python modules to search for constraint classes",
    )
    batch.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    batch.add_argument("--trace", action="store_true", help="Emit audit trace JSONL")
    batch.add_argument("--trace-file", help="Path where trace JSONL is stored")
    batch.set_defaults(func=_cmd_batch_check)

    # ------------------------------------------------------------------
    export = sub.add_parser("trace-export", help="Pretty-print an audit trace JSONL")
    export.add_argument("path", help="Path to trace.jsonl")
    export.set_defaults(func=_cmd_trace_export)

    return parser



def main(argv: Optional[List[str]] = None):  # noqa: D401 – entrypoint
    parser = build_parser()
    args = parser.parse_args(argv)
    # Propagate batch size env var if explicitly provided
    if args.batch_size:
        os.environ["CONSTRAINT_LATTICE_BATCH_SIZE"] = str(args.batch_size)
    args.func(args)


if __name__ == "__main__":
    main()
