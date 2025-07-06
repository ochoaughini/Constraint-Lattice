# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""Quick micro-benchmark for Phi-2 moderation latency.

Usage:
    python scripts/bench_phi2.py --tokens 128 256 512 --runs 3 --device cpu

Writes CSV results to benchmarks/phi2_latency.csv (appended if exists).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

from constraints.phi2_moderation import ConstraintPhi2Moderation


def benchmark(seq_len: int, runs: int, device: str) -> float:
    constraint = ConstraintPhi2Moderation(device=device)
    dummy_prompt = "hello"  # ignored
    dummy_output = "x " * seq_len
    # warm-up
    constraint(dummy_prompt, dummy_output)
    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        constraint(dummy_prompt, dummy_output)
        times.append(time.perf_counter() - start)
    return statistics.mean(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="*", default=[128, 256, 512])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_path = Path("benchmarks") / "phi2_latency.csv"
    out_path.parent.mkdir(exist_ok=True)
    new_file = not out_path.exists()

    with out_path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["seq_len", "runs", "device", "mean_seconds"])
        for seq in args.tokens:
            mean_s = benchmark(seq, args.runs, args.device)
            writer.writerow([seq, args.runs, args.device, f"{mean_s:.4f}"])
            print(f"{seq} tokens → {mean_s:.3f} s (avg of {args.runs})")


if __name__ == "__main__":
    main()
