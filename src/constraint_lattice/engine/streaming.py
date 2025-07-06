# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. All rights reserved.
# See LICENSE for full terms.


class StreamingConstraintEngine:
    def __init__(self, constraints, window_size=5):
        self.constraints = constraints
        self.window_size = window_size
        self.buffer = []

    def process_token(self, token, prompt):
        self.buffer.append(token)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        window_text = "".join(self.buffer)
        processed, _ = apply_constraints(
            prompt, window_text, self.constraints, return_trace=True
        )
        return processed[-1] if processed else token
