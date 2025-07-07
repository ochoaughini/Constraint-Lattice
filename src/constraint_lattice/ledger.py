# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. All rights reserved.
# See LICENSE for full terms.
"""Simplified Cross-Agent Alignment Ledger."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional


@dataclass
class LedgerEvent:
    timestamp: str
    agent_id: str
    constraint: str
    action: str
    metadata: Dict[str, Any]


class CrossAgentAlignmentLedger:
    """Append-only JSONL ledger for constraint events."""

    def __init__(self, path: str = "alignment_ledger.jsonl") -> None:
        self.path = path
        if not os.path.exists(self.path):
            open(self.path, "a", encoding="utf-8").close()

    def record(
        self,
        agent_id: str,
        constraint: str,
        action: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        event = LedgerEvent(
            timestamp=datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
            agent_id=agent_id,
            constraint=constraint,
            action=action,
            metadata=metadata or {},
        )
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(event)) + "\n")

    def read(self) -> Iterable[LedgerEvent]:
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                data = json.loads(line)
                yield LedgerEvent(**data)
