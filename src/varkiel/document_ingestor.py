from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Dict

import PyPDF2

from .constraint_lattice import ConstraintLattice


class DocumentIngestor:
    """Parse documents and register constraints."""

    def __init__(self, lattice: ConstraintLattice) -> None:
        self.lattice = lattice

    def ingest_pdf(self, path: str | Path, *, source: str) -> None:
        reader = PyPDF2.PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            node_id = f"{source}_{i}"
            self.lattice.add_node(node_id, line, source=source)

    def ingest_json(self, path: str | Path, *, source: str) -> None:
        data = json.loads(Path(path).read_text())
        for key, value in data.items():
            node_id = f"{source}_{key}"
            self.lattice.add_node(node_id, str(value), source=source)

