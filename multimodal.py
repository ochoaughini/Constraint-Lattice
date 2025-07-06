# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Multimodal support stub for Constraint Lattice
# Extend this module to handle text, image, audio, and video constraints.

from dataclasses import dataclass
from typing import Optional


@dataclass
class MultimodalInput:
    """Container for multimodal content passed to Constraint Lattice."""

    text: Optional[str] = None
    image: Optional[bytes] = None  # raw bytes or path/URI in future
    audio: Optional[bytes] = None
    video: Optional[bytes] = None

    def as_dict(self):
        """Return a serialisable representation of the multimodal input."""
        return {
            "text": self.text,
            "image": "<bytes>" if self.image else None,
            "audio": "<bytes>" if self.audio else None,
            "video": "<bytes>" if self.video else None,
        }


def apply_multimodal_constraints(input_obj: MultimodalInput):
    """Stub for future multimodal constraint enforcement."""
    raise NotImplementedError(
        "Multimodal constraint enforcement is not yet implemented. "
        "Track progress in issue #multimodal-support."
    )


# Extend Constraint classes and apply_constraints to handle these types.
