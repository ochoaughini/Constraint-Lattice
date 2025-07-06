# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""Unit tests for ConstraintPhi2Moderation.

The real Phi-2 model is ~5 GB and not downloaded in CI.  We therefore patch the
private ``_analyse`` and ``_regenerate`` helpers so the tests stay lightweight
and deterministic.
"""

from unittest.mock import patch

import pytest

from constraints.phi2_moderation import ConstraintPhi2Moderation


@pytest.fixture(scope="module")
def moderator():
    # Instantiate with quantize=False to avoid bits-and-bytes CUDA deps even if
    # GPU runners are present.
    mod = ConstraintPhi2Moderation(quantize=False, fallback_strategy="block")
    # Inject dummy objects so the guard (model/tokenizer is None) is bypassed
    mod.model = object()  # type: ignore
    mod.tokenizer = object()  # type: ignore
    return mod


def test_safe_content(moderator):
    with patch.object(
        moderator, "_analyse", return_value={"is_safe": True, "violations": {}}
    ):
        assert moderator.moderate("Hello world!") == "Hello world!"


def test_blocking_strategy(moderator):
    moderator.fallback_strategy = "block"
    with patch.object(
        moderator,
        "_analyse",
        return_value={"is_safe": False, "violations": {"hate_speech": 0.99}},
    ):
        text = "I hate you!"
        moderated = moderator.moderate(text)
        assert moderated.startswith("[Content removed")


def test_mask_strategy(moderator):
    moderator.fallback_strategy = "mask"
    with patch.object(
        moderator,
        "_analyse",
        return_value={"is_safe": False, "violations": {"violence": 0.9}},
    ):
        moderated = moderator.moderate("Kill them!")
        assert moderated == "[REDACTED]"


def test_regenerate_strategy(moderator):
    moderator.fallback_strategy = "regenerate"
    with (
        patch.object(
            moderator,
            "_analyse",
            return_value={"is_safe": False, "violations": {"harassment": 0.8}},
        ),
        patch.object(
            moderator,
            "_regenerate",
            return_value="Let's keep things civil.",
        ),
    ):
        moderated = moderator.moderate("You are stupid!")
        assert moderated == "Let's keep things civil."
