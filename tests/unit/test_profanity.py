# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import pytest

from constraints.constraint_profanity_filter import ConstraintProfanityFilter


@pytest.mark.parametrize(
    "inp,expected",
    [
        ("This is shit.", "This is ***."),
        ("No bad words here", "No bad words here"),
        ("Fuck off", "*** off"),
    ],
)
def test_profanity_filter(inp, expected):
    filt = ConstraintProfanityFilter()
    assert filt.filter_constraint("", inp) == expected
