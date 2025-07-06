# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import pytest
from constraint_lattice.constraints import ProfanityFilter


@pytest.mark.parametrize(
    "inp,expected",
    [
        ("This is shit.", "This is ***."),
        ("No bad words here", "No bad words here"),
        ("Fuck off", "*** off"),
    ],
)
def test_profanity_filter(inp, expected):
    filt = ProfanityFilter(profanity_list=["shit", "fuck", "ass"])
    assert filt.process_text(inp) == expected
