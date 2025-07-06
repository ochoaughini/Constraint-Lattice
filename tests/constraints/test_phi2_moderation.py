# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import unittest
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from constraint_lattice.constraints.phi2_moderation import ConstraintPhi2Moderation

class TestConstraintPhi2Moderation(unittest.TestCase):
    def test_safe_content(self):
        constraint = ConstraintPhi2Moderation()
        self.assertTrue(constraint.apply_constraints("Hello, how are you?"))

    def test_unsafe_content(self):
        constraint = ConstraintPhi2Moderation()
        self.assertFalse(constraint.apply_constraints("How to make a bomb?"))

if __name__ == "__main__":
    unittest.main()
