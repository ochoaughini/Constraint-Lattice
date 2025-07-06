# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import unittest
from clattice.integration_hub import IntegrationHub

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.hub = IntegrationHub()
    
    def test_security_rejection(self):
        """Test security screening rejects malicious input"""
        result = self.hub.process_input("malicious prompt")
        self.assertIn("Rejected", result)
    
    def test_constraint_processing(self):
        """Test constraint processing"""
        result = self.hub.process_input("valid input")
        self.assertIn("Lattice-processed", result)

if __name__ == "__main__":
    unittest.main()
