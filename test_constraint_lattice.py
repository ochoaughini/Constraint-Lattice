# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. All rights reserved.
# See LICENSE for full terms.

class TestConstraintLattice(unittest.TestCase):
    def test_acyclic_constraints(self):
        lattice = ConstraintLattice()
        lattice.nodes = {'A': 1, 'B': None, 'C': None}
        
        def constraint1(a):
            return a + 1
        
        def constraint2(b):
            return b * 2
            
        lattice.add_constraint(constraint1, inputs=['A'], outputs=['B'])
        lattice.add_constraint(constraint2, inputs=['B'], outputs=['C'])
        lattice.propagate()
        
        self.assertEqual(lattice.nodes['B'], 2)
        self.assertEqual(lattice.nodes['C'], 4)
        
    def test_cyclic_constraints(self):
        lattice = ConstraintLattice()
        lattice.nodes = {'A': 1, 'B': 2}
        
        def constraint1(a):
            return a + 1
        
        def constraint2(b):
            return b - 1
            
        lattice.add_constraint(constraint1, inputs=['A'], outputs=['B'])
        lattice.add_constraint(constraint2, inputs=['B'], outputs=['A'])
        lattice.propagate()
        
        self.assertEqual(lattice.nodes['A'], 1)
        self.assertEqual(lattice.nodes['B'], 2)
        
    def test_three_node_cycle(self):
        lattice = ConstraintLattice()
        lattice.nodes = {'A': 1, 'B': 0, 'C': 0}
        
        def constraint1(a, c):
            return a + c
        
        def constraint2(b):
            return b + 1
            
        def constraint3(b):
            return b * 2
            
        lattice.add_constraint(constraint1, inputs=['A', 'C'], outputs=['B'])
        lattice.add_constraint(constraint2, inputs=['B'], outputs=['C'])
        lattice.add_constraint(constraint3, inputs=['B'], outputs=['A'])
        lattice.propagate()
        
        # Verify convergence
        self.assertEqual(lattice.nodes['A'], lattice.nodes['B'] * 2)
        self.assertEqual(lattice.nodes['C'], lattice.nodes['B'] + 1)
        self.assertEqual(lattice.nodes['B'], lattice.nodes['A'] + lattice.nodes['C'])

if __name__ == '__main__':
    unittest.main()
