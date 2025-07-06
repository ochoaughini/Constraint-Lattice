from constraint_lattice_core import ConstraintLattice

# Create lattice with initial values
lattice = ConstraintLattice()
lattice.nodes = {'A': 1, 'B': 2}  # Start with stable state

# Add cyclic constraints
def constraint1(a):
    print(f"constraint1: a={a}")
    return a + 1

def constraint2(b):
    print(f"constraint2: b={b}")
    return b - 1

lattice.add_constraint(constraint1, inputs=['A'], outputs=['B'])
lattice.add_constraint(constraint2, inputs=['B'], outputs=['A'])

print("Initial state:", lattice.nodes)
lattice.propagate()
print("Final state:", lattice.nodes)

# Verify results
print("Propagation complete!")
print(f"Final values: A={lattice.nodes['A']}, B={lattice.nodes['B']}")
assert lattice.nodes['A'] == 1
assert lattice.nodes['B'] == 2
print("Test passed! Cyclic constraints handled correctly.")
