import logging
from constraint_lattice.constraints.length import LengthConstraint
from constraint_lattice.constraints.profanity import ProfanityFilter
from constraint_lattice.engine.apply import apply_constraints

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Test constraints
constraints = [
    ProfanityFilter(replacement="[FILTERED]"),
    LengthConstraint(max_length=40, truncate=True, ellipsis="[...]")
]

# Test text
text = "This is a bad long sentence that needs filtering and truncation"

# Apply constraints
processed, trace = apply_constraints(
    prompt="", 
    output=text, 
    constraints=constraints, 
    return_trace=True
)

print(f"Original: {text}")
print(f"Processed: {processed}")
print(f"Steps: {len(trace)}")
for step in trace:
    print(f"  {step.constraint}: {step.pre_text[:20]}... -> {step.post_text[:20]}...")
