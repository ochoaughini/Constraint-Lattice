from constraint_lattice.reflexivity import SemanticReflexivityIndex

def test_reflexivity_basic():
    idx = SemanticReflexivityIndex()
    assert idx.score == 0.0
    idx.record_self_correction()
    idx.record_external_correction()
    idx.record_violation()
    assert idx.score != 0.0
