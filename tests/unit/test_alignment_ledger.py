from constraint_lattice.ledger import CrossAgentAlignmentLedger

def test_ledger_record(tmp_path):
    path = tmp_path / 'ledger.jsonl'
    ledger = CrossAgentAlignmentLedger(str(path))
    ledger.record('agent', 'C1', 'applied', {'x': 1})
    events = list(ledger.read())
    assert len(events) == 1
    assert events[0].agent_id == 'agent'
    assert events[0].constraint == 'C1'
