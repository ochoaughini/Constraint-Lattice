from cognitive_arch import (
    AdaptiveEthics,
    EthicalRule,
    HierarchicalMemory,
    MetaConstraintLog,
    GovernanceCoordinator,
    Agent,
)


def test_hierarchical_memory(tmp_path):
    mem = HierarchicalMemory(tmp_path / "mem.json")
    mem.add(["session", "message"], "hello")
    assert mem.get(["session", "message"]) == "hello"


def test_meta_constraint_log(tmp_path):
    log = MetaConstraintLog(path=tmp_path / "log.json")
    log.log("rule1", "before", "after")
    assert log.summary()["rule1"] == 1


def test_governance_coordinator():
    def echo(msg: str) -> str:
        return msg
    gov = GovernanceCoordinator()
    gov.register(Agent("a", echo))
    assert gov.broadcast("hi") == {"a": "hi"}


def test_adaptive_ethics():
    ethics = AdaptiveEthics([EthicalRule(name="no-x", forbidden_terms=["x"])])
    assert ethics.check("contains x") == ["no-x"]
