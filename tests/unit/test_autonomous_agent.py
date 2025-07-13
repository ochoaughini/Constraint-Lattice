from autonomous_agent import AutonomousAgent


def test_event_classification_and_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = AutonomousAgent()
    agent.run(["normal", "fertile_exception", "fatal_contradiction"])

    assert len(agent.event_history) == 3
    assert agent.event_history[-1]["classification"] == "fatal"
    assert (tmp_path / "event_history.json").exists()
