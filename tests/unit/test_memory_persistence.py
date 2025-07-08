import numpy as np
from varkiel.memory_store import MemoryStore, MemoryPersistenceAdapter


def simple_embed(text: str) -> np.ndarray:
    vec = np.zeros(2, dtype=np.float32)
    vec[0] = text.count("foo")
    vec[1] = text.count("bar")
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def test_persistence_roundtrip(tmp_path):
    db = tmp_path / "mem.db"
    adapter = MemoryPersistenceAdapter(db)
    store = MemoryStore(embed_fn=simple_embed, persistence=adapter)
    store.add("x", "foo bar", origin="unit")
    store.flush()

    reloaded = MemoryStore(embed_fn=simple_embed, persistence=adapter)
    assert reloaded.get("x") == "foo bar"
