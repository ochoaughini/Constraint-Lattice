import numpy as np

from varkiel.memory_store import MemoryStore


def simple_embed(text: str) -> np.ndarray:
    vec = np.zeros(3, dtype=np.float32)
    vec[0] = text.count("hello")
    vec[1] = text.count("goodbye")
    vec[2] = text.count("world")
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def test_similarity_search():
    store = MemoryStore(embed_fn=simple_embed)
    store.add("a", "hello world", origin="test")
    store.add("b", "goodbye world", origin="test")
    results = list(store.search_similar("hello", threshold=0.5))
    assert "hello world" in results
