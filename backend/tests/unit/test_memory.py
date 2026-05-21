"""Unit tests for memory vector store."""
import pytest
import math


@pytest.mark.unit
class TestVectorStore:
    def test_cosine_similarity_identical(self):
        from app.memory.vector_store import cosine_similarity
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        from app.memory.vector_store import cosine_similarity
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert abs(cosine_similarity(v1, v2)) < 1e-6

    def test_vector_store_add_and_search(self):
        from app.memory.vector_store import VectorStore
        store = VectorStore()
        store.add("1", [1.0, 0.0, 0.0], {"text": "test"})
        store.add("2", [0.0, 1.0, 0.0], {"text": "other"})

        results = store.search([1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        assert results[0].id == "1"
        assert results[0].score > 0.99

    def test_vector_store_delete(self):
        from app.memory.vector_store import VectorStore
        store = VectorStore()
        store.add("1", [1.0, 0.0], {"text": "test"})
        store.delete("1")
        results = store.search([1.0, 0.0], k=10)
        assert all(r.id != "1" for r in results)
