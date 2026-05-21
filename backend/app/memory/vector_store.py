"""
In-process vector similarity search store.

No external vector database required.  Embeddings and metadata are held in
numpy arrays; persisted to disk via pickle for durability between restarts.

Classes:
- SearchResult:  typed result of a similarity search
- VectorStore:   add / search / delete / persist / load operations
- cosine_similarity: standalone utility function
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Args:
        a: First vector (list of floats).
        b: Second vector (list of floats).

    Returns:
        Float in [-1, 1]; higher is more similar.
        Returns 0.0 if either vector is a zero vector.
    """
    if not a or not b:
        return 0.0

    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)

    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(va, vb) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Single result from a vector similarity search."""

    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<SearchResult id={self.id!r} score={self.score:.4f}>"


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------


class VectorStore:
    """
    Thread-safe in-process vector store backed by numpy arrays.

    Supports:
    - O(n) cosine similarity search (suitable for up to ~100k embeddings)
    - Optional metadata filtering
    - Disk persistence via pickle

    For production deployments with > 100k vectors, replace with
    Qdrant, Weaviate, or pgvector.
    """

    def __init__(self) -> None:
        # ids[i] corresponds to embeddings[i] and metadata_list[i]
        self._ids: list[str] = []
        self._embeddings: list[np.ndarray] = []  # each shape: (dim,)
        self._metadata: list[dict[str, Any]] = []

        # Index for O(1) id → position lookup
        self._id_to_index: dict[str, int] = {}

        self._lock = threading.RLock()

        logger.debug("VectorStore initialised.")

    # ------------------------------------------------------------------
    # Mutation operations
    # ------------------------------------------------------------------

    def add(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a vector to the store.

        If *id* already exists the call is silently ignored.  Use
        ``update()`` to modify an existing entry.

        Args:
            id:        Unique string identifier.
            embedding: Dense float vector (must be consistent dimension).
            metadata:  Arbitrary key-value pairs stored alongside the vector.
        """
        with self._lock:
            if id in self._id_to_index:
                logger.debug("VectorStore.add: id %r already exists; skipped.", id)
                return

            vec = np.array(embedding, dtype=np.float32)
            index = len(self._ids)

            self._ids.append(id)
            self._embeddings.append(vec)
            self._metadata.append(metadata or {})
            self._id_to_index[id] = index

    def update(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Update an existing vector and / or its metadata.

        If *id* does not exist, adds a new entry.

        Args:
            id:        Existing vector identifier.
            embedding: New embedding vector.
            metadata:  New metadata (replaces existing metadata entirely).
        """
        with self._lock:
            if id not in self._id_to_index:
                self.add(id, embedding, metadata)
                return

            idx = self._id_to_index[id]
            self._embeddings[idx] = np.array(embedding, dtype=np.float32)
            self._metadata[idx] = metadata or {}

    def delete(self, id: str) -> bool:
        """
        Remove the vector with the given *id*.

        Uses swap-and-pop for O(1) deletion.

        Args:
            id: Vector identifier to remove.

        Returns:
            True if found and deleted; False if not found.
        """
        with self._lock:
            if id not in self._id_to_index:
                return False

            idx = self._id_to_index.pop(id)
            last_idx = len(self._ids) - 1

            if idx != last_idx:
                # Swap with last element
                last_id = self._ids[last_idx]
                self._ids[idx] = last_id
                self._embeddings[idx] = self._embeddings[last_idx]
                self._metadata[idx] = self._metadata[last_idx]
                self._id_to_index[last_id] = idx

            self._ids.pop()
            self._embeddings.pop()
            self._metadata.pop()
            return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Return the *k* most similar vectors to *query_embedding*.

        Args:
            query_embedding: Query vector.
            k:               Maximum number of results.
            filter:          Optional dict of metadata key-value pairs; only
                             entries where ALL filter conditions match are
                             considered.

        Returns:
            List of SearchResult objects sorted by descending similarity score.
        """
        with self._lock:
            if not self._ids:
                return []

            q = np.array(query_embedding, dtype=np.float32)
            q_norm = float(np.linalg.norm(q))
            if q_norm == 0.0:
                return []
            q = q / q_norm

            # Build candidate indices (respecting filter)
            if filter:
                candidates = [
                    i
                    for i, meta in enumerate(self._metadata)
                    if all(meta.get(key) == val for key, val in filter.items())
                ]
            else:
                candidates = list(range(len(self._ids)))

            if not candidates:
                return []

            # Batch cosine similarity: stack candidate vectors into matrix
            mat = np.stack(
                [self._embeddings[i] for i in candidates], axis=0
            )  # shape: (n_candidates, dim)

            # Normalise rows
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms

            scores = mat @ q  # shape: (n_candidates,)

            # Top-k
            k_actual = min(k, len(candidates))
            top_indices = np.argpartition(scores, -k_actual)[-k_actual:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            results: list[SearchResult] = []
            for local_idx in top_indices:
                global_idx = candidates[int(local_idx)]
                results.append(
                    SearchResult(
                        id=self._ids[global_idx],
                        score=float(scores[local_idx]),
                        metadata=dict(self._metadata[global_idx]),
                    )
                )

            return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self, path: str) -> None:
        """
        Save the vector store to disk using pickle.

        Args:
            path: File path to write (will be created or overwritten).
        """
        with self._lock:
            state = {
                "ids": self._ids,
                "embeddings": self._embeddings,
                "metadata": self._metadata,
                "id_to_index": self._id_to_index,
            }
            tmp_path = path + ".tmp"
            try:
                with open(tmp_path, "wb") as f:
                    pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_path, path)
                logger.info(
                    "VectorStore persisted: %d vectors → %s",
                    len(self._ids),
                    path,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("VectorStore.persist failed: %s", exc)
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

    def load(self, path: str) -> None:
        """
        Load vector store state from a pickle file.

        Replaces any existing in-memory data.

        Args:
            path: File path to read.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"VectorStore file not found: {path}")

        with self._lock:
            with open(path, "rb") as f:
                state = pickle.load(f)

            self._ids = state["ids"]
            self._embeddings = state["embeddings"]
            self._metadata = state["metadata"]
            self._id_to_index = state["id_to_index"]

        logger.info(
            "VectorStore loaded: %d vectors from %s",
            len(self._ids),
            path,
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._ids)

    def __contains__(self, id: str) -> bool:
        with self._lock:
            return id in self._id_to_index

    def get_metadata(self, id: str) -> dict[str, Any] | None:
        """Return the metadata dict for *id*, or None if not found."""
        with self._lock:
            idx = self._id_to_index.get(id)
            if idx is None:
                return None
            return dict(self._metadata[idx])

    def ids(self) -> list[str]:
        """Return a copy of all stored IDs."""
        with self._lock:
            return list(self._ids)
