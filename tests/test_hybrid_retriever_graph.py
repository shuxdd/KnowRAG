"""Tests for graph retrieval integration in hybrid_retriever.

These tests mock external services (Milvus, Redis, PostgreSQL) so they
can run without any infrastructure dependencies.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Install lightweight mocks into sys.modules BEFORE importing hybrid_retriever
# to prevent real Milvus / Redis / PostgreSQL connections.
# ---------------------------------------------------------------------------

def _make_fake_module(name: str, attrs: dict) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


# Build mock objects for the attributes that hybrid_retriever imports
_mock_parent_store = MagicMock(name="parent_store")
_mock_parent_store.get_by_ids.return_value = []

_mock_embedding_service = MagicMock(name="embedding_service")
_mock_embedding_service.embed.return_value = []

_mock_vector_service = MagicMock(name="vector_service")
_mock_vector_service.get_all_chunks.return_value = []
_mock_vector_service.similarity_search.return_value = []

_mock_redis_module = MagicMock(name="redis_module")
_mock_redis_client = MagicMock(name="redis_client")
_mock_redis_module.Redis.from_url.return_value = _mock_redis_client

# Map of module_name -> mock module to install
_MOCK_MODULES = {
    "redis": _make_fake_module("redis", {
        "Redis": _mock_redis_module.Redis,
    }),
    "backend.services.parent_store": _make_fake_module("backend.services.parent_store", {
        "parent_store": _mock_parent_store,
    }),
    "backend.services.embedding_service": _make_fake_module("backend.services.embedding_service", {
        "embedding_service": _mock_embedding_service,
    }),
    "backend.services.vector_service": _make_fake_module("backend.services.vector_service", {
        "vector_service": _mock_vector_service,
    }),
}

# Save originals so we can restore after import
_original_modules: dict = {}
for _name, _mock in _MOCK_MODULES.items():
    _original_modules[_name] = sys.modules.get(_name)
    sys.modules[_name] = _mock

# Remove cached hybrid_retriever so it re-imports with our mocks
if "backend.services.hybrid_retriever" in sys.modules:
    del sys.modules["backend.services.hybrid_retriever"]

try:
    from backend.services.hybrid_retriever import rrf_fusion, HybridRetriever
finally:
    # Restore original modules (so other test files aren't affected)
    for _name, _orig in _original_modules.items():
        if _orig is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _orig


# ---------------------------------------------------------------------------
# Tests: RRF Fusion with graph results
# ---------------------------------------------------------------------------

class TestRRFFusionWithGraph:
    """Verify RRF fusion works correctly with 3 doc_lists (including graph)."""

    def test_graph_results_included_in_rrf_fusion(self):
        """Graph results should be included as a third doc_list in RRF fusion."""
        doc_a = Document(page_content="content A", metadata={})
        doc_b = Document(page_content="content B", metadata={})
        doc_c = Document(page_content="content C", metadata={})

        # doc_c only appears in graph results (third list)
        fused = rrf_fusion(
            doc_lists=[[doc_a, doc_b], [doc_b, doc_a], [doc_c, doc_a]],
            k=60,
            top_n=3,
        )
        contents = [d.page_content for d in fused]
        # doc_a appears in all 3 lists -> highest score
        assert "content A" in contents
        # doc_b appears in 2 lists -> second highest
        assert "content B" in contents
        # doc_c appears in 1 list -> included since top_n=3
        assert "content C" in contents

    def test_three_way_rrf_ranking_order(self):
        """Documents appearing in more lists should rank higher."""
        doc_shared = Document(page_content="shared", metadata={})
        doc_vec_only = Document(page_content="vec_only", metadata={})
        doc_graph_only = Document(page_content="graph_only", metadata={})

        fused = rrf_fusion(
            doc_lists=[[doc_shared, doc_vec_only], [doc_shared], [doc_shared, doc_graph_only]],
            k=60,
            top_n=3,
        )
        # doc_shared appears in all 3 -> should be first
        assert fused[0].page_content == "shared"


# ---------------------------------------------------------------------------
# Tests: New HybridRetriever methods
# ---------------------------------------------------------------------------

class TestHybridRetrieverGraphMethods:
    """Test new methods added to HybridRetriever."""

    def test_update_retrieval_stats_method_exists(self):
        """_update_retrieval_stats should exist as a method on HybridRetriever."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        assert hasattr(retriever, '_update_retrieval_stats')
        assert callable(getattr(retriever, '_update_retrieval_stats', None))

    def test_graph_retrieve_method_exists(self):
        """_graph_retrieve should exist as a method on HybridRetriever."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        assert hasattr(retriever, '_graph_retrieve')
        assert callable(getattr(retriever, '_graph_retrieve', None))

    def test_trigger_extraction_method_exists(self):
        """_trigger_extraction should exist as a method on HybridRetriever."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        assert hasattr(retriever, '_trigger_extraction')
        assert callable(getattr(retriever, '_trigger_extraction', None))

    def test_expand_to_parents_accepts_user_id(self):
        """_expand_to_parents should accept a user_id parameter."""
        import inspect
        sig = inspect.signature(HybridRetriever._expand_to_parents)
        assert "user_id" in sig.parameters
