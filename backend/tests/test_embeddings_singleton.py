"""Embeddings singleton behavior (no real model load)."""

from unittest.mock import MagicMock, patch

from rag.embeddings import get_embedding_model, reset_embedding_model_cache


def test_embedding_singleton_single_factory_call():
    reset_embedding_model_cache()
    mock_emb = MagicMock()
    with patch("rag.embeddings._create_embeddings", return_value=mock_emb) as factory:
        a = get_embedding_model()
        b = get_embedding_model()
        assert a is b is mock_emb
        assert factory.call_count == 1
