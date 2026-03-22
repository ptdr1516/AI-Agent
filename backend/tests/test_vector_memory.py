"""Lightweight checks for vector memory helpers (no FAISS / model load)."""

from agent.vector_memory import _make_turn_summary


def test_make_turn_summary_format():
    s = _make_turn_summary("Hello world", "Hi there")
    assert s.startswith("User:")
    assert "Assistant:" in s
    assert "Hello world" in s
    assert "Hi there" in s
