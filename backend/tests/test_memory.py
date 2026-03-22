"""Tests for memory persistence.

Uses the MemoryManager singleton which is backed by SQLite, validating that
messages written in one call are retrievable in a subsequent call — simulating
production request-across-restart behaviour.
"""
import pytest
from agent.memory import memory_manager


def test_memory_persistence():
    """Messages added to a session should survive between get_memory calls."""
    session_id = "test-persist-123"

    # Ensure a clean slate
    memory_manager.clear_memory(session_id)

    # Write messages
    mem = memory_manager.get_memory(session_id)
    mem.chat_memory.add_user_message("Hello, my name is Alice.")
    mem.chat_memory.add_ai_message("Hello Alice, I am an AI.")

    # Simulate a new request by evicting from the in-memory cache...
    if session_id in memory_manager._sessions:
        del memory_manager._sessions[session_id]

    # ...then re-loading from SQLite
    mem_reloaded = memory_manager.get_memory(session_id)
    messages = mem_reloaded.chat_memory.messages

    assert len(messages) == 2
    assert messages[0].content == "Hello, my name is Alice."
    assert messages[1].content == "Hello Alice, I am an AI."

    # Cleanup
    memory_manager.clear_memory(session_id)


def test_memory_isolation():
    """Two different sessions must not share messages."""
    s1 = "isolation-session-A"
    s2 = "isolation-session-B"

    memory_manager.clear_memory(s1)
    memory_manager.clear_memory(s2)

    memory_manager.get_memory(s1).chat_memory.add_user_message("Secret A")
    memory_manager.get_memory(s2).chat_memory.add_user_message("Secret B")

    assert memory_manager.get_memory(s1).chat_memory.messages[0].content == "Secret A"
    assert memory_manager.get_memory(s2).chat_memory.messages[0].content == "Secret B"

    memory_manager.clear_memory(s1)
    memory_manager.clear_memory(s2)


def test_clear_memory_removes_messages():
    """clear_memory should wipe everything from both cache and DB."""
    session_id = "test-clear-mem"
    memory_manager.clear_memory(session_id)

    mem = memory_manager.get_memory(session_id)
    mem.chat_memory.add_user_message("This should be deleted.")
    memory_manager.clear_memory(session_id)

    fresh = memory_manager.get_memory(session_id)
    assert len(fresh.chat_memory.messages) == 0

    memory_manager.clear_memory(session_id)
