from typing import Any

from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from core.config import settings
from core.logger import log
from .vector_memory import vector_memory_store


class ConversationBufferWindowMemoryWithVectorContext(ConversationBufferWindowMemory):
    """Extends window memory so extra prompt-only keys (e.g. FAISS context) are not saved.

    LangChain's save_context expects exactly one user input key; passing `memory_context`
    alongside `input` would raise ValueError. We strip non-persistent keys before save.
    """

    _PROMPT_ONLY_KEYS = frozenset({"memory_context"})

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        cleaned = {k: v for k, v in inputs.items() if k not in self._PROMPT_ONLY_KEYS}
        super().save_context(cleaned, outputs)


class MemoryManager:
    """Manages per-session chat histories backed by SQLite for robust persistence."""

    def __init__(self):
        self._sessions: dict[str, ConversationBufferWindowMemoryWithVectorContext] = {}
        log.info(
            f"MemoryManager initialised — "
            f"db={settings.MEMORY_DB_PATH}, window={settings.MEMORY_WINDOW_SIZE}"
        )

    def get_memory(self, session_id: str) -> ConversationBufferWindowMemoryWithVectorContext:
        """Retrieves or creates SQLite-backed memory for a given session."""
        if session_id not in self._sessions:
            log.info(f"Loading persistent SQL memory for session: {session_id}")
            chat_history = SQLChatMessageHistory(
                session_id=session_id,
                connection_string=settings.MEMORY_DB_PATH,
            )
            self._sessions[session_id] = ConversationBufferWindowMemoryWithVectorContext(
                chat_memory=chat_history,
                memory_key="chat_history",
                return_messages=True,
                output_key="output",
                k=settings.MEMORY_WINDOW_SIZE,
            )
        return self._sessions[session_id]

    def clear_memory(self, session_id: str) -> None:
        """Permanently clears memory for a session from both the cache and SQLite."""
        vector_memory_store.clear_session(session_id)
        if session_id in self._sessions:
            self._sessions[session_id].chat_memory.clear()
            del self._sessions[session_id]
            log.info(f"Wiped memory for session: {session_id}")
        else:
            # Session exists in DB but not in the hot cache — wipe it cold
            SQLChatMessageHistory(
                session_id=session_id,
                connection_string=settings.MEMORY_DB_PATH,
            ).clear()
            log.info(f"Wiped cold memory for session: {session_id}")


# Global singleton — one MemoryManager serves all sessions
memory_manager = MemoryManager()

