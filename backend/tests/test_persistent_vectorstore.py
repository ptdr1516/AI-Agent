"""PersistentVectorStore: FAISS + Chroma persistence (FakeEmbeddings, no HF)."""

import pytest
from langchain_core.documents import Document

from langchain_community.embeddings import DeterministicFakeEmbedding

from rag.vectorstore import PersistentVectorStore, _faiss_index_exists


@pytest.fixture
def fake_emb():
    return DeterministicFakeEmbedding(size=16)


@pytest.mark.asyncio
async def test_faiss_persist_roundtrip(tmp_path, fake_emb, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    pdir = tmp_path / "faiss_idx"
    docs = [Document(page_content="hello world", metadata={"source": "a.txt"})]

    s1 = await PersistentVectorStore.open(
        persist_path=str(pdir),
        backend="faiss",
        embedding=fake_emb,
        default_top_k=2,
    )
    await s1.aadd_documents(docs)
    await s1.apersist()
    assert _faiss_index_exists(pdir)

    s2 = await PersistentVectorStore.open(
        persist_path=str(pdir),
        backend="faiss",
        embedding=fake_emb,
        default_top_k=2,
    )
    hits = await s2.asimilarity_search("hello")
    assert len(hits) >= 1
    assert "hello" in hits[0].page_content


@pytest.mark.asyncio
async def test_chroma_persist_and_search(tmp_path, fake_emb, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    pdir = tmp_path / "chroma_data"

    s1 = await PersistentVectorStore.open(
        persist_path=str(pdir),
        backend="chroma",
        embedding=fake_emb,
        collection_name="test_rag",
        default_top_k=3,
    )
    await s1.aadd_documents(
        [
            Document(page_content="alpha beta", metadata={"source": "1"}),
            Document(page_content="gamma delta", metadata={"source": "2"}),
        ]
    )
    await s1.apersist()

    s2 = await PersistentVectorStore.open(
        persist_path=str(pdir),
        backend="chroma",
        embedding=fake_emb,
        collection_name="test_rag",
        default_top_k=3,
    )
    hits = await s2.asimilarity_search("alpha", k=1)
    assert len(hits) == 1
