"""RAGRetriever: top_k, payloads, documents + metadata."""

from langchain_core.documents import Document

from rag.retriever import RAGRetriever, documents_to_payloads


class _FakeVS:
    """Minimal vector store stub."""

    def __init__(self):
        self.last_k = None

    async def asimilarity_search(self, query: str, k: int = 4, **kwargs):
        self.last_k = k
        return [
            Document(
                page_content="chunk",
                metadata={"source": "x", "page": 1},
            )
        ]

    def similarity_search(self, query: str, k: int = 4, **kwargs):
        self.last_k = k
        return [
            Document(
                page_content="chunk",
                metadata={"source": "x", "page": 1},
            )
        ]


def test_documents_to_payloads():
    docs = [Document(page_content="a", metadata={"m": 1})]
    p = documents_to_payloads(docs)
    assert p == [{"page_content": "a", "metadata": {"m": 1}}]


def test_retriever_top_k_override():
    vs = _FakeVS()
    r = RAGRetriever(vs, default_top_k=4)
    import asyncio

    async def _run():
        await r.aretrieve("q", top_k=7)
        assert vs.last_k == 7
        await r.aretrieve("q", k=2)
        assert vs.last_k == 2

    asyncio.run(_run())


def test_retrieve_payloads():
    vs = _FakeVS()
    r = RAGRetriever(vs)
    p = r.retrieve_payloads("hi")
    assert p[0]["page_content"] == "chunk"
    assert p[0]["metadata"]["source"] == "x"
