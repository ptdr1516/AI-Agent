"""Lightweight RAG module tests (no embedding model load)."""

from pathlib import Path

import pytest

from rag.chunker import split_documents
from rag.loader import documents_from_strings, load_pdf_file, load_text_file
from rag.rag_chain import format_documents


def test_documents_from_strings():
    docs = documents_from_strings(["a", "b"])
    assert len(docs) == 2
    assert docs[0].page_content == "a"


def test_load_text_metadata(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("hello", encoding="utf-8")
    docs = load_text_file(p)
    assert len(docs) == 1
    md = docs[0].metadata
    assert md["filename"] == "note.txt"
    assert md["page"] == 1
    assert Path(md["source"]) == p.resolve()


def test_load_pdf_metadata(tmp_path):
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=72, height=72)
    path = tmp_path / "one.pdf"
    with path.open("wb") as f:
        w.write(f)
    docs = load_pdf_file(path)
    assert len(docs) >= 1
    assert docs[0].metadata["filename"] == "one.pdf"
    assert docs[0].metadata["page"] == 1


def test_split_documents_recursive_and_metadata(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("word " * 200, encoding="utf-8")
    docs = load_text_file(p)
    chunks = split_documents(docs, chunk_size=80, chunk_overlap=10)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.metadata.get("filename") == "t.txt"
        assert "source" in c.metadata
        assert c.metadata.get("page") == 1


def test_format_documents():
    from langchain_core.documents import Document

    s = format_documents([Document(page_content="hello")])
    assert "hello" in s
    assert "[1]" in s
    assert "Source:" in s


def test_build_context_and_extract_sources():
    from langchain_core.documents import Document

    from rag.rag_chain import build_context, extract_sources

    docs = [
        Document(page_content="a", metadata={"filename": "one.txt"}),
        Document(page_content="b", metadata={"filename": "two.txt"}),
    ]
    ctx = build_context(docs)
    assert "one.txt" in ctx and "two.txt" in ctx
    assert extract_sources(docs) == ["one.txt", "two.txt"]


def test_parse_build_context_roundtrip():
    from langchain_core.documents import Document

    from rag.rag_chain import build_context, parse_build_context

    docs = [
        Document(page_content="body one", metadata={"filename": "a.pdf"}),
        Document(page_content="body two\nline", metadata={"filename": "b.pdf"}),
    ]
    ctx = build_context(docs)
    parsed = parse_build_context(ctx)
    assert parsed["sources"] == ["a.pdf", "b.pdf"]
    assert len(parsed["chunks"]) == 2
    assert parsed["chunks"][0]["filename"] == "a.pdf"
    assert parsed["chunks"][0]["preview"] == "body one"
    assert parsed["chunks"][1]["preview"] == "body two\nline"
