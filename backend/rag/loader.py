"""Document loaders (LangChain) for PDF, text, and Markdown with unified metadata."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Iterable, Sequence

from langchain_core.documents import Document


def _install_pwd_stub_on_windows() -> None:
    """`langchain_community.document_loaders` imports `pebblo` which imports `pwd` (Unix-only).

    Stub `pwd` on Windows so PyPDFLoader / TextLoader imports succeed; PebbloSafeLoader is unused here.
    """
    if sys.platform != "win32" or "pwd" in sys.modules:
        return

    pwd_mod = types.ModuleType("pwd")

    class _Pw:
        pw_name = "unknown"

    def getpwuid(_uid: int) -> _Pw:  # noqa: ANN001
        return _Pw()

    def getpwnam(_name: str) -> _Pw:  # noqa: ANN001
        return _Pw()

    pwd_mod.getpwuid = getpwuid  # type: ignore[attr-defined]
    pwd_mod.getpwnam = getpwnam  # type: ignore[attr-defined]
    sys.modules["pwd"] = pwd_mod


_install_pwd_stub_on_windows()


def standardize_document_metadata(documents: list[Document], file_path: str | Path) -> list[Document]:
    """Ensure every document has ``source``, ``filename``, and ``page`` (1-based).

    PyPDFLoader/LangChain uses 0-based ``page``; we normalize to 1-based page numbers.
    """
    resolved = Path(file_path).resolve()
    filename = resolved.name
    out: list[Document] = []
    for doc in documents:
        md = {**(doc.metadata or {})}
        md["source"] = str(resolved)
        md["filename"] = filename
        if "page" in md and md["page"] is not None:
            try:
                md["page"] = int(md["page"]) + 1
            except (TypeError, ValueError):
                md["page"] = 1
        else:
            md["page"] = 1
        out.append(Document(page_content=doc.page_content, metadata=md))
    return out


def load_pdf_file(path: str | Path) -> list[Document]:
    """Load a PDF with LangChain ``PyPDFLoader`` (one document per page, ``pypdf``)."""
    from langchain_community.document_loaders import PyPDFLoader

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    docs = PyPDFLoader(str(p)).load()
    return standardize_document_metadata(docs, p)


def load_text_file(path: str | Path, *, encoding: str | None = "utf-8") -> list[Document]:
    """Load a plain text file with LangChain ``TextLoader``."""
    from langchain_community.document_loaders import TextLoader

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    docs = TextLoader(str(p), encoding=encoding).load()
    return standardize_document_metadata(docs, p)


def load_markdown_file(path: str | Path, *, encoding: str | None = "utf-8") -> list[Document]:
    """Load a Markdown file with LangChain ``TextLoader`` (UTF-8 text)."""
    from langchain_community.document_loaders import TextLoader

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    docs = TextLoader(str(p), encoding=encoding).load()
    return standardize_document_metadata(docs, p)


def load_file(path: str | Path, *, encoding: str | None = "utf-8") -> list[Document]:
    """Dispatch by extension: ``.pdf`` → PyPDFLoader; ``.md`` / ``.markdown`` / text → TextLoader."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".pdf":
        return load_pdf_file(p)
    if suf in {".md", ".markdown", ".txt", ".text"}:
        if suf in {".md", ".markdown"}:
            return load_markdown_file(p, encoding=encoding)
        return load_text_file(p, encoding=encoding)
    raise ValueError(f"Unsupported file type: {suf} (path={p})")


def load_text_files(paths: Sequence[str | Path], *, encoding: str | None = "utf-8") -> list[Document]:
    """Load many text files; order preserved."""
    out: list[Document] = []
    for path in paths:
        out.extend(load_text_file(path, encoding=encoding))
    return out


def load_directory(
    directory: str | Path,
    *,
    extensions: Sequence[str] = (".txt", ".text", ".md", ".markdown", ".pdf"),
    recursive: bool = True,
    encoding: str | None = "utf-8",
) -> list[Document]:
    """Load all files under ``directory`` whose suffix is in ``extensions``."""
    base = Path(directory)
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {base}")
    ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    if recursive:
        candidates = [p for p in base.rglob("*") if p.is_file()]
    else:
        candidates = [p for p in base.glob("*") if p.is_file()]
    candidates.sort(key=lambda x: str(x))
    out: list[Document] = []
    for p in candidates:
        if p.suffix.lower() not in ext_set:
            continue
        try:
            out.extend(load_file(p, encoding=encoding))
        except OSError:
            continue
    return out


def documents_from_strings(
    texts: Iterable[str],
    *,
    metadatas: Iterable[dict] | None = None,
) -> list[Document]:
    """Build Documents from raw strings (optional parallel metadatas).

    Per-string metadata may include ``source`` / ``filename`` / ``page``; missing keys are not filled.
    """
    texts_list = list(texts)
    if metadatas is None:
        return [Document(page_content=t) for t in texts_list]
    meta_list = list(metadatas)
    if len(meta_list) != len(texts_list):
        raise ValueError("metadatas must be the same length as texts")
    return [
        Document(page_content=t, metadata=m) for t, m in zip(texts_list, meta_list)
    ]
