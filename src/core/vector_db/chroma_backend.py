"""ChromaDB vector-database backend — persistent RAG with incremental indexing.

File changes are tracked via per-file MD5 hashes stored in
``.file_hashes.json`` next to the database directory.
"""

import json
import hashlib
import shutil
from time import time
from pathlib import Path
from tqdm import tqdm

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.config import cfg


class LocalEmbeddings:
    """Adapter wrapping SentenceTransformer so Chroma can use it."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text])[0].tolist()


class ChromaVectorDB:
    """Persistent vector store backed by ChromaDB.

    Usage::

        db = ChromaVectorDB()
        db.initialize()
        context = db.search("what did I do yesterday?")
    """

    def __init__(
        self,
        journal_path: str | Path | None = None,
        db_path: str | Path | None = None,
        model_name: str | None = None,
    ):
        self.journal_path = Path(journal_path or cfg.journal)
        self.db_path = Path(db_path or cfg.vector_db)
        self.model_name = model_name or cfg.rag_embedding_model
        self.vector_store = None

    # ── hash tracking ───────────────────────────────────────────

    def _hash_content(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _file_hashes_path(self) -> Path:
        return self.db_path / ".file_hashes.json"

    def _load_file_hashes(self) -> dict[str, str]:
        p = self._file_hashes_path()
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {}

    def _save_file_hashes(self, hashes: dict[str, str]):
        self._file_hashes_path().write_text(
            json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── initialisation ──────────────────────────────────────────

    def initialize(self):
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.journal_path.mkdir(parents=True, exist_ok=True)

        embed_model = LocalEmbeddings(self.model_name)
        db_exists = (self.db_path / "chroma.sqlite3").exists()

        if db_exists and not self._file_hashes_path().exists():
            print("Migrating from legacy DB format. Rebuilding...")
            shutil.rmtree(str(self.db_path))
            db_exists = False

        if not db_exists:
            print("Creating new vector database...")
            self._create_new_db(embed_model)
        else:
            self.vector_store = Chroma(
                persist_directory=str(self.db_path),
                embedding_function=embed_model,
            )
            self._update_incremental(embed_model)

    # ── fresh build ─────────────────────────────────────────────

    def _create_new_db(self, embed_model):
        md_files = sorted(self.journal_path.rglob("*.md"))
        docs: list[Document] = []
        for f in md_files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                if content.strip():
                    docs.append(Document(
                        page_content=content,
                        metadata={"source": f.name},
                    ))
            except Exception as e:
                print(f"Error reading {f}: {e}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.rag_chunk_size,
            chunk_overlap=cfg.rag_chunk_overlap,
        )
        chunks = splitter.split_documents(docs)
        total = len(chunks)
        print(f"\n[INFO] Total chunks to process: {total}")

        self.vector_store = Chroma(
            persist_directory=str(self.db_path),
            embedding_function=embed_model,
        )

        batch_size = 1000
        print(f"[INFO] Indexing in batches of {batch_size}...\n")
        with tqdm(total=total, desc="Creating Vector DB", unit="chunk") as pbar:
            for i in range(0, total, batch_size):
                batch = chunks[i : i + batch_size]
                t0 = time()
                self.vector_store.add_documents(batch)
                elapsed = time() - t0
                pbar.set_postfix({
                    "batch_s": f"{elapsed:.2f}",
                    "chunk/s": f"{len(batch) / elapsed:.1f}",
                })
                pbar.update(len(batch))

        print(f"\nVector DB created at {self.db_path}")
        file_hashes = {}
        for f in md_files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                file_hashes[f.name] = self._hash_content(content)
            except Exception as e:
                print(f"Error hashing {f}: {e}")
        self._save_file_hashes(file_hashes)

    # ── incremental update ──────────────────────────────────────

    def _update_incremental(self, embed_model):
        stored = self._load_file_hashes()
        current_files = {f.name: f for f in sorted(self.journal_path.rglob("*.md"))}
        current_hashes: dict[str, str] = {}
        any_change = False
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.rag_chunk_size,
            chunk_overlap=cfg.rag_chunk_overlap,
        )

        for name, f in current_files.items():
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"  Skipping {f}: {e}")
                continue
            content_hash = self._hash_content(content)
            current_hashes[name] = content_hash

            if name not in stored:
                any_change = True
                print(f"  + Adding: {name}")
                doc = Document(page_content=content, metadata={"source": name})
                chunks = splitter.split_documents([doc])
                if chunks:
                    self.vector_store.add_documents(chunks)
            elif stored[name] != content_hash:
                any_change = True
                print(f"  ~ Updating: {name}")
                try:
                    self.vector_store.delete(where={"source": name})
                except Exception:
                    pass
                doc = Document(page_content=content, metadata={"source": name})
                chunks = splitter.split_documents([doc])
                if chunks:
                    self.vector_store.add_documents(chunks)

        for name in stored:
            if name not in current_files:
                any_change = True
                print(f"  - Removing: {name}")
                try:
                    self.vector_store.delete(where={"source": name})
                except Exception:
                    pass

        if any_change:
            self._save_file_hashes(current_hashes)
            print("Incremental update complete.")
        else:
            print("No changes detected. Vector DB is up to date.")

    # ── retrieval ───────────────────────────────────────────────

    def _format_hits(self, hits, max_chars):
        parts: list[str] = []
        total = 0
        for h in hits:
            text = h.page_content.strip()
            if total + len(text) > max_chars:
                allowed = max_chars - total
                if allowed > 80:
                    parts.append(text[:allowed] + "...")
                break
            parts.append(text)
            total += len(text)
        return "\n---\n".join(parts)

    def search(self, query, k=None, max_chars=None):
        k = k if k is not None else cfg.rag_k
        max_chars = max_chars if max_chars is not None else cfg.rag_max_chars
        if not self.vector_store:
            return "No context available (Database not initialised)."
        try:
            hits = self.vector_store.similarity_search(query, k=k)
            return self._format_hits(hits, max_chars)
        except Exception as e:
            return f"RAG search error: {e}"

    def search_priority(
        self, query, priority_source, k=None, max_chars=None, priority_k=None
    ):
        k = k if k is not None else cfg.rag_k
        max_chars = max_chars if max_chars is not None else cfg.rag_max_chars
        priority_k = priority_k if priority_k is not None else cfg.rag_priority_k
        if not self.vector_store:
            return "No context available (Database not initialised)."
        try:
            seen = set()
            combined = []

            priority_hits = self.vector_store.similarity_search(
                query, k=priority_k, filter={"source": priority_source},
            )
            for h in priority_hits:
                h.page_content = f"[priority] {h.page_content.strip()}"
                combined.append(h)
                seen.add(self._hash_content(h.page_content))

            general_hits = self.vector_store.similarity_search(query, k=k)
            for h in general_hits:
                if len(combined) >= k:
                    break
                key = self._hash_content(h.page_content)
                if key not in seen:
                    seen.add(key)
                    combined.append(h)

            return self._format_hits(combined, max_chars)
        except Exception as e:
            return f"RAG priority search error: {e}"
