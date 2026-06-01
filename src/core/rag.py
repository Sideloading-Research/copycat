import os
import json

# Thread limits are set globally in ``setup_env`` (imported earlier during
# startup).  Here we only disable tokeniser parallelism to avoid deadlocks
# with HuggingFace's internal multiprocessing.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm
from time import time
import hashlib
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class LocalEmbeddings:
    """Adapter that wraps SentenceTransformer so Chroma can use it."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text])[0].tolist()


class RAGManager:
    """Manages a persistent Chroma vector store over a journal of .md files.

    Behaviour
    ---------
    - On first run, the entire journal is chunked and indexed into Chroma.
    - On subsequent runs, *only changed files* are re-indexed (incremental).
      Unchanged files are skipped, making repeated startups fast.
    - File identity is tracked via an MD5 hash stored in
      ``data/vector_db/.file_hashes.json``.

    Usage
    -----
        rag = RAGManager(journal_path, db_path, model_name)
        rag.initialize()
        context = rag.search("what did I do yesterday?")
    """

    def __init__(self, journal_path, db_path, model_name):
        self.journal_path = Path(journal_path)
        self.db_path = Path(db_path)
        self.model_name = model_name
        self.vector_store = None

    # ── per-file hash tracking ───────────────────────────────────

    def _hash_content(self, content: str) -> str:
        """Return MD5 hex digest of *content* (not file name / size)."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _file_hashes_path(self) -> Path:
        return self.db_path / ".file_hashes.json"

    def _load_file_hashes(self) -> dict[str, str]:
        """Load the persisted filename→hash mapping, or return empty dict."""
        p = self._file_hashes_path()
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {}

    def _save_file_hashes(self, hashes: dict[str, str]):
        self._file_hashes_path().write_text(
            json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── initialisation ───────────────────────────────────────────

    def initialize(self):
        """Load existing DB (incrementally updated) or create a new one."""
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.journal_path.mkdir(parents=True, exist_ok=True)

        embed_model = LocalEmbeddings(self.model_name)
        db_exists = (self.db_path / "chroma.sqlite3").exists()

        # Migration from old global-hash system: rebuild cleanly to avoid
        # vector duplicates (the old code used .content_hash, the new
        # code uses .file_hashes.json for per-file tracking).
        if db_exists and not self._file_hashes_path().exists():
            print("Migrating from legacy DB format. Rebuilding...")
            import shutil
            shutil.rmtree(str(self.db_path))
            db_exists = False

        if not db_exists:
            print("Creating new vector database...")
            self._create_new_db(embed_model)
        else:
            # Load the existing store, then apply incremental changes.
            self.vector_store = Chroma(
                persist_directory=str(self.db_path),
                embedding_function=embed_model,
            )
            self._update_incremental(embed_model)

    # ── fresh build (used when no DB exists yet) ─────────────────

    def _create_new_db(self, embed_model):
        """Read all .md files, split into chunks, and index into a fresh DB."""
        md_files = sorted(self.journal_path.rglob("*.md"))

        # Load all non-empty files into Documents.
        docs: list[Document] = []
        for f in md_files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                if content.strip():
                    docs.append(
                        Document(
                            page_content=content,
                            metadata={"source": f.name},
                        )
                    )
            except Exception as e:
                print(f"Error reading {f}: {e}")

        # Split into overlapping chunks for better retrieval.
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        total = len(chunks)
        print(f"\n[INFO] Total chunks to process: {total}")

        # Create the empty collection, then batch-insert.
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

        # Persist the hash snapshot so the next run can diff.
        file_hashes = {}
        for f in md_files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                file_hashes[f.name] = self._hash_content(content)
            except Exception as e:
                print(f"Error hashing {f}: {e}")
        self._save_file_hashes(file_hashes)

    # ── incremental update ───────────────────────────────────────

    def _update_incremental(self, embed_model):
        """Compare current files against stored hashes; add/remove vectors as needed.

        Only files whose content actually changed are re-chunked and re-embedded.
        This is *much* faster than rebuilding the entire DB every time.
        """
        stored = self._load_file_hashes()
        current_files = {f.name: f for f in sorted(self.journal_path.rglob("*.md"))}
        current_hashes: dict[str, str] = {}
        any_change = False
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

        # --- process new / modified files ---
        for name, f in current_files.items():
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"  Skipping {f}: {e}")
                continue

            content_hash = self._hash_content(content)
            current_hashes[name] = content_hash

            if name not in stored:
                # Brand new file → add all its chunks.
                any_change = True
                print(f"  + Adding: {name}")
                doc = Document(page_content=content, metadata={"source": name})
                chunks = splitter.split_documents([doc])
                if chunks:
                    self.vector_store.add_documents(chunks)

            elif stored[name] != content_hash:
                # Content changed → delete old vectors, then re-add.
                any_change = True
                print(f"  ~ Updating: {name}")
                try:
                    self.vector_store.delete(where={"source": name})
                except Exception:
                    pass  # No existing vectors to delete.
                doc = Document(page_content=content, metadata={"source": name})
                chunks = splitter.split_documents([doc])
                if chunks:
                    self.vector_store.add_documents(chunks)

        # --- process deleted files ---
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

    # ── retrieval ────────────────────────────────────────────────

    def _format_hits(self, hits, max_chars):
        """Concatenate *hits* (list of Document), truncate to *max_chars*."""
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

    def search(self, query, k=3, max_chars=2000):
        if not self.vector_store:
            return "No context available (Database not initialised)."
        try:
            hits = self.vector_store.similarity_search(query, k=k)
            return self._format_hits(hits, max_chars)
        except Exception as e:
            return f"RAG search error: {e}"

    def search_priority(self, query, priority_source, k=3,
                        max_chars=2000, priority_k=1):
        """Always include *priority_k* chunk(s) from *priority_source*, then fill
        the rest from the full index.  Avoids duplicates."""
        if not self.vector_store:
            return "No context available (Database not initialised)."
        try:
            seen = set()
            combined = []

            # 1 — forced chunk(s) from the priority source.
            priority_hits = self.vector_store.similarity_search(
                query, k=priority_k,
                filter={"source": priority_source},
            )
            for h in priority_hits:
                h.page_content = f"[priority] {h.page_content.strip()}"
                combined.append(h)
                seen.add(self._hash_content(h.page_content))

            # 2 — general search, skipping anything already collected.
            general_hits = self.vector_store.similarity_search(query, k=k)
            for h in general_hits:
                if len(combined) >= k:
                    break
                key = self._hash_content(h.page_content)
                if key not in seen:
                    seen.add(key)
                    combined.append(h)
                elif len(combined) < priority_k + 1:
                    pass  # still looking for more

            return self._format_hits(combined, max_chars)
        except Exception as e:
            return f"RAG priority search error: {e}"
