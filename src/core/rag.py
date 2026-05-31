from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class LocalEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text])[0].tolist()


class RAGManager:
    def __init__(self, journal_path, db_path, model_name):
        self.journal_path = Path(journal_path)
        self.db_path = Path(db_path)
        self.model_name = model_name
        self.vector_store = None

    def initialize(self):
        """Load or create the vector DB. Forces path creation."""
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.journal_path.mkdir(parents=True, exist_ok=True)

        embed_model = LocalEmbeddings(self.model_name)

        is_already_created = any(self.db_path.glob("*.sqlite3")) or (self.db_path / "chroma.sqlite3").exists()

        if is_already_created:
            print(f"Loading vector database from {self.db_path}")
            self.vector_store = Chroma(
                persist_directory=str(self.db_path),
                embedding_function=embed_model
            )
        else:
            print("Creating new vector database...")
            self._create_new_db(embed_model)

    def _create_new_db(self, embed_model):
        docs = []
        md_files = list(self.journal_path.glob("**/*.md"))

        for f in md_files:
            try:
                content = f.read_text(encoding="utf-8")
                if content.strip():
                    docs.append(Document(page_content=content, metadata={"source": str(f)}))
            except Exception as e:
                print(f"Error reading {f}: {e}")

        if not docs:
            docs = [Document(page_content="System: Copycat memory initialized. Awaiting interactions.",
                             metadata={"source": "system"})]

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embed_model,
            persist_directory=str(self.db_path)
        )
        print(f"Vector DB created with {len(chunks)} chunks.")

    def search(self, query, k=2):
        """Safe search against NoneType."""
        if not self.vector_store:
            return "No context available (Database not initialized)."

        try:
            hits = self.vector_store.similarity_search(query, k=k)
            return "\n---\n".join(h.page_content for h in hits)
        except Exception as e:
            return f"RAG search error: {e}"
