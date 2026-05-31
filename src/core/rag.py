from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer



class LocalEmbeddings:
    def __init__(self, model_name: str):
        # Aseguramos que use CPU explícitamente para evitar conflictos con XTTS/Wav2Lip
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text])[0].tolist()  # Ajuste de formato para query


class RAGManager:
    def __init__(self, journal_path, db_path, model_name):
        self.journal_path = Path(journal_path)
        self.db_path = Path(db_path)
        self.model_name = model_name
        self.vector_store = None

    def initialize(self):
        """Carga o crea la DB. Forzamos la creación de rutas."""
        # 1. Asegurar que las rutas existan ANTES de inicializar Chroma
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.journal_path.mkdir(parents=True, exist_ok=True)

        embed_model = LocalEmbeddings(self.model_name)

        # Comprobamos si hay archivos de sqlite (indicativo de una DB de Chroma)
        is_already_created = any(self.db_path.glob("*.sqlite3")) or (self.db_path / "chroma.sqlite3").exists()

        if is_already_created:
            print(f"-> Cargando base de datos vectorial desde {self.db_path}")
            self.vector_store = Chroma(
                persist_directory=str(self.db_path),
                embedding_function=embed_model
            )
        else:
            print("-> Creando nueva base de datos vectorial...")
            self._create_new_db(embed_model)

    def _create_new_db(self, embed_model):
        docs = []
        # Buscamos archivos .md
        md_files = list(self.journal_path.glob("**/*.md"))

        for f in md_files:
            try:
                content = f.read_text(encoding="utf-8")
                if content.strip():
                    docs.append(Document(page_content=content, metadata={"source": str(f)}))
            except Exception as e:
                print(f"Error leyendo {f}: {e}")

        # Si no hay documentos, creamos uno de sistema para inicializar la DB
        if not docs:
            docs = [Document(page_content="System: Inicio de memoria de Copycat. Esperando interacciones.",
                             metadata={"source": "system"})]

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        # Esto debería forzar la creación de archivos en data/vector_db/
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embed_model,
            persist_directory=str(self.db_path)
        )
        print(f"✅ Vector DB creada con {len(chunks)} fragmentos.")

    def search(self, query, k=2):
        """Búsqueda segura contra NoneType."""
        if not self.vector_store:
            return "No hay contexto disponible (Base de datos no inicializada)."

        try:
            hits = self.vector_store.similarity_search(query, k=k)
            return "\n---\n".join(h.page_content for h in hits)
        except Exception as e:
            return f"Error en búsqueda RAG: {e}"