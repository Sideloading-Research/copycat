from src.core.rag import RAGManager
from src.utils.paths import PATHS

rag = RAGManager(PATHS['journal'], PATHS['vector_db'], "all-MiniLM-L6-v2")
rag.initialize()
print(rag.search("¿Qué cené ayer?"))