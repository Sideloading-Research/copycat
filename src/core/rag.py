"""Re-export of ``ChromaVectorDB`` for backward compatibility.

New code should import directly from ``src.core.vector_db.chroma_backend``.
"""

from src.core.vector_db.chroma_backend import ChromaVectorDB

# Kept for compatibility with existing test scripts.
RAGManager = ChromaVectorDB
