"""RAG retrieval module with FAISS vectorstore.

Loads a FAISS index for clinical document retrieval.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from src.config import CONFIG, PROJECT_ROOT

logger = logging.getLogger(__name__)

_PATHS = CONFIG["paths"]


class RAGRetriever:
    """FAISS-based retrieval for clinical excerpts.

    Args:
        vectorstore_dir: Path to the FAISS index directory.
    """

    def __init__(
        self,
        vectorstore_dir: Path,
        source_dir: Optional[Path] = None,
    ) -> None:
        self.vector_db = None
        self._embeddings = None
        self._source_dir = source_dir or (PROJECT_ROOT / _PATHS["source_dir"])
        self._vs_dir = vectorstore_dir
        self._initialize(vectorstore_dir)

    def _initialize(self, vs_path: Path) -> None:
        """Load existing vectorstore or build from source documents."""
        api_key = os.getenv("MISTRAL_API_KEY")

        if not api_key:
            logger.warning(
                "RAG disabled: MISTRAL_API_KEY not set. "
                "ECO mode will work without clinical excerpts."
            )
            return

        try:
            from langchain_mistralai import MistralAIEmbeddings

            self._embeddings = MistralAIEmbeddings(
                api_key=api_key, model="mistral-embed"
            )
        except Exception as exc:
            logger.error("Failed to initialize embeddings: %s", exc)
            return

        if vs_path.exists():
            self._load_existing(vs_path)
        else:
            logger.info(
                "Vectorstore not found at %s — building from source documents...",
                vs_path,
            )
            self._build_from_sources(vs_path)

    def _load_existing(self, vs_path: Path) -> None:
        """Load an existing FAISS index from disk."""
        try:
            from langchain_community.vectorstores import FAISS

            self.vector_db = FAISS.load_local(
                str(vs_path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("RAG vectorstore loaded from %s", vs_path)
        except Exception as exc:
            logger.error(
                "Failed to load vectorstore from %s: %s. "
                "Delete the directory and restart to trigger rebuild.",
                vs_path,
                exc,
            )

    def _build_from_sources(self, vs_path: Path) -> None:
        """Build FAISS index from source documents (PDF + TXT).

        Replaces the manual notebook 3 step. Chunking parameters
        match the original notebook: 800 chars, 100 overlap.
        """
        import time

        if not self._source_dir.exists():
            logger.error(
                "Source directory not found: %s. "
                "Cannot build vectorstore without clinical documents.",
                self._source_dir,
            )
            return

        try:
            from langchain_community.document_loaders import (
                PyPDFLoader,
                TextLoader,
            )
            from langchain_community.vectorstores import FAISS
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            # 1. Load documents
            documents = []
            for filename in sorted(os.listdir(self._source_dir)):
                file_path = self._source_dir / filename
                loader = None

                if filename.endswith(".pdf"):
                    loader = PyPDFLoader(str(file_path))
                elif filename.endswith(".txt"):
                    loader = TextLoader(str(file_path), encoding="utf-8")

                if loader:
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info("  Loaded: %s (%d pages/sections)", filename, len(docs))

            if not documents:
                logger.error("No documents found in %s", self._source_dir)
                return

            # 2. Chunk (same params as notebook 3)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", " ", ""],
            )
            chunks = splitter.split_documents(documents)
            logger.info("  Chunked: %d chunks from %d documents", len(chunks), len(documents))

            # 3. Embed and index
            start = time.time()
            self.vector_db = FAISS.from_documents(chunks, self._embeddings)
            elapsed = time.time() - start
            logger.info("  Indexed in %.1fs", elapsed)

            # 4. Save to disk
            vs_path.mkdir(parents=True, exist_ok=True)
            self.vector_db.save_local(str(vs_path))
            logger.info("  Vectorstore saved to %s", vs_path)

        except Exception as exc:
            logger.error("Failed to build vectorstore: %s", exc)

    @property
    def is_loaded(self) -> bool:
        """Check if the vectorstore is ready for queries."""
        return self.vector_db is not None

    def get_clinical_excerpt(
        self,
        query: str,
        context: str = "general",
    ) -> Optional[dict]:
        """Retrieve a clinical excerpt with source metadata.

        The query is adapted to the detected context to improve
        retrieval relevance.

        Args:
            query: Search query (emotion or coping strategy).
            context: Detected situational context from analysis.py.

        Returns:
            Dictionary with 'content' and 'source' keys, or None.
        """
        if not self.is_loaded:
            return None

        try:
            # Context-aware query construction
            context_prefixes = {
                "work": "workplace stress coping techniques for",
                "relationship": "relationship difficulties support for",
                "academic": "academic stress management for",
                "health": "health anxiety coping strategies for",
                "financial": "financial stress management techniques for",
                "family": "family conflict resolution strategies for",
                "social": "social anxiety and loneliness coping for",
                "general": "techniques for managing",
            }
            prefix = context_prefixes.get(context, context_prefixes["general"])
            enriched_query = f"{prefix} {query} emotion"

            results = self.vector_db.similarity_search(enriched_query, k=1)

            if not results:
                return None

            doc = results[0]
            content = doc.page_content

            # Truncate at last complete sentence (not mid-list)
            if len(content) > 500:
                truncated = content[:500]
                last_period = truncated.rfind(".")
                if last_period > 300:
                    content = truncated[: last_period + 1]
                else:
                    last_newline = truncated.rfind("\n")
                    if last_newline > 200:
                        content = truncated[:last_newline].rstrip()
                    else:
                        content = truncated + "..."
            
            # Extract source filename from FAISS metadata
            source_path = doc.metadata.get("source", "Unknown source")
            source_name = Path(source_path).name if source_path else "Unknown"

            return {
                "content": content,
                "source": source_name,
            }

        except Exception as exc:
            logger.error("RAG excerpt retrieval failed: %s", exc)
            return None
        

    def query_knowledge_base(self, query: str) -> str:
        """Search the RAG knowledge base for detailed information.

        Args:
            query: Search query.

        Returns:
            Concatenated content from the top 2 matching chunks.
        """
        if not self.is_loaded:
            return "Knowledge base unavailable."

        try:
            results = self.vector_db.similarity_search(query, k=2)
            return "\n".join(doc.page_content for doc in results)
        except Exception as exc:
            logger.error("Knowledge base query failed: %s", exc)
            return f"Knowledge base error: {exc}"
        

