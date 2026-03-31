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

    def __init__(self, vectorstore_dir: Path) -> None:
        self.vector_db = None
        self._load_vectorstore(vectorstore_dir)

    def _load_vectorstore(self, vs_path: Path) -> None:
        """Load the FAISS vectorstore."""
        api_key = os.getenv("MISTRAL_API_KEY")

        # Original: single vague error message
        if not vs_path.exists() or not api_key:
            logger.warning("RAG vectorstore not loaded (path or API key missing)")
            return

        try:
            from langchain_community.vectorstores import FAISS
            from langchain_mistralai import MistralAIEmbeddings

            embeddings = MistralAIEmbeddings(api_key=api_key, model="mistral-embed")
            self.vector_db = FAISS.load_local(
                str(vs_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("RAG vectorstore loaded from %s", vs_path)
        except Exception as exc:
            logger.error("Failed to load vectorstore: %s", exc)

    @property
    def is_loaded(self) -> bool:
        """Check if the vectorstore is ready for queries."""
        return self.vector_db is not None

    def get_clinical_excerpt(self, query: str) -> Optional[str]:
        """Retrieve a clinical excerpt from the RAG vectorstore.

        Args:
            query: Search query (emotion or coping strategy).

        Returns:
            Relevant text excerpt (max 500 chars), or None.
        """
        if not self.is_loaded:
            return None

        try:
            # Original naive query — same for all contexts
            enriched_query = (
                f"techniques for managing {query} emotion coping strategies"
            )
            results = self.vector_db.similarity_search(enriched_query, k=1)

            if not results:
                return None

            content = results[0].page_content

            # Original: brute truncation at 500 chars
            if len(content) > 500:
                truncated = content[:500]
                last_period = truncated.rfind(".")
                content = (
                    truncated[: last_period + 1]
                    if last_period > 400
                    else truncated + "..."
                )

            return content

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
        

        