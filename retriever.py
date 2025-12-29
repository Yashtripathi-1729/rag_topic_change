import logging
from config import TOP_K

# ------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)


# ------------------------------------------------------------------
# Document Retrieval
# ------------------------------------------------------------------
def retrieve_documents(
    vector_db,
    query: str,
    context_summary: str | None,
    relation: str
) -> list[dict]:
    """
    Performs vector search.
    """
    logger.info("retrieve_documents called | relation=%s", relation)
    logger.debug("Query length: %d", len(query))

    try:
        # ----------------------------------------------------------
        # Primary retrieval
        # ----------------------------------------------------------
        logger.info("Performing primary vector search | top_k=%d", TOP_K)
        docs = vector_db.search(query, top_k=TOP_K)
        logger.info("Primary retrieval returned %d documents", len(docs))

        # ----------------------------------------------------------
        # Context-based retrieval (same topic only)
        # ----------------------------------------------------------
        if relation == "same_topic" and context_summary:
            logger.info(
                "Same topic detected — performing context-based retrieval"
            )
            logger.debug(
                "Context summary length: %d",
                len(context_summary)
            )

            context_docs = vector_db.search(
                context_summary,
                top_k=TOP_K
            )

            logger.info(
                "Context-based retrieval returned %d documents",
                len(context_docs)
            )

            docs.extend(context_docs)

        logger.info(
            "Total documents returned after merge: %d",
            len(docs)
        )

        return docs

    except Exception:
        logger.exception("Document retrieval failed")
        raise
