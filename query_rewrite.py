import logging

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
# Query Rewriting
# ------------------------------------------------------------------
def rewrite_query(
    user_query: str,
    conversation_summary: str | None,
    relation: str,
    llm
) -> str:
    """
    Rewrites the query based on topic relation.
    """
    logger.info("rewrite_query called | relation=%s", relation)
    logger.debug("User query length: %d", len(user_query))

    # --------------------------------------------------------------
    # Case 1: New topic or no memory → no rewrite
    # --------------------------------------------------------------
    if relation == "new_topic" or not conversation_summary:
        logger.info(
            "Skipping query rewrite (new topic or empty summary)"
        )
        return user_query

    try:
        # ----------------------------------------------------------
        # Case 2: Same topic
        # ----------------------------------------------------------
        if relation == "same_topic":
            logger.info("Rewriting query for SAME topic")
            prompt = f"""
            Rewrite the user question using the context below.

            Context:
            {conversation_summary}

            Question:
            {user_query}
            """

        # ----------------------------------------------------------
        # Case 3: Partial topic overlap
        # ----------------------------------------------------------
        else:  # partial
            logger.info("Rewriting query for PARTIAL topic overlap")
            prompt = f"""
            Rewrite the question as a standalone query.
            Use the context only if clearly relevant.

            Context:
            {conversation_summary}

            Question:
            {user_query}
            """

        logger.info("Sending rewrite prompt to LLM")
        rewritten_query = llm(prompt).strip()

        logger.debug(
            "Rewritten query length: %d",
            len(rewritten_query)
        )

        return rewritten_query

    except Exception:
        logger.exception("Query rewrite failed")
        raise
