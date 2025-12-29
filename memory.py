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
# Conversation Memory Update
# ------------------------------------------------------------------
def update_summary(
    llm,
    previous_summary: str | None,
    user_query: str,
    answer: str
) -> str:
    """
    Updates long-term conversation summary.
    """
    logger.info("update_summary called")

    try:
        previous_summary = previous_summary or ""

        logger.debug(
            "Previous summary length: %d",
            len(previous_summary)
        )
        logger.debug(
            "User query length: %d | Answer length: %d",
            len(user_query), len(answer)
        )

        prompt = f"""
        Update the summary with the new interaction.

        Existing Summary:
        {previous_summary}

        User Question:
        {user_query}

        Answer:
        {answer}
        """

        logger.info("Sending summary update prompt to LLM")
        updated_summary = llm(prompt).strip()

        logger.info("Conversation summary updated successfully")
        logger.debug(
            "Updated summary length: %d",
            len(updated_summary)
        )

        return updated_summary

    except Exception:
        logger.exception("Failed to update conversation summary")
        raise
