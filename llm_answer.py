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
# Answer Generation
# ------------------------------------------------------------------
def generate_answer(
    llm,
    user_query: str,
    retrieved_docs: list[dict],
    *,
    is_first_turn: bool = False
) -> str:
    """
    Generates an answer using retrieved context.
    On first turn, allows a best-effort answer.
    """
    logger.info("generate_answer called")
    logger.info("Number of retrieved documents: %d", len(retrieved_docs))

    if not retrieved_docs:
        logger.warning("No retrieved documents provided to LLM")
        return "I don't know"

    context = "\n\n".join(doc["content"] for doc in retrieved_docs)

    if is_first_turn:
        prompt = f"""
        You are a helpful AI assistant.

        Use the context below to answer the question.
        If the context is partial, still give a concise, correct explanation.

        Context:
        {context}

        Question:
        {user_query}
        """
    else:
        prompt = f"""
        Answer the question using ONLY the context below.

        Rules:
        - Do not use prior conversation.
        - If the answer is not present, say "I don't know".

        Context:
        {context}

        Question:
        {user_query}
        """

    response = llm(prompt).strip()
    return response

