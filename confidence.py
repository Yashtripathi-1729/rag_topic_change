import logging
from config import MIN_RETRIEVAL_SCORE

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
# Confidence Check
# ------------------------------------------------------------------
def is_confident(retrieved_docs: list[dict]) -> bool:
    """
    Checks whether retrieval quality is sufficient.
    """
    logger.info("is_confident called")

    if not retrieved_docs:
        logger.warning("No documents retrieved — confidence check failed")
        return False

    try:
        scores = [doc["score"] for doc in retrieved_docs]
        best_score = max(scores)

        logger.info(
            "Best retrieval score: %.4f | Threshold: %.4f",
            best_score,
            MIN_RETRIEVAL_SCORE
        )

        is_conf = best_score >= MIN_RETRIEVAL_SCORE
        logger.info("Confidence result: %s", is_conf)

        return is_conf

    except KeyError:
        logger.exception("Retrieved document missing 'score' key")
        raise

    except Exception:
        logger.exception("Unexpected error during confidence evaluation")
        raise
