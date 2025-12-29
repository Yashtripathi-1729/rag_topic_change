import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import SAME_TOPIC_THRESHOLD, PARTIAL_TOPIC_THRESHOLD

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
# Cosine Similarity
# ------------------------------------------------------------------
def cosine_sim(vec1, vec2) -> float:
    """
    Computes cosine similarity between two vectors.
    """
    try:
        similarity = cosine_similarity(
            np.array(vec1).reshape(1, -1),
            np.array(vec2).reshape(1, -1)
        )[0][0]

        logger.debug("Cosine similarity computed: %.4f", similarity)
        return similarity

    except Exception:
        logger.exception("Cosine similarity computation failed")
        raise


# ------------------------------------------------------------------
# Topic Relation Detection
# ------------------------------------------------------------------
def detect_topic_relation(
    query_embedding: list[float],
    summary_embedding: list[float]
) -> dict:
    """
    Determines whether the query belongs to the same topic,
    partially related topic, or a new topic.
    """
    logger.info("detect_topic_relation called")

    try:
        similarity = cosine_sim(query_embedding, summary_embedding)

        if similarity > SAME_TOPIC_THRESHOLD:
            relation = "same_topic"
        elif similarity >= PARTIAL_TOPIC_THRESHOLD:
            relation = "partial"
        else:
            relation = "new_topic"

        logger.info(
            "Topic relation determined: %s | Similarity: %.4f "
            "(same_topic > %.2f, partial >= %.2f)",
            relation,
            similarity,
            SAME_TOPIC_THRESHOLD,
            PARTIAL_TOPIC_THRESHOLD
        )

        return {
            "relation": relation,
            "similarity": similarity
        }

    except Exception:
        logger.exception("Topic relation detection failed")
        raise
