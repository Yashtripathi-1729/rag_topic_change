import logging
import numpy as np

from embedding import embed_text
from similarity import detect_topic_relation
from query_rewrite import rewrite_query
from retriever import retrieve_documents
from confidence import is_confident
from llm_answer import generate_answer
from memory import update_summary

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
# RAG Pipeline
# ------------------------------------------------------------------
def run_rag_pipeline(
    user_query: str,
    *,
    llm,
    embedder,
    vector_db,
    conversation_summary: str | None,
    conversation_summary_embedding: list[float] | None,
    current_topic_embedding: list[float] | None,   
):
    logger.info("RAG pipeline started")

    try:
        # --------------------------------------------------------------
        # Step 1: Embed user query
        # --------------------------------------------------------------
        logger.info("Step 1: Embedding user query")
        query_embedding = embed_text(embedder, user_query)

        # --------------------------------------------------------------
        # Step 2: Topic similarity detection (FIXED)
        # --------------------------------------------------------------
        logger.info("Step 2: Topic similarity detection")

        if current_topic_embedding is None:
            topic_info = {"relation": "new_topic", "similarity": 0.0}
            logger.info("No current topic → new topic")
        else:
            topic_info = detect_topic_relation(
                query_embedding,
                current_topic_embedding
            )

            # 🔥 Promote strong partial → same_topic
            if (
                topic_info["relation"] == "partial"
                and topic_info.get("similarity", 0.0) >= 0.65
            ):
                topic_info["relation"] = "same_topic"

            logger.info(
                "Topic relation: %s | Similarity: %.4f",
                topic_info["relation"],
                topic_info.get("similarity", 0.0)
            )

        # --------------------------------------------------------------
        # Step 3: Query rewrite (unchanged)
        # --------------------------------------------------------------
        logger.info("Step 3: Query rewrite")

        rewritten_query = rewrite_query(
            user_query,
            conversation_summary,
            topic_info["relation"],
            llm
        )

        # --------------------------------------------------------------
        # Step 4: Retrieval
        # --------------------------------------------------------------
        logger.info("Step 4: Retrieving documents")

        retrieved_docs = retrieve_documents(
            vector_db,
            rewritten_query,
            conversation_summary,
            topic_info["relation"]
        )

        logger.info("Retrieved %d documents", len(retrieved_docs))

        # --------------------------------------------------------------
        # Step 5: Confidence gate
        # --------------------------------------------------------------
        logger.info("Step 5: Confidence evaluation")

        is_first_turn = conversation_summary is None or conversation_summary.strip() == ""
        confident = is_confident(retrieved_docs)

        logger.info("Confidence result: %s", confident)

        if not confident and not is_first_turn:
            return {
                "answer": "I don’t have enough relevant information to answer this confidently.",
                "conversation_summary": conversation_summary,
                "conversation_summary_embedding": conversation_summary_embedding,
                "current_topic_embedding": current_topic_embedding,
                "topic_relation": topic_info["relation"],
                "topic_similarity": topic_info["similarity"],
            }

        # --------------------------------------------------------------
        # Step 6: Answer generation
        # --------------------------------------------------------------
        logger.info("Step 6: Generating answer")

        answer = generate_answer(
            llm,
            user_query,
            retrieved_docs,
            is_first_turn=is_first_turn
        )

        logger.info("Answer generated successfully")

        # --------------------------------------------------------------
        # Step 7: Update conversation summary (LONG-TERM MEMORY ONLY)
        # --------------------------------------------------------------
        logger.info("Step 7: Updating conversation summary")

        updated_summary = update_summary(
            llm,
            conversation_summary,
            user_query,
            answer
        )

        # --------------------------------------------------------------
        # Step 8: Update CURRENT TOPIC embedding (FIXED)
        # --------------------------------------------------------------
        if topic_info["relation"] == "new_topic" or current_topic_embedding is None:
            updated_topic_embedding = query_embedding
        elif topic_info["relation"] == "same_topic":
            updated_topic_embedding = (
                0.7 * np.array(current_topic_embedding, dtype=np.float32)
                + 0.3 * np.array(query_embedding, dtype=np.float32)
            )
        else:  # partial
            updated_topic_embedding = (
                0.9 * np.array(current_topic_embedding, dtype=np.float32)
                + 0.1 * np.array(query_embedding, dtype=np.float32)
            ).tolist()

        logger.info("RAG pipeline completed successfully")

        return {
            "answer": answer,
            "conversation_summary": updated_summary,
            "conversation_summary_embedding": embed_text(embedder, updated_summary),
            "current_topic_embedding": updated_topic_embedding,   # ✅ RETURNED
            "topic_relation": topic_info["relation"],
            "topic_similarity": topic_info["similarity"],
        }

    except Exception:
        logger.exception("RAG pipeline execution failed")
        raise

