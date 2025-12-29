import logging
import gradio as gr
from chroma_store import search as chroma_search
from main import run_rag_pipeline
from embedding import embed_text
from llm_client import llm
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# Logging
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
# Models
# ------------------------------------------------------------------
logger.info("Loading SentenceTransformer model")
model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("SentenceTransformer model loaded successfully")

# ------------------------------------------------------------------
# Vector DB Adapter
# ------------------------------------------------------------------
class VectorDBAdapter:
    def search(self, query: str, top_k=5):
        logger.info("VectorDB search called | top_k=%d", top_k)
        query_embedding = embed_text(model.encode, query)
        return chroma_search(query_embedding, top_k)

vector_db = VectorDBAdapter()

# ------------------------------------------------------------------
# 🔥 SESSION STATE (ChatInterface-safe)
# ------------------------------------------------------------------
session_state = {
    "conversation_summary": None,
    "conversation_summary_embedding": None,
    "current_topic_embedding": None,
}

# ------------------------------------------------------------------
# Chat Handler (ChatInterface compliant)
# ------------------------------------------------------------------
def chat_fn(user_message: str, history):
    logger.info("New chat message received")

    result = run_rag_pipeline(
        user_query=user_message,
        llm=llm,
        embedder=model.encode,
        vector_db=vector_db,
        conversation_summary=session_state["conversation_summary"],
        conversation_summary_embedding=session_state["conversation_summary_embedding"],
        current_topic_embedding=session_state["current_topic_embedding"],
    )

    # -------------------------------
    # Update session memory
    # -------------------------------
    session_state["conversation_summary"] = result["conversation_summary"]
    session_state["conversation_summary_embedding"] = result["conversation_summary_embedding"]
    session_state["current_topic_embedding"] = result["current_topic_embedding"]

    topic_relation = result["topic_relation"]

    # -------------------------------
    # UI topic indicator
    # -------------------------------
    if topic_relation == "new_topic":
        ui_prefix = "New topic \n\n"
    elif topic_relation == "same_topic":
        ui_prefix = "Continuing on the same topic\n\n"
    elif topic_relation == "partial":
        ui_prefix = "Partially related topic\n\n"
    else:
        ui_prefix = ""

    return ui_prefix + result["answer"]

# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------
demo = gr.ChatInterface(
    fn=chat_fn,
    title="RAG Chatbot (Topic-Aware)",
    description="RAG chatbot with topic detection, confidence gating, and memory.",
    examples=[
        "What is a transformer model?",
        "Explain self-attention",
        "What is NLP?",
        "What is tokenization?",
        "What is SQL LEFT JOIN?",
    ],
)

if __name__ == "__main__":
    demo.launch()
