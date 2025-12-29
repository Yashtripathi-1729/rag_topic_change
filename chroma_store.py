import logging
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


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
# ChromaDB Client Initialization
# ------------------------------------------------------------------
try:
    logger.info("Initializing ChromaDB client")
    
    CHROMA_PATH = "C:/Users/Yash Tripathi/intraintel/chroma_db"
    client = chromadb.PersistentClient(
    path=CHROMA_PATH
    )

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def embedding_fn(texts):
        return embedding_model.encode(texts).tolist()

    collection = client.get_or_create_collection(
        name="rag_docs",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    logger.info("ChromaDB collection 'rag_docs' initialized successfully")

except Exception as e:
    logger.exception("Failed to initialize ChromaDB client or collection")
    raise

# ------------------------------------------------------------------
# Batch Helper
# ------------------------------------------------------------------
def batch(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

# ------------------------------------------------------------------
# Add Documents
# ------------------------------------------------------------------
def add_documents(documents, batch_size=100):
    """
    Adds documents to the ChromaDB collection in batches.

    Args:
        documents (list): List of dicts with keys: id, content, embedding
        batch_size (int): Max batch size for ChromaDB
    """
    logger.info("add_documents called")
    logger.debug("Number of documents received: %d", len(documents))

    if not documents:
        logger.warning("No documents to add")
        return

    total = len(documents)

    try:
        for start_idx in range(0, total, batch_size):
            end_idx = start_idx + batch_size
            batch_docs = documents[start_idx:end_idx]

            logger.info(
                "Inserting batch %d–%d (%d docs)",
                start_idx,
                min(end_idx, total),
                len(batch_docs),
            )

            collection.add(
                ids=[doc["id"] for doc in batch_docs],
                documents=[doc["content"] for doc in batch_docs],
                embeddings=[
                    doc["embedding"].tolist()
                    if hasattr(doc["embedding"], "tolist")
                    else doc["embedding"]
                    for doc in batch_docs
                ],
            )

        logger.info("Successfully added %d documents to collection", total)

    except KeyError as e:
        logger.exception("Document schema error. Missing key: %s", str(e))
        raise

    except Exception:
        logger.exception("Failed to add documents to ChromaDB")
        raise



# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------
def search(query_embedding, top_k=5):
    """
    Searches the ChromaDB collection using an embedding.

    Args:
        query_embedding (list): Query embedding vector
        top_k (int): Number of results to return

    Returns:
        list: List of documents with content and similarity score
    """
    logger.info("search called with top_k=%d", top_k)

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        docs = []
        num_results = len(results["documents"][0])
        logger.debug("Number of results retrieved: %d", num_results)

        for i in range(num_results):
            docs.append({
                "content": results["documents"][0][i],
                "score": 1 - results["distances"][0][i]
            })

        logger.info("Search completed successfully")
        return docs

    except Exception as e:
        logger.exception("Search operation failed")
        raise
