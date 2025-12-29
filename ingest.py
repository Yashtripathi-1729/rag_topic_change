# ingest.py

import os
import logging
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from embedding import embed_text
from chroma_store import add_documents

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
# Constants & Model Initialization
# ------------------------------------------------------------------
DOCS_PATH = "data/"

try:
    logger.info("Loading SentenceTransformer model for ingestion")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("SentenceTransformer model loaded successfully")
except Exception:
    logger.exception("Failed to load SentenceTransformer model")
    raise
# ------------------------------------------------------------------
# Chunking Function
# ------------------------------------------------------------------
def chunk_text(text, chunk_size=400, overlap=80):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# ------------------------------------------------------------------
# Document Loader
# ------------------------------------------------------------------
def load_documents():
    logger.info("Starting document ingestion from path: %s", DOCS_PATH)

    documents = []
    doc_id = 0

    try:
        files = os.listdir(DOCS_PATH)
        logger.info("Found %d files in data directory", len(files))
    except Exception:
        logger.exception("Failed to list files in data directory")
        raise

    for file in files:
        path = os.path.join(DOCS_PATH, file)
        logger.info("Processing file: %s", file)

        try:
            if file.endswith(".pdf"):
                reader = PdfReader(path)
                text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
                logger.debug("Extracted text from PDF: %s", file)

            elif file.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                logger.debug("Read text file: %s", file)

            else:
                logger.warning("Skipping unsupported file type: %s", file)
                continue

            chunks = chunk_text(text)

            logger.info(
                "File '%s' split into %d chunks before filtering",
                file, len(chunks)
            )

            for chunk in chunks:
                chunk = chunk.strip()

                if len(chunk) < 150:
                    continue

                embedding = embed_text(model.encode, chunk)

                documents.append({
                    "id": f"doc_{doc_id}",
                    "content": chunk,
                    "embedding": embedding
                })

                doc_id += 1

        except Exception:
            logger.exception("Failed to process file: %s", file)
            continue

    logger.info("Total chunks prepared for ingestion: %d", len(documents))
    return documents


# ------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Ingestion script started")

    try:
        docs = load_documents()
        add_documents(docs)

        logger.info(
            "Successfully ingested %d chunks into ChromaDB",
            len(docs)
        )

    except Exception:
        logger.exception("Document ingestion failed")
        raise
