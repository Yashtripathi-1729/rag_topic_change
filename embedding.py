def embed_text(embedder, text: str):
    embedding = embedder(text)

    # Convert numpy array → Python list (required by Chroma)
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    return embedding
