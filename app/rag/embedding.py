from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def generate_embeddings(chunks):

    texts = [chunk.page_content for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):

        embedded_chunks.append(
            {
                "text": chunk.page_content,
                "metadata": chunk.metadata,
                "embedding": embedding.tolist()
            }
        )

    return embedded_chunks
