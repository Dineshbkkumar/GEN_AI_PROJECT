from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_documents(
    query,
    retrieved_chunks,
    top_k=3
):

    sentence_pairs = []

    for chunk in retrieved_chunks:

        sentence_pairs.append(
            (query, chunk)
        )

    scores = reranker.predict(sentence_pairs)

    ranked = sorted(
        zip(retrieved_chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    reranked_chunks = []

    for chunk, score in ranked[:top_k]:

        reranked_chunks.append(chunk)

    return reranked_chunks