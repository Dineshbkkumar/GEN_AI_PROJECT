from rag.retriever import retrieve_documents
from rag.reranker import rerank_documents
from rag.prompt_builder import build_prompt
from rag.llm import generate_response


def rag_pipeline(
    client,
    query
):

    retrieved_chunks = retrieve_documents(
        client,
        query,
        top_k=10
    )

    if not retrieved_chunks:
        return (
            "I couldn't find any relevant information in the uploaded documents.",
            []
        )

    reranked_chunks = rerank_documents(
        query,
        [chunk["text"] for chunk in retrieved_chunks],
        top_k=5
    )

    final_chunks = [
        chunk for chunk in retrieved_chunks
        if chunk["text"] in reranked_chunks
    ]

    prompt = build_prompt(
        query,
        final_chunks
    )

    answer = generate_response(
        prompt
    )

    return answer, final_chunks
