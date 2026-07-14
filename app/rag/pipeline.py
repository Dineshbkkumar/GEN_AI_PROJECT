from rag.retriever import retrieve_documents
from rag.prompt_builder import build_prompt
from rag.llm import generate_response


def rag_pipeline(
    client,
    query
):

    retrieved_chunks = retrieve_documents(
        client,
        query,
        top_k=5
    )

    # If no relevant chunks are found
    if not retrieved_chunks:
        return (
            "I couldn't find any relevant information in the uploaded documents.",
            []
        )

    prompt = build_prompt(
        query,
        retrieved_chunks
    )

    answer = generate_response(
        prompt
    )

    return answer, retrieved_chunks