
def build_prompt(
    query,
    retrieved_chunks
):

    context = ""

    for chunk in retrieved_chunks:

        if isinstance(chunk, dict):

            context += chunk["text"] + "\n\n"

        else:

            context += chunk + "\n\n"

    prompt = f"""
You are an intelligent AI assistant.

Answer ONLY using the provided context.

IMPORTANT INSTRUCTIONS:
- Match information even if case differs (e.g., SEARCH-404 matches search-404)
- Extract exact data from tables
- Provide exact matches from the context

If the answer is not available in the context, say:

"I couldn't find the answer in the uploaded documents."

Context:
-----------------------
{context}
-----------------------

Question:
{query}

Answer:
"""

    return prompt