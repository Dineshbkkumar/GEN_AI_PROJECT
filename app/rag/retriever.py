from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def retrieve_documents(
    client,
    query,
    top_k=5,
    collection_name="RAGDocuments"
):

    try:

        collection = client.collections.get(
            collection_name
        )

        query_embedding = model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=["distance"]
        )

        retrieved_chunks = []

        for obj in response.objects:

            retrieved_chunks.append(
                {
                    "text": obj.properties.get("text"),
                    "source": obj.properties.get("source"),
                    "page": obj.properties.get("page"),
                    "type": obj.properties.get("type"),
                    "distance": obj.metadata.distance
                }
            )

        return retrieved_chunks

    except Exception as e:

        print(f"Retrieval Error: {e}")

        return []
