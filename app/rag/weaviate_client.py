import os

import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType


load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

COLLECTION_NAME = "RAGDocuments"


def get_weaviate_client():
    """
    Connect to Weaviate Cloud
    """

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(
            WEAVIATE_API_KEY
        )
    )

    return client


def create_collection(client):
    """
    Create collection with metadata fields.
    """

    try:
        client.collections.get(COLLECTION_NAME)
        print("Collection already exists.")

    except Exception:

        client.collections.create(

            name=COLLECTION_NAME,

            properties=[

                Property(
                    name="text",
                    data_type=DataType.TEXT
                ),

                Property(
                    name="source",
                    data_type=DataType.TEXT
                ),

                Property(
                    name="page",
                    data_type=DataType.INT
                ),

                Property(
                    name="type",
                    data_type=DataType.TEXT
                )

            ]

        )

        print("Collection created successfully.")


def delete_collection(client):
    """
    Delete existing collection.
    """

    try:

        client.collections.delete(
            COLLECTION_NAME
        )

        print("Old collection deleted.")

    except Exception:

        print("Collection not found.")


def store_embeddings(
    client,
    embedded_chunks
):
    """
    Store embeddings along with metadata.
    """

    collection = client.collections.get(
        COLLECTION_NAME
    )

    count = 0

    for item in embedded_chunks:

        metadata = item.get("metadata", {})

        collection.data.insert(

            properties={

                "text": item["text"],

                "source": metadata.get(
                    "source",
                    ""
                ),

                "page": metadata.get(
                    "page",
                    0
                ),

                "type": metadata.get(
                    "type",
                    ""
                )

            },

            vector=item["embedding"]

        )

        count += 1

    print(f"{count} chunks stored successfully.")