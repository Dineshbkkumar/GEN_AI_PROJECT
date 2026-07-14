import streamlit as st


from rag.ingestion import extract_pdf_content
    


from rag.embedding import generate_embeddings

from rag.weaviate_client import (
    get_weaviate_client,
    create_collection,
    delete_collection,
    store_embeddings
)

from rag.pipeline import rag_pipeline


st.set_page_config(
    page_title="Enterprise RAG System",
    layout="wide"
)

st.title("Enterprise RAG System")

# -----------------------------
# Session State
# -----------------------------
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# -----------------------------
# File Upload
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload Files",
    accept_multiple_files=True,
    type=["pdf", "xlsx", "xls", "csv", "txt", "docx"]
)

# -----------------------------
# Process Documents
# -----------------------------
if uploaded_files and not st.session_state.documents_processed:

    with st.spinner("Processing uploaded documents..."):

        all_chunks = []

        for file in uploaded_files:

             chunks = extract_pdf_content(file)

             all_chunks.extend(chunks)

        st.success(f"Created {len(all_chunks)} chunks.")

        embedded_chunks = generate_embeddings(all_chunks)

        st.success(f"Created {len(chunks)} chunks.")

        # -----------------------------
        # Embeddings
        # -----------------------------
        embedded_chunks = generate_embeddings(chunks)

        st.success(
            f"Generated {len(embedded_chunks)} embeddings."
        )

        # -----------------------------
        # Store in Weaviate
        # -----------------------------
        client = get_weaviate_client()

        try:

            delete_collection(client)

            create_collection(client)

            store_embeddings(
                client,
                embedded_chunks
            )

            st.session_state.documents_processed = True

            st.success(
                "Documents indexed successfully!"
            )

        except Exception as e:

            st.error("Error storing embeddings.")

            st.exception(e)

        finally:

            client.close()

# -----------------------------
# Ask Questions
# -----------------------------
if st.session_state.documents_processed:

    st.divider()

    st.subheader("Ask Questions")

    query = st.text_input(
        "Ask a question about the uploaded documents"
    )

    if query:

        with st.spinner("Searching documents..."):

            client = get_weaviate_client()

            try:

                answer, retrieved_docs = rag_pipeline(
                    client,
                    query
                )

            except Exception as e:

                st.error("RAG Pipeline Error")

                st.exception(e)

                answer = None

                retrieved_docs = []

            finally:

                client.close()

        # -----------------------------
        # Display Answer
        # -----------------------------
        if answer:

            st.subheader("Answer")

            st.success(answer)

        # -----------------------------
        # Retrieved Chunks
        # -----------------------------
        if retrieved_docs:

            with st.expander("Retrieved Chunks"):

                for i, doc in enumerate(
                    retrieved_docs,
                    start=1
                ):

                    st.markdown(f"### Chunk {i}")

                    st.write(doc["text"])

                    if "distance" in doc:

                        st.caption(
                            f"Distance: {doc['distance']:.4f}"
                        )

                    st.divider()

        else:

            st.warning(
                "No relevant chunks found."
            )