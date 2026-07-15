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

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

uploaded_files = st.file_uploader(
    "Upload Files",
    accept_multiple_files=True,
    type=["pdf", "xlsx", "xls", "csv", "txt", "docx"]
)

if uploaded_files and not st.session_state.documents_processed:

    with st.spinner("Processing uploaded documents..."):

        all_chunks = []

        for file in uploaded_files:

             chunks = extract_pdf_content(file)

             all_chunks.extend(chunks)

        st.success(f"Created {len(all_chunks)} chunks.")

        embedded_chunks = generate_embeddings(all_chunks)

        st.success(
            f"Generated {len(embedded_chunks)} embeddings."
        )

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

        if answer:

            st.subheader("Answer")

            st.success(answer)

        if retrieved_docs:

            with st.expander("Retrieved Chunks"):

                for i, doc in enumerate(
                    retrieved_docs,
                    start=1
                ):

                    is_table = doc.get("type") == "Table"
                    badge = "Table" if is_table else "Text"

                    st.markdown(f"### Chunk {i} [{badge}]")

                    st.write(doc["text"])

                    st.caption(
                        f"Source: {doc.get('source', 'Unknown')} | Page: {doc.get('page', 'N/A')}"
                    )

                    if "distance" in doc:

                        st.caption(
                            f"Relevance: {(1 - doc['distance'])*100:.1f}%"
                        )

                    st.divider()

        else:

            st.warning(
                "No relevant chunks found."
            )
