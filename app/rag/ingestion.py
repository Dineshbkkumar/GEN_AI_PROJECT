from unstructured.partition.pdf import partition_pdf

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_pdf_content(uploaded_file):

    elements = partition_pdf(
        file=uploaded_file,
        strategy="hi_res",
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=1200,
        combine_text_under_n_chars=300
    )

    documents = []

    for element in elements:

        text = str(element).strip()

        if not text:
            continue

        page = None

        if hasattr(element.metadata, "page_number"):
            page = element.metadata.page_number

        metadata = {
            "source": uploaded_file.name,
            "page": page,
            "type": element.category
        }

        documents.append(
            Document(
                page_content=text,
                metadata=metadata
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    return chunks