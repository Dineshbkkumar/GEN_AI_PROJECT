from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_pdf_content(uploaded_file):
    elements = partition_pdf(
        file=uploaded_file,
        strategy="fast",
        infer_table_structure=True
    )

    print(f"\n=== PDF Extraction Debug ===")
    print(f"Total elements: {len(elements)}")
    for i, elem in enumerate(elements[:5]):
        print(f"Element {i}: {elem.category} - {str(elem)[:100]}")
    print("=== End Debug ===\n")

    documents = []

    for element in elements:
        text = str(element).strip()

        if not text:
            continue

        page = None
        if hasattr(element.metadata, "page_number"):
            page = element.metadata.page_number

        is_table = element.category == "Table"

        metadata = {
            "source": uploaded_file.name,
            "page": page,
            "type": element.category,
            "is_table": is_table
        }

        doc = Document(
            page_content=text,
            metadata=metadata
        )

        if is_table:
            doc.metadata["chunk_size"] = "preserve"

        documents.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []
    for doc in documents:
        if doc.metadata.get("chunk_size") == "preserve":
            chunks.append(doc)
        else:
            chunks.extend(splitter.split_documents([doc]))

    return chunks

