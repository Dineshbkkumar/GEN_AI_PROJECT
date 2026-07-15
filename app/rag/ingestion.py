from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_pdf_content(uploaded_file):
    elements = partition_pdf(
        file=uploaded_file,
        strategy="hi_res",
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

        documents.append(
            Document(
                page_content=text,
                metadata=metadata
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
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

