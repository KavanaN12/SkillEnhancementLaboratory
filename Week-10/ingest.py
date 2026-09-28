from pathlib import Path
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = Path("data/2023060228.pdf")
VECTOR_DIR = Path("vectorstore/ipc_faiss")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# Matches actual IPC section headings such as:
# 302. Punishment for murder —
# 304A. Causing death by negligence —
SECTION_PATTERN = re.compile(
    r"(?m)^\s*(\d{1,3}[A-Z]?)\.\s+(.+?)\s*(?:—|-)\s*$"
)


def clean_text(text: str) -> str:
    """Clean PDF extraction artifacts without destroying paragraph structure."""

    text = text.replace("\x00", " ")

    # Remove excessive spaces while preserving newlines.
    text = re.sub(r"[ \t]+", " ", text)

    # Fix spaces before punctuation caused by PDF extraction.
    text = re.sub(r"\s+([,.;:])", r"\1", text)

    # Keep at most two consecutive newlines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_sections(pages):
    """
    Convert the page-based PDF into section-aware documents.

    A section may continue across multiple PDF pages, so we keep track
    of the current section while walking through the document.
    """

    sections = []

    current_section = "General"
    current_title = "General"
    current_pages = []
    current_text = []

    def save_current_section():
        if not current_text:
            return

        text = "\n\n".join(current_text).strip()

        if not text:
            return

        sections.append(
            {
                "section": current_section,
                "title": current_title,
                "text": text,
                "pages": sorted(set(current_pages)),
            }
        )

    for page in pages:
        page_number = int(page.metadata.get("page", 0)) + 1
        text = clean_text(page.page_content)

        if not text:
            continue

        lines = text.splitlines()
        section_found_on_page = False
        buffer = []

        for line in lines:
            line = line.strip()

            match = SECTION_PATTERN.match(line)

            if match:
                # Save text belonging to the previous section.
                if buffer:
                    current_text.append("\n".join(buffer))
                    buffer = []

                save_current_section()

                current_section = match.group(1)
                current_title = match.group(2).strip()

                current_pages = [page_number]
                current_text = [line]

                section_found_on_page = True

            else:
                buffer.append(line)

        # Remaining text belongs to the current section.
        if buffer:
            current_text.append("\n".join(buffer))

        if not section_found_on_page:
            current_pages.append(page_number)

    # Save final section.
    save_current_section()

    return sections


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"IPC PDF not found: {PDF_PATH}"
        )

    print("Loading IPC PDF...")

    loader = PyPDFLoader(str(PDF_PATH))
    pages = loader.load()

    pages = [
        page for page in pages
        if page.page_content and page.page_content.strip()
    ]

    print(f"Loaded {len(pages)} non-empty pages.")

    print("Building section-aware document structure...")

    sections = extract_sections(pages)

    print(f"Detected {len(sections)} section/document blocks.")

    # Split long sections into smaller retrieval chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            " ",
            "",
        ],
    )

    chunks = []

    chunk_id = 0

    for section_data in sections:

        section = section_data["section"]
        title = section_data["title"]
        text = section_data["text"]
        pages_used = section_data["pages"]

        # Create a temporary document for splitting.
        from langchain_core.documents import Document

        section_doc = Document(
            page_content=text,
            metadata={
                "section": section,
                "section_title": title,
                "pages": pages_used,
                "source_file": PDF_PATH.name,
                "source_type": "Indian Penal Code document",
            },
        )

        section_chunks = splitter.split_documents([section_doc])

        for chunk in section_chunks:

            chunk.metadata["chunk_id"] = chunk_id

            # Keep the exact section metadata.
            chunk.metadata["section"] = section
            chunk.metadata["section_title"] = title
            chunk.metadata["pages"] = pages_used
            chunk.metadata["source_file"] = PDF_PATH.name

            chunks.append(chunk)

            chunk_id += 1

    print(f"Created {len(chunks)} section-aware chunks.")

    print("Creating embeddings and FAISS index...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    db = FAISS.from_documents(chunks, embeddings)

    VECTOR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    db.save_local(str(VECTOR_DIR))

    print(f"FAISS index saved to: {VECTOR_DIR}")
    print("Ingestion complete.")


if __name__ == "__main__":
    main()