import fitz  # pymupdf
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document extracted from a PDF."""
    content: str
    metadata: dict


def extract_text_from_page(page: fitz.Page) -> str:
    """
    Extracts text from a page, ignoring non-textual elements.
    Blocks are sorted by vertical position to preserve reading order.
    """
    blocks = page.get_text("blocks")
    blocks_sorted = sorted(blocks, key=lambda b: (b[1], b[0]))

    texts = []
    for block in blocks_sorted:
        block_type = block[6]
        if block_type == 0:  # type 0 = text, type 1 = image
            text = block[4].strip()
            if text:
                texts.append(text)

    return "\n".join(texts)


def clean_text(text: str) -> str:
    """
    Basic cleanup of extracted text.
    Removes excessive blank lines and unnecessary whitespace.
    """
    lines = text.splitlines()
    cleaned = []
    previous_empty = False

    for line in lines:
        stripped = line.strip()
        is_empty = stripped == ""

        if is_empty and previous_empty:
            continue

        cleaned.append(stripped)
        previous_empty = is_empty

    return "\n".join(cleaned).strip()


def load_pdf(path: str | Path) -> Document:
    """
    Loads a PDF and returns a Document with extracted text and metadata.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {path}")

    logger.info(f"Loading PDF: {path.name}")

    doc = fitz.open(str(path))
    pages_text = []
    total_pages = len(doc)  # ← salva antes de fechar

    for page_num in range(total_pages):
        page = doc[page_num]
        text = extract_text_from_page(page)
        text = clean_text(text)

        if text:
            pages_text.append({
                "page": page_num + 1,
                "text": text
            })

    doc.close()

    full_text = "\n\n".join(p["text"] for p in pages_text)

    metadata = {
        "source": path.name,
        "total_pages": total_pages,  # ← usa a variável salva
        "pages_with_text": len(pages_text),
        "language": "en" if path.stem.endswith("_en") else "pt"
    }

    logger.info(
        f"Extracted: {len(pages_text)}/{total_pages} pages "
        f"with text from {path.name}"
    )

    return Document(content=full_text, metadata=metadata)


def load_all_pdfs(directory: str | Path) -> list[Document]:
    """
    Loads all PDFs from a directory.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDFs found in: {directory}")
        return []

    logger.info(f"Found {len(pdf_files)} PDFs in {directory}")

    documents = []
    for pdf_path in pdf_files:
        try:
            doc = load_pdf(pdf_path)
            documents.append(doc)
        except Exception as e:
            logger.error(f"Failed to load {pdf_path.name}: {e}")
            continue

    return documents