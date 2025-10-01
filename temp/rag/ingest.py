import fitz  # PyMuPDF
import re

def load_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    pages = []
    for p in doc:
        text = p.get_text("text")
        text = re.sub(r"\s{2,}", " ", text)
        pages.append(text)
    return "\f".join(pages)

def clean_text(full_text: str) -> str:
    pages = full_text.split("\f")
    cleaned = []
    for pg in pages:
        pg = re.sub(r"전자공시시스템.*?(Page\s*\d+)?", "", pg, flags=re.IGNORECASE)
        pg = re.sub(r"\s{2,}", " ", pg)
        cleaned.append(pg.strip())
    return "\n".join(cleaned)