from typing import List, Tuple

from pypdf import PdfReader


def load_pdf_text(pdf_path: str) -> List[Tuple[int, str]]:
    """Return list of (page_number starting at 1, page_text)."""
    reader = PdfReader(pdf_path)
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        page_no = i + 1
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append((page_no, txt))
    return pages
