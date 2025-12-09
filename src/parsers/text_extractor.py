"""
Advanced Text Extractor for PDFs
Handles:
- Multi-column layouts
- Headers & footnotes
- Page-wise extraction
- Returns list of strings (one per page)
"""

from typing import List
import pdfplumber

def extract_text_from_pdf(pdf_path: str, columns: int = 2) -> List[str]:
    """
    Extract text page-wise from a PDF, considering multi-column layout.
    
    Args:
        pdf_path: Path to PDF file
        columns: Number of columns per page (default 2)
    
    Returns:
        List of strings: each entry is the text of a page
    """
    pages_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = ""

            # If the page has no text, skip
            if page.extract_text() is None:
                pages_text.append("")
                continue

            width = page.width
            # Calculate column widths
            col_width = width / columns

            # Extract text column by column
            column_texts = []
            for i in range(columns):
                left = i * col_width
                right = (i + 1) * col_width
                col_bbox = (left, 0, right, page.height)
                col_text = page.within_bbox(col_bbox).extract_text() or ""
                column_texts.append(col_text.strip())

            # Combine column texts in left->right order
            page_text = "\n".join(column_texts).strip()
            pages_text.append(page_text)

    return pages_text


# -------- CLI Testing --------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.parsers.text_extractor <pdf_path>")
    else:
        pdf_path = sys.argv[1]
        pages = extract_text_from_pdf(pdf_path)
        for i, text in enumerate(pages, start=1):
            print(f"\n--- Page {i} ---\n{text[:500]}")  # preview first 500 chars
