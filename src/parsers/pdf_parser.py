"""
src/parsers/pdf_parser.py

Simple, robust PDF parsing utilities using pdfplumber and pypdf.
Provides:
 - full text extraction (per page & whole doc)
 - metadata extraction
 - basic cleaning helpers
"""

from typing import Dict, List, Optional
import pdfplumber
from pypdf import PdfReader


def extract_text_pages(pdf_path: str) -> List[str]:
    """Return list of page texts (one string per page)."""
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)
    except Exception as e:
        raise RuntimeError(f"Failed to extract pages from {pdf_path}: {e}")
    return pages


def extract_text_full(pdf_path: str, join_with: str = "\n\n") -> str:
    """Return whole-document text by joining page-level text."""
    pages = extract_text_pages(pdf_path)
    return join_with.join(pages)


def extract_metadata(pdf_path: str) -> Dict[str, Optional[str]]:
    """Return simple metadata from the PDF (title, author, producer, etc.)."""
    try:
        reader = PdfReader(pdf_path)
        md = reader.metadata or {}
        # Normalize keys to simple names
        return {
            "title": md.get("/Title"),
            "author": md.get("/Author"),
            "creator": md.get("/Creator"),
            "producer": md.get("/Producer"),
            "creation_date": md.get("/CreationDate"),
            "mod_date": md.get("/ModDate"),
        }
    except Exception as e:
        raise RuntimeError(f"Failed to read metadata from {pdf_path}: {e}")


def extract_images(pdf_path: str, max_pages: Optional[int] = None) -> List[Dict]:
    """
    Extract embedded raster images using pdfplumber.
    Returns list of dicts: { 'page': int, 'name': str, 'stream': bytes, 'width': int, 'height': int }
    Note: Writing images to disk is left to caller.
    """
    images = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
            for i, page in enumerate(pages, start=1):
                for img_idx, im in enumerate(page.images, start=1):
                    # pdfplumber's image dict contains "stream" for image bytes
                    images.append({
                        "page": i,
                        "name": f"page{i}_img{img_idx}",
                        "stream": im.get("stream"),
                        "width": im.get("width"),
                        "height": im.get("height"),
                        "object": im,
                    })
    except Exception as e:
        # don't fail hard — return what we have or raise depending on caller needs
        raise RuntimeError(f"Failed to extract images from {pdf_path}: {e}")
    return images


def save_image_bytes(image_dict: Dict, out_path: str):
    """
    Save a single image (from extract_images) to disk.
    image_dict must contain 'stream' (bytes) and out_path is the target filename.
    """
    stream = image_dict.get("stream")
    if not stream:
        raise ValueError("image_dict missing 'stream' bytes")
    with open(out_path, "wb") as f:
        f.write(stream)


if __name__ == "__main__":
    # quick local test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.parsers.pdf_parser <pdf_path>")
    else:
        p = sys.argv[1]
        print("Metadata:", extract_metadata(p))
        print("First page text preview:")
        pages = extract_text_pages(p)
        print(pages[0][:1000] if pages else "<no pages>")
