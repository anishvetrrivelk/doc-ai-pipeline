"""
src/parsers/image_parser.py

Extracts images from PDF using pdfplumber.
Returns image bytes or numpy arrays.
"""

from typing import List, Dict, Any
import pdfplumber
from PIL import Image
import io
import numpy as np


def extract_images(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract images from all PDF pages.

    Returns list of:
    {
        "page": int,
        "bbox": (x0, y0, x1, y1),
        "image": PIL.Image,
        "numpy_array": np.ndarray
    }
    """
    results = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for img in page.images:
                x0, y0, x1, y1 = img["x0"], img["y0"], img["x1"], img["y1"]
                
                # Crop the image from the page
                cropped = page.within_bbox((x0, y0, x1, y1)).to_image(resolution=300)
                pil_image = cropped.original

                results.append({
                    "page": page_num,
                    "bbox": (x0, y0, x1, y1),
                    "image": pil_image,
                    "numpy_array": np.array(pil_image)
                })

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.parsers.image_parser <pdf_path>")
    else:
        pdf = sys.argv[1]
        pics = extract_images(pdf)
        print(f"Extracted {len(pics)} images.")
