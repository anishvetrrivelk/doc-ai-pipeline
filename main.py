"""
Main pipeline: PDF → preprocessing → text → tables → images → equations → JSON
"""

from src.parsers.table_extractor import extract_tables
from src.parsers.image_parser import extract_images
from src.parsers.equation_detector import detect_equations_in_text
from src.slm.slm_model import tag
from src.utils.unified_output import build_page_json, build_document_json

import pdfplumber


def process_pdf(pdf_path: str):
    pages_output = []

    tables = extract_tables(pdf_path)
    images = extract_images(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            equations = detect_equations_in_text(text)

            pages_output.append(
                build_page_json(
                    page_num=i,
                    text=text,
                    tables=tables,
                    images=images,
                    equations=equations
                )
            )

    return build_document_json(pages_output)


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py <pdf_path>")
        exit()

    result = process_pdf(sys.argv[1])
    print(json.dumps(result, indent=2))
