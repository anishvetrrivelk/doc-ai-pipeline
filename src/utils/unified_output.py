"""
Combines all extracted components into one unified JSON schema.
"""

from typing import Dict, Any, List


def build_page_json(
    page_num: int,
    text: str,
    tables: list,
    images: list,
    equations_text: list,
    equations_img: list = None
) -> dict:
    """
    equations_img: optional list of dicts for equations detected in images
    """
    return {
        "page": page_num,
        "text": text,
        "tables": [
            {"table_index": t["table_index"], "data": t["data"]}
            for t in tables if t["page"] == page_num
        ],
        "images": [
            {
                "bbox": img["bbox"],
                "equations_in_image": equations_img if equations_img else []
            }
            for img in images if img["page"] == page_num
        ],
        "equations": equations_text
    }


def build_document_json(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "total_pages": len(pages),
        "pages": pages
    }
