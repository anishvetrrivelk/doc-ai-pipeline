"""
Simple Small Language Model (SLM) for tagging and section analysis.
"""

from typing import Dict, Any

KEYWORDS = {
    "introduction": "section:introduction",
    "methodology": "section:method",
    "results": "section:results",
    "table": "contains:table",
    "figure": "contains:image",
    "equation": "contains:math"
}


def encode(text: str):
    return text.lower().strip()


def tag(text: str) -> Dict[str, Any]:
    encoded = encode(text)
    labels = []

    for k, v in KEYWORDS.items():
        if k in encoded:
            labels.append(v)

    return {
        "text": text,
        "labels": labels
    }
