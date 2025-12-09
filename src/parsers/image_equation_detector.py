"""
Detect equations in images using OpenCV.
Outputs bounding boxes of potential equations.
"""

import cv2
import numpy as np
from typing import List, Dict

def detect_equations_in_image(img_array: np.ndarray) -> List[Dict[str, int]]:
    """
    Input: numpy array image
    Output: list of bounding boxes {"x": x, "y": y, "w": w, "h": h}
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter by size: likely equation areas
        if w > 30 and h > 10:  # tweak thresholds for your documents
            boxes.append({"x": x, "y": y, "w": w, "h": h})

    return boxes


if __name__ == "__main__":
    import sys
    from PIL import Image
    import numpy as np

    if len(sys.argv) < 2:
        print("Usage: python -m src.parsers.image_equation_detector <image_path>")
    else:
        image_path = sys.argv[1]
        img = np.array(Image.open(image_path).convert("RGB"))
        boxes = detect_equations_in_image(img)
        print(f"Detected {len(boxes)} potential equation regions")
