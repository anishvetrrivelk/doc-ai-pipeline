"""
src/parsers/equation_detector.py

Detect equation-like regions inside images (or PDF page images).
- Uses classical CV (grayscale -> adaptive threshold -> morphology -> contours)
- Filters contours by aspect, area, and stroke density heuristics
- Returns list of crops and optional OCR using pytesseract (best-effort)
- Intended to be a preprocessor for a stronger math OCR (pix2tex / MathPix)
"""

import cv2
import numpy as np
from PIL import Image
import os
import pytesseract
from typing import List, Dict, Tuple


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _binarize(gray: np.ndarray) -> np.ndarray:
    # Use adaptive threshold — good for variable lighting
    return cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 25, 10)


def _morph_close(bin_img: np.ndarray, kx: int = 15, ky: int = 3) -> np.ndarray:
    # Close gaps horizontally to join long equations or inline math
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    return cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)


def _find_candidate_contours(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _is_equation_like(contour: np.ndarray, img_shape: Tuple[int, int]) -> bool:
    x, y, w, h = cv2.boundingRect(contour)
    img_h, img_w = img_shape[:2]

    # Heuristics:
    # - Minimum area
    area = w * h
    if area < 400:  # too small
        return False

    # - Not a full-page box
    if w > img_w * 0.95 and h > img_h * 0.95:
        return False

    # - Aspect ratio: equations often are wide or tall with fractions; accept wider boxes too
    ar = w / (h + 1e-6)
    if ar < 0.2 and ar > 10:
        return False

    # - Accept typical eq sizes
    if h < 10 or w < 10:
        return False

    return True


def detect_equations_in_image(img: np.ndarray,
                              save_crops: bool = False,
                              out_dir: str = "data/equation_crops",
                              ocr: bool = True) -> List[Dict]:
    """
    Detect equation regions in an image (numpy array, BGR or grayscale).
    Returns list of dicts:
    {
       "bbox": (x,y,w,h),
       "crop": np.ndarray,
       "saved_path": optional path,
       "tesseract_text": optional OCR text
    }
    """
    _ensure_dir(out_dir) if save_crops else None

    gray = _to_gray(img)
    bin_img = _binarize(gray)
    closed = _morph_close(bin_img, kx=25, ky=3)

    contours = _find_candidate_contours(closed)
    candidates = []

    # sort contours left->right, top->bottom (helpful when saving)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1] * img.shape[1] + cv2.boundingRect(c)[0])

    for idx, c in enumerate(contours):
        if not _is_equation_like(c, img.shape):
            continue

        x, y, w, h = cv2.boundingRect(c)
        margin_x = int(w * 0.03)  # small margin
        margin_y = int(h * 0.05)
        x0 = max(0, x - margin_x)
        y0 = max(0, y - margin_y)
        x1 = min(img.shape[1], x + w + margin_x)
        y1 = min(img.shape[0], y + h + margin_y)
        crop = img[y0:y1, x0:x1].copy()

        entry = {
            "bbox": (x0, y0, x1 - x0, y1 - y0),
            "crop": crop
        }

        if save_crops:
            fname = f"eq_{idx:03d}.png"
            path = os.path.join(out_dir, fname)
            cv2.imwrite(path, crop)
            entry["saved_path"] = path

        if ocr:
            try:
                # Use Tesseract with a math-friendly config (best-effort)
                pil = Image.fromarray(crop if len(crop.shape) == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                # psm 6: Assume a single uniform block of text; OEM 3 default
                custom_config = r'--psm 6'
                txt = pytesseract.image_to_string(pil, config=custom_config)
                entry["tesseract_text"] = txt.strip()
            except Exception as e:
                entry["tesseract_text"] = ""
                entry["ocr_error"] = str(e)

        candidates.append(entry)

    return candidates


def detect_equations_from_pdf_page_image(page_image_path: str,
                                         save_crops: bool = True,
                                         out_dir: str = "data/equation_crops",
                                         ocr: bool = True) -> List[Dict]:
    """
    Convenience helper: load an image file path (e.g., a PDF page rendered to PNG),
    and run detect_equations_in_image.
    """
    img = cv2.imread(page_image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {page_image_path}")
    return detect_equations_in_image(img, save_crops=save_crops, out_dir=out_dir, ocr=ocr)


if __name__ == "__main__":
    # quick demo (python -m src.parsers.equation_detector <image_path>)
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.parsers.equation_detector <image_path>")
    else:
        imgp = sys.argv[1]
        res = detect_equations_from_pdf_page_image(imgp, save_crops=True, out_dir="data/equation_crops", ocr=True)
        print(f"Found {len(res)} candidate equation regions")
        for i, r in enumerate(res):
            print(i, r.get("bbox"), "->", r.get("tesseract_text", "")[:160])
