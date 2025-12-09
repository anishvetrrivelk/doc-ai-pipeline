import cv2
import numpy as np
from PIL import Image
import pytesseract
import easyocr


class ImagePreprocessor:
    def __init__(self, use_easyocr=True, use_tesseract=True):
        self.use_easyocr = use_easyocr
        self.use_tesseract = use_tesseract

        if self.use_easyocr:
            self.reader = easyocr.Reader(['en'], gpu=False)

    # -------------------------
    # Load image
    # -------------------------
    def load_image(self, path):
        return cv2.imread(path)

    # -------------------------
    # Convert to grayscale
    # -------------------------
    def to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # Noise removal
    # -------------------------
    def denoise(self, img):
        return cv2.fastNlMeansDenoising(img, h=10)

    # -------------------------
    # Threshold / Binarization
    # -------------------------
    def threshold(self, img):
        return cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

    # -------------------------
    # Sharpening Filter
    # -------------------------
    def sharpen(self, img):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

    # -------------------------
    # Deskew image (fix tilt)
    # -------------------------
    def deskew(self, img):
        gray = self.to_gray(img)
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    # -------------------------
    # Full processing pipeline
    # -------------------------
    def process(self, img_path):
        img = self.load_image(img_path)
        gray = self.to_gray(img)
        den = self.denoise(gray)
        th = self.threshold(den)
        sh = self.sharpen(th)
        final = self.deskew(sh)
        return final

    # -------------------------
    # OCR functions
    # -------------------------
    def ocr_easyocr(self, img_path):
        result = self.reader.readtext(img_path, detail=0)
        return "\n".join(result)

    def ocr_tesseract(self, img):
        text = pytesseract.image_to_string(Image.fromarray(img))
        return text

    # -------------------------
    # Unified OCR
    # -------------------------
    def extract_text(self, img_path):
        processed = self.process(img_path)

        text_out = {}

        if self.use_easyocr:
            text_out["easyocr"] = self.ocr_easyocr(img_path)

        if self.use_tesseract:
            text_out["tesseract"] = self.ocr_tesseract(processed)

        return {
            "processed_image": processed,
            "text": text_out
        }

