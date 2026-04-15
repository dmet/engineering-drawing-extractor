import cv2
import os
import shutil
import pytesseract
import numpy as np
from table_parser import TableParser

# --- Tesseract path: auto-detect or set via environment variable --- #
_tess_env = os.environ.get("TESSERACT_CMD")
if _tess_env:
    pytesseract.pytesseract.tesseract_cmd = _tess_env
elif shutil.which("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
else:
    pytesseract.pytesseract.tesseract_cmd = (
        r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    )

_parser = TableParser()


def GetString(img, keyword1, keyword2):
    """Extract a value from a table cell identified by keyword matching.

    Scans contours in *img* for cells containing *keyword1* or *keyword2*,
    then returns the first non-keyword text line below the match.
    """
    copy = img.copy()
    nrow, ncol = img.shape

    blur = cv2.GaussianBlur(copy, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 127, 1, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if not (40000 < area < 5000000):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), -1)
        ROI = copy[y:y + h, x:x + w]

        string = pytesseract.image_to_string(ROI, config='--psm 6').strip()
        if not string:
            continue

        if keyword1 not in string and keyword2 not in string:
            continue

        # Extend region downward to capture the value below the keyword
        ROI = copy[y:y + h + 100, x + 10:x + w]
        # Reuse TableParser's line-removal instead of duplicating logic
        cleaned = _parser.clean_cell_for_ocr(ROI)

        string = pytesseract.image_to_string(cleaned,
                                             config='--psm 6').strip()
        lines = string.splitlines()

        for i, line in enumerate(lines):
            if keyword1 in line or keyword2 in line:
                # Return the first non-empty line after the keyword
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        return lines[j].strip()
                return ""

        return ""

    return ""
