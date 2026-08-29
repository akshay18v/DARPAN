# ============================================================
# DARPAN - OCR TEXT REGION EXTRACTION
# Tesseract OCR + Bounding Boxes
# ============================================================

import os
import sys
import cv2
import pytesseract
from pytesseract import Output


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# OCR FUNCTION
# ============================================================

def extract_text_regions(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Unable to read image:\n{image_path}"
        )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )

    processed = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # --------------------------------------------------------
    # OCR WITH POSITION INFORMATION
    # --------------------------------------------------------

    data = pytesseract.image_to_data(
        processed,
        config="--psm 6",
        output_type=Output.DICT
    )

    regions = []

    n = len(data["text"])

    for i in range(n):

        text = data["text"][i].strip()

        try:
            confidence = float(
                data["conf"][i]
            )
        except:
            confidence = -1

        # Ignore empty text
        if not text:
            continue

        # Ignore very poor OCR detections
        if confidence < 30:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        region = {
            "text": text,
            "confidence": confidence,
            "x": x,
            "y": y,
            "width": w,
            "height": h
        }

        regions.append(region)

    return regions


# ============================================================
# DISPLAY OCR REGIONS
# ============================================================

def display_regions(regions):

    print()
    print("=" * 70)
    print("OCR TEXT REGIONS")
    print("=" * 70)

    print(
        f"\nTotal detected regions: {len(regions)}"
    )

    print()

    for index, region in enumerate(regions, start=1):

        print(
            f"[{index}] "
            f"{region['text']}"
        )

        print(
            f"    Confidence : "
            f"{region['confidence']:.2f}%"
        )

        print(
            f"    Position   : "
            f"x={region['x']}, "
            f"y={region['y']}, "
            f"w={region['width']}, "
            f"h={region['height']}"
        )

        print("-" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DARPAN - OCR TEXT REGION EXTRACTION")
    print("=" * 70)

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print(
            "python src/text_extract_regions.py "
            "<image_path>"
        )

        print()
        print("Example:")
        print(
            r"python src\text_extract_regions.py "
            r"data\yolo_dataset\images\misdirection_2.3.png"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    print()
    print("Image:")
    print(image_path)

    print()
    print("Extracting text regions...")
    print("-" * 70)

    try:

        regions = extract_text_regions(
            image_path
        )

        display_regions(regions)

        print()
        print("=" * 70)
        print("OCR REGION EXTRACTION COMPLETED")
        print("=" * 70)

    except Exception as e:

        print()
        print("OCR ERROR")
        print("-" * 70)
        print(str(e))
        print("-" * 70)