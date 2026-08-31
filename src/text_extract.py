import os
import sys
import cv2
import pytesseract


# ============================================================
# DARPAN - TEXT EXTRACTION USING TESSERACT OCR
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# Tesseract installation path
TESSERACT_PATH = r"D:\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# OCR FUNCTION
# ============================================================

def extract_text_from_image(image_path):
    """
    Extract text from an image using Tesseract OCR.

    Parameters:
        image_path (str): Path to screenshot/image

    Returns:
        str: Extracted OCR text
    """

    # --------------------------------------------------------
    # Check image path
    # --------------------------------------------------------

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
    # Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------
    # Improve OCR readability
    # --------------------------------------------------------

    gray = cv2.resize(
        gray,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # Thresholding
    # --------------------------------------------------------

    processed = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # --------------------------------------------------------
    # Tesseract OCR
    # --------------------------------------------------------

    text = pytesseract.image_to_string(
        processed,
        config="--psm 6"
    )

    # --------------------------------------------------------
    # Clean basic whitespace
    # --------------------------------------------------------

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:
            lines.append(line)

    extracted_text = "\n".join(lines)

    return extracted_text


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DARPAN - TESSERACT OCR TEXT EXTRACTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Image path from command line
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print("python src/text_extract.py <image_path>")
        print()

        print("Example:")
        print(
            r"python src/text_extract.py data\yolo_dataset\images\misdirection_2.3.png"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    print()
    print("Image:")
    print(image_path)

    print()
    print("Extracting text...")
    print("-" * 70)

    try:

        extracted_text = extract_text_from_image(image_path)

        print()
        print("EXTRACTED TEXT")
        print("=" * 70)

        if extracted_text:

            print(extracted_text)

        else:

            print("No text detected.")

        print("=" * 70)

        print()
        print("OCR extraction completed successfully.")

    except Exception as e:

        print()
        print("OCR ERROR")
        print("-" * 70)
        print(str(e))
        print("-" * 70)