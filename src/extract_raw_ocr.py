import os
import cv2
import pandas as pd
import pytesseract
from tqdm import tqdm

# ============================================================
# DARPAN - RAW SCREENSHOT OCR EXTRACTION
# IMPORTANT:
# This script DOES NOT modify/delete any existing data.
# It only creates a new temporary OCR CSV.
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "raw_ocr_candidates.csv"
)

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# OCR
# ============================================================

def extract_text(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return ""

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Upscale for better OCR
    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # OTSU threshold
    processed = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    text = pytesseract.image_to_string(
        processed,
        config="--psm 6"
    )

    # Basic cleaning
    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:
            lines.append(line)

    return " ".join(lines)


# ============================================================
# FIND RAW IMAGES
# ============================================================

image_files = []

for root, dirs, files in os.walk(RAW_DIR):

    for file in files:

        if file.lower().endswith(
            (".png", ".jpg", ".jpeg", ".webp")
        ):

            image_files.append(
                os.path.join(root, file)
            )

image_files.sort()


print("=" * 70)
print("DARPAN - RAW SCREENSHOT OCR EXTRACTION")
print("=" * 70)

print()
print("Raw directory:")
print(RAW_DIR)

print()
print("Images found:", len(image_files))

print()
print("Starting OCR...")
print("-" * 70)


# ============================================================
# PROCESS
# ============================================================

rows = []

for image_path in tqdm(
    image_files,
    desc="OCR"
):

    relative_path = os.path.relpath(
        image_path,
        PROJECT_ROOT
    )

    image_name = os.path.basename(
        image_path
    )

    # Website = first folder inside data/raw
    relative_to_raw = os.path.relpath(
        image_path,
        RAW_DIR
    )

    parts = relative_to_raw.split(
        os.sep
    )

    website = parts[0] if len(parts) > 1 else "unknown"

    text = extract_text(
        image_path
    )

    rows.append({
        "image_name": image_name,
        "image_path": relative_path,
        "website": website,
        "source": "raw",
        "ocr_text": text
    })


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(rows)

# Remove completely empty OCR
df["ocr_text"] = (
    df["ocr_text"]
    .fillna("")
    .astype(str)
)

usable = df[
    df["ocr_text"].str.strip() != ""
].copy()


# ============================================================
# SAVE
# ============================================================

usable.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("RAW OCR EXTRACTION COMPLETED")
print("=" * 70)

print()
print("Images processed :", len(df))
print("Usable OCR rows  :", len(usable))
print("Empty OCR rows   :", len(df) - len(usable))

print()
print("Websites:")
print(
    usable["website"].value_counts()
)

print()
print("Output:")
print(OUTPUT_CSV)

print()
print("=" * 70)
print("IMPORTANT:")
print("No existing image, annotation or metadata was modified.")
print("No labels were assigned automatically.")
print("=" * 70)