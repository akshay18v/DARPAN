import os
import re
import cv2
import pandas as pd
import pytesseract
from tqdm import tqdm

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "yolo_dataset",
    "images"
)

LABEL_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "yolo_dataset",
    "labels"
)

CLASS_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "annotations",
    "classes.txt"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "text_regions.csv"
)

# ============================================================
# TESSERACT
# ============================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_FILE, "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f if line.strip()]

print("=" * 70)
print("DARPAN - TEXT DATASET CREATION")
print("=" * 70)

print("\nClasses:")

for i, class_name in enumerate(classes):
    print(i, "->", class_name)

# ============================================================
# FIND ALL IMAGES
# ============================================================

image_files = []

for root, dirs, files in os.walk(IMAGE_DIR):

    for file in files:

        if file.lower().endswith(
            (".png", ".jpg", ".jpeg", ".webp")
        ):
            image_files.append(
                os.path.join(root, file)
            )

print("\nTotal images found:", len(image_files))

# ============================================================
# FIND LABEL FILES
# ============================================================

label_files = {}

for root, dirs, files in os.walk(LABEL_DIR):

    for file in files:

        if file.lower().endswith(".txt"):

            full_path = os.path.join(root, file)

            # Use filename without extension
            stem = os.path.splitext(file)[0].lower()

            label_files[stem] = full_path

print("Total label files found:", len(label_files))

# ============================================================
# OCR CLEANING FUNCTION
# ============================================================

def clean_ocr_text(text):

    if not text:
        return ""

    # Replace newlines/tabs with spaces
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    # Remove common OCR decorative symbols/noise
    text = re.sub(
        r"[|¦•●○◆◇★☆►◄▲▼■□~`^*_{}[\]\\]+",
        " ",
        text
    )

    # Keep useful characters:
    # letters, numbers, spaces and useful punctuation
    text = re.sub(
        r"[^A-Za-z0-9₹£$€%.,!?():;/+\-'\"]",
        " ",
        text
    )

    # Multiple spaces -> one space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# OCR FUNCTION
# ============================================================

def extract_text(crop):

    if crop is None or crop.size == 0:
        return ""

    # Convert to grayscale
    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    # Enlarge crop
    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Improve contrast
    gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # OCR
    text = pytesseract.image_to_string(
        gray,
        config="--psm 6"
    )

    return clean_ocr_text(text)


# ============================================================
# PROCESS DATASET
# ============================================================

dataset_rows = []

for image_path in tqdm(
    image_files,
    desc="Processing images"
):

    image_name = os.path.basename(image_path)

    image_stem = os.path.splitext(
        image_name
    )[0].lower()

    # --------------------------------------------------------
    # Find corresponding label
    # --------------------------------------------------------

    label_path = label_files.get(image_stem)

    if label_path is None:

        print(
            f"\n⚠ No label found for: {image_name}"
        )

        continue

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:

        print(
            f"\n⚠ Could not read: {image_name}"
        )

        continue

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # Read labels
    # --------------------------------------------------------

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = f.readlines()

    # --------------------------------------------------------
    # Process every bounding box
    # --------------------------------------------------------

    region_id = 0

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        try:

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

        except ValueError:

            continue

        # ----------------------------------------------------
        # Check class ID
        # ----------------------------------------------------

        if class_id < 0 or class_id >= len(classes):

            print(
                f"\n⚠ Invalid class ID {class_id}"
                f" in {label_path}"
            )

            continue

        pattern_category = classes[class_id]

        # ----------------------------------------------------
        # YOLO coordinates -> pixels
        # ----------------------------------------------------

        x_center *= width
        y_center *= height

        box_width *= width
        box_height *= height

        x1 = int(
            x_center - box_width / 2
        )

        y1 = int(
            y_center - box_height / 2
        )

        x2 = int(
            x_center + box_width / 2
        )

        y2 = int(
            y_center + box_height / 2
        )

        # Keep inside image
        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(width, x2)
        y2 = min(height, y2)

        # ----------------------------------------------------
        # Crop annotated region
        # ----------------------------------------------------

        crop = image[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            continue

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        text = extract_text(crop)

        region_id += 1

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        dataset_rows.append({

            "image_name": image_name,

            "image_path": os.path.relpath(
                image_path,
                PROJECT_ROOT
            ),

            "region_id": region_id,

            "class_id": class_id,

            "pattern_category": pattern_category,

            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,

            "extracted_text": text
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(dataset_rows)

# ============================================================
# REMOVE EMPTY OCR RESULTS
# ============================================================

before = len(df)

df = df[
    df["extracted_text"].notna()
]

df = df[
    df["extracted_text"].str.strip() != ""
]

after = len(df)

# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TEXT DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print("\nOutput:")
print(OUTPUT_CSV)

print("\nTotal annotated regions:", before)

print("Regions with usable OCR:", after)

print("\nCategory distribution:")

print(
    df["pattern_category"]
    .value_counts()
)

print("\nSample data:")
print(
    df[
        [
            "image_name",
            "pattern_category",
            "extracted_text"
        ]
    ].head(10)
)

print("\n" + "=" * 70)