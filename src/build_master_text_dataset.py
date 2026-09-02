import os
import re
import cv2
import pandas as pd
import pytesseract
from tqdm import tqdm

# ============================================================
# DARPAN - MASTER TEXT DATASET BUILDER
# ============================================================
# IMPORTANT:
# - Does NOT delete existing data
# - Does NOT modify raw/synthetic/synthetic2/YOLO data
# - Creates ONE new master CSV
# - Final labels = 6 dark patterns + normal
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "darpan_master_text_dataset.csv"
)

# ------------------------------------------------------------
# TESSERACT
# ------------------------------------------------------------

TESSERACT_PATH = r"D:\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ------------------------------------------------------------
# FINAL 7 CLASSES
# ------------------------------------------------------------

VALID_LABELS = [
    "hidden_fees",
    "false_urgency",
    "sneaking",
    "misdirection",
    "interface_interference",
    "obstruction",
    "normal"
]

DARK_LABELS = set(VALID_LABELS[:-1])

# ------------------------------------------------------------
# CREATE OUTPUT DIRECTORY
# ------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 75)
print("DARPAN - MASTER TEXT DATASET BUILDER")
print("=" * 75)

print("\nFinal classes:")
for i, label in enumerate(VALID_LABELS):
    print(f"{i}: {label}")


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(value):
    """
    Convert different existing label spellings
    into the six canonical dark-pattern labels.

    Unknown / blank values return None.
    """

    if value is None:
        return None

    value = str(value).strip().lower()

    if not value:
        return None

    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    mapping = {

        "hidden fees": "hidden_fees",
        "hidden fee": "hidden_fees",
        "hidden_fees": "hidden_fees",

        "false urgency": "false_urgency",
        "false urgency ": "false_urgency",
        "false_urgency": "false_urgency",

        "sneaking": "sneaking",

        "misdirection": "misdirection",

        "interface interference": "interface_interference",
        "interface_interference": "interface_interference",

        "obstruction": "obstruction",

        # DO NOT automatically convert this to either class.
        "obstruct sneaking": None,
        "obstruct_sneaking": None,
    }

    return mapping.get(value)


# ============================================================
# OCR CLEANING
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    # Common OCR noise
    text = re.sub(
        r"[|¦•★☆►◄▲▼■~`^*_{}[\]\\]+",
        " ",
        text
    )

    # Keep useful characters
    text = re.sub(
        r"[^A-Za-z0-9₹£$€%.,!?():;/+\-'\"]",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# OCR
# ============================================================

def extract_ocr(image):

    if image is None or image.size == 0:
        return ""

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=2.5,
        fy=2.5,
        interpolation=cv2.INTER_CUBIC
    )

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

    return clean_text(text)


# ============================================================
# DATA ROW FORMAT
# ============================================================
#
# EXACTLY 7 COLUMNS:
#
# text
# label
# source
# image_name
# website
# pattern_type
# comment
#
# ============================================================

rows = []


# ============================================================
# SOURCE 1
# YOLO ANNOTATED REGIONS
# ============================================================

print("\n" + "=" * 75)
print("SOURCE 1: YOLO ANNOTATED REGIONS")
print("=" * 75)

YOLO_IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "yolo_dataset",
    "images"
)

YOLO_LABEL_DIR = os.path.join(
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

if not os.path.exists(CLASS_FILE):

    raise FileNotFoundError(
        f"Missing classes file:\n{CLASS_FILE}"
    )

with open(
    CLASS_FILE,
    "r",
    encoding="utf-8"
) as f:

    yolo_classes = [
        line.strip()
        for line in f
        if line.strip()
    ]

print("\nYOLO classes:")
for i, name in enumerate(yolo_classes):
    print(i, "->", name)


# ------------------------------------------------------------
# Find images recursively
# ------------------------------------------------------------

yolo_images = []

for root, dirs, files in os.walk(YOLO_IMAGE_DIR):

    for file in files:

        if file.lower().endswith(
            (".png", ".jpg", ".jpeg", ".webp")
        ):

            yolo_images.append(
                os.path.join(root, file)
            )

print("\nYOLO images found:", len(yolo_images))


# ------------------------------------------------------------
# Label lookup
# ------------------------------------------------------------

yolo_label_files = {}

for root, dirs, files in os.walk(YOLO_LABEL_DIR):

    for file in files:

        if file.lower().endswith(".txt"):

            stem = os.path.splitext(
                file
            )[0].lower()

            yolo_label_files[stem] = os.path.join(
                root,
                file
            )


# ------------------------------------------------------------
# Process annotations
# ------------------------------------------------------------

yolo_rows = 0

for image_path in tqdm(
    yolo_images,
    desc="YOLO OCR"
):

    image_name = os.path.basename(image_path)

    stem = os.path.splitext(
        image_name
    )[0].lower()

    label_path = yolo_label_files.get(stem)

    if label_path is None:
        continue

    image = cv2.imread(image_path)

    if image is None:
        continue

    height, width = image.shape[:2]

    try:

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            annotation_lines = f.readlines()

    except Exception:
        continue

    for line in annotation_lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        try:

            class_id = int(parts[0])

            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

        except ValueError:
            continue

        if class_id < 0 or class_id >= len(yolo_classes):
            continue

        raw_label = yolo_classes[class_id]

        label = normalize_label(raw_label)

        # Unknown class is NOT forced into a class
        if label not in DARK_LABELS:
            continue

        # YOLO -> pixels
        xc *= width
        yc *= height
        bw *= width
        bh *= height

        x1 = max(
            0,
            int(xc - bw / 2)
        )

        y1 = max(
            0,
            int(yc - bh / 2)
        )

        x2 = min(
            width,
            int(xc + bw / 2)
        )

        y2 = min(
            height,
            int(yc + bh / 2)
        )

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[
            y1:y2,
            x1:x2
        ]

        text = extract_ocr(crop)

        if len(text) < 3:
            continue

        rows.append({

            "text": text,

            "label": label,

            "source": "yolo_annotated",

            "image_name": image_name,

            "website": "unknown",

            "pattern_type": label,

            "comment": "OCR from annotated YOLO region"

        })

        yolo_rows += 1


print("\nYOLO text rows added:", yolo_rows)


# ============================================================
# SOURCE 2 + SOURCE 3
# SYNTHETIC / SYNTHETIC2
# ============================================================

def process_folder_dataset(
    folder_name,
    source_name
):

    folder = os.path.join(
        PROJECT_ROOT,
        "data",
        folder_name
    )

    if not os.path.exists(folder):

        print(
            f"\n{folder_name}: folder not found, skipped."
        )

        return 0

    print("\n" + "=" * 75)
    print(f"SOURCE: {folder_name}")
    print("=" * 75)

    added = 0

    # --------------------------------------------------------
    # We ONLY trust folder names that map to known classes.
    # --------------------------------------------------------

    for root, dirs, files in os.walk(folder):

        current_folder = os.path.basename(
            root
        )

        folder_label = normalize_label(
            current_folder
        )

        # Normal is intentionally not inferred here
        if folder_label not in DARK_LABELS:
            continue

        for file in files:

            if not file.lower().endswith(
                (".png", ".jpg", ".jpeg", ".webp")
            ):
                continue

            image_path = os.path.join(
                root,
                file
            )

            image = cv2.imread(image_path)

            if image is None:
                continue

            text = extract_ocr(image)

            if len(text) < 3:
                continue

            rows.append({

                "text": text,

                "label": folder_label,

                "source": source_name,

                "image_name": file,

                "website": "unknown",

                "pattern_type": folder_label,

                "comment": "OCR from labelled synthetic screenshot"

            })

            added += 1

    print(
        f"{source_name} text rows added:",
        added
    )

    return added


synthetic_rows = process_folder_dataset(
    "synthetic",
    "synthetic"
)

synthetic2_rows = process_folder_dataset(
    "synthetic2",
    "synthetic2"
)


# ============================================================
# SOURCE 4
# EXISTING METADATA DATASET
# ============================================================

print("\n" + "=" * 75)
print("SOURCE: EXISTING CLEANED METADATA")
print("=" * 75)

METADATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "metadata",
    "cleaned_dataset.csv"
)

metadata_rows = 0

if os.path.exists(METADATA_FILE):

    metadata_df = pd.read_csv(
        METADATA_FILE
    )

    print(
        "Metadata rows found:",
        len(metadata_df)
    )

    for _, record in metadata_df.iterrows():

        label = normalize_label(
            record.get("category")
        )

        # Do not trust blank or mixed categories
        if label not in DARK_LABELS:
            continue

        text = record.get(
            "cleaned_text",
            ""
        )

        text = clean_text(text)

        if len(text) < 3:
            continue

        image_name = str(
            record.get(
                "image_name",
                ""
            )
        )

        website = str(
            record.get(
                "website",
                "unknown"
            )
        )

        rows.append({

            "text": text,

            "label": label,

            "source": "metadata",

            "image_name": image_name,

            "website": website,

            "pattern_type": label,

            "comment": "Existing labelled metadata text"

        })

        metadata_rows += 1

else:

    print(
        "Metadata file not found. Skipped."
    )


print(
    "\nMetadata rows added:",
    metadata_rows
)


# ============================================================
# SOURCE 5
# NORMAL DATA FROM METADATA
# ============================================================
#
# IMPORTANT:
# Blank category is NOT automatically normal.
#
# Only rows explicitly marked as normal are used.
# ============================================================

print("\n" + "=" * 75)
print("SOURCE: EXPLICIT NORMAL TEXT")
print("=" * 75)

normal_rows = 0

if os.path.exists(METADATA_FILE):

    for _, record in metadata_df.iterrows():

        category = str(
            record.get(
                "category",
                ""
            )
        ).strip().lower()

        if category != "normal":
            continue

        text = clean_text(
            record.get(
                "cleaned_text",
                ""
            )
        )

        if len(text) < 3:
            continue

        rows.append({

            "text": text,

            "label": "normal",

            "source": "metadata",

            "image_name": str(
                record.get(
                    "image_name",
                    ""
                )
            ),

            "website": str(
                record.get(
                    "website",
                    "unknown"
                )
            ),

            "pattern_type": "normal",

            "comment": "Explicitly labelled normal text"

        })

        normal_rows += 1


print(
    "\nExplicit normal rows added:",
    normal_rows
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

print("\n" + "=" * 75)
print("BUILDING MASTER DATAFRAME")
print("=" * 75)

df = pd.DataFrame(
    rows,
    columns=[
        "text",
        "label",
        "source",
        "image_name",
        "website",
        "pattern_type",
        "comment"
    ]
)

if len(df) == 0:

    raise RuntimeError(
        "No usable text samples were created."
    )


# ============================================================
# CLEAN
# ============================================================

df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["label"] = (
    df["label"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[
    df["text"].str.len() >= 3
].copy()

df = df[
    df["label"].isin(VALID_LABELS)
].copy()


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["text", "label"]
).reset_index(drop=True)

duplicates_removed = (
    before_duplicates - len(df)
)


# ============================================================
# SHUFFLE
# ============================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# FINAL SAVE
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 75)
print("MASTER DATASET CREATED SUCCESSFULLY")
print("=" * 75)

print("\nOutput:")
print(OUTPUT_CSV)

print("\nTotal usable samples:", len(df))

print(
    "Exact duplicate text+label rows removed:",
    duplicates_removed
)

print("\nClass distribution:")
print(
    df["label"]
    .value_counts()
    .reindex(VALID_LABELS, fill_value=0)
)

print("\nSource distribution:")
print(
    df["source"]
    .value_counts()
)

print("\nColumns:")
print(
    list(df.columns)
)

print("\nFirst 10 samples:")
print(
    df.head(10).to_string(index=False)
)

print("\n" + "=" * 75)
print("IMPORTANT")
print("=" * 75)

print(
    "\nExisting data was NOT deleted or modified."
)

print(
    "\nUnlabelled/mixed categories were NOT automatically "
    "converted into normal."
)

print(
    "\nFinal labels are only:"
)

for label in VALID_LABELS:
    print(" -", label)

print("\n" + "=" * 75)