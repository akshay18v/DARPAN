import os
import re
import random

import cv2
import pandas as pd
import easyocr

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "../../"

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "yolo_dataset",
    "images"
)

LABEL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "yolo_dataset",
    "labels"
)

CLASSES_FILE = os.path.join(
    BASE_DIR,
    "data",
    "annotations",
    "classes.txt"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "metadata"
)

ALL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "nlp_dataset_v5.csv"
)

TRAIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "train_nlp_v5.csv"
)

VAL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "validation_nlp_v5.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

VAL_SIZE = 0.20

MIN_TEXT_LENGTH = 3

VALID_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp"
}


# ============================================================
# EXPECTED DARPAN CLASSES
# ============================================================

EXPECTED_CLASSES = [
    "hidden_fees",
    "false_urgency",
    "sneaking",
    "misdirection",
    "interface_interference",
    "obstruction"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("DARPAN - BUILDING CLEAN NLP DATASET V5")
print("=" * 70)

print()
print("Pipeline:")
print("YOLO annotation -> crop -> EasyOCR -> cleaning -> NLP CSV")
print()


# ============================================================
# CHECK DIRECTORIES
# ============================================================

if not os.path.isdir(IMAGE_DIR):
    raise FileNotFoundError(
        f"Image directory not found:\n{IMAGE_DIR}"
    )

if not os.path.isdir(LABEL_DIR):
    raise FileNotFoundError(
        f"Label directory not found:\n{LABEL_DIR}"
    )

if not os.path.isfile(CLASSES_FILE):
    raise FileNotFoundError(
        f"classes.txt not found:\n{CLASSES_FILE}"
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD CLASSES
# ============================================================

with open(
    CLASSES_FILE,
    "r",
    encoding="utf-8"
) as f:

    classes = [
        line.strip()
        for line in f.readlines()
        if line.strip()
    ]


print("Classes found:")

for class_id, class_name in enumerate(classes):
    print(
        f"{class_id} -> {class_name}"
    )


if len(classes) != 6:
    raise ValueError(
        f"Expected 6 classes but found {len(classes)}"
    )


for expected_class in EXPECTED_CLASSES:

    if expected_class not in classes:

        raise ValueError(
            f"Missing expected class: "
            f"{expected_class}"
        )


ID2LABEL = {
    i: label
    for i, label in enumerate(classes)
}


# ============================================================
# INITIALIZE EASYOCR
# ============================================================

print()
print("=" * 70)
print("INITIALIZING EASYOCR")
print("=" * 70)

print()
print("Loading EasyOCR English model...")
print("First initialization may take some time.")

reader = easyocr.Reader(
    ["en"],
    gpu=False
)

print()
print("EasyOCR initialized successfully.")


# ============================================================
# OCR CLEANING
# ============================================================

def clean_ocr_text(text):

    if text is None:
        return ""

    text = str(text)

    # Remove null characters
    text = text.replace(
        "\x00",
        " "
    )

    # Common OCR encoding corrections
    replacements = {

        "â‚¹": "₹",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "Â": "",
        "Ã": "",

        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',

        "|": " ",
        "_": " "
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    # Remove unwanted characters
    text = re.sub(
        r"[^\w\s₹$€£%.,:;!?+\-()/&@#']",
        " ",
        text,
        flags=re.UNICODE
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# OCR QUALITY CHECK
# ============================================================

def is_usable_text(text):

    if not text:
        return False

    text = text.strip()

    if len(text) < MIN_TEXT_LENGTH:
        return False

    # Remove spaces and punctuation
    meaningful_chars = re.sub(
        r"[\s\W_]",
        "",
        text,
        flags=re.UNICODE
    )

    if len(meaningful_chars) < MIN_TEXT_LENGTH:
        return False

    return True


# ============================================================
# OCR FUNCTION
# ============================================================

def extract_text_from_crop(crop):

    if crop is None:
        return ""

    if crop.size == 0:
        return ""

    # --------------------------------------------------------
    # Upscale crop
    # --------------------------------------------------------

    crop = cv2.resize(
        crop,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # EasyOCR
    # --------------------------------------------------------

    results = reader.readtext(
        crop,
        detail=1,
        paragraph=False
    )

    if not results:
        return ""

    extracted_text = []

    for result in results:

        if len(result) < 2:
            continue

        text = result[1]

        if not text:
            continue

        extracted_text.append(
            text
        )

    if not extracted_text:
        return ""

    # Join OCR detections
    combined_text = " ".join(
        extracted_text
    )

    # Clean
    combined_text = clean_ocr_text(
        combined_text
    )

    return combined_text


# ============================================================
# YOLO -> PIXEL COORDINATES
# ============================================================

def yolo_to_pixel_box(
    x_center,
    y_center,
    width,
    height,
    image_width,
    image_height
):

    x1 = int(
        (x_center - width / 2)
        * image_width
    )

    y1 = int(
        (y_center - height / 2)
        * image_height
    )

    x2 = int(
        (x_center + width / 2)
        * image_width
    )

    y2 = int(
        (y_center + height / 2)
        * image_height
    )

    # Clamp coordinates
    x1 = max(
        0,
        min(
            x1,
            image_width - 1
        )
    )

    y1 = max(
        0,
        min(
            y1,
            image_height - 1
        )
    )

    x2 = max(
        1,
        min(
            x2,
            image_width
        )
    )

    y2 = max(
        1,
        min(
            y2,
            image_height
        )
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# FIND IMAGES
# ============================================================

image_files = []

for filename in os.listdir(
    IMAGE_DIR
):

    extension = os.path.splitext(
        filename
    )[1].lower()

    if extension in VALID_IMAGE_EXTENSIONS:

        image_files.append(
            filename
        )


image_files.sort()

print()
print(
    f"Images found: {len(image_files)}"
)


# ============================================================
# PROCESS DATA
# ============================================================

records = []

missing_labels = []

invalid_annotations = []

processed_images = 0

processed_boxes = 0

ocr_success = 0

ocr_failed = 0


print()
print("=" * 70)
print("PROCESSING ANNOTATIONS")
print("=" * 70)

print()


for image_index, image_name in enumerate(
    image_files,
    start=1
):

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    stem = os.path.splitext(
        image_name
    )[0]

    label_path = os.path.join(
        LABEL_DIR,
        stem + ".txt"
    )

    # --------------------------------------------------------
    # Missing label
    # --------------------------------------------------------

    if not os.path.exists(
        label_path
    ):

        missing_labels.append(
            image_name
        )

        continue


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            f"WARNING: Could not read {image_name}"
        )

        continue


    image_height, image_width = image.shape[:2]

    processed_images += 1


    # --------------------------------------------------------
    # Read YOLO annotation
    # --------------------------------------------------------

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]


    # --------------------------------------------------------
    # Process every YOLO box
    # --------------------------------------------------------

    for annotation_index, line in enumerate(
        lines
    ):

        parts = line.split()


        # ----------------------------------------------------
        # Validate YOLO line
        # ----------------------------------------------------

        if len(parts) != 5:

            invalid_annotations.append({

                "image_name": image_name,

                "annotation": line,

                "reason":
                    "Expected 5 YOLO values"

            })

            continue


        try:

            class_id = int(
                parts[0]
            )

            x_center = float(
                parts[1]
            )

            y_center = float(
                parts[2]
            )

            width = float(
                parts[3]
            )

            height = float(
                parts[4]
            )

        except ValueError:

            invalid_annotations.append({

                "image_name": image_name,

                "annotation": line,

                "reason":
                    "Invalid numeric value"

            })

            continue


        # ----------------------------------------------------
        # Validate class
        # ----------------------------------------------------

        if class_id not in ID2LABEL:

            invalid_annotations.append({

                "image_name": image_name,

                "annotation": line,

                "reason":
                    "Invalid class ID"

            })

            continue


        # ----------------------------------------------------
        # Validate YOLO coordinates
        # ----------------------------------------------------

        if not (
            0 <= x_center <= 1
            and
            0 <= y_center <= 1
            and
            0 < width <= 1
            and
            0 < height <= 1
        ):

            invalid_annotations.append({

                "image_name": image_name,

                "annotation": line,

                "reason":
                    "Invalid YOLO coordinates"

            })

            continue


        # ----------------------------------------------------
        # Convert YOLO box to pixels
        # ----------------------------------------------------

        x1, y1, x2, y2 = yolo_to_pixel_box(

            x_center,
            y_center,
            width,
            height,

            image_width,
            image_height

        )


        # ----------------------------------------------------
        # Crop image
        # ----------------------------------------------------

        crop = image[
            y1:y2,
            x1:x2
        ]


        if crop.size == 0:

            invalid_annotations.append({

                "image_name": image_name,

                "annotation": line,

                "reason":
                    "Empty crop"

            })

            continue


        processed_boxes += 1


        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        text = extract_text_from_crop(
            crop
        )


        # ----------------------------------------------------
        # Skip unusable OCR
        # ----------------------------------------------------

        if not is_usable_text(
            text
        ):

            ocr_failed += 1

            continue


        ocr_success += 1


        # ----------------------------------------------------
        # Pattern information
        # ----------------------------------------------------

        category = ID2LABEL[
            class_id
        ]


        # ----------------------------------------------------
        # Add clean NLP record
        # ----------------------------------------------------

        records.append({

            "text": text,

            "pattern_category":
                category,

            "pattern_type":
                category,

            "deceptive":
                "Yes",

            "source_image":
                image_name

        })


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        image_index % 25 == 0
        or
        image_index == len(image_files)
    ):

        print(
            f"Processed images: "
            f"{image_index}/{len(image_files)}"
        )


# ============================================================
# CHECK RESULTS
# ============================================================

print()
print("=" * 70)
print("OCR EXTRACTION RESULTS")
print("=" * 70)

print()
print(
    f"Images processed : {processed_images}"
)

print(
    f"YOLO boxes       : {processed_boxes}"
)

print(
    f"OCR success      : {ocr_success}"
)

print(
    f"OCR failed       : {ocr_failed}"
)

print(
    f"Missing labels   : {len(missing_labels)}"
)

print(
    f"Invalid boxes    : {len(invalid_annotations)}"
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    records
)


if df.empty:

    raise RuntimeError(
        "No usable OCR text was extracted."
    )


# ============================================================
# REMOVE DUPLICATES
# ============================================================

before_duplicates = len(
    df
)


df = df.drop_duplicates(
    subset=[
        "text",
        "pattern_category",
        "source_image"
    ]
).copy()


removed_duplicates = (
    before_duplicates
    - len(df)
)


print()
print(
    f"Duplicate rows removed: "
    f"{removed_duplicates}"
)


# ============================================================
# SHUFFLE DATA
# ============================================================

df = df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(
    drop=True
)


# ============================================================
# SAVE COMPLETE NLP DATASET
# ============================================================

df.to_csv(
    ALL_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# IMAGE-LEVEL TRAIN / VALIDATION SPLIT
# ============================================================

unique_images = (
    df[
        "source_image"
    ]
    .drop_duplicates()
    .tolist()
)


random.seed(
    RANDOM_SEED
)

random.shuffle(
    unique_images
)


split_index = int(
    len(unique_images)
    * (1 - VAL_SIZE)
)


train_images = set(
    unique_images[
        :split_index
    ]
)

val_images = set(
    unique_images[
        split_index:
    ]
)


train_df = df[
    df[
        "source_image"
    ].isin(
        train_images
    )
].copy()


val_df = df[
    df[
        "source_image"
    ].isin(
        val_images
    )
].copy()


# ============================================================
# CHECK IMAGE LEAKAGE
# ============================================================

overlap = (
    train_images
    &
    val_images
)


if overlap:

    raise RuntimeError(
        "IMAGE LEAKAGE DETECTED!"
    )


# ============================================================
# SAVE TRAIN / VALIDATION
# ============================================================

train_df.to_csv(
    TRAIN_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


val_df.to_csv(
    VAL_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL NLP DATASET")
print("=" * 70)

print()
print(
    f"Total NLP rows : {len(df)}"
)

print(
    f"Training rows  : {len(train_df)}"
)

print(
    f"Validation rows: {len(val_df)}"
)

print(
    f"Training images: {len(train_images)}"
)

print(
    f"Validation images: {len(val_images)}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("ALL CLASS DISTRIBUTION")
print("=" * 70)

print(
    df[
        "pattern_category"
    ].value_counts()
)


print()
print("=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

print(
    train_df[
        "pattern_category"
    ].value_counts()
)


print()
print("=" * 70)
print("VALIDATION CLASS DISTRIBUTION")
print("=" * 70)

print(
    val_df[
        "pattern_category"
    ].value_counts()
)


# ============================================================
# SAMPLE DATA
# ============================================================

print()
print("=" * 70)
print("SAMPLE NLP DATA")
print("=" * 70)

print()

print(
    df.head(15).to_string(
        index=False
    )
)


# ============================================================
# OUTPUT FILES
# ============================================================

print()
print("=" * 70)
print("DATASET FILES CREATED")
print("=" * 70)

print()
print(
    f"Complete dataset:\n{ALL_OUTPUT}"
)

print()
print(
    f"Training dataset:\n{TRAIN_OUTPUT}"
)

print()
print(
    f"Validation dataset:\n{VAL_OUTPUT}"
)


# ============================================================
# SAFETY CONFIRMATION
# ============================================================

print()
print("=" * 70)
print("SOURCE DATA SAFETY")
print("=" * 70)

print()
print(
    "Original YOLO images: READ ONLY"
)

print(
    "Original YOLO labels : READ ONLY"
)

print(
    "Original classes.txt : READ ONLY"
)

print(
    "No crops were saved."
)

print(
    "No YOLO files were modified."
)

print(
    "Only CSV files were created in data/metadata."
)

print()
print("=" * 70)
print("DARPAN NLP DATASET V5 COMPLETE")
print("=" * 70)