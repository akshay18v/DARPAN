import os
import re
import html
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "../../"

TRAIN_INPUT = os.path.join(
    BASE_DIR, "data", "metadata", "train_v4.csv"
)

VAL_INPUT = os.path.join(
    BASE_DIR, "data", "metadata", "validation_v4.csv"
)

TRAIN_OUTPUT = os.path.join(
    BASE_DIR, "data", "metadata", "train_v5.csv"
)

VAL_OUTPUT = os.path.join(
    BASE_DIR, "data", "metadata", "validation_v5.csv"
)


# ============================================================
# OCR CLEANING
# ============================================================

def clean_ocr_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # --------------------------------------------------------
    # HTML entities
    # --------------------------------------------------------

    text = html.unescape(text)

    # --------------------------------------------------------
    # Common UTF-8 / OCR encoding corruption
    # --------------------------------------------------------

    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "â€¦": "...",
        "Â°": "°",
        "Â₹": "₹",
        "Â": "",
        "Ã©": "é",
        "Ã¨": "è",
        "Ã": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # --------------------------------------------------------
    # Remove NULL / control characters
    # --------------------------------------------------------

    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", text)

    # --------------------------------------------------------
    # Normalize common separators
    # --------------------------------------------------------

    text = text.replace("::", " ")
    text = text.replace("â€¢", " ")
    text = text.replace("•", " ")

    # --------------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------------

    text = re.sub(r"\s+", " ", text)

    # --------------------------------------------------------
    # Remove spaces around punctuation
    # --------------------------------------------------------

    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # --------------------------------------------------------
    # Remove repeated punctuation
    # --------------------------------------------------------

    text = re.sub(r"([!?.,])\1{2,}", r"\1", text)

    # --------------------------------------------------------
    # Strip
    # --------------------------------------------------------

    text = text.strip()

    return text


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("DARPAN - CLEANING OCR DATASET V5")
print("=" * 70)

print("\nLoading V4 datasets...")

train_df = pd.read_csv(TRAIN_INPUT)
val_df = pd.read_csv(VAL_INPUT)

print(f"Training rows   : {len(train_df)}")
print(f"Validation rows : {len(val_df)}")


# ============================================================
# CHECK TEXT COLUMN
# ============================================================

if "text" not in train_df.columns:
    raise ValueError(
        "ERROR: train_v4.csv does not contain a 'text' column."
    )

if "text" not in val_df.columns:
    raise ValueError(
        "ERROR: validation_v4.csv does not contain a 'text' column."
    )


# ============================================================
# BACKUP ORIGINAL TEXT
# ============================================================

train_df["original_text"] = train_df["text"]
val_df["original_text"] = val_df["text"]


# ============================================================
# CLEAN
# ============================================================

print("\nCleaning OCR text...")

train_df["text"] = train_df["text"].apply(clean_ocr_text)
val_df["text"] = val_df["text"].apply(clean_ocr_text)


# ============================================================
# REMOVE EMPTY TEXT
# ============================================================

before_train = len(train_df)
before_val = len(val_df)

train_df = train_df[
    train_df["text"].str.len() > 0
].copy()

val_df = val_df[
    val_df["text"].str.len() > 0
].copy()

print(
    f"\nRemoved empty training rows   : "
    f"{before_train - len(train_df)}"
)

print(
    f"Removed empty validation rows : "
    f"{before_val - len(val_df)}"
)


# ============================================================
# SAVE
# ============================================================

train_df.to_csv(
    TRAIN_OUTPUT,
    index=False
)

val_df.to_csv(
    VAL_OUTPUT,
    index=False
)


# ============================================================
# SAMPLE COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("OCR CLEANING EXAMPLES")
print("=" * 70)

sample_count = min(10, len(train_df))

for i in range(sample_count):

    original = train_df.iloc[i]["original_text"]
    cleaned = train_df.iloc[i]["text"]

    print("\n------------------------------------------------------------")
    print("BEFORE:")
    print(str(original)[:500])

    print("\nAFTER:")
    print(str(cleaned)[:500])


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("V5 DATASET CREATED")
print("=" * 70)

print(f"\nTraining:")
print(TRAIN_OUTPUT)

print(f"\nValidation:")
print(VAL_OUTPUT)

print("\nRows:")
print(f"Training   : {len(train_df)}")
print(f"Validation : {len(val_df)}")

print("\nDone.")