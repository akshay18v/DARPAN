import pandas as pd
import re
import os

# ============================================================
# PATHS
# ============================================================

INPUT_DIR = "../../data/metadata"
OUTPUT_DIR = "../../data/metadata"

FILES = [
    "train.csv",
    "validation.csv",
    "test.csv"
]


# ============================================================
# OCR CLEANING FUNCTION
# ============================================================

def clean_ocr_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Fix common OCR encoding problems
    replacements = {
        "â€”": "-",
        "â€“": "-",
        "â€˜": "'",
        "â€™": "'",
        "â€œ": '"',
        "â€�": '"',
        "Â": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Replace newlines/tabs with spaces
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Keep letters, numbers and useful symbols
    text = re.sub(
        r"[^a-zA-Z0-9₹$%.,!?+:/()'&@#*_-]",
        " ",
        text
    )

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove repeated garbage symbols
    text = re.sub(r"([*@#])\1{2,}", r"\1", text)

    return text.strip()


# ============================================================
# PROCESS DATASETS
# ============================================================

for filename in FILES:

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(
        OUTPUT_DIR,
        filename.replace(".csv", "_clean.csv")
    )

    print("\nProcessing:", filename)

    df = pd.read_csv(input_path)

    print("Original rows:", len(df))

    # Clean OCR text
    df["text"] = df["text"].fillna("").apply(clean_ocr_text)

    # Remove completely empty text
    df = df[df["text"].str.strip() != ""]

    # Save
    df.to_csv(
        output_path,
        index=False
    )

    print("Saved:", output_path)
    print("Rows:", len(df))

    # Show sample
    print("\nSample cleaned text:")
    print(df["text"].iloc[0][:500])
    print("-" * 60)


print("\nOCR CLEANING COMPLETE")