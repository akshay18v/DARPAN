import os
import pandas as pd

# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "text_regions_cleaned.csv"
)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("DARPAN - DATASET CHECK")
print("=" * 60)

print("\nTotal rows:", len(df))

# ==========================================
# CATEGORY COUNT
# ==========================================

print("\nCategory distribution:")
print("-" * 40)

counts = df["pattern_category"].value_counts()

print(counts)

# ==========================================
# PERCENTAGE
# ==========================================

print("\nCategory percentage:")
print("-" * 40)

percentage = (
    df["pattern_category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(percentage)

# ==========================================
# EMPTY TEXT CHECK
# ==========================================

print("\nEmpty text:")
print("-" * 40)

empty = df["cleaned_text"].isna().sum()

print(empty)

# ==========================================
# VERY SHORT TEXT
# ==========================================

print("\nVery short text:")
print("-" * 40)

short_text = df[
    df["cleaned_text"].str.len() < 5
]

print("Count:", len(short_text))

print("\nExamples:")

print(
    short_text[
        [
            "pattern_category",
            "cleaned_text"
        ]
    ].head(20).to_string(index=False)
)

print("\n" + "=" * 60)