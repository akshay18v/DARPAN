import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "normal_text.csv"
)

df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("NORMAL TEXT DATASET CHECK")
print("=" * 60)

print("\nTotal normal rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing text:")
print(
    df["extracted_text"]
    .isna()
    .sum()
)

print("\nDuplicate text:")
print(
    df["extracted_text"]
    .duplicated()
    .sum()
)

print("\nText length statistics:")
print(
    df["extracted_text"]
    .str.len()
    .describe()
)

print("\nShortest examples:")
print(
    df.sort_values(
        by="extracted_text",
        key=lambda x: x.str.len()
    )[
        ["image_name", "extracted_text"]
    ].head(20).to_string(index=False)
)

print("\n" + "=" * 60)