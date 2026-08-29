import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "final_text_dataset.csv"
)

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("DARPAN - FINAL DATASET QUALITY CHECK")
print("=" * 70)

print("\nTotal samples:", len(df))

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nMissing text:")
print(df["text"].isna().sum())

print("\nEmpty text:")
print((df["text"].str.strip() == "").sum())

# ------------------------------------------------------------
# TEXT LENGTH
# ------------------------------------------------------------

df["text_length"] = df["text"].str.len()

print("\nText length statistics:")
print(df["text_length"].describe())

# ------------------------------------------------------------
# VERY SHORT TEXT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VERY SHORT TEXT (< 10 characters)")
print("=" * 70)

short_df = df[df["text_length"] < 10]

print("Count:", len(short_df))

print(
    short_df[
        ["text", "label"]
    ].head(50).to_string(index=False)
)

# ------------------------------------------------------------
# SUSPICIOUS OCR TEXT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE FROM EACH CLASS")
print("=" * 70)

for label in df["label"].unique():

    print("\n---", label, "---")

    sample = df[
        df["label"] == label
    ].sample(
        min(10, len(df[df["label"] == label])),
        random_state=42
    )

    for text in sample["text"]:
        print("-", text[:250])

print("\n" + "=" * 70)