import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

INPUT_CSV = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "final_text_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "distilbert"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("DARPAN - DISTILBERT DATASET SPLIT")
print("=" * 70)

df = pd.read_csv(INPUT_CSV)

print("\nTotal samples:", len(df))

# ============================================================
# REMOVE EMPTY TEXT
# ============================================================

df["text"] = df["text"].fillna("").astype(str)

df = df[
    df["text"].str.strip() != ""
].copy()

# ============================================================
# FIRST SPLIT
# 80% TRAIN
# 20% TEMPORARY
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# ============================================================
# SECOND SPLIT
# TEMPORARY → 10% VALIDATION + 10% TEST
# ============================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

# ============================================================
# RESET INDEX
# ============================================================

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# ============================================================
# SAVE
# ============================================================

train_df.to_csv(
    os.path.join(OUTPUT_DIR, "train.csv"),
    index=False,
    encoding="utf-8-sig"
)

val_df.to_csv(
    os.path.join(OUTPUT_DIR, "validation.csv"),
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    os.path.join(OUTPUT_DIR, "test.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("DATASET SPLIT COMPLETED")
print("=" * 70)

print("\nTrain samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))

print("\nTRAIN CLASS DISTRIBUTION")
print(train_df["label"].value_counts())

print("\nVALIDATION CLASS DISTRIBUTION")
print(val_df["label"].value_counts())

print("\nTEST CLASS DISTRIBUTION")
print(test_df["label"].value_counts())

print("\nOutput directory:")
print(OUTPUT_DIR)

print("=" * 70)