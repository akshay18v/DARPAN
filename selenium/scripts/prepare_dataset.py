import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# PATH
# ============================================================

CSV_PATH = "../../data/metadata/dataset.csv"

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(CSV_PATH, engine="python")

print("=" * 60)
print("DARPAN - DISTILBERT DATASET PREPARATION")
print("=" * 60)

print(f"\nOriginal dataset size: {len(df)}")

# ============================================================
# KEEP ONLY LABELED SAMPLES
# ============================================================

df = df.dropna(subset=["category"]).copy()

print(f"Labeled samples: {len(df)}")

# ============================================================
# NORMALIZE CATEGORY NAMES
# ============================================================

df["category"] = df["category"].astype(str).str.strip()

# Fix inconsistent capitalization
df["category"] = df["category"].replace({
    "Interface interference": "interface interference"
})

# ============================================================
# REMOVE ROWS WITH EMPTY OCR TEXT
# ============================================================

df["text"] = df["text"].fillna("").astype(str).str.strip()

before_text_filter = len(df)

df = df[df["text"] != ""].copy()

print(f"Removed empty-text rows: {before_text_filter - len(df)}")
print(f"Final usable samples: {len(df)}")

# ============================================================
# DISPLAY FINAL CLASS DISTRIBUTION
# ============================================================

print("\nFinal class distribution:")
print(df["category"].value_counts())

# ============================================================
# LABEL ENCODING
# ============================================================

labels = sorted(df["category"].unique())

label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

df["label"] = df["category"].map(label2id)

print("\nLabel mapping:")

for label, idx in label2id.items():
    print(f"{idx} -> {label}")

# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First split:
# 80% training
# 20% temporary set

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Split temporary set equally:
# 10% validation
# 10% test

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

print("\nDataset split:")
print(f"Training   : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")

# ============================================================
# SAVE PREPARED DATASETS
# ============================================================

OUTPUT_DIR = "../../data/metadata"

train_df.to_csv(
    f"{OUTPUT_DIR}/train.csv",
    index=False
)

val_df.to_csv(
    f"{OUTPUT_DIR}/validation.csv",
    index=False
)

test_df.to_csv(
    f"{OUTPUT_DIR}/test.csv",
    index=False
)

# Save label mapping
label_mapping = pd.DataFrame({
    "label_id": list(id2label.keys()),
    "category": list(id2label.values())
})

label_mapping.to_csv(
    f"{OUTPUT_DIR}/label_mapping.csv",
    index=False
)

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\nFiles created:")
print("  data/metadata/train.csv")
print("  data/metadata/validation.csv")
print("  data/metadata/test.csv")
print("  data/metadata/label_mapping.csv")

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)