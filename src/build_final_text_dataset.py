import os
import pandas as pd

# ============================================================
# DARPAN - FINAL TEXT DATASET BUILDER
# ============================================================
# Existing files are READ ONLY.
# Nothing inside data/raw, data/synthetic, data/synthetic2,
# annotations, or yolo_dataset is deleted or modified.
#
# Final labels:
# hidden_fees
# false_urgency
# sneaking
# misdirection
# interface_interference
# obstruction
# normal
#
# Final columns:
# text
# label
# source
# image_name
# website
# pattern_type
# comment
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

MASTER_FILE = os.path.join(
    PROCESSED_DIR,
    "darpan_master_text_dataset.csv"
)

RAW_FILE = os.path.join(
    PROCESSED_DIR,
    "raw_curated_text.csv"
)

NORMAL_FILE = os.path.join(
    PROCESSED_DIR,
    "normal_text_dataset.csv"
)

OUTPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "darpan_final_text_dataset.csv"
)

# ============================================================
# FINAL LABELS
# ============================================================

LABELS = [
    "hidden_fees",
    "false_urgency",
    "sneaking",
    "misdirection",
    "interface_interference",
    "obstruction",
    "normal"
]

# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("DARPAN - FINAL TEXT DATASET BUILDER")
print("=" * 75)

master = pd.read_csv(MASTER_FILE)
raw = pd.read_csv(RAW_FILE)
normal = pd.read_csv(NORMAL_FILE)

print("\nInput datasets:")
print("Master :", len(master))
print("Raw    :", len(raw))
print("Normal :", len(normal))

# ============================================================
# STANDARDIZE COLUMNS
# ============================================================

required_columns = [
    "text",
    "label",
    "source",
    "image_name",
    "website",
    "pattern_type",
    "comment"
]

def standardize(df):

    df = df.copy()

    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    df = df[required_columns]

    for col in required_columns:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return df


master = standardize(master)
raw = standardize(raw)
normal = standardize(normal)

# ============================================================
# COMBINE
# ============================================================

df = pd.concat(
    [
        master,
        raw,
        normal
    ],
    ignore_index=True
)

print("\nCombined before cleaning:", len(df))

# ============================================================
# NORMALIZE LABELS
# ============================================================

label_map = {
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

    "normal": "normal"
}

df["label"] = (
    df["label"]
    .str.lower()
    .map(label_map)
)

# ============================================================
# REMOVE INVALID LABELS
# ============================================================

before = len(df)

df = df[
    df["label"].isin(LABELS)
].copy()

invalid_removed = before - len(df)

print(
    "Invalid-label rows removed:",
    invalid_removed
)

# ============================================================
# CLEAN TEXT
# ============================================================

df["text"] = (
    df["text"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Remove empty / tiny samples
before = len(df)

df = df[
    df["text"].str.len() >= 4
].copy()

short_removed = before - len(df)

print(
    "Empty/tiny rows removed:",
    short_removed
)

# ============================================================
# PATTERN TYPE = LABEL
# ============================================================

df["pattern_type"] = df["label"]

# ============================================================
# REMOVE EXACT DUPLICATES
#
# Same text with same label = duplicate.
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=["text", "label"]
).reset_index(drop=True)

duplicates_removed = before - len(df)

print(
    "Exact text+label duplicates removed:",
    duplicates_removed
)

# ============================================================
# HANDLE CONFLICTING LABELS
#
# Same text labelled differently is unsafe for training.
# Remove conflicting texts completely instead of guessing.
# ============================================================

label_counts_per_text = (
    df.groupby("text")["label"]
    .nunique()
)

conflicting_texts = set(
    label_counts_per_text[
        label_counts_per_text > 1
    ].index
)

print(
    "Conflicting text strings:",
    len(conflicting_texts)
)

if conflicting_texts:

    df = df[
        ~df["text"].isin(conflicting_texts)
    ].copy()

# ============================================================
# FINAL CLASS COUNTS BEFORE BALANCING
# ============================================================

print("\nClass distribution BEFORE balancing:")

counts = (
    df["label"]
    .value_counts()
    .reindex(
        LABELS,
        fill_value=0
    )
)

print(counts)

# ============================================================
# BALANCING STRATEGY
#
# We DO NOT duplicate minority samples.
# We keep all available samples when possible.
#
# Target = maximum amount that every class can support,
# but capped at 200.
#
# Therefore:
#   - no class dominates the training data
#   - no artificial duplicate rows
#   - no oversampling by copying text
# ============================================================

MAX_PER_CLASS = 200

available_counts = (
    df["label"]
    .value_counts()
    .reindex(
        LABELS,
        fill_value=0
    )
)

target_count = min(
    MAX_PER_CLASS,
    int(available_counts.min())
)

print(
    "\nTarget samples per class:",
    target_count
)

balanced_parts = []

for label in LABELS:

    class_df = df[
        df["label"] == label
    ].copy()

    if len(class_df) == 0:
        continue

    # --------------------------------------------------------
    # Shuffle deterministically
    # --------------------------------------------------------

    class_df = class_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Downsample only when necessary
    # --------------------------------------------------------

    if len(class_df) > target_count:

        class_df = class_df.iloc[
            :target_count
        ].copy()

    balanced_parts.append(
        class_df
    )

# ============================================================
# CREATE FINAL DATASET
# ============================================================

final_df = pd.concat(
    balanced_parts,
    ignore_index=True
)

# Shuffle final dataset
final_df = final_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ============================================================
# SAVE
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 75)
print("FINAL DATASET CREATED")
print("=" * 75)

print(
    "\nTotal final samples:",
    len(final_df)
)

print(
    "\nFINAL CLASS DISTRIBUTION:"
)

print(
    final_df["label"]
    .value_counts()
    .reindex(
        LABELS,
        fill_value=0
    )
)

print(
    "\nFINAL COLUMNS:"
)

print(
    list(final_df.columns)
)

print(
    "\nSOURCE DISTRIBUTION:"
)

print(
    final_df["source"].value_counts()
)

print(
    "\nOUTPUT:"
)

print(
    OUTPUT_FILE
)

print("\n" + "=" * 75)
print("SAFETY CHECKS")
print("=" * 75)

print(
    "\nExisting source files were NOT deleted or modified."
)

print(
    "No conflicting text-label pairs remain."
)

print(
    "No class was artificially duplicated."
)

print(
    "Pattern type is identical to label."
)

print("\n" + "=" * 75)