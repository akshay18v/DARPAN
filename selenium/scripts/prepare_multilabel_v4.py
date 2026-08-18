import os
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "../../"

TRAIN_V3 = os.path.join(
    BASE_DIR, "data", "metadata", "train_v3.csv"
)

VAL_V3 = os.path.join(
    BASE_DIR, "data", "metadata", "validation_v3.csv"
)

UNIFIED = os.path.join(
    BASE_DIR, "data", "metadata", "unified_predictions_v1.csv"
)

TRAIN_V4 = os.path.join(
    BASE_DIR, "data", "metadata", "train_multilabel_v4.csv"
)

VAL_V4 = os.path.join(
    BASE_DIR, "data", "metadata", "validation_multilabel_v4.csv"
)


LABELS = [
    "Sneaking",
    "hidden fees",
    "interface interference",
    "misdirection",
    "obstruction",
    "potential false urgency",
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("DARPAN - PREPARING MULTI-LABEL DATASET V4")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train = pd.read_csv(TRAIN_V3)
val = pd.read_csv(VAL_V3)
unified = pd.read_csv(UNIFIED)

print(f"V3 training samples   : {len(train)}")
print(f"V3 validation samples : {len(val)}")
print(f"Unified predictions   : {len(unified)}")


# ============================================================
# NORMALIZE
# ============================================================

for df in [train, val]:
    df["image_name"] = df["image_name"].astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str)
    df["category"] = df["category"].astype(str).str.strip()


unified["image_name"] = (
    unified["image_name"]
    .astype(str)
    .str.strip()
)

unified["predicted_category"] = (
    unified["predicted_category"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# GET POTENTIAL FALSE URGENCY
# ============================================================

pfu = unified[
    unified["potential_false_urgency"]
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
].copy()

print("\nPotential false urgency samples found:")
print(len(pfu))


# ============================================================
# CREATE LOOKUP
# ============================================================

pfu_names = set(pfu["image_name"])

# Map image -> predicted category
pfu_category = dict(
    zip(
        pfu["image_name"],
        pfu["predicted_category"]
    )
)


# ============================================================
# FIND PFU SAMPLES IN V3 DATA
# ============================================================

train_pfu_mask = train["image_name"].isin(pfu_names)
val_pfu_mask = val["image_name"].isin(pfu_names)

train_pfu = train[train_pfu_mask].copy()
existing_val_pfu = val[val_pfu_mask].copy()

print("\nPotential false urgency distribution:")
print(
    f"Available in V3 training   : {len(train_pfu)}"
)
print(
    f"Already in V3 validation    : {len(existing_val_pfu)}"
)


# ============================================================
# IMPORTANT:
# MOVE SOME PFU SAMPLES FROM TRAIN → VALIDATION
#
# This ensures validation actually contains positive
# potential-false-urgency examples.
# ============================================================

rng = np.random.default_rng(42)

# Target approximately 30% of available PFU training samples
validation_count = max(
    1,
    int(round(len(train_pfu) * 0.30))
)

# Do not move more than available
validation_count = min(
    validation_count,
    len(train_pfu)
)

if validation_count > 0:

    validation_pfu_indices = rng.choice(
        train_pfu.index.to_numpy(),
        size=validation_count,
        replace=False
    )

else:
    validation_pfu_indices = []


validation_pfu = train.loc[
    validation_pfu_indices
].copy()

train = train.drop(
    validation_pfu_indices
)


# ============================================================
# COMBINE WITH EXISTING VALIDATION
# ============================================================

val = pd.concat(
    [
        val,
        validation_pfu
    ],
    ignore_index=True
)


# Remove duplicate image names
train = train.drop_duplicates(
    subset=["image_name"]
)

val = val.drop_duplicates(
    subset=["image_name"]
)


# ============================================================
# CREATE SIX LABEL COLUMNS
# ============================================================

def create_multilabel_columns(df):

    df = df.copy()

    # --------------------------------------------------------
    # Five actual dark-pattern labels
    # --------------------------------------------------------

    for label in LABELS[:5]:

        df[label] = (
            df["category"]
            .str.lower()
            .eq(label.lower())
            .astype(int)
        )

    # --------------------------------------------------------
    # Potential false urgency
    #
    # IMPORTANT:
    # This is independent of the dark-pattern category.
    #
    # Example:
    #
    # obstruction = 1
    # potential false urgency = 1
    #
    # --------------------------------------------------------

    df["potential false urgency"] = (
        df["image_name"]
        .isin(pfu_names)
        .astype(int)
    )

    return df


train = create_multilabel_columns(train)
val = create_multilabel_columns(val)


# ============================================================
# CHECK FOR DATA LEAKAGE
# ============================================================

overlap = set(train["image_name"]) & set(val["image_name"])

if overlap:

    print("\nWARNING: Duplicate images found between train/validation.")
    print(
        f"Removing {len(overlap)} duplicated validation images..."
    )

    val = val[
        ~val["image_name"].isin(overlap)
    ].copy()


# ============================================================
# PRINT DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("MULTI-LABEL DISTRIBUTION")
print("=" * 70)

print("\nTRAINING:")

for label in LABELS:

    print(
        f"{label:<30}: "
        f"{train[label].sum()}"
    )


print("\nVALIDATION:")

for label in LABELS:

    print(
        f"{label:<30}: "
        f"{val[label].sum()}"
    )


# ============================================================
# SANITY CHECK
# ============================================================

print("\n")
print("=" * 70)
print("MULTI-LABEL SANITY CHECK")
print("=" * 70)

# Every sample should have exactly one of the five
# dark-pattern labels.

train_dark_count = train[LABELS[:5]].sum(axis=1)

val_dark_count = val[LABELS[:5]].sum(axis=1)

print(
    "\nTraining rows with invalid dark-pattern count:",
    (train_dark_count != 1).sum()
)

print(
    "Validation rows with invalid dark-pattern count:",
    (val_dark_count != 1).sum()
)


# ============================================================
# SHOW PFU EXAMPLES
# ============================================================

print("\n")
print("=" * 70)
print("POTENTIAL FALSE URGENCY EXAMPLES")
print("=" * 70)

pfu_train = train[
    train["potential false urgency"] == 1
]

pfu_val = val[
    val["potential false urgency"] == 1
]

print("\nTRAINING PFU:")
print(
    pfu_train[
        [
            "image_name",
            "category",
            "potential false urgency"
        ]
    ].to_string(index=False)
)

print("\nVALIDATION PFU:")
print(
    pfu_val[
        [
            "image_name",
            "category",
            "potential false urgency"
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

output_columns = [
    "image_name",
    "image_path",
    "website",
    "source",
    "page",
    "text",
    "category",
    "severity",
] + LABELS


# Keep only columns that exist
train_columns = [
    c for c in output_columns
    if c in train.columns
]

val_columns = [
    c for c in output_columns
    if c in val.columns
]


train[
    train_columns
].to_csv(
    TRAIN_V4,
    index=False
)

val[
    val_columns
].to_csv(
    VAL_V4,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("MULTI-LABEL V4 DATASET CREATED")
print("=" * 70)

print("\nTraining:")
print(TRAIN_V4)

print("\nValidation:")
print(VAL_V4)

print("\nTraining samples :", len(train))
print("Validation samples:", len(val))

print("\nPotential false urgency:")
print(
    f"Train      : {train['potential false urgency'].sum()}"
)

print(
    f"Validation : {val['potential false urgency'].sum()}"
)

print("\nREADY FOR MULTI-LABEL DISTILBERT TRAINING")
print("=" * 70)