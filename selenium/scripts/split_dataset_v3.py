import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "../../data/metadata/dataset.csv"

TRAIN = "../../data/metadata/train_v3.csv"
VAL = "../../data/metadata/validation_v3.csv"
TEST = "../../data/metadata/test_v3.csv"

df = pd.read_csv(INPUT)

# Keep only rows that have both text and a category
df = df[
    df["category"].notna() &
    df["text"].notna() &
    (df["text"].str.strip() != "")
].copy()

print("=" * 60)
print("DARPAN V3 DATASET SPLIT")
print("=" * 60)

print(f"\nUsable labeled samples: {len(df)}")

print("\nClass distribution:")
print(df["category"].value_counts())

# ---------------------------------------------------------
# 80% TRAIN / 10% VALIDATION / 10% FINAL TEST
# ---------------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["category"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["category"],
    random_state=42
)

# Save
train_df.to_csv(TRAIN, index=False)
val_df.to_csv(VAL, index=False)
test_df.to_csv(TEST, index=False)

print("\n" + "=" * 60)
print("FINAL SPLIT")
print("=" * 60)

print(f"Training   : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Final Test : {len(test_df)}")

print("\nTraining distribution:")
print(train_df["category"].value_counts())

print("\nValidation distribution:")
print(val_df["category"].value_counts())

print("\nFinal test distribution:")
print(test_df["category"].value_counts())

print("\nFiles created:")
print(TRAIN)
print(VAL)
print(TEST)