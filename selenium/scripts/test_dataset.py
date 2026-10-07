import pandas as pd

path = "../../data/metadata/dataset.csv"

print("Trying Python CSV parser...\n")

df = pd.read_csv(
    path,
    engine="python",
    on_bad_lines="warn"
)

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nCategory distribution:")
print(df["category"].value_counts(dropna=False))

print("\nFirst 5 image names:")
print(df["image_name"].head().tolist())

print("\nFirst 5 text samples:")
print(df["text"].head().tolist())