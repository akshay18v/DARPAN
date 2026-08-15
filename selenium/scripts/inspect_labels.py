import pandas as pd

path = "../../data/metadata/dataset.csv"

df = pd.read_csv(path, engine="python")

print("CATEGORY COUNTS:")
print(df["category"].value_counts(dropna=False))

print("\nROWS WITH MISSING CATEGORY:")
missing = df[df["category"].isna()]

print("Count:", len(missing))
print(missing[["image_name", "text"]].head(20).to_string(index=False))

print("\nSEVERITY COUNTS:")
print(df["severity"].value_counts(dropna=False))