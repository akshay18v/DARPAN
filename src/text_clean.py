import pandas as pd
import re

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
df = pd.read_csv("data/metadata/dataset.csv")

print("=" * 50)
print("Dataset Loaded Successfully!")
print("=" * 50)

# --------------------------------------------------
# Text Cleaning Function
# --------------------------------------------------
def clean_text(text):

    # Handle missing values
    if pd.isna(text):
        return ""

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # --------------------------------------------------
    # Remove image/file names
    # Example: travel_2.png, hotel_0.jpg
    # --------------------------------------------------
    text = re.sub(r'\b\w+\.(png|jpg|jpeg)\b', ' ', text)

    # --------------------------------------------------
    # Remove all special characters/decorators
    # Keep only alphabets, numbers and spaces
    # --------------------------------------------------
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # --------------------------------------------------
    # Remove single-letter OCR noise
    # Example: q, v, x, b, i
    # --------------------------------------------------
    text = re.sub(r'\b[a-z]\b', ' ', text)

    # --------------------------------------------------
    # Remove common OCR garbage words
    # --------------------------------------------------
    noise_words = [
        "oo", "vv", "ww", "gb", "oe", "sx",
        "qx", "xx", "yy", "zz", "lll",
        "iii", "ooo", "mapdata"
    ]

    for word in noise_words:
        text = re.sub(rf'\b{word}\b', ' ', text)

    # --------------------------------------------------
    # Remove extra spaces
    # --------------------------------------------------
    text = re.sub(r'\s+', ' ', text)

    # Remove leading and trailing spaces
    text = text.strip()

    return text

# --------------------------------------------------
# Apply Cleaning
# --------------------------------------------------
df["cleaned_text"] = df["text"].apply(clean_text)

# --------------------------------------------------
# Save Cleaned Dataset
# --------------------------------------------------
output_path = "data/metadata/cleaned_dataset.csv"

df.to_csv(output_path, index=False)

# --------------------------------------------------
# Display Sample Output
# --------------------------------------------------
print("\nText Cleaning Completed Successfully!")
print(f"Saved at: {output_path}")

print("\nSample Cleaned Text:\n")

print(df[["text", "cleaned_text"]].head(10))