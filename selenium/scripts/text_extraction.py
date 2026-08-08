import os
import re
import pandas as pd
from PIL import Image
import pytesseract
from tqdm import tqdm

# ==========================================
# Tesseract Path
# ==========================================

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ==========================================
# Project Paths
# ==========================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "metadata",
    "dataset.csv"
)

print("=" * 60)
print("TEXT EXTRACTION USING OCR")
print("=" * 60)

print("Project Root :", PROJECT_ROOT)
print("CSV Path     :", CSV_PATH)

# ==========================================
# Load CSV
# ==========================================

df = pd.read_csv(CSV_PATH,dtype=str)
df=df.fillna("")
# ==========================================
# Create text column if missing
# ==========================================

if "text" not in df.columns:
    df["text"] = ""

# ==========================================
# OCR Loop
# ==========================================

for index in tqdm(range(len(df)), desc="Processing Images"):

    image_path = os.path.join(PROJECT_ROOT, df.loc[index, "image_path"])

    try:

        image = Image.open(image_path)

        text = pytesseract.image_to_string(image, lang="eng")

        # ----------------------------
        # Clean OCR Text
        # ----------------------------

        text = text.replace("\n", " ")

        text = re.sub(r"\s+", " ", text)

        text = text.strip()

        df.loc[index, "text"] = text

    except Exception as e:

        print(f"\nSkipped : {image_path}")

        print(e)

        df.loc[index, "text"] = ""

# ==========================================
# Save CSV
# ==========================================

df.to_csv(CSV_PATH, index=False, encoding="utf-8")

print("\n" + "=" * 60)
print("OCR COMPLETED SUCCESSFULLY")
print("Updated CSV :", CSV_PATH)
print("Total Images:", len(df))
print("=" * 60)