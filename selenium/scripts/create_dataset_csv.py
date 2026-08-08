import os
import csv

# ==========================
# Project Root
# ==========================

print("Script Started")

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

print(PROJECT_ROOT)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic")
SYNTHETIC2_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic2")
METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")

print("PROJECT_ROOT  :", PROJECT_ROOT)
print("RAW_DIR       :", RAW_DIR)
print("SYNTHETIC_DIR :", SYNTHETIC_DIR)
print("SYNTHETIC2_DIR:", SYNTHETIC2_DIR)
print("METADATA_DIR  :", METADATA_DIR)

os.makedirs(METADATA_DIR, exist_ok=True)

CSV_PATH = os.path.join(METADATA_DIR, "dataset.csv")

# ==========================
# CSV Header
# ==========================

header = [
    "image_name",
    "image_path",
    "website",
    "source",
    "page",
    "text",
    "category",
    "severity"
]

rows = []

# ===================================================
# REAL IMAGES
# ===================================================

if os.path.exists(RAW_DIR):

    for website in sorted(os.listdir(RAW_DIR)):

        website_path = os.path.join(RAW_DIR, website)

        if not os.path.isdir(website_path):
            continue

        for file in sorted(os.listdir(website_path)):

            if file.lower().endswith((".png", ".jpg", ".jpeg")):

                rows.append([
                    file,
                    os.path.relpath(
                        os.path.join(website_path, file),
                        PROJECT_ROOT
                    ),
                    website,
                    "Real",
                    "",
                    "",
                    "",
                    ""
                ])

# ===================================================
# SYNTHETIC IMAGES
# ===================================================

if os.path.exists(SYNTHETIC_DIR):

    for pattern in sorted(os.listdir(SYNTHETIC_DIR)):

        pattern_path = os.path.join(SYNTHETIC_DIR, pattern)

        if not os.path.isdir(pattern_path):
            continue

        for file in sorted(os.listdir(pattern_path)):

            if file.lower().endswith((".png", ".jpg", ".jpeg")):

                rows.append([
                    file,
                    os.path.relpath(
                        os.path.join(pattern_path, file),
                        PROJECT_ROOT
                    ),
                    "Synthetic",
                    "Figma",
                    "",
                    "",
                    pattern,
                    ""
                ])

# ===================================================
# SYNTHETIC2 IMAGES
# ===================================================

if os.path.exists(SYNTHETIC2_DIR):

    for pattern in sorted(os.listdir(SYNTHETIC2_DIR)):

        pattern_path = os.path.join(SYNTHETIC2_DIR, pattern)

        if not os.path.isdir(pattern_path):
            continue

        for file in sorted(os.listdir(pattern_path)):

            if file.lower().endswith((".png", ".jpg", ".jpeg")):

                rows.append([
                    file,
                    os.path.relpath(
                        os.path.join(pattern_path, file),
                        PROJECT_ROOT
                    ),
                    "Synthetic",
                    "Figma",
                    "",
                    "",
                    pattern,
                    ""
                ])

# ===================================================
# WRITE CSV
# ===================================================

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow(header)

    writer.writerows(rows)

print("=" * 50)
print("CSV Created Successfully!")
print("Location :", CSV_PATH)
print("Total Images :", len(rows))
print("=" * 50)