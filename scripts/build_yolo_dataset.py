from pathlib import Path
import shutil

# ============================================================
# DARPAN → YOLO DATASET BUILDER
# ============================================================

ROOT = Path(r"D:\DARPAN")

LABEL_DIR = ROOT / "data" / "annotations" / "labels"
DATA_DIR = ROOT / "data"

OUTPUT = ROOT / "data" / "yolo_dataset"

# Image extensions that we accept
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp"
}

# ------------------------------------------------------------
# These are the 6 duplicate-name cases we already resolved
# ------------------------------------------------------------

SPECIAL_CASES = {
    "hotel_0": ROOT / "data" / "raw" / "agoda" / "hotel_0.png",

    "hotel_1": ROOT / "data" / "raw" / "easemytrip" / "hotel_1.png",

    "hotel_2": ROOT / "data" / "raw" / "easemytrip" / "hotel_2.png",

    "hotel_3": ROOT / "data" / "raw" / "easemytrip" / "hotel_3.png",

    "hotel_5": ROOT / "data" / "raw" / "easemytrip" / "hotel_5.png",

    "hotel_7": ROOT / "data" / "raw" / "easemytrip" / "hotel_7.png",
}


# ------------------------------------------------------------
# Create output directories
# ------------------------------------------------------------

IMAGE_OUTPUT = OUTPUT / "images"
LABEL_OUTPUT = OUTPUT / "labels"

IMAGE_OUTPUT.mkdir(parents=True, exist_ok=True)
LABEL_OUTPUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Find all images in DARPAN
# ------------------------------------------------------------

print("=" * 70)
print("SEARCHING FOR IMAGES")
print("=" * 70)

all_images = []

for file in DATA_DIR.rglob("*"):

    if not file.is_file():
        continue

    if file.suffix.lower() in IMAGE_EXTENSIONS:

        # Do not use files already inside yolo_dataset
        if OUTPUT in file.parents:
            continue

        all_images.append(file)


print(f"Images found: {len(all_images)}")


# ------------------------------------------------------------
# Create image lookup by filename stem
# ------------------------------------------------------------

image_lookup = {}

for image in all_images:

    stem = image.stem.lower()

    if stem not in image_lookup:
        image_lookup[stem] = []

    image_lookup[stem].append(image)


# ------------------------------------------------------------
# Process annotation files
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PROCESSING ANNOTATIONS")
print("=" * 70)

processed = 0
missing = 0
skipped = 0

used_names = set()

for label_file in sorted(LABEL_DIR.rglob("*.txt")):

    label_stem = label_file.stem

    # --------------------------------------------------------
    # Skip classes.txt
    # --------------------------------------------------------

    if label_file.name.lower() == "classes.txt":

        print(f"SKIP: {label_file.name}")
        skipped += 1
        continue

    # --------------------------------------------------------
    # Skip empty annotation files
    # --------------------------------------------------------

    if label_file.stat().st_size == 0:

        print(f"SKIP EMPTY: {label_file.name}")
        skipped += 1
        continue

    # --------------------------------------------------------
    # Check special duplicate cases
    # --------------------------------------------------------

    if label_stem in SPECIAL_CASES:

        image_file = SPECIAL_CASES[label_stem]

        if not image_file.exists():

            print(f"ERROR: Special image missing: {image_file}")
            missing += 1
            continue

        # Give unique name
        new_stem = image_file.parent.name + "_" + image_file.stem

    else:

        # ----------------------------------------------------
        # Normal case
        # ----------------------------------------------------

        candidates = image_lookup.get(label_stem.lower(), [])

        if len(candidates) == 0:

            print(f"MISSING IMAGE: {label_file.name}")
            missing += 1
            continue

        elif len(candidates) == 1:

            image_file = candidates[0]

            new_stem = image_file.stem

        else:

            # ------------------------------------------------
            # Unexpected duplicate
            # ------------------------------------------------

            print("\nWARNING: Multiple images found")
            print(f"Label: {label_file.name}")

            for candidate in candidates:
                print("   ", candidate)

            print("Skipping this file for safety.")

            missing += 1
            continue

    # --------------------------------------------------------
    # Prevent duplicate output names
    # --------------------------------------------------------

    original_new_stem = new_stem
    counter = 2

    while new_stem.lower() in used_names:

        new_stem = f"{original_new_stem}_{counter}"
        counter += 1

    used_names.add(new_stem.lower())

    # --------------------------------------------------------
    # Destination paths
    # --------------------------------------------------------

    new_image = IMAGE_OUTPUT / f"{new_stem}{image_file.suffix.lower()}"
    new_label = LABEL_OUTPUT / f"{new_stem}.txt"

    # --------------------------------------------------------
    # Copy image and label
    # --------------------------------------------------------

    shutil.copy2(image_file, new_image)
    shutil.copy2(label_file, new_label)

    processed += 1

    print(f"[{processed}] {image_file.name} -> {new_image.name}")


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET BUILD COMPLETE")
print("=" * 70)

print(f"Images copied : {processed}")
print(f"Labels copied : {processed}")
print(f"Skipped       : {skipped}")
print(f"Missing       : {missing}")

print("\nDataset location:")
print(OUTPUT)

print("\nImages:")
print(IMAGE_OUTPUT)

print("\nLabels:")
print(LABEL_OUTPUT)

print("=" * 70)