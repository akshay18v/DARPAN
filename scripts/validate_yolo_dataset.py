from pathlib import Path
from PIL import Image

# ============================================================
# DARPAN YOLO DATASET VALIDATION
# ============================================================

ROOT = Path(r"D:\DARPAN\data\yolo_dataset")

IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels"

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp"
}

NUM_CLASSES = 6

CLASS_NAMES = {
    0: "hidden_fees",
    1: "false_urgency",
    2: "sneaking",
    3: "misdirection",
    4: "interface_interference",
    5: "obstruction"
}


print("=" * 70)
print("DARPAN YOLO DATASET VALIDATION")
print("=" * 70)


# ============================================================
# 1. FIND IMAGES
# ============================================================

images = [
    f for f in IMAGE_DIR.iterdir()
    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
]

labels = [
    f for f in LABEL_DIR.iterdir()
    if f.is_file() and f.suffix.lower() == ".txt"
]

print(f"\nImages found : {len(images)}")
print(f"Labels found : {len(labels)}")


# ============================================================
# 2. IMAGE-LABEL MATCHING
# ============================================================

image_stems = {
    image.stem.lower()
    for image in images
}

label_stems = {
    label.stem.lower()
    for label in labels
}

images_without_labels = image_stems - label_stems
labels_without_images = label_stems - image_stems

print("\n" + "=" * 70)
print("FILE MATCHING")
print("=" * 70)

print(f"Images without labels : {len(images_without_labels)}")
print(f"Labels without images : {len(labels_without_images)}")

if images_without_labels:
    print("\nImages without labels:")
    for name in sorted(images_without_labels):
        print("  ", name)

if labels_without_images:
    print("\nLabels without images:")
    for name in sorted(labels_without_images):
        print("  ", name)


# ============================================================
# 3. CHECK LABEL CONTENT
# ============================================================

invalid_files = []
empty_files = []
invalid_class = []
invalid_coordinates = []
invalid_box_size = []

class_counts = {
    class_id: 0
    for class_id in range(NUM_CLASSES)
}

total_boxes = 0


print("\n" + "=" * 70)
print("CHECKING LABEL CONTENT")
print("=" * 70)


for label_file in sorted(labels):

    # --------------------------------------------------------
    # Read label
    # --------------------------------------------------------

    try:
        with open(label_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

    except Exception as e:

        invalid_files.append(
            (label_file.name, f"Could not read file: {e}")
        )

        continue

    # --------------------------------------------------------
    # Empty file
    # --------------------------------------------------------

    if len(lines) == 0:

        empty_files.append(label_file.name)
        continue

    # --------------------------------------------------------
    # Check every annotation
    # --------------------------------------------------------

    for line_number, line in enumerate(lines, start=1):

        parts = line.split()

        # YOLO format:
        # class x_center y_center width height

        if len(parts) != 5:

            invalid_files.append(
                (
                    label_file.name,
                    f"Line {line_number}: expected 5 values, found {len(parts)}"
                )
            )

            continue

        try:

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:

            invalid_files.append(
                (
                    label_file.name,
                    f"Line {line_number}: non-numeric value"
                )
            )

            continue

        # ----------------------------------------------------
        # Class validation
        # ----------------------------------------------------

        if class_id < 0 or class_id >= NUM_CLASSES:

            invalid_class.append(
                (
                    label_file.name,
                    line_number,
                    class_id
                )
            )

        else:

            class_counts[class_id] += 1

        # ----------------------------------------------------
        # Coordinate validation
        # ----------------------------------------------------

        coordinates = [
            x_center,
            y_center,
            width,
            height
        ]

        if not all(0 <= value <= 1 for value in coordinates):

            invalid_coordinates.append(
                (
                    label_file.name,
                    line_number,
                    coordinates
                )
            )

        # ----------------------------------------------------
        # Width / height validation
        # ----------------------------------------------------

        if width <= 0 or height <= 0:

            invalid_box_size.append(
                (
                    label_file.name,
                    line_number,
                    width,
                    height
                )
            )

        total_boxes += 1


# ============================================================
# 4. CHECK IMAGE FILES
# ============================================================

corrupted_images = []

print("\n" + "=" * 70)
print("CHECKING IMAGE FILES")
print("=" * 70)

for image_file in images:

    try:

        with Image.open(image_file) as img:
            img.verify()

    except Exception as e:

        corrupted_images.append(
            (image_file.name, str(e))
        )


print(f"Corrupted images : {len(corrupted_images)}")


# ============================================================
# 5. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

for class_id, count in class_counts.items():

    class_name = CLASS_NAMES[class_id]

    print(
        f"{class_id} - {class_name:<25} {count}"
    )

print("-" * 50)
print(f"Total bounding boxes: {total_boxes}")


# ============================================================
# 6. PROBLEMS FOUND
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(f"Invalid label entries : {len(invalid_files)}")
print(f"Empty label files     : {len(empty_files)}")
print(f"Invalid class IDs     : {len(invalid_class)}")
print(f"Invalid coordinates   : {len(invalid_coordinates)}")
print(f"Invalid box sizes     : {len(invalid_box_size)}")
print(f"Corrupted images      : {len(corrupted_images)}")


# ============================================================
# 7. PRINT DETAILS IF PROBLEMS EXIST
# ============================================================

if invalid_files:

    print("\n" + "-" * 70)
    print("INVALID LABEL ENTRIES")
    print("-" * 70)

    for item in invalid_files:
        print(" ", item)


if empty_files:

    print("\n" + "-" * 70)
    print("EMPTY LABEL FILES")
    print("-" * 70)

    for item in empty_files:
        print(" ", item)


if invalid_class:

    print("\n" + "-" * 70)
    print("INVALID CLASS IDs")
    print("-" * 70)

    for item in invalid_class:
        print(" ", item)


if invalid_coordinates:

    print("\n" + "-" * 70)
    print("INVALID COORDINATES")
    print("-" * 70)

    for item in invalid_coordinates:
        print(" ", item)


if invalid_box_size:

    print("\n" + "-" * 70)
    print("INVALID BOX SIZES")
    print("-" * 70)

    for item in invalid_box_size:
        print(" ", item)


if corrupted_images:

    print("\n" + "-" * 70)
    print("CORRUPTED IMAGES")
    print("-" * 70)

    for item in corrupted_images:
        print(" ", item)


# ============================================================
# 8. FINAL STATUS
# ============================================================

total_problems = (
    len(images_without_labels)
    + len(labels_without_images)
    + len(invalid_files)
    + len(empty_files)
    + len(invalid_class)
    + len(invalid_coordinates)
    + len(invalid_box_size)
    + len(corrupted_images)
)


print("\n" + "=" * 70)

if total_problems == 0:

    print("✓ DATASET VALIDATION PASSED")
    print("✓ Dataset is ready for YOLO train/validation/test splitting.")

else:

    print("⚠ DATASET VALIDATION FOUND PROBLEMS")
    print(f"Total problem count: {total_problems}")
    print("Fix the reported problems before training.")

print("=" * 70)