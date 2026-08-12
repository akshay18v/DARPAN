from pathlib import Path
import yaml

# ============================================================
# DARPAN YOLO SETUP VERIFICATION
# ============================================================

PROJECT_ROOT = Path(r"D:\DARPAN")
DATASET_ROOT = PROJECT_ROOT / "data" / "yolo_dataset"
YAML_FILE = DATASET_ROOT / "data.yaml"

EXPECTED_CLASSES = {
    0: "hidden_fees",
    1: "false_urgency",
    2: "sneaking",
    3: "misdirection",
    4: "interface_interference",
    5: "obstruction",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


print("=" * 70)
print("DARPAN YOLO SETUP VERIFICATION")
print("=" * 70)


# ============================================================
# 1. CHECK DATASET ROOT
# ============================================================

print("\n[1] DATASET ROOT")

if DATASET_ROOT.exists():
    print("✓ Dataset root exists:")
    print(DATASET_ROOT)
else:
    print("✗ Dataset root NOT FOUND")
    raise SystemExit


# ============================================================
# 2. CHECK DATA.YAML
# ============================================================

print("\n[2] DATA.YAML")

if not YAML_FILE.exists():
    print("✗ data.yaml NOT FOUND")
    raise SystemExit

print("✓ data.yaml found:")
print(YAML_FILE)

try:
    with open(YAML_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("✓ YAML syntax is valid")

except Exception as e:
    print("✗ YAML could not be read")
    print("Error:", e)
    raise SystemExit


# ============================================================
# 3. CHECK REQUIRED YAML FIELDS
# ============================================================

print("\n[3] YAML CONFIGURATION")

required_fields = [
    "path",
    "train",
    "val",
    "test",
    "nc",
    "names"
]

for field in required_fields:

    if field in config:
        print(f"✓ {field}: {config[field]}")
    else:
        print(f"✗ Missing field: {field}")


# ============================================================
# 4. CHECK NUMBER OF CLASSES
# ============================================================

print("\n[4] CLASS CONFIGURATION")

if config.get("nc") == 6:
    print("✓ nc = 6")
else:
    print(f"✗ Expected nc=6, found {config.get('nc')}")


# ============================================================
# 5. CHECK CLASS NAMES
# ============================================================

yaml_names = config.get("names")

if isinstance(yaml_names, dict):

    yaml_names = {
        int(k): v
        for k, v in yaml_names.items()
    }

print("\nConfigured classes:")

class_names_correct = True

for class_id in range(6):

    expected = EXPECTED_CLASSES[class_id]
    actual = yaml_names.get(class_id)

    if actual == expected:

        print(
            f"✓ {class_id}: {actual}"
        )

    else:

        print(
            f"✗ {class_id}: expected '{expected}', "
            f"found '{actual}'"
        )

        class_names_correct = False


# ============================================================
# 6. RESOLVE DATASET PATH
# ============================================================

yaml_path = Path(config["path"])

if not yaml_path.is_absolute():
    yaml_path = (YAML_FILE.parent / yaml_path).resolve()

print("\n[5] RESOLVED DATASET PATH")

print(yaml_path)

if yaml_path.exists():
    print("✓ Dataset path exists")
else:
    print("✗ Dataset path does NOT exist")


# ============================================================
# 7. CHECK TRAIN / VAL / TEST DIRECTORIES
# ============================================================

print("\n[6] DATASET DIRECTORIES")

splits = {
    "train": config["train"],
    "val": config["val"],
    "test": config["test"],
}

split_paths = {}

for split_name, relative_path in splits.items():

    split_path = yaml_path / relative_path

    split_paths[split_name] = split_path

    if split_path.exists():

        print(
            f"✓ {split_name:<5}: {split_path}"
        )

    else:

        print(
            f"✗ {split_name:<5}: NOT FOUND"
        )


# ============================================================
# 8. CHECK IMAGE COUNTS
# ============================================================

print("\n[7] IMAGE COUNTS")

expected_counts = {
    "train": 299,
    "val": 85,
    "test": 43,
}

image_counts = {}

for split_name, split_path in split_paths.items():

    if not split_path.exists():
        image_counts[split_name] = 0
        continue

    count = sum(
        1
        for f in split_path.iterdir()
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    image_counts[split_name] = count

    expected = expected_counts[split_name]

    if count == expected:

        print(
            f"✓ {split_name:<5}: "
            f"{count} images"
        )

    else:

        print(
            f"⚠ {split_name:<5}: "
            f"{count} images "
            f"(expected {expected})"
        )


# ============================================================
# 9. CHECK LABEL COUNTS
# ============================================================

print("\n[8] LABEL COUNTS")

label_counts = {}

for split_name in ["train", "val", "test"]:

    label_path = DATASET_ROOT / "labels" / split_name

    if not label_path.exists():

        print(
            f"✗ {split_name:<5}: label directory missing"
        )

        label_counts[split_name] = 0
        continue

    count = sum(
        1
        for f in label_path.iterdir()
        if f.is_file()
        and f.suffix.lower() == ".txt"
    )

    label_counts[split_name] = count

    if count == image_counts[split_name]:

        print(
            f"✓ {split_name:<5}: "
            f"{count} labels "
            f"(matches images)"
        )

    else:

        print(
            f"✗ {split_name:<5}: "
            f"{count} labels vs "
            f"{image_counts[split_name]} images"
        )


# ============================================================
# 10. CHECK IMAGE-LABEL PAIRS
# ============================================================

print("\n[9] IMAGE-LABEL PAIR CHECK")

pair_errors = 0

for split_name in ["train", "val", "test"]:

    image_path = split_paths[split_name]
    label_path = DATASET_ROOT / "labels" / split_name

    image_stems = {
        f.stem
        for f in image_path.iterdir()
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTENSIONS
    }

    label_stems = {
        f.stem
        for f in label_path.iterdir()
        if f.is_file()
        and f.suffix.lower() == ".txt"
    }

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    if not missing_labels and not missing_images:

        print(
            f"✓ {split_name:<5}: all image-label pairs match"
        )

    else:

        pair_errors += (
            len(missing_labels)
            + len(missing_images)
        )

        print(
            f"✗ {split_name}: "
            f"{len(missing_labels)} missing labels, "
            f"{len(missing_images)} missing images"
        )


# ============================================================
# 11. CHECK TOTAL DATASET
# ============================================================

print("\n[10] TOTAL DATASET")

total_images = sum(image_counts.values())
total_labels = sum(label_counts.values())

print(f"Total images : {total_images}")
print(f"Total labels : {total_labels}")

if total_images == 427:
    print("✓ Total images = 427")
else:
    print("⚠ Expected 427 images")

if total_labels == 427:
    print("✓ Total labels = 427")
else:
    print("⚠ Expected 427 labels")


# ============================================================
# 12. FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

all_good = (
    YAML_FILE.exists()
    and config.get("nc") == 6
    and class_names_correct
    and yaml_path.exists()
    and total_images == 427
    and total_labels == 427
    and pair_errors == 0
)

if all_good:

    print("✓ YOLO DATASET SETUP VERIFIED")
    print("✓ data.yaml is valid")
    print("✓ All 6 classes are configured correctly")
    print("✓ Train/Val/Test directories exist")
    print("✓ 427 images found")
    print("✓ 427 labels found")
    print("✓ All image-label pairs match")
    print("\nREADY FOR YOLOv8n ENVIRONMENT CHECK")

else:

    print("⚠ VERIFICATION FAILED")
    print("Review the warnings/errors above.")
    print("\nDO NOT START TRAINING YET.")

print("=" * 70)