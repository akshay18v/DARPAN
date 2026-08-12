from pathlib import Path
import shutil
import random
from collections import Counter

# ============================================================
# DARPAN - MULTI-LABEL STRATIFIED YOLO DATASET SPLITTER
# ============================================================

ROOT = Path(r"D:\DARPAN\data\yolo_dataset")

IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels"

# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42
NUM_CLASSES = 6

CLASS_NAMES = {
    0: "hidden_fees",
    1: "false_urgency",
    2: "sneaking",
    3: "misdirection",
    4: "interface_interference",
    5: "obstruction"
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp"
}


# ============================================================
# CHECK RATIOS
# ============================================================

assert abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) < 1e-6


# ============================================================
# FIND ORIGINAL IMAGES
# ============================================================

images = [
    f for f in IMAGE_DIR.iterdir()
    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
]

print("=" * 70)
print("DARPAN MULTI-LABEL STRATIFIED DATASET SPLITTER")
print("=" * 70)

print(f"\nTotal images found: {len(images)}")


# ============================================================
# READ LABELS
# ============================================================

image_info = []

for image in images:

    label_file = LABEL_DIR / f"{image.stem}.txt"

    if not label_file.exists():
        print(f"WARNING: Missing label: {image.name}")
        continue

    classes = []

    with open(label_file, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])

                if 0 <= class_id < NUM_CLASSES:
                    classes.append(class_id)

            except ValueError:
                continue

    if len(classes) == 0:
        print(f"WARNING: No valid annotations: {image.name}")
        continue

    # Store UNIQUE classes for stratification
    unique_classes = set(classes)

    image_info.append({
        "image": image,
        "label": label_file,
        "classes": unique_classes
    })


print(f"Valid image-label pairs: {len(image_info)}")


# ============================================================
# TARGET NUMBER OF IMAGES PER SPLIT
# ============================================================

total = len(image_info)

target_train = round(total * TRAIN_RATIO)
target_val = round(total * VAL_RATIO)
target_test = total - target_train - target_val

targets = {
    "train": target_train,
    "val": target_val,
    "test": target_test
}

print("\nTarget split:")
print(f"Train : {target_train}")
print(f"Val   : {target_val}")
print(f"Test  : {target_test}")


# ============================================================
# RANDOMIZE
# ============================================================

random.seed(RANDOM_SEED)

random.shuffle(image_info)


# ============================================================
# MULTI-LABEL STRATIFICATION
# ============================================================
#
# We assign images one-by-one.
#
# Images containing rare classes are handled first.
#
# For each image, we select the split that:
#   1. Still needs images
#   2. Needs the image's classes the most
#
# This gives us a much better class distribution than
# a simple random split.
# ============================================================

remaining = {
    "train": target_train,
    "val": target_val,
    "test": target_test
}

splits = {
    "train": [],
    "val": [],
    "test": []
}


# ============================================================
# GLOBAL CLASS COUNTS
# ============================================================

global_class_counts = Counter()

for item in image_info:

    for class_id in item["classes"]:
        global_class_counts[class_id] += 1


# ============================================================
# PROCESS RARE / DIFFICULT IMAGES FIRST
# ============================================================

def image_priority(item):

    rarity_score = 0

    for class_id in item["classes"]:

        count = global_class_counts[class_id]

        if count > 0:
            rarity_score += 1 / count

    # More classes + rarer classes = higher priority
    return rarity_score + (len(item["classes"]) * 0.001)


image_info.sort(
    key=image_priority,
    reverse=True
)


# ============================================================
# CURRENT CLASS COUNTS
# ============================================================

split_class_counts = {
    "train": Counter(),
    "val": Counter(),
    "test": Counter()
}


# ============================================================
# CALCULATE IDEAL CLASS TARGETS
# ============================================================

ideal_class_counts = {
    split: {
        class_id: global_class_counts[class_id] * ratio
        for class_id in range(NUM_CLASSES)
    }
    for split, ratio in [
        ("train", TRAIN_RATIO),
        ("val", VAL_RATIO),
        ("test", TEST_RATIO)
    ]
}


# ============================================================
# CHOOSE BEST SPLIT
# ============================================================

def score_split(item, split):

    # If split is full, don't use it
    if remaining[split] <= 0:
        return float("-inf")

    score = 0

    for class_id in item["classes"]:

        current = split_class_counts[split][class_id]

        ideal = ideal_class_counts[split][class_id]

        # How much this class is still needed
        need = ideal - current

        if ideal > 0:
            score += need / ideal

    # Encourage filling the split according to target size
    fill_ratio = (
        remaining[split] /
        max(targets[split], 1)
    )

    score += fill_ratio * 0.5

    # Tiny random component to avoid identical tie decisions
    score += random.random() * 0.001

    return score


# ============================================================
# ASSIGN IMAGES
# ============================================================

for item in image_info:

    scores = {
        split: score_split(item, split)
        for split in splits
    }

    best_split = max(
        scores,
        key=scores.get
    )

    splits[best_split].append(item)

    remaining[best_split] -= 1

    for class_id in item["classes"]:
        split_class_counts[best_split][class_id] += 1


# ============================================================
# VERIFY COUNTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL IMAGE SPLIT")
print("=" * 70)

print(f"Train : {len(splits['train'])}")
print(f"Val   : {len(splits['val'])}")
print(f"Test  : {len(splits['test'])}")


# ============================================================
# DELETE PREVIOUS SPLIT
# ============================================================

print("\nRemoving previous train/val/test folders...")

split_folders = [
    IMAGE_DIR / "train",
    IMAGE_DIR / "val",
    IMAGE_DIR / "test",
    LABEL_DIR / "train",
    LABEL_DIR / "val",
    LABEL_DIR / "test"
]

for folder in split_folders:

    if folder.exists():
        shutil.rmtree(folder)


# ============================================================
# CREATE NEW FOLDERS
# ============================================================

for split in ["train", "val", "test"]:

    (IMAGE_DIR / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (LABEL_DIR / split).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# COPY DATA
# ============================================================

for split in ["train", "val", "test"]:

    print(f"\nCopying {split} data...")

    for item in splits[split]:

        image = item["image"]
        label = item["label"]

        shutil.copy2(
            image,
            IMAGE_DIR / split / image.name
        )

        shutil.copy2(
            label,
            LABEL_DIR / split / label.name
        )


# ============================================================
# COUNT ACTUAL BOUNDING BOXES
# ============================================================

def count_boxes(items):

    counts = Counter()

    for item in items:

        label = item["label"]

        with open(label, "r", encoding="utf-8") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                try:
                    class_id = int(parts[0])

                    if 0 <= class_id < NUM_CLASSES:
                        counts[class_id] += 1

                except ValueError:
                    continue

    return counts


train_boxes = count_boxes(splits["train"])
val_boxes = count_boxes(splits["val"])
test_boxes = count_boxes(splits["test"])


# ============================================================
# DISPLAY CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("BOUNDING BOX DISTRIBUTION")
print("=" * 70)

print(
    f"{'Class':<28}"
    f"{'Train':>10}"
    f"{'Val':>10}"
    f"{'Test':>10}"
    f"{'Total':>10}"
)

print("-" * 70)

for class_id in range(NUM_CLASSES):

    train = train_boxes[class_id]
    val = val_boxes[class_id]
    test = test_boxes[class_id]

    total_count = train + val + test

    print(
        f"{CLASS_NAMES[class_id]:<28}"
        f"{train:>10}"
        f"{val:>10}"
        f"{test:>10}"
        f"{total_count:>10}"
    )


# ============================================================
# CLASS PRESENCE CHECK
# ============================================================

print("\n" + "=" * 70)
print("CLASS PRESENCE CHECK")
print("=" * 70)

for class_id in range(NUM_CLASSES):

    name = CLASS_NAMES[class_id]

    train = train_boxes[class_id]
    val = val_boxes[class_id]
    test = test_boxes[class_id]

    status = "OK"

    if train == 0:
        status = "WARNING: no training boxes"

    elif val == 0:
        status = "WARNING: no validation boxes"

    elif test == 0:
        status = "WARNING: no test boxes"

    print(
        f"{name:<28} "
        f"Train={train:<4} "
        f"Val={val:<4} "
        f"Test={test:<4} "
        f"{status}"
    )


# ============================================================
# DATA LEAKAGE CHECK
# ============================================================

train_names = {
    item["image"].stem
    for item in splits["train"]
}

val_names = {
    item["image"].stem
    for item in splits["val"]
}

test_names = {
    item["image"].stem
    for item in splits["test"]
}

train_val = train_names & val_names
train_test = train_names & test_names
val_test = val_names & test_names


print("\n" + "=" * 70)
print("DATA LEAKAGE CHECK")
print("=" * 70)

print(f"Train ∩ Validation : {len(train_val)}")
print(f"Train ∩ Test       : {len(train_test)}")
print(f"Validation ∩ Test  : {len(val_test)}")


# ============================================================
# FINAL CHECK
# ============================================================

total_split_images = (
    len(splits["train"])
    + len(splits["val"])
    + len(splits["test"])
)

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"Original images : {total}")
print(f"Split images    : {total_split_images}")

if (
    total_split_images == total
    and len(train_val) == 0
    and len(train_test) == 0
    and len(val_test) == 0
):

    print("\n✓ SPLIT COMPLETED SUCCESSFULLY")
    print("✓ No image was lost.")
    print("✓ No image appears in multiple splits.")
    print("✓ Multi-label stratification applied.")

else:

    print("\n⚠ CHECK REQUIRED")


print("\nDataset:")
print(ROOT)

print("=" * 70)