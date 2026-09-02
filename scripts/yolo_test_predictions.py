from pathlib import Path
from ultralytics import YOLO
import yaml

# ============================================================
# DARPAN YOLO PER-IMAGE PREDICTIONS
# ============================================================

PROJECT_ROOT = Path(r"D:\DARPAN")

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "darpan_yolov8n-2"
    / "weights"
    / "best.pt"
)

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "yolo_dataset"
    / "data.yaml"
)

TEST_IMAGES = (
    PROJECT_ROOT
    / "data"
    / "yolo_dataset"
    / "images"
    / "test"
)

TEST_LABELS = (
    PROJECT_ROOT
    / "data"
    / "yolo_dataset"
    / "labels"
    / "test"
)

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

CONF_THRESHOLD = 0.25

# ------------------------------------------------------------
# LOAD CLASS NAMES
# ------------------------------------------------------------

with open(DATA_YAML, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

names = data["names"]

if isinstance(names, dict):
    CLASS_NAMES = {
        int(k): v
        for k, v in names.items()
    }
else:
    CLASS_NAMES = {
        i: name
        for i, name in enumerate(names)
    }

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("=" * 75)
print("DARPAN YOLO PER-IMAGE PREDICTIONS")
print("=" * 75)

print("\nModel:")
print(MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found:\n{MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

print("✓ YOLO model loaded")

# ------------------------------------------------------------
# TEST IMAGES
# ------------------------------------------------------------

extensions = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp"
}

image_files = sorted(
    [
        p
        for p in TEST_IMAGES.iterdir()
        if p.is_file()
        and p.suffix.lower() in extensions
    ]
)

print("\nTest images:", len(image_files))
print("Confidence threshold:", CONF_THRESHOLD)

# ------------------------------------------------------------
# GROUND TRUTH
# ------------------------------------------------------------

def get_ground_truth(image_path):

    label_path = (
        TEST_LABELS
        / f"{image_path.stem}.txt"
    )

    labels = set()

    if not label_path.exists():
        return labels

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                continue

            if class_id in CLASS_NAMES:
                labels.add(
                    CLASS_NAMES[class_id]
                )

    return labels


# ------------------------------------------------------------
# RUN YOLO
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("IMAGE | GROUND TRUTH | YOLO PREDICTIONS")
print("=" * 75)

for index, image_path in enumerate(
    image_files,
    start=1
):

    ground_truth = get_ground_truth(
        image_path
    )

    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        device="cpu",
        verbose=False
    )

    result = results[0]

    predictions = {}

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            if class_id not in CLASS_NAMES:
                continue

            label = CLASS_NAMES[class_id]

            # Keep highest confidence for each class
            if (
                label not in predictions
                or confidence
                > predictions[label]
            ):
                predictions[label] = confidence

    # --------------------------------------------------------
    # FORMAT
    # --------------------------------------------------------

    gt_text = (
        ",".join(sorted(ground_truth))
        if ground_truth
        else "NONE"
    )

    pred_text = (
        ",".join(
            f"{label}({confidence:.3f})"
            for label, confidence
            in sorted(
                predictions.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )
        if predictions
        else "NONE"
    )

    print(
        f"[{index:02d}/{len(image_files):02d}] "
        f"{image_path.name}"
        f" | GT: {gt_text}"
        f" | YOLO: {pred_text}"
    )

print("\n" + "=" * 75)
print("YOLO PREDICTION EXTRACTION COMPLETED")
print("=" * 75)