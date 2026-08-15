from pathlib import Path
from ultralytics import YOLO
import cv2
import yaml
import torch

# ============================================================
# DARPAN YOLO ERROR VISUALIZATION
# ============================================================

print("=" * 70)
print("DARPAN YOLO TEST SET ERROR VISUALIZATION")
print("=" * 70)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
MODEL_PATH = Path(
    r"D:\DARPAN\runs\darpan_yolo11s\weights\best.pt"
)

DATA_YAML = Path(
    r"D:\DARPAN\data\yolo_dataset\data.yaml"
)

OUTPUT_DIR = Path(
    r"D:\DARPAN\runs\error_analysis_yolo11s\visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

print("\n[1] DEVICE")

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    DEVICE = 0
    print("GPU:", torch.cuda.get_device_name(0))
else:
    DEVICE = "cpu"
    print("Using CPU")

# ------------------------------------------------------------
# FILE CHECK
# ------------------------------------------------------------

print("\n[2] FILE CHECK")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"data.yaml not found:\n{DATA_YAML}"
    )

print("✓ Model found")
print("✓ data.yaml found")

# ------------------------------------------------------------
# LOAD DATASET YAML
# ------------------------------------------------------------

print("\n[3] DATASET")

with open(DATA_YAML, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

names = data["names"]

if isinstance(names, dict):
    CLASS_NAMES = [names[i] for i in sorted(names.keys())]
else:
    CLASS_NAMES = names

print("Classes:")

for i, name in enumerate(CLASS_NAMES):
    print(f"  {i}: {name}")

# ------------------------------------------------------------
# TEST PATH
# ------------------------------------------------------------

TEST_IMAGES = Path(
    r"D:\DARPAN\data\yolo_dataset\images\test"
)

TEST_LABELS = Path(
    r"D:\DARPAN\data\yolo_dataset\labels\test"
)

if not TEST_IMAGES.exists():
    raise FileNotFoundError(
        f"Test images not found:\n{TEST_IMAGES}"
    )

if not TEST_LABELS.exists():
    raise FileNotFoundError(
        f"Test labels not found:\n{TEST_LABELS}"
    )

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("\n[4] LOADING MODEL")

model = YOLO(str(MODEL_PATH))

print("✓ best.pt loaded successfully")

# ------------------------------------------------------------
# IMAGE LIST
# ------------------------------------------------------------

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

image_files = sorted(
    [
        p
        for p in TEST_IMAGES.iterdir()
        if p.suffix.lower() in image_extensions
    ]
)

print("\n[5] TEST SET")

print("Test images:", len(image_files))

# ------------------------------------------------------------
# OUTPUT FOLDERS
# ------------------------------------------------------------

CORRECT_DIR = OUTPUT_DIR / "correct"
MISSED_DIR = OUTPUT_DIR / "missed"
FALSE_POSITIVE_DIR = OUTPUT_DIR / "false_positives"
WRONG_CLASS_DIR = OUTPUT_DIR / "wrong_class"

for folder in [
    CORRECT_DIR,
    MISSED_DIR,
    FALSE_POSITIVE_DIR,
    WRONG_CLASS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.50

print("\n[6] SETTINGS")

print("Confidence threshold:", CONF_THRESHOLD)
print("IoU threshold:", IOU_THRESHOLD)

# ------------------------------------------------------------
# IOU FUNCTION
# ------------------------------------------------------------

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ------------------------------------------------------------
# DRAW FUNCTIONS
# ------------------------------------------------------------

def draw_ground_truth(
    image,
    box,
    class_name
):

    x1, y1, x2, y2 = map(int, box)

    # GREEN = ground truth
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    text = f"GT: {class_name}"

    cv2.putText(
        image,
        text,
        (x1, max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )


def draw_prediction(
    image,
    box,
    class_name,
    confidence,
    label_color=(0, 0, 255)
):

    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        label_color,
        2
    )

    text = f"PRED: {class_name} {confidence:.2f}"

    cv2.putText(
        image,
        text,
        (x1, min(image.shape[0] - 5, y2 + 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        label_color,
        2
    )


# ------------------------------------------------------------
# PROCESS IMAGES
# ------------------------------------------------------------

print("\n[7] PROCESSING TEST IMAGES")

summary = {
    "correct": 0,
    "missed": 0,
    "false_positive": 0,
    "wrong_class": 0
}

for index, image_path in enumerate(image_files, start=1):

    print(
        f"[{index:02d}/{len(image_files):02d}] "
        f"{image_path.name}"
    )

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:
        print("  WARNING: Could not read image")
        continue

    visualization = image.copy()

    # --------------------------------------------------------
    # LOAD GROUND TRUTH
    # --------------------------------------------------------

    label_path = TEST_LABELS / (
        image_path.stem + ".txt"
    )

    ground_truth = []

    if label_path.exists():

        with open(label_path, "r") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])

                width = float(parts[3])
                height = float(parts[4])

                img_h, img_w = image.shape[:2]

                x1 = (x_center - width / 2) * img_w
                y1 = (y_center - height / 2) * img_h

                x2 = (x_center + width / 2) * img_w
                y2 = (y_center + height / 2) * img_h

                ground_truth.append({
                    "class_id": class_id,
                    "box": [x1, y1, x2, y2],
                    "matched": False
                })

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        device=DEVICE,
        verbose=False
    )

    predictions = []

    result = results[0]

    if result.boxes is not None:

        for box in result.boxes:

            xyxy = box.xyxy[0].cpu().numpy()

            confidence = float(
                box.conf[0].cpu().numpy()
            )

            class_id = int(
                box.cls[0].cpu().numpy()
            )

            predictions.append({
                "class_id": class_id,
                "box": xyxy.tolist(),
                "confidence": confidence,
                "matched": False
            })

    # --------------------------------------------------------
    # DRAW GROUND TRUTH
    # --------------------------------------------------------

    for gt in ground_truth:

        draw_ground_truth(
            visualization,
            gt["box"],
            CLASS_NAMES[gt["class_id"]]
        )

    # --------------------------------------------------------
    # MATCH PREDICTIONS
    # --------------------------------------------------------

    for pred in predictions:

        best_iou = 0.0
        best_gt = None

        for gt in ground_truth:

            if gt["matched"]:
                continue

            iou = calculate_iou(
                pred["box"],
                gt["box"]
            )

            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        # ----------------------------------------------------
        # CORRECT CLASS
        # ----------------------------------------------------

        if (
            best_gt is not None
            and best_iou >= IOU_THRESHOLD
            and pred["class_id"] == best_gt["class_id"]
        ):

            pred["matched"] = True
            best_gt["matched"] = True

            summary["correct"] += 1

            draw_prediction(
                visualization,
                pred["box"],
                CLASS_NAMES[pred["class_id"]],
                pred["confidence"],
                (255, 0, 0)
            )

        # ----------------------------------------------------
        # WRONG CLASS
        # ----------------------------------------------------

        elif (
            best_gt is not None
            and best_iou >= IOU_THRESHOLD
            and pred["class_id"] != best_gt["class_id"]
        ):

            pred["matched"] = True
            best_gt["matched"] = True

            summary["wrong_class"] += 1

            draw_prediction(
                visualization,
                pred["box"],
                CLASS_NAMES[pred["class_id"]],
                pred["confidence"],
                (0, 0, 255)
            )

        # ----------------------------------------------------
        # FALSE POSITIVE
        # ----------------------------------------------------

        else:

            summary["false_positive"] += 1

            draw_prediction(
                visualization,
                pred["box"],
                CLASS_NAMES[pred["class_id"]],
                pred["confidence"],
                (0, 165, 255)
            )

    # --------------------------------------------------------
    # MISSED GROUND TRUTH
    # --------------------------------------------------------

    for gt in ground_truth:

        if not gt["matched"]:

            summary["missed"] += 1

    # --------------------------------------------------------
    # SAVE BASE IMAGE
    # --------------------------------------------------------

    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(
        str(output_path),
        visualization
    )

    # --------------------------------------------------------
    # SAVE SPECIAL ERROR COPIES
    # --------------------------------------------------------

    has_missed = any(
        not gt["matched"]
        for gt in ground_truth
    )

    has_false_positive = any(
        not pred["matched"]
        for pred in predictions
    )

    if has_missed:

        cv2.imwrite(
            str(MISSED_DIR / image_path.name),
            visualization
        )

    if has_false_positive:

        cv2.imwrite(
            str(FALSE_POSITIVE_DIR / image_path.name),
            visualization
        )

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETED")
print("=" * 70)

print("\nCorrect detections :", summary["correct"])
print("Wrong-class        :", summary["wrong_class"])
print("Missed detections  :", summary["missed"])
print("False positives    :", summary["false_positive"])

print("\nOutput directory:")

print(OUTPUT_DIR)

print("\nFolders:")

print("All visualizations:")
print(OUTPUT_DIR)

print("\nMissed detections:")
print(MISSED_DIR)

print("\nFalse positives:")
print(FALSE_POSITIVE_DIR)

print("\n✓ Test dataset was NOT modified.")
print("✓ best.pt was NOT modified.")