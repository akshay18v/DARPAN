from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import torch
import cv2

# ============================================================
# DARPAN YOLO TEST SET ERROR ANALYSIS
# ============================================================

print("=" * 70)
print("DARPAN YOLO TEST SET ERROR ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

MODEL_PATH = Path(
    r"D:\DARPAN\runs\darpan_yolov8n\weights\best.pt"
)

DATASET_ROOT = Path(
    r"D:\DARPAN\data\yolo_dataset"
)

TEST_IMAGES = DATASET_ROOT / "images" / "test"
TEST_LABELS = DATASET_ROOT / "labels" / "test"

OUTPUT_DIR = Path(
    r"D:\DARPAN\runs\error_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# CLASS NAMES
# ------------------------------------------------------------

CLASS_NAMES = {
    0: "hidden_fees",
    1: "false_urgency",
    2: "sneaking",
    3: "misdirection",
    4: "interface_interference",
    5: "obstruction",
}

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

print("\n[1] DEVICE")

if torch.cuda.is_available():
    DEVICE = 0
    print("CUDA available : True")
    print("GPU             :", torch.cuda.get_device_name(0))
else:
    DEVICE = "cpu"
    print("CUDA available : False")
    print("Using CPU")

# ------------------------------------------------------------
# FILE CHECK
# ------------------------------------------------------------

print("\n[2] FILE CHECK")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not TEST_IMAGES.exists():
    raise FileNotFoundError(
        f"Test images folder not found:\n{TEST_IMAGES}"
    )

if not TEST_LABELS.exists():
    raise FileNotFoundError(
        f"Test labels folder not found:\n{TEST_LABELS}"
    )

print("✓ Model found")
print("✓ Test images found")
print("✓ Test labels found")

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("\n[3] LOADING MODEL")

model = YOLO(str(MODEL_PATH))

print("✓ best.pt loaded successfully")

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def yolo_to_xyxy(class_id, x_center, y_center, width, height,
                 image_width, image_height):

    x1 = (x_center - width / 2) * image_width
    y1 = (y_center - height / 2) * image_height
    x2 = (x_center + width / 2) * image_width
    y2 = (y_center + height / 2) * image_height

    return [
        x1,
        y1,
        x2,
        y2
    ]


def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width * intersection_height
    )

    area1 = max(0, box1[2] - box1[0]) * \
            max(0, box1[3] - box1[1])

    area2 = max(0, box2[2] - box2[0]) * \
            max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_ground_truth(label_path, image_width, image_height):

    ground_truth = []

    if not label_path.exists():
        return ground_truth

    with open(label_path, "r", encoding="utf-8") as file:

        for line in file:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            box = yolo_to_xyxy(
                class_id,
                x_center,
                y_center,
                width,
                height,
                image_width,
                image_height
            )

            ground_truth.append({
                "class_id": class_id,
                "box": box
            })

    return ground_truth


# ------------------------------------------------------------
# FIND TEST IMAGES
# ------------------------------------------------------------

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

image_paths = sorted(
    [
        p for p in TEST_IMAGES.iterdir()
        if p.suffix.lower() in image_extensions
    ]
)

print("\n[4] TEST DATASET")

print("Test images:", len(image_paths))

# ------------------------------------------------------------
# STATISTICS
# ------------------------------------------------------------

class_gt = defaultdict(int)
class_correct = defaultdict(int)
class_missed = defaultdict(int)
class_wrong_class = defaultdict(int)

total_gt = 0
total_predictions = 0

correct_detections = 0
missed_detections = 0
wrong_class_detections = 0
false_positive_detections = 0

image_results = []

# ------------------------------------------------------------
# ANALYSIS
# ------------------------------------------------------------

print("\n[5] RUNNING ERROR ANALYSIS")
print("IoU threshold :", IOU_THRESHOLD)
print("Confidence    :", CONF_THRESHOLD)

for index, image_path in enumerate(image_paths, start=1):

    image = cv2.imread(str(image_path))

    if image is None:
        print(
            f"WARNING: Could not read {image_path.name}"
        )
        continue

    image_height, image_width = image.shape[:2]

    label_path = TEST_LABELS / (
        image_path.stem + ".txt"
    )

    ground_truth = load_ground_truth(
        label_path,
        image_width,
        image_height
    )

    # --------------------------------------------------------
    # MODEL PREDICTIONS
    # --------------------------------------------------------

    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        device=DEVICE,
        verbose=False
    )

    result = results[0]

    predictions = []

    if result.boxes is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, class_id, confidence in zip(
            boxes,
            classes,
            confidences
        ):

            predictions.append({
                "class_id": int(class_id),
                "confidence": float(confidence),
                "box": box.tolist()
            })

    total_gt += len(ground_truth)
    total_predictions += len(predictions)

    for gt in ground_truth:
        class_gt[gt["class_id"]] += 1

    # --------------------------------------------------------
    # MATCHING
    # --------------------------------------------------------

    matched_gt = set()
    matched_predictions = set()

    correct_for_image = []
    wrong_class_for_image = []
    missed_for_image = []
    false_positive_for_image = []

    # --------------------------------------------------------
    # FIRST: MATCH SAME-CLASS BOXES
    # --------------------------------------------------------

    candidate_matches = []

    for gt_index, gt in enumerate(ground_truth):

        for pred_index, pred in enumerate(predictions):

            if gt["class_id"] != pred["class_id"]:
                continue

            iou = calculate_iou(
                gt["box"],
                pred["box"]
            )

            if iou >= IOU_THRESHOLD:

                candidate_matches.append(
                    (
                        iou,
                        gt_index,
                        pred_index
                    )
                )

    # Highest IoU first
    candidate_matches.sort(
        reverse=True
    )

    for iou, gt_index, pred_index in candidate_matches:

        if gt_index in matched_gt:
            continue

        if pred_index in matched_predictions:
            continue

        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)

        class_id = ground_truth[
            gt_index
        ]["class_id"]

        class_correct[class_id] += 1
        correct_detections += 1

        correct_for_image.append({
            "class": CLASS_NAMES[class_id],
            "iou": iou
        })

    # --------------------------------------------------------
    # FIND WRONG CLASS PREDICTIONS
    # --------------------------------------------------------

    for gt_index, gt in enumerate(ground_truth):

        if gt_index in matched_gt:
            continue

        best_iou = 0
        best_pred_index = None

        for pred_index, pred in enumerate(predictions):

            if pred_index in matched_predictions:
                continue

            iou = calculate_iou(
                gt["box"],
                pred["box"]
            )

            if iou > best_iou:
                best_iou = iou
                best_pred_index = pred_index

        if (
            best_pred_index is not None
            and best_iou >= IOU_THRESHOLD
        ):

            pred = predictions[
                best_pred_index
            ]

            gt_class = gt["class_id"]
            pred_class = pred["class_id"]

            matched_gt.add(gt_index)
            matched_predictions.add(
                best_pred_index
            )

            class_wrong_class[
                gt_class
            ] += 1

            wrong_class_detections += 1

            wrong_class_for_image.append({
                "actual": CLASS_NAMES[gt_class],
                "predicted": CLASS_NAMES[pred_class],
                "confidence": pred["confidence"],
                "iou": best_iou
            })

    # --------------------------------------------------------
    # MISSED GROUND TRUTH
    # --------------------------------------------------------

    for gt_index, gt in enumerate(ground_truth):

        if gt_index in matched_gt:
            continue

        class_id = gt["class_id"]

        class_missed[class_id] += 1
        missed_detections += 1

        missed_for_image.append({
            "class": CLASS_NAMES[class_id]
        })

    # --------------------------------------------------------
    # FALSE POSITIVES
    # --------------------------------------------------------

    for pred_index, pred in enumerate(predictions):

        if pred_index in matched_predictions:
            continue

        false_positive_detections += 1

        false_positive_for_image.append({
            "class": CLASS_NAMES[
                pred["class_id"]
            ],
            "confidence": pred["confidence"]
        })

    # --------------------------------------------------------
    # STORE IMAGE RESULT
    # --------------------------------------------------------

    image_results.append({
        "image": image_path.name,
        "gt_count": len(ground_truth),
        "prediction_count": len(predictions),
        "correct": correct_for_image,
        "wrong_class": wrong_class_for_image,
        "missed": missed_for_image,
        "false_positive": false_positive_for_image
    })

    print(
        f"[{index:02d}/{len(image_paths)}] "
        f"{image_path.name}"
    )

# ============================================================
# SAVE DETAILED REPORT
# ============================================================

report_path = OUTPUT_DIR / "detailed_error_analysis.txt"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "DARPAN YOLO TEST SET ERROR ANALYSIS\n"
    )

    report.write("=" * 70 + "\n\n")

    report.write(
        f"Test images: {len(image_paths)}\n"
    )

    report.write(
        f"Ground truth boxes: {total_gt}\n"
    )

    report.write(
        f"Predictions: {total_predictions}\n\n"
    )

    report.write(
        f"Correct detections: {correct_detections}\n"
    )

    report.write(
        f"Wrong-class detections: "
        f"{wrong_class_detections}\n"
    )

    report.write(
        f"Missed detections: "
        f"{missed_detections}\n"
    )

    report.write(
        f"False positives: "
        f"{false_positive_detections}\n\n"
    )

    report.write("=" * 70 + "\n")
    report.write("CLASS-WISE ANALYSIS\n")
    report.write("=" * 70 + "\n\n")

    for class_id, class_name in CLASS_NAMES.items():

        report.write(
            f"{class_name}\n"
        )

        report.write(
            f"  Ground truth : "
            f"{class_gt[class_id]}\n"
        )

        report.write(
            f"  Correct      : "
            f"{class_correct[class_id]}\n"
        )

        report.write(
            f"  Wrong class  : "
            f"{class_wrong_class[class_id]}\n"
        )

        report.write(
            f"  Missed       : "
            f"{class_missed[class_id]}\n\n"
        )

    report.write("=" * 70 + "\n")
    report.write("IMAGE-BY-IMAGE ANALYSIS\n")
    report.write("=" * 70 + "\n\n")

    for item in image_results:

        report.write(
            f"\nIMAGE: {item['image']}\n"
        )

        report.write(
            f"Ground truth boxes: "
            f"{item['gt_count']}\n"
        )

        report.write(
            f"Predictions: "
            f"{item['prediction_count']}\n"
        )

        if item["correct"]:

            report.write("\n  CORRECT:\n")

            for x in item["correct"]:

                report.write(
                    f"    {x['class']} "
                    f"(IoU={x['iou']:.3f})\n"
                )

        if item["wrong_class"]:

            report.write(
                "\n  WRONG CLASS:\n"
            )

            for x in item["wrong_class"]:

                report.write(
                    f"    Actual: {x['actual']} | "
                    f"Predicted: {x['predicted']} | "
                    f"Confidence: {x['confidence']:.3f} | "
                    f"IoU: {x['iou']:.3f}\n"
                )

        if item["missed"]:

            report.write(
                "\n  MISSED:\n"
            )

            for x in item["missed"]:

                report.write(
                    f"    {x['class']}\n"
                )

        if item["false_positive"]:

            report.write(
                "\n  FALSE POSITIVE:\n"
            )

            for x in item["false_positive"]:

                report.write(
                    f"    {x['class']} | "
                    f"Confidence: "
                    f"{x['confidence']:.3f}\n"
                )

# ============================================================
# CONFUSION MATRIX STYLE REPORT
# ============================================================

confusion_path = (
    OUTPUT_DIR /
    "class_confusions.txt"
)

confusions = defaultdict(int)

for item in image_results:

    for x in item["wrong_class"]:

        key = (
            x["actual"],
            x["predicted"]
        )

        confusions[key] += 1

with open(
    confusion_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "DARPAN CLASS CONFUSION ANALYSIS\n"
    )

    file.write("=" * 70 + "\n\n")

    if not confusions:

        file.write(
            "No wrong-class detections found.\n"
        )

    else:

        sorted_confusions = sorted(
            confusions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for (actual, predicted), count in sorted_confusions:

            file.write(
                f"Actual: {actual:<25} "
                f"Predicted: {predicted:<25} "
                f"Count: {count}\n"
            )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETED")
print("=" * 70)

print("\nTotal test images      :", len(image_paths))
print("Ground truth boxes     :", total_gt)
print("Predictions            :", total_predictions)

print("\nCorrect detections     :", correct_detections)
print("Wrong-class detections :", wrong_class_detections)
print("Missed detections      :", missed_detections)
print("False positives        :", false_positive_detections)

print("\nClass-wise summary:")

for class_id, class_name in CLASS_NAMES.items():

    print(
        f"{class_name:<27}"
        f" GT={class_gt[class_id]:<4}"
        f" Correct={class_correct[class_id]:<4}"
        f" Wrong={class_wrong_class[class_id]:<4}"
        f" Missed={class_missed[class_id]}"
    )

print("\nReports saved to:")

print(
    OUTPUT_DIR /
    "detailed_error_analysis.txt"
)

print(
    OUTPUT_DIR /
    "class_confusions.txt"
)

print("\n✓ Dataset was NOT modified.")
print("✓ Test images were NOT modified.")
print("✓ best.pt was NOT modified.")