from pathlib import Path
import sys
import json
import csv
import contextlib
import io


# ============================================================
# DARPAN FUSION MODEL EVALUATION
# YOLOv8n + DistilBERT/OCR/EVIDENCE
# ============================================================
#
# Evaluates the EXISTING fusion_predict.py on the YOLO test set.
#
# IMPORTANT:
# - Does NOT modify pipeline.py
# - Does NOT duplicate fusion rules
# - Uses fusion_predict() directly
# - Uses YOLO annotation files for image-level ground truth
# - Treats the task as MULTI-LABEL
# - Saves per-image predictions for later analysis
# ============================================================


print("=" * 75)
print("DARPAN MULTIMODAL FUSION EVALUATION")
print("YOLOv8n + DistilBERT/OCR/EVIDENCE")
print("=" * 75)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\DARPAN")

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "fusion_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CLASS MAPPING
# ============================================================

CLASS_NAMES = [
    "hidden_fees",
    "false_urgency",
    "sneaking",
    "misdirection",
    "interface_interference",
    "obstruction",
]


# ============================================================
# 3. IMPORT FUSION PIPELINE
# ============================================================

print("\n[1] LOADING FUSION PIPELINE")

SCRIPTS_DIR = (
    PROJECT_ROOT
    / "scripts"
)

if not SCRIPTS_DIR.exists():

    raise FileNotFoundError(
        f"scripts directory not found:\n{SCRIPTS_DIR}"
    )


sys.path.insert(
    0,
    str(SCRIPTS_DIR)
)


try:

    from fusion_predict import fusion_predict

except Exception as e:

    print("\nERROR while importing fusion_predict.py:")
    print(e)

    sys.exit(1)


print("✓ fusion_predict.py imported successfully")
print("✓ fusion_predict() found")


# ============================================================
# 4. CHECK DATASET
# ============================================================

print("\n[2] DATASET CHECK")

if not TEST_IMAGES.exists():

    raise FileNotFoundError(
        f"Test image directory not found:\n{TEST_IMAGES}"
    )

if not TEST_LABELS.exists():

    raise FileNotFoundError(
        f"Test label directory not found:\n{TEST_LABELS}"
    )


IMAGE_EXTENSIONS = {
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
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
)


print(
    "Test images:",
    len(image_files)
)


if len(image_files) == 0:

    raise RuntimeError(
        "No test images found."
    )


# ============================================================
# 5. GROUND TRUTH READER
# ============================================================

def read_ground_truth(
    label_path
):

    """
    Reads a YOLO annotation file and returns
    the set of classes present in the image.

    Example:

        1 0.8 0.2 0.2 0.2

    becomes:

        {"false_urgency"}
    """

    ground_truth = set()

    if not label_path.exists():

        return ground_truth


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

                class_id = int(
                    parts[0]
                )

            except ValueError:

                continue

            if (
                class_id >= 0
                and class_id < len(CLASS_NAMES)
            ):

                ground_truth.add(
                    CLASS_NAMES[class_id]
                )


    return ground_truth


# ============================================================
# 6. EXTRACT FUSION PREDICTIONS
# ============================================================

def extract_fusion_predictions(
    result
):

    """
    Extract confirmed labels from the structured
    output of fusion_predict().

    Expected:

        result["patterns"] = [
            {
                "label": "...",
                "confidence": ...,
                "decision": "confirmed"
            }
        ]
    """

    predictions = {}

    if not isinstance(
        result,
        dict
    ):

        return predictions


    patterns = result.get(
        "patterns",
        []
    )


    if not isinstance(
        patterns,
        list
    ):

        return predictions


    for pattern in patterns:

        if not isinstance(
            pattern,
            dict
        ):

            continue


        label = pattern.get(
            "label"
        )


        if label not in CLASS_NAMES:

            continue


        confidence = float(
            pattern.get(
                "confidence",
                0.0
            ) or 0.0
        )


        decision = pattern.get(
            "decision",
            ""
        )


        # Only count confirmed fusion decisions.
        if decision != "confirmed":

            continue


        predictions[label] = {

            "confidence":
                confidence,

            "confidence_percent":
                confidence * 100,

            "decision":
                decision,

            "reason":
                pattern.get(
                    "reason",
                    ""
                )
        }


    return predictions


# ============================================================
# 7. METRIC STORAGE
# ============================================================

statistics = {}

for label in CLASS_NAMES:

    statistics[label] = {

        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0
    }


# ============================================================
# 8. DETAILED RESULTS
# ============================================================

detailed_results = []


# ============================================================
# 9. RUN FUSION ON TEST SET
# ============================================================

print("\n[3] RUNNING FUSION MODEL")
print("=" * 75)


for index, image_path in enumerate(
    image_files,
    start=1
):

    print(
        f"[{index:02d}/{len(image_files):02d}] "
        f"{image_path.name}"
    )


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    label_path = (
        TEST_LABELS
        / f"{image_path.stem}.txt"
    )


    ground_truth = read_ground_truth(
        label_path
    )


    # --------------------------------------------------------
    # Run fusion
    # --------------------------------------------------------
    #
    # fusion_predict() itself prints YOLO/text/fusion
    # diagnostics.
    #
    # Suppress those diagnostics here because we want
    # a clean evaluation report.
    # --------------------------------------------------------

    captured_output = io.StringIO()


    try:

        with contextlib.redirect_stdout(
            captured_output
        ):

            result = fusion_predict(
                str(image_path)
            )


    except Exception as e:

        print(
            f"  ERROR: {e}"
        )


        detailed_results.append({

            "image":
                image_path.name,

            "ground_truth":
                sorted(ground_truth),

            "predictions":
                {},

            "pipeline_error":
                str(e)
        })


        continue


    # --------------------------------------------------------
    # Handle unexpected None
    # --------------------------------------------------------

    if result is None:

        print(
            "  WARNING: fusion_predict() returned None."
        )


        detailed_results.append({

            "image":
                image_path.name,

            "ground_truth":
                sorted(ground_truth),

            "predictions":
                {},

            "pipeline_error":
                "fusion_predict returned None"
        })


        continue


    # --------------------------------------------------------
    # Extract confirmed fusion predictions
    # --------------------------------------------------------

    predictions = extract_fusion_predictions(
        result
    )


    predicted_labels = set(
        predictions.keys()
    )


    # --------------------------------------------------------
    # Print compact result
    # --------------------------------------------------------

    if predicted_labels:

        compact = ", ".join(

            f"{label}("
            f"{predictions[label]['confidence']:.3f}"
            f")"

            for label in sorted(
                predicted_labels
            )
        )

    else:

        compact = "NONE"


    gt_text = ",".join(
        sorted(ground_truth)
    )

    print(
        f"  GT: {gt_text or 'NONE'}"
    )

    print(
        f"  FUSION: {compact}"
    )


    # --------------------------------------------------------
    # Calculate TP / FP / FN / TN
    # --------------------------------------------------------

    for label in CLASS_NAMES:

        actual = (
            label in ground_truth
        )

        predicted = (
            label in predicted_labels
        )


        if actual and predicted:

            statistics[label]["TP"] += 1

        elif not actual and predicted:

            statistics[label]["FP"] += 1

        elif actual and not predicted:

            statistics[label]["FN"] += 1

        else:

            statistics[label]["TN"] += 1


    # --------------------------------------------------------
    # Save detailed result
    # --------------------------------------------------------

    detailed_results.append({

        "image":
            image_path.name,

        "ground_truth":
            sorted(ground_truth),

        "predictions":
            predictions,

        "dark_pattern_detected":
            result.get(
                "dark_pattern_detected",
                False
            ),

        "total_patterns_detected":
            result.get(
                "total_patterns_detected",
                0
            ),

        "primary_pattern":
            result.get(
                "primary_pattern"
            ),

        "primary_confidence":
            result.get(
                "primary_confidence",
                0.0
            ),

        "visual_predictions":
            result.get(
                "visual_predictions",
                {}
            ),

        "text_predictions":
            result.get(
                "text_predictions",
                {}
            )
    })


# ============================================================
# 10. METRIC FUNCTIONS
# ============================================================

def calculate_precision(
    tp,
    fp
):

    denominator = tp + fp

    if denominator == 0:

        return 0.0

    return tp / denominator


def calculate_recall(
    tp,
    fn
):

    denominator = tp + fn

    if denominator == 0:

        return 0.0

    return tp / denominator


def calculate_f1(
    precision,
    recall
):

    denominator = (
        precision + recall
    )

    if denominator == 0:

        return 0.0

    return (
        2
        * precision
        * recall
        / denominator
    )


# ============================================================
# 11. PER-CLASS METRICS
# ============================================================

metrics = {}


for label in CLASS_NAMES:

    tp = statistics[label]["TP"]
    fp = statistics[label]["FP"]
    fn = statistics[label]["FN"]
    tn = statistics[label]["TN"]


    precision = calculate_precision(
        tp,
        fp
    )


    recall = calculate_recall(
        tp,
        fn
    )


    f1 = calculate_f1(
        precision,
        recall
    )


    support = tp + fn


    metrics[label] = {

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "support":
            support,

        "TP":
            tp,

        "FP":
            fp,

        "FN":
            fn,

        "TN":
            tn
    }


# ============================================================
# 12. PRINT PER-CLASS REPORT
# ============================================================

print("\n")
print("=" * 75)
print("FUSION MODEL TEST REPORT")
print("=" * 75)


print(
    f"{'Class':<28}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'Support':>10}"
)


print("-" * 75)


for label in CLASS_NAMES:

    m = metrics[label]


    print(

        f"{label:<28}"

        f"{m['precision']:>12.4f}"

        f"{m['recall']:>12.4f}"

        f"{m['f1']:>12.4f}"

        f"{m['support']:>10}"
    )


# ============================================================
# 13. MICRO AVERAGE
# ============================================================

total_tp = sum(
    statistics[label]["TP"]
    for label in CLASS_NAMES
)

total_fp = sum(
    statistics[label]["FP"]
    for label in CLASS_NAMES
)

total_fn = sum(
    statistics[label]["FN"]
    for label in CLASS_NAMES
)


micro_precision = calculate_precision(
    total_tp,
    total_fp
)


micro_recall = calculate_recall(
    total_tp,
    total_fn
)


micro_f1 = calculate_f1(
    micro_precision,
    micro_recall
)


print("\n" + "-" * 75)
print("MICRO AVERAGE")


print(
    f"Precision : {micro_precision:.4f}"
)


print(
    f"Recall    : {micro_recall:.4f}"
)


print(
    f"F1        : {micro_f1:.4f}"
)


# ============================================================
# 14. MACRO AVERAGE
# ============================================================

macro_precision = (

    sum(
        metrics[label]["precision"]
        for label in CLASS_NAMES
    )

    / len(CLASS_NAMES)
)


macro_recall = (

    sum(
        metrics[label]["recall"]
        for label in CLASS_NAMES
    )

    / len(CLASS_NAMES)
)


macro_f1 = (

    sum(
        metrics[label]["f1"]
        for label in CLASS_NAMES
    )

    / len(CLASS_NAMES)
)


print("\nMACRO AVERAGE")


print(
    f"Precision : {macro_precision:.4f}"
)


print(
    f"Recall    : {macro_recall:.4f}"
)


print(
    f"F1        : {macro_f1:.4f}"
)


# ============================================================
# 15. SAVE METRICS JSON
# ============================================================

metrics_output = {

    "model":
        "YOLOv8n + DistilBERT/OCR/evidence fusion",

    "evaluation_type":
        "multimodal_fusion",

    "dataset":
        "DARPAN YOLO test set",

    "test_images":
        len(image_files),

    "classes":
        metrics,

    "micro_average": {

        "precision":
            micro_precision,

        "recall":
            micro_recall,

        "f1":
            micro_f1
    },

    "macro_average": {

        "precision":
            macro_precision,

        "recall":
            macro_recall,

        "f1":
            macro_f1
    }
}


metrics_json = (
    OUTPUT_DIR
    / "fusion_metrics.json"
)


with open(
    metrics_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics_output,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 16. SAVE DETAILED PREDICTIONS
# ============================================================

details_json = (
    OUTPUT_DIR
    / "fusion_predictions.json"
)


with open(
    details_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        detailed_results,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 17. SAVE CSV REPORT
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "fusion_metrics.csv"
)


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)


    writer.writerow([

        "class",
        "precision",
        "recall",
        "f1",
        "support",
        "TP",
        "FP",
        "FN",
        "TN"
    ])


    for label in CLASS_NAMES:

        m = metrics[label]


        writer.writerow([

            label,

            round(
                m["precision"],
                6
            ),

            round(
                m["recall"],
                6
            ),

            round(
                m["f1"],
                6
            ),

            m["support"],

            m["TP"],

            m["FP"],

            m["FN"],

            m["TN"]
        ])


# ============================================================
# 18. SAVE PER-IMAGE SUMMARY CSV
# ============================================================

predictions_csv = (
    OUTPUT_DIR
    / "fusion_predictions.csv"
)


with open(
    predictions_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)


    writer.writerow([

        "image",
        "ground_truth",
        "predictions",
        "primary_pattern",
        "primary_confidence"
    ])


    for item in detailed_results:

        prediction_text = ";".join(

            f"{label}:"
            f"{info['confidence']:.4f}"

            for label, info
            in item.get(
                "predictions",
                {}
            ).items()
        )


        writer.writerow([

            item.get(
                "image",
                ""
            ),

            ";".join(
                item.get(
                    "ground_truth",
                    []
                )
            ),

            prediction_text,

            item.get(
                "primary_pattern",
                ""
            ),

            item.get(
                "primary_confidence",
                0.0
            )
        ])


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("FUSION EVALUATION COMPLETED")
print("=" * 75)


print("\nMetrics saved to:")
print(metrics_json)


print("\nDetailed predictions saved to:")
print(details_json)


print("\nMetrics CSV saved to:")
print(csv_path)


print("\nPer-image predictions CSV saved to:")
print(predictions_csv)


print("\n✓ fusion_predict.py was NOT modified.")
print("✓ pipeline.py was NOT modified.")
print("✓ Test images were NOT modified.")
print("✓ Test labels were NOT modified.")