from pathlib import Path
import sys
import json
import csv
from collections import defaultdict

# ============================================================
# DARPAN TEXT MODEL EVALUATION
# ============================================================
#
# Evaluates the trained DistilBERT classifier on the DARPAN
# YOLO test images.
#
# IMPORTANT:
# - Reuses OCR and chunking from pipeline.py
# - Reuses the trained DistilBERT model
# - Uses predict_chunk()
# - DOES NOT call analyze_image()
# - DOES NOT run semantic evidence analysis
# - DOES NOT modify pipeline.py
#
# This is intentionally a CLASSIFIER evaluation.
# ============================================================


print("=" * 75)
print("DARPAN DISTILBERT TEXT MODEL EVALUATION")
print("=" * 75)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\DARPAN"
)

SRC_DIR = (
    PROJECT_ROOT
    / "src"
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "text_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CLASS MAPPING
# ============================================================

CLASS_NAMES = {
    0: "hidden_fees",
    1: "false_urgency",
    2: "sneaking",
    3: "misdirection",
    4: "interface_interference",
    5: "obstruction"
}

CLASS_LABELS = list(
    CLASS_NAMES.values()
)


# ============================================================
# 3. IMPORT EXISTING PIPELINE COMPONENTS
# ============================================================

print("\n[1] LOADING TEXT MODEL")

if not SRC_DIR.exists():

    raise FileNotFoundError(
        f"src directory not found:\n{SRC_DIR}"
    )


sys.path.insert(
    0,
    str(SRC_DIR)
)


try:

    import pipeline

except Exception as e:

    print(
        "\nERROR importing pipeline.py:"
    )

    print(e)

    sys.exit(1)


# ------------------------------------------------------------
# Get required functions
# ------------------------------------------------------------

extract_text = pipeline.extract_text

create_chunks = pipeline.create_chunks

predict_chunk = pipeline.predict_chunk


print(
    "✓ pipeline.py imported"
)

print(
    "✓ OCR function loaded"
)

print(
    "✓ chunking function loaded"
)

print(
    "✓ DistilBERT prediction function loaded"
)

print(
    "✓ Semantic evidence will NOT be executed"
)


# ============================================================
# 4. DATASET CHECK
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
        and p.suffix.lower()
        in IMAGE_EXTENSIONS
    ]
)


print(
    "Test images:",
    len(image_files)
)


if not image_files:

    raise RuntimeError(
        "No test images found."
    )


# ============================================================
# 5. GROUND TRUTH
# ============================================================

def read_ground_truth(
    label_path
):

    """
    Read YOLO annotation file.

    Returns image-level set of classes.

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

            parts = (
                line.strip()
                .split()
            )


            if len(parts) != 5:

                continue


            try:

                class_id = int(
                    parts[0]
                )

            except ValueError:

                continue


            if class_id in CLASS_NAMES:

                ground_truth.add(
                    CLASS_NAMES[class_id]
                )


    return ground_truth


# ============================================================
# 6. STATISTICS
# ============================================================

statistics = {

    label: {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0
    }

    for label in CLASS_LABELS
}


# ============================================================
# 7. DETAILED RESULTS
# ============================================================

detailed_results = []


# ============================================================
# 8. PROCESS TEST IMAGES
# ============================================================

print("\n[3] RUNNING DISTILBERT CLASSIFIER")
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


    ground_truth = (
        read_ground_truth(
            label_path
        )
    )


    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    try:

        raw_text = extract_text(
            str(image_path)
        )

    except Exception as e:

        print(
            f"  OCR ERROR: {e}"
        )

        detailed_results.append({

            "image": image_path.name,

            "ground_truth": sorted(
                ground_truth
            ),

            "ocr_text": "",

            "chunks": [],

            "predictions": {},

            "error": str(e)

        })

        continue


    if not raw_text or not raw_text.strip():

        print(
            "  No OCR text detected."
        )

        detailed_results.append({

            "image": image_path.name,

            "ground_truth": sorted(
                ground_truth
            ),

            "ocr_text": "",

            "chunks": [],

            "predictions": {},

            "error": "No OCR text"

        })

        # No predictions
        predicted_labels = set()


        for label in CLASS_LABELS:

            actual = (
                label in ground_truth
            )

            predicted = False


            if actual:

                statistics[label]["FN"] += 1

            else:

                statistics[label]["TN"] += 1


        continue


    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    try:

        chunks = create_chunks(
            raw_text
        )

    except Exception as e:

        print(
            f"  CHUNKING ERROR: {e}"
        )

        detailed_results.append({

            "image": image_path.name,

            "ground_truth": sorted(
                ground_truth
            ),

            "ocr_text": raw_text,

            "chunks": [],

            "predictions": {},

            "error": str(e)

        })

        continue


    print(
        f"  OCR characters: {len(raw_text)}"
    )

    print(
        f"  Chunks: {len(chunks)}"
    )


    # --------------------------------------------------------
    # DistilBERT predictions
    # --------------------------------------------------------

    chunk_predictions = []


    # Keep strongest confidence for each class
    class_predictions = {}


    for chunk_index, chunk in enumerate(
        chunks,
        start=1
    ):

        if not chunk or not chunk.strip():

            continue


        try:

            predicted_label, confidence = (
                predict_chunk(
                    chunk
                )
            )

        except Exception as e:

            print(
                f"  Prediction error on chunk "
                f"{chunk_index}: {e}"
            )

            continue


        confidence = float(
            confidence
        )


        chunk_result = {

            "chunk_index":
                chunk_index,

            "text":
                chunk,

            "predicted_label":
                predicted_label,

            "confidence":
                confidence
        }


        chunk_predictions.append(
            chunk_result
        )


        # ----------------------------------------------------
        # Keep strongest prediction per class
        # ----------------------------------------------------

        if predicted_label in CLASS_LABELS:

            if (
                predicted_label
                not in class_predictions
            ):

                class_predictions[
                    predicted_label
                ] = {

                    "confidence":
                        confidence,

                    "supporting_chunks":
                        1,

                    "best_text":
                        chunk
                }

            else:

                existing = (
                    class_predictions[
                        predicted_label
                    ]
                )


                existing[
                    "supporting_chunks"
                ] += 1


                if (
                    confidence
                    >
                    existing[
                        "confidence"
                    ]
                ):

                    existing[
                        "confidence"
                    ] = confidence

                    existing[
                        "best_text"
                    ] = chunk


    # --------------------------------------------------------
    # Image-level prediction
    # --------------------------------------------------------

    predicted_labels = set(
        class_predictions.keys()
    )


    # --------------------------------------------------------
    # Print strongest predictions
    # --------------------------------------------------------

    print(
        "  Image predictions:"
    )


    if class_predictions:

        for label, info in sorted(
            class_predictions.items(),
            key=lambda x: x[1]["confidence"],
            reverse=True
        ):

            print(
                f"    {label:<25}"
                f"CONF={info['confidence'] * 100:6.2f}% "
                f"chunks={info['supporting_chunks']}"
            )

    else:

        print(
            "    None"
        )


    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    for label in CLASS_LABELS:

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
    # Primary prediction
    # --------------------------------------------------------

    if class_predictions:

        primary_label, primary_info = max(
            class_predictions.items(),
            key=lambda x:
                x[1]["confidence"]
        )

        primary_confidence = (
            primary_info["confidence"]
        )

    else:

        primary_label = None

        primary_confidence = 0.0


    # --------------------------------------------------------
    # Save detailed result
    # --------------------------------------------------------

    detailed_results.append({

        "image":
            image_path.name,

        "ground_truth":
            sorted(ground_truth),

        "ocr_text":
            raw_text,

        "total_chunks":
            len(chunks),

        "dark_pattern_detected":
            bool(predicted_labels),

        "total_patterns_detected":
            len(predicted_labels),

        "primary_pattern":
            primary_label,

        "primary_confidence":
            primary_confidence,

        "predictions":
            class_predictions,

        "chunk_predictions":
            chunk_predictions

    })


# ============================================================
# 9. METRIC FUNCTIONS
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
# 10. PER-CLASS METRICS
# ============================================================

metrics = {}


for label in CLASS_LABELS:

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
# 11. MICRO AVERAGE
# ============================================================

total_tp = sum(
    statistics[label]["TP"]
    for label in CLASS_LABELS
)

total_fp = sum(
    statistics[label]["FP"]
    for label in CLASS_LABELS
)

total_fn = sum(
    statistics[label]["FN"]
    for label in CLASS_LABELS
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


# ============================================================
# 12. MACRO AVERAGE
# ============================================================

macro_precision = sum(
    metrics[label]["precision"]
    for label in CLASS_LABELS
) / len(CLASS_LABELS)


macro_recall = sum(
    metrics[label]["recall"]
    for label in CLASS_LABELS
) / len(CLASS_LABELS)


macro_f1 = sum(
    metrics[label]["f1"]
    for label in CLASS_LABELS
) / len(CLASS_LABELS)


# ============================================================
# 13. PRINT REPORT
# ============================================================

print("\n")
print("=" * 75)
print("DISTILBERT TEXT MODEL TEST REPORT")
print("=" * 75)


print(
    f"{'Class':<28}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'Support':>10}"
)

print("-" * 75)


for label in CLASS_LABELS:

    m = metrics[label]

    print(
        f"{label:<28}"
        f"{m['precision']:>12.4f}"
        f"{m['recall']:>12.4f}"
        f"{m['f1']:>12.4f}"
        f"{m['support']:>10}"
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
# 14. SAVE METRICS JSON
# ============================================================

metrics_output = {

    "model":
        "DistilBERT classifier",

    "evaluation_type":
        "classifier_only",

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
    / "text_metrics.json"
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
# 15. SAVE DETAILED PREDICTIONS
# ============================================================

details_json = (
    OUTPUT_DIR
    / "text_predictions.json"
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
# 16. SAVE CSV REPORT
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "text_metrics.csv"
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


    for label in CLASS_LABELS:

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
# 17. FINAL
# ============================================================

print("\n")
print("=" * 75)
print("TEXT MODEL EVALUATION COMPLETED")
print("=" * 75)


print(
    "\nMetrics JSON:"
)

print(
    metrics_json
)


print(
    "\nDetailed predictions:"
)

print(
    details_json
)


print(
    "\nCSV report:"
)

print(
    csv_path
)


print("\n✓ DistilBERT model evaluated.")

print("✓ Semantic evidence was NOT executed.")

print("✓ pipeline.py was NOT modified.")

print("✓ Test images were NOT modified.")

print("✓ Test labels were NOT modified.")