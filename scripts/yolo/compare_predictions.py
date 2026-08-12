from pathlib import Path
from collections import Counter


# ============================================================
# DARPAN GROUND TRUTH VS PREDICTION COMPARISON
# ============================================================

print("=" * 70)
print("DARPAN TEST SET GROUND TRUTH VS PREDICTIONS")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

GROUND_TRUTH_DIR = Path(
    r"D:\DARPAN\data\yolo_dataset\labels\test"
)

PREDICTION_DIR = Path(
    r"D:\DARPAN\runs\test_predictions\labels"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "hidden_fees",
    1: "false_urgency",
    2: "sneaking",
    3: "misdirection",
    4: "interface_interference",
    5: "obstruction",
}


# ============================================================
# CHECK DIRECTORIES
# ============================================================

if not GROUND_TRUTH_DIR.exists():
    raise FileNotFoundError(
        f"Ground truth directory not found:\n"
        f"{GROUND_TRUTH_DIR}"
    )

if not PREDICTION_DIR.exists():
    raise FileNotFoundError(
        f"Prediction directory not found:\n"
        f"{PREDICTION_DIR}\n\n"
        "Run analyze_test_predictions.py first."
    )


# ============================================================
# COUNT GROUND TRUTH
# ============================================================

ground_truth = Counter()

for file in GROUND_TRUTH_DIR.glob("*.txt"):

    with open(file, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            class_id = int(line.split()[0])

            ground_truth[class_id] += 1


# ============================================================
# COUNT PREDICTIONS
# ============================================================

predictions = Counter()

for file in PREDICTION_DIR.glob("*.txt"):

    with open(file, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            class_id = int(line.split()[0])

            predictions[class_id] += 1


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)

print(
    f"{'CLASS':<28}"
    f"{'GROUND TRUTH':>15}"
    f"{'PREDICTIONS':>15}"
)

print("-" * 58)

for class_id, class_name in CLASS_NAMES.items():

    gt = ground_truth[class_id]
    pred = predictions[class_id]

    print(
        f"{class_name:<28}"
        f"{gt:>15}"
        f"{pred:>15}"
    )


print("-" * 58)

print(
    f"{'TOTAL':<28}"
    f"{sum(ground_truth.values()):>15}"
    f"{sum(predictions.values()):>15}"
)


# ============================================================
# SAVE REPORT
# ============================================================

OUTPUT_FILE = Path(
    r"D:\DARPAN\runs\test_predictions\comparison.txt"
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    file.write(
        "DARPAN TEST SET GROUND TRUTH VS PREDICTIONS\n"
    )

    file.write("=" * 70 + "\n\n")

    for class_id, class_name in CLASS_NAMES.items():

        gt = ground_truth[class_id]
        pred = predictions[class_id]

        file.write(
            f"{class_name:<28}"
            f"GT={gt:<8}"
            f"Predictions={pred}\n"
        )

    file.write("\n")
    file.write(
        f"Total ground truth boxes: "
        f"{sum(ground_truth.values())}\n"
    )

    file.write(
        f"Total predictions: "
        f"{sum(predictions.values())}\n"
    )


print("\n✓ Comparison completed.")
print("Report:", OUTPUT_FILE)