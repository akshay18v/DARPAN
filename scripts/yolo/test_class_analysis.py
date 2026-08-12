from pathlib import Path
from collections import Counter


# ============================================================
# DARPAN TEST CLASS DISTRIBUTION ANALYSIS
# ============================================================

print("=" * 70)
print("DARPAN TEST SET CLASS ANALYSIS")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

LABEL_DIR = Path(
    r"D:\DARPAN\data\yolo_dataset\labels\test"
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
# CHECK DIRECTORY
# ============================================================

if not LABEL_DIR.exists():
    raise FileNotFoundError(
        f"Test label directory not found:\n{LABEL_DIR}"
    )


# ============================================================
# READ LABELS
# ============================================================

class_counts = Counter()

label_files = sorted(LABEL_DIR.glob("*.txt"))

print("\nTest label files:", len(label_files))


for label_file in label_files:

    with open(label_file, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            class_id = int(parts[0])

            class_counts[class_id] += 1


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("GROUND-TRUTH BOX DISTRIBUTION")
print("=" * 70)

total = 0

for class_id, class_name in CLASS_NAMES.items():

    count = class_counts[class_id]

    total += count

    print(
        f"{class_id} - "
        f"{class_name:<25} "
        f"{count:>4}"
    )


print("-" * 50)
print("Total bounding boxes:", total)


# ============================================================
# SAVE REPORT
# ============================================================

OUTPUT_FILE = Path(
    r"D:\DARPAN\runs\test_predictions\ground_truth_distribution.txt"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    file.write("DARPAN TEST SET GROUND TRUTH DISTRIBUTION\n")
    file.write("=" * 60 + "\n\n")

    for class_id, class_name in CLASS_NAMES.items():

        file.write(
            f"{class_id} - "
            f"{class_name}: "
            f"{class_counts[class_id]}\n"
        )

    file.write(
        f"\nTotal bounding boxes: {total}\n"
    )


print("\n✓ Analysis completed.")
print("Report:", OUTPUT_FILE)