import os
from collections import defaultdict

# ===========================
# CHANGE THIS PATH
# ===========================
LABEL_FOLDER = r"D:\DARPAN\data\annotations\labels"

CLASS_NAMES = {
    0: "hidden_fees",
    1: "false_urgency",
    2: "sneaking",
    3: "misdirection",
    4: "interface_interference",
    5: "obstruction"
}

bbox_per_class = defaultdict(int)
images_per_class = defaultdict(int)

total_images = 0
negative_images = 0

for filename in os.listdir(LABEL_FOLDER):

    if not filename.endswith(".txt"):
        continue

    total_images += 1

    filepath = os.path.join(LABEL_FOLDER, filename)

    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Empty annotation file
    if len(lines) == 0:
        negative_images += 1
        continue

    classes_present = set()

    for line in lines:

        values = line.split()

        # Skip invalid rows
        if len(values) != 5:
            continue

        cls = int(values[0])

        bbox_per_class[cls] += 1
        classes_present.add(cls)

    for cls in classes_present:
        images_per_class[cls] += 1


print("\n================ DATASET SUMMARY ================\n")

print(f"Total Images           : {total_images}")
print(f"Negative Images        : {negative_images}")
print(f"Images with Objects    : {total_images-negative_images}")

print("\n================ CLASS STATISTICS ================\n")

print(f"{'Class':30s}{'Images':>10s}{'Boxes':>10s}")

total_boxes = 0

for cls in sorted(CLASS_NAMES.keys()):

    img = images_per_class[cls]
    box = bbox_per_class[cls]

    total_boxes += box

    print(f"{CLASS_NAMES[cls]:30s}{img:10d}{box:10d}")

print("\n===============================================")
print(f"Total Bounding Boxes : {total_boxes}")
