from ultralytics import YOLO
from pathlib import Path
import torch
import shutil


# ============================================================
# DARPAN TEST SET PREDICTION ANALYSIS
# ============================================================

print("=" * 70)
print("DARPAN YOLOv8n TEST SET PREDICTION ANALYSIS")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(
    r"D:\DARPAN\runs\darpan_yolov8n\weights\best.pt"
)

TEST_IMAGES = Path(
    r"D:\DARPAN\data\yolo_dataset\images\test"
)

OUTPUT_DIR = Path(
    r"D:\DARPAN\runs\test_predictions"
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
# DEVICE
# ============================================================

print("\n[1] DEVICE")

if torch.cuda.is_available():
    DEVICE = 0
    print("CUDA available : True")
    print("GPU             :", torch.cuda.get_device_name(0))
else:
    DEVICE = "cpu"
    print("CUDA available : False")
    print("Using CPU")


# ============================================================
# CHECK MODEL
# ============================================================

print("\n[2] MODEL")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

print("Model:", MODEL_PATH)


# ============================================================
# CHECK TEST IMAGES
# ============================================================

print("\n[3] TEST IMAGES")

if not TEST_IMAGES.exists():
    raise FileNotFoundError(
        f"Test image directory not found:\n{TEST_IMAGES}"
    )

image_files = []

for extension in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
    image_files.extend(TEST_IMAGES.glob(extension))

image_files = sorted(image_files)

print("Test images found:", len(image_files))

if len(image_files) == 0:
    raise RuntimeError("No test images found.")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

print("\n[4] OUTPUT DIRECTORY")

if OUTPUT_DIR.exists():
    print("Removing previous prediction results...")
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("Saving predictions to:")
print(OUTPUT_DIR)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n[5] LOADING MODEL")

model = YOLO(str(MODEL_PATH))

print("✓ Model loaded successfully")


# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\n[6] RUNNING PREDICTIONS")

print("Confidence threshold : 0.25")
print("Image size           : 640")
print("Device               :", DEVICE)

results = model.predict(
    source=str(TEST_IMAGES),
    imgsz=640,
    conf=0.25,
    iou=0.7,
    device=DEVICE,
    save=True,
    save_txt=True,
    save_conf=True,
    project=str(OUTPUT_DIR.parent),
    name=OUTPUT_DIR.name,
    exist_ok=True,
    verbose=True,
)


# ============================================================
# DISPLAY DETECTION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION SUMMARY")
print("=" * 70)

total_detections = 0

class_counts = {
    class_id: 0
    for class_id in CLASS_NAMES
}

for result in results:

    boxes = result.boxes

    if boxes is None:
        continue

    count = len(boxes)

    total_detections += count

    for cls in boxes.cls:

        class_id = int(cls.item())

        if class_id in class_counts:
            class_counts[class_id] += 1


print("\nTotal detections:", total_detections)

print("\nPredictions by class:")

for class_id, class_name in CLASS_NAMES.items():

    print(
        f"{class_id} - "
        f"{class_name:<25} "
        f"{class_counts[class_id]}"
    )


# ============================================================
# SAVE SIMPLE REPORT
# ============================================================

REPORT_FILE = OUTPUT_DIR / "prediction_summary.txt"

with open(REPORT_FILE, "w", encoding="utf-8") as file:

    file.write("DARPAN TEST SET PREDICTION SUMMARY\n")
    file.write("=" * 60 + "\n\n")

    file.write(
        f"Test images: {len(image_files)}\n"
    )

    file.write(
        f"Total detections: {total_detections}\n\n"
    )

    file.write("Predictions by class:\n")

    for class_id, class_name in CLASS_NAMES.items():

        file.write(
            f"{class_id} - "
            f"{class_name}: "
            f"{class_counts[class_id]}\n"
        )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("TEST PREDICTION ANALYSIS COMPLETED")
print("=" * 70)

print("\nPredicted images:")
print(OUTPUT_DIR)

print("\nPrediction labels:")
print(OUTPUT_DIR / "labels")

print("\nSummary:")
print(REPORT_FILE)

print("\n✓ Original test images were NOT modified.")
print("✓ Dataset was NOT modified.")
print("✓ Model was NOT retrained.")
print("✓ Predictions were generated using best.pt.")