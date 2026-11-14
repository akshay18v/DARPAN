from ultralytics import YOLO
import torch
from pathlib import Path

print("=" * 70)
print("DARPAN YOLOv8n TEST SET EVALUATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. GPU
# ------------------------------------------------------------

print("\n[1] DEVICE")

if torch.cuda.is_available():
    print("CUDA available : True")
    print("GPU             :", torch.cuda.get_device_name(0))
    DEVICE = 0
else:
    print("CUDA available : False")
    print("Using CPU")
    DEVICE = "cpu"

# ------------------------------------------------------------
# 2. PATHS
# ------------------------------------------------------------

MODEL_PATH = r"D:\DARPAN\runs\darpan_yolov8n\weights\best.pt"
DATA_YAML = r"D:\DARPAN\data\yolo_dataset\data.yaml"

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

if not Path(DATA_YAML).exists():
    raise FileNotFoundError(f"data.yaml not found:\n{DATA_YAML}")

print("\n[2] FILES")

print("Model  :", MODEL_PATH)
print("Dataset:", DATA_YAML)

# ------------------------------------------------------------
# 3. LOAD MODEL
# ------------------------------------------------------------

print("\n[3] LOADING BEST MODEL")

model = YOLO(MODEL_PATH)

print("✓ best.pt loaded successfully")

# ------------------------------------------------------------
# 4. TEST SET EVALUATION
# ------------------------------------------------------------

print("\n[4] TEST SET EVALUATION")
print("Evaluating ONLY the untouched test split.")
print("=" * 70)

results = model.val(
    data=DATA_YAML,
    split="test",
    imgsz=640,
    batch=4,
    device=DEVICE,
    workers=0,
    plots=True,
    save_json=True,
    project=r"D:\DARPAN\runs",
    name="test_evaluation",
    exist_ok=False,
    verbose=True,
)


# ------------------------------------------------------------
# 5. RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TEST SET EVALUATION COMPLETED")
print("=" * 70)

# Dynamically count test images
TEST_IMAGES_DIR = Path(
    r"D:\DARPAN\data\yolo_dataset\images\test"
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

test_images = [
    p for p in TEST_IMAGES_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"\nTest images: {len(test_images)}")

print("\nOverall metrics:")

print(f"Precision : {results.box.mp:.4f}")
print(f"Recall    : {results.box.mr:.4f}")
print(f"mAP50     : {results.box.map50:.4f}")
print(f"mAP50-95  : {results.box.map:.4f}")

print("\nResults saved to:")
print(results.save_dir)

print("\n✓ Test evaluation completed.")
print("✓ The model was NOT retrained.")
print("✓ Test set remained separate from training.")
