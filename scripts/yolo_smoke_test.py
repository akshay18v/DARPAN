from ultralytics import YOLO
import torch

print("=" * 70)
print("DARPAN YOLOv8n SMOKE TEST")
print("=" * 70)

# ------------------------------------------------------------
# GPU CHECK
# ------------------------------------------------------------

print("\n[1] GPU")

print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("GPU:", torch.cuda.get_device_name(0))

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("\n[2] LOADING YOLOv8n")

model = YOLO("yolov8n.pt")

print("✓ YOLOv8n loaded successfully")

# ------------------------------------------------------------
# DATASET CHECK
# ------------------------------------------------------------

print("\n[3] DATASET")

DATA_YAML = r"D:\DARPAN\data\yolo_dataset\data.yaml"

print("Dataset:", DATA_YAML)

# ------------------------------------------------------------
# SHORT TRAINING TEST
# ------------------------------------------------------------

print("\n[4] RUNNING SMOKE TEST")

print("This is NOT the real training.")
print("Running only 2 epochs with a small image size.")

results = model.train(
    data=DATA_YAML,

    # Very short test
    epochs=2,

    # GTX 1650 4 GB
    imgsz=640,
    batch=4,

    # GPU
    device=0,

    # Small dataloader workload for Windows
    workers=0,

    # Don't perform extensive augmentation during smoke test
    mosaic=0.0,

    # Save test run separately
    project=r"D:\DARPAN\runs",
    name="smoke_test",

    # Reproducibility
    seed=42,

    # Cache disabled to avoid unnecessary RAM usage
    cache=False,

    # Save model
    save=True,

    verbose=True
)

print("\n" + "=" * 70)
print("SMOKE TEST COMPLETED")
print("=" * 70)

print("\n✓ YOLOv8n successfully trained for the test run.")
print("✓ Dataset was accepted.")
print("✓ CUDA training worked.")
print("\nIf there were no CUDA/label/dataset errors,")
print("we are ready for the real training.")