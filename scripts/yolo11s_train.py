from ultralytics import YOLO
import torch
from pathlib import Path

# ============================================================
# DARPAN YOLO11s  REAL TRAINING
# ============================================================

print("=" * 70)
print("DARPAN YOLO11s  REAL TRAINING")
print("=" * 70)

# ------------------------------------------------------------
# 1. GPU CHECK
# ------------------------------------------------------------

print("\n[1] GPU CHECK")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Training requires the GPU.")

gpu_name = torch.cuda.get_device_name(0)

print("CUDA available :", True)
print("GPU             :", gpu_name)
print("CUDA version    :", torch.version.cuda)

# ------------------------------------------------------------
# 2. PATHS
# ------------------------------------------------------------

DATA_YAML = r"D:\DARPAN\data\yolo_dataset\data.yaml"
PROJECT_DIR = r"D:\DARPAN\runs"

data_path = Path(DATA_YAML)

if not data_path.exists():
    raise FileNotFoundError(
        f"data.yaml not found:\n{DATA_YAML}"
    )

print("\n[2] DATASET")
print("Dataset YAML:", DATA_YAML)

# ------------------------------------------------------------
# 3. LOAD PRETRAINED YOLO11s
# ------------------------------------------------------------

print("\n[3] LOADING yolo11s")

model = YOLO("yolo11s.pt")

print("✓ yolo11s loaded successfully")

# ------------------------------------------------------------
# 4. TRAINING CONFIGURATION
# ------------------------------------------------------------

print("\n[4] TRAINING CONFIGURATION")

print("Epochs       : 100")
print("Image size   : 640")
print("Batch size   : 4")
print("Device       : 0")
print("Workers      : 0")
print("AMP          : False")
print("Patience     : 20")
print("Seed         : 42")

# ------------------------------------------------------------
# 5. REAL TRAINING
# ------------------------------------------------------------

print("\n[5] STARTING REAL TRAINING")
print("=" * 70)

results = model.train(
    # Dataset
    data=DATA_YAML,

    # Training duration
    epochs=100,

    # Image size
    imgsz=640,

    # GTX 1650 4GB
    batch=4,

    # GPU
    device=0,

    # Windows stability
    workers=0,

    # Disable AMP because GTX 1650 smoke test warned about AMP
    amp=False,

    # Reproducibility
    seed=42,

    # Early stopping
    patience=20,

    # Pretrained model
    pretrained=True,

    # Standard augmentation
    mosaic=1.0,

    # Save checkpoints
    save=True,
    save_period=10,

    # Validation during training
    val=True,

    # Generate training plots
    plots=True,

    # Store results
    project=PROJECT_DIR,
    name="darpan_yolo11s",

    # Do not overwrite an existing run
    exist_ok=False,

    # Cache disabled to reduce RAM usage
    cache=False,

    # Verbose output
    verbose=True,
)

# ------------------------------------------------------------
# 6. TRAINING COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print("\nTraining results saved to:")
print(r"D:\DARPAN\runs\darpan_yolo11s")

print("\nBest model:")
print(r"D:\DARPAN\runs\darpan_yolo11s\weights\best.pt")

print("\nLast model:")
print(r"D:\DARPAN\runs\darpan_yolo11s\weights\last.pt")

print("\n✓ YOLOv11s training completed.")
print("✓ Best weights were saved.")
print("✓ Training/validation metrics were generated.")