import sys
import platform
import importlib.util

print("=" * 70)
print("DARPAN YOLOv8n TRAINING ENVIRONMENT CHECK")
print("=" * 70)

# ------------------------------------------------------------
# Python
# ------------------------------------------------------------

print("\n[1] PYTHON")

print("Python version :", sys.version)
print("Platform       :", platform.platform())

# ------------------------------------------------------------
# PyTorch
# ------------------------------------------------------------

print("\n[2] PYTORCH")

torch_spec = importlib.util.find_spec("torch")

if torch_spec is None:
    print("✗ PyTorch is NOT installed")
else:
    import torch

    print("✓ PyTorch installed")
    print("PyTorch version :", torch.__version__)

    print("CUDA available  :", torch.cuda.is_available())

    if torch.cuda.is_available():

        print("CUDA version    :", torch.version.cuda)
        print("GPU count       :", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):

            print(
                f"GPU {i}          : "
                f"{torch.cuda.get_device_name(i)}"
            )

    else:

        print("CUDA version    :", torch.version.cuda)
        print("GPU             : Not available")
        print("Training will use CPU if no GPU is configured.")


# ------------------------------------------------------------
# Ultralytics
# ------------------------------------------------------------

print("\n[3] ULTRALYTICS")

ultralytics_spec = importlib.util.find_spec("ultralytics")

if ultralytics_spec is None:

    print("✗ Ultralytics is NOT installed")

else:

    import ultralytics

    print("✓ Ultralytics installed")
    print("Ultralytics version :", ultralytics.__version__)


# ------------------------------------------------------------
# GPU recommendation
# ------------------------------------------------------------

print("\n[4] TRAINING DEVICE")

if torch_spec is not None:

    import torch

    if torch.cuda.is_available():

        print("✓ CUDA GPU detected")
        print("Recommended device: 0")

    else:

        print("⚠ CUDA GPU not detected")
        print("Recommended device: cpu")


# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ENVIRONMENT CHECK COMPLETE")
print("=" * 70)