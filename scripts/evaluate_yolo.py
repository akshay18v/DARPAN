from ultralytics import YOLO

MODEL_PATH = r"D:\DARPAN\runs\darpan_yolov8n-2\weights\best.pt"
DATA_YAML = r"D:\DARPAN\data\yolo_dataset\data.yaml"

model = YOLO(MODEL_PATH)

metrics = model.val(
    data=DATA_YAML,
    split="test",
    imgsz=640,
    batch=4,
    device="cpu",
    plots=True
)

print("\n" + "=" * 70)
print("YOLOv8n TEST EVALUATION")
print("=" * 70)

print("\nOverall:")
print("Precision :", metrics.box.mp)
print("Recall    :", metrics.box.mr)
print("mAP50     :", metrics.box.map50)
print("mAP50-95  :", metrics.box.map)

print("\nPer-class:")
print("-" * 70)

for i, name in enumerate(model.names.values()):

    print(
        f"{name:<25} "
        f"P={metrics.box.p[i]:.4f}  "
        f"R={metrics.box.r[i]:.4f}  "
        f"mAP50={metrics.box.ap50[i]:.4f}  "
        f"mAP50-95={metrics.box.ap[i]:.4f}"
    )