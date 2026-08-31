from ultralytics import YOLO
from pathlib import Path
import json
import sys
import torch


# ============================================================
# DARPAN YOLO STRUCTURED PREDICTION
# ============================================================


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

MODEL_PATH = Path(
    r"D:\DARPAN\runs\darpan_yolov8n-2\weights\best.pt"
)

CLASS_NAMES = [
    "hidden_fees",
    "false_urgency",
    "sneaking",
    "misdirection",
    "interface_interference",
    "obstruction"
]


# ------------------------------------------------------------
# LOAD YOLO MODEL
# ------------------------------------------------------------

print("=" * 70)
print("DARPAN YOLO STRUCTURED PREDICTION")
print("=" * 70)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found:\n{MODEL_PATH}"
    )

print("\nLoading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully.")


# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

if torch.cuda.is_available():

    DEVICE = 0
    print(
        "Device:",
        torch.cuda.get_device_name(0)
    )

else:

    DEVICE = "cpu"
    print("Device: CPU")


# ------------------------------------------------------------
# STRUCTURED PREDICTION FUNCTION
# ------------------------------------------------------------

def predict_image(image_path, conf_threshold=0.25):

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # RUN YOLO PREDICTION
    # --------------------------------------------------------

    results = model.predict(

        source=str(image_path),

        conf=conf_threshold,

        device=DEVICE,

        verbose=False
    )

    result = results[0]

    predictions = []

    # --------------------------------------------------------
    # EXTRACT DETECTIONS
    # --------------------------------------------------------

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0].cpu().numpy()
            )

            confidence = float(
                box.conf[0].cpu().numpy()
            )

            xyxy = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .tolist()
            )

            prediction = {

                "label": CLASS_NAMES[class_id],

                "class_id": class_id,

                "confidence": round(
                    confidence,
                    4
                ),

                "confidence_percent": round(
                    confidence * 100,
                    2
                ),

                "bounding_box": {

                    "x1": round(xyxy[0], 2),

                    "y1": round(xyxy[1], 2),

                    "x2": round(xyxy[2], 2),

                    "y2": round(xyxy[3], 2)
                }
            }

            predictions.append(
                prediction
            )

    # --------------------------------------------------------
    # SORT BY CONFIDENCE
    # --------------------------------------------------------

    predictions = sorted(

        predictions,

        key=lambda x: x["confidence"],

        reverse=True
    )

    # --------------------------------------------------------
    # BUILD STRUCTURED RESULT
    # --------------------------------------------------------

    structured_result = {

        "image_path": str(
            image_path
        ),

        "visual_model": "YOLOv8n",

        "dark_pattern_detected": (
            len(predictions) > 0
        ),

        "total_detections": len(
            predictions
        ),

        "primary_pattern": (

            predictions[0]["label"]

            if predictions

            else None
        ),

        "primary_confidence": (

            predictions[0]["confidence"]

            if predictions

            else 0.0
        ),

        "patterns": predictions
    }

    return structured_result


# ============================================================
# COMMAND LINE EXECUTION
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python scripts/yolo_predict.py <image_path>"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    try:

        result = predict_image(
            image_path
        )

        print("\n" + "=" * 70)
        print("YOLO STRUCTURED OUTPUT")
        print("=" * 70)

        print(

            json.dumps(

                result,

                indent=4,

                ensure_ascii=False
            )
        )

    except Exception as e:

        print("\n" + "=" * 70)
        print("DARPAN YOLO ERROR")
        print("=" * 70)

        print(
            "\n",
            str(e)
        )

        sys.exit(1)