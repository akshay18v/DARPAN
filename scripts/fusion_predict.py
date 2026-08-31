from pathlib import Path
import sys
import json
import contextlib
import io

from ultralytics import YOLO


# ============================================================
# DARPAN MULTIMODAL FUSION
# YOLOv8n + DistilBERT/OCR TEXT PIPELINE
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\DARPAN")

YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "darpan_yolov8n-2"
    / "weights"
    / "best.pt"
)

SRC_DIR = PROJECT_ROOT / "src"

CLASS_NAMES = [
    "hidden_fees",
    "false_urgency",
    "sneaking",
    "misdirection",
    "interface_interference",
    "obstruction",
]

# ------------------------------------------------------------
# YOLO candidate threshold
# ------------------------------------------------------------
#
# Keep this relatively low because fusion will make the
# final decision.
#
YOLO_CONF_THRESHOLD = 0.10


# ============================================================
# 2. CHECK PATHS
# ============================================================

if not YOLO_MODEL_PATH.exists():

    raise FileNotFoundError(
        f"YOLO model not found:\n{YOLO_MODEL_PATH}"
    )

if not SRC_DIR.exists():

    raise FileNotFoundError(
        f"Source directory not found:\n{SRC_DIR}"
    )


# ============================================================
# 3. IMPORT TEXT PIPELINE
# ============================================================

sys.path.insert(
    0,
    str(SRC_DIR)
)

print("=" * 75)
print("DARPAN MULTIMODAL FUSION")
print("YOLOv8n + DistilBERT TEXT PIPELINE")
print("=" * 75)

print("\nLoading text pipeline...")

try:

    from pipeline import analyze_image

except Exception as e:

    raise RuntimeError(
        "Could not import src\\pipeline.py\n\n"
        f"Original error:\n{e}"
    )

print("✓ Text pipeline loaded")


# ============================================================
# 4. LOAD YOLO
# ============================================================

print("\nLoading YOLO model...")

yolo_model = YOLO(
    str(YOLO_MODEL_PATH)
)

print("✓ YOLO model loaded")

print(
    "YOLO model:",
    YOLO_MODEL_PATH
)


# ============================================================
# 5. RUN YOLO
# ============================================================

def run_yolo(image_path):

    results = yolo_model.predict(
        source=str(image_path),
        conf=YOLO_CONF_THRESHOLD,
        device="cpu",
        verbose=False
    )

    result = results[0]

    predictions = {}

    if result.boxes is None:
        return predictions

    for box in result.boxes:

        class_id = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        if class_id < 0 or class_id >= len(CLASS_NAMES):
            continue

        label = CLASS_NAMES[class_id]

        # Keep strongest prediction for each class
        if (
            label not in predictions
            or confidence > predictions[label]["confidence"]
        ):

            xyxy = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .tolist()
            )

            predictions[label] = {

                "class_id": class_id,

                "confidence": confidence,

                "confidence_percent":
                    confidence * 100,

                "bounding_box": {

                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3])
                }
            }

    return predictions


# ============================================================
# 6. EXTRACT TEXT PREDICTIONS
# ============================================================

def extract_text_predictions(
    text_result
):

    predictions = {}

    if not isinstance(
        text_result,
        dict
    ):

        return predictions

    patterns = text_result.get(
        "patterns",
        []
    )

    if not isinstance(
        patterns,
        list
    ):

        return predictions

    for pattern in patterns:

        if not isinstance(
            pattern,
            dict
        ):

            continue

        label = pattern.get(
            "label"
        )

        if label not in CLASS_NAMES:
            continue

        confidence = float(
            pattern.get(
                "confidence",
                0.0
            ) or 0.0
        )

        evidence_score = float(
            pattern.get(
                "evidence_score",
                0.0
            ) or 0.0
        )

        supporting_chunks = int(
            pattern.get(
                "supporting_chunks",
                0
            ) or 0
        )

        semantic_score = float(
            pattern.get(
                "best_semantic_score",
                0.0
            ) or 0.0
        )

        exact_evidence = pattern.get(
            "exact_evidence",
            []
        )

        fuzzy_evidence = pattern.get(
            "fuzzy_evidence",
            []
        )

        semantic_evidence = pattern.get(
            "semantic_evidence",
            []
        )

        supporting_text = pattern.get(
            "supporting_text",
            ""
        )

        prediction = {

            "confidence": confidence,

            "confidence_percent":
                confidence * 100,

            "evidence_score":
                evidence_score,

            "supporting_chunks":
                supporting_chunks,

            "best_semantic_score":
                semantic_score,

            "exact_evidence":
                exact_evidence,

            "fuzzy_evidence":
                fuzzy_evidence,

            "semantic_evidence":
                semantic_evidence,

            "supporting_text":
                supporting_text
        }

        # Keep strongest prediction if duplicated
        if (
            label not in predictions
            or confidence >
               predictions[label]["confidence"]
        ):

            predictions[label] = prediction

    return predictions


# ============================================================
# 7. FUSION DECISION
# ============================================================

def fuse_class(
    label,
    visual,
    text
):

    """
    Class-specific multimodal fusion.

    IMPORTANT:
    This is intentionally conservative.

    We do NOT simply do:

        YOLO OR TEXT

    because the text evaluation showed many false positives.
    """

    yolo_conf = 0.0

    if visual:

        yolo_conf = float(
            visual.get(
                "confidence",
                0.0
            )
        )

    text_conf = 0.0
    evidence = 0.0
    semantic = 0.0
    exact_count = 0
    fuzzy_count = 0
    semantic_count = 0
    supporting_chunks = 0

    if text:

        text_conf = float(
            text.get(
                "confidence",
                0.0
            )
        )

        evidence = float(
            text.get(
                "evidence_score",
                0.0
            )
        )

        semantic = float(
            text.get(
                "best_semantic_score",
                0.0
            )
        )

        exact_count = len(
            text.get(
                "exact_evidence",
                []
            )
            or []
        )

        fuzzy_count = len(
            text.get(
                "fuzzy_evidence",
                []
            )
            or []
        )

        semantic_count = len(
            text.get(
                "semantic_evidence",
                []
            )
            or []
        )

        supporting_chunks = int(
            text.get(
                "supporting_chunks",
                0
            )
        )

    yolo_present = visual is not None
    text_present = text is not None

    # --------------------------------------------------------
    # SNEAKING
    # --------------------------------------------------------
    #
    # YOLO is very strong for this class.
    # --------------------------------------------------------

    if label == "sneaking":

        if yolo_conf >= 0.60:

            return True, (
                "Strong YOLO visual detection"
            )

        if (
            yolo_conf >= 0.45
            and text_present
            and text_conf >= 0.70
        ):

            return True, (
                "YOLO detection supported by text model"
            )

        if (
            text_present
            and text_conf >= 0.85
            and evidence >= 3.0
        ):

            return True, (
                "Strong textual evidence for sneaking"
            )

        return False, "Insufficient evidence"

    # --------------------------------------------------------
    # OBSTRUCTION
    # --------------------------------------------------------
    #
    # YOLO is strong here as well.
    # --------------------------------------------------------

    if label == "obstruction":

        if yolo_conf >= 0.60:

            return True, (
                "Strong YOLO visual detection"
            )

        if (
            yolo_conf >= 0.45
            and text_present
            and text_conf >= 0.70
        ):

            return True, (
                "YOLO detection supported by text model"
            )

        if (
            text_present
            and text_conf >= 0.85
            and evidence >= 3.0
        ):

            return True, (
                "Strong textual evidence for obstruction"
            )

        return False, "Insufficient evidence"

    # --------------------------------------------------------
    # HIDDEN FEES
    # --------------------------------------------------------
    #
    # Text is useful because explicit fee language is strong.
    # --------------------------------------------------------

    if label == "hidden_fees":

        if (
            text_present
            and exact_count > 0
            and text_conf >= 0.75
            and evidence >= 3.0
        ):

            return True, (
                "Explicit textual fee evidence"
            )

        if yolo_conf >= 0.70:

            return True, (
                "Strong YOLO visual detection"
            )

        if (
            yolo_conf >= 0.45
            and text_present
            and text_conf >= 0.70
        ):

            return True, (
                "YOLO and text models agree"
            )

        return False, "Insufficient evidence"

    # --------------------------------------------------------
    # FALSE URGENCY
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # Text classifier has very poor precision here.
    #
    # Therefore confidence alone is NOT sufficient.
    # --------------------------------------------------------

    if label == "false_urgency":

        strong_text_evidence = (

            text_present

            and text_conf >= 0.75

            and evidence >= 4.0

            and (
                exact_count > 0
                or fuzzy_count > 0
                or semantic_count > 0
            )
        )

        if strong_text_evidence:

            return True, (
                "Strong textual urgency/scarcity evidence"
            )

        if (
            yolo_conf >= 0.60
            and text_present
            and text_conf >= 0.70
        ):

            return True, (
                "Strong YOLO detection supported by text"
            )

        return False, "Insufficient evidence"

    # --------------------------------------------------------
    # MISDIRECTION
    # --------------------------------------------------------
    #
    # Both models are relatively weak.
    # Require stronger evidence.
    # --------------------------------------------------------

    if label == "misdirection":

    # Strong YOLO evidence
        if yolo_conf >= 0.65:

            return True, (
                "Strong YOLO visual detection"
            )

        # Moderate YOLO evidence supported by text
        if (
            yolo_conf >= 0.45
            and text_present
            and text_conf >= 0.65
        ):

            return True, (
                "YOLO detection supported by text model"
            )

        # Strong textual evidence
        if (
            text_present
            and text_conf >= 0.85
            and evidence >= 3.0
            and semantic >= 0.82
        ):

            return True, (
                "Strong textual and semantic evidence"
            )

        return False, "Insufficient evidence"
    # --------------------------------------------------------
    # INTERFACE INTERFERENCE
    # --------------------------------------------------------
    #
    # Text has extremely high false-positive rate.
    # Be conservative.
    # --------------------------------------------------------

    if label == "interface_interference":

        # Strong YOLO evidence
        if yolo_conf >= 0.70:

            return True, (
                "Strong YOLO visual detection"
            )

        # Moderate YOLO evidence supported by text
        if (
            yolo_conf >= 0.50
            and text_present
            and text_conf >= 0.65
        ):

            return True, (
                "YOLO detection supported by text model"
            )

        # Strong textual evidence
        if (
            text_present
            and text_conf >= 0.90
            and evidence >= 4.0
            and (
                exact_count > 0
                or semantic >= 0.88
            )
        ):

            return True, (
                "Strong textual interface evidence"
            )

        return False, "Insufficient evidence"
    return False, "Unknown class"


# ============================================================
# 8. COMPLETE FUSION
# ============================================================

def fusion_predict(
    image_path
):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    print("\n" + "=" * 75)
    print("RUNNING YOLO")
    print("=" * 75)

    visual_predictions = run_yolo(
        image_path
    )

    print(
        "YOLO detections:",
        len(visual_predictions)
    )

    for label, info in visual_predictions.items():

        print(
            f"  {label:<25}"
            f"{info['confidence'] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # TEXT PIPELINE
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("RUNNING TEXT PIPELINE")
    print("=" * 75)

    # pipeline.py prints a lot of diagnostic output.
    # We suppress it here so the fusion result remains clean.

    captured_output = io.StringIO()

    try:

        with contextlib.redirect_stdout(
            captured_output
        ):

            text_result = analyze_image(
                str(image_path)
            )

    except Exception as e:

        raise RuntimeError(
            "Text pipeline failed:\n"
            f"{e}"
        )

    text_predictions = extract_text_predictions(
        text_result
    )

    print(
        "Text detections:",
        len(text_predictions)
    )

    for label, info in text_predictions.items():

        print(
            f"  {label:<25}"
            f"{info['confidence'] * 100:.2f}%"
            f" | evidence={info['evidence_score']:.2f}"
        )

    # --------------------------------------------------------
    # FUSION
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("FUSION")
    print("=" * 75)

    final_patterns = []

    for label in CLASS_NAMES:

        visual = visual_predictions.get(
            label
        )

        text = text_predictions.get(
            label
        )

        confirmed, reason = fuse_class(
            label,
            visual,
            text
        )

        if confirmed:

            # ------------------------------------------------
            # Determine final confidence
            # ------------------------------------------------

            yolo_conf = (
                visual["confidence"]
                if visual
                else 0.0
            )

            text_conf = (
                text["confidence"]
                if text
                else 0.0
            )

            # Evidence-aware confidence
            #
            # This is NOT a trained probability.
            # It is a fusion score used for ranking.

            if visual and text:

                final_confidence = (
                    0.50 * yolo_conf
                    + 0.50 * text_conf
                )

            elif visual:

                final_confidence = yolo_conf

            else:

                final_confidence = text_conf

            # Evidence bonus, capped
            if text:

                evidence_bonus = min(
                    0.10,
                    evidence_score_bonus(
                        text.get(
                            "evidence_score",
                            0.0
                        )
                    )
                )

                final_confidence = min(
                    1.0,
                    final_confidence
                    + evidence_bonus
                )

            final_patterns.append({

                "label": label,

                "confidence":
                    round(
                        final_confidence,
                        4
                    ),

                "confidence_percent":
                    round(
                        final_confidence * 100,
                        2
                    ),

                "decision":
                    "confirmed",

                "reason":
                    reason,

                "visual":
                    visual,

                "text":
                    text
            })

            print(
                f"✓ {label:<25}"
                f"{final_confidence * 100:.2f}%"
            )

            print(
                f"    Reason: {reason}"
            )

        else:

            print(
                f"✗ {label:<25} NOT CONFIRMED"
            )

    # --------------------------------------------------------
    # RANK FINAL PATTERNS
    # --------------------------------------------------------

    final_patterns.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    if final_patterns:

        primary_pattern = (
            final_patterns[0]["label"]
        )

        primary_confidence = (
            final_patterns[0]["confidence"]
        )

    else:

        primary_pattern = None
        primary_confidence = 0.0

    # --------------------------------------------------------
    # FINAL STRUCTURED OUTPUT
    # --------------------------------------------------------

    output = {

        "image_path":
            str(image_path),

        "visual_model":
            "YOLOv8n",

        "text_model":
            "DistilBERT + OCR + evidence",

        "dark_pattern_detected":
            len(final_patterns) > 0,

        "total_patterns_detected":
            len(final_patterns),

        "primary_pattern":
            primary_pattern,

        "primary_confidence":
            primary_confidence,

        "patterns":
            final_patterns,

        "visual_predictions":
            visual_predictions,

        "text_predictions":
            text_predictions
    }

    return output


# ============================================================
# 9. EVIDENCE BONUS
# ============================================================

def evidence_score_bonus(
    evidence_score
):

    """
    Small ranking bonus.

    Evidence score is not treated as probability.
    """

    evidence_score = float(
        evidence_score or 0.0
    )

    return min(
        0.10,
        evidence_score / 100.0
    )


# ============================================================
# 10. COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")

        print(
            r'python scripts\fusion_predict.py '
            r'"D:\DARPAN\data\yolo_dataset\images\test\false_ur_03.png"'
        )

        sys.exit(1)

    image_path = sys.argv[1]

    try:

        result = fusion_predict(
            image_path
        )

        print("\n" + "=" * 75)
        print("FINAL FUSED STRUCTURED OUTPUT")
        print("=" * 75)

        print(
            json.dumps(
                result,
                indent=4,
                ensure_ascii=False
            )
        )

    except Exception as e:

        print("\n" + "=" * 75)
        print("DARPAN FUSION ERROR")
        print("=" * 75)

        print(
            str(e)
        )

        sys.exit(1)