# ============================================================
# DARPAN - OCR REGION CLASSIFIER
# ============================================================
#
# Pipeline:
# Screenshot
#     ↓
# Tesseract OCR
#     ↓
# Word-level bounding boxes
#     ↓
# Better line grouping
#     ↓
# DistilBERT classification
#     ↓
# Suspicious regions + coordinates
#     ↓
# JSON output
#
# ============================================================

import os
import sys
import json
import cv2
import torch
import torch.nn.functional as F
import pytesseract

from collections import defaultdict
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/distilbert"

OUTPUT_DIR = "models/distilbert"

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR,
    "region_analysis.json"
)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.60

# Ignore OCR text shorter than this
MIN_TEXT_LENGTH = 10

# OCR confidence threshold
OCR_CONFIDENCE_THRESHOLD = 40

# Maximum vertical distance for words to belong
# to the same text line
LINE_Y_TOLERANCE = 18

# Maximum horizontal gap between words
# before considering them separate regions
MAX_HORIZONTAL_GAP = 180


# ============================================================
# LABELS
# ============================================================

LABELS = [
    "false_urgency",
    "hidden_fees",
    "interface_interference",
    "misdirection",
    "normal",
    "obstruction",
    "sneaking"
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DARPAN - OCR REGION CLASSIFIER")
print("=" * 70)

print()
print(f"Device: {device}")

if device.type == "cuda":
    print("GPU available.")
else:
    print("CPU mode enabled.")


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print()
    print("ERROR: Trained DistilBERT model not found.")
    print(f"Expected location: {MODEL_PATH}")

    sys.exit(1)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print()
print("Loading tokenizer...")

tokenizer = DistilBertTokenizer.from_pretrained(
    MODEL_PATH
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained DistilBERT model...")

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=len(LABELS)
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# OCR WORD EXTRACTION
# ============================================================

def extract_ocr_words(image_path):

    """
    Extract individual OCR words together with:
        - text
        - confidence
        - x
        - y
        - width
        - height
    """

    image = cv2.imread(image_path)

    if image is None:

        raise ValueError(
            f"Unable to read image:\n{image_path}"
        )

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    gray = cv2.resize(
        gray,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    processed = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # --------------------------------------------------------
    # OCR DATA
    # --------------------------------------------------------

    data = pytesseract.image_to_data(
        processed,
        config="--psm 6",
        output_type=pytesseract.Output.DICT
    )

    words = []

    total = len(data["text"])

    for i in range(total):

        text = data["text"][i].strip()

        if not text:
            continue

        try:

            confidence = float(
                data["conf"][i]
            )

        except:

            confidence = 0

        if confidence < OCR_CONFIDENCE_THRESHOLD:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])

        w = int(data["width"][i])
        h = int(data["height"][i])

        words.append({
            "text": text,
            "confidence": confidence,
            "x": x,
            "y": y,
            "w": w,
            "h": h
        })

    return words


# ============================================================
# GROUP WORDS INTO LINES
# ============================================================

def group_words_into_lines(words):

    """
    Groups OCR words into meaningful text lines.

    Words are grouped primarily according to their vertical
    position. Large vertical differences create new lines.
    """

    if not words:
        return []

    # Sort top-to-bottom
    words = sorted(
        words,
        key=lambda item: (
            item["y"],
            item["x"]
        )
    )

    lines = []

    for word in words:

        word_center_y = (
            word["y"] +
            word["h"] / 2
        )

        best_line = None
        best_difference = float("inf")

        for line in lines:

            line_center_y = line["center_y"]

            difference = abs(
                word_center_y -
                line_center_y
            )

            if (
                difference <= LINE_Y_TOLERANCE
                and difference < best_difference
            ):

                best_line = line
                best_difference = difference

        # ----------------------------------------------------
        # Existing line
        # ----------------------------------------------------

        if best_line is not None:

            best_line["words"].append(word)

            # Update line center
            centers = [
                w["y"] + w["h"] / 2
                for w in best_line["words"]
            ]

            best_line["center_y"] = sum(
                centers
            ) / len(centers)

        # ----------------------------------------------------
        # New line
        # ----------------------------------------------------

        else:

            lines.append({
                "center_y": word_center_y,
                "words": [word]
            })

    # --------------------------------------------------------
    # Sort words inside every line
    # --------------------------------------------------------

    for line in lines:

        line["words"].sort(
            key=lambda item: item["x"]
        )

    return lines


# ============================================================
# SPLIT LONG LINES INTO REGIONS
# ============================================================

def split_line_into_regions(line):

    """
    A single OCR line can sometimes contain multiple
    physically separated UI regions.

    Example:

        "Average rating       Fees waived"

    should not necessarily become one region.
    """

    words = line["words"]

    if not words:
        return []

    regions = []

    current_region = [words[0]]

    for word in words[1:]:

        previous = current_region[-1]

        previous_right = (
            previous["x"] +
            previous["w"]
        )

        gap = (
            word["x"] -
            previous_right
        )

        # ----------------------------------------------------
        # Large gap -> new region
        # ----------------------------------------------------

        if gap > MAX_HORIZONTAL_GAP:

            regions.append(
                current_region
            )

            current_region = [word]

        else:

            current_region.append(word)

    regions.append(current_region)

    return regions


# ============================================================
# BUILD REGION
# ============================================================

def build_region(words):

    """
    Converts OCR words into a single region.
    """

    words = sorted(
        words,
        key=lambda item: item["x"]
    )

    text = " ".join(
        word["text"]
        for word in words
    )

    x1 = min(
        word["x"]
        for word in words
    )

    y1 = min(
        word["y"]
        for word in words
    )

    x2 = max(
        word["x"] + word["w"]
        for word in words
    )

    y2 = max(
        word["y"] + word["h"]
        for word in words
    )

    average_confidence = sum(
        word["confidence"]
        for word in words
    ) / len(words)

    return {
        "text": text.strip(),
        "confidence": round(
            average_confidence,
            2
        ),
        "x": x1,
        "y": y1,
        "w": x2 - x1,
        "h": y2 - y1
    }


# ============================================================
# CREATE MEANINGFUL REGIONS
# ============================================================

def create_regions(words):

    lines = group_words_into_lines(words)

    regions = []

    for line in lines:

        line_regions = split_line_into_regions(
            line
        )

        for region_words in line_regions:

            region = build_region(
                region_words
            )

            text = region["text"]

            # Ignore very short OCR fragments
            if len(text) < MIN_TEXT_LENGTH:
                continue

            regions.append(region)

    return regions


# ============================================================
# CLASSIFY TEXT
# ============================================================

def classify_text(text):

    """
    Classifies one OCR region using DistilBERT.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        probabilities = F.softmax(
            outputs.logits,
            dim=1
        )[0]

    predicted_id = torch.argmax(
        probabilities
    ).item()

    confidence = probabilities[
        predicted_id
    ].item()

    prediction = LABELS[
        predicted_id
    ]

    # --------------------------------------------------------
    # Confidence handling
    # --------------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        final_prediction = "uncertain"

        reason = "Low model confidence"

    else:

        final_prediction = prediction

        reason = "High-confidence prediction"

    return {
        "raw_prediction": prediction,
        "prediction": final_prediction,
        "confidence": round(
            confidence,
            4
        ),
        "reason": reason,
        "probabilities": {
            label: round(
                probabilities[i].item(),
                4
            )
            for i, label in enumerate(LABELS)
        }
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def analyze_image(image_path):

    print()
    print("=" * 70)
    print("STEP 1: OCR REGION EXTRACTION")
    print("=" * 70)

    print()
    print("Image:")
    print(image_path)

    print()
    print("Extracting OCR words...")
    print("-" * 70)

    words = extract_ocr_words(
        image_path
    )

    print(
        f"OCR words detected: {len(words)}"
    )

    print()
    print("Grouping words into meaningful regions...")

    regions = create_regions(
        words
    )

    print(
        f"Meaningful regions: {len(regions)}"
    )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 2: DISTILBERT REGION CLASSIFICATION")
    print("=" * 70)

    results = []

    for index, region in enumerate(
        regions,
        start=1
    ):

        text = region["text"]

        classification = classify_text(
            text
        )

        result = {
            "region_id": index,
            "text": text,
            "ocr_confidence": region[
                "confidence"
            ],
            "position": {
                "x": region["x"],
                "y": region["y"],
                "width": region["w"],
                "height": region["h"]
            },
            **classification
        }

        results.append(result)

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print()
        print("-" * 70)

        print(
            f"[Region {index}]"
        )

        print(
            f"Text       : {text}"
        )

        print(
            f"Position   : "
            f"x={region['x']}, "
            f"y={region['y']}, "
            f"w={region['w']}, "
            f"h={region['h']}"
        )

        print(
            f"OCR Conf.  : "
            f"{region['confidence']:.2f}%"
        )

        print(
            f"Prediction : "
            f"{classification['prediction']}"
        )

        print(
            f"Confidence : "
            f"{classification['confidence'] * 100:.2f}%"
        )

        print(
            f"Reason     : "
            f"{classification['reason']}"
        )

    return results


# ============================================================
# SAVE JSON
# ============================================================

def save_results(
    image_path,
    results
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Suspicious regions only
    # --------------------------------------------------------

    suspicious_regions = [
        result
        for result in results
        if (
            result["prediction"] != "normal"
            and
            result["prediction"] != "uncertain"
        )
    ]

    output = {

        "image": image_path,

        "total_regions": len(
            results
        ),

        "suspicious_regions": len(
            suspicious_regions
        ),

        "results": results,

        "suspicious": suspicious_regions
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    return suspicious_regions


# ============================================================
# FINAL RESULT
# ============================================================

def display_final_result(
    suspicious_regions
):

    print()
    print("=" * 70)
    print("DARPAN REGION ANALYSIS RESULT")
    print("=" * 70)

    print()
    print(
        f"Suspicious regions : "
        f"{len(suspicious_regions)}"
    )

    if not suspicious_regions:

        print()
        print(
            "No high-confidence dark pattern "
            "regions detected."
        )

    else:

        print()
        print(
            "Potential dark-pattern regions:"
        )

        print("-" * 70)

        for region in suspicious_regions:

            print()
            print(
                f"Region ID : "
                f"{region['region_id']}"
            )

            print(
                f"Text      : "
                f"{region['text']}"
            )

            print(
                f"Pattern   : "
                f"{region['prediction']}"
            )

            print(
                f"Confidence: "
                f"{region['confidence'] * 100:.2f}%"
            )

            position = region[
                "position"
            ]

            print(
                f"Position  : "
                f"x={position['x']}, "
                f"y={position['y']}, "
                f"w={position['width']}, "
                f"h={position['height']}"
            )

            print("-" * 70)

    print()
    print(
        f"JSON saved at:"
    )

    print(
        os.path.abspath(
            OUTPUT_JSON
        )
    )

    print()
    print("=" * 70)
    print(
        "REGION CLASSIFICATION COMPLETED"
    )
    print("=" * 70)


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print(
            "python src\\region_classifier.py "
            "<image_path>"
        )

        print()
        print("Example:")

        print(
            r"python src\region_classifier.py "
            r"data\yolo_dataset\images\misdirection_2.3.png"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):

        print()
        print(
            f"ERROR: Image not found:\n{image_path}"
        )

        sys.exit(1)

    try:

        results = analyze_image(
            image_path
        )

        suspicious_regions = save_results(
            image_path,
            results
        )

        display_final_result(
            suspicious_regions
        )

    except KeyboardInterrupt:

        print()
        print(
            "\nProcess interrupted by user."
        )

        sys.exit(0)

    except Exception as e:

        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)

        print()
        print(str(e))

        sys.exit(1)