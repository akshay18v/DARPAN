"""
============================================================
DARPAN - UNIFIED PREDICTION PIPELINE V1
============================================================

Combines:
1. DistilBERT text dark-pattern classifier
2. Urgency detector
3. Produces one unified JSON result

Visual model integration will be added separately when its
final inference script is ready.
============================================================
"""

import os
import json
import re
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "distilbert_dark_pattern_v3"
)

LABEL_MAP_PATH = os.path.join(
    BASE_DIR,
    "data",
    "metadata",
    "train_v3.csv"
)


# ============================================================
# MODEL LABELS
# ============================================================

# IMPORTANT:
# Keep this mapping consistent with the labels used while
# training DistilBERT.

LABELS = {
    0: "hidden fees",
    1: "interface interference",
    2: "misdirection",
    3: "obstruction",
    4: "Sneaking"
}


# ============================================================
# LOAD DISTILBERT
# ============================================================

print("=" * 60)
print("DARPAN UNIFIED PREDICTION PIPELINE V1")
print("=" * 60)

print("\nLoading DistilBERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print(f"Device: {device}")
print("DistilBERT loaded successfully.")


# ============================================================
# URGENCY PATTERNS
# ============================================================

URGENCY_PATTERNS = {

    "explicit_urgency": [
        r"\bhurry\b",
        r"\bact now\b",
        r"\bbook now\b",
        r"\bbuy now\b",
        r"\bgrab now\b",
        r"\bdon't wait\b",
        r"\bdo not wait\b",
        r"\bquickly\b",
        r"\bimmediately\b",
        r"\bnow\b"
    ],

    "scarcity": [
        r"\bonly\s+\d+\s+(?:room|rooms|seat|seats|ticket|tickets|spot|spots)\s+(?:left|remaining)\b",
        r"\b\d+\s+(?:room|rooms|seat|seats|ticket|tickets|spot|spots)\s+(?:left|remaining)\b",
        r"\blimited availability\b",
        r"\blimited stock\b",
        r"\blimited seats\b",
        r"\blimited rooms\b",
        r"\bfew left\b",
        r"\bonly a few left\b"
    ],

    "social_pressure": [
        r"\b\d+\s+people\s+(?:are\s+)?viewing\b",
        r"\b\d+\s+people\s+(?:are\s+)?looking\b",
        r"\b\d+\s+people\s+booked\b",
        r"\b\d+\s+people\s+booked\s+(?:this|in the last)\b",
        r"\b\d+\s+people\s+viewing\s+(?:this|now)\b",
        r"\bpopular choice\b",
        r"\bmost bookers choose\b",
        r"\bpeople are viewing\b"
    ],

    "high_demand": [
        r"\bin high demand\b",
        r"\bhigh demand\b",
        r"\bselling fast\b",
        r"\bbooked \d+ times\b",
        r"\bbooked \d+ times today\b",
        r"\bpopular\b",
        r"\bpeople are looking\b"
    ],

    "limited_time_promotion": [
        r"\blimited time offer\b",
        r"\blimited time sale\b",
        r"\blimited time\b",
        r"\boffer expires\b",
        r"\bexpires in\b",
        r"\bvalid for the next\b",
        r"\bprice valid for\b",
        r"\bends tonight\b",
        r"\bends today\b",
        r"\btoday only\b"
    ],

    "countdown_expiry": [
        r"\bexpires?\s+in\s+\d+\s*[:]\s*\d+\b",
        r"\b\d+\s*[:]\s*\d+\s*(?:remaining|left)\b",
        r"\b\d+\s+(?:minutes?|mins?|seconds?|secs?)\s+(?:left|remaining)\b",
        r"\btime\s+left\b",
        r"\bcountdown\b"
    ]
}


# ============================================================
# URGENCY DETECTOR
# ============================================================

def detect_urgency(text):

    if not isinstance(text, str):
        text = ""

    text = text.lower()

    detected_signals = []
    matched_phrases = []

    for signal_type, patterns in URGENCY_PATTERNS.items():

        for pattern in patterns:

            matches = re.findall(pattern, text)

            if matches:

                detected_signals.append(signal_type)

                # Get actual matching text
                regex_matches = re.finditer(pattern, text)

                for match in regex_matches:

                    phrase = match.group(0).strip()

                    if phrase and phrase not in matched_phrases:
                        matched_phrases.append(phrase)

                break

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    signal_count = len(detected_signals)

    if "countdown_expiry" in detected_signals:
        severity = "HIGH"

    elif "scarcity" in detected_signals and (
        "social_pressure" in detected_signals
        or "high_demand" in detected_signals
    ):
        severity = "HIGH"

    elif signal_count >= 2:
        severity = "MEDIUM"

    elif signal_count == 1:
        severity = "LOW"

    else:
        severity = "NONE"

    urgency_detected = signal_count > 0

    # --------------------------------------------------------
    # Important terminology:
    #
    # We DO NOT claim that urgency is definitely false.
    # It is only potential false urgency.
    # --------------------------------------------------------

    potential_false_urgency = (
        urgency_detected
        and severity in ["MEDIUM", "HIGH"]
    )

    return {
        "urgency_detected": urgency_detected,
        "urgency_signals": detected_signals,
        "matched_phrases": matched_phrases,
        "urgency_severity": severity,
        "potential_false_urgency": potential_false_urgency
    }


# ============================================================
# DISTILBERT PREDICTION
# ============================================================

def predict_dark_pattern(text):

    if not isinstance(text, str):
        text = ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=-1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    return {
        "dark_pattern": LABELS.get(
            predicted_class,
            f"class_{predicted_class}"
        ),
        "dark_pattern_confidence": round(
            confidence,
            4
        )
    }


# ============================================================
# UNIFIED PREDICTION
# ============================================================

def analyze_text(text):

    dark_pattern_result = predict_dark_pattern(text)

    urgency_result = detect_urgency(text)

    result = {
        **dark_pattern_result,
        **urgency_result
    }

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("TEST PREDICTION")
    print("=" * 60)

    sample_text = """
    ONLY 2 ROOMS LEFT!
    44 people are viewing this right now.
    Hurry! Price valid for next 10 minutes.
    Book now before the deal expires.
    """

    print("\nInput:")
    print(sample_text)

    result = analyze_text(sample_text)

    print("\n" + "=" * 60)
    print("DARPAN RESULT")
    print("=" * 60)

    print(
        json.dumps(
            result,
            indent=4
        )
    )

    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETE")
    print("=" * 60)