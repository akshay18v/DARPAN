import os
import re
import json
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

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "metadata",
    "cleaned_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "distilbert_dark_pattern_v3"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "metadata",
    "unified_predictions_v1.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("DARPAN FULL DATASET PREDICTION V1")
print("=" * 60)

print("\nReading dataset:")
print(DATASET_PATH)

df = pd.read_csv(DATASET_PATH)

print(f"\nTotal samples: {len(df)}")

if "text" not in df.columns:
    raise ValueError(
        "Dataset does not contain a 'text' column."
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print(f"Device: {device}")


# ============================================================
# IMPORTANT:
# GET LABEL MAPPING DIRECTLY FROM MODEL
# ============================================================

id2label = model.config.id2label

print("\nModel label mapping:")

for key, value in id2label.items():
    print(f"  {key} -> {value}")


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
        r"\bdon'?t wait\b",
        r"\bquickly\b",
        r"\bimmediately\b"
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
        r"\b\d+\s+people\s+viewing\s+(?:this|now)\b",
        r"\bpopular choice\b",
        r"\bmost bookers choose\b"
    ],

    "high_demand": [
        r"\bin high demand\b",
        r"\bhigh demand\b",
        r"\bselling fast\b",
        r"\bbooked \d+ times\b",
        r"\bbooked \d+ times today\b",
        r"\bpopular\b"
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

    if pd.isna(text):
        text = ""

    text = str(text).lower()

    signals = []
    matched_phrases = []

    for signal_type, patterns in URGENCY_PATTERNS.items():

        signal_found = False

        for pattern in patterns:

            matches = list(
                re.finditer(pattern, text)
            )

            if matches:

                signal_found = True

                if signal_type not in signals:
                    signals.append(signal_type)

                for match in matches:

                    phrase = match.group(0).strip()

                    if (
                        phrase
                        and phrase not in matched_phrases
                    ):
                        matched_phrases.append(
                            phrase
                        )

        # Continue checking other signal types

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    signal_count = len(signals)

    if "countdown_expiry" in signals:
        severity = "HIGH"

    elif (
        "scarcity" in signals
        and (
            "social_pressure" in signals
            or "high_demand" in signals
        )
    ):
        severity = "HIGH"

    elif signal_count >= 2:
        severity = "MEDIUM"

    elif signal_count == 1:
        severity = "LOW"

    else:
        severity = "NONE"

    urgency_detected = signal_count > 0

    potential_false_urgency = (
        urgency_detected
        and severity in ["MEDIUM", "HIGH"]
    )

    return {
        "urgency_detected": urgency_detected,
        "urgency_signals": "; ".join(signals),
        "matched_phrases": "; ".join(
            matched_phrases
        ),
        "urgency_severity": severity,
        "potential_false_urgency":
            potential_false_urgency
    }


# ============================================================
# DISTILBERT PREDICTION
# ============================================================

def predict_dark_pattern(text):

    if pd.isna(text):
        text = ""

    text = str(text)

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

        predicted_id = torch.argmax(
            probabilities,
            dim=-1
        ).item()

        confidence = probabilities[
            0,
            predicted_id
        ].item()

    predicted_label = id2label.get(
        predicted_id,
        f"class_{predicted_id}"
    )

    return (
        predicted_label,
        confidence
    )


# ============================================================
# PROCESS ALL 461 SAMPLES
# ============================================================

results = []

print("\n" + "=" * 60)
print("PROCESSING DATASET")
print("=" * 60)

for index, row in df.iterrows():

    image_name = row.get(
        "image_name",
        ""
    )

    actual_category = row.get(
        "category",
        ""
    )

    text = row.get(
        "text",
        ""
    )

    # --------------------------------------------------------
    # DistilBERT
    # --------------------------------------------------------

    predicted_category, confidence = (
        predict_dark_pattern(text)
    )

    # --------------------------------------------------------
    # Urgency
    # --------------------------------------------------------

    urgency = detect_urgency(text)

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    result = {

        "image_name": image_name,

        "actual_category": actual_category,

        "predicted_category":
            predicted_category,

        "dark_pattern_confidence":
            round(confidence, 4),

        "urgency_detected":
            urgency["urgency_detected"],

        "urgency_severity":
            urgency["urgency_severity"],

        "urgency_signals":
            urgency["urgency_signals"],

        "matched_phrases":
            urgency["matched_phrases"],

        "potential_false_urgency":
            urgency[
                "potential_false_urgency"
            ]
    }

    results.append(result)

    # Progress

    if (index + 1) % 25 == 0:

        print(
            f"Processed "
            f"{index + 1}/{len(df)}"
        )


# ============================================================
# SAVE
# ============================================================

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FULL DATASET PREDICTION COMPLETE")
print("=" * 60)

print(
    f"\nTotal processed: "
    f"{len(result_df)}"
)

print(
    "\nUrgency distribution:"
)

print(
    result_df[
        "urgency_detected"
    ].value_counts()
)

print(
    "\nSeverity distribution:"
)

print(
    result_df[
        "urgency_severity"
    ].value_counts()
)

print(
    "\nPotential false urgency:"
)

print(
    result_df[
        "potential_false_urgency"
    ].value_counts()
)

print(
    "\nOutput saved to:"
)

print(OUTPUT_PATH)

print("\n" + "=" * 60)