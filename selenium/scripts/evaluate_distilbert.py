import pandas as pd
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "../../models/distilbert_dark_pattern"
TEST_PATH = "../../data/metadata/test.csv"

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("DARPAN - DISTILBERT MODEL EVALUATION")
print("=" * 60)

print("\nLoading trained model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

# ============================================================
# LOAD TEST DATA
# ============================================================

test_df = pd.read_csv(TEST_PATH)

test_df["text"] = (
    test_df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# ============================================================
# LABEL MAPPING FROM MODEL
# ============================================================

id2label = model.config.id2label

print("\nClasses:")

for label_id, label_name in id2label.items():
    print(f"{label_id} -> {label_name}")

# ============================================================
# PREDICTIONS
# ============================================================

true_labels = []
predicted_labels = []
confidences = []

print("\nRunning predictions...")

for _, row in test_df.iterrows():

    text = row["text"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        prediction = torch.argmax(
            probabilities,
            dim=-1
        ).item()

        confidence = probabilities[
            0, prediction
        ].item()

    true_label = row["category"]

    predicted_label = id2label[prediction]

    true_labels.append(true_label)
    predicted_labels.append(predicted_label)
    confidences.append(confidence)

# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

print("\n" + "=" * 60)
print("OVERALL RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        true_labels,
        predicted_labels,
        zero_division=0
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = list(id2label.values())

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=labels
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\nLabels:")
print(labels)

print("\nMatrix:")
print(cm)

# ============================================================
# SAVE PREDICTIONS
# ============================================================

test_df["predicted_category"] = predicted_labels
test_df["confidence"] = confidences

output_path = "../../data/metadata/test_predictions.csv"

test_df.to_csv(
    output_path,
    index=False
)

print("\nPredictions saved to:")
print(output_path)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)