import pandas as pd
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# DARPAN - DISTILBERT V2 FINAL TEST
# ============================================================

print("=" * 60)
print("DARPAN - DISTILBERT V2 FINAL TEST")
print("=" * 60)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "../../models/distilbert_dark_pattern_v2"
TEST_PATH = "../../data/metadata/test_clean.csv"


# ============================================================
# LABELS
# ============================================================

LABELS = [
    "Sneaking",
    "hidden fees",
    "interface interference",
    "misdirection",
    "obstruct_sneaking",
    "obstruction",
]

label2id = {
    label: i
    for i, label in enumerate(LABELS)
}

id2label = {
    i: label
    for i, label in enumerate(LABELS)
}


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading final test dataset...")

df = pd.read_csv(TEST_PATH)

df = df.dropna(
    subset=["text", "category"]
).copy()

df = df[
    df["category"].isin(LABELS)
].copy()

texts = df["text"].tolist()

true_labels = [
    label2id[x]
    for x in df["category"]
]


print("Test samples:", len(df))


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading V2 model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model.to(device)

print("Device:", device)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nRunning predictions...")

predictions = []
confidences = []

batch_size = 8

for i in range(
    0,
    len(texts),
    batch_size
):

    batch_texts = texts[
        i:i + batch_size
    ]

    encoded = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    encoded = {
        k: v.to(device)
        for k, v in encoded.items()
    }

    with torch.no_grad():

        outputs = model(
            **encoded
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        batch_predictions = torch.argmax(
            probabilities,
            dim=-1
        )

        batch_confidences = torch.max(
            probabilities,
            dim=-1
        ).values

    predictions.extend(
        batch_predictions.cpu().numpy()
    )

    confidences.extend(
        batch_confidences.cpu().numpy()
    )


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predictions
)

print("\n")
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        true_labels,
        predictions,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(
    true_labels,
    predictions,
    labels=list(range(len(LABELS)))
)

print("Confusion Matrix:")

print(matrix)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

df["true_label"] = [
    id2label[x]
    for x in true_labels
]

df["predicted_label"] = [
    id2label[x]
    for x in predictions
]

df["confidence"] = confidences

output_path = (
    "../../data/metadata/test_predictions_v2.csv"
)

df.to_csv(
    output_path,
    index=False
)

print(
    f"\nPredictions saved to:\n{output_path}"
)


print("\n")
print("=" * 60)
print("V2 FINAL TEST COMPLETE")
print("=" * 60)