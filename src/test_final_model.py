import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/darpan_final_distilbert"
TEST_PATH = "data/processed/darpan_final_splits/test.csv"

print("=" * 70)
print("DARPAN - FINAL MODEL TEST")
print("=" * 70)

# ============================================================
# LOAD TEST DATA
# ============================================================

df = pd.read_csv(TEST_PATH)

texts = df["text"].fillna("").astype(str).tolist()
true_labels = df["label"].astype(str).tolist()

print(f"\nTest samples: {len(df)}")

# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print("Loading final DistilBERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

print("Model loaded successfully.")

# ============================================================
# MODEL LABEL MAPPING
# ============================================================

id2label = {
    int(k): v
    for k, v in model.config.id2label.items()
}

label2id = {
    v: k
    for k, v in id2label.items()
}

print("\nModel labels:")
for k, v in id2label.items():
    print(f"  {k} -> {v}")

# ============================================================
# CONVERT TRUE LABELS TO IDS
# ============================================================

true_ids = []

for label in true_labels:

    if label not in label2id:
        raise ValueError(
            f"Label '{label}' found in test.csv "
            f"but not present in model label mapping."
        )

    true_ids.append(label2id[label])

# ============================================================
# PREDICTION
# ============================================================

predicted_ids = []

print("\nRunning predictions...")

with torch.no_grad():

    for text in texts:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        outputs = model(**inputs)

        predicted_label_id = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

        predicted_ids.append(predicted_label_id)

# ============================================================
# CONVERT PREDICTIONS TO LABEL NAMES
# ============================================================

predicted_labels = [
    id2label[p]
    for p in predicted_ids
]

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    true_ids,
    predicted_ids
)

precision, recall, f1, _ = precision_recall_fscore_support(
    true_ids,
    predicted_ids,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"\nAccuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

all_label_ids = sorted(id2label.keys())

target_names = [
    id2label[i]
    for i in all_label_ids
]

print(
    classification_report(
        true_ids,
        predicted_ids,
        labels=all_label_ids,
        target_names=target_names,
        zero_division=0
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    true_ids,
    predicted_ids,
    labels=all_label_ids
)

print("\nLabels:")
print(target_names)

print("\n")
print(cm)

# ============================================================
# SAVE PREDICTIONS
# ============================================================

df["predicted_label"] = predicted_labels
df["correct"] = (
    df["label"].astype(str) == df["predicted_label"]
)

output_path = "data/processed/final_test_predictions.csv"

df.to_csv(
    output_path,
    index=False
)

print(f"\nPredictions saved to:")
print(output_path)

print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)