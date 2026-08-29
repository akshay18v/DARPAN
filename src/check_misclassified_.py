import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ============================================================
# DARPAN - CHECK MISCLASSIFIED TEST SAMPLES
# ============================================================

TEST_FILE = r"data\\processed\\distilbert_final\\test.csv"
MODEL_DIR = r"models\\distilbert_final"

print("=" * 70)
print("DARPAN - MISCLASSIFIED TEST SAMPLES")
print("=" * 70)

# Load test data
df = pd.read_csv(TEST_FILE)

# Load model and tokenizer
print("\nLoading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model.eval()

# Label mapping
id2label = {
    int(k): v
    for k, v in model.config.id2label.items()
}

print("Model loaded successfully.")

# ============================================================
# PREDICTION
# ============================================================

misclassified = []

print("\nChecking test samples...")
print("-" * 70)

for index, row in df.iterrows():

    text = str(row["text"])
    true_label = str(row["label"])

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = model(**inputs)

        prediction_id = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

    predicted_label = id2label[prediction_id]

    if predicted_label != true_label:

        misclassified.append({
            "index": index,
            "true": true_label,
            "predicted": predicted_label,
            "text": text
        })


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MISCLASSIFIED RESULTS")
print("=" * 70)

print(
    f"\nTotal test samples : {len(df)}"
)

print(
    f"Misclassified      : {len(misclassified)}"
)

print(
    f"Correct             : {len(df) - len(misclassified)}"
)

print("\n")

for i, item in enumerate(misclassified, 1):

    print("-" * 70)

    print(f"[{i}] Test Index : {item['index']}")

    print(
        f"TRUE      : {item['true']}"
    )

    print(
        f"PREDICTED : {item['predicted']}"
    )

    print(
        f"TEXT      : {item['text'][:700]}"
    )

print("\n" + "=" * 70)
print("CHECK COMPLETED")
print("=" * 70)
