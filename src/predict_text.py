import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# DARPAN - DISTILBERT PREDICTION
# ============================================================

MODEL_DIR = r"models\distilbert_final"

print("=" * 70)
print("DARPAN - TEXT PREDICTION")
print("=" * 70)

# Load model
print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print("Model loaded successfully.")
print("Device:", device)


def predict(text):

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

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        prediction_id = torch.argmax(
            probabilities,
            dim=1
        ).item()

    predicted_label = model.config.id2label[
        prediction_id
    ]

    confidence = probabilities[
        0,
        prediction_id
    ].item()

    return predicted_label, confidence


# ============================================================
# TEST PREDICTIONS
# ============================================================

test_texts = [
    "Only 2 rooms left",
    "Platform fee $18.08",
    "Recommended for you",
    "Protection automatically added",
    "Continue without protection",
    "Safety information"
]

print("\n" + "=" * 70)
print("PREDICTIONS")
print("=" * 70)

for text in test_texts:

    label, confidence = predict(text)

    print("\nText       :", text)
    print("Prediction :", label)
    print("Confidence :", f"{confidence * 100:.2f}%")

print("\n" + "=" * 70)
print("PREDICTION TEST COMPLETED")
print("=" * 70)
