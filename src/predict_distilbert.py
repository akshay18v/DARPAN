# ============================================================
# DARPAN - DISTILBERT PREDICTION
# ============================================================

import os
import torch
import torch.nn.functional as F

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/distilbert_final"

CONFIDENCE_THRESHOLD = 0.60
MIN_TEXT_LENGTH = 10


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LABELS
# ============================================================

labels = [
    "false_urgency",
    "hidden_fees",
    "interface_interference",
    "misdirection",
    "normal",
    "obstruction",
    "sneaking"
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("DARPAN - DISTILBERT PREDICTION")
    print("=" * 70)

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print("GPU available.")
    else:
        print("CPU mode enabled.")

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Trained model not found!\n"
            f"Expected location: {MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = DistilBertTokenizer.from_pretrained(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("Loading trained model...")

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=len(labels)
    )

    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    return tokenizer, model


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_text(
    text,
    tokenizer=None,
    model=None,
    display=True
):

    # --------------------------------------------------------
    # Load model if not already loaded
    # --------------------------------------------------------

    if tokenizer is None or model is None:

        tokenizer, model = load_model()

    # --------------------------------------------------------
    # Clean input
    # --------------------------------------------------------

    text = text.strip()

    if not text:

        return {
            "label": "uncertain",
            "raw_prediction": None,
            "confidence": 0.0,
            "reason": "No text provided.",
            "probabilities": {}
        }

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(**inputs)

        logits = outputs.logits

        probabilities = F.softmax(
            logits,
            dim=1
        )

        predicted_id = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0
        ][predicted_id].item()

    # --------------------------------------------------------
    # Raw prediction
    # --------------------------------------------------------

    raw_prediction = labels[predicted_id]

    # --------------------------------------------------------
    # Confidence handling
    # --------------------------------------------------------

    text_length = len(text)

    if text_length < MIN_TEXT_LENGTH:

        final_prediction = "uncertain"

        reason = (
            f"Text is too short "
            f"({text_length} characters < "
            f"{MIN_TEXT_LENGTH})"
        )

    elif confidence < CONFIDENCE_THRESHOLD:

        final_prediction = "uncertain"

        reason = (
            f"Model confidence is below "
            f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
        )

    else:

        final_prediction = raw_prediction

        reason = "High-confidence prediction"

    # --------------------------------------------------------
    # Probability dictionary
    # --------------------------------------------------------

    probability_dict = {}

    for label, probability in zip(
        labels,
        probabilities[0].cpu().tolist()
    ):

        probability_dict[label] = (
            probability * 100
        )

    # ========================================================
    # RESULT
    # ========================================================

    result = {
        "label": final_prediction,
        "raw_prediction": raw_prediction,
        "confidence": confidence * 100,
        "reason": reason,
        "probabilities": probability_dict
    }

    # ========================================================
    # DISPLAY
    # ========================================================

    if display:

        print("\n" + "=" * 70)
        print("PREDICTION")
        print("=" * 70)

        print("\nText:")
        print(text)

        print("\n--------------------------------------------------")

        print(
            f"Raw Model Prediction : "
            f"{raw_prediction}"
        )

        print(
            f"Model Confidence     : "
            f"{confidence * 100:.2f}%"
        )

        print("--------------------------------------------------")

        print(
            f"Final Prediction     : "
            f"{final_prediction}"
        )

        print(
            f"Status               : "
            f"{reason}"
        )

        print("--------------------------------------------------")

        print("\nAll class probabilities:")

        for label, probability in probability_dict.items():

            print(
                f"{label:25s}: "
                f"{probability:.2f}%"
            )

        print("=" * 70)

    return result


# ============================================================
# MANUAL TESTING MODE
# ============================================================

def interactive_mode():

    tokenizer, model = load_model()

    print("\nEnter OCR text to classify.")
    print("Type 'exit' to stop.")

    while True:

        print("\n" + "-" * 70)

        text = input("Text: ")

        if text.strip().lower() == "exit":

            print(
                "\nExiting prediction system..."
            )

            break

        predict_text(
            text,
            tokenizer,
            model,
            display=True
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    interactive_mode()
