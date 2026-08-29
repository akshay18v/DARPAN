import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# DARPAN - PAGE CHUNK TEST
# Existing model is NOT modified.
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "darpan_final_distilbert"
)

print("=" * 70)
print("DARPAN - PAGE CHUNK TEST")
print("=" * 70)

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.to(device)
model.eval()

# ------------------------------------------------------------
# OCR TEXT FROM YOUR TEST IMAGE
# ------------------------------------------------------------

ocr_text = """
Looking for instant coupons? See all coupons
Check out our Coupons & Deals page for today's discounts
Hurry! 51% of properties on our site are fully booked!
Rooms in Goa are in high demand on your selected dates.
Reserve yours now before prices go up.
5,972 properties in Goa
Sort by: Best match
Bloom Hotel - Calangute
8.0 Excellent
1,554 reviews
Calangute, Goa - City center
8.6 Location score
Best price for 3-star properties
Per night before taxes and fees
Rs. 6,473
-75%
Rs. 1,618
AGODASPONSORED applied - Rs. 310 off!
FREE CANCELLATION
"""

# ------------------------------------------------------------
# SPLIT PAGE INTO MEANINGFUL LINES
# ------------------------------------------------------------

lines = []

for line in ocr_text.splitlines():

    line = re.sub(
        r"\s+",
        " ",
        line
    ).strip()

    if len(line) >= 4:
        lines.append(line)

print("\nTotal OCR chunks:", len(lines))

# ------------------------------------------------------------
# PREDICT EACH CHUNK
# ------------------------------------------------------------

results = []

print("\n" + "=" * 70)
print("CHUNK PREDICTIONS")
print("=" * 70)

for i, text in enumerate(lines):

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
        )[0]

    prediction_id = int(
        torch.argmax(probabilities).item()
    )

    label = model.config.id2label[
        prediction_id
    ]

    confidence = float(
        probabilities[prediction_id].item()
    )

    results.append({
        "text": text,
        "label": label,
        "confidence": confidence
    })

    print(
        f"\n[{i + 1}] {label:<25} "
        f"{confidence * 100:6.2f}%"
    )

    print(
        "    ",
        text
    )


# ------------------------------------------------------------
# DARK-PATTERN EVIDENCE
# ------------------------------------------------------------

dark_results = [
    r for r in results
    if r["label"] != "normal"
]

print("\n" + "=" * 70)
print("DARK-PATTERN EVIDENCE")
print("=" * 70)

if dark_results:

    dark_results.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    for r in dark_results[:10]:

        print(
            f"\n{r['label']:<25} "
            f"{r['confidence'] * 100:.2f}%"
        )

        print(
            r["text"]
        )

else:

    print(
        "\nNo dark-pattern chunks detected."
    )


# ------------------------------------------------------------
# FINAL PAGE RESULT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PAGE-LEVEL RESULT")
print("=" * 70)

if dark_results:

    best = max(
        dark_results,
        key=lambda x: x["confidence"]
    )

    print(
        "\nDetected Pattern :",
        best["label"]
    )

    print(
        "Confidence        :",
        f"{best['confidence'] * 100:.2f}%"
    )

    print(
        "\nEvidence:"
    )

    print(
        best["text"]
    )

else:

    print(
        "\nDetected Pattern : normal"
    )

    print(
        "Confidence        : no dark-pattern evidence"
    )

print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)