import os
import re
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "raw_ocr_candidates.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "raw_curated_text.csv"
)

VALID_LABELS = {
    "hidden_fees",
    "false_urgency",
    "sneaking",
    "misdirection",
    "interface_interference",
    "obstruction",
    "normal"
}

df = pd.read_csv(INPUT_FILE)

rows = []

# ------------------------------------------------------------
# Patterns for genuine dark-pattern text
# ------------------------------------------------------------

patterns = {

    "hidden_fees": [
        r"taxes?\s*(and|&)\s*fees?",
        r"additional\s+fees?",
        r"convenience\s+fees?",
        r"service\s+fees?",
        r"booking\s+fees?",
        r"charges?\s+and\s+taxes?",
        r"fees?\s+(may|will)\s+apply",
        r"excluding\s+taxes?",
        r"taxes?\s+not\s+included",
        r"taxes?\s+and\s+fees?\s+calculated\s+later",
    ],

    "false_urgency": [
        r"limited\s+availability",
        r"limited\s+time",
        r"only\s+\d+\s+(room|rooms|left|spot|spots)",
        r"\d+\s+left\s+at\s+this\s+price",
        r"high\s+demand",
        r"hurry",
        r"book\s+now",
        r"book\s+while\s+you\s+still\s+can",
        r"last\s+rooms?\s+available",
        r"offer\s+expires",
        r"session\s+will\s+expire",
        r"only\s+\d+\s+left",
    ],

    "sneaking": [
        r"travel\s+insurance",
        r"airport\s+transfer",
        r"premium\s+protection",
        r"travel\s+protection",
        r"cancellation\s+protection",
        r"extra\s+baggage",
        r"priority\s+boarding",
        r"carbon\s+offset",
        r"donation",
        r"add[- ]on",
        r"premium\s+package",
        r"upgrade\s+package",
        r"selected\s+automatically",
        r"automatically\s+selected",
    ],

    "misdirection": [
        r"guest\s+favourite",
        r"guest\s+favorite",
        r"most\s+popular",
        r"best\s+value",
        r"recommended",
        r"editor'?s\s+choice",
        r"travell?er'?s\s+choice",
        r"smart\s+choice",
        r"best\s+seller",
        r"top\s+choice",
        r"highly\s+recommended",
        r"suggested\s+for\s+you",
        r"deal\s+of\s+the\s+day",
        r"best\s+option",
        r"great\s+value",
    ],

    "interface_interference": [
        r"choose\s+enhanced\s+protection",
        r"keep\s+my\s+seats",
        r"select\s+premium",
        r"confirm\s+and\s+pay",
        r"add\s+for\s+\$?\d+",
        r"choose\s+luxury\s+package",
        r"select\s+smart\s+option",
        r"continue\s+without",
        r"no\s+thanks.*continue",
        r"continue\s+without\s+membership",
    ],

    "obstruction": [
        r"cannot\s+go\s+back",
        r"cannot\s+continue",
        r"checkout\s+locked",
        r"complete\s+all\s+steps",
        r"cannot\s+be\s+edited",
        r"cannot\s+return",
        r"verification\s+required",
        r"payment\s+is\s+locked",
        r"previous\s+step\s+cannot",
        r"must\s+complete\s+this\s+step",
        r"return\s+option\s+is\s+unavailable",
        r"continue\s+is\s+disabled",
    ]
}


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ------------------------------------------------------------
# Re-scan RAW OCR text
# ------------------------------------------------------------

for _, row in df.iterrows():

    text = normalize(row["ocr_text"])

    if not text:
        continue

    for label, regexes in patterns.items():

        for pattern in regexes:

            match = re.search(pattern, text)

            if match:

                matched_text = match.group(0).strip()

                # Ignore extremely short generic words
                if len(matched_text) < 4:
                    continue

                rows.append({
                    "text": matched_text,
                    "label": label,
                    "source": "raw",
                    "image_name": row.get("image_name", ""),
                    "website": row.get("website", ""),
                    "comment": "Matched from raw OCR"
                })

                break


# ------------------------------------------------------------
# DataFrame
# ------------------------------------------------------------

out = pd.DataFrame(rows)

if len(out) > 0:

    out["text"] = (
        out["text"]
        .astype(str)
        .str.strip()
    )

    out = out.drop_duplicates(
        subset=["text", "label"]
    )

else:

    out = pd.DataFrame(
        columns=[
            "text",
            "label",
            "source",
            "image_name",
            "website",
            "comment"
        ]
    )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

out.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 70)
print("DARPAN - CURATED RAW TEXT")
print("=" * 70)

print("\nRaw OCR rows:", len(df))
print("Curated rows:", len(out))

print("\nClass distribution:")

if len(out):
    print(out["label"].value_counts())
else:
    print("No candidates found.")

print("\nOutput:")
print(OUTPUT_FILE)

print("\nDone.")