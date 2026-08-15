import re
import pandas as pd
from pathlib import Path


# ============================================================
# DARPAN - URGENCY DETECTOR V1
# ============================================================
#
# Purpose:
# Detect urgency-related language from OCR extracted text.
#
# IMPORTANT:
# This component detects URGENCY SIGNALS.
# It does NOT automatically prove that the urgency is false.
#
# The final system should describe the result as:
# "Potential False Urgency"
# rather than claiming falsity from OCR alone.
# ============================================================


# ------------------------------------------------------------
# DATASET PATH
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# CHANGE THIS ONLY IF YOUR ACTUAL OCR CSV HAS A DIFFERENT NAME
OCR_FILE = BASE_DIR / "data" / "metadata" / "cleaned_dataset.csv"


# ------------------------------------------------------------
# URGENCY PATTERNS
# ------------------------------------------------------------
#
# These patterns are based on the actual urgency expressions
# found in your OCR dataset.
#
# Categories:
#   1. scarcity
#   2. countdown / expiry
#   3. social pressure
#   4. explicit urgency
#   5. limited-time promotion
#   6. high-demand / availability
# ------------------------------------------------------------

URGENCY_PATTERNS = {

    "scarcity": [

        r"\bonly\s+\d+\s+(?:room|rooms|seat|seats|spot|spots|ticket|tickets|item|items)"
        r"(?:\s+(?:left|remaining))?\b",

        r"\b\d+\s+(?:room|rooms|seat|seats|spot|spots|ticket|tickets)"
        r"\s+(?:left|remaining)\b",

        r"\bonly\s+\d+\s+left\b",

        r"\bonly\s+\d+\s+remain(?:ing)?\b",

        r"\blimited\s+availability\b",

        r"\bavailable\s+sections?\b.*?\b\d+\s+seats?\s+remaining\b",
    ],


    "countdown_expiry": [

        r"\boffer\s+expires?\b",

        r"\bprice\s+valid\s+for\s+(?:the\s+)?next\s+\d+\s*(?:min|mins|minute|minutes|sec|secs|second|seconds|hr|hrs|hour|hours)\b",

        r"\bexpires?\s+in\s+\d+\s*(?:min|mins|minute|minutes|sec|secs|second|seconds|hr|hrs|hour|hours)\b",

        r"\b(?:offer|deal|price|session)\s+expires?\s+in\b",

        r"\byour\s+session\s+will\s+expire\b",

        r"\btime\s+remaining\b",

        r"\boffer\s+expires?\s+in\s+\d+:\d+\b",
    ],


    "social_pressure": [

        r"\b\d+\s+(?:people|person|users?)\s+(?:are\s+)?viewing\b",

        r"\b\d+\s+others?\s+are\s+viewing\b",

        r"\b\d+\s+people\s+are\s+looking\s+right\s+now\b",

        r"\b\d+\s+people\s+viewing\s+(?:this\s+)?(?:right\s+now)?\b",

        r"\bbooked\s+\d+\s+times?\s+(?:today|in\s+the\s+last)\b",

        r"\b\d+\s+people\s+booked\b",

        r"\b\d+\s+people\s+are\s+looking\b",

        r"\b\d+\s+fans\s+attending\b",
    ],


    "explicit_urgency": [

        r"\bhurry\b",

        r"\bhurry\s*up\b",

        r"\bbook\s+now\b",

        r"\bact\s+now\b",

        r"\bbuy\s+now\b",

        r"\breserve\s+now\b",

        r"\block\s+in\b",

        r"\bdon'?t\s+miss\b",

        r"\bdeal\s+is\s+slipping\s+away\b",

        r"\bselling\s+fast\b",

        r"\bfast[-\s]?selling\b",
    ],


    "limited_time_promotion": [

        r"\blimited\s+time\s+offer\b",

        r"\blimited\s+time\s+sale\b",

        r"\blimited\s+time\b",

        r"\bflash\s+sale\b",

        r"\bdeal\s+of\s+the\s+day\b",

        r"\blast\s+minute\s+(?:price|deal|offer)\b",

        r"\bprice\s+drop\b",

        r"\bspecial\s+discount\b",

        r"\boffer\s+expires?\b",
    ],


    "high_demand": [

        r"\bin\s+high\s+demand\b",

        r"\bhigh\s+demand\b",

        r"\bbooked\s+\d+\s+times?\s+today\b",

        r"\b\d+\s+people\s+viewing\b",

        r"\b\d+\s+people\s+looking\b",

        r"\bonly\s+\d+\s+(?:rooms?|seats?|spots?)\s+(?:left|remaining)\b",
    ],
}


# ------------------------------------------------------------
# NORMALIZE OCR TEXT
# ------------------------------------------------------------

def normalize_text(text):
    """
    Clean OCR text while preserving useful numbers and words.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Lowercase for matching
    text = text.lower().strip()

    return text


# ------------------------------------------------------------
# DETECT URGENCY
# ------------------------------------------------------------

def detect_urgency(text):

    text = normalize_text(text)

    if not text:
        return {
            "urgency_detected": False,
            "urgency_score": 0.0,
            "signals": [],
            "evidence": []
        }

    detected_categories = set()
    evidence = []

    for category, patterns in URGENCY_PATTERNS.items():

        for pattern in patterns:

            matches = re.findall(pattern, text, flags=re.IGNORECASE)

            if matches:

                detected_categories.add(category)

                # Find actual matching text
                for match in re.finditer(
                    pattern,
                    text,
                    flags=re.IGNORECASE
                ):
                    evidence.append(match.group(0))

    # Remove duplicate evidence
    evidence = list(dict.fromkeys(evidence))

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------
    #
    # Multiple independent urgency categories increase score.
    #
    # This is NOT a probability.
    # It is an interpretable heuristic score.
    # --------------------------------------------------------

    category_count = len(detected_categories)

    if category_count == 0:
        score = 0.0

    elif category_count == 1:
        score = 0.40

    elif category_count == 2:
        score = 0.60

    elif category_count == 3:
        score = 0.75

    elif category_count >= 4:
        score = 0.90

    else:
        score = 0.0

    return {
        "urgency_detected": category_count > 0,
        "urgency_score": round(score, 2),
        "signals": sorted(detected_categories),
        "evidence": evidence
    }


# ------------------------------------------------------------
# PROCESS DATASET
# ------------------------------------------------------------

def process_dataset():

    print("=" * 60)
    print("DARPAN URGENCY DETECTOR V1")
    print("=" * 60)

    print(f"\nReading OCR dataset:")
    print(OCR_FILE)

    if not OCR_FILE.exists():

        print("\nERROR:")
        print("OCR dataset not found.")

        print("\nExpected location:")
        print(OCR_FILE)

        print("\nChange OCR_FILE at the top of this script")
        print("to your actual extracted-text CSV.")

        return

    df = pd.read_csv(OCR_FILE)

    print(f"\nTotal samples: {len(df)}")

    # Check text column
    possible_columns = [
        "text",
        "extracted_text",
        "ocr_text",
        "Text"
    ]

    text_column = None

    for column in possible_columns:

        if column in df.columns:
            text_column = column
            break

    if text_column is None:

        print("\nERROR: Could not find OCR text column.")

        print("\nAvailable columns:")
        print(list(df.columns))

        return

    print(f"OCR text column: {text_column}")

    # --------------------------------------------------------
    # DETECTION
    # --------------------------------------------------------

    results = df[text_column].apply(detect_urgency)

    df["urgency_detected"] = results.apply(
        lambda x: x["urgency_detected"]
    )

    df["urgency_score"] = results.apply(
        lambda x: x["urgency_score"]
    )

    df["urgency_signals"] = results.apply(
        lambda x: ", ".join(x["signals"])
    )

    df["urgency_evidence"] = results.apply(
        lambda x: " | ".join(x["evidence"])
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = (
        BASE_DIR
        / "data"
        / "metadata"
        / "urgency_detection_v1.csv"
    )

    df.to_csv(output_file, index=False)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    detected = df["urgency_detected"].sum()

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print(f"\nTotal samples      : {len(df)}")
    print(f"Urgency detected   : {detected}")
    print(f"No urgency         : {len(df) - detected}")

    print("\nSignal distribution:")

    signal_counts = {}

    for signals in df["urgency_signals"]:

        if not signals:
            continue

        for signal in signals.split(", "):

            signal_counts[signal] = (
                signal_counts.get(signal, 0) + 1
            )

    for signal, count in sorted(
        signal_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(f"{signal:25s}: {count}")

    print("\nOutput saved to:")
    print(output_file)

    print("\n" + "=" * 60)
    print("URGENCY DETECTION COMPLETE")
    print("=" * 60)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    process_dataset()