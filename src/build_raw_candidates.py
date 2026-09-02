import os
import re
import pandas as pd

# ============================================================
# DARPAN - RAW OCR -> TEXT CANDIDATES
# ============================================================

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
    "raw_text_candidates.csv"
)

# ============================================================
# PATTERN PHRASES
# ============================================================

PATTERNS = {

    "hidden_fees": [
        "taxes and fees",
        "taxes & fees",
        "additional fee",
        "additional fees",
        "service fee",
        "convenience fee",
        "reservation fee",
        "booking fee",
        "processing fee",
        "charges and taxes",
        "fees calculated later",
        "fees apply",
        "fee will be charged",
        "plus taxes",
        "plus fees",
        "excluding taxes",
        "excluding fees"
    ],

    "false_urgency": [
        "only 1 room left",
        "only 2 rooms left",
        "only 3 rooms left",
        "only 1 room",
        "only 2 rooms",
        "only 3 rooms",
        "only 1 left",
        "only 2 left",
        "only 3 left",
        "rooms left",
        "last room",
        "last rooms",
        "limited time",
        "limited offer",
        "hurry",
        "book while you can",
        "high demand",
        "in high demand",
        "selling fast",
        "few rooms left",
        "spots left",
        "only 2 spots left",
        "session will expire",
        "expires in",
        "offer expires",
        "hours left",
        "days left"
    ],

    "sneaking": [
        "already selected",
        "automatically selected",
        "automatically added",
        "has been added",
        "already added",
        "added to your booking",
        "added to booking",
        "protection included",
        "insurance included",
        "travel protection",
        "travel insurance",
        "cancellation protection",
        "extra baggage",
        "priority boarding",
        "premium seat",
        "airport transfer",
        "meal package",
        "lounge access",
        "carbon offset",
        "donation",
        "contribution"
    ],

    "misdirection": [
        "most popular",
        "best value",
        "recommended for you",
        "recommended",
        "top pick",
        "guest favourite",
        "guest favorite",
        "editor's choice",
        "traveler's choice",
        "traveller's choice",
        "smart choice",
        "best seller",
        "top choice",
        "highly recommended",
        "suggested for you",
        "most chosen",
        "most booked",
        "popular choice",
        "our recommendation",
        "best option",
        "great value",
        "favorite choice",
        "favourite choice",
        "top rated",
        "deal of the day",
        "special pick",
        "chosen by travelers",
        "chosen by travellers"
    ],

    "interface_interference": [
        "continue to payment",
        "continue to checkout",
        "confirm booking",
        "confirm and pay",
        "add protection",
        "add insurance",
        "add trip protection",
        "choose premium",
        "select premium",
        "continue without",
        "no thanks"
    ],

    "obstruction": [
        "cannot go back",
        "can't go back",
        "cannot continue",
        "can't continue",
        "checkout locked",
        "payment is locked",
        "locked until",
        "cannot be edited",
        "can't be edited",
        "cannot return",
        "can't return",
        "previous step",
        "complete all steps",
        "complete all required",
        "must complete",
        "verification required",
        "continue is disabled",
        "cannot be modified",
        "changes are not allowed"
    ]
}

# ============================================================
# LOAD OCR
# ============================================================

print("=" * 70)
print("DARPAN - RAW OCR CANDIDATE BUILDER")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nRaw OCR rows:", len(df))

# ============================================================
# EXTRACT MATCHES
# ============================================================

rows = []

for _, row in df.iterrows():

    text = str(row.get("ocr_text", ""))

    if not text or text.lower() == "nan":
        continue

    lower = text.lower()

    for label, phrases in PATTERNS.items():

        for phrase in phrases:

            if phrase.lower() in lower:

                rows.append({
                    "text": phrase,
                    "label": label,
                    "source": "raw",
                    "image_name": row["image_name"],
                    "website": row["website"],
                    "comment": "Matched dark-pattern phrase in raw OCR"
                })


# ============================================================
# DATAFRAME
# ============================================================

candidate_df = pd.DataFrame(
    rows,
    columns=[
        "text",
        "label",
        "source",
        "image_name",
        "website",
        "comment"
    ]
)

# ============================================================
# REMOVE DUPLICATES
# ============================================================

before = len(candidate_df)

candidate_df = candidate_df.drop_duplicates(
    subset=[
        "text",
        "label",
        "image_name"
    ]
).reset_index(drop=True)

duplicates_removed = before - len(candidate_df)

# ============================================================
# SAVE
# ============================================================

candidate_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RAW CANDIDATE DATASET CREATED")
print("=" * 70)

print("\nCandidate rows:", len(candidate_df))

print("Duplicates removed:", duplicates_removed)

print("\nClass distribution:")

if len(candidate_df) > 0:
    print(
        candidate_df["label"].value_counts()
    )
else:
    print("No candidates found.")

print("\nColumns:")
print(list(candidate_df.columns))

print("\nSample candidates:")

if len(candidate_df) > 0:
    print(
        candidate_df.head(30).to_string(
            index=False
        )
    )

print("\nOutput:")
print(OUTPUT_FILE)

print("\nIMPORTANT:")
print("Raw images were NOT modified.")
print("OCR data was NOT modified.")
print("No normal labels were assigned.")

print("\n" + "=" * 70)