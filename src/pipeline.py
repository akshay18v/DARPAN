import os
import sys
import re
import cv2
import torch
import pytesseract

from difflib import SequenceMatcher
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# DARPAN - ROBUST MULTI-PATTERN PIPELINE
#
# IMAGE
#   ↓
# MULTI-PASS OCR
#   ↓
# CLEAN / NORMALIZE OCR
#   ↓
# DISTILBERT
#   ↓
# ALL 6 PATTERN EVIDENCE
#   ↓
# MULTIPLE DARK PATTERNS
#
# Existing model and dataset are NOT modified.
# ============================================================


# ============================================================
# PATHS
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

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pytesseract.pytesseract.tesseract_cmd = (
    TESSERACT_PATH
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 75)
print("DARPAN - ROBUST MULTI-PATTERN PIPELINE")
print("=" * 75)

print("\nDevice:", device)

print(
    "Loading trained DistilBERT model..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# OCR
# ============================================================

def extract_text(image_path):

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise ValueError(
            f"Unable to read image:\n{image_path}"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # OCR PASS 1
    # --------------------------------------------------------

    processed_1 = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )[1]

    text_1 = pytesseract.image_to_string(
        processed_1,
        config="--psm 6"
    )

    # --------------------------------------------------------
    # OCR PASS 2
    # --------------------------------------------------------
    # PSM 11 is useful for scattered webpage text.

    text_2 = pytesseract.image_to_string(
        gray,
        config="--psm 11"
    )

    # --------------------------------------------------------
    # Combine both OCR passes
    # --------------------------------------------------------

    return (
        text_1 +
        "\n" +
        text_2
    )


# ============================================================
# NORMALIZE OCR
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    # OCR sometimes inserts symbols inside words.
    text = text.replace(
        "™",
        ""
    )

    text = text.replace(
        "®",
        ""
    )

    text = text.replace(
        "©",
        ""
    )

    text = text.replace(
        "—",
        " "
    )

    text = text.replace(
        "–",
        " "
    )

    text = text.replace(
        "’",
        "'"
    )

    # Common OCR character substitutions.
    replacements = {

        "selec ed":
            "selected",

        "selec ed automatically":
            "selected automatically",

        "selecled":
            "selected",

        "seieced":
            "selected",

        "seiected":
            "selected",

        "automaticaliy":
            "automatically",

        "automaticaly":
            "automatically",

        "automatica11y":
            "automatically",

        "scked":
            "locked",

        "Iocked":
            "locked",

        "ont":
            "until",

        "coc s":
            "continue"
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    # Remove excessive punctuation
    text = re.sub(
        r"[^\w\s$₹€£%+&:/.,!?'-]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# CLEAN LINE
# ============================================================

def clean_line(text):

    return normalize_text(
        text
    )


# ============================================================
# GARBAGE FILTER
# ============================================================

def is_garbage(text):

    text = clean_line(
        text
    )

    if len(text) < 5:
        return True

    letters = sum(
        c.isalpha()
        for c in text
    )

    digits = sum(
        c.isdigit()
        for c in text
    )

    meaningful = (
        letters +
        digits
    )

    if meaningful < 3:
        return True

    ratio = (
        meaningful /
        max(len(text), 1)
    )

    if (
        len(text) > 8
        and ratio < 0.25
    ):
        return True

    return False


# ============================================================
# CREATE CHUNKS
# ============================================================

def create_chunks(text):

    chunks = []

    for raw_line in text.splitlines():

        line = clean_line(
            raw_line
        )

        if not line:
            continue

        if is_garbage(line):
            continue

        if len(line) > 350:

            parts = re.split(
                r"(?<=[.!?])\s+",
                line
            )

            for part in parts:

                part = clean_line(
                    part
                )

                if (
                    len(part) >= 5
                    and not is_garbage(part)
                ):

                    chunks.append(
                        part
                    )

        else:

            chunks.append(
                line
            )

    # --------------------------------------------------------
    # Remove exact duplicates
    # --------------------------------------------------------

    unique_chunks = []

    seen = set()

    for chunk in chunks:

        key = chunk.lower()

        if key not in seen:

            seen.add(key)

            unique_chunks.append(
                chunk
            )

    return unique_chunks


# ============================================================
# PATTERN EVIDENCE
# ============================================================

EVIDENCE = {

    # --------------------------------------------------------
    # FALSE URGENCY
    # --------------------------------------------------------

    "false_urgency": [

        r"\bhurry\b",

        r"\blimited availability\b",

        r"\blimited time\b",

        r"\blimited offer\b",

        r"\bhigh demand\b",

        r"\bin high demand\b",

        r"\bonly\s+\d+\b",

        r"\b\d+\s+left\b",

        r"\bleft\s+at\s+this\s+price\b",

        r"\blast room\b",

        r"\blast rooms\b",

        r"\brooms left\b",

        r"\bspots left\b",

        r"\bseats left\b",

        r"\bselling fast\b",

        r"\bfilling fast\b",

        r"\bbook while you can\b",

        r"\bbook before\b",

        r"\boffer expires\b",

        r"\bdeal expires\b",

        r"\bexpires in\b",

        r"\bsession will expire\b",

        r"\bprices?\s+go\s+up\b",

        r"\bprices?\s+may\s+increase\b",

        r"\bprice increases?\b",

        r"\bfirst come, first served\b"
    ],


    # --------------------------------------------------------
    # HIDDEN FEES
    # --------------------------------------------------------

    "hidden_fees": [

        r"\btaxes?\s+and\s+fees?\b",

        r"\btaxes?\s*&\s*fees?\b",

        r"\bmandatory\s+[\w\s-]{0,50}\bfee\b",

        r"\bproperty\s+cleaning\s+fee\b",

        r"\bcleaning\s+fee\b",

        r"\bservice\s+fee\b",

        r"\bbooking\s+fee\b",

        r"\bconvenience\s+fee\b",

        r"\bprocessing\s+fee\b",

        r"\badditional\s+fee\b",

        r"\badditional\s+fees\b",

        r"\bextra\s+charges?\b",

        r"\bcharges?\s+and\s+taxes?\b",

        r"\bfees\s+calculated\s+later\b",

        r"\bfees\s+apply\b",

        r"\bfee\s+will\s+be\s+charged\b",

        r"\bexcluding\s+taxes?\b",

        r"\bexcluding\s+fees?\b"
    ],


    # --------------------------------------------------------
    # MISDIRECTION
    # --------------------------------------------------------

    "misdirection": [

        r"\bmost popular\b",

        r"\bbest value\b",

        r"\brecommended\b",

        r"\brecommended for you\b",

        r"\btop pick\b",

        r"\btop choice\b",

        r"\bguest favourite\b",

        r"\bguest favorite\b",

        r"\beditor'?s choice\b",

        r"\btravell?er'?s choice\b",

        r"\bsmart choice\b",

        r"\bbest seller\b",

        r"\bmost chosen\b",

        r"\bmost booked\b",

        r"\bpopular choice\b",

        r"\bour recommendation\b",

        r"\bsuggested for you\b",

        r"\bdeal of the day\b",

        r"\bspecial pick\b",

        r"\btop rated\b",

        r"\blower than other sites\b",

        r"\bless than other sites\b",

        r"\bbetter than\b",

        r"\bbest price\b",

        r"\bbest deal\b",

        r"\bhot hotel deals\b",

        r"\bhot deals\b",

        r"\bprice drop\b",

        r"\bprice dropped\b",

        r"\bsave\s+\d+%\b",

        r"\b\d+%\s+lower\b",

        r"\b\d+%\s+off\b"
    ],


    # --------------------------------------------------------
    # SNEAKING
    # --------------------------------------------------------

    "sneaking": [

        r"\balready selected\b",

        r"\bselected automatically\b",

        r"\bautomatically selected\b",

        r"\bautomatically added\b",

        r"\balready added\b",

        r"\bhas been added\b",

        r"\badded to your booking\b",

        r"\badded to booking\b",

        r"\bprotection included\b",

        r"\binsurance included\b",

        r"\bcarbon offset\b",

        r"\bdonation added\b",

        r"\bcontribution added\b",

        r"\bprotection added\b"
    ],


    # --------------------------------------------------------
    # INTERFACE INTERFERENCE
    # --------------------------------------------------------

    "interface_interference": [

        r"\bcontinue without\b",

        r"\bno thanks\b",

        r"\badd protection\b",

        r"\badd insurance\b",

        r"\badd trip protection\b",

        r"\bchoose premium\b",

        r"\bselect premium\b",

        r"\bconfirm and pay\b",

        r"\bupgrade to\b",

        r"\bcontinue with protection\b",

        r"\bcontinue with premium\b"
    ],


    # --------------------------------------------------------
    # OBSTRUCTION
    # --------------------------------------------------------

    "obstruction": [

        r"\bcannot go back\b",

        r"\bcan'?t go back\b",

        r"\bcannot continue\b",

        r"\bcan'?t continue\b",

        r"\bcheckout locked\b",

        r"\bpayment is locked\b",

        r"\bpayment locked\b",

        r"\bpayment\s+locked\s+until\b",

        r"\blocked until\b",

        r"\bcannot be edited\b",

        r"\bcan'?t be edited\b",

        r"\bcannot return\b",

        r"\bcan'?t return\b",

        r"\bcomplete all steps\b",

        r"\bcomplete all required\b",

        r"\bmust complete\b",

        r"\bverification required\b",

        r"\bverification.*before continuing\b",

        r"\bcontinue is disabled\b",

        r"\bcannot be modified\b",

        r"\bchanges are not allowed\b",

        r"\breturn option is unavailable\b"
    ]
}


# ============================================================
# EXACT EVIDENCE
# ============================================================

def exact_evidence(
    text,
    label
):

    normalized = normalize_text(
        text
    )

    matches = []

    for pattern in EVIDENCE.get(
        label,
        []
    ):

        if re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE
        ):

            matches.append(
                pattern
            )

    return matches


# ============================================================
# FUZZY KEY PHRASE DETECTION
# ============================================================
#
# Helps when OCR slightly corrupts words.
# This is intentionally limited to high-value phrases.
# ============================================================

FUZZY_PHRASES = {

    "sneaking": [
        "selected automatically",
        "automatically selected",
        "already selected",
        "automatically added",
        "already added",
        "protection added"
    ],

    "obstruction": [
        "payment locked",
        "payment locked until",
        "locked until verification",
        "verification required",
        "cannot continue",
        "complete all steps"
    ],

    "false_urgency": [
        "high demand",
        "limited availability",
        "hurry",
        "book before",
        "prices go up"
    ],

    "hidden_fees": [
        "service fee",
        "booking fee",
        "cleaning fee",
        "property cleaning fee",
        "mandatory fee",
        "convenience fee"
    ],

    "misdirection": [
        "most popular",
        "best value",
        "recommended",
        "top choice",
        "guest favourite",
        "guest favorite"
    ],

    "interface_interference": [
        "continue without",
        "no thanks",
        "continue with protection",
        "add protection",
        "confirm and pay"
    ]
}


def fuzzy_similarity(
    text,
    phrase
):

    text = normalize_text(
        text
    )

    phrase = normalize_text(
        phrase
    )

    if not text or not phrase:
        return 0.0

    # Direct substring gets maximum score
    if phrase in text:
        return 1.0

    text_words = text.split()
    phrase_words = phrase.split()

    if len(text_words) < len(
        phrase_words
    ):
        return 0.0

    best = 0.0

    window_size = len(
        phrase_words
    )

    for i in range(
        len(text_words)
        - window_size
        + 1
    ):

        window = " ".join(
            text_words[
                i:i + window_size
            ]
        )

        score = SequenceMatcher(
            None,
            window,
            phrase
        ).ratio()

        best = max(
            best,
            score
        )

    return best


def fuzzy_evidence(
    text,
    label
):

    matches = []

    phrases = FUZZY_PHRASES.get(
        label,
        []
    )

    for phrase in phrases:

        score = fuzzy_similarity(
            text,
            phrase
        )

        # Conservative threshold
        if score >= 0.82:

            matches.append(
                phrase
            )

    return matches


# ============================================================
# ALL-CLASS EVIDENCE
# ============================================================

def get_all_evidence(
    text
):

    detected = {}

    for label in EVIDENCE:

        exact = exact_evidence(
            text,
            label
        )

        fuzzy = fuzzy_evidence(
            text,
            label
        )

        combined = []

        for item in (
            exact + fuzzy
        ):

            if item not in combined:

                combined.append(
                    item
                )

        if combined:

            detected[label] = {
                "score":
                    len(combined),

                "matches":
                    combined
            }

    return detected


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_chunk(
    text
):

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

        outputs = model(
            **inputs
        )

        probabilities = (
            torch.softmax(
                outputs.logits,
                dim=1
            )[0]
        )

    prediction_id = int(
        torch.argmax(
            probabilities
        ).item()
    )

    predicted_label = (
        model.config.id2label[
            prediction_id
        ]
    )

    confidence = float(
        probabilities[
            prediction_id
        ]
    )

    return (
        predicted_label,
        confidence
    )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(
    image_path
):

    print("\n" + "=" * 75)
    print("STEP 1 - IMAGE")
    print("=" * 75)

    print(
        "\nImage:",
        image_path
    )


    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("STEP 2 - MULTI-PASS OCR")
    print("=" * 75)

    raw_text = extract_text(
        image_path
    )

    if not raw_text.strip():

        print(
            "\nNo text detected."
        )

        return


    # --------------------------------------------------------
    # CHUNKS
    # --------------------------------------------------------

    chunks = create_chunks(
        raw_text
    )

    print(
        "\nUsable chunks:",
        len(chunks)
    )


    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("STEP 3 - MULTI-PATTERN ANALYSIS")
    print("=" * 75)

    accepted = []

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        predicted_label, confidence = (
            predict_chunk(
                chunk
            )
        )

        all_evidence = (
            get_all_evidence(
                chunk
            )
        )

        print(
            f"\n[{index}] "
            f"MODEL="
            f"{predicted_label:<25} "
            f"CONF="
            f"{confidence * 100:6.2f}%"
        )

        print(
            "    ",
            chunk
        )

        if all_evidence:

            print(
                "    Evidence:"
            )

            for (
                evidence_label,
                info
            ) in all_evidence.items():

                print(
                    f"       "
                    f"{evidence_label}: "
                    f"{info['score']}"
                )

                print(
                    "          ",
                    info["matches"]
                )


        # ----------------------------------------------------
        # Accept independently for EVERY class.
        #
        # Evidence count >= 1 AND:
        #
        #   model confidence >= 0.40
        #
        # OR:
        #
        #   strong evidence >= 2
        # ----------------------------------------------------

        for (
            evidence_label,
            info
        ) in all_evidence.items():

            evidence_count = (
                info["score"]
            )

            model_support = (
                confidence >= 0.40
            )

            strong_evidence = (
                evidence_count >= 2
            )

            if (
                model_support
                or strong_evidence
            ):

                accepted.append({

                    "label":
                        evidence_label,

                    "confidence":
                        confidence,

                    "evidence":
                        evidence_count,

                    "text":
                        chunk,

                    "matches":
                        info["matches"]
                })


    # ========================================================
    # GROUP RESULTS
    # ========================================================

    grouped = {}

    for item in accepted:

        label = item["label"]

        if label not in grouped:

            grouped[label] = {

                "signals": 0,

                "chunks": 0,

                "best_confidence": 0.0,

                "best_text": "",

                "matches": []
            }

        grouped[label]["signals"] += (
            item["evidence"]
        )

        grouped[label]["chunks"] += 1

        if (
            item["confidence"]
            >
            grouped[label][
                "best_confidence"
            ]
        ):

            grouped[label][
                "best_confidence"
            ] = item["confidence"]

            grouped[label][
                "best_text"
            ] = item["text"]

        for match in item["matches"]:

            if match not in grouped[
                label
            ]["matches"]:

                grouped[label][
                    "matches"
                ].append(match)


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 75)
    print("STEP 4 - DARPAN FINAL RESULT")
    print("=" * 75)


    if not grouped:

        print(
            "\nDetected Patterns : normal"
        )

        print(
            "Reason             : "
            "No strong pattern-specific "
            "evidence detected."
        )

        print("\n" + "=" * 75)
        print(
            "DARPAN ANALYSIS COMPLETED"
        )
        print("=" * 75)

        return


    # --------------------------------------------------------
    # Rank patterns.
    # --------------------------------------------------------

    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            item[1]["signals"],
            item[1]["chunks"],
            item[1]["best_confidence"]
        ),
        reverse=True
    )


    print(
        "\nDetected Patterns:"
    )


    for position, (
        label,
        info
    ) in enumerate(
        ranked,
        start=1
    ):

        print(
            f"\n{position}. {label}"
        )

        print(
            "   Model Confidence  :",
            f"{info['best_confidence'] * 100:.2f}%"
        )

        print(
            "   Evidence Signals  :",
            info["signals"]
        )

        print(
            "   Supporting Chunks :",
            info["chunks"]
        )

        print(
            "   Evidence Matches  :",
            ", ".join(
                info["matches"]
            )
        )

        print(
            "   Evidence Text:"
        )

        print(
            "   ",
            info["best_text"]
        )


    print("\n" + "=" * 75)

    print(
        "TOTAL PATTERNS DETECTED:",
        len(ranked)
    )

    print("=" * 75)

    print(
        "DARPAN ANALYSIS COMPLETED"
    )

    print("=" * 75)


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            r"python src\pipeline.py "
            r"data\test_images\test_booking.png"
        )

        sys.exit(1)


    image_path = sys.argv[1]


    try:

        analyze_image(
            image_path
        )

    except Exception as e:

        print(
            "\n" + "=" * 75
        )

        print(
            "DARPAN ERROR"
        )

        print(
            "=" * 75
        )

        print(
            "\n",
            str(e)
        )