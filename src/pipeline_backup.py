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

try:
    from sentence_transformers import SentenceTransformer, util
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False


# ============================================================
# DARPAN - FINAL TEXT DETECTION PIPELINE
#
# IMAGE
#   ↓
# MULTI-PASS OCR
#   ↓
# OCR CLEANING
#   ↓
# SMART TEXT RECONSTRUCTION
#   ↓
# SMART CHUNKING
#   ↓
# DISTILBERT
#   ↓
# EXACT + FUZZY + SEMANTIC EVIDENCE
#   ↓
# FINAL VALIDATION
#   ↓
# CONFIRMED PATTERNS
#
# The pipeline does NOT depend on one particular phrase.
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
    r"D:\Tesseract-OCR\tesseract.exe"
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
# THRESHOLDS
# ============================================================

# Minimum model confidence when model and
# evidence support the same pattern.
MODEL_SUPPORT_THRESHOLD = 0.40

# Fuzzy evidence is less reliable than exact evidence.
STRONG_MODEL_THRESHOLD = 0.70

# Semantic similarity threshold.
SEMANTIC_THRESHOLD = 0.82

# Very strong semantic similarity.
STRONG_SEMANTIC_THRESHOLD = 0.88

# Strong evidence score.
STRONG_EVIDENCE_SCORE = 3.0

# Maximum text length for one chunk.
MAX_CHUNK_LENGTH = 300

# Number of nearby OCR lines to combine.
MAX_COMBINE_LINES = 4


# ============================================================
# START
# ============================================================

print("=" * 75)
print("DARPAN - FINAL TEXT DETECTION PIPELINE")
print("=" * 75)

print("\nDevice:", device)


# ============================================================
# LOAD DISTILBERT
# ============================================================

print("\nLoading trained DistilBERT model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.to(device)
model.eval()

print(
    "DistilBERT model loaded successfully."
)


# ============================================================
# LOAD SEMANTIC MODEL
# ============================================================

semantic_model = None

if SEMANTIC_AVAILABLE:

    print(
        "\nLoading semantic evidence model..."
    )

    try:

        semantic_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print(
            "Semantic model loaded successfully."
        )

    except Exception as e:

        print(
            "\nWARNING: Semantic model could not be loaded."
        )

        print(
            "Semantic evidence disabled."
        )

        print(
            "Reason:",
            str(e)
        )

else:

    print(
        "\nWARNING: sentence-transformers is not installed."
    )

    print(
        "Install using:"
    )

    print(
        "py -3.11 -m pip install sentence-transformers"
    )


# ============================================================
# SEMANTIC REFERENCE EXAMPLES
#
# These represent the MEANING/concept of each pattern.
# They are not intended to cover every possible wording.
# ============================================================

SEMANTIC_EXAMPLES = {

    "false_urgency": [

        "Only a small number of rooms are remaining",

        "Only a few seats are available",

        "The availability of this option is very limited",

        "This room is in high demand",

        "Availability is running low",

        "The user is pressured to book quickly",

        "The offer creates pressure to act immediately",

        "The price may increase if the user waits",

        "The deal is available only for a limited time",

        "The website creates a sense of scarcity"
    ],


    "hidden_fees": [

        "Additional fees are added later",

        "Extra charges are added at checkout",

        "A mandatory fee is charged in addition to the displayed price",

        "An additional service fee is charged",

        "A booking fee is added",

        "A cleaning fee is added",

        "The final amount contains additional charges",

        "Fees are revealed later in the booking process",

        "The displayed price does not include all mandatory charges",

        "Additional charges appear during checkout"
    ],


    "sneaking": [

        "An additional option has already been selected",

        "Protection has already been added",

        "Insurance is selected automatically",

        "An extra service is automatically included",

        "An additional product is added to the booking",

        "The option was selected without the user choosing it",

        "The website automatically adds an extra service",

        "An additional charge is included by default",

        "The user is enrolled in an additional service automatically"
    ],


    "misdirection": [

        "One option is presented as the recommended choice",

        "The website highlights one option as the preferred choice",

        "A particular option is emphasized over other choices",

        "The interface directs attention toward one option",

        "One choice is visually or textually promoted over alternatives",

        "The website guides the user toward a particular option"
    ],


    "interface_interference": [

        "The preferred option is made more prominent",

        "The user is encouraged to select the premium option",

        "The upgrade option is emphasized",

        "The interface encourages the user to add protection",

        "The premium option receives more prominent treatment",

        "The interface makes one choice easier to select than another"
    ],


    "obstruction": [

        "The user cannot continue without completing a step",

        "The booking cannot proceed",

        "Verification is required before continuing",

        "Payment is locked until verification is completed",

        "The user cannot go back",

        "The booking cannot be modified",

        "Changes are not allowed",

        "A required step prevents the user from proceeding",

        "The return option is unavailable"
    ]
}


# ============================================================
# CREATE SEMANTIC EMBEDDINGS
# ============================================================

SEMANTIC_EMBEDDINGS = {}

if semantic_model is not None:

    print(
        "\nCreating semantic reference embeddings..."
    )

    for pattern, examples in SEMANTIC_EXAMPLES.items():

        SEMANTIC_EMBEDDINGS[pattern] = (
            semantic_model.encode(
                examples,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        )

    print(
        "Semantic reference embeddings created."
    )


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

    # Increase resolution.
    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )


    # ========================================================
    # OCR PASS 1
    # ========================================================

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


    # ========================================================
    # OCR PASS 2
    # ========================================================

    text_2 = pytesseract.image_to_string(
        gray,
        config="--psm 11"
    )


    # ========================================================
    # OCR PASS 3
    # ========================================================

    processed_3 = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    text_3 = pytesseract.image_to_string(
        processed_3,
        config="--psm 11"
    )


    return (
        text_1
        + "\n"
        + text_2
        + "\n"
        + text_3
    )


# ============================================================
# NORMALIZE OCR
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    # Remove common symbols.
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


    # Common OCR mistakes.
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

        "iocked":
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


    # Keep useful characters.
    text = re.sub(
        r"[^\w\s$₹€£%+&:/.,!?'-]",
        " ",
        text
    )

    # Normalize spaces.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# GARBAGE FILTER
# ============================================================

def is_garbage(text):

    text = normalize_text(
        text
    )

    if len(text) < 3:
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

    if meaningful < 2:
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
# LINE QUALITY
# ============================================================

def is_useful_line(text):

    text = normalize_text(
        text
    )

    if not text:
        return False

    if is_garbage(text):
        return False

    return True


# ============================================================
# SMART TEXT RECONSTRUCTION
#
# OCR can produce:
#
#   Only
#   3 rooms
#   left
#
# instead of:
#
#   Only 3 rooms left
#
# This function reconstructs nearby lines.
# ============================================================

def reconstruct_lines(text):

    raw_lines = text.splitlines()

    lines = []

    for raw_line in raw_lines:

        line = normalize_text(
            raw_line
        )

        if not line:
            continue

        if is_useful_line(line):

            lines.append(
                line
            )

    return lines


# ============================================================
# CREATE SMART CHUNKS
#
# Generates:
#
# 1. Individual OCR lines
# 2. Adjacent line combinations
# 3. Sentence-level text
#
# This makes the pipeline robust to OCR fragmentation.
# ============================================================

def create_chunks(text):

    lines = reconstruct_lines(
        text
    )

    candidates = []


    # ========================================================
    # INDIVIDUAL LINES
    # ========================================================

    for line in lines:

        if len(line) <= MAX_CHUNK_LENGTH:

            candidates.append(
                line
            )


    # ========================================================
    # ADJACENT LINE RECONSTRUCTION
    # ========================================================

    for start in range(
        len(lines)
    ):

        combined = ""

        for end in range(
            start,
            min(
                start + MAX_COMBINE_LINES,
                len(lines)
            )
        ):

            if combined:

                combined += " "

            combined += lines[end]

            if (
                len(combined)
                <= MAX_CHUNK_LENGTH
            ):

                candidates.append(
                    combined
                )

            else:

                break


    # ========================================================
    # SENTENCE SPLITTING
    # ========================================================

    sentence_candidates = []

    for chunk in candidates:

        if len(chunk) <= 350:

            sentence_candidates.append(
                chunk
            )

            continue


        parts = re.split(
            r"(?<=[.!?])\s+",
            chunk
        )

        for part in parts:

            part = normalize_text(
                part
            )

            if (
                len(part) >= 3
                and not is_garbage(part)
            ):

                sentence_candidates.append(
                    part
                )


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    unique_chunks = []

    seen = set()

    for chunk in sentence_candidates:

        chunk = normalize_text(
            chunk
        )

        if not chunk:
            continue

        if is_garbage(chunk):
            continue

        key = chunk.lower()

        if key not in seen:

            seen.add(key)

            unique_chunks.append(
                chunk
            )


    return unique_chunks


# ============================================================
# EXACT EVIDENCE
#
# These are strong linguistic indicators.
#
# They are NOT the only source of evidence.
# ============================================================

EVIDENCE = {

    # ========================================================
    # FALSE URGENCY
    # ========================================================

    "false_urgency": [

        r"\bhurry\b",

        r"\blimited availability\b",

        r"\blimited time\b",

        r"\blimited offer\b",

        r"\bhigh demand\b",

        r"\bin high demand\b",

        r"\bonly\s+\d+\s+(?:room|rooms|seat|seats|ticket|tickets|spot|spots)\b",

        r"\b\d+\s+(?:room|rooms|seat|seats|ticket|tickets|spot|spots)\s+left\b",

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

        r"\bprices?\s+go\s+up\b",

        r"\bprices?\s+may\s+increase\b",

        r"\bprice increases?\b",

        r"\bfirst come, first served\b",

        r"\bavailability\s+is\s+running\s+low\b",

        r"\balmost\s+sold\s+out\b",

        r"\blast\s+few\b"
    ],


    # ========================================================
    # HIDDEN FEES
    # ========================================================

    "hidden_fees": [

        r"\bmandatory\s+(?:\w+\s+){0,5}fee\b",

        r"\badditional\s+(?:\w+\s+){0,5}fee\b",

        r"\badditional\s+(?:\w+\s+){0,5}fees\b",

        r"\bextra\s+(?:\w+\s+){0,5}charges?\b",

        r"\bservice\s+fee\b",

        r"\bbooking\s+fee\b",

        r"\bconvenience\s+fee\b",

        r"\bprocessing\s+fee\b",

        r"\bcleaning\s+fee\b",

        r"\bproperty\s+cleaning\s+fee\b",

        r"\bresort\s+fee\b",

        r"\bfees?\s+(?:will\s+be\s+)?added\s+(?:later|at\s+checkout)\b",

        r"\bcharges?\s+(?:will\s+be\s+)?added\s+(?:later|at\s+checkout)\b",

        r"\bfees?\s+calculated\s+later\b",

        r"\badditional\s+charges?\s+apply\b"
    ],


    # ========================================================
    # MISDIRECTION
    # ========================================================

    "misdirection": [

        r"\brecommended\s+for\s+you\b",

        r"\bsuggested\s+for\s+you\b",

        r"\btop\s+pick\s+for\s+you\b",

        r"\bpopular\s+choice\b",

        r"\bmost\s+popular\s+choice\b",

        r"\bguest\s+favo[u]?rite\b",

        r"\bmost\s+booked\s+option\b",

        r"\bmost\s+chosen\s+option\b",

        r"\bsmart\s+choice\b",

        r"\bour\s+recommendation\b",

        r"\bdeal\s+of\s+the\s+day\b",

        r"\bspecial\s+pick\b"
    ],


    # ========================================================
    # SNEAKING
    # ========================================================

    "sneaking": [

        r"\balready\s+selected\b",

        r"\bselected\s+automatically\b",

        r"\bautomatically\s+selected\b",

        r"\bautomatically\s+added\b",

        r"\balready\s+added\b",

        r"\bhas\s+been\s+added\b",

        r"\badded\s+to\s+your\s+booking\b",

        r"\badded\s+to\s+booking\b",

        r"\bprotection\s+included\b",

        r"\binsurance\s+included\b",

        r"\bdonation\s+added\b",

        r"\bcontribution\s+added\b",

        r"\bprotection\s+added\b",

        r"\bincluded\s+by\s+default\b",

        r"\bpre[- ]selected\b",

        r"\bselected\s+by\s+default\b"
    ],


    # ========================================================
    # INTERFACE INTERFERENCE
    # ========================================================

    "interface_interference": [

        r"\bcontinue\s+without\b",

        r"\bno\s+thanks\b",

        r"\badd\s+protection\b",

        r"\badd\s+insurance\b",

        r"\badd\s+trip\s+protection\b",

        r"\bchoose\s+premium\b",

        r"\bselect\s+premium\b",

        r"\bcontinue\s+with\s+protection\b",

        r"\bcontinue\s+with\s+premium\b",

        r"\bupgrade\s+to\b",

        r"\bupgrade\s+your\b",

        r"\bdecline\s+protection\b"
    ],


    # ========================================================
    # OBSTRUCTION
    # ========================================================

    "obstruction": [

        r"\bcannot\s+go\s+back\b",

        r"\bcan'?t\s+go\s+back\b",

        r"\bcannot\s+continue\b",

        r"\bcan'?t\s+continue\b",

        r"\bcheckout\s+locked\b",

        r"\bpayment\s+is\s+locked\b",

        r"\bpayment\s+locked\b",

        r"\bpayment\s+locked\s+until\b",

        r"\blocked\s+until\b",

        r"\bcannot\s+be\s+edited\b",

        r"\bcan'?t\s+be\s+edited\b",

        r"\bcannot\s+return\b",

        r"\bcan'?t\s+return\b",

        r"\bcomplete\s+all\s+steps\b",

        r"\bcomplete\s+all\s+required\b",

        r"\bmust\s+complete\b",

        r"\bverification\s+required\b",

        r"\bverification.*before\s+continuing\b",

        r"\bcontinue\s+is\s+disabled\b",

        r"\bcannot\s+be\s+modified\b",

        r"\bchanges\s+are\s+not\s+allowed\b",

        r"\breturn\s+option\s+is\s+unavailable\b"
    ]
}


# ============================================================
# EXACT MATCH
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
# FUZZY PHRASES
#
# Used mainly for OCR errors.
# ============================================================

FUZZY_PHRASES = {

    "false_urgency": [

        "high demand",

        "limited availability",

        "hurry",

        "book before",

        "prices go up",

        "availability is running low",

        "almost sold out"
    ],


    "hidden_fees": [

        "service fee",

        "booking fee",

        "cleaning fee",

        "mandatory fee",

        "additional fee",

        "extra charges",

        "fees calculated later"
    ],


    "sneaking": [

        "selected automatically",

        "automatically selected",

        "already selected",

        "automatically added",

        "already added",

        "protection added",

        "included by default"
    ],


    "misdirection": [

        "recommended for you",

        "top pick for you",

        "popular choice",

        "guest favourite",

        "guest favorite",

        "most booked option"
    ],


    "interface_interference": [

        "continue without",

        "no thanks",

        "add protection",

        "continue with protection",

        "choose premium",

        "upgrade to"
    ],


    "obstruction": [

        "payment locked",

        "payment locked until",

        "locked until verification",

        "verification required",

        "cannot continue",

        "complete all steps"
    ]
}


# ============================================================
# FUZZY SIMILARITY
# ============================================================

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

    if (
        not text
        or not phrase
    ):

        return 0.0


    # Exact substring.
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


# ============================================================
# FUZZY EVIDENCE
# ============================================================

def fuzzy_evidence(
    text,
    label
):

    matches = []

    for phrase in FUZZY_PHRASES.get(
        label,
        []
    ):

        score = fuzzy_similarity(
            text,
            phrase
        )


        if score >= 0.82:

            matches.append(
                {
                    "phrase": phrase,
                    "score": score
                }
            )


    return matches


# ============================================================
# SEMANTIC EVIDENCE
# ============================================================

def semantic_evidence(
    text,
    label
):

    if (
        semantic_model is None
        or
        label not in SEMANTIC_EMBEDDINGS
    ):

        return {
            "found": False,
            "score": 0.0,
            "matched_example": None
        }


    normalized = normalize_text(
        text
    )


    if len(normalized) < 5:

        return {
            "found": False,
            "score": 0.0,
            "matched_example": None
        }


    try:

        text_embedding = (
            semantic_model.encode(
                normalized,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        )


        similarities = util.cos_sim(
            text_embedding,
            SEMANTIC_EMBEDDINGS[label]
        )[0]


        best_index = int(
            similarities.argmax().item()
        )


        best_score = float(
            similarities[
                best_index
            ].item()
        )


        best_example = (
            SEMANTIC_EXAMPLES[label][
                best_index
            ]
        )


        return {

            "found":
                best_score >= SEMANTIC_THRESHOLD,

            "score":
                best_score,

            "matched_example":
                best_example
        }


    except Exception:

        return {
            "found": False,
            "score": 0.0,
            "matched_example": None
        }


# ============================================================
# ALL EVIDENCE
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


        semantic = semantic_evidence(
            text,
            label
        )


        # ====================================================
        # EVIDENCE SCORE
        #
        # Exact  = 2
        # Fuzzy  = 1.5
        # Semantic = 1
        # ====================================================

        evidence_score = 0.0


        evidence_score += (
            len(exact) * 2.0
        )


        evidence_score += (
            len(fuzzy) * 1.5
        )


        if semantic["found"]:

            evidence_score += 1.0


        if (
            exact
            or
            fuzzy
            or
            semantic["found"]
        ):

            detected[label] = {

                "exact":
                    exact,

                "fuzzy":
                    fuzzy,

                "semantic":
                    semantic,

                "evidence_score":
                    evidence_score
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
        ].item()
    )


    return (
        predicted_label,
        confidence
    )


# ============================================================
# FINAL DECISION
# ============================================================

def should_accept_pattern(
    pattern_label,
    predicted_label,
    model_confidence,
    evidence
):

    exact_count = len(
        evidence["exact"]
    )

    fuzzy_count = len(
        evidence["fuzzy"]
    )

    semantic_found = (
        evidence["semantic"]["found"]
    )

    semantic_score = (
        evidence["semantic"]["score"]
    )

    evidence_score = (
        evidence["evidence_score"]
    )


    # ========================================================
    # CASE 1:
    # MODEL + EVIDENCE AGREE
    # ========================================================

    if pattern_label == predicted_label:

        # Exact evidence + reasonable model support.
        if (
            exact_count >= 1
            and
            model_confidence
            >= MODEL_SUPPORT_THRESHOLD
        ):

            return True


        # Fuzzy evidence requires stronger model confidence.
        if (
            fuzzy_count >= 1
            and
            model_confidence
            >= STRONG_MODEL_THRESHOLD
        ):

            return True


        # Strong semantic evidence.
        if (
            semantic_found
            and
            semantic_score
            >= STRONG_SEMANTIC_THRESHOLD
            and
            model_confidence
            >= MODEL_SUPPORT_THRESHOLD
        ):

            return True


        # Multiple evidence signals.
        if (
            evidence_score
            >= STRONG_EVIDENCE_SCORE
            and
            model_confidence
            >= MODEL_SUPPORT_THRESHOLD
        ):

            return True


    # ========================================================
    # CASE 2:
    # MODEL SAYS NORMAL
    # BUT EVIDENCE IS STRONG
    #
    # This handles hard examples.
    # ========================================================

    if predicted_label == "normal":

        # Two independent exact signals.
        if exact_count >= 2:

            return True


        # Strong combined evidence.
        if (
            evidence_score
            >= STRONG_EVIDENCE_SCORE
        ):

            return True


        # Very strong semantic evidence + another
        # supporting signal.
        if (
            semantic_found
            and
            semantic_score
            >= STRONG_SEMANTIC_THRESHOLD
            and
            (
                exact_count
                + fuzzy_count
            ) >= 1
        ):

            return True


    # ========================================================
    # CASE 3:
    # MODEL PREDICTS ONE PATTERN
    # BUT EVIDENCE SUPPORTS ANOTHER
    # ========================================================

    if (
        pattern_label != predicted_label
        and
        predicted_label != "normal"
    ):

        # Strong exact evidence.
        if (
            exact_count >= 2
            and
            evidence_score
            >= STRONG_EVIDENCE_SCORE
        ):

            return True


        # Strong semantic + supporting evidence.
        if (
            semantic_found
            and
            semantic_score
            >= STRONG_SEMANTIC_THRESHOLD
            and
            evidence_score
            >= STRONG_EVIDENCE_SCORE
        ):

            return True


    return False


# ============================================================
# ANALYZE IMAGE
# ============================================================

def analyze_image(
    image_path
):

    print(
        "\n" + "=" * 75
    )

    print(
        "STEP 1 - IMAGE"
    )

    print(
        "=" * 75
    )

    print(
        "\nImage:",
        image_path
    )


    # ========================================================
    # OCR
    # ========================================================

    print(
        "\n" + "=" * 75
    )

    print(
        "STEP 2 - MULTI-PASS OCR"
    )

    print(
        "=" * 75
    )


    raw_text = extract_text(
        image_path
    )


    if not raw_text.strip():

        print(
            "\nNo text detected."
        )

        return


    # ========================================================
    # CHUNKS
    # ========================================================

    chunks = create_chunks(
        raw_text
    )


    print(
        "\nUsable chunks:",
        len(chunks)
    )


    # ========================================================
    # ANALYSIS
    # ========================================================

    print(
        "\n" + "=" * 75
    )

    print(
        "STEP 3 - TEXT MODEL + EVIDENCE ANALYSIS"
    )

    print(
        "=" * 75
    )


    accepted = []


    # ========================================================
    # PROCESS EACH CHUNK
    # ========================================================

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        predicted_label, confidence = (
            predict_chunk(
                chunk
            )
        )


        # Evidence is ALWAYS checked independently.
        all_evidence = get_all_evidence(
            chunk
        )


        print(
            f"\n[{index}] "
            f"MODEL={predicted_label:<25}"
            f"CONF={confidence * 100:6.2f}%"
        )


        print(
            "     Text:",
            chunk
        )


        # ----------------------------------------------------
        # No evidence
        # ----------------------------------------------------

        if not all_evidence:

            print(
                "     Evidence: NONE"
            )

            continue


        # ----------------------------------------------------
        # Evidence found
        # ----------------------------------------------------

        for (
            evidence_label,
            evidence
        ) in all_evidence.items():

            print(
                f"\n     Pattern Evidence: "
                f"{evidence_label}"
            )


            print(
                "     Evidence Score:",
                f"{evidence['evidence_score']:.2f}"
            )


            # =================================================
            # EXACT
            # =================================================

            if evidence["exact"]:

                print(
                    "     Exact Evidence:"
                )

                for match in evidence["exact"]:

                    print(
                        "        ",
                        match
                    )


            # =================================================
            # FUZZY
            # =================================================

            if evidence["fuzzy"]:

                print(
                    "     Fuzzy Evidence:"
                )

                for item in evidence["fuzzy"]:

                    print(
                        "        ",
                        item["phrase"],
                        "->",
                        f"{item['score'] * 100:.2f}%"
                    )


            # =================================================
            # SEMANTIC
            # =================================================

            semantic = evidence[
                "semantic"
            ]


            if semantic["found"]:

                print(
                    "     Semantic Evidence:"
                )

                print(
                    "        Similarity:",
                    f"{semantic['score'] * 100:.2f}%"
                )

                print(
                    "        Reference:",
                    semantic["matched_example"]
                )

            else:

                if semantic["score"] > 0:

                    print(
                        "     Semantic Similarity:",
                        f"{semantic['score'] * 100:.2f}%"
                    )

                    print(
                        "     Semantic Evidence: NOT STRONG"
                    )


            # =================================================
            # FINAL DECISION
            # =================================================

            accepted_pattern = (
                should_accept_pattern(
                    evidence_label,
                    predicted_label,
                    confidence,
                    evidence
                )
            )


            if accepted_pattern:

                print(
                    "     Decision: ACCEPTED"
                )


                accepted.append(
                    {

                        "label":
                            evidence_label,

                        "confidence":
                            confidence,

                        "evidence_score":
                            evidence[
                                "evidence_score"
                            ],

                        "semantic_score":
                            evidence[
                                "semantic"
                            ]["score"],

                        "text":
                            chunk,

                        "evidence":
                            evidence
                    }
                )


            else:

                print(
                    "     Decision: REJECTED"
                )


    # ========================================================
    # GROUP RESULTS
    # ========================================================

    grouped = {}


    for item in accepted:

        label = item["label"]


        if label not in grouped:

            grouped[label] = {

                "signals":
                    0.0,

                "chunks":
                    0,

                "best_confidence":
                    0.0,

                "best_semantic":
                    0.0,

                "best_text":
                    "",

                "exact_matches":
                    [],

                "fuzzy_matches":
                    [],

                "semantic_matches":
                    []
            }


        grouped[label]["signals"] += (
            item["evidence_score"]
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


        grouped[label]["best_semantic"] = max(
            grouped[label]["best_semantic"],
            item["semantic_score"]
        )


        evidence = item["evidence"]


        # Exact matches.
        for match in evidence["exact"]:

            if match not in grouped[
                label
            ]["exact_matches"]:

                grouped[
                    label
                ]["exact_matches"].append(
                    match
                )


        # Fuzzy matches.
        for fuzzy in evidence["fuzzy"]:

            phrase = fuzzy["phrase"]

            if phrase not in grouped[
                label
            ]["fuzzy_matches"]:

                grouped[
                    label
                ]["fuzzy_matches"].append(
                    phrase
                )


        # Semantic matches.
        semantic_match = (
            evidence[
                "semantic"
            ]["matched_example"]
        )


        if (
            semantic_match
            and
            semantic_match
            not in grouped[
                label
            ]["semantic_matches"]
        ):

            grouped[
                label
            ]["semantic_matches"].append(
                semantic_match
            )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print(
        "\n" + "=" * 75
    )

    print(
        "STEP 4 - DARPAN FINAL RESULT"
    )

    print(
        "=" * 75
    )


    # ========================================================
    # NO CONFIRMED PATTERN
    # ========================================================

    if not grouped:

        print(
            "\nDetected Patterns:"
        )

        print(
            "NORMAL / NO CONFIRMED DARK PATTERN"
        )

        print(
            "\nReason:"
        )

        print(
            "No sufficiently strong pattern-specific "
            "evidence was found."
        )

        print(
            "\nModel predictions alone were NOT treated "
            "as confirmed dark patterns."
        )

        print(
            "\n" + "=" * 75
        )

        print(
            "DARPAN ANALYSIS COMPLETED"
        )

        print(
            "=" * 75
        )

        return


    # ========================================================
    # RANK PATTERNS
    # ========================================================

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
            "   Model Confidence :",
            f"{info['best_confidence'] * 100:.2f}%"
        )


        print(
            "   Evidence Score   :",
            f"{info['signals']:.2f}"
        )


        print(
            "   Supporting Chunks:",
            info["chunks"]
        )


        print(
            "   Best Semantic    :",
            f"{info['best_semantic'] * 100:.2f}%"
        )


        if info["exact_matches"]:

            print(
                "   Exact Evidence:"
            )

            for match in info["exact_matches"]:

                print(
                    "      ",
                    match
                )


        if info["fuzzy_matches"]:

            print(
                "   Fuzzy Evidence:"
            )

            for match in info["fuzzy_matches"]:

                print(
                    "      ",
                    match
                )


        if info["semantic_matches"]:

            print(
                "   Semantic Evidence:"
            )

            for match in info["semantic_matches"]:

                print(
                    "      ",
                    match
                )


        print(
            "   Supporting Text:"
        )

        print(
            "      ",
            info["best_text"]
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 75
    )


    print(
        "TOTAL PATTERNS DETECTED:",
        len(ranked)
    )


    print(
        "\nPrimary Pattern:",
        ranked[0][0]
    )


    print(
        "Primary Confidence:",
        f"{ranked[0][1]['best_confidence'] * 100:.2f}%"
    )


    print(
        "Primary Evidence Score:",
        f"{ranked[0][1]['signals']:.2f}"
    )


    print(
        "\nAll Detected Patterns:"
    )


    for label, info in ranked:

        print(
            f"   {label:<25}"
            f"| confidence="
            f"{info['best_confidence'] * 100:.2f}%"
            f"| evidence="
            f"{info['signals']:.2f}"
        )


    print(
        "\n" + "=" * 75
    )

    print(
        "DARPAN ANALYSIS COMPLETED"
    )

    print(
        "=" * 75
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            r"py -3.11 src\pipeline.py "
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

        sys.exit(1)