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
# DARPAN - ROBUST TEXT DARK-PATTERN DETECTION PIPELINE
#
# IMAGE
#   ↓
# POSITIONAL OCR
#   ↓
# VISUAL LINE RECONSTRUCTION
#   ↓
# TEXT CLEANING
#   ↓
# SMART CHUNKING
#   ↓
# DISTILBERT
#   ↓
# EXACT + FUZZY + SEMANTIC EVIDENCE
#   ↓
# CONTEXT / CONFIDENCE VALIDATION
#   ↓
# FINAL CONFIRMED PATTERNS
#
# IMPORTANT:
# - Does NOT depend on one fixed phrase.
# - Does NOT accept model prediction alone.
# - Does NOT generate hundreds of overlapping chunks.
# - Normal UI text is filtered before final classification.
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
# THRESHOLDS
# ============================================================

MODEL_SUPPORT_THRESHOLD = 0.40

STRONG_MODEL_THRESHOLD = 0.70

VERY_STRONG_MODEL_THRESHOLD = 0.85

FUZZY_THRESHOLD = 0.82

SEMANTIC_THRESHOLD = 0.82

STRONG_SEMANTIC_THRESHOLD = 0.88

STRONG_EVIDENCE_SCORE = 3.0

MAX_CHUNK_LENGTH = 300

MAX_COMBINE_LINES = 3

MIN_TEXT_LENGTH = 5

MIN_MEANINGFUL_RATIO = 0.25

# Similar chunks above this ratio are treated
# as duplicate/overlapping OCR fragments.
CHUNK_DUPLICATE_THRESHOLD = 0.90

# A dark-pattern phrase can be split across OCR lines. These phrases
# are kept together in a small local context window instead of sending
# the entire screenshot text to DistilBERT.
EVIDENCE_CONTEXT_LINES = 2
MAX_EVIDENCE_CONTEXT_WORDS = 80


# ============================================================
# START
# ============================================================

print("=" * 75)
print("DARPAN - ROBUST TEXT DARK-PATTERN DETECTION")
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

        "The website creates a sense of scarcity",

        "The website tells the user to hurry",

        "The website indicates that availability is almost gone"
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

        "The user is enrolled in an additional service automatically",

        "An optional service is preselected for the user"
    ],


    "misdirection": [

        "One option is presented as the recommended choice",

        "The website highlights one option as the preferred choice",

        "A particular option is emphasized over other choices",

        "The interface directs attention toward one option",

        "One choice is visually or textually promoted over alternatives",

        "The website guides the user toward a particular option",

        "One alternative is deliberately presented as the preferred choice"
    ],


    "interface_interference": [

        "The preferred option is made more prominent",

        "The user is encouraged to select the premium option",

        "The upgrade option is emphasized",

        "The interface encourages the user to add protection",

        "The premium option receives more prominent treatment",

        "The interface makes one choice easier to select than another",

        "The user is encouraged toward an expensive option"
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
# NORMALIZE OCR TEXT
# ============================================================

def normalize_text(text):

    text = str(text).lower()

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


    # --------------------------------------------------------
    # OCR correction
    # --------------------------------------------------------

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
            "continue",

        "hurr y":
            "hurry",

        "hurrv":
            "hurry",

        "hurr":
            "hurry",

        "availabiiity":
            "availability",

        "availabiity":
            "availability",

        "deman d":
            "demand",

        "autom atically":
            "automatically"
    }


    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )


    # --------------------------------------------------------
    # Join common OCR splits
    # --------------------------------------------------------

    text = re.sub(
        r"\bselect\s+ed\b",
        "selected",
        text
    )

    text = re.sub(
        r"\bautomatically\s+added\b",
        "automatically added",
        text
    )


    # --------------------------------------------------------
    # Common OCR exaggeration:
    # "hurryyyyy", "hurryyyy!", "hurryy" -> "hurry"
    # --------------------------------------------------------

    text = re.sub(
        r"\bhur+r+y+\b",
        "hurry",
        text,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Preserve useful UI characters.
    # --------------------------------------------------------

    text = re.sub(
        r"[^\w\s$₹€£%+&:/.,!?'\-]",
        " ",
        text
    )


    # --------------------------------------------------------
    # Normalize spaces.
    # --------------------------------------------------------

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

    if len(text) < MIN_TEXT_LENGTH:

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
        and
        ratio < MIN_MEANINGFUL_RATIO
    ):

        return True


    # --------------------------------------------------------
    # OCR fragment protection
    #
    # Examples:
    # vy
    # a
    # p 7
    # kkk kk
    # fey
    # --------------------------------------------------------

    words = text.split()


    if len(words) <= 2:

        alphabetic_words = [
            w for w in words
            if w.isalpha()
        ]


        if alphabetic_words:

            tiny_words = sum(
                len(w) <= 2
                for w in alphabetic_words
            )


            if (
                tiny_words
                ==
                len(alphabetic_words)
            ):

                return True


    return False


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


    # ========================================================
    # POSITIONAL OCR
    # ========================================================

    print(
        "\nRunning positional OCR..."
    )


    data = pytesseract.image_to_data(
        gray,
        config="--psm 11",
        output_type=pytesseract.Output.DICT
    )


    words = []


    for i in range(
        len(data["text"])
    ):

        text = str(
            data["text"][i]
        ).strip()


        if not text:

            continue


        try:

            confidence = float(
                data["conf"][i]
            )

        except:

            confidence = 0


        if confidence < 15:

            continue


        words.append({

            "text":
                text,

            "x":
                int(data["left"][i]),

            "y":
                int(data["top"][i]),

            "w":
                int(data["width"][i]),

            "h":
                int(data["height"][i]),

            "conf":
                confidence,

            "block":
                int(data["block_num"][i]),

            "par":
                int(data["par_num"][i]),

            "line":
                int(data["line_num"][i])
        })


    print(
        "OCR words detected:",
        len(words)
    )


    if not words:

        return ""


    # ========================================================
    # REMOVE DUPLICATE OCR WORDS
    # ========================================================

    unique_words = []

    seen = set()


    for word in words:

        key = (
            normalize_text(word["text"]),
            round(word["x"] / 10),
            round(word["y"] / 10)
        )


        if key in seen:

            continue


        seen.add(key)

        unique_words.append(
            word
        )


    words = unique_words


    print(
        "Unique OCR words:",
        len(words)
    )


    # ========================================================
    # VISUAL LINE RECONSTRUCTION
    #
    # Uses actual OCR coordinates.
    # ========================================================

    words.sort(
        key=lambda item: (
            item["y"],
            item["x"]
        )
    )


    lines = []


    for word in words:

        placed = False


        word_center_y = (
            word["y"]
            +
            word["h"] / 2
        )


        for line in lines:

            if not line:

                continue


            avg_center_y = sum(
                item["y"]
                +
                item["h"] / 2
                for item in line
            ) / len(line)


            avg_height = sum(
                item["h"]
                for item in line
            ) / len(line)


            vertical_distance = abs(
                word_center_y
                -
                avg_center_y
            )


            # Words with similar vertical position
            # belong to the same visual line.
            if (
                vertical_distance
                <=
                max(
                    avg_height * 0.65,
                    12
                )
            ):

                line.append(
                    word
                )

                placed = True

                break


        if not placed:

            lines.append(
                [word]
            )


    # ========================================================
    # SORT EACH LINE LEFT → RIGHT
    # ========================================================

    reconstructed_lines = []


    for line in lines:

        line.sort(
            key=lambda item:
                item["x"]
        )


        text = " ".join(
            item["text"]
            for item in line
        )


        text = normalize_text(
            text
        )


        if (
            text
            and
            not is_garbage(text)
        ):

            reconstructed_lines.append(
                text
            )


    print(
        "Visual lines reconstructed:",
        len(reconstructed_lines)
    )


    # ========================================================
    # RETURN LINES
    # ========================================================

    return "\n".join(
        reconstructed_lines
    )


# ============================================================
# RECONSTRUCT LINES
# ============================================================

def reconstruct_lines(text):

    lines = []


    for raw_line in text.splitlines():

        line = normalize_text(
            raw_line
        )


        if not line:

            continue


        if is_garbage(line):

            continue


        lines.append(
            line
        )


    return lines


# ============================================================
# DETECT UI / NORMAL TEXT
#
# These patterns are NOT automatically dark patterns.
# They are used to prevent model hallucinations from
# becoming final detections.
# ============================================================

NORMAL_UI_PATTERNS = [

    r"^\d[\d,.\s]*$",

    r"^\d[\d,.\s]*\s+(?:properties|reviews?)$",

    r"^\d[\d,.\s]*\s+(?:reviews?|rating)$",

    r"^\d+(?:\.\d+)?\s+(?:excellent|very good|good)$",

    r"^(?:sort by|sort by:)$",

    r"^best match$",

    r"^best price$",

    r"^price for$",

    r"^per night$",

    r"^properties per night$",

    r"^walking distance$",

    r"^location score$",

    r"^free cancellation$",

    r"^preferred$",

    r"^agoda preferred$",

    r"^text search$",

    r"^your budget$",

    r"^budget per night$",

    r"^(?:map|map data)$",

    r"^today'?s discounts$",

    r"^coupons?(?: & deals)?$",

    r"^see all coupons$"
]


def is_normal_ui_text(text):

    normalized = normalize_text(
        text
    )


    if not normalized:

        return True


    # --------------------------------------------------------
    # Exact normal UI patterns.
    # --------------------------------------------------------

    for pattern in NORMAL_UI_PATTERNS:

        if re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE
        ):

            return True


    # --------------------------------------------------------
    # Rating / review numbers.
    # --------------------------------------------------------

    if re.fullmatch(
        r"\d+(?:\.\d+)?\s+reviews?",
        normalized
    ):

        return True


    if re.fullmatch(
        r"\d+(?:\.\d+)?",
        normalized
    ):

        return True


    # --------------------------------------------------------
    # Hotel/location information.
    # --------------------------------------------------------

    if (
        "reviews" in normalized
        and
        not any(
            word in normalized
            for word in [
                "hurry",
                "limited",
                "only",
                "left",
                "demand"
            ]
        )
    ):

        return True


    # --------------------------------------------------------
    # Normal cancellation information.
    # --------------------------------------------------------

    if normalized == "free cancellation":

        return True


    return False


# ============================================================
# SMART CHUNKING
#
# IMPORTANT:
# Do NOT create every possible sliding window.
#
# We keep:
# 1. Full visual lines
# 2. Only meaningful adjacent combinations
# 3. Sentence-level combinations
#
# This prevents 400+ repeated chunks.
# ============================================================

def create_chunks(text):
    """
    Build small, meaningful model inputs.

    We intentionally do NOT send the entire OCR output as one input.
    Instead we keep:
      1. complete visual lines;
      2. small neighbouring-line context windows;
      3. evidence-centred windows around dark-pattern language.

    The evidence-centred windows are important for cases such as:
        "Only"
        "5 seats available"
    or:
        "Rooms in Goa are"
        "in high demand"
        "on your selected dates"
    """

    lines = reconstruct_lines(text)

    if not lines:
        return []

    candidates = []

    # ------------------------------------------------------------
    # 1. Keep complete visual lines.
    # ------------------------------------------------------------
    for line in lines:
        if len(line) <= MAX_CHUNK_LENGTH:
            candidates.append(line)

    # ------------------------------------------------------------
    # 2. Small neighbouring context windows.
    #
    # Do not combine the whole page. Only 2-3 nearby lines.
    # ------------------------------------------------------------
    for i in range(len(lines)):
        for span in (2, 3):
            if i + span > len(lines):
                continue

            group = lines[i:i + span]
            combined = normalize_text(" ".join(group))

            if (
                combined
                and len(combined) <= MAX_CHUNK_LENGTH
                and len(combined.split()) <= MAX_EVIDENCE_CONTEXT_WORDS
            ):
                candidates.append(combined)

    # ------------------------------------------------------------
    # 3. Evidence-centred context windows.
    #
    # Search the reconstructed OCR lines for any known dark-pattern
    # phrase. When found, retain the nearby visual context.
    # This fixes fragmented OCR such as:
    #
    #     5 seats
    #     available
    #
    # without sending the entire screenshot to the model.
    # ------------------------------------------------------------
    evidence_trigger_patterns = []

    for patterns in EVIDENCE.values():
        evidence_trigger_patterns.extend(patterns)

    for i, line in enumerate(lines):
        normalized_line = normalize_text(line)

        hit = False

        for pattern in evidence_trigger_patterns:
            try:
                if re.search(pattern, normalized_line, re.IGNORECASE):
                    hit = True
                    break
            except re.error:
                continue

        # Also detect common scarcity fragments even when OCR split
        # the phrase in an unusual way.
        if not hit:
            scarcity_terms = (
                "seat", "seats", "room", "rooms",
                "ticket", "tickets", "spot", "spots",
                "available", "remaining", "left",
                "hurry", "demand", "sold out"
            )

            if (
                any(term in normalized_line for term in scarcity_terms)
                and any(
                    term in normalized_line
                    for term in (
                        "available", "remaining", "left",
                        "only", "hurry", "demand",
                        "sold out"
                    )
                )
            ):
                hit = True

        if not hit:
            continue

        lo = max(0, i - EVIDENCE_CONTEXT_LINES)
        hi = min(len(lines), i + EVIDENCE_CONTEXT_LINES + 1)

        context_lines = lines[lo:hi]
        context = normalize_text(" ".join(context_lines))

        if (
            context
            and len(context) <= MAX_CHUNK_LENGTH
            and len(context.split()) <= MAX_EVIDENCE_CONTEXT_WORDS
        ):
            candidates.append(context)

        # Also keep a tighter local window around the hit.
        lo = max(0, i - 1)
        hi = min(len(lines), i + 2)

        local_context = normalize_text(
            " ".join(lines[lo:hi])
        )

        if (
            local_context
            and len(local_context) <= MAX_CHUNK_LENGTH
        ):
            candidates.append(local_context)

    # ------------------------------------------------------------
    # 4. Sentence reconstruction.
    # ------------------------------------------------------------
    sentence_candidates = []

    for chunk in candidates:
        parts = re.split(
            r"(?<=[.!?])\s+",
            chunk
        )

        if len(parts) <= 1:
            sentence_candidates.append(chunk)
            continue

        for part in parts:
            part = normalize_text(part)

            if (
                len(part) >= MIN_TEXT_LENGTH
                and not is_garbage(part)
            ):
                sentence_candidates.append(part)

    candidates = sentence_candidates

    # ------------------------------------------------------------
    # 5. Normalize and remove garbage.
    # ------------------------------------------------------------
    candidates = [
        normalize_text(c)
        for c in candidates
        if c
    ]

    candidates = [
        c
        for c in candidates
        if not is_garbage(c)
    ]

    # ------------------------------------------------------------
    # 6. Do not discard a long chunk merely because it contains
    # normal UI text. Only obvious short standalone labels go.
    # ------------------------------------------------------------
    filtered = []

    for chunk in candidates:
        if is_normal_ui_text(chunk):
            if len(chunk.split()) <= 5:
                continue

        filtered.append(chunk)

    candidates = filtered

    # ------------------------------------------------------------
    # 7. Exact duplicate removal first.
    # ------------------------------------------------------------
    exact_unique = []
    seen = set()

    for candidate in candidates:
        key = candidate.lower()

        if key in seen:
            continue

        seen.add(key)
        exact_unique.append(candidate)

    candidates = exact_unique

    # ------------------------------------------------------------
    # 8. Remove highly overlapping OCR fragments.
    #
    # Prefer the longer / more contextual chunk.
    # ------------------------------------------------------------
    candidates.sort(
        key=lambda x: (
            len(x.split()),
            len(x)
        ),
        reverse=True
    )

    unique_chunks = []

    for candidate in candidates:
        duplicate = False
        candidate_words = set(candidate.split())

        if not candidate_words:
            continue

        for existing in unique_chunks:
            existing_words = set(existing.split())

            intersection = len(
                candidate_words & existing_words
            )

            union = len(
                candidate_words | existing_words
            )

            jaccard = (
                intersection / max(union, 1)
            )

            similarity = SequenceMatcher(
                None,
                candidate,
                existing
            ).ratio()

            if (
                similarity >= CHUNK_DUPLICATE_THRESHOLD
                or jaccard >= 0.90
            ):
                duplicate = True
                break

        if not duplicate:
            unique_chunks.append(candidate)

    # ------------------------------------------------------------
    # 9. Return in compact model-analysis order.
    # Longer contextual chunks first, but still only local chunks.
    # ------------------------------------------------------------
    unique_chunks.sort(
        key=lambda x: (
            len(x.split()),
            len(x)
        ),
        reverse=True
    )

    return unique_chunks


# ============================================================
# EXACT EVIDENCE
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

        # Scarcity / availability statements.
        # Detect both "only 5 seats available" and "5 seats available".
        r"\bonly\s+\d+\s+(?:room|rooms|seat|seats|ticket|tickets|spot|spots)\s+(?:are\s+)?(?:available|left|remaining)\b",

        r"\b\d+\s+(?:room|rooms|seat|seats|ticket|tickets|spot|spots)\s+(?:are\s+)?(?:available|left|remaining)\b",

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

        r"\blast\s+few\b",

        r"\bonly\s+a\s+few\b",

        r"\bfew\s+(?:rooms?|seats?|tickets?|spots?)\s+(?:remain|remaining|left|available)\b",

        r"\bavailability\s+(?:is\s+)?(?:low|limited)\b"
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

        r"\badditional\s+charges?\s+apply\b",

        r"\bextra\s+fees?\s+apply\b",

        r"\bcharges?\s+apply\s+at\s+checkout\b",

        r"\badditional\s+fees?\s+at\s+checkout\b"
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

        r"\bspecial\s+pick\b",

        r"\brecommended\s+choice\b",

        r"\bpreferred\s+choice\b",

        r"\bpreferred\s+option\b"
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

        r"\bselected\s+by\s+default\b",

        r"\bautomatically\s+included\b",

        r"\badded\s+automatically\b",

        r"\boption\s+is\s+preselected\b"
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

        r"\bdecline\s+protection\b",

        r"\bchoose\s+the\s+premium\b",

        r"\bselect\s+the\s+premium\b"
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

        r"\breturn\s+option\s+is\s+unavailable\b",

        r"\bcannot\s+proceed\b",

        r"\bunable\s+to\s+continue\b"
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
# ============================================================

FUZZY_PHRASES = {

    "false_urgency": [

        "high demand",

        "limited availability",

        "limited time",

        "hurry",

        "book before",

        "prices go up",

        "availability is running low",

        "almost sold out",

        "few rooms remaining",

        "few seats remaining",

        "seats available",

        "rooms available",

        "tickets available",

        "spots available"
    ],


    "hidden_fees": [

        "service fee",

        "booking fee",

        "cleaning fee",

        "mandatory fee",

        "additional fee",

        "extra charges",

        "fees calculated later",

        "additional charges at checkout"
    ],


    "sneaking": [

        "selected automatically",

        "automatically selected",

        "already selected",

        "automatically added",

        "already added",

        "protection added",

        "included by default",

        "pre selected"
    ],


    "misdirection": [

        "recommended for you",

        "top pick for you",

        "popular choice",

        "guest favourite",

        "guest favorite",

        "most booked option",

        "preferred choice",

        "preferred option"
    ],


    "interface_interference": [

        "continue without",

        "no thanks",

        "add protection",

        "continue with protection",

        "choose premium",

        "upgrade to",

        "select premium"
    ],


    "obstruction": [

        "payment locked",

        "payment locked until",

        "locked until verification",

        "verification required",

        "cannot continue",

        "complete all steps",

        "cannot go back",

        "cannot be modified"
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
        or
        not phrase
    ):

        return 0.0


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
        -
        window_size
        +
        1
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


        if score >= FUZZY_THRESHOLD:

            matches.append(
                {
                    "phrase":
                        phrase,

                    "score":
                        score
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

            "found":
                False,

            "score":
                0.0,

            "matched_example":
                None
        }


    normalized = normalize_text(
        text
    )


    if len(normalized) < 5:

        return {

            "found":
                False,

            "score":
                0.0,

            "matched_example":
                None
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
                best_score
                >=
                SEMANTIC_THRESHOLD,

            "score":
                best_score,

            "matched_example":
                best_example
        }


    except Exception:

        return {

            "found":
                False,

            "score":
                0.0,

            "matched_example":
                None
        }


# ============================================================
# EVIDENCE SCORE
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


        # ----------------------------------------------------
        # Prevent duplicate evidence inflation.
        #
        # If 3 regexes detect the same phrase,
        # they should not produce unlimited score.
        # ----------------------------------------------------

        unique_exact = []


        for item in exact:

            if item not in unique_exact:

                unique_exact.append(
                    item
                )


        unique_fuzzy = []


        fuzzy_seen = set()


        for item in fuzzy:

            key = item["phrase"]


            if key in fuzzy_seen:

                continue


            fuzzy_seen.add(
                key
            )


            unique_fuzzy.append(
                item
            )


        exact_score = min(
            len(unique_exact),
            2
        ) * 2.0


        fuzzy_score = min(
            len(unique_fuzzy),
            2
        ) * 1.5


        semantic_score = 1.0 if (
            semantic["found"]
        ) else 0.0


        evidence_score = (
            exact_score
            +
            fuzzy_score
            +
            semantic_score
        )


        if (
            unique_exact
            or
            unique_fuzzy
            or
            semantic["found"]
        ):

            detected[label] = {

                "exact":
                    unique_exact,

                "fuzzy":
                    unique_fuzzy,

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
        key:
            value.to(device)
        for key, value
        in inputs.items()
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
# CONTEXTUAL EVIDENCE BOOST
#
# This handles OCR where one exact phrase may be
# fragmented but the complete chunk has enough context.
# ============================================================

def contextual_signal(
    text,
    label
):

    normalized = normalize_text(
        text
    )


    # --------------------------------------------------------
    # FALSE URGENCY
    # --------------------------------------------------------

    if label == "false_urgency":

        urgency_words = [

            "hurry",
            "limited",
            "high demand",
            "fully booked",
            "only",
            "left",
            "remaining",
            "expires",
            "book now",
            "reserve now",
            "prices go up",
            "price increase",
            "availability",
            "available",
            "seats",
            "rooms",
            "tickets",
            "spots",
            "sold out"
        ]


        count = sum(
            1
            for word in urgency_words
            if word in normalized
        )


        if count >= 3:

            return 2.0


        if count >= 2:

            return 1.0


    # --------------------------------------------------------
    # HIDDEN FEES
    # --------------------------------------------------------

    if label == "hidden_fees":

        fee_words = [

            "fee",
            "fees",
            "charge",
            "charges",
            "mandatory",
            "additional",
            "extra",
            "checkout",
            "added later",
            "calculated later"
        ]


        count = sum(
            1
            for word in fee_words
            if word in normalized
        )


        if count >= 3:

            return 2.0


        if count >= 2:

            return 1.0


    # --------------------------------------------------------
    # SNEAKING
    # --------------------------------------------------------

    if label == "sneaking":

        sneak_words = [

            "selected",
            "automatically",
            "added",
            "already",
            "default",
            "included",
            "preselected",
            "pre-selected"
        ]


        count = sum(
            1
            for word in sneak_words
            if word in normalized
        )


        if count >= 3:

            return 2.0


        if count >= 2:

            return 1.0


    # --------------------------------------------------------
    # MISDIRECTION
    # --------------------------------------------------------

    if label == "misdirection":

        misdirection_words = [

            "recommended",
            "recommended for you",
            "preferred",
            "popular choice",
            "top pick",
            "smart choice",
            "most popular",
            "most booked",
            "suggested"
        ]


        count = sum(
            1
            for word in misdirection_words
            if word in normalized
        )


        if count >= 2:

            return 2.0


        if count >= 1:

            return 1.0


    # --------------------------------------------------------
    # INTERFACE INTERFERENCE
    # --------------------------------------------------------

    if label == "interface_interference":

        interface_words = [

            "continue without",
            "no thanks",
            "premium",
            "upgrade",
            "protection",
            "insurance",
            "add protection",
            "choose premium"
        ]


        count = sum(
            1
            for word in interface_words
            if word in normalized
        )


        if count >= 2:

            return 2.0


        if count >= 1:

            return 1.0


    # --------------------------------------------------------
    # OBSTRUCTION
    # --------------------------------------------------------

    if label == "obstruction":

        obstruction_words = [

            "cannot",
            "can't",
            "locked",
            "required",
            "verification",
            "must complete",
            "not allowed",
            "unavailable",
            "cannot continue",
            "cannot go back"
        ]


        count = sum(
            1
            for word in obstruction_words
            if word in normalized
        )


        if count >= 2:

            return 2.0


        if count >= 1:

            return 1.0


    return 0.0


def hard_evidence_override(
    pattern_label,
    evidence,
    text
):
    """
    Certain booking-platform phrases are explicit dark-pattern
    evidence and should not be rejected only because DistilBERT
    classified the same chunk as another label.

    Example:
        "5 seats available"
        "only 2 rooms left"
        "rooms in high demand"
        "prices may go up"
    """

    if pattern_label != "false_urgency":
        return False

    normalized = normalize_text(text)

    # Strong scarcity statements.
    hard_patterns = [
        r"\bonly\s+\d+\s+(?:room|rooms|seat|seats|ticket|tickets|spot|spots)\s+(?:are\s+)?(?:available|left|remaining)\b",
        r"\b\d+\s+(?:room|rooms|seat|seats|ticket|tickets|spot|spots)\s+(?:are\s+)?(?:available|left|remaining)\b",
        r"\bonly\s+\d+\s+(?:room|rooms|seat|seats|ticket|tickets|spot|spots)\b",
        r"\b(?:last|only)\s+few\s+(?:rooms?|seats?|tickets?|spots?)\b",
        r"\bfew\s+(?:rooms?|seats?|tickets?|spots?)\s+(?:remain|remaining|left|available)\b",
        r"\b(?:hurry|hurry+)\b",
        r"\bhigh\s+demand\b",
        r"\balmost\s+sold\s+out\b",
        r"\bprices?\s+(?:may\s+)?(?:go\s+up|increase)\b",
        r"\bavailability\s+(?:is\s+)?(?:running\s+low|low|limited)\b",
    ]

    for pattern in hard_patterns:
        if re.search(pattern, normalized, re.IGNORECASE):
            return True

    # If the evidence engine already found an exact false-urgency
    # phrase, preserve it even when the classifier disagrees.
    if evidence.get("exact"):
        return True

    return False


# ============================================================
# FINAL DECISION
# ============================================================

def should_accept_pattern(
    pattern_label,
    predicted_label,
    model_confidence,
    evidence,
    text
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


    context_score = contextual_signal(
        text,
        pattern_label
    )


    total_evidence = (
        evidence_score
        +
        context_score
    )


    # ========================================================
    # NEVER ACCEPT OBVIOUS NORMAL UI
    # ========================================================

    if is_normal_ui_text(text):

        return False


    # ========================================================
    # HARD BOOKING-SCARCITY EVIDENCE
    #
    # Explicit phrases such as "5 seats available" are stronger
    # than a conflicting model label. This is still NOT based on
    # model prediction alone: an evidence rule must match.
    # ========================================================

    if hard_evidence_override(
        pattern_label,
        evidence,
        text
    ):

        return True


    # ========================================================
    # CASE 1
    #
    # MODEL + EVIDENCE AGREE
    # ========================================================

    if pattern_label == predicted_label:

        # Exact linguistic evidence.
        if (
            exact_count >= 1
            and
            model_confidence
            >= MODEL_SUPPORT_THRESHOLD
        ):

            return True


        # Strong contextual evidence.
        if (
            context_score >= 1.0
            and
            model_confidence
            >= MODEL_SUPPORT_THRESHOLD
            and
            len(text.split()) >= 5
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
            total_evidence
            >= STRONG_EVIDENCE_SCORE
            and
            model_confidence
            >= MODEL_SUPPORT_THRESHOLD
        ):

            return True


    # ========================================================
    # CASE 2
    #
    # MODEL SAYS NORMAL
    # BUT STRONG EVIDENCE EXISTS
    # ========================================================

    if predicted_label == "normal":

        # Two exact signals.
        if exact_count >= 2:

            return True


        # Strong contextual + exact/fuzzy support.
        if (
            context_score >= 2.0
            and
            (
                exact_count
                +
                fuzzy_count
            ) >= 1
        ):

            return True


        # Strong evidence.
        if (
            total_evidence
            >=
            STRONG_EVIDENCE_SCORE
        ):

            return True


        # Very strong semantic evidence +
        # another linguistic signal.
        if (
            semantic_found
            and
            semantic_score
            >= STRONG_SEMANTIC_THRESHOLD
            and
            (
                exact_count
                +
                fuzzy_count
            ) >= 1
        ):

            return True


    # ========================================================
    # CASE 3
    #
    # MODEL PREDICTS ANOTHER DARK PATTERN
    # BUT EVIDENCE SUPPORTS THIS PATTERN.
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
            total_evidence
            >= STRONG_EVIDENCE_SCORE
        ):

            return True


        # Strong contextual evidence.
        if (
            context_score >= 2.0
            and
            exact_count >= 1
            and
            model_confidence >= MODEL_SUPPORT_THRESHOLD
        ):

            return True


        # Very strong semantic evidence.
        if (
            semantic_found
            and
            semantic_score
            >= STRONG_SEMANTIC_THRESHOLD
            and
            total_evidence
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
        "STEP 2 - POSITIONAL OCR + TEXT RECONSTRUCTION"
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
        "Final text chunks:",
        len(chunks)
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


        # ====================================================
        # MODEL-ONLY PREDICTION
        #
        # Do NOT print as evidence.
        # ====================================================

        if not all_evidence:

            print(
                "     Evidence: NONE"
            )

            continue


        # ====================================================
        # EVIDENCE
        # ====================================================

        for (
            evidence_label,
            evidence
        ) in all_evidence.items():

            print(
                f"\n     Pattern Evidence: "
                f"{evidence_label}"
            )


            context_score = contextual_signal(
                chunk,
                evidence_label
            )


            total_score = (
                evidence["evidence_score"]
                +
                context_score
            )


            print(
                "     Evidence Score:",
                f"{total_score:.2f}"
            )


            # ------------------------------------------------
            # EXACT
            # ------------------------------------------------

            if evidence["exact"]:

                print(
                    "     Exact Evidence:"
                )


                for match in evidence["exact"]:

                    print(
                        "        ",
                        match
                    )


            # ------------------------------------------------
            # FUZZY
            # ------------------------------------------------

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


            # ------------------------------------------------
            # SEMANTIC
            # ------------------------------------------------

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


            # ------------------------------------------------
            # CONTEXT
            # ------------------------------------------------

            if context_score > 0:

                print(
                    "     Contextual Signal:",
                    f"{context_score:.2f}"
                )


            # ------------------------------------------------
            # FINAL DECISION
            # ------------------------------------------------

            accepted_pattern = (
                should_accept_pattern(
                    evidence_label,
                    predicted_label,
                    confidence,
                    evidence,
                    chunk
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
                            total_score,

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


        # ----------------------------------------------------
        # DO NOT endlessly accumulate repeated evidence.
        # ----------------------------------------------------

        grouped[label]["signals"] += min(
            item["evidence_score"],
            5.0
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


        # ----------------------------------------------------
        # Exact matches
        # ----------------------------------------------------

        for match in evidence["exact"]:

            if match not in grouped[
                label
            ]["exact_matches"]:

                grouped[
                    label
                ]["exact_matches"].append(
                    match
                )


        # ----------------------------------------------------
        # Fuzzy matches
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Semantic matches
        # ----------------------------------------------------

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
    # NO PATTERN
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
    # RANK
    # ========================================================

    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            item[1]["signals"],
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