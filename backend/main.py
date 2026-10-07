from pathlib import Path
import sys
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import AnalyzeResponse
from recommendations import get_recommendation


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if not SCRIPTS_DIR.exists():
    raise RuntimeError(
        f"Scripts directory not found:\n{SCRIPTS_DIR}"
    )


# ============================================================
# IMPORT EXISTING FUSION
# ============================================================

sys.path.insert(
    0,
    str(SCRIPTS_DIR)
)

try:
    from fusion_predict import fusion_predict
except Exception as exc:
    raise RuntimeError(
        "Could not import fusion_predict.py\n"
        f"Original error:\n{exc}"
    ) from exc


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="DARPAN API",
    description="Backend API for DARPAN multimodal dark-pattern detection.",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "success": True,
        "message": "DARPAN backend is running."
    }


@app.get("/health")
def health():
    return {
        "success": True,
        "status": "healthy"
    }


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@app.post(
    "/api/analyze",
    response_model=AnalyzeResponse
)
async def analyze_page(
    image: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate uploaded file
    # --------------------------------------------------------

    if not image.filename:
        raise HTTPException(
            status_code=400,
            detail="No image filename provided."
        )

    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp"
    }

    suffix = Path(
        image.filename
    ).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use PNG, JPG, JPEG, WEBP, or BMP."
            )
        )


    # --------------------------------------------------------
    # Save uploaded image temporarily
    # --------------------------------------------------------

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="darpan_"
        )
    )

    temp_image = temp_dir / (
        f"input{suffix}"
    )


    try:

        with open(
            temp_image,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                image.file,
                buffer
            )


        # ----------------------------------------------------
        # Run EXISTING fusion pipeline
        # ----------------------------------------------------

        result = fusion_predict(
            str(temp_image)
        )


        if not isinstance(
            result,
            dict
        ):
            raise RuntimeError(
                "fusion_predict() returned an invalid result."
            )


        # ----------------------------------------------------
        # Convert fusion output for frontend
        # ----------------------------------------------------

        final_patterns = []

        for pattern in result.get(
            "patterns",
            []
        ):

            if not isinstance(
                pattern,
                dict
            ):
                continue


            label = pattern.get(
                "label"
            )

            if not label:
                continue


            confidence = float(
                pattern.get(
                    "confidence",
                    0.0
                ) or 0.0
            )


            confidence_percent = float(
                pattern.get(
                    "confidence_percent",
                    confidence * 100
                ) or 0.0
            )


            visual = pattern.get(
                "visual"
            ) or {}

            text = pattern.get(
                "text"
            ) or {}


            bbox = None

            if isinstance(
                visual,
                dict
            ):

                raw_bbox = visual.get(
                    "bounding_box"
                )

                if isinstance(
                    raw_bbox,
                    dict
                ):

                    try:

                        bbox = {
                            "x1": float(
                                raw_bbox["x1"]
                            ),
                            "y1": float(
                                raw_bbox["y1"]
                            ),
                            "x2": float(
                                raw_bbox["x2"]
                            ),
                            "y2": float(
                                raw_bbox["y2"]
                            )
                        }

                    except (
                        KeyError,
                        TypeError,
                        ValueError
                    ):

                        bbox = None


            supporting_text = ""

            if isinstance(
                text,
                dict
            ):

                supporting_text = (
                    text.get(
                        "supporting_text",
                        ""
                    )
                    or ""
                )


            final_patterns.append({

                "label": label,

                "confidence": confidence,

                "confidence_percent":
                    confidence_percent,

                "decision":
                    pattern.get(
                        "decision",
                        "confirmed"
                    ),

                "reason":
                    pattern.get(
                        "reason",
                        ""
                    ),

                "supporting_text":
                    supporting_text,

                "bounding_box":
                    bbox,

                "why_it_matters":
                    get_why_it_matters(
                        label
                    ),

                "ethical_recommendation":
                    get_recommendation(
                        label
                    )
            })


        primary_pattern = result.get(
            "primary_pattern"
        )

        primary_confidence = float(
            result.get(
                "primary_confidence",
                0.0
            ) or 0.0
        )


        return AnalyzeResponse(

            success=True,

            dark_pattern_detected=bool(
                result.get(
                    "dark_pattern_detected",
                    False
                )
            ),

            total_patterns_detected=len(
                final_patterns
            ),

            primary_pattern=
                primary_pattern,

            primary_confidence=
                primary_confidence,

            patterns=
                final_patterns,

            message=(
                "Analysis completed successfully."
            )
        )


   
    except Exception as exc:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                f"DARPAN analysis failed: "
                f"{type(exc).__name__}: {exc}"
            )
        ) from exc

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

# ============================================================
# UX EXPLANATIONS
# ============================================================

def get_why_it_matters(
    pattern: str
) -> str:

    explanations = {

        "hidden_fees": (
            "Additional costs may not be sufficiently visible "
            "when the user makes an important decision."
        ),

        "false_urgency": (
            "Scarcity or time-pressure cues may encourage users "
            "to make a rushed decision."
        ),

        "sneaking": (
            "An optional service or choice may be introduced "
            "without sufficiently clear user intent."
        ),

        "misdirection": (
            "The wording or presentation may steer users toward "
            "one choice instead of presenting meaningful alternatives neutrally."
        ),

        "interface_interference": (
            "The visual hierarchy or interaction structure may "
            "make one choice disproportionately prominent."
        ),

        "obstruction": (
            "A user-controlled action may involve unnecessary "
            "friction or difficulty."
        ),
    }

    return explanations.get(
        pattern,
        "The interface may contain a potentially misleading design choice."
    )