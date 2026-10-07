from typing import Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class PatternResult(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_percent: float
    decision: str

    reason: str = ""

    supporting_text: str = ""

    bounding_box: BoundingBox | None = None

    why_it_matters: str = ""
    ethical_recommendation: str = ""


class AnalyzeResponse(BaseModel):
    success: bool

    dark_pattern_detected: bool
    total_patterns_detected: int

    primary_pattern: str | None = None
    primary_confidence: float = 0.0

    patterns: list[PatternResult]

    message: str = ""