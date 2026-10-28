"""Customer sentiment & frustration analysis service for ApexSupport-AI.

Runs real-time sentiment scoring on customer turns to detect distress,
frustration, or high urgency, allowing operators and the agent system to flag
conversations for priority human attention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FRUSTRATION_KEYWORDS = {
    "furious", "unacceptable", "scam", "lawyer", "attorney", "ridiculous",
    "terrible", "worst", "awful", "disaster", "horrible", "stolen", "cheat",
    "fraud", "sue", "complaint", "better business bureau", "bbb", "garbage",
    "hate", "useless", "incompetent", "chargeback", "dispute"
}

URGENCY_KEYWORDS = {
    "immediately", "urgent", "emergency", "asap", "right now", "critical",
    "today", "stranded", "vital"
}

POSITIVE_KEYWORDS = {
    "thanks", "thank you", "great", "helpful", "appreciate", "awesome",
    "excellent", "good", "perfect", "resolved"
}


@dataclass
class SentimentResult:
    score: float  # -1.0 (extremely negative) to +1.0 (positive)
    label: str    # "positive", "neutral", "frustrated"
    detected_urgency: bool
    frustration_signals: list[str]


def analyze_sentiment(text: str) -> SentimentResult:
    """Analyze the emotional tone and urgency of customer input."""
    if not text:
        return SentimentResult(score=0.0, label="neutral", detected_urgency=False, frustration_signals=[])

    lower_text = text.lower()
    words = set(re.findall(r"\b\w+\b", lower_text))

    frustrated_matches = [k for k in FRUSTRATION_KEYWORDS if k in lower_text or k in words]
    urgency_matches = [k for k in URGENCY_KEYWORDS if k in lower_text or k in words]
    positive_matches = [k for k in POSITIVE_KEYWORDS if k in lower_text or k in words]

    # Check for excessive caps or punctuation (e.g., "REFUND ME NOW!!!")
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = caps_count / max(len(text), 1)
    shouting = len(text) > 15 and caps_ratio > 0.45
    exclamations = text.count("!") >= 3

    base_score = 0.0
    base_score += len(positive_matches) * 0.35
    base_score -= len(frustrated_matches) * 0.4
    if shouting:
        base_score -= 0.3
        frustrated_matches.append("excessive_caps")
    if exclamations:
        base_score -= 0.2
        frustrated_matches.append("multiple_exclamations")

    score = max(-1.0, min(1.0, round(base_score, 2)))

    if score <= -0.35 or len(frustrated_matches) >= 2 or shouting:
        label = "frustrated"
    elif score >= 0.2:
        label = "positive"
    else:
        label = "neutral"

    detected_urgency = len(urgency_matches) > 0 or ("urgent" in lower_text)

    return SentimentResult(
        score=score,
        label=label,
        detected_urgency=detected_urgency,
        frustration_signals=frustrated_matches,
    )
