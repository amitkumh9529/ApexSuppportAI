"""Tests for the customer sentiment and frustration detection service."""

import pytest
from app.services.sentiment import analyze_sentiment


def test_positive_sentiment():
    res = analyze_sentiment("Thank you so much! That was extremely helpful and resolved my question.")
    assert res.label == "positive"
    assert res.score > 0.0
    assert not res.detected_urgency


def test_frustrated_sentiment_keywords():
    res = analyze_sentiment("This is ridiculous and horrible! You cheated me and I want a refund now!")
    assert res.label == "frustrated"
    assert res.score < 0.0
    assert "ridiculous" in res.frustration_signals or "horrible" in res.frustration_signals


def test_frustrated_shouting():
    res = analyze_sentiment("GIVE ME MY MONEY BACK IMMEDIATELY THIS IS UNACCEPTABLE")
    assert res.label == "frustrated"
    assert res.detected_urgency
    assert "excessive_caps" in res.frustration_signals


def test_neutral_sentiment():
    res = analyze_sentiment("Could you please look up order ORD-1002?")
    assert res.label == "neutral"
    assert -0.3 <= res.score <= 0.3
