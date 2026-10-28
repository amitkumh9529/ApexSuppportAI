"""Tests for outbound webhook signing and event recording."""

import pytest
from app.db import session as db
from app.db.models import WebhookEvent
from app.services.webhook import dispatch_webhook_event, generate_signature
from sqlalchemy import select


def test_hmac_signature_generation():
    secret = "my-secret-key-123"
    payload = b'{"event":"refund.approved"}'
    sig = generate_signature(secret, payload)
    assert isinstance(sig, str)
    assert len(sig) == 64  # SHA256 hex digest length
    sig2 = generate_signature(secret, b'{"event":"refund.denied"}')
    assert sig != sig2


@pytest.mark.asyncio
async def test_webhook_event_logged(engine):
    await dispatch_webhook_event(
        event_type="refund.approved",
        payload={"order_id": "ORD-1234", "amount": 99.50},
    )
    async with db.SessionLocal() as session:
        events = (await session.scalars(select(WebhookEvent))).all()
        assert len(events) >= 1
        event = events[-1]
        assert event.event_type == "refund.approved"
        assert event.payload["order_id"] == "ORD-1234"
