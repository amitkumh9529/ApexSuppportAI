"""Asynchronous Outbound Webhook Dispatcher for ApexSupport-AI.

Emits cryptographically signed HTTP webhooks (HMAC-SHA256) when refund
decisions (approved, denied, escalated) are committed to the ledger, allowing
instant integration with downstream ERP, CRM, or Slack alerting channels.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any

import httpx

from app.config import get_settings
from app.db import session as db
from app.db.models import WebhookEvent

log = logging.getLogger(__name__)


def generate_signature(secret: str, payload_bytes: bytes) -> str:
    """Compute an HMAC-SHA256 hex digest for payload verification."""
    return hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()


async def dispatch_webhook_event(
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """Persist and asynchronously dispatch an outbound webhook if configured."""
    settings = get_settings()

    status = "disabled"
    http_status = None
    error_message = None

    if settings.webhook_url:
        status = "pending"
        payload_body = {
            "event": event_type,
            "timestamp": int(time.time()),
            "data": payload,
        }
        body_bytes = json.dumps(payload_body, sort_keys=True).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"{settings.app_name}-Webhook/1.0",
        }
        if settings.webhook_secret:
            sig = generate_signature(settings.webhook_secret, body_bytes)
            headers["X-Apex-Signature"] = sig

        try:
            async with httpx.AsyncClient(timeout=settings.webhook_timeout_s) as client:
                resp = await client.post(settings.webhook_url, content=body_bytes, headers=headers)
                http_status = resp.status_code
                if 200 <= resp.status_code < 300:
                    status = "sent"
                    log.info("Webhook %s delivered successfully to %s", event_type, settings.webhook_url)
                else:
                    status = "failed"
                    error_message = f"HTTP status {resp.status_code}: {resp.text[:200]}"
                    log.warning("Webhook delivery failed: %s", error_message)
        except Exception as e:
            status = "failed"
            error_message = str(e)
            log.warning("Exception during webhook delivery: %s", error_message)

    # Record event in the WebhookEvent audit table
    try:
        async with db.SessionLocal() as session:
            evt = WebhookEvent(
                event_type=event_type,
                payload=payload,
                status=status,
                http_status=http_status,
                error_message=error_message,
            )
            session.add(evt)
            await session.commit()
    except Exception as e:
        log.error("Failed to persist webhook event log: %s", e)
