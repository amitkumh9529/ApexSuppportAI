import hmac
import json
import logging
import random
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sse_starlette.sse import EventSourceResponse

from app import redis as shared
from app import tracing
from app.agent.llm import primary_model_name
from app.agent.runner import (
    ConversationNotFound,
    create_conversation,
    load_owned_conversation,
    new_conversation_id,
    run_agent_turn,
)
from app.auth import (
    ROLE_ADMIN,
    ROLE_CUSTOMER,
    Principal,
    bearer_token,
    mint_token,
    principal_from_request,
    require_admin,
    require_customer,
)
from app.config import get_settings
from app.db import session as db
from app.db.models import (
    Conversation,
    Customer,
    Message,
    Order,
    ReasoningEvent,
    Refund,
    SentimentLog,
    WebhookEvent,
)
from app.db.seed import seed_if_empty
from app.demo import order_out
from app.demo import router as demo_router
from app.events import broadcaster
from app.observability import (
    REJECTIONS,
    SSE_STREAMS,
    RequestContextMiddleware,
    configure_logging,
    configure_sentry,
    customer_id_var,
    log_event,
)
from app.schemas import (
    AdminStats,
    ChatRequest,
    ConversationDetail,
    ConversationSummary,
    MessageOut,
    OrderOut,
    ReasoningEventOut,
)
from app.services.webhook import dispatch_webhook_event

settings = get_settings()
logger = logging.getLogger("app")

STREAM_ERROR_MESSAGE = (
    "Sorry — something went wrong on our side while handling that. "
    "Please try again in a moment."
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    configure_sentry()
    tracing.configure()
    if settings.seed_on_startup:
        await seed_if_empty(create_tables=True)
    log_event(logger, "startup", app=settings.app_name, env=settings.env, llm_provider=settings.llm_provider)
    yield
    tracing.flush()
    await shared.close_redis()
    await db.dispose()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise Autonomous Customer Support Backend with Deterministic Guardrails & Policy Engine",
    lifespan=lifespan,
)

if settings.cors_origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["Retry-After", "X-Request-ID"],
        max_age=3600,
    )
app.add_middleware(RequestContextMiddleware, max_body_bytes=settings.max_request_bytes)
app.include_router(demo_router)


@app.exception_handler(Exception)
async def _unhandled(_request: Request, exc: Exception):
    log_event(
        logger,
        "unhandled error",
        level=logging.ERROR,
        error_type=type(exc).__name__,
        error=str(exc)[:500],
    )
    return JSONResponse(status_code=500, content={"detail": "internal error"})


# ---------------------------------------------------------------------------
# Health & Diagnostics
# ---------------------------------------------------------------------------


@app.get("/api/health", tags=["Health"])
@app.get("/api/health/live", tags=["Health"])
async def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "llm_provider": settings.llm_provider,
        "llm_key_configured": settings.has_llm_key,
    }


@app.get("/api/health/ready", tags=["Health"])
async def ready() -> Response:
    checks = {}
    try:
        async with db.SessionLocal() as session:
            await session.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as exc:
        checks["db"] = f"error: {type(exc).__name__}"
    try:
        await shared.get_redis().ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {type(exc).__name__}"
    ok = all(v == "ok" for v in checks.values())
    return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, **checks})


@app.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    if settings.metrics_token:
        given = bearer_token(request) or ""
        if not hmac.compare_digest(given.encode(), settings.metrics_token.encode()):
            raise HTTPException(status_code=401, detail="metrics token required")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/meta", tags=["Metadata"])
async def meta() -> dict:
    scripted = settings.llm_provider == "fake"
    return {
        "app_name": settings.app_name,
        "auth_mode": settings.auth_mode,
        "model": "scripted" if scripted else settings.llm_provider,
        "model_name": None if scripted else primary_model_name(),
        "demo": {
            "session_ttl_s": settings.demo_token_ttl_s,
            "data_ttl_hours": settings.demo_data_ttl_hours,
            "transcript_retention_days": settings.retention_days,
        }
        if settings.is_demo
        else None,
    }


# ---------------------------------------------------------------------------
# Dev Auth Endpoints (AUTH_MODE=dev)
# ---------------------------------------------------------------------------


class DevTokenRequest(BaseModel):
    customer_id: str | None = None
    admin: bool = False


@app.post("/api/dev/token", tags=["Dev Auth"])
async def dev_token(req: DevTokenRequest) -> dict:
    if settings.auth_mode != "dev" or settings.env == "prod":
        raise HTTPException(status_code=404, detail="Not Found")
    if req.admin:
        return {"token": mint_token("dev-admin", ROLE_ADMIN), "role": ROLE_ADMIN}
    if not req.customer_id:
        raise HTTPException(status_code=400, detail="customer_id is required")
    async with db.SessionLocal() as session:
        if await session.get(Customer, req.customer_id) is None:
            raise HTTPException(status_code=404, detail="unknown customer")
    return {"token": mint_token(req.customer_id, ROLE_CUSTOMER), "role": ROLE_CUSTOMER}


@app.get("/api/dev/customers", tags=["Dev Auth"])
async def dev_customers() -> list[dict]:
    if settings.auth_mode != "dev" or settings.env == "prod":
        raise HTTPException(status_code=404, detail="Not Found")
    async with db.SessionLocal() as session:
        rows = (await session.scalars(select(Customer).order_by(Customer.id).limit(20))).all()
    return [{"id": c.id, "name": c.name, "email": c.email} for c in rows]


# ---------------------------------------------------------------------------
# Customer Chat
# ---------------------------------------------------------------------------


def _rate_limit_key(request: Request) -> str:
    principal = principal_from_request(request)
    if principal is not None:
        return f"sub:{principal.sub}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


async def enforce_chat_rate_limit(request: Request) -> None:
    if principal_from_request(request) is None:
        return
    allowed, reset_in = await shared.hit_rate_limit(
        f"chat:{_rate_limit_key(request)}", settings.chat_rate_limit
    )
    if not allowed:
        REJECTIONS.labels("rate_limited").inc()
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {settings.chat_rate_limit}",
            headers={"Retry-After": str(reset_in)},
        )


@app.post("/api/chat", dependencies=[Depends(enforce_chat_rate_limit)], tags=["Chat"])
async def chat(req: ChatRequest, principal: Principal = Depends(require_customer)):
    customer_id_var.set(principal.sub)
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    if len(req.message) > settings.max_message_chars:
        REJECTIONS.labels("too_long").inc()
        raise HTTPException(
            status_code=400,
            detail=(
                f"message is too long ({len(req.message)} chars); the limit is "
                f"{settings.max_message_chars}."
            ),
        )

    if req.conversation_id:
        try:
            await load_owned_conversation(principal.sub, req.conversation_id)
        except ConversationNotFound:
            raise HTTPException(status_code=404, detail="conversation not found") from None

        async with db.SessionLocal() as session:
            existing = (
                await session.scalar(
                    select(func.count(Message.id)).where(
                        Message.conversation_id == req.conversation_id
                    )
                )
                or 0
            )
        if existing >= settings.max_conversation_turns:
            REJECTIONS.labels("turn_cap").inc()
            raise HTTPException(
                status_code=429,
                detail=(
                    f"this conversation has reached its message limit "
                    f"({settings.max_conversation_turns}). Please start a new chat."
                ),
            )
        cid = req.conversation_id
        is_new = False
    else:
        cid = new_conversation_id()
        is_new = True

    lock_ttl = settings.turn_timeout_s + 30
    lock = await shared.acquire_turn_lock(cid, lock_ttl)
    if lock is None:
        REJECTIONS.labels("turn_in_progress").inc()
        raise HTTPException(
            status_code=409, detail="a reply is still being generated for this conversation"
        )

    slot = await shared.acquire_turn_slot(settings.max_concurrent_turns, lock_ttl)
    if slot is None:
        await shared.release_turn_lock(cid, lock)
        REJECTIONS.labels("overloaded").inc()
        raise HTTPException(
            status_code=503,
            detail="High server demand. Please try again shortly.",
            headers={"Retry-After": str(random.randint(5, 15))},
        )

    if is_new:
        await create_conversation(principal.sub, cid)

    async def event_stream():
        SSE_STREAMS.labels("chat").inc()
        try:
            async for event in run_agent_turn(principal.sub, cid, req.message):
                yield {"data": json.dumps(event)}
        except Exception as exc:
            log_event(
                logger,
                "chat stream failed",
                level=logging.ERROR,
                error_type=type(exc).__name__,
                error=str(exc)[:500],
            )
            yield {"data": json.dumps({"kind": "error", "message": STREAM_ERROR_MESSAGE})}
        finally:
            SSE_STREAMS.labels("chat").dec()
            await shared.release_turn_slot(slot)
            await shared.release_turn_lock(cid, lock)

    return EventSourceResponse(event_stream(), ping=15)


def _messages_out(messages) -> list[MessageOut]:
    return [
        MessageOut(id=m.id, role=m.role, content=m.content, created_at=m.created_at)
        for m in messages
    ]


@app.get("/api/me/conversations", response_model=list[ConversationSummary], tags=["Customer"])
async def my_conversations(
    principal: Principal = Depends(require_customer),
    limit: int = Query(20, ge=1, le=100),
    before: datetime | None = None,
) -> list[ConversationSummary]:
    return await _conversation_summaries(limit, before, customer_id=principal.sub)


@app.get("/api/me/conversations/{conversation_id}", response_model=ConversationDetail, tags=["Customer"])
async def my_conversation(
    conversation_id: str, principal: Principal = Depends(require_customer)
) -> ConversationDetail:
    try:
        convo = await load_owned_conversation(principal.sub, conversation_id)
    except ConversationNotFound:
        raise HTTPException(status_code=404, detail="conversation not found") from None
    async with db.SessionLocal() as session:
        messages = (
            await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.id)
            )
        ).all()
    return ConversationDetail(
        id=convo.id,
        customer_id=convo.customer_id,
        customer_name=None,
        messages=_messages_out(messages),
        events=[],
    )


@app.get("/api/me/orders", response_model=list[OrderOut], tags=["Customer"])
async def my_orders(principal: Principal = Depends(require_customer)) -> list[OrderOut]:
    async with db.SessionLocal() as session:
        orders = (
            await session.scalars(
                select(Order)
                .where(Order.customer_id == principal.sub)
                .order_by(Order.id)
                .limit(100)
            )
        ).all()
    return [order_out(o) for o in orders]


# ---------------------------------------------------------------------------
# Admin Endpoints
# ---------------------------------------------------------------------------


async def _conversation_summaries(
    limit: int, before: datetime | None, customer_id: str | None = None
) -> list[ConversationSummary]:
    msg_count = (
        select(func.count(Message.id))
        .where(Message.conversation_id == Conversation.id)
        .correlate(Conversation)
        .scalar_subquery()
    )
    last_message = (
        select(Message.content)
        .where(Message.conversation_id == Conversation.id)
        .order_by(Message.id.desc())
        .limit(1)
        .correlate(Conversation)
        .scalar_subquery()
    )
    stmt = (
        select(Conversation, Customer.name, msg_count, last_message)
        .outerjoin(Customer, Customer.id == Conversation.customer_id)
        .order_by(Conversation.created_at.desc())
        .limit(limit)
    )
    if before is not None:
        stmt = stmt.where(Conversation.created_at < before)
    if customer_id is not None:
        stmt = stmt.where(Conversation.customer_id == customer_id)
    async with db.SessionLocal() as session:
        rows = (await session.execute(stmt)).all()
    return [
        ConversationSummary(
            id=c.id,
            customer_id=c.customer_id,
            customer_name=name,
            created_at=c.created_at,
            message_count=count or 0,
            last_message=last,
        )
        for c, name, count, last in rows
    ]


@app.get("/api/conversations", response_model=list[ConversationSummary], tags=["Admin"])
async def list_conversations(
    limit: int = Query(50, ge=1, le=200),
    before: datetime | None = Query(None, description="cursor: created_at of last row"),
    principal: Principal = Depends(require_admin),
) -> list[ConversationSummary]:
    return await _conversation_summaries(limit, before, customer_id=principal.scope)


def _visible(convo: Conversation | None, principal: Principal) -> bool:
    return convo is not None and (principal.scope is None or convo.customer_id == principal.scope)


@app.get("/api/admin/stats", response_model=AdminStats, tags=["Admin"])
async def admin_stats(principal: Principal = Depends(require_admin)) -> AdminStats:
    def scoped(stmt, conv_col):
        if principal.scope is None:
            return stmt
        owned = select(Conversation.id).where(Conversation.customer_id == principal.scope)
        return stmt.where(conv_col.in_(owned))

    async with db.SessionLocal() as session:
        conv_stmt = select(
            func.count(Conversation.id), func.coalesce(func.sum(Conversation.tokens_used), 0)
        )
        if principal.scope is not None:
            conv_stmt = conv_stmt.where(Conversation.customer_id == principal.scope)
        conversations, tokens = (await session.execute(conv_stmt)).one()
        messages = await session.scalar(
            scoped(select(func.count(Message.id)), Message.conversation_id)
        )
        decisions = (
            await session.execute(
                scoped(
                    select(Refund.decision, func.count(Refund.id)).group_by(Refund.decision),
                    Refund.conversation_id,
                )
            )
        ).all()
        flags = await session.scalar(
            scoped(
                select(func.count(ReasoningEvent.id)).where(
                    ReasoningEvent.step_type == "injection_flag"
                ),
                ReasoningEvent.conversation_id,
            )
        )
    return AdminStats(
        conversations=conversations or 0,
        messages=messages or 0,
        decisions={d: n for d, n in decisions},
        injection_flags=flags or 0,
        tokens_used=int(tokens or 0),
    )


@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail, tags=["Admin"])
async def get_conversation(
    conversation_id: str, principal: Principal = Depends(require_admin)
) -> ConversationDetail:
    async with db.SessionLocal() as session:
        convo = await session.get(Conversation, conversation_id)
        if not _visible(convo, principal):
            raise HTTPException(status_code=404, detail="conversation not found")
        name = None
        if convo.customer_id:
            cust = await session.get(Customer, convo.customer_id)
            name = cust.name if cust else None
        messages = (
            await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.id)
            )
        ).all()
        events = (
            await session.scalars(
                select(ReasoningEvent)
                .where(ReasoningEvent.conversation_id == conversation_id)
                .order_by(ReasoningEvent.seq)
            )
        ).all()
    return ConversationDetail(
        id=convo.id,
        customer_id=convo.customer_id,
        customer_name=name,
        messages=_messages_out(messages),
        events=[
            ReasoningEventOut(
                id=e.id,
                seq=e.seq,
                step_type=e.step_type,
                node=e.node,
                payload=e.payload,
                created_at=e.created_at,
            )
            for e in events
        ],
    )


@app.get("/api/conversations/{conversation_id}/stream", tags=["Admin"])
async def stream_conversation(
    conversation_id: str, principal: Principal = Depends(require_admin)
):
    if principal.scope is not None:
        async with db.SessionLocal() as session:
            convo = await session.get(Conversation, conversation_id)
        if not _visible(convo, principal):
            raise HTTPException(status_code=404, detail="conversation not found")

    async def event_stream():
        SSE_STREAMS.labels("admin").inc()
        try:
            async for event in broadcaster.subscribe(conversation_id):
                yield {"data": json.dumps(event)}
        finally:
            SSE_STREAMS.labels("admin").dec()

    return EventSourceResponse(event_stream(), ping=15)


# ---------------------------------------------------------------------------
# Resume Upgrade Endpoints: Sentiment Telemetry & Webhook Inspection
# ---------------------------------------------------------------------------


@app.get("/api/admin/sentiment/stats", tags=["Telemetry Upgrades"])
async def sentiment_stats(principal: Principal = Depends(require_admin)) -> dict:
    """Return customer sentiment and frustration distribution."""
    async with db.SessionLocal() as session:
        total_logs = await session.scalar(select(func.count(SentimentLog.id))) or 0
        frustrated_count = await session.scalar(
            select(func.count(SentimentLog.id)).where(SentimentLog.sentiment_label == "frustrated")
        ) or 0
        urgent_count = await session.scalar(
            select(func.count(SentimentLog.id)).where(SentimentLog.detected_urgency.is_(True))
        ) or 0
        avg_score = await session.scalar(select(func.avg(SentimentLog.sentiment_score))) or 0.0

    return {
        "total_analyzed_turns": total_logs,
        "frustrated_turns": frustrated_count,
        "urgent_turns": urgent_count,
        "frustration_rate": round(frustrated_count / max(total_logs, 1), 3),
        "average_sentiment_score": round(float(avg_score), 2),
    }


@app.get("/api/admin/webhooks", tags=["Telemetry Upgrades"])
async def list_webhooks(
    limit: int = Query(20, ge=1, le=100),
    principal: Principal = Depends(require_admin),
) -> list[dict]:
    """Inspect recent outbound webhook dispatch events and delivery statuses."""
    async with db.SessionLocal() as session:
        events = (
            await session.scalars(
                select(WebhookEvent).order_by(WebhookEvent.id.desc()).limit(limit)
            )
        ).all()
    return [
        {
            "id": e.id,
            "event_type": e.event_type,
            "status": e.status,
            "http_status": e.http_status,
            "error_message": e.error_message,
            "payload": e.payload,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in events
    ]


@app.post("/api/admin/webhooks/test", tags=["Telemetry Upgrades"])
async def test_webhook(principal: Principal = Depends(require_admin)) -> dict:
    """Dispatch a simulated test webhook event."""
    await dispatch_webhook_event(
        event_type="refund.test",
        payload={
            "test": True,
            "message": "ApexSupport AI Webhook Dispatcher Verified",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
    return {"status": "dispatched", "message": "Test webhook queued and logged"}
