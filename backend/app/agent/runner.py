"""Drives one agent turn and emits reasoning events for ApexSupport-AI.

`run_agent_turn` is an async generator: it executes the LangGraph agent over the
conversation, persisting each step (model thoughts, tool calls, tool results,
decisions, and real-time sentiment analysis) as a ReasoningEvent, publishing it
to Redis pub/sub (for the admin trace dashboard), and yielding it to the client SSE stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy import insert, select, update

from app import redis as shared
from app import tracing
from app.agent.graph import get_agent
from app.agent.guard import detect_injection
from app.agent.llm import primary_model_name, use_prompt_caching
from app.agent.prompts import get_system_prompt
from app.agent.tools import UNTRUSTED_PLACEHOLDER, ToolContext, tool_config
from app.config import get_settings
from app.db import session as db
from app.db.models import Conversation, Message, ReasoningEvent, SentimentLog, utcnow
from app.events import broadcaster
from app.observability import (
    INJECTION_FLAGS,
    LLM_FALLBACKS,
    LLM_TOKENS,
    REFUND_DECISIONS,
    TOOL_CALLS,
    TURN_LATENCY,
    TURNS,
    TURNS_IN_FLIGHT,
    conversation_id_var,
    log_event,
)
from app.services.sentiment import analyze_sentiment

logger = logging.getLogger(__name__)

_HISTORY_LIMIT = 40

_ERROR_MESSAGE = (
    "Sorry — something went wrong on our side while handling that. "
    "Please try again in a moment."
)

_DECISION_TOOLS = {"issue_refund", "escalate_to_human"}

_APPROVAL_CLAIM = re.compile(
    r"(?:"
    r"(?:your |the )?refund(?:\s+of\s+\$[\d,.]+)?\s+(?:has been|is|was)\s+"
    r"(?:approved|processed|issued|completed|refunded)"
    r"|(?:successfully|now)\s+(?:approved|processed|issued|refunded)"
    r"|i(?:'ve| have)\s+(?:approved|processed|issued)\s+(?:your |the )?refund"
    r"|(?:approved|processed|issued)\s+your\s+refund"
    r")",
    re.I,
)

_NEGATED = re.compile(r"\b(?:no|not|never|without|nor)\b[\s\w,'-]{0,12}$", re.I)
_DECLINED = re.compile(
    r"\b(?:can[’']?t|cannot|can not|won[’']t|will not|unable to|not able to)\b"
    r"[^.!?\n]{0,60}?\b(?:confirm|say|state|claim|pretend|report|promise|guarantee|tell you|include|write)\s+"
    r"(?:that\s+)?[“\"']?(?:[\w$,.*]+\s+){0,3}$",
    re.I,
)


def claims_approval(text: str) -> bool:
    """True if `text` affirmatively says a refund was granted."""
    return any(
        not _NEGATED.search(text[max(0, m.start() - 24) : m.start()])
        and not _DECLINED.search(text[max(0, m.start() - 100) : m.start()])
        for m in _APPROVAL_CLAIM.finditer(text)
    )


_BUDGET_MESSAGE = (
    "I'm sorry, this conversation has reached its usage limit for the day. "
    "Please start a new chat or contact a human specialist if you still need help."
)

_CORRECTION_PREFIX = (
    "_System note: the refund system did not record this as approved. "
    "Please disregard any approval language below — the database is the "
    "source of truth._\n\n"
)


class ConversationNotFound(Exception):
    pass


def new_conversation_id() -> str:
    return f"conv-{uuid.uuid4().hex}"


async def create_conversation(customer_id: str, cid: str | None = None) -> str:
    cid = cid or new_conversation_id()
    async with db.SessionLocal() as session:
        session.add(Conversation(id=cid, customer_id=customer_id))
        await session.commit()
    return cid


async def load_owned_conversation(customer_id: str, conversation_id: str) -> Conversation:
    async with db.SessionLocal() as session:
        convo = await session.get(Conversation, conversation_id)
    if convo is None or convo.customer_id != customer_id:
        raise ConversationNotFound(conversation_id)
    return convo


async def _history(conversation_id: str) -> list:
    async with db.SessionLocal() as session:
        rows = (
            await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.id.desc())
                .limit(_HISTORY_LIMIT)
            )
        ).all()
    return [
        HumanMessage(m.content) if m.role == "user" else AIMessage(m.content)
        for m in reversed(rows)
    ]


def _system_message() -> SystemMessage:
    prompt = get_system_prompt()
    if use_prompt_caching():
        return SystemMessage(
            content=[{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]
        )
    return SystemMessage(prompt)


def _step_type_for_tool(name: str) -> str:
    if name == "check_refund_eligibility":
        return "policy_eval"
    if name in _DECISION_TOOLS:
        return "decision"
    return "tool_result"


async def _save_message(cid: str, role: str, content: str) -> int:
    async with db.SessionLocal() as session:
        msg = Message(conversation_id=cid, role=role, content=content)
        session.add(msg)
        await session.commit()
        return msg.id


async def run_agent_turn(
    customer_id: str, conversation_id: str | None, user_text: str
) -> AsyncIterator[dict]:
    settings = get_settings()
    started = time.perf_counter()
    if conversation_id is None:
        conversation_id = await create_conversation(customer_id)
    convo = await load_owned_conversation(customer_id, conversation_id)
    cid = convo.id
    conversation_id_var.set(cid)

    # Persist the user's message
    user_msg_id = await _save_message(cid, "user", user_text)
    await broadcaster.publish(cid, {"kind": "message", "role": "user", "content": user_text})

    async def emit(step_type: str, node: str, payload: dict, tokens: int = 0) -> dict:
        async with db.SessionLocal() as session:
            values = {"event_seq": Conversation.event_seq + 1}
            if tokens:
                values["tokens_used"] = Conversation.tokens_used + tokens
            seq = await session.scalar(
                update(Conversation)
                .where(Conversation.id == cid)
                .values(**values)
                .returning(Conversation.event_seq)
            )
            created = utcnow()
            row_id = await session.scalar(
                insert(ReasoningEvent)
                .values(
                    conversation_id=cid,
                    message_id=user_msg_id,
                    seq=seq,
                    step_type=step_type,
                    node=node,
                    payload=payload,
                    created_at=created,
                )
                .returning(ReasoningEvent.id)
            )
            await session.commit()
        event = {
            "kind": "step",
            "id": row_id,
            "seq": seq,
            "step_type": step_type,
            "node": node,
            "payload": payload,
            "created_at": created.isoformat(),
        }
        await broadcaster.publish(cid, event)
        return event

    async def finish(final_text: str, outcome: str) -> list[dict]:
        await _save_message(cid, "assistant", final_text)
        final_event = {"kind": "message", "role": "assistant", "content": final_text}
        await broadcaster.publish(cid, final_event)
        TURNS.labels(outcome).inc()
        TURN_LATENCY.observe(time.perf_counter() - started)
        return [final_event, {"kind": "done"}]

    yield {"kind": "conversation", "conversation_id": cid}

    # Sentiment & Frustration Analysis
    if settings.sentiment_analysis_enabled:
        sentiment = analyze_sentiment(user_text)
        async with db.SessionLocal() as session:
            slog = SentimentLog(
                conversation_id=cid,
                sentiment_score=sentiment.score,
                sentiment_label=sentiment.label,
                detected_urgency=sentiment.detected_urgency,
            )
            session.add(slog)
            await session.commit()
        if sentiment.label == "frustrated" or sentiment.detected_urgency:
            yield await emit(
                "sentiment",
                "telemetry",
                {
                    "score": sentiment.score,
                    "label": sentiment.label,
                    "urgent": sentiment.detected_urgency,
                    "signals": sentiment.frustration_signals,
                },
            )

    # Prompt manipulation / injection detection
    flags = detect_injection(user_text)
    if flags:
        INJECTION_FLAGS.inc()
        yield await emit("injection_flag", "guard", {"patterns": flags, "text": user_text})

    # Token budget verification
    prior_tokens = convo.tokens_used or 0
    today_tokens = await shared.customer_tokens_today(customer_id)
    over_budget = (
        prior_tokens >= settings.max_conversation_tokens
        or today_tokens >= settings.customer_daily_token_budget
    )
    scripted = False
    if settings.is_demo:
        global_cap = settings.demo_global_daily_token_budget
        if over_budget or (global_cap and await shared.global_tokens_today() >= global_cap):
            scripted = True
            yield await emit(
                "notice",
                "guard",
                {
                    "reason": "demo_budget",
                    "message": "Demo token budget exhausted; running scripted model fallback with identical policy gates.",
                },
            )
    elif over_budget:
        yield await emit(
            "budget_exhausted",
            "guard",
            {
                "prior_tokens": prior_tokens,
                "limit": settings.max_conversation_tokens,
                "customer_tokens_today": today_tokens,
                "customer_daily_limit": settings.customer_daily_token_budget,
            },
        )
        for event in await finish(_BUDGET_MESSAGE, "budget_exhausted"):
            yield event
        return

    ctx = ToolContext(conversation_id=cid, verified_customer_id=customer_id)
    agent = get_agent(scripted=scripted)
    tracer = tracing.tracer_for_turn(cid, customer_id, scripted=scripted)
    run_config = {
        "recursion_limit": settings.agent_recursion_limit,
        "run_name": "refund-turn",
        **tool_config(ctx),
    }
    if tracer is not None:
        run_config["callbacks"] = [tracer]
    primary_model = primary_model_name()

    messages = [_system_message()] + await _history(cid)

    final_text = ""
    approved_in_turn = False
    decision_in_turn: str | None = None
    fallback_in_turn = False
    TURNS_IN_FLIGHT.inc()
    try:
        async with asyncio.timeout(settings.turn_timeout_s):
            async for chunk in agent.astream(
                {"messages": messages},
                stream_mode="updates",
                config=run_config,
            ):
                for node, update_ in chunk.items():
                    for msg in update_.get("messages", []):
                        if isinstance(msg, AIMessage):
                            usage = getattr(msg, "usage_metadata", None)
                            meta = getattr(msg, "response_metadata", None) or {}
                            model = meta.get("model") or meta.get("model_name") or ""
                            if model and not scripted and not model.startswith(primary_model):
                                fallback_in_turn = True
                                LLM_FALLBACKS.labels(model).inc()
                            if isinstance(msg.content, str) and msg.content.strip():
                                yield await emit("model", node, {"text": msg.content})
                                final_text = msg.content
                            elif isinstance(msg.content, list):
                                text = "".join(
                                    b.get("text", "")
                                    for b in msg.content
                                    if isinstance(b, dict) and b.get("type") == "text"
                                )
                                if text.strip():
                                    yield await emit("model", node, {"text": text})
                                    final_text = text
                            for call in msg.tool_calls or []:
                                TOOL_CALLS.labels(call["name"]).inc()
                                yield await emit(
                                    "tool_call",
                                    node,
                                    {"tool": call["name"], "args": call.get("args", {})},
                                )
                            if usage and not scripted:
                                tin = int(usage.get("input_tokens", 0))
                                tout = int(usage.get("output_tokens", 0))
                                LLM_TOKENS.labels("input").inc(tin)
                                LLM_TOKENS.labels("output").inc(tout)
                                await shared.add_customer_tokens(customer_id, tin + tout)
                                yield await emit(
                                    "usage",
                                    node,
                                    {"input_tokens": tin, "output_tokens": tout, "model": model},
                                    tokens=tin + tout,
                                )
                        elif isinstance(msg, ToolMessage):
                            try:
                                result = json.loads(msg.content)
                            except (json.JSONDecodeError, TypeError):
                                result = {"raw": str(msg.content)}
                            if UNTRUSTED_PLACEHOLDER in str(msg.content):
                                flags.append(f"tool:{msg.name}")
                                INJECTION_FLAGS.inc()
                                yield await emit(
                                    "injection_flag",
                                    "guard",
                                    {
                                        "patterns": ["instructions_in_tool_output"],
                                        "source": "tool",
                                        "tool": msg.name,
                                    },
                                )
                            if msg.name in _DECISION_TOOLS and isinstance(result, dict):
                                decision = result.get("decision")
                                if decision:
                                    REFUND_DECISIONS.labels(decision).inc()
                                    decision_in_turn = decision
                                if msg.name == "issue_refund" and decision == "approved":
                                    approved_in_turn = True
                            yield await emit(
                                _step_type_for_tool(msg.name),
                                node,
                                {"tool": msg.name, "result": result},
                            )
    except Exception as exc:
        log_event(
            logger,
            "agent turn failed",
            level=logging.ERROR,
            error_type=type(exc).__name__,
            error=str(exc)[:500],
        )
        TURNS.labels("error").inc()
        TURN_LATENCY.observe(time.perf_counter() - started)
        yield await emit("error", "agent", {"error_type": type(exc).__name__})
        trace = tracing.finish_turn(
            tracer,
            tracing.TurnOutcome(
                decision=decision_in_turn,
                injection_flagged=bool(flags),
                fallback=fallback_in_turn,
                error=True,
            ),
        )
        if trace:
            yield await emit("trace", "opik", trace)
        yield {"kind": "error", "message": _ERROR_MESSAGE}
        return
    finally:
        TURNS_IN_FLIGHT.dec()

    if not final_text:
        final_text = "I'm sorry, I wasn't able to complete that request."

    corrected = not approved_in_turn and claims_approval(final_text)
    if corrected:
        yield await emit(
            "output_correction",
            "sanitizer",
            {
                "reason": "Assistant claimed an approval that issue_refund did not produce this turn.",
                "original": final_text,
            },
        )
        final_text = _CORRECTION_PREFIX + final_text

    trace = tracing.finish_turn(
        tracer,
        tracing.TurnOutcome(
            decision=decision_in_turn,
            corrected=corrected,
            injection_flagged=bool(flags),
            fallback=fallback_in_turn,
        ),
    )
    if trace:
        yield await emit("trace", "opik", trace)

    for event in await finish(final_text, "ok"):
        yield event
