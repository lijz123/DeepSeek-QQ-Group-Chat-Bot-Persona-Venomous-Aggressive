"""Lossless text-part delivery through the existing OneBot send transaction.

Parts are a single logical model response, not new model requests or permission
grants. Every physical message still passes the bot's locked send-time checks.
No failed or uncertain physical request is retried.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time

from output_policy import split_output, wide_output


log = logging.getLogger("dafish.output")

QUOTA_WAIT_SECONDS = 61.0
POLL_SECONDS = 0.1
_pause = asyncio.sleep


def wide_delivery_current(bot, context: dict) -> bool:
    """Recheck ownership and stop conditions without consuming any send quota."""
    if not isinstance(context, dict) or getattr(bot, "_closing", False):
        return False
    group_id, user_id = context.get("group_id"), context.get("user_id")
    contexts = getattr(bot, "_wide_delivery_context", {})
    if not isinstance(contexts, dict) or contexts.get(group_id) is not context:
        return False
    if getattr(bot.client, "self_id", None) != context.get("bot_id"):
        return False
    if getattr(bot.client, "ws", None) is not context.get("ws"):
        return False
    if getattr(bot, "_active_turn", {}).get(group_id) is not context.get("turn"):
        return False
    allowed = getattr(bot, "_group_allowed", None)
    if callable(allowed) and not allowed(group_id):
        return False
    if user_id in getattr(bot, "_user_blacklist", ()):
        return False
    muted = getattr(bot, "_is_muted", None)
    if callable(muted) and muted(group_id, user_id):
        return False
    turn = context.get("turn")
    if turn is not None:
        if turn.group_id != group_id or turn.user_id != user_id:
            return False
        if str(turn.message_id or "") != str(context.get("message_id") or ""):
            return False
        superseded = getattr(bot, "_superseded", None)
        if callable(superseded) and superseded(turn):
            return False
        elsewhere = getattr(bot, "_addressed_elsewhere", None)
        if callable(elsewhere) and elsewhere(turn):
            return False
        blocked = getattr(bot, "safety", {}).get("blocked_input", ()) or ()
        if any(word and str(word) in turn.message.text for word in blocked):
            return False
    owner = getattr(bot, "_delivery_owners", {}).get(group_id)
    if owner and (owner[0] is not turn or owner[1] != context.get("bot_id")
                  or owner[2] is not context.get("ws")):
        return False
    return True


async def _wait_quota(bot, context: dict, deadline: float) -> str:
    """Wait at most one quota window, yielding often enough to observe stops."""
    limits = bot._second_reply_limits(context["group_id"])
    while wide_delivery_current(bot, context):
        if time.monotonic() > deadline:
            return "quota_wait_expired"
        if bot.limiter.allow_many(limits, consume=False):
            return "ready"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "quota_wait_expired"
        await _pause(min(POLL_SECONDS, remaining))
    return "cancelled"


def _receipt(bot, group_id: int, context: dict, status: str) -> None:
    receipts = getattr(bot, "_wide_delivery_receipts", None)
    if not isinstance(receipts, dict):
        receipts = {}
        bot._wide_delivery_receipts = receipts
    if len(receipts) >= 256 and group_id not in receipts:
        receipts.pop(next(iter(receipts)))
    turn = context.get("turn")
    if status != "complete":
        log.warning("长输出未完整发送 status=%s sent_parts=%d total_parts=%d", status, context["sent_parts"], context["total_parts"])
    receipts[group_id] = {
        "complete": status == "complete",
        "sent_parts": context["sent_parts"],
        "total_parts": context["total_parts"],
        "status": status,
        "source_turn": (getattr(turn, "arrival_revision", 0), turn.message_id) if turn is not None else None,
    }


async def deliver_wide_output(bot, group_id: int, text: str, message_id,
                              user_id: int, reason: str) -> bool:
    """Deliver full text in lossless physical parts; never silently truncate it.

The callback ``_send_single`` must consume the original group/global quotas at
send time while the context is present, including legacy non-planning mode.
Only a callback status of ``quota`` proves no outbound request happened and may
be retried after waiting; every other false/uncertain result stops this reply.
"""
    if not isinstance(text, str) or not text:
        return False
    settings = bot.cfg.get("output", {}) if isinstance(bot.cfg, dict) else {}
    chunk_chars = settings.get("chunk_chars", 1500) if isinstance(settings, dict) else 1500
    if type(chunk_chars) is not int:
        chunk_chars = 1500
    chunk_chars = min(2000, max(256, chunk_chars))
    if not wide_output(bot.cfg) or len(text) <= chunk_chars:
        return await bot._send_single(group_id, text, message_id, user_id, reason)
    parts = split_output(text, chunk_chars)
    if not parts or "".join(parts) != text:
        return False
    contexts = getattr(bot, "_wide_delivery_context", None)
    if not isinstance(contexts, dict):
        contexts = {}
        bot._wide_delivery_context = contexts
    if group_id in contexts:
        # Do not replace an in-flight source ownership context with another task.
        return False
    turn = getattr(bot, "_active_turn", {}).get(group_id)
    context = {
        "group_id": group_id, "user_id": user_id,
        "message_id": str(message_id or ""), "turn": turn,
        "bot_id": bot.client.self_id, "ws": getattr(bot.client, "ws", None),
        "reason": reason, "total_parts": len(parts), "sent_parts": 0,
        "last_send_status": "pending",
    }
    contexts[group_id] = context
    status = "cancelled"
    try:
        for part in parts:
            deadline = time.monotonic() + QUOTA_WAIT_SECONDS
            while True:
                ready = await _wait_quota(bot, context, deadline)
                if ready != "ready":
                    status = ready
                    return False
                context["last_send_status"] = "pending"
                sent = await bot._send_single(group_id, part, message_id, user_id, reason)
                if sent:
                    context["sent_parts"] += 1
                    break
                result = context.get("last_send_status")
                if result == "quota":
                    # A competing group can consume quota between the optimistic
                    # check and the locked send gate. This status means no send.
                    if time.monotonic() >= deadline:
                        status = "quota_wait_expired"
                        return False
                    await _pause(min(POLL_SECONDS, max(0.0, deadline-time.monotonic())))
                    continue
                status = "uncertain" if result == "uncertain" else (
                    "cancelled" if not wide_delivery_current(bot, context) else "send_failed")
                return False
        status = "complete"
        return True
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    except Exception:
        status = "send_failed"
        return False
    finally:
        try:
            commit = getattr(bot, "_commit_wide_delivery", None)
            if callable(commit):
                result = commit(context)
                if inspect.isawaitable(result):
                    await result
        finally:
            _receipt(bot, group_id, context, status)
            if contexts.get(group_id) is context:
                contexts.pop(group_id, None)
