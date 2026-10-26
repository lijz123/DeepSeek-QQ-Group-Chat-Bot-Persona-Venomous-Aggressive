"""Ephemeral, identity-scoped duel sessions; one idle followup, never a spam loop.

The tracker never sends, sleeps, calls a model, or stores account data on disk.
The mixin submits through the existing guarded notification sender. A session is
claimed before a send attempt, so an uncertain acknowledgement is never retried.
"""
from __future__ import annotations

from output_policy import daily_limit, limit_daily_text

import asyncio
from collections import deque
from dataclasses import dataclass, field
import logging
import math
import re
import time
from duel_generation import normalize_followup


log = logging.getLogger("dafish")


def _finite(value, default: float, low: float, high: float) -> float:
    if type(value) is bool:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) and low <= number <= high else default


def _stamp(value: float | None) -> float:
    if type(value) is bool:
        raise ValueError("Duel timestamps must be numeric, not booleans")
    stamp = time.monotonic() if value is None else float(value)
    if not math.isfinite(stamp) or stamp < 0:
        raise ValueError("Duel timestamps must be finite and nonnegative")
    return stamp

@dataclass(frozen=True)
class DuelSession:
    group_id: int
    user_id: int
    bot_id: int
    revision: int
    opened_at: float
    due_at: float
    expires_at: float
    user_text: str
    bot_text: str
    connection: object | None = field(default=None, repr=False, compare=False)


class DuelTracker:
    """One replaceable session per group, with identity-based atomic claims."""

    def __init__(self, max_groups: int = 256) -> None:
        self.max_groups = int(_finite(max_groups, 256, 1, 256))
        self._sessions: dict[int, DuelSession] = {}

    def __len__(self) -> int:
        return len(self._sessions)

    def current(self, group_id: int) -> DuelSession | None:
        return self._sessions.get(group_id)

    def cancel(self, group_id: int) -> DuelSession | None:
        return self._sessions.pop(group_id, None)

    def clear(self) -> None:
        self._sessions.clear()

    def arm(self, group_id: int, user_id: int, bot_id: int, revision: int,
            user_text: str, bot_text: str, *, now: float | None = None,
            idle_seconds: float = 60, expiry_seconds: float = 180,
            connection: object | None = None) -> DuelSession | None:
        if (any(type(value) is not int or value <= 0 for value in
                (group_id, user_id, bot_id, revision))
                or not isinstance(user_text, str) or not user_text.strip()
                or not isinstance(bot_text, str) or not bot_text.strip()):
            return None
        stamp = _stamp(now)
        idle = _finite(idle_seconds, 60, 1, 600)
        expiry = _finite(expiry_seconds, 180, 2, 1800)
        if expiry <= idle:
            expiry = min(1800, idle + max(60, idle))
        self.due(stamp)  # Expire old entries before applying the memory bound.
        old = self._sessions.get(group_id)
        same = bool(old and old.user_id == user_id and old.bot_id == bot_id
                    and old.revision == revision and old.connection is connection)
        opened = old.opened_at if same else stamp
        expires = old.expires_at if same else stamp + expiry
        if stamp + idle >= expires:
            self.cancel(group_id)
            return None
        if old is None and len(self._sessions) >= self.max_groups:
            oldest = min(self._sessions, key=lambda key: self._sessions[key].opened_at)
            self.cancel(oldest)
        session = DuelSession(group_id, user_id, bot_id, revision, opened,
                              stamp + idle, expires, user_text[:1500],
                              bot_text[:4000], connection)
        self._sessions[group_id] = session
        return session

    def due(self, now: float | None = None) -> tuple[DuelSession, ...]:
        stamp = _stamp(now)
        for group_id, session in tuple(self._sessions.items()):
            if stamp >= session.expires_at:
                self.cancel(group_id)
        return tuple(sorted((session for session in self._sessions.values()
                             if stamp >= session.due_at), key=lambda session: session.due_at))

    def claim(self, session: DuelSession, now: float | None = None) -> bool:
        stamp = _stamp(now)
        if self.current(session.group_id) is not session:
            return False
        if stamp >= session.expires_at:
            self.cancel(session.group_id)
            return False
        if stamp < session.due_at:
            return False
        self.cancel(session.group_id)
        return True


class DuelMixin:
    def _init_duels(self) -> None:
        self._duels = DuelTracker(self._duel_settings()["max_groups"])
        self._duel_task = None
        self._duel_generating = {}
        self._duel_recent = {}

    def _duel_settings(self) -> dict:
        raw = self.cfg.get("duel", {})
        raw = raw if isinstance(raw, dict) else {}
        idle = _finite(raw.get("idle_seconds", 60), 60, 1, 600)
        expiry = _finite(raw.get("expiry_seconds", 180), 180, 2, 1800)
        if expiry <= idle:
            expiry = min(1800, idle + max(60, idle))
        return {
            "enabled": raw.get("enabled", True) is True,
            "idle_seconds": idle, "expiry_seconds": expiry,
            "max_groups": int(_finite(raw.get("max_groups", 256), 256, 1, 256)),
            "generation_timeout": _finite(raw.get("generation_timeout", 15), 15, .1, 60),
            "poll_interval": _finite(raw.get("poll_interval", 1), 1, .2, 10),
        }

    def _start_duels(self) -> None:
        if (not self._closing and self._duel_settings()["enabled"]
                and not self.cfg["runtime"].get("dry_run")
                and (self._duel_task is None or self._duel_task.done())):
            self._duel_task = asyncio.create_task(self._duel_loop())

    def _duel_activity(self, group_id: int, user_id: int, revision: int) -> None:
        # Any human message or poke invalidates the idle premise immediately.
        tracker = getattr(self, "_duels", None)
        if tracker is not None:
            tracker.cancel(group_id)
        pending = getattr(self, "_duel_generating", {}).get(group_id)
        if pending is not None:
            pending[1].cancel()

    def _duel_record_delivery(self, group_id: int, user_id: int, text: str, reason: str) -> None:
        """Called only after root validates an actual positive delivery receipt."""
        settings = self._duel_settings()
        if (self._closing or not settings["enabled"] or self.cfg["runtime"].get("dry_run")
                or not self._notification_ready()):
            self._duels.clear()
            return
        if reason in ("fixed", "image", "wake", "mute", "duel", "duel_followup"):
            return
        turn = self._active_turn.get(group_id)
        if (turn is None or turn.user_id != user_id
                or self._turn_tones.get(group_id) != (user_id, "retaliation")
                or not self._retaliation_for(turn)):
            return
        try:
            if not self._duel_semantics_confirmed(turn, text):
                return
        except Exception:
            log.warning("互怼会话未建立：生成回执无法核对")
            return
        dialogue = self._dialogues.get(group_id)
        revision = self._arrival_revisions.get(group_id, 0)
        if (dialogue is None or dialogue.user_id != user_id or dialogue.interrupted
                or dialogue.user_text != turn.message.text
                or revision != turn.arrival_revision
                or not self._notification_allowed(group_id, user_id, (user_id,))):
            return
        self._duels.arm(group_id, user_id, self.client.self_id, revision,
            dialogue.user_text, dialogue.bot_text, now=time.monotonic(),
            idle_seconds=settings["idle_seconds"], expiry_seconds=settings["expiry_seconds"],
            connection=self.client.ws)

    def _duel_follow_state(self, session: DuelSession, stamp: float, body: str | None = None) -> str:
        """Rechecked after all awaits and immediately before claiming the send."""
        if self._duels.current(session.group_id) is not session:
            return "invalid"
        if (self._closing or not self._duel_settings()["enabled"]
                or self.cfg["runtime"].get("dry_run") or stamp >= session.expires_at
                or not self._notification_ready()
                or self.client.self_id != session.bot_id
                or self.client.ws is not session.connection
                or self._arrival_revisions.get(session.group_id) != session.revision):
            return "invalid"
        dialogue = self._dialogues.get(session.group_id)
        if (dialogue is None or dialogue.user_id != session.user_id or dialogue.interrupted
                or dialogue.user_text != session.user_text or dialogue.bot_text != session.bot_text
                or not self._notification_allowed(session.group_id, session.user_id, (session.user_id,))
                or (body is not None and not self._notification_body_allowed(body))):
            return "invalid"
        if self.store.trailing_bot_count(session.group_id) >= int(self.beh["max_chain"]):
            return "invalid"
        worker = self._workers.get(session.group_id)
        if (session.group_id in self._active_turn or self._pending.get(session.group_id)
                or (worker is not None and not worker.done())):
            return "busy"
        return "ready" if stamp >= session.due_at else "early"

    def _duel_cancel_session(self, session: DuelSession) -> None:
        if self._duels.current(session.group_id) is session:
            self._duels.cancel(session.group_id)

    def _duel_recent_key(self, session: DuelSession) -> tuple[int, int, int]:
        return (session.bot_id, session.group_id, session.user_id)

    def _duel_remember_text(self, session: DuelSession, body: str) -> None:
        key = self._duel_recent_key(session)
        if key not in self._duel_recent and len(self._duel_recent) >= 256:
            self._duel_recent.pop(next(iter(self._duel_recent)))
        self._duel_recent.setdefault(key, deque(maxlen=5)).append(body)

    def _duel_daily_remaining(self, session) -> int:
        limit = daily_limit(self.cfg)
        if not limit:
            return 0
        used = sum(value[0] for key, value in getattr(self, "_daily_delivery_used", {}).items()
                   if len(key) >= 4 and key[0] == session.group_id and key[1] == session.user_id
                   and key[3] == session.revision)
        return max(0, limit-used)

    async def _duel_tick(self, now: float | None = None) -> None:
        settings = self._duel_settings()
        if (self._closing or not settings["enabled"] or self.cfg["runtime"].get("dry_run")
                or not self._notification_ready()):
            self._duels.clear()
            for _, task in tuple(self._duel_generating.values()):
                task.cancel()
            return
        stamp = _stamp(now)
        started = time.monotonic()
        for session in self._duels.due(stamp):
            current_stamp = stamp + max(0.0, time.monotonic() - started)
            state = self._duel_follow_state(session, current_stamp)
            if state == "invalid":
                self._duel_cancel_session(session)
                continue
            if state != "ready" or session.group_id in self._duel_generating:
                continue
            if not self.limiter.allow_many(self._second_reply_limits(session.group_id), consume=False):
                continue
            remaining = self._duel_daily_remaining(session) if daily_limit(self.cfg) else 0
            if daily_limit(self.cfg) and remaining <= 1:
                self._duel_cancel_session(session)
                continue
            generate = getattr(self.brain, "duel_followup", None)
            if not callable(generate):
                self._duel_cancel_session(session)
                continue
            recent = tuple(self._duel_recent.get(self._duel_recent_key(session), ()))
            generation_started = time.monotonic()
            try:
                task = asyncio.create_task(generate(session.group_id, session.user_id,
                    session.user_text, session.bot_text, recent_texts=recent,
                    **({"remaining_chars": max(0, remaining-1)} if daily_limit(self.cfg) and callable(getattr(self.brain, "output_kind_for", None)) else {})))
            except Exception:
                self._duel_cancel_session(session)
                log.warning("互怼追问生成接口不可用，跳过且不使用预制回复")
                continue
            owned = (session, task)
            self._duel_generating[session.group_id] = owned
            try:
                body = await asyncio.wait_for(task, timeout=settings["generation_timeout"])
                if time.monotonic() - generation_started >= settings["generation_timeout"]:
                    log.info("互怼追问生成超时，跳过且不使用预制回复")
                    continue
                current_stamp = stamp + max(0.0, time.monotonic() - started)
                limit = max(0, self._duel_daily_remaining(session)-1) if daily_limit(self.cfg) else int(_finite(self.beh.get("max_chars", 40), 40, 1, 200))
                if daily_limit(self.cfg) and isinstance(body, str):
                    body = limit_daily_text(body, limit)
                if (not isinstance(body, str) or not body.strip() or len(body) > limit
                        or re.search(r"@\S|\[CQ:|[\r\n]", body, re.I)
                        or self._duel_follow_state(session, current_stamp, body) != "ready"):
                    log.info("互怼追问跳过：会话已变更或模型未提供可发送文本")
                    continue
                body = body.strip()
                norm = normalize_followup
                if not norm(body) or any(norm(body) == norm(old) for old in recent):
                    log.info("互怼追问跳过：与近期措辞重复")
                    continue

                def claim(s=session) -> bool:
                    current_stamp = stamp + max(0.0, time.monotonic() - started)
                    if self._duel_follow_state(s, current_stamp, body) != "ready":
                        return False
                    if daily_limit(self.cfg) and len(body)+1 > self._duel_daily_remaining(s):
                        return False
                    if not self._duels.claim(s, current_stamp):
                        return False
                    self._duel_remember_text(s, body)
                    return True

                status = await self._send_notification(session.group_id, session.user_id,
                    (session.user_id,), body, before_send=claim,
                    expected_bot_id=session.bot_id)
                if daily_limit(self.cfg) and status in ("sent", "submitted", "uncertain"):
                    for key, value in list(getattr(self, "_daily_delivery_used", {}).items()):
                        if (len(key) >= 4 and key[0] == session.group_id and key[1] == session.user_id
                                and key[3] == session.revision):
                            self._daily_delivery_used[key] = (value[0] + len(body) + 1, time.monotonic())
                            break
                log.info("模型决定的一次性互怼追问状态=%s", status)
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise
                log.info("互怼追问取消：等待期间有新消息或会话结束")
            except Exception:
                # If the guarded claim occurred, it is already gone. Never retry.
                log.warning("互怼追问未完成，跳过且不使用预制回复")
            finally:
                self._duel_cancel_session(session)
                if self._duel_generating.get(session.group_id) is owned:
                    self._duel_generating.pop(session.group_id, None)
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def _duel_loop(self) -> None:
        while not self._closing:
            try:
                await self._duel_tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("互怼会话检查异常；未确认发送不自动重试")
            await asyncio.sleep(self._duel_settings()["poll_interval"])

    async def close(self) -> None:
        self._closing = True
        task = self._duel_task
        self._duel_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        tasks = [value[1] for value in tuple(self._duel_generating.values())]
        for child in tasks:
            child.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._duel_generating.clear()
        self._duel_recent.clear()
        self._duels.clear()
        await super().close()
