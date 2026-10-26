"""Local conversation routing and per-group message batching.

QQ identities and reply relations, never display names, establish the recipient.
A short uninterrupted session allows natural topic changes; outside that grace
period, topic/response cues are required. No extra model request is used.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import re
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

log = logging.getLogger("dafish.dialogue")


@dataclass(frozen=True)
class ParsedMessage:
    text: str
    at_self: bool
    media: bool
    reply_id: str
    mentioned_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class IndexedMessage:
    group_id: int
    user_id: int
    speaker: str
    text: str
    ts: float
    is_bot: bool = False


@dataclass(frozen=True)
class PendingTurn:
    group_id: int
    user_id: int
    message_id: str
    message: Any
    at_self: bool
    mentioned_ids: tuple[str, ...]
    reply_id: str
    reply_sender_id: str
    received_at: float
    first_received_at: float
    source_ids: tuple[str, ...]
    is_poke: bool = False
    rechecks: int = 0
    reserved_reason: str = ""
    target_revision: int = 0
    arrival_revision: int = 0


@dataclass
class DialogueState:
    user_id: int
    user_text: str
    bot_text: str
    updated_at: float
    followups: int = 0
    interrupted: bool = False


TOPIC_ENDINGS = ("不聊了", "先不聊", "不用回", "别回复", "换个话题", "不说这个", "说点别的")
_FILLERS = re.compile(
    "你们|我们|他们|这个|那个|这些|那些|怎么|什么|为什么|怎么样|怎么办|"
    "可以|觉得|一下|然后|还是|但是|就是|不是|一个|今天|现在|刚才|你好|哈哈|"
    "我|你|他|她|它|的|了|吗|呢|啊|呀|吧"
)


def topic_terms(text: str) -> set[str]:
    """Meaningful ASCII words and Chinese bigrams; no tokenizer dependency."""
    cleaned = _FILLERS.sub(" ", text.lower())
    terms = set(re.findall(r"[a-z][a-z0-9_]{1,}", cleaned))
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", cleaned):
        terms.update(chunk[i:i + 2] for i in range(len(chunk) - 1))
    return terms


class DialogueMixin:
    def _init_dialogue(self) -> None:
        self._pending: dict[int, deque[PendingTurn]] = {}
        self._workers: dict[int, asyncio.Task] = {}
        self._pending_changed: dict[int, asyncio.Event] = {}
        self._latest_from_user: dict[tuple[int, int], PendingTurn] = {}
        self._dialogues: dict[int, DialogueState] = {}
        self._active_turn: dict[int, PendingTurn] = {}
        # Tone belongs to a member's conversation, not the group's latest route.
        self._friendly_sessions: dict[tuple[int, int], float] = {}
        self._turn_tones: dict[int, tuple[int, str]] = {}
        self._closing = False
        self._arrival_revisions: dict[int, int] = {}
        self._second_cancel_events: dict[int, asyncio.Event] = {}
        self._overlap_tickets: dict[tuple[int, int], dict] = {}
        self._overlap_cohorts: dict[tuple[int, int], dict] = {}
        self._overlap_open: dict[int, tuple[int, int]] = {}
        self._overlap_interrupted: dict[int, tuple] = {}
        self._overlap_activity: dict[int, deque] = {}

    def _note_group_activity(self, group_id: int, user_id: int) -> int:
        revision = self._arrival_revisions.get(group_id, 0) + 1
        self._arrival_revisions[group_id] = revision
        if self._overlap_window():
            self._overlap_prune()
            if group_id in self._overlap_activity or len(self._overlap_activity) < 512:
                self._overlap_activity.setdefault(group_id, deque(maxlen=256)).append(
                    {"revision": revision, "user_id": user_id, "cohort": None, "created": time.monotonic()})
            state = self._dialogues.get(group_id)
            if state and state.user_id != user_id and not state.interrupted:
                if group_id in self._overlap_interrupted or len(self._overlap_interrupted) < 512:
                    self._overlap_interrupted[group_id] = (replace(state), time.monotonic(), revision)
        duel_activity = getattr(self, "_duel_activity", None)
        if callable(duel_activity):
            duel_activity(group_id, user_id, revision)
        event = self._second_cancel_events.get(group_id)
        if event is not None:
            event.set()
        state = self._dialogues.get(group_id)
        if state and state.user_id != user_id:
            state.interrupted = True
        return revision

    def _overlap_window(self) -> float:
        value = self.beh.get("concurrent_reply_window", 0)
        if isinstance(value, bool):
            return 0.0
        try:
            value = float(value)
        except (ValueError, TypeError, OverflowError):
            return 0.0
        return min(value, 2.0) if math.isfinite(value) and value > 0 else 0.0

    def _overlap_prune(self) -> None:
        now = time.monotonic()
        for key, ticket in list(self._overlap_tickets.items()):
            if now - ticket["created"] > 120 and key[0] not in self._workers:
                self._overlap_tickets.pop(key, None)
        live = {ticket["cohort"] for ticket in self._overlap_tickets.values()}
        for key in list(self._overlap_cohorts):
            if key not in live:
                self._overlap_cohorts.pop(key, None)
        for group, key in list(self._overlap_open.items()):
            if key not in self._overlap_cohorts:
                self._overlap_open.pop(group, None)
        for group, snapshot in list(self._overlap_interrupted.items()):
            if now - snapshot[1] > self._overlap_window():
                self._overlap_interrupted.pop(group, None)
        for group, events in list(self._overlap_activity.items()):
            if not events or (now - events[-1]["created"] > 120 and group not in self._workers):
                self._overlap_activity.pop(group, None)

    def _overlap_safe_turn(self, turn: PendingTurn) -> bool:
        if (self._closing or not self._overlap_window() or turn.is_poke or not turn.message_id
                or self._addressed_elsewhere(turn)
                or any(marker in turn.message.text for marker in TOPIC_ENDINGS)
                or self._is_tone_session_end(turn.message.text)
                or any(word and str(word) in turn.message.text for word in (self.beh.get("mute_keywords") or []))
                or any(word and str(word) in turn.message.text for word in (self.safety.get("blocked_input") or []))):
            return False
        allowed = getattr(self, "_group_allowed", None)
        if callable(allowed) and not allowed(turn.group_id):
            return False
        if turn.user_id in getattr(self, "_user_blacklist", set()):
            return False
        muted = getattr(self, "_is_muted", None)
        if callable(muted) and muted(turn.group_id, turn.user_id) and not turn.at_self:
            return False
        return True

    def _overlap_register(self, turn: PendingTurn) -> None:
        if not self._overlap_safe_turn(turn):
            return
        reason = ""
        if turn.at_self and self.beh.get("at_reply", True):
            reason = "at"
        elif turn.reply_sender_id == str(self.client.self_id):
            reason = "quote"
        elif self._calls_bot(turn.message.text):
            reason = "keyword"
        elif (callable(getattr(self, "_owner_chat_request", None))
                and self._owner_chat_request(turn) is not None):
            reason = "owner"
        elif (float(self.beh.get("followup_rate", 1)) > 0
                and self._related_followup(turn.group_id, turn.user_id, turn.message.text)):
            reason = "followup"
        if not reason:
            snapshot = self._overlap_interrupted.get(turn.group_id)
            if snapshot and snapshot[0].user_id == turn.user_id:
                state, interrupted_at, revision = snapshot
                breaker = self._overlap_tickets.get((turn.group_id, revision))
                cohort = self._overlap_cohorts.get(breaker["cohort"]) if breaker else None
                if (breaker and breaker["reason"] != "followup" and cohort
                        and turn.received_at <= cohort["deadline"]
                        and turn.received_at >= cohort["anchor"]
                        and float(self.beh.get("followup_rate", 1)) > 0
                        and self._related_followup_state(state, turn.message.text, time.monotonic())):
                    events = self._overlap_activity.get(turn.group_id, ())
                    if all(event["user_id"] == turn.user_id or event["cohort"] == breaker["cohort"]
                           for event in events if event["revision"] >= revision):
                        reason = "followup"
        if not reason:
            # An unaddressed suffix can inherit only its own earlier addressed
            # turn in this fixed window, even if another member spoke between.
            for prior in reversed(list(self._overlap_tickets.values())):
                cohort = self._overlap_cohorts.get(prior["cohort"])
                if (prior["cohort"][0] == turn.group_id and prior["user_id"] == turn.user_id
                        and prior["revision"] == turn.target_revision and cohort
                        and cohort["anchor"] <= turn.received_at <= cohort["deadline"]
                        and self._can_merge(prior["turn"], turn)):
                    reason = prior["reason"]
                    break
        if not reason:
            return
        self._overlap_prune()
        if len(self._overlap_tickets) >= 4096:
            return
        cohort_key = self._overlap_open.get(turn.group_id)
        cohort = self._overlap_cohorts.get(cohort_key)
        if not cohort or not cohort["anchor"] <= turn.received_at <= cohort["deadline"]:
            if len(self._overlap_cohorts) >= 512:
                return
            cohort_key = (turn.group_id, turn.arrival_revision)
            cohort = {"anchor": turn.received_at, "deadline": turn.received_at + self._overlap_window(),
                      "members": set(), "contended": False}
            self._overlap_cohorts[cohort_key] = cohort
            self._overlap_open[turn.group_id] = cohort_key
        cohort["members"].add(turn.user_id)
        cohort["contended"] = len(cohort["members"]) > 1
        ticket = {"cohort": cohort_key, "reason": reason, "user_id": turn.user_id,
                  "revision": turn.target_revision, "ids": frozenset(turn.source_ids or (turn.message_id,)),
                  "created": time.monotonic(), "arrival": turn.arrival_revision,
                  "bot_id": self.client.self_id, "ws": getattr(self.client, "ws", None), "turn": turn}
        self._overlap_tickets[(turn.group_id, turn.arrival_revision)] = ticket
        for event in self._overlap_activity.get(turn.group_id, ()):
            if event["revision"] == turn.arrival_revision:
                event["cohort"] = cohort_key
                break

    def _overlap_ticket(self, turn: PendingTurn):
        ids = set(turn.source_ids or (turn.message_id,))
        matches = [ticket for (group, _), ticket in self._overlap_tickets.items()
                   if group == turn.group_id and ticket["user_id"] == turn.user_id
                   and ticket["revision"] == turn.target_revision and ticket["ids"] & ids]
        return min(matches, key=lambda ticket: ticket["arrival"]) if matches else None

    def _overlap_ticket_valid(self, turn: PendingTurn, ticket) -> bool:
        if not ticket or not self._overlap_safe_turn(turn) or time.monotonic() - ticket["created"] > 120:
            return False
        cohort = self._overlap_cohorts.get(ticket["cohort"])
        latest = self._latest_from_user.get((turn.group_id, turn.user_id))
        if (not cohort or not cohort["contended"]
                or (latest and latest.target_revision > turn.target_revision)
                or self.client.self_id != ticket["bot_id"]
                or getattr(self.client, "ws", None) is not ticket["ws"]):
            return False
        if ticket["reason"] == "followup":
            events = self._overlap_activity.get(turn.group_id, ())
            if not events or events[0]["revision"] > ticket["arrival"] + 1:
                return False
            if any(event["user_id"] != turn.user_id and event["cohort"] != ticket["cohort"]
                   for event in events if event["revision"] > ticket["arrival"]):
                return False
        return True

    def _overlap_turn_ready(self, turn: PendingTurn) -> bool:
        return self._overlap_ticket_valid(turn, self._overlap_ticket(turn))

    def _overlap_followup(self, turn: PendingTurn) -> bool:
        ticket = self._overlap_ticket(turn)
        return bool(ticket and ticket["reason"] == "followup" and self._overlap_ticket_valid(turn, ticket))

    def _overlap_quote(self, turn: PendingTurn) -> bool:
        return self._overlap_turn_ready(turn)

    def _overlap_invalid(self, turn: PendingTurn) -> bool:
        ticket = self._overlap_ticket(turn)
        cohort = self._overlap_cohorts.get(ticket["cohort"]) if ticket else None
        return bool(ticket and cohort and cohort["contended"] and not self._overlap_ticket_valid(turn, ticket))

    def _second_reply_ready(self, turn: PendingTurn) -> bool:
        owner_request = getattr(self, "_owner_chat_request", None)
        task_material = callable(owner_request) and owner_request(turn) is not None
        return bool(not self._closing and turn.arrival_revision
                    and self._arrival_revisions.get(turn.group_id) == turn.arrival_revision
                    and self._followup_is_current(turn)
                    and (task_material or not any(marker in turn.message.text for marker in TOPIC_ENDINGS)))

    def _calls_bot(self, text: str) -> bool:
        if not self.beh.get("keyword_reply", True):
            return False
        for word in self.beh.get("keywords") or []:
            word = str(word).strip().lower()
            if not word:
                continue
            if word.isascii():
                if re.search(r"(?<![a-z0-9_])" + re.escape(word) + r"(?![a-z0-9_])", text.lower()):
                    return True
            elif word in text.lower():
                return True
        return False

    def _addressed_elsewhere(self, turn: PendingTurn) -> bool:
        owner_request = getattr(self, "_owner_chat_request", None)
        if callable(owner_request) and owner_request(turn) is not None:
            return False
        self_id = str(self.client.self_id)
        # An explicit @self or a known reply to self wins over mixed @ mentions.
        if turn.at_self or turn.reply_sender_id == self_id or turn.is_poke:
            return False
        if any(uid != self_id for uid in turn.mentioned_ids):
            return True
        # Unknown references are not guessed from a nickname or resolved via HTTP.
        return bool(turn.reply_id)

    def _is_direct_turn(self, turn: PendingTurn) -> bool:
        return not self._addressed_elsewhere(turn) and (
            turn.at_self or turn.reply_sender_id == str(self.client.self_id)
            or turn.is_poke or self._calls_bot(turn.message.text)
            or (callable(getattr(self, "_owner_chat_request", None))
                and self._owner_chat_request(turn) is not None)
        )

    def _enqueue_turn(self, turn: PendingTurn) -> None:
        if self._closing:
            return
        if not turn.arrival_revision:
            turn = replace(turn, arrival_revision=self._note_group_activity(turn.group_id, turn.user_id))
        key = (turn.group_id, turn.user_id)
        previous = self._latest_from_user.get(key)
        revision = previous.target_revision if previous else 0
        if (self._addressed_elsewhere(turn)
                or any(marker in turn.message.text for marker in TOPIC_ENDINGS)
                or self._is_tone_session_end(turn.message.text)
                or (self._overlap_window() and any(word and str(word) in turn.message.text
                                                  for word in (self.beh.get("mute_keywords") or [])))
                or any(word and str(word) in turn.message.text for word in (self.safety.get("blocked_input") or []))):
            revision += 1
        turn = replace(turn, target_revision=revision)
        self._latest_from_user[key] = turn
        self._overlap_register(turn)
        # Bound routing bookkeeping, including on long-lived public groups.
        if len(self._latest_from_user) > 4096:
            cutoff = time.monotonic() - 600
            for old_key, value in list(self._latest_from_user.items()):
                if value.received_at < cutoff:
                    self._latest_from_user.pop(old_key, None)
            while len(self._latest_from_user) > 4096:
                self._latest_from_user.pop(next(iter(self._latest_from_user)))
        state = self._dialogues.get(turn.group_id)
        if state and (state.user_id != turn.user_id or self._addressed_elsewhere(turn)):
            state.interrupted = True
        pending = self._pending.setdefault(turn.group_id, deque())
        limit = max(1, int(self.beh.get("pending_message_limit", 100)))
        if len(pending) >= limit:
            # Prefer dropping an unsolicited item over an explicitly addressed request.
            disposable = next((item for item in pending if not self._is_direct_turn(item)), pending[0])
            pending.remove(disposable)
            log.warning("群 %s 待处理消息超过上限，丢弃最早待处理项", turn.group_id)
        pending.append(turn)
        self._pending_changed.setdefault(turn.group_id, asyncio.Event()).set()
        if turn.group_id not in self._workers:
            self._workers[turn.group_id] = asyncio.create_task(
                self._group_worker(turn.group_id), name=f"qq-dialogue-{turn.group_id}"
            )

    def _merge_turns(self, first: PendingTurn, last: PendingTurn) -> PendingTurn:
        mentions = tuple(dict.fromkeys(first.mentioned_ids + last.mentioned_ids))
        text = (first.message.text + "\n" + last.message.text).strip()
        reply_id = last.reply_id or first.reply_id
        reply_sender = last.reply_sender_id if last.reply_id else first.reply_sender_id
        message = replace(
            last.message, text=text, mentions=mentions,
            quote=last.message.quote or first.message.quote,
            reply_to_id=reply_id, reply_to_sender_id=reply_sender,
        )
        return replace(
            last, message=message, at_self=first.at_self or last.at_self,
            mentioned_ids=mentions, reply_id=reply_id, reply_sender_id=reply_sender,
            source_ids=first.source_ids + last.source_ids,
            first_received_at=first.first_received_at,
            rechecks=max(first.rechecks, last.rechecks),
            reserved_reason=first.reserved_reason or last.reserved_reason,
        )

    def _can_merge(self, first: PendingTurn, last: PendingTurn) -> bool:
        if first.user_id != last.user_id or first.target_revision != last.target_revision or first.is_poke or last.is_poke:
            return False
        if len(first.source_ids) >= max(1, int(self.beh.get("merge_max_messages", 20))):
            return False
        # A new explicit addressee/reference starts a separate turn, not a suffix
        # to an earlier question to the bot. An unaddressed suffix may inherit.
        if last.mentioned_ids or last.reply_id:
            return (
                first.mentioned_ids == last.mentioned_ids
                and first.reply_id == last.reply_id
                and first.reply_sender_id == last.reply_sender_id
            ) or (not first.mentioned_ids and not first.reply_id)
        return True

    async def _take_batch(self, group_id: int) -> list[PendingTurn]:
        changed = self._pending_changed[group_id]
        pending = self._pending[group_id]
        while pending:
            changed.clear()
            items = list(pending)
            first = items[0]
            deadline = min(
                first.first_received_at + max(0.0, float(self.beh.get("merge_max_wait", 5.0))),
                items[-1].received_at + max(0.0, float(self.beh.get("merge_delay", 1.6))),
            )
            for item in items:
                if self._is_direct_turn(item):
                    deadline = min(deadline, item.first_received_at + max(
                        0.0, float(self.beh.get("merge_at_delay", 0.6))))
            overlap_deadlines = [self._overlap_cohorts[ticket["cohort"]]["deadline"]
                for item in items if (ticket := self._overlap_ticket(item)) is not None
                and ticket["cohort"] in self._overlap_cohorts]
            if overlap_deadlines:
                # A fixed first-request window, never a sliding wait after every arrival.
                deadline = max(deadline, min(overlap_deadlines))
            if (len(items) >= max(1, int(self.beh.get("merge_max_messages", 20)))
                    and (not overlap_deadlines or time.monotonic() >= min(overlap_deadlines))):
                break
            delay = deadline - time.monotonic()
            if delay <= 0:
                break
            try:
                await asyncio.wait_for(changed.wait(), timeout=delay)
            except asyncio.TimeoutError:
                break
        count = min(len(pending), max(1, int(self.beh.get("merge_max_messages", 20))))
        items = [pending.popleft() for _ in range(count)]
        merged: list[PendingTurn] = []
        for item in items:
            merge_index = len(merged) - 1 if merged and self._can_merge(merged[-1], item) else None
            if self._overlap_window() and self._overlap_ticket(item) is not None:
                for index in range(len(merged) - 1, -1, -1):
                    prior = merged[index]
                    first_ticket = self._overlap_ticket(prior)
                    last_ticket = self._overlap_ticket(item)
                    if (first_ticket and last_ticket and first_ticket["cohort"] == last_ticket["cohort"]
                            and self._can_merge(prior, item)):
                        merge_index = index
                        break
            if merge_index is not None:
                merged[merge_index] = self._merge_turns(merged[merge_index], item)
            else:
                merged.append(item)
        return merged

    async def _group_worker(self, group_id: int) -> None:
        try:
            while self._pending.get(group_id):
                batch = await self._take_batch(group_id)
                for turn in batch:
                    self._active_turn[group_id] = turn
                    try:
                        await self._process_turn(turn)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        log.exception("群 %s 对话批次处理失败", group_id)
                    finally:
                        owners = getattr(self, "_delivery_owners", None)
                        if owners is not None and owners.get(group_id, (None,))[0] is turn:
                            owners.pop(group_id, None)
                        self._active_turn.pop(group_id, None)
                        self._turn_tones.pop(group_id, None)
        finally:
            self._workers.pop(group_id, None)
            self._pending.pop(group_id, None)
            self._pending_changed.pop(group_id, None)
            for key in [key for key in self._overlap_tickets if key[0] == group_id]:
                self._overlap_tickets.pop(key, None)
            for key in [key for key in self._overlap_cohorts if key[0] == group_id]:
                self._overlap_cohorts.pop(key, None)
            self._overlap_open.pop(group_id, None)
            self._overlap_interrupted.pop(group_id, None)
            self._overlap_activity.pop(group_id, None)

    async def wait_idle(self, group_id: int | None = None) -> None:
        """Wait for currently queued turns; mainly useful for offline checks."""
        while True:
            tasks = [task for gid, task in list(self._workers.items())
                     if group_id is None or group_id == gid]
            if not tasks:
                return
            await asyncio.gather(*(asyncio.shield(task) for task in tasks))

    async def close(self) -> None:
        self._closing = True
        tasks = list(self._workers.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._pending.clear()
        self._pending_changed.clear()
        self._latest_from_user.clear()
        self._arrival_revisions.clear()
        self._second_cancel_events.clear()
        self._overlap_tickets.clear()
        self._overlap_cohorts.clear()
        self._overlap_open.clear()
        self._overlap_interrupted.clear()
        self._overlap_activity.clear()
        self._friendly_sessions.clear()
        self._turn_tones.clear()
        owners = getattr(self, "_delivery_owners", None)
        if owners is not None:
            owners.clear()
        await self.brain.close()

    def _superseded(self, turn: PendingTurn) -> bool:
        latest = self._latest_from_user.get((turn.group_id, turn.user_id))
        return bool(latest and latest.target_revision > turn.target_revision) or self._overlap_invalid(turn)

    def _defer_late_supplement(self, turn: PendingTurn, reason: str) -> bool:
        """Replan at most once if the same speaker adds a contiguous suffix."""
        if turn.rechecks >= 1 or turn.is_poke:
            return False
        pending = self._pending.get(turn.group_id)
        if not pending or not self._can_merge(turn, pending[0]):
            return False
        merged = turn
        while pending and self._can_merge(merged, pending[0]):
            merged = self._merge_turns(merged, pending.popleft())
        pending.appendleft(replace(merged, rechecks=1, reserved_reason=reason))
        self._pending_changed[turn.group_id].set()
        return True

    def _related_followup(self, group_id: int, user_id: int, text: str) -> bool:
        state = self._dialogues.get(group_id)
        if not state or state.user_id != user_id or state.interrupted:
            return False
        return self._related_followup_state(state, text, time.monotonic())

    def _related_followup_state(self, state: DialogueState, text: str, now: float) -> bool:
        age = now - state.updated_at
        window = max(0.0, float(self.beh.get("followup_window", 120)))
        if age > window:
            return False
        if state.followups >= max(0, int(self.beh.get("followup_max_turns", 6))):
            return False
        if not text.strip() or any(marker in text for marker in TOPIC_ENDINGS):
            return False
        # Once a conversation is established, the same uninterrupted speaker can
        # answer naturally or change subject without repeating topic keywords.
        grace = min(window, max(0.0, float(self.beh.get("followup_grace_window", 30))))
        if age <= grace:
            return True
        # Beyond the open grace period, explicit conversational linkage remains
        # useful even when the wording differs from the previous question.
        if re.search(r"(?:找你|跟你|和你|陪你|与你).{0,6}(?:聊|说话)|"
                     r"(?:你陪我|你跟我|你和我).{0,6}(?:聊|说话)|"
                     r"(?:我想问你|问问你|你刚才|你刚刚|你说的|你觉得|你怎么看)|"
                     r"(?:chat|talk|speak)\s+(?:to|with)\s+you|you\s+(?:said|asked|mean)",
                     text, re.I):
            return True
        # Strip configured bot aliases from both sides: merely sharing the bot's
        # name is not evidence of a shared topic.
        def terms(value: str) -> set[str]:
            for alias in self.beh.get("keywords") or []:
                if alias:
                    value = re.sub(re.escape(str(alias)), " ", value, flags=re.I)
            return topic_terms(value)
        shared = terms(text) & (terms(state.user_text) | terms(state.bot_text))
        if shared:
            return True
        return bool(not state.interrupted and age <= min(10.0, float(self.beh.get("followup_window", 90)))
                    and re.fullmatch(r"\s*(?:那|这个|这样|所以)?(?:呢|是吗|为什么|为啥|然后呢|还有呢|可以吗|行吗)[？?！!。 ]*", text))

    def _followup_is_current(self, turn: PendingTurn) -> bool:
        state = self._dialogues.get(turn.group_id)
        return bool((state and state.user_id == turn.user_id
                    and not state.interrupted and not self._superseded(turn))
                    or self._overlap_followup(turn))

    async def _wait_followup_gap(self, turn: PendingTurn) -> bool:
        """A normal fast answer waits for direct cooldown instead of being lost."""
        now = time.monotonic()
        ready = max(self._last_any + float(self.beh["min_gap"]),
                    self._last_group.get(turn.group_id, 0.0) + float(self.beh["direct_cooldown"]),
                    self._last_user.get((turn.group_id, turn.user_id), 0.0)
                    + float(self.beh["direct_user_cooldown"]))
        if ready > now:
            log.info("[群 %s] 续聊等待冷却 %.2f 秒", turn.group_id, ready - now)
            await asyncio.sleep(ready - now)
        return self._followup_is_current(turn)

    def _poke_related_message(self, turn: PendingTurn) -> dict[str, Any] | None:
        """A poke may remind us of a recent unanswered utterance, not revive a topic.

        Match stable IDs, never a nickname or a guessed pronoun. Only the latest
        human message in this group is eligible; another member's intervening
        speech, a delivered answer, missing identity, or age breaks the link.
        """
        window = max(0.0, float(self.beh.get("poke_context_window", 45)))
        if not turn.is_poke or not window:
            return None
        records = list(self.store._data.get(turn.group_id) or ())
        candidate = next((m for m in reversed(records) if not m.is_bot), None)
        if candidate is None or str(candidate.sender_id) != str(turn.user_id):
            return None
        message_id = str(candidate.message_id or "")
        # Do not attach a message that arrived after this selected poke while
        # the group worker was busy; it belongs to its own pending turn.
        age = float(turn.message.ts) - float(candidate.ts)
        wall_age = time.time() - float(candidate.ts)
        if (not message_id or not candidate.text.strip()
                or not (0 <= age <= window and 0 <= wall_age <= window)):
            return None
        if any(m.is_bot and str(m.reply_to_id or "") == message_id for m in records):
            return None
        related = {
            "role": "group_member", "event_kind": "message",
            "sender": {"id": str(candidate.sender_id), "display_name": candidate.speaker,
                       "nickname": candidate.nickname or None, "card": candidate.card or None},
            "message_id": message_id, "content": candidate.text[:2000],
        }
        if candidate.quote:
            related["quote"] = candidate.quote
        if candidate.reply_to_id or candidate.reply_to_sender_id:
            related["reply_to"] = {"message_id": candidate.reply_to_id or None,
                                   "sender_id": candidate.reply_to_sender_id or None}
        if candidate.mentions:
            related["mentions"] = list(candidate.mentions)
        return related

    def _reply_context(self, turn: PendingTurn) -> tuple[str, str]:
        rows = [json.loads(row) for row in self.store.transcript(turn.group_id).splitlines() if row.strip()]
        source_ids = set(turn.source_ids)
        rows = [row for row in rows if str(row.get("message_id", "")) not in source_ids]
        # The selected turn must be last even when another group member spoke
        # later; the prompt explicitly targets these stable IDs, not that member.
        target = {
            "role": "group_member",
            "event_kind": "poke" if turn.is_poke else "message",
            "sender": {"id": str(turn.user_id), "display_name": turn.message.speaker,
                       "nickname": turn.message.nickname or None, "card": turn.message.card or None},
            "message_id": turn.message_id or None,
            "source_message_ids": list(turn.source_ids),
            "content": turn.message.text[:2000],
        }
        if turn.message.quote:
            target["quote"] = turn.message.quote
        if turn.reply_id:
            target["reply_to"] = {"message_id": turn.reply_id, "sender_id": turn.reply_sender_id or None}
        if turn.mentioned_ids:
            target["mentions"] = list(turn.mentioned_ids)
        if turn.is_poke:
            related = self._poke_related_message(turn)
            if related is not None:
                target["related_message"] = related
        rows.append(target)
        note = "【程序选定回复对象，不可改成其他群友】" + json.dumps(
            {"target_user_id": str(turn.user_id), "target_message_id": turn.message_id or None,
             "source_message_ids": list(turn.source_ids)}, ensure_ascii=False
        ) + "。最后一行是本次选定目标的合并发言，不一定是群里按时间最新的一条。其他人的话仅作背景，不得归给目标。"
        if turn.is_poke:
            note += ("最后一行 event_kind=poke 是动作事件，不是群友说了标记里的字。"
                     "若带 related_message，它是该成员近期尚未收到回复的原话，戳可能是在提醒你接这句话；"
                     "先结合原话理解意图，不要只回一句又戳。没有关联原话时不猜测、更不翻旧话题。")
        return "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows), note
