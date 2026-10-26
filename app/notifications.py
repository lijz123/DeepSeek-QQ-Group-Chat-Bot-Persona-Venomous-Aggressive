"""Explicit QQ mentions and durable reminders; no model decides whom to notify."""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from reminders import ReminderStore, parse_command, format_time, LOCAL_TZ
from voice import QQVoice, VoiceUnavailable, parse_voice_command
from project_paths import project_path

log = logging.getLogger("dafish")


def human_reminder_time(timestamp: float, now: float | None = None) -> str:
    """Describe the persisted deadline, not a guessed time from the user's text."""
    current = dt.datetime.fromtimestamp(time.time() if now is None else now, LOCAL_TZ)
    due = dt.datetime.fromtimestamp(timestamp, LOCAL_TZ)
    days = (due.date() - current.date()).days
    if days in (0, 1, 2):
        day = ("今天", "明天", "后天")[days]
    else:
        day = (f"{due.year}年" if due.year != current.year else "") + f"{due.month}月{due.day}日"
    hour = due.hour
    if hour < 6:
        period, display_hour = "凌晨", hour
    elif hour < 12:
        period, display_hour = "早上", hour
    elif hour == 12:
        period, display_hour = "中午", 12
    elif hour < 18:
        period, display_hour = "下午", hour - 12
    else:
        period, display_hour = "晚上", hour - 12
    clock = f"{display_hour}点"
    if due.minute == 30 and not due.second:
        clock += "半"
    elif due.minute:
        clock += f"{due.minute}分"
    if due.second:
        clock += f"{due.second}秒"
    return day + period + clock


class NotificationMixin:
    def _init_notifications(self) -> None:
        self._reminder_store = None
        self._reminder_task = None
        self._qq_voice = None
        self._voice_deferred_until = {}

    def _voice_engine(self) -> QQVoice:
        if self._qq_voice is None:
            self._qq_voice = QQVoice(self.cfg.get("voice", {}))
        return self._qq_voice

    def _notification_settings(self) -> dict:
        return self.cfg.get("notifications", {})

    def _notification_store(self) -> ReminderStore:
        if self._reminder_store is None:
            path = Path(self._notification_settings().get("database", "logs/reminders.sqlite3"))
            if not path.is_absolute():
                path = project_path(path)
            self._reminder_store = ReminderStore(path)
        return self._reminder_store

    def _notification_ready(self) -> bool:
        client = self.client
        ws = getattr(client, "ws", None)
        return bool(getattr(client, "_ready", False) and client.self_id
                    and ws is not None and not ws.closed)

    def _start_notifications(self) -> None:
        if (self._notification_settings().get("enabled", True)
                and not self.cfg["runtime"].get("dry_run")
                and self._reminder_task is None):
            self._reminder_task = asyncio.create_task(self._notification_loop())

    async def close(self) -> None:
        self._closing = True
        if self._reminder_task is not None:
            self._reminder_task.cancel()
            await asyncio.gather(self._reminder_task, return_exceptions=True)
            self._reminder_task = None
        # Wait for any command admitted immediately before shutdown to release DB.
        async with self._delivery_lock:
            if self._reminder_store is not None:
                self._reminder_store.close()
                self._reminder_store = None
        await super().close()

    def _notification_allowed(self, group_id: int, user_id: int, targets: tuple[int, ...]) -> bool:
        if not self._group_allowed(group_id) or user_id in self._user_blacklist:
            return False
        if self._is_muted(group_id, user_id):
            return False
        return all(type(t) is int and t > 0 and t != self.client.self_id and t not in self._user_blacklist
                   and not self._is_muted(group_id, t) for t in targets)

    def _notification_body_allowed(self, text: str) -> bool:
        return bool(text.strip() and len(text) <= 500 and not any(
            word and str(word) in text
            for key in ("blocked_input", "blocked_output")
            for word in (self.safety.get(key) or [])))

    def _notification_addressed(self, group_id: int, user_id: int, parsed) -> bool:
        if parsed.at_self or self._calls_bot(parsed.text):
            return True
        quote = self._msg_index.get((group_id, parsed.reply_id)) if parsed.reply_id else None
        if quote and quote.is_bot:
            return True
        state = self._dialogues.get(group_id)
        return bool(state and state.user_id == user_id and not state.interrupted
                    and time.monotonic() - state.updated_at <= float(self.beh.get("followup_window", 120)))

    def _reminder_confirmation(self, job, user_id: int) -> str:
        """Acknowledge only a saved job; keep IDs/maintenance details in the list."""
        status = getattr(job, "status", "pending")
        if status != "pending":
            return {
                "sent": "这条提醒之前已经叫过啦，别想让本肥鱼再喊一遍。",
                "submitted": "这条语音提醒已经提交啦，不会再重复发；有没有听到要看群消息哦。",
                "cancelled": "这条提醒已经取消啦，要重新约时间再告诉我。",
                "expired": "这条提醒已经错过时间啦，要不要重新约一个？",
                "uncertain": "这条提醒之前有没有发成功，我没拿到准信，先看看群消息，免得叫重了。",
                "sending": "这条提醒正在叫呢，等本肥鱼一下。",
            }.get(status, "这条提醒的状态需要核对一下，用“查看提醒”看看吧。")
        when = human_reminder_time(job.due)
        voice = "本肥鱼" if "鱼" in self.name else self.name
        if getattr(job, "delivery", "text") == "voice":
            if tuple(job.targets) == (user_id,):
                if job.body in ("起床", "起床啦", "起床哦", "起床吧"):
                    return f"好的呢，{when}{voice}用语音叫你起床，别赖床哦，小懒虫。"
                return f"记下啦，{when}{voice}会 @ 你，用语音提醒：{job.body}。"
            return f"好呀，{when}{voice}会 @ 指定的群友，用语音提醒「{job.body}」。"
        if tuple(job.targets) == (user_id,):
            if job.body in ("起床", "起床啦", "起床哦", "起床吧"):
                return f"好的呢，{when}{voice}叫你起床，别赖床哦，小懒虫。"
            return f"记下啦，{when}{voice}提醒你：{job.body}。"
        recipient = ("你和这几位群友" if user_id in job.targets else
                     "这位群友" if len(job.targets) == 1 else "这几位群友")
        return f"好呀，{when}{voice}会 @ {recipient}，提醒「{job.body}」。"

    def _notification_command_text(self, text: str) -> str:
        """Remove a known bot vocative from a parsing copy, never the stored words."""
        value = text.strip()
        aliases = {str(v).strip() for v in
                   (self.name, self.client.nickname, *(self.beh.get("keywords") or []))
                   if v and str(v).strip()}
        for alias in sorted(aliases, key=len, reverse=True):
            if value.casefold().startswith(alias.casefold()):
                return value[len(alias):].lstrip(" \t，,:：")
        return value

    async def _handle_notification_command(self, group_id: int, user_id: int, parsed,
                                           message_id: str = "") -> bool:
        if not self._notification_addressed(group_id, user_id, parsed):
            return False
        command_text = self._notification_command_text(parsed.text)
        voice_command = parse_voice_command(command_text)
        command = (None if voice_command else
                   parse_command(command_text, parsed.mentioned_ids, user_id, self.client.self_id))
        if voice_command is None and (command is None or not self._notification_settings().get("enabled", True)):
            return False
        targets = () if voice_command else command.targets
        # Interception happens before the ordinary worker, so repeat its filters.
        if self._closing or not self._notification_ready():
            return True
        if self._is_muted(group_id, user_id):
            if not parsed.at_self:
                return True
            self._unmute(group_id, user_id)
        if not self._notification_allowed(group_id, user_id, targets):
            return True
        if not self._notification_body_allowed(parsed.text):
            return True
        if self._apply_mute(group_id, user_id, parsed.text):
            return True
        if self.cfg["runtime"].get("dry_run"):
            log.info("[dry-run] 通知命令未保存、未发送")
            return True
        if voice_command is not None:
            try:
                if not self.cfg.get("voice", {}).get("enabled", True):
                    raise VoiceUnavailable("语音消息暂时没有开启。")
                if voice_command.action == "voices":
                    expected = self.client.self_id
                    voices = await self._voice_engine().characters(self.client, group_id, expected)
                    if self._closing or not self._notification_ready() or self.client.self_id != expected:
                        return True
                    lines = [f"{v.character_name}：{v.character_id}" for v in voices[:15]]
                    text = "这个群可用的语音音色：\n" + "\n".join(lines) if lines else "这个群暂时没有可用的语音音色。"
                    await self._notification_feedback(group_id, user_id, text)
                else:
                    status = await self._send_voice_notification(group_id, user_id, (), voice_command.text)
                    if status == "blocked":
                        await self._notification_feedback(group_id, user_id, "这段话暂时不能转成语音，请控制在200字以内并检查内容。")
                    elif status == "uncertain":
                        log.warning("语音提交结果不确定；不自动重发，请在群内确认")
            except VoiceUnavailable as exc:
                await self._notification_feedback(group_id, user_id, str(exc))
            return True
        if command.action == "error":
            await self._notification_feedback(group_id, user_id, command.error)
            return True
        if command.action in ("immediate", "schedule") and not self._notification_body_allowed(command.body):
            await self._notification_feedback(group_id, user_id, "正文为空、过长或包含不支持的内容，请换个说法。")
            return True
        try:
            if command.action == "immediate":
                status = await self._send_notification(group_id, user_id, command.targets, command.body)
                if status in ("blocked", "deferred"):
                    log.info("立即通知未发送，状态=%s", status)
                elif status == "uncertain":
                    log.warning("立即通知回执不确定；不自动重发，请在群内确认")
            elif command.action == "schedule":
                expected = self.client.self_id
                if command.delivery == "voice":
                    if not self.cfg.get("voice", {}).get("enabled", True):
                        raise VoiceUnavailable("语音消息暂时没有开启，语音提醒还没保存。")
                    if len(command.body) > self._voice_engine().max_chars:
                        raise VoiceUnavailable("语音提醒正文太长啦，请控制在200字以内；这次还没保存。")
                    await self._voice_engine().character(self.client, group_id, expected)
                if (self._closing or not self._notification_ready() or self.client.self_id != expected
                        or not self._notification_allowed(group_id, user_id, command.targets)):
                    return True
                store = self._notification_store()
                job = SimpleNamespace(**store.add(group_id, user_id, command.targets, command.body, command.due,
                                bot_id=expected, source_id=message_id, delivery=command.delivery))
                await self._notification_feedback(
                    group_id, user_id,
                    self._reminder_confirmation(job, user_id))
            elif command.action == "list":
                jobs = self._notification_store().list_owned(group_id, user_id, bot_id=self.client.self_id)
                lines = [f"{j.id}｜{format_time(j.due)}｜{j.status}｜{'语音' if j.delivery == 'voice' else '文字'}｜{j.body[:60]}" for j in map(lambda v: SimpleNamespace(**v), jobs)]
                await self._notification_feedback(group_id, user_id,
                                                  "你的提醒（北京时间）：\n" + "\n".join(lines)
                                                  if lines else "你在本群没有提醒记录。")
            elif command.action == "cancel":
                count = self._notification_store().cancel_owned(group_id, user_id, command.job_id,
                                                                 bot_id=self.client.self_id)
                await self._notification_feedback(group_id, user_id,
                                                  f"已取消 {count} 条提醒。" if count else "未找到你可取消的待发送提醒。")
        except VoiceUnavailable as exc:
            await self._notification_feedback(group_id, user_id, str(exc))
        except ValueError as exc:
            log.warning("通知命令未完成：%s", type(exc).__name__)
            await self._notification_feedback(group_id, user_id, "提醒未保存：" + str(exc)[:150])
        except Exception:
            log.exception("通知命令处理失败")
            await self._notification_feedback(group_id, user_id, "提醒操作失败，未能确认结果，请用“查看提醒”核对。")
        return True

    async def _notification_feedback(self, group_id: int, user_id: int, text: str) -> str:
        # Plain confirmations are distinct from reminders: never a bare @.
        return await self._send_notification(group_id, user_id, (), text[:1000], feedback=True)

    async def _send_notification(self, group_id: int, user_id: int, targets: tuple[int, ...],
                                 body: str, *, before_send: Callable[[], bool] | None = None,
                                 feedback: bool = False, expected_bot_id: int | None = None) -> str:
        if not body.strip() or len(body) > (1000 if feedback else 500):
            return "blocked"
        if len(targets) > 5 or len(set(targets)) != len(targets):
            return "blocked"
        if not self._notification_body_allowed(body) and not feedback:
            return "blocked"
        if self.cfg["runtime"].get("dry_run"):
            return "dry_run"
        expected_bot_id = self.client.self_id if expected_bot_id is None else expected_bot_id
        async with self._delivery_lock:
            await self._wait_delivery_gap()
            if (self._closing or not self._notification_ready()
                    or self.client.self_id != expected_bot_id):
                return "deferred"
            if not self._notification_allowed(group_id, user_id, targets):
                return "blocked"
            if not self.limiter.allow_many(self._second_reply_limits(group_id), consume=False):
                return "deferred"
            if before_send is not None and not before_send():
                return "blocked"
            self.limiter.allow_many(self._second_reply_limits(group_id))
            segments = [{"type": "at", "data": {"qq": str(target)}} for target in targets]
            segments.append({"type": "text", "data": {"text": (" " if targets else "") + body}})
            try:
                connection = self.client.ws
                data = await self.client.send_action("send_group_msg", {"group_id": group_id, "message": segments})
                new_id = data.get("message_id") if isinstance(data, dict) else None
                if (isinstance(new_id, str) and len(new_id) <= 32
                        and new_id.isascii() and new_id.isdecimal()):
                    new_id = int(new_id)
                if type(new_id) is not int or new_id <= 0:
                    return "uncertain"
                if self.client.self_id != expected_bot_id or self.client.ws is not connection:
                    # The original action has a positive receipt, but its owner
                    # is no longer current. Do not index it as the new account.
                    self._last_any = time.monotonic()
                    return "sent"
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("通知发送无确定回执；不自动重发")
                return "uncertain"
            self._last_any = time.monotonic()
            # Do not call _record_delivery: scheduled work must not resurrect old
            # dialogue/targets, supersede a live turn, or renew a friendly session.
            from dialogue import IndexedMessage
            name = self.client.nickname or self.name
            stamp = time.time()
            self.store.add(group_id, self._notification_message_type(name, body, True, stamp,
                sender_id=str(self.client.self_id), nickname=name, message_id=str(new_id),
                mentions=tuple(map(str, targets))))
            self._msg_index[(group_id, str(new_id))] = IndexedMessage(
                group_id, self.client.self_id, name, body, stamp, True)
            return "sent"

    def _voice_quota_available(self, group_id: int, cost: int) -> bool:
        # Reserve two DIFFERENT sends, not duplicate keys in allow_many.
        limits = [(key, cap - cost + 1) for key, cap in self._second_reply_limits(group_id)]
        return self.limiter.allow_many(limits, consume=False)

    def _voice_turn_current(self, group_id: int, turn, reason: str) -> bool:
        if turn is None:
            return True
        if self._active_turn.get(group_id) is not turn or self._superseded(turn):
            return False
        if reason in ("followup", "taunt") and not self._followup_is_current(turn):
            return False
        if reason in ("second", "continuation"):
            return (self._second_reply_ready(turn)
                    and self.store.trailing_bot_count(group_id) < int(self.beh["max_chain"]))
        return True

    def _record_voice_transport(self, group_id: int, body: str, new_id, targets=()) -> None:
        # Voice content belongs in context, but a scheduled job is NOT a chat turn.
        from dialogue import IndexedMessage
        name = self.client.nickname or self.name
        stamp = time.time()
        self.store.add(group_id, self._notification_message_type(name, body, True, stamp,
            sender_id=str(self.client.self_id), nickname=name,
            message_id=str(new_id) if new_id else "", mentions=tuple(map(str, targets))))
        if new_id:
            self._msg_index[(group_id, str(new_id))] = IndexedMessage(
                group_id, self.client.self_id, name, body, stamp, True)

    async def _send_voice_notification(self, group_id: int, user_id: int, targets: tuple[int, ...],
                                       body: str, *, before_send: Callable[[], bool] | None = None,
                                       expected_bot_id: int | None = None, turn=None,
                                       reason: str = "voice") -> str:
        if self.cfg["runtime"].get("dry_run"):
            return "dry_run"
        if (not self.cfg.get("voice", {}).get("enabled", True)
                or not self._notification_body_allowed(body)
                or len(body) > self._voice_engine().max_chars
                or len(targets) > 5 or len(set(targets)) != len(targets)
                or not self._notification_allowed(group_id, user_id, targets)):
            return "blocked"
        expected = self.client.self_id if expected_bot_id is None else expected_bot_id
        cost = 2 if targets else 1
        if self._closing or not self._notification_ready() or self.client.self_id != expected:
            return "deferred"
        if not self._voice_turn_current(group_id, turn, reason):
            return "blocked"
        if not self._voice_quota_available(group_id, cost):
            return "deferred"
        # Metadata lookup is read-only and does not hold up ordinary text sends.
        character = await self._voice_engine().character(self.client, group_id, expected)
        async with self._delivery_lock:
            await self._wait_delivery_gap()
            if self._closing or not self._notification_ready() or self.client.self_id != expected:
                return "deferred"
            if (not self._notification_allowed(group_id, user_id, targets)
                    or not self._voice_turn_current(group_id, turn, reason)):
                return "blocked"
            if not self._voice_quota_available(group_id, cost):
                return "deferred"
            if before_send is not None and not before_send():
                return "blocked"
            for _ in range(cost):
                self.limiter.allow_many(self._second_reply_limits(group_id))
            if targets:
                segments = [{"type": "at", "data": {"qq": str(target)}} for target in targets]
                segments.append({"type": "text", "data": {"text": " " + body}})
                try:
                    data = await self.client.send_action("send_group_msg", {"group_id": group_id, "message": segments})
                    new_id = data.get("message_id") if isinstance(data, dict) else None
                    if (isinstance(new_id, str) and len(new_id) <= 32
                            and new_id.isascii() and new_id.isdecimal()):
                        new_id = int(new_id)
                    if type(new_id) is not int or new_id <= 0:
                        return "uncertain"
                    if self.client.self_id != expected:
                        return "uncertain"
                    self._last_any = time.monotonic()
                    self._record_voice_transport(group_id, body, new_id, targets)
                    await self._wait_delivery_gap()
                    if (self._closing or not self._notification_ready() or self.client.self_id != expected
                            or not self._notification_allowed(group_id, user_id, targets)):
                        return "uncertain"
                except asyncio.CancelledError:
                    raise
                except Exception:
                    return "uncertain"
            try:
                result = await self._voice_engine().send(self.client, group_id, character, body)
            except asyncio.CancelledError:
                raise
            except VoiceUnavailable:
                # A claimed job must not repeat a potentially delivered @ message.
                return "uncertain"
            self._last_any = time.monotonic()
            # A reply can arrive after reconnection/account changes. Do not
            # attach an old account's receipt or old turn to the new dialogue.
            if self.client.self_id != expected:
                return "uncertain"
            if result.state in ("sent", "submitted"):
                if turn is not None and self._voice_turn_current(group_id, turn, reason):
                    self._record_delivery(group_id, user_id, body, result.message_id, reason)
                else:
                    self._record_voice_transport(group_id, body, result.message_id)
            log.info("语音群消息状态=%s（不代表对方已收听）", result.state)
            return result.state

    async def _notification_tick(self, now: float | None = None) -> None:
        if (self._closing or not self._notification_settings().get("enabled", True)
                or self.cfg["runtime"].get("dry_run") or not self._notification_ready()):
            return
        stamp = time.time() if now is None else now
        store = self._notification_store()
        grace = min(900, max(0, float(self._notification_settings().get("grace_seconds", 300))))
        started = time.monotonic()
        for row in store.due_candidates(stamp, bot_id=self.client.self_id, limit=10, grace_seconds=grace,
                                       include_voice=self.cfg.get("voice", {}).get("enabled", True)):
            job = SimpleNamespace(**row)
            if job.bot_id != self.client.self_id or not self._notification_ready():
                break
            if (getattr(job, "delivery", "text") == "voice"
                    and not self.cfg.get("voice", {}).get("enabled", True)):
                # Temporarily disabled: retain the job without claiming/cancelling.
                continue
            if stamp - job.due > grace:
                store.finish(job.id, "expired", bot_id=job.bot_id)
                continue
            if (not self._notification_allowed(job.group_id, job.requester_id, job.targets)
                    or not self._notification_body_allowed(job.body)):
                store.finish(job.id, "cancelled", bot_id=job.bot_id)
                continue
            # Claim only inside the delivery lock after readiness/rate checks.
            voice_job = getattr(job, "delivery", "text") == "voice"
            if voice_job and time.monotonic() < self._voice_deferred_until.get(job.id, 0):
                continue
            sender = self._send_voice_notification if voice_job else self._send_notification
            try:
                status = await sender(
                    job.group_id, job.requester_id, job.targets, job.body,
                    before_send=lambda j=job: store.claim(j.id, stamp + time.monotonic() - started, bot_id=j.bot_id, grace_seconds=grace),
                    expected_bot_id=job.bot_id)
            except VoiceUnavailable:
                self._voice_deferred_until[job.id] = time.monotonic() + 15
                log.warning("语音音色暂不可用，保留提醒，15秒后再检查（尚未发送）")
                continue
            if status in ("sent", "submitted", "uncertain"):
                store.finish(job.id, status, bot_id=job.bot_id)
                log.info("提醒 %s 状态=%s", job.id, status)
            elif status == "blocked" and voice_job:
                store.finish(job.id, "cancelled", bot_id=job.bot_id)
            elif status == "deferred":
                break

    async def _notification_loop(self) -> None:
        while not self._closing:
            try:
                await self._notification_tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("提醒调度异常，保留任务状态等待下轮检查")
            await asyncio.sleep(max(.2, min(60, float(self._notification_settings().get("poll_interval", 1)))))
