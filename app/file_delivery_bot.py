"""Source-bound delivery of newly generated plain-text group files.

The model cannot choose local paths, a recipient, an action name, or a QQ
account. Every upload reuses the current turn's permission/quota transaction.
No upload with an uncertain outcome is automatically retried.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import re
import time
import unicodedata

from file_delivery import GeneratedFileStore, FileRequest
from output_policy import limit_daily_text
from project_paths import project_path

log = logging.getLogger("dafish.files")
GENERATED_ROOT = project_path("generated_files")
FILE_QUOTA_WAIT_SECONDS = 61.0
FILE_QUOTA_POLL_SECONDS = 0.1
FILE_PENDING_SECONDS = 120.0
FILE_PENDING_MAX = 256
_pause = asyncio.sleep


def _source_signature(turn):
    return (turn.group_id, turn.user_id, str(turn.message_id or ""),
            getattr(turn, "arrival_revision", 0), getattr(turn, "target_revision", 0),
            tuple(getattr(turn, "source_ids", ()) or ()), str(turn.message.text))


def _positive_file_id(data) -> bool:
    if not isinstance(data, dict):
        return False
    value = data.get("file_id")
    return ((type(value) is int and value > 0)
            or (isinstance(value, str) and 0 < len(value) <= 512
                and bool(value.strip()) and value.strip() != "0"
                and all(c.isprintable() for c in value)))


class FileDeliveryMixin:
    def _file_pending_state(self) -> dict:
        pending = getattr(self, "_file_pending", None)
        if not isinstance(pending, dict):
            pending = {}
            self._file_pending = pending
        now = time.monotonic()
        for key, entry in list(pending.items()):
            if (not isinstance(entry, dict) or not isinstance(entry.get("request"), FileRequest)
                    or not isinstance(entry.get("stamp"), (int, float))
                    or now-entry["stamp"] > FILE_PENDING_SECONDS or now < entry["stamp"]):
                pending.pop(key, None)
        while len(pending) > FILE_PENDING_MAX:
            pending.pop(min(pending, key=lambda key: pending[key]["stamp"]), None)
        return pending

    def _clear_file_pending(self, turn) -> None:
        pending = getattr(self, "_file_pending", None)
        if isinstance(pending, dict):
            pending.pop((turn.group_id, turn.user_id), None)

    def _remember_file_pending(self, context: dict, request: FileRequest) -> None:
        if not self._file_context_current(context):
            return
        pending = self._file_pending_state()
        turn = context["turn"]
        key = (turn.group_id, turn.user_id)
        if key not in pending and len(pending) >= FILE_PENDING_MAX:
            pending.pop(min(pending, key=lambda item: pending[item]["stamp"]), None)
        # The revision is captured after the clarification was delivered, not
        # while its request was still waiting behind another member's turn.
        revision = getattr(self, "_arrival_revisions", {}).get(
            turn.group_id, getattr(turn, "arrival_revision", 0))
        question_id = ""
        rows = getattr(getattr(self, "store", None), "_data", {})
        recent = rows.get(turn.group_id, ()) if isinstance(rows, dict) else ()
        last = recent[-1] if recent else None
        if (last is not None and getattr(last, "is_bot", False) is True
                and str(getattr(last, "sender_id", "")) == str(context["bot_id"])
                and str(getattr(last, "reply_to_sender_id", "")) == str(turn.user_id)
                and str(getattr(last, "reply_to_id", "")) == str(turn.message_id or "")):
            value = getattr(last, "message_id", "")
            if (type(value) in (str, int) and str(value) not in {"", "0"}
                    and len(str(value)) <= 128 and all(c.isprintable() for c in str(value))):
                question_id = str(value)
        pending[key] = {"request": request, "source": context["source"],
                        "stamp": time.monotonic(), "bot_id": context["bot_id"],
                        "ws": context["ws"], "after_revision": revision,
                        "question_id": question_id}

    @staticmethod
    def _file_short_supplement(text: str):
        """Only implementation values, never an open-ended new instruction."""
        if not isinstance(text, str) or not text.strip() or len(text) > 100:
            return None
        value = text.strip()
        if (any(c in value for c in '\n\r\t"\'“”「」『』<>/\\:@')
                or any(unicodedata.category(c) in {"Cc", "Cf", "Cs"} for c in value)
                or re.search(r"取消|算了|不用|不要|别|停止|不写|不发|换个|换话题|先聊|他说|她说|别人|群友", value)):
            return None
        value = value.rstrip("。！! ")
        value = re.sub(r"^(?:就用|使用|用|选用|选择|选|改用|我要|要|是|保存为|保存成|后缀用|后缀是|格式用|格式是)\s*", "", value, count=1)
        value = re.sub(r"\s*(?:就行|即可|吧|就好|格式|文件)$", "", value, count=1).strip()
        if not value:
            return None
        if re.fullmatch(r"(?:python(?:\s*3(?:\.\d+){0,2})?|pytorch(?:\s*\d+(?:\.\d+){0,2})?|d2l|\.?py)", value, re.I):
            return value, "py", ""
        if re.fullmatch(r"(?:C(?:\s*语言|\s*(?:89|99|11|17|23))?|\.c)", value, re.I):
            return value, "c", ""
        if re.fullmatch(r"(?:\.?txt|纯文本|文本)", value, re.I):
            return value, "txt", ""
        if re.fullmatch(r"(?:C\+\+|C#|cpp|javascript|typescript|java|rust|go|sql|bash|shell)", value, re.I):
            return value, "txt", ""
        # A basename is a format/name clarification, never a local path.
        if (re.fullmatch(r"[\w\-\u3400-\u9fff][\w .\-\u3400-\u9fff]{0,75}\.(?:py|c|txt)", value, re.I)
                and unicodedata.normalize("NFKC", value) == value
                and not re.fullmatch(r"(?:con|prn|aux|nul|com[1-9]|lpt[1-9])", value.split(".", 1)[0], re.I)
                and len(value.encode("utf-8")) <= 240):
            return value, Path(value).suffix.lower().lstrip("."), value
        # These values cannot start unrelated writing/chat tasks; without a
        # language hint, the existing request retains responsibility to clarify.
        if re.fullmatch(r"v?\d+\.\d+(?:\.\d+){0,2}|STM32[A-Za-z0-9]{2,24}|ESP32[A-Za-z0-9-]{0,20}|[A-Za-z0-9]{1,12}51", value, re.I):
            return value, "", ""
        return None

    def _resume_file_request(self, turn) -> FileRequest | None:
        """Resume only a delivered, recent, same-member clarification.

        Inspection is idempotent because receiving and executing the same turn
        both query the router. The entry is cleared when generation starts.
        """
        pending = self._file_pending_state()
        key = (turn.group_id, turn.user_id)
        entry = pending.get(key)
        if entry is None:
            return None
        def reject():
            pending.pop(key, None)
            return None
        settings = self.cfg.get("file_delivery", {})
        if (not isinstance(settings, dict) or settings.get("enabled") is not True
                or self.cfg.get("runtime", {}).get("dry_run") or self._closing
                or getattr(turn, "is_poke", False)
                or not self.client._ready or self.client.self_id != entry["bot_id"]
                or self.client.ws is not entry["ws"] or getattr(self.client.ws, "closed", False)
                or not self._group_allowed(turn.group_id) or turn.user_id in self._user_blacklist
                or self._is_muted(turn.group_id, turn.user_id) or self._addressed_elsewhere(turn)):
            return reject()
        reply_id = str(getattr(turn, "reply_id", "") or "")
        if reply_id and (not entry.get("question_id") or reply_id != entry["question_id"]
                         or str(getattr(turn, "reply_sender_id", "")) != str(entry["bot_id"])):
            return reject()
        if any(str(member) != str(self.client.self_id) for member in getattr(turn, "mentioned_ids", ())):
            return reject()
        if any(word and str(word) in turn.message.text for word in (self.safety.get("blocked_input") or ())):
            return reject()
        routing_source = (turn.group_id, turn.user_id, str(turn.message_id or ""),
                          getattr(turn, "arrival_revision", 0), tuple(getattr(turn, "source_ids", ()) or ()),
                          str(turn.message.text))
        if entry.get("resume_source") == routing_source:
            return entry["resumed_request"]
        old_source = entry["source"]
        if str(turn.message_id or "") == old_source[2]:
            return None
        revision = getattr(turn, "arrival_revision", 0)
        after = entry.get("after_revision", 0)
        if type(revision) is int and type(after) is int and revision > 0 and after > 0 and revision != after+1:
            return reject()
        supplement = self._file_short_supplement(turn.message.text)
        if supplement is None:
            return reject()
        text, extension, filename = supplement
        original = entry["request"]
        extension = extension or original.extension
        filename = filename or original.filename
        if filename and extension and Path(filename).suffix.lower() != "."+extension:
            filename = str(Path(filename).with_suffix("."+extension))
        instruction = (original.instruction + "\n【当前成员对刚才文件任务的明确补充】" + text
                       + "。按此补充完成原文件任务，不是另起闲聊。")
        if len(instruction) > 12000:
            return reject()
        resumed = FileRequest(instruction, filename, extension)
        entry["resume_source"] = routing_source
        entry["resumed_request"] = resumed
        return resumed

    def _file_context_current(self, context: dict) -> bool:
        """Fail closed when any original sender/source/account condition changes."""
        try:
            turn = context["turn"]
            group_id, user_id = turn.group_id, turn.user_id
            settings = self.cfg.get("file_delivery", {})
            if (not isinstance(settings, dict) or settings.get("enabled") is not True
                    or self.cfg.get("runtime", {}).get("dry_run")):
                return False
            if (type(group_id) is not int or group_id <= 0 or type(user_id) is not int or user_id <= 0
                    or context["ws"] is None or type(context["bot_id"]) is not int or context["bot_id"] <= 0):
                return False
            if self._closing or getattr(turn, "is_poke", False):
                return False
            if self._active_turn.get(group_id) is not turn:
                return False
            if _source_signature(turn) != context["source"]:
                return False
            if not self.client._ready or self.client.self_id != context["bot_id"]:
                return False
            if self.client.ws is not context["ws"] or getattr(self.client.ws, "closed", False):
                return False
            if not self._group_allowed(group_id) or user_id in self._user_blacklist:
                return False
            if self._is_muted(group_id, user_id) or self._superseded(turn) or self._addressed_elsewhere(turn):
                return False
            if any(word and str(word) in turn.message.text
                   for word in (self.safety.get("blocked_input") or ())):
                return False
            owner = self._delivery_owners.get(group_id)
            if owner is not None and (owner[0] is not turn or owner[1] != context["bot_id"]
                                      or owner[2] is not context["ws"]):
                return False
            captured = getattr(self, "_turn_tones", {}).get(group_id)
            if captured == (user_id, "owner") and self._owner_chat_request(turn) is None:
                return False
            if captured == (user_id, "roast_pushback"):
                return False
            return True
        except (AttributeError, TypeError, ValueError, KeyError, IndexError):
            return False

    async def _file_feedback(self, context: dict, text: str) -> bool:
        if not text or not self._file_context_current(context):
            return False
        turn = context["turn"]
        # File receipts are short chat metadata, never a second code generation.
        text = limit_daily_text(text, 100)
        if not text:
            return False
        return bool(await self._send(turn.group_id, text, turn.message_id, turn.user_id, "file"))

    async def _wait_file_quota(self, context: dict, deadline: float) -> str:
        turn = context["turn"]
        limits = self._second_reply_limits(turn.group_id)
        while self._file_context_current(context):
            if self.limiter.allow_many(limits, consume=False):
                return "ready"
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                return "quota"
            await _pause(min(FILE_QUOTA_POLL_SECONDS, remaining))
        return "cancelled"

    def _record_file_ack(self, context: dict, name: str) -> None:
        """A confirmed upload consumes cooldown time even if its sender moved on."""
        turn = context["turn"]
        now = time.monotonic()
        self._last_any = self._last_awake = now
        self._last_group[turn.group_id] = now
        self._last_user[(turn.group_id, turn.user_id)] = now
        if self._file_context_current(context):
            # Metadata only: do not put long file contents in daily chat history,
            # and do not arm the separate idle/duel follow-up path.
            self._record_delivery(turn.group_id, turn.user_id,
                                  "[文件 " + name + "]", None, "file_upload")

    async def _upload_generated_file(self, context: dict, store, stored) -> str:
        deadline = time.monotonic()+FILE_QUOTA_WAIT_SECONDS
        turn = context["turn"]
        while self._file_context_current(context):
            ready = await self._wait_file_quota(context, deadline)
            if ready != "ready":
                return ready
            async with self._delivery_lock:
                if not self._file_context_current(context):
                    return "cancelled"
                await self._wait_delivery_gap()
                if not self._file_context_current(context):
                    return "cancelled"
                if not store.verify(stored):
                    return "invalid_file"
                if not self.limiter.allow_many(self._second_reply_limits(turn.group_id)):
                    if time.monotonic() >= deadline:
                        return "quota"
                    continue
                # Mark before awaiting: cancellation or a broken connection may
                # leave NapCat reading this file; its bytes must remain in place.
                context["possibly_pending"].add(stored.path)
                try:
                    data = await self.client.send_action("upload_group_file", {
                        "group_id": turn.group_id,
                        "file": str(Path(stored.path).absolute()),
                        "name": stored.name,
                    }, require_sync=True)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.warning("文件上传回执未确认 code=%s；不自动重传", type(exc).__name__)
                    return "uncertain"
                if not _positive_file_id(data):
                    log.warning("文件上传缺少确认编号；不自动重传")
                    return "uncertain"
                context["possibly_pending"].discard(stored.path)
                context["confirmed"].append(stored.name)
                self._record_file_ack(context, stored.name)
                return "sent" if self._file_context_current(context) else "cancelled"
        return "cancelled"

    async def _handle_file_request(self, turn, request, *, transcript: str = "",
                                   tone: str = "", focus_note: str = "") -> None:
        self._clear_file_pending(turn)
        settings = self.cfg.get("file_delivery", {})
        if not isinstance(settings, dict) or settings.get("enabled") is not True:
            return
        if self.cfg.get("runtime", {}).get("dry_run"):
            log.info("[dry-run] 文件任务未调用模型、未创建或上传文件")
            return
        connection = getattr(self.client, "ws", None)
        if connection is None or type(self.client.self_id) is not int or self.client.self_id <= 0:
            return
        context = {"turn": turn, "source": _source_signature(turn),
                   "bot_id": self.client.self_id, "ws": connection,
                   "possibly_pending": set(), "confirmed": []}
        if not self._file_context_current(context):
            return
        generate = getattr(self.brain, "generate_files", None)
        if not callable(generate):
            await self._file_feedback(context, "当前模型组件还不能生成文件，本次没有上传。")
            return
        store = None
        stored_files = ()
        try:
            result = await generate(request, transcript=transcript, tone=tone, focus_note=focus_note)
            if not self._file_context_current(context):
                return
            if result is None:
                await self._file_feedback(context, "这次没生成出有效文件，本次没有上传。")
                return
            if result.action == "wait":
                return
            if result.action == "clarify":
                text = result.text
                if not text or re.search(r"已(?:经)?(?:上传|发送|发出)|上传成功|发过去了|\b(?:uploaded|sent successfully)\b", text, re.I):
                    text = "要生成哪种内容或编程语言的文件？请再具体说一下。"
                if await self._file_feedback(context, text):
                    self._remember_file_pending(context, request)
                return
            if result.action != "files" or not result.files:
                await self._file_feedback(context, "没有拿到实际文件内容，本次没有上传。")
                return
            if not self._file_context_current(context):
                return
            store = GeneratedFileStore(GENERATED_ROOT)
            stored_files = tuple(store.create(result.files))
            if not stored_files:
                await self._file_feedback(context, "没有拿到实际文件内容，本次没有上传。")
                return
            status = "sent"
            for stored in stored_files:
                if not self._file_context_current(context):
                    status = "cancelled"
                    break
                status = await self._upload_generated_file(context, store, stored)
                if status != "sent":
                    break
            if not self._file_context_current(context):
                return
            confirmed = len(context["confirmed"])
            total = len(stored_files)
            if confirmed == total:
                names = "、".join(context["confirmed"])
                note = "代码未在本机执行验证。" if any(Path(item.name).suffix.lower() in {".py", ".c"} for item in stored_files) else ""
                await self._file_feedback(context, "已上传 " + names + "，可以在群文件里下载。" + note)
            elif confirmed:
                await self._file_feedback(context,
                    f"已确认上传{confirmed}/{total}个文件；其余未确认完成，没有自动重传，请先核对群文件。")
            elif status == "uncertain":
                await self._file_feedback(context, "文件上传回执不确定，未确认送达，也没有自动重传，请先核对群文件。")
            elif status == "quota":
                await self._file_feedback(context, "这次发送额度不足，文件尚未上传。")
            elif status == "invalid_file":
                await self._file_feedback(context, "文件校验未通过，本次没有上传。")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("文件任务停止 code=%s；未自动上传或重传", type(exc).__name__)
            if self._file_context_current(context):
                if context["confirmed"]:
                    await self._file_feedback(context, "部分文件已确认上传，其余未完成，请先核对群文件；没有自动重传。")
                elif context["possibly_pending"]:
                    await self._file_feedback(context, "文件上传结果未确认，请先核对群文件；没有自动重传。")
                else:
                    await self._file_feedback(context, "文件生成或校验失败，本次没有上传。")
        finally:
            if store is not None and stored_files:
                # Only known completed or never-attempted files may be removed.
                # An unknown API outcome is not proof that NapCat stopped reading.
                cleanup = tuple(item for item in stored_files
                                if item.path not in context["possibly_pending"])
                if cleanup:
                    try:
                        store.cleanup(cleanup)
                    except Exception as exc:
                        log.warning("生成文件清理未完成 code=%s", type(exc).__name__)
