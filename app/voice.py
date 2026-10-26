"""Explicit QQ AI voice messages through the existing OneBot connection.

No credentials, media downloads, preview playback, calling, or automatic retries.
NapCat's get_ai_record uses the SAME send-capable operation as
send_group_ai_record; never fetch a record and then resend it.
"""
from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class VoiceCommand:
    action: Literal["speak", "voices"]
    text: str = ""


@dataclass(frozen=True)
class VoiceCharacter:
    character_id: str
    character_name: str
    type: str = ""


@dataclass(frozen=True)
class VoiceResult:
    state: Literal["sent", "submitted", "uncertain"]
    message_id: int | None = None


class VoiceUnavailable(RuntimeError):
    """The operation was rejected before a voice send could be attempted."""


_POLITE = r"(?:(?:请|麻烦你?|劳烦你?)\s*)?(?:你\s*)?"
_SPEAK = re.compile(
    r"^" + _POLITE + r"(?:用语音(?:说|念|读)(?:一下)?\s*[：:]\s*(?P<say>.+)"
    r"|发(?:一)?条语音\s*[：:]\s*(?P<send>.+))$",
    re.DOTALL,
)
_VOICES = re.compile(r"^" + _POLITE + r"(?:查看|列出|看看)语音音色[。！!？?]*$")
_VOICE_REPLY = re.compile(
    r"^" + _POLITE + r"用语音(?:回复|回答)(?:一下)?(?:我)?(?:一下)?(?P<tail>.*)$",
    re.DOTALL,
)
_QUESTION_OPENER = re.compile(r"^(?:什么|怎么|为什么|为何|如何|哪|谁|几|多少|是否|有没有|是不是|能否|能不能)")
# A trailing correction/report is not consent. This deliberately errs toward
# text rather than interpreting a discussion, hypothetical, or retraction as a
# fresh request to publish audio. Speech payloads themselves are never rewritten.
_NON_REQUEST_CLAUSE = re.compile(
    r"(?:^|[，,。；;！？!?\n])\s*(?:(?:但是?|不过|其实|还是|现在|这次)\s*)?"
    r"(?:不(?:要|用|必|需要|想)|别|先别|算了|取消|撤回|改(?:用|成)文字|"
    r"只(?:是|想)(?:问|说|举|讨论)|我(?:只是|说的是|在说|没让你|不需要|不想)|"
    r"这(?:句|段|条).{0,16}(?:原话|意思|举例|例子|别人说|他说|她说)|"
    r"这(?:是|个).{0,16}(?:原话|说的|例子|举例|引用)|(?:是|什么)意思|"
    r"如果|假如|假设|要是|倘若|(?:他|她|别人|有人)(?:说|让|提到|问)|"
    r"据说|例如|比如|举个例子|开玩笑)"
)
_RETRACT_SEND = re.compile(
    r"(?:[，,。；;]\s*|^)(?:算了[，,]?\s*)?(?:先|还是|这次|现在)?"
    r"(?:别(?:真的?|真)?发(?:送)?(?:语音|消息|了|吧)?|"
    r"不(?:要|用)(?:真的?|真|实际)?发(?:送)?(?:语音|消息|了|吧)?|"
    r"取消(?:发送|语音)?|撤回(?:命令|语音)?|只是(?:举例|例子|测试|讨论)|"
    r"不是(?:命令|让你|要你)|开玩笑的)"
)
_REPORTED_PAYLOAD = re.compile(
    r"(?:^|[，,。；;]\s*)(?:这(?:是|句(?:话)?|段(?:话)?|条(?:消息)?).{0,20}"
    r"(?:原话|说的|意思|举例|例子|引用)|(?:是)?(?:他|她|别人|有人|朋友).{0,12}"
    r"(?:说的|原话)|(?:是什么|什么)意思)"
)


def _balanced_quotes(value: str) -> bool:
    # An unmatched closing quote commonly ends a quoted/reported command.
    return (all(value.count(left) == value.count(right)
                for left, right in (("“", "”"), ("「", "」"), ("『", "』")))
            and value.count('"') % 2 == 0)


def parse_voice_command(text: str) -> VoiceCommand | None:
    """Only recognize direct, literal commands, not discussion about voice."""
    if not isinstance(text, str):
        return None
    value = text.strip()
    if _VOICES.fullmatch(value):
        return VoiceCommand("voices")
    match = _SPEAK.fullmatch(value)
    if match:
        body = (match.group("say") or match.group("send") or "").strip()
        if (body and body.strip(" \t\r\n：:")
                and _balanced_quotes(body) and not _RETRACT_SEND.search(body)
                and not _REPORTED_PAYLOAD.search(body)):
            return VoiceCommand("speak", body)
    return None


def wants_voice_reply(text: str) -> bool:
    """Opt in for this reply only; a mention of calls/recordings is not consent."""
    if not isinstance(text, str):
        return False
    match = _VOICE_REPLY.fullmatch(text.strip())
    if not match:
        return False
    tail = match.group("tail")
    if not tail or not tail.strip(" \t\r\n。！!？?"):
        return True
    separated = tail[0] in " \t\r\n，,:：。；;"
    body = tail.lstrip(" \t\r\n，,:：。；;") if separated else tail
    if (not body or not _balanced_quotes(body) or _NON_REQUEST_CLAUSE.search(body)
            or _RETRACT_SEND.search(body) or _REPORTED_PAYLOAD.search(body)):
        return False
    # Without a separator, only an unmistakable question opener can follow the
    # directive: 回答什么 is allowed, 回复不了吗 / 回复我不要 are not.
    return separated or bool(_QUESTION_OPENER.match(body))


def _positive_number(value: Any, default: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) and 0 < number <= maximum else default


def _group(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise VoiceUnavailable("群聊信息不完整，暂时不能发语音。")
    group = str(value).strip()
    if len(group) > 32 or not group.isascii() or not group.isdecimal() or int(group) <= 0:
        raise VoiceUnavailable("群聊信息不完整，暂时不能发语音。")
    return group


class QQVoice:
    """Discover real group voices on demand and submit at most one native send."""

    def __init__(self, settings: dict | None = None) -> None:
        self.settings = dict(settings or {})
        self.max_chars = max(1, min(200, int(_positive_number(self.settings.get("max_chars", 200), 200, 1000))))
        self.list_timeout = _positive_number(self.settings.get("list_timeout", 10), 10, 60)
        self.send_timeout = _positive_number(self.settings.get("send_timeout", 30), 30, 120)
        self.cache_seconds = _positive_number(self.settings.get("cache_seconds", 300), 300, 3600)
        self._cache: dict[tuple[str, str], tuple[float, tuple[VoiceCharacter, ...]]] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    def _enabled(self) -> None:
        if not self.settings.get("enabled", True):
            raise VoiceUnavailable("语音消息暂时没有开启。")

    async def characters(self, client: Any, group_id: int, bot_id: int) -> tuple[VoiceCharacter, ...]:
        """Return only character metadata; discard private preview URLs entirely."""
        self._enabled()
        group = _group(group_id)
        key = (str(bot_id), group)
        cached = self._cache.get(key)
        if cached and time.monotonic() < cached[0]:
            return cached[1]
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._cache.get(key)
            if cached and time.monotonic() < cached[0]:
                return cached[1]
            try:
                raw = await asyncio.wait_for(
                    client.send_action("get_ai_characters", {"group_id": group, "chat_type": 1},
                                       timeout=self.list_timeout),
                    timeout=self.list_timeout,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise VoiceUnavailable("这个群的语音音色暂时取不到，请稍后再试。") from exc
            if not isinstance(raw, list):
                raise VoiceUnavailable("这个群没有返回可用的语音音色。")
            voices: list[VoiceCharacter] = []
            seen: set[str] = set()
            for category in raw:
                if not isinstance(category, dict) or not isinstance(category.get("characters"), list):
                    continue
                kind = category.get("type")
                kind = kind.strip()[:80] if isinstance(kind, str) else ""
                for entry in category["characters"]:
                    if not isinstance(entry, dict):
                        continue
                    identity = entry.get("character_id")
                    if not isinstance(identity, str):
                        continue
                    identity = identity.strip()
                    if not identity or len(identity) > 200 or any(ord(c) < 32 for c in identity):
                        continue
                    if identity in seen:
                        continue
                    name = entry.get("character_name")
                    name = name.strip()[:80] if isinstance(name, str) else ""
                    voices.append(VoiceCharacter(identity, name or identity, kind))
                    seen.add(identity)
            result = tuple(voices)
            self._cache[key] = (time.monotonic() + self.cache_seconds, result)
            return result

    async def character(self, client: Any, group_id: int, bot_id: int) -> str:
        voices = await self.characters(client, group_id, bot_id)
        if not voices:
            raise VoiceUnavailable("这个群暂时没有可用的语音音色。")
        configured = self.settings.get("character_id", "auto")
        if configured is None or configured == "" or configured == "auto":
            return voices[0].character_id
        if isinstance(configured, str):
            configured = configured.strip()
            if configured in {voice.character_id for voice in voices}:
                return configured
        raise VoiceUnavailable("设定的语音音色不在这个群的可用列表里，请先查看语音音色。")

    async def send(self, client: Any, group_id: int, character: str, text: str) -> VoiceResult:
        self._enabled()
        group = _group(group_id)
        if not isinstance(character, str) or not character.strip() or len(character) > 200:
            raise VoiceUnavailable("没有可用的语音音色。")
        if not isinstance(text, str) or not text.strip():
            raise VoiceUnavailable("要说的话不能为空。")
        body = text.strip()
        if len(body) > self.max_chars:
            raise VoiceUnavailable(f"这段话太长啦，语音请控制在{self.max_chars}字以内。")
        # An already disconnected transport is known to precede any send.
        if hasattr(client, "ws"):
            ws = client.ws
            if ws is None or getattr(ws, "closed", False):
                raise VoiceUnavailable("机器人还没连上，暂时不能发语音。")
        try:
            # Do not call get_ai_record: it can already send the voice message.
            data = await asyncio.wait_for(
                client.send_action("send_group_ai_record", {
                    "group_id": group, "character": character.strip(), "text": body,
                }, timeout=self.send_timeout),
                timeout=self.send_timeout,
            )
        except asyncio.CancelledError:
            # The caller must persist an in-flight operation as uncertain.
            raise
        except Exception:
            # Even a timeout or a negative reply can follow a completed QQ send.
            # Never retry or fall back to another send after this boundary.
            return VoiceResult("uncertain")
        identity = data.get("message_id") if isinstance(data, dict) else None
        if (isinstance(identity, str) and len(identity) <= 32
                and identity.isascii() and identity.isdecimal()):
            identity = int(identity)
        if type(identity) is not int or identity < 0:
            return VoiceResult("uncertain")
        if identity == 0:
            # Current NapCat acknowledges submission with an intentional zero.
            # It is not a quotable message ID and must never enter the ID index.
            return VoiceResult("submitted")
        return VoiceResult("sent", identity)
