"""Explicit QQ mention commands and restart-safe, account-scoped reminder storage.

This module never calls a model or sends a message. Recipient identity comes only
from OneBot metadata (or the sender's explicit ``我``), never a nickname in text.
Times use China Standard Time (UTC+08:00); no external timezone package is needed.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, replace
from typing import Iterable

LOCAL_TZ = dt.timezone(dt.timedelta(hours=8))
MAX_TARGETS = 5
MAX_BODY = 300
MAX_HORIZON = 30 * 86400


@dataclass(frozen=True)
class Command:
    action: str
    targets: tuple[int, ...] = ()
    body: str = ""
    due: float = 0.0
    job_id: str = ""
    error: str = ""
    delivery: str = "text"


def format_time(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp, LOCAL_TZ).strftime("%Y-%m-%d %H:%M")


def _error(message: str) -> Command:
    return Command("error", error=message)


def _number(value: str) -> int:
    if value.isascii() and value.isdigit():
        return int(value)
    chars = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3,
             "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value in chars:
        return chars[value]
    if value == "十":
        return 10
    if "十" in value and value.count("十") == 1:
        a, b = value.split("十")
        if (not a or a in chars) and (not b or b in chars):
            return (chars[a] if a else 1) * 10 + (chars[b] if b else 0)
    raise ValueError("时间数字无法识别")


_NUM = r"(?:\d{1,5}|[零〇一二两三四五六七八九十]{1,3})"
_RELATIVE = re.compile(rf"(?P<n>{_NUM})\s*(?P<unit>秒钟?|分钟|小时|天)\s*(?:之后|以后|后)")
_CLOCK = re.compile(
    rf"(?P<day>今天|明天|后天)?\s*(?P<period>凌晨|早上|早晨|上午|中午|下午|晚上|夜里)?\s*"
    rf"(?P<h>{_NUM})\s*(?:(?P<colon>[:：])\s*(?P<cm>\d{{1,2}})|点(?:(?P<half>半)|\s*(?P<m>{_NUM})\s*分?)?)")
_INTENT = re.compile(r"艾特|(?<!\w)at(?=\s|我|他|她|$)|提醒|通知|叫一下|喊一下|@(?:我|他|她|\d)", re.I)


def _parse_time(prefix: str, now: float) -> tuple[float, tuple[int, int]] | None:
    """Only inspect text before the first command verb, never reminder body."""
    if re.search(r"每天|每日|每周|每隔|每小时|循环|重复|周[一二三四五六日天]|星期|下周|月|年|号|日", prefix):
        raise ValueError("暂只支持一次性提醒，请写“明天早上8点”或“10分钟后”。")
    relatives = list(_RELATIVE.finditer(prefix))
    clocks = list(_CLOCK.finditer(prefix))
    if len(relatives) + len(clocks) > 1:
        raise ValueError("一次请指定一个明确时间。")
    if relatives:
        match = relatives[0]
        amount = _number(match["n"])
        scale = {"秒": 1, "秒钟": 1, "分钟": 60, "小时": 3600, "天": 86400}[match["unit"]]
        seconds = amount * scale
        if seconds <= 0 or seconds > MAX_HORIZON:
            raise ValueError("提醒时间须在现在之后、30天以内。")
        return now + seconds, match.span()
    if not clocks:
        if re.search(r"今天|明天|后天|点|[:：]\d|分钟|小时|秒|时候|届时|到时", prefix):
            raise ValueError("时间没有写完整，请写“明天早上8点”或“10分钟后”。")
        return None
    match = clocks[0]
    hour = _number(match["h"])
    minute = 30 if match["half"] else _number(match["cm"] or match["m"] or "0")
    period = match["period"] or ""
    if minute not in range(60):
        raise ValueError("分钟须在0到59之间。")
    if period:
        if hour not in range(1, 13) and not (hour == 0 and period in ("凌晨", "夜里")):
            raise ValueError("带早上/下午等时段时，请使用0到12点的时间。")
        if hour == 12 and period != "中午":
            raise ValueError("“12点”的日期/时段可能有歧义，请写明确日期及24小时制00:00或12:00。")
        if period in ("下午", "晚上", "夜里") and 0 < hour < 12:
            hour += 12
        elif period == "中午":
            if hour in (1, 2):
                hour += 12
            elif hour not in (11, 12):
                raise ValueError("请把中午的时间写成24小时制，例如12:30。")
    elif not match["colon"] and hour < 13:
        raise ValueError("“几点”可能有早晚歧义，请写“早上8点”或24小时制“08:00”。")
    if hour not in range(24):
        raise ValueError("小时须在0到23之间。")
    current = dt.datetime.fromtimestamp(now, LOCAL_TZ)
    day = match["day"] or ""
    offset = {"": 0, "今天": 0, "明天": 1, "后天": 2}[day]
    date = current.date() + dt.timedelta(days=offset)
    result = dt.datetime.combine(date, dt.time(hour, minute), LOCAL_TZ).timestamp()
    if result <= now:
        if day:
            raise ValueError("指定时间已经过去，请重新指定未来时间。")
        result += 86400  # Unqualified clock means its next occurrence.
    return result, match.span()


def _clean_body(value: str) -> str:
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)
    value = re.sub(r"\s+", " ", value).strip(" ,，。.!！?？:：;；、")
    return value


def _parse_explicit_command(text: str, mentioned_ids: Iterable[str | int], requester_id: int,
                  self_id: int, now: float | None = None) -> Command | None:
    """Parse an explicit command. Caller must separately require bot addressing.

    OneBot's at segments have normally been removed from ``text``. Merely having
    mention metadata is not a command. Bare names and numeric QQ strings are not
    accepted as recipient identity. Unsupported/ambiguous commands return error,
    so callers can explain instead of making a false scheduling promise.
    """
    text = str(text or "").strip()
    now = time.time() if now is None else float(now)
    normalized = re.sub(r"[\s，,。！!？?：:]", "", text)
    if normalized in ("查看提醒", "我的提醒", "列出提醒", "查看我的提醒", "提醒列表"):
        return Command("list")
    if normalized in ("取消全部提醒", "取消所有提醒", "取消我的全部提醒", "取消我的所有提醒"):
        return Command("cancel", job_id="all")
    cancel = re.fullmatch(r"(?:请|帮我)?取消(?:我的)?提醒\s*[:：]?\s*(全部|所有|all|[a-fA-F0-9]{8,32})[。！! ]*", text, re.I)
    if cancel:
        value = cancel[1].lower()
        return Command("cancel", job_id="all" if value in ("全部", "所有", "all") else value)
    if re.match(r"(?:请|帮我)?取消(?:我的)?提醒", text):
        return _error("请写“取消提醒 编号”或“取消提醒 全部”，编号可用“查看提醒”获取。")
    intent = _INTENT.search(text)
    if not intent:
        return None
    # First establish a request-shaped header. Merely discussing 艾特/提醒
    # must not enter target validation, time errors or any outbox operation.
    prefix = text[:intent.start()]
    if not _notification_request_header(prefix):
        return None
    question_head = re.split(r"[，,;；]|(?:说|告诉)[：:]", text, maxsplit=1)[0]
    meta_tail = question_head[intent.end():].lstrip()
    if (re.search(r"什么意思|是什么意思|是什么|指什么|怎么用|如何使用|怎么触发|"
                  r"会怎样|会怎么样|会发生什么|是不是|是否|(?:有什么|有啥|什么)区别|区别(?:是什么|在哪)", question_head)
            or re.search(r"(?:吗|么|呢|不)[?？]?$|[?？]$", question_head)
            or re.match(r"(?:的)?(?:用法|作用|机制|规则|功能)", meta_tail)):
        return None
    head = text[:intent.end()]
    if re.search(r"如果|假如|假设|要是|比如|例如|据说|(?:他|她|别人|群友|有人)说|[“\"「『]", head):
        return None  # A quoted or hypothetical command is ordinary chat, not consent.
    if re.search(r"不要|不用|别|停止|取消", head):
        return _error("没有新增提醒。已有任务请用“查看提醒”及“取消提醒 编号”管理。")
    # Capability questions are about the command, not a question inside its body.
    # For example the body '你能准时到吗' must survive unchanged.
    question_head = re.split(r"[，,;；]|(?:说|告诉)[：:]", text, maxsplit=1)[0]
    capability = re.search(r"可以|能|会|支持", text[:intent.start()])
    if re.search(r"能不能|能否|会不会", head) or (capability and re.search(r"(?:吗|么)[?？]?$", question_head)):
        return None  # A capability question belongs to normal conversation.
    # All recipients are metadata, never values named in arbitrary chat text.
    ids: list[int] = []
    for value in mentioned_ids:
        raw = str(value)
        if raw.lower() == "all":
            return _error("不支持@全体成员，请明确选择最多5位群友。")
        if not re.fullmatch(r"[1-9]\d{0,19}", raw):
            return _error("无法确认@对象，请使用QQ真正的@成员功能。")
        number = int(raw)
        if number != int(self_id) and number not in ids:
            ids.append(number)
    # '帮我' is a request prefix, not an at-self recipient declaration.
    tail = text[intent.end():]
    at_me = bool(re.match(r"\s*我(?!们|的|朋友|同事|老板|家人|爸|妈|哥|姐)", tail)) or text[intent.start():].startswith("@我")
    if at_me and int(requester_id) != int(self_id) and int(requester_id) not in ids:
        ids.insert(0, int(requester_id))
    if not ids:
        return _error("请说“艾特我”，或用QQ真正的@选择要通知的群友；不能仅凭昵称猜对象。")
    if len(ids) > MAX_TARGETS:
        return _error("一次最多@5位群友。")
    prefix = text[:intent.start()]
    try:
        timing = _parse_time(prefix, now)
    except (ValueError, OverflowError) as exc:
        return _error(str(exc))
    # '提醒我...' without a time is not an immediate-at request.
    if not timing and intent[0] == "提醒" and not re.search(r"现在|立即|马上", prefix):
        return _error("请明确提醒时间，例如“10分钟后提醒我喝水”。立即通知可说“现在艾特我，说：喝水”。")
    if not timing and re.search(r"定时|闹钟|到时|届时|等会|待会|一会", text):
        return _error("请明确提醒时间，例如“10分钟后艾特我”或“明天早上8点艾特我”。")
    # Require an imperative request prefix. This intentionally does not parse
    # every sentence containing 提醒/艾特: '他说10分钟后提醒我' is not an order.
    command_prefix = prefix
    if timing:
        a, b = timing[1]
        command_prefix = prefix[:a] + prefix[b:]
    command_prefix = re.sub(r"[\s，,。！!？?：:、]", "", command_prefix)
    command_prefix = re.sub(r"(?:我想让你|我要你|请|帮我|麻烦你?|拜托你?|你|给我|现在|立即|马上|那|就|在|到|于|等到|的时候|时|一下)+", "", command_prefix)
    if command_prefix:
        return None
    body = tail
    body = re.sub(r"^(?:\s*(?:我|他|她|他们|她们|我们|一下|一声|一次|一条|的时候|时|吧|哦|呀|啊|[，,。:：;；、]))+", "", body)
    body = re.sub(r"^(?:(?:并且|然后|并|顺便|再|要|帮我)?\s*(?:说|告诉(?:我|他|她|他们|她们)?|提醒(?:我|他|她|他们|她们)?|附带|附上|内容(?:是|为)?)\s*[:：，,]?\s*)+", "", body)
    body = _clean_body(body)
    if len(body) > MAX_BODY:
        return _error(f"附带正文请控制在{MAX_BODY}字以内。")
    body = body or ("到时间啦。" if timing else "找你一下，来聊聊吧。")
    return Command("schedule" if timing else "immediate", tuple(ids), body,
                   timing[0] if timing else 0.0)


# Natural requests share the same validated outbox protocol as explicit commands.
# The verb is only one slot: a request header, time and real recipient are required.
_NATURAL_ACTION = re.compile(r"艾特|叫醒|喊醒|唤醒|提醒|通知|催|叫|喊")
_ALIAS_CLOCK = re.compile(rf"(?:明早|明晚|今晚|今早)\s*{_NUM}\s*(?:[:：]\s*\d{{1,2}}|点(?:半|\s*{_NUM}\s*分?)?)")
_HALF_DELAY = re.compile(r"半(?:个)?小时(?:之后|以后|后)|一刻钟(?:之后|以后|后)")
_REQUEST_WORDS = re.compile(r"(?:(?:别|不要)忘(?:了|记(?:了)?)|准时|过来|我想(?:请|让)你|我要你|能不能|可不可以|能否|记得|记住|一定要|一定|请|帮我|麻烦你?|拜托你?|给我|你|可以|能|在|等到|到|于|那|就|现在|立即|马上|的时候|时候|时|来|再|一下|给)+")


def _natural_clock_slots(value: str) -> list[tuple[int, int, str]]:
    found = [(m.start(), m.end(), m.group()) for m in _ALIAS_CLOCK.finditer(value)]
    found += [(m.start(), m.end(), m.group()) for m in _HALF_DELAY.finditer(value)]
    for expression in (_RELATIVE, _CLOCK):
        for m in expression.finditer(value):
            if not any(m.start() < b and m.end() > a for a, b, _ in found):
                found.append((m.start(), m.end(), m.group()))
    return sorted(found)


def _natural_time_text(value: str) -> str:
    for old, new in (("明早", "明天早上"), ("明晚", "明天晚上"),
                     ("今晚", "今天晚上"), ("今早", "今天早上")):
        if value.startswith(old):
            value = new + value[len(old):]
            break
    return re.sub(r"半(?:个)?小时", "30分钟", value).replace("一刻钟", "15分钟")


def _request_prefix(value: str) -> bool:
    value = re.sub(r"[\s，,。！!？?：:、]", "", value)
    return not _REQUEST_WORDS.sub("", value)


def _notification_request_header(prefix: str) -> bool:
    """Admit a request header before interpreting its validation errors."""
    value = prefix
    for start, end, _ in reversed(_natural_clock_slots(prefix)):
        value = value[:start] + value[end:]
    value = re.sub(
        r"每天|每日|每周|每隔|每小时|循环|重复|星期[一二三四五六日天]?|下周|"
        r"明早|明晚|今晚|今早|今天|明天|后天|早上|早晨|上午|中午|下午|晚上|夜里|凌晨|"
        r"等会|待会|一会|到时|届时|定时", "", value)
    value = re.sub(r"(?:别|不要)忘(?:了|记(?:了)?)", "记得", value)
    value = re.sub(r"不要|不用|别|不想|不需要|无需|停止|取消", "", value)
    return _request_prefix(value)


def _parse_natural_request(text: str, mentioned_ids: tuple[str | int, ...],
                           requester_id: int, self_id: int, now: float) -> Command | None:
    action = _NATURAL_ACTION.search(text)
    alarm = re.search(r"(?:设置|设|定|订)(?:一个|个|一下)?", text)
    is_alarm = bool(alarm and "闹钟" in text[alarm.end():])
    if is_alarm and (action is None or alarm.start() < action.start()):
        action = alarm
    else:
        is_alarm = False
    if action is None:
        return None
    prefix, tail = text[:action.start()], text[action.end():]
    if not _notification_request_header(prefix):
        return None
    head = text[:action.end()]
    if re.search(r"如果|假如|假设|要是|比如|例如|据说|(?:他|她|别人|群友|有人)说|[“\"「『]", head):
        return None
    negative_head = re.sub(r"(?:别|不要)忘(?:了|记(?:了)?)", "记得", head)
    if re.search(r"不要|不用|别|不想|不需要|无需|停止|取消", negative_head):
        return _error("没有新增提醒。已有任务可以用“查看提醒”或“取消提醒 编号”管理。")
    if re.search(r"每天|每日|每周|每隔|循环|重复|星期|下周", prefix + (tail if is_alarm else "")):
        return _error("暂只支持一次性提醒，请指定一个明确时间。")
    target = "我" if is_alarm else ""
    if not is_alarm:
        who = re.match(r"\s*(?:一下|一声)?\s*(我|他们|她们|他|她)(?!们|的|朋友|同事|老板|家人|爸|妈|哥|姐)", tail)
        if who:
            target, tail = who[1], tail[who.end():]
        elif not any(str(n) != str(self_id) for n in mentioned_ids):
            return None
    slots = _natural_clock_slots(prefix)
    if len(slots) > 1:
        return _error("一次请指定一个明确时间。")
    if slots:
        a, b, when = slots[0]
        request = prefix[:a] + prefix[b:]
    else:
        # Only a leading time after the recipient is a deadline. A time inside
        # the reminder body is content, not a reason to schedule an extra job.
        after = _natural_clock_slots(tail)
        if not after:
            if _request_prefix(prefix) and re.search(r"定时|到时|届时|待会|等会|一会|明早|明晚|今晚|今早|明天|后天", prefix + tail):
                return _error("想让我什么时候叫你？说清具体时间，我才好记下来。")
            return None
        a, b, when = after[0]
        before_time = tail[:a]
        if not _request_prefix(before_time.replace("一个", "").replace("个", "")):
            return None
        request, tail = prefix, tail[b:]
    if not _request_prefix(request):
        return None
    if re.search(r"每天|每日|每周|每隔|循环|重复|星期|下周", request):
        return _error("暂只支持一次性提醒，请指定一个明确时间。")
    if is_alarm:
        tail = re.sub(r"^\s*(?:的)?\s*闹钟\s*", "", tail)
        tail = re.sub(r"^\s*(?:一个|个|一下)?\s*(?:的)?\s*闹钟\s*", "", tail)
    else:
        tail = re.sub(r"^\s*(?:的时候|时候|时|再|来|一下|一声)+", "", tail)
    explicit_body = bool(re.match(r"\s*[，,：:]?\s*(?:说|告诉)", tail))
    if not explicit_body:
        if re.search(r"[，,；;。.!！?？]\s*(?:算了(?:\s*(?:不用了|不需要了))?|不用了|不需要了|(?:不要|别)真的(?:设置|安排|提醒|叫|喊)|(?:不要|别)(?:设置|安排)(?:了|提醒|闹钟))", tail):
            return _error("这条提醒先不安排啦。已有的提醒可以用“查看提醒”核对。")
        if re.search(r"是什么意思|什么意思|这句话|只是举例|开玩笑|不是要你", tail):
            return Command("ignore")
        tail = re.split(r"[，,；;。.!！?？]\s*(?:因为|不然|否则|免得|我(?:要|得|想|还|怕))", tail, maxsplit=1)[0]
        tail = re.sub(r"^(\s*起床(?:啦|哦|吧)?)(?=我(?:要|得|想|还|怕))[\s\S]*$", r"\1", tail)
        tail = re.sub(r"(?:好不好|好么|好吗|行吗|可以吗|能不能)[？?。！! ]*$", "", tail)
        if re.search(r"能不能|可不可以|能否|可以|能", request):
            tail = re.sub(r"吗[？?。！! ]*$", "", tail)
    body = _clean_body(tail)
    if action.group() in ("叫醒", "喊醒", "唤醒") and not body:
        body = "起床"
    if not body:
        body = "到时间啦。"
    canonical = _natural_time_text(when) + "提醒" + target + "，" + body
    return _parse_explicit_command(canonical, mentioned_ids, requester_id, self_id, now)


def _voice_request_copy(text: str) -> tuple[str, bool]:
    """Remove an explicit delivery modifier only from scheduling request slots."""
    payload = re.search(r"(?:说|告诉(?:我|他|她|他们|她们)?)\s*[:：]", text)
    limit = payload.start() if payload else len(text)
    request = text[:limit]
    if not _natural_clock_slots(request):
        return text, False
    if re.search(r"(?:不要|不用|别|无需|不需要)\s*(?:再)?\s*(?:用|以|通过)?语音", request):
        return text, False
    action = _NATURAL_ACTION.search(request)
    alarm = re.search(r"(?:设置|设|定|订)(?:一个|个|一下)?", request)
    if alarm and "闹钟" in request[alarm.end():] and (not action or alarm.start() < action.start()):
        action = alarm
    if action is None:
        return text, False
    prefix = request[:action.start()]
    modifier = r"(?:用|以|通过)语音(?:消息)?(?:的方式)?|语音(?=\s*$)"
    cleaned, count = re.subn(modifier, "", prefix)
    value = cleaned + text[action.start():]
    if payload is None:
        value, tail_count = re.subn(
            r"[，,；;。.!！?？]\s*(?:(?:请|记得|要|改成|换成|到时)\s*)?"
            r"(?:用|以|通过)语音(?:消息)?(?:的方式)?(?:吧|哦|呀|哈)?[。.!！?？ ]*$", "", value)
        count += tail_count
    return value, bool(count)


def parse_command(text: str, mentioned_ids: Iterable[str | int], requester_id: int,
                  self_id: int, now: float | None = None) -> Command | None:
    """Recognize natural reminder requests, then retain explicit command support."""
    value = str(text or "").strip()
    ids = tuple(mentioned_ids)
    stamp = time.time() if now is None else float(now)
    explicit = _parse_explicit_command(value, ids, requester_id, self_id, stamp)
    if explicit is not None and (explicit.action in ("list", "cancel") or value.startswith("取消")):
        return explicit
    value, voice_requested = _voice_request_copy(value)
    if voice_requested:
        explicit = _parse_explicit_command(value, ids, requester_id, self_id, stamp)
    natural = _parse_natural_request(value, ids, requester_id, self_id, stamp)
    if natural is not None:
        result = None if natural.action == "ignore" else natural
    else:
        result = explicit
    if result is not None and result.action == "schedule" and voice_requested:
        return replace(result, delivery="voice")
    return result


class ReminderStore:
    """SQLite outbox with one atomic pending->sending claim, no blind retries.

    A crash after sending is inherently ambiguous in OneBot. On next open, any
    ``sending`` row becomes ``uncertain`` and will NOT be sent again. This avoids
    duplicate pings; it cannot guarantee delivery exactly once across a crash.
    """
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._connection: sqlite3.Connection | None = None

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=5)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("""CREATE TABLE IF NOT EXISTS reminders (
                id TEXT PRIMARY KEY, bot_id INTEGER NOT NULL,
                group_id INTEGER NOT NULL, requester_id INTEGER NOT NULL,
                targets TEXT NOT NULL, body TEXT NOT NULL,
                due REAL NOT NULL, created REAL NOT NULL,
                status TEXT NOT NULL, error TEXT NOT NULL DEFAULT '',
                source_id TEXT NOT NULL DEFAULT '',
                delivery TEXT NOT NULL DEFAULT 'text')""")
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(reminders)")}
            if "delivery" not in columns:
                connection.execute("ALTER TABLE reminders ADD COLUMN delivery TEXT NOT NULL DEFAULT 'text'")
            connection.execute("CREATE INDEX IF NOT EXISTS reminder_due ON reminders(bot_id,status,due)")
            connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS reminder_source ON reminders(bot_id,group_id,requester_id,source_id) WHERE source_id != ''")
            connection.execute("UPDATE reminders SET status='uncertain',error='程序曾在发送中退出，未自动重发' WHERE status='sending'")
            connection.commit()
            self._connection = connection
        return self._connection

    @staticmethod
    def _job(row: sqlite3.Row) -> dict:
        value = dict(row)
        value["targets"] = tuple(int(n) for n in json.loads(value["targets"]))
        return value

    def add(self, group_id: int, requester_id: int, targets: Iterable[int], body: str,
            due: float, now: float | None = None, *, bot_id: int, source_id: str = "",
            delivery: str = "text") -> dict:
        now = time.time() if now is None else float(now)
        bot_id, group_id, requester_id = int(bot_id), int(group_id), int(requester_id)
        targets = tuple(dict.fromkeys(int(n) for n in targets))
        body = _clean_body(str(body or ""))
        if delivery not in ("text", "voice"):
            raise ValueError("提醒发送方式只能是text或voice。")
        if min(bot_id, group_id, requester_id) <= 0 or not targets or len(targets) > MAX_TARGETS:
            raise ValueError("提醒对象无效。")
        if any(n <= 0 or n == bot_id or n > 2**63 - 1 for n in targets):
            raise ValueError("提醒对象无效。")
        if not body or len(body) > MAX_BODY:
            raise ValueError("提醒必须附带1到300字的正文。")
        if not now < float(due) <= now + MAX_HORIZON:
            raise ValueError("提醒时间须在现在之后、30天以内。")
        source_id = str(source_id or "")[:128]
        db = self._db()
        try:
            db.execute("BEGIN IMMEDIATE")
            if source_id:
                existing = db.execute("SELECT * FROM reminders WHERE bot_id=? AND group_id=? AND requester_id=? AND source_id=?",
                                      (bot_id, group_id, requester_id, source_id)).fetchone()
                if existing:
                    db.commit()
                    return self._job(existing)
            counts = db.execute("""SELECT count(*) AS total,
                sum(CASE WHEN group_id=? THEN 1 ELSE 0 END) AS grp,
                sum(CASE WHEN group_id=? AND requester_id=? THEN 1 ELSE 0 END) AS own
                FROM reminders WHERE bot_id=? AND status IN ('pending','sending')""",
                                (group_id, group_id, requester_id, bot_id)).fetchone()
            if counts["total"] >= 200 or (counts["grp"] or 0) >= 30 or (counts["own"] or 0) >= 5:
                raise ValueError("待办提醒数量已达上限：每人每群5条、每群30条、每个机器人200条。")
            identity = uuid.uuid4().hex[:12]
            db.execute("INSERT INTO reminders(id,bot_id,group_id,requester_id,targets,body,due,created,status,error,source_id,delivery) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                       (identity, bot_id, group_id, requester_id, json.dumps(targets), body,
                        float(due), now, "pending", "", source_id, delivery))
            row = db.execute("SELECT * FROM reminders WHERE id=?", (identity,)).fetchone()
            db.commit()
            return self._job(row)
        except BaseException:
            db.rollback()
            raise

    def list_owned(self, group_id: int, requester_id: int, *, bot_id: int) -> list[dict]:
        rows = self._db().execute("SELECT * FROM reminders WHERE bot_id=? AND group_id=? AND requester_id=? ORDER BY CASE WHEN status='pending' THEN 0 ELSE 1 END,CASE WHEN status='pending' THEN due ELSE -created END,id LIMIT 20",
                                  (int(bot_id), int(group_id), int(requester_id))).fetchall()
        return [self._job(row) for row in rows]

    def cancel_owned(self, group_id: int, requester_id: int, job_id: str, *, bot_id: int) -> int:
        db = self._db()
        params: list = [int(bot_id), int(group_id), int(requester_id)]
        extra = ""
        if job_id != "all":
            extra = " AND id=?"
            params.append(str(job_id))
        with db:
            cursor = db.execute("UPDATE reminders SET status='cancelled' WHERE bot_id=? AND group_id=? AND requester_id=? AND status='pending'" + extra, params)
        return cursor.rowcount

    def due_candidates(self, now: float | None = None, limit: int = 20, *, bot_id: int,
                       grace_seconds: float = 300, include_voice: bool = True) -> list[dict]:
        now = time.time() if now is None else float(now)
        grace = max(0.0, min(900.0, float(grace_seconds)))
        db = self._db()
        with db:
            db.execute("UPDATE reminders SET status='expired',error='已超过迟到发送窗口' WHERE bot_id=? AND status='pending' AND due<? AND (? OR delivery<>'voice')",
                       (int(bot_id), now - grace, int(bool(include_voice))))
        rows = db.execute("SELECT * FROM reminders WHERE bot_id=? AND status='pending' AND due<=? AND due>=? AND (? OR delivery<>'voice') ORDER BY due,id LIMIT ?",
                          (int(bot_id), now, now - grace, int(bool(include_voice)), max(1, min(100, int(limit))))).fetchall()
        return [self._job(row) for row in rows]

    def claim(self, job_id: str, now: float | None = None, *, bot_id: int,
              grace_seconds: float = 300) -> bool:
        now = time.time() if now is None else float(now)
        grace = max(0.0, min(900.0, float(grace_seconds)))
        db = self._db()
        with db:
            cursor = db.execute("UPDATE reminders SET status='sending' WHERE id=? AND bot_id=? AND status='pending' AND due<=? AND due>=?",
                                (str(job_id), int(bot_id), now, now - grace))
        return cursor.rowcount == 1

    def finish(self, job_id: str, status: str, error: str = "", *, bot_id: int) -> None:
        if status not in ("sent", "submitted", "uncertain", "cancelled", "expired"):
            raise ValueError("非法提醒状态")
        db = self._db()
        with db:
            db.execute("UPDATE reminders SET status=?,error=? WHERE id=? AND bot_id=? AND status IN ('pending','sending')",
                       (status, str(error)[:300], str(job_id), int(bot_id)))

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
