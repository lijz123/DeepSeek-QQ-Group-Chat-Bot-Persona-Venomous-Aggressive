# -*- coding: utf-8 -*-
"""QQ 群「deepseek」机器人。    可爱捏 >_<

链路：QQ ← NapCat / LLOneBot(OneBot v11) ←WebSocket→ 本程序 → DeepSeek(OpenAI 兼容)

用法：  //    建议仔细阅读readme，同时结合ai食用
    set DEEPSEEK_API_KEY=sk-xxx
    python bot.py --config config.json
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import copy
import itertools
import json
import logging
import logging.handlers
import os
import random
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from persona import ReplyPlan, TurnUnderstanding, build_system_prompt, build_user_prompt, parse_reply_plan, sanitize_reply
from dialogue import DialogueMixin, IndexedMessage, ParsedMessage, PendingTurn, DialogueState, TOPIC_ENDINGS

log = logging.getLogger("dafish")

# 配置

DEFAULTS: dict[str, Any] = {
    "onebot": {
        "ws_url": "ws://127.0.0.1:3001",
        "access_token": "",
        "heartbeat": 25,
        "reconnect_min": 3,
        "reconnect_max": 60,
        "request_timeout": 15,
    },
    "llm": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "env:DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
        "temperature": 1.15,
        "max_tokens": 100,
        "planning_max_tokens": 192,
        "timeout": 30,
        "retries": 1,
    },
    "persona": {"name": "聊天机器人", "style": "sharp", "extra": ""},
    "behavior": {
        # Batching only waits locally; it does not add any model requests.
        "merge_delay": 1.6,
        "merge_at_delay": 0.6,
        "merge_max_wait": 5.0,
        "merge_max_messages": 20,
        "pending_message_limit": 100,
        "followup_max_turns": 6,
        "followup_grace_window": 30,
        "model_reply_planning": True,
        "model_burst_max_messages": 3,
        # Legacy random addendum mode; ignored when model_reply_planning=True.
        "second_reply_rate": 0.60,
        "second_reply_delay_min": 0.8,
        "second_reply_delay_max": 0.8,
        "at_reply": True,
        "keyword_reply": True,
        "keywords": ["聊天机器人", "机器人", "大肥鱼", "肥鱼", "deepseek", "Deepseek", "ds"],
        "question_reply": False,
        "question_rate": 0.35,
        "random_rate": 0.03,
        # 话题分母：命中 game_keywords 算游戏内容，狠狠骂，其余算日常，要温油。
        "game_keywords": [
            "英雄", "出装", "补刀", "打野", "中单", "上单", "辅助", "射手", "排位",
            "上分", "段位", "王者", "三角洲", "吃鸡", "开黑", "团战", "送人头",
            "战绩", "皮肤", "装备", "兵线", "水晶", "野区", "对局", "队友", "操作",
            "意识", "走位", "技能", "经济", "开局", "翻盘", "打团", "带线", "越塔",
        ],
        # 未被 @ 时的随机接话概率：日常可友好可攻击，游戏只攻击
        "topic_rates": {
            "daily": {"friendly": 0.10, "attack": 0.05},
            "game": {"friendly": 0.0, "attack": 0.10},
        },
        "friendly_tone": (
            "这一条是正常聊天接话：友好、自然，像普通群友搭话，"
            "不攻击、不阴阳怪气、不挑刺、不贬低。"
        ),
        "attack_tone": "按你的性格怼他，可以刻薄。",
        "attack_game_tone": (
            "按你的性格怼他，但必须贴着游戏内容骂：操作、出装、意识、走位、段位、战绩、"
            "队友配合，别脱离游戏去泛泛地人身攻击。"
        ),
        "group_cooldown": 25,
        "direct_cooldown": 3,
        "user_cooldown": 60,
        "min_gap": 1.5,
        "per_group_per_minute": 18,
        "global_per_minute": 25,
        "history_size": 14,
        "history_ttl": 900,
        "quote_reply": True,
        "prefix_at": False,
        "reply_poke": True,
        "ignore_at_all": True,
        "max_chain": 3,
        "max_chars": 120,
        "followup_window": 120,
        "followup_rate": 1.0,
        "direct_user_cooldown": 3,
        "taunt_keywords": [
            "你妈", "尼玛", "妈死", "傻逼", "煞笔", "傻b", "废物",
            "垃圾", "滚", "草", "cnm", "sb",
        ],
        "mute_keywords": ["肥鱼闭嘴"],
        "mute_reply": ["呜呜呜，接下来我再也不会理你了"],
        # 引用回复：never 永不 / always 永远 / auto 只在群里刷得快时才引用
        "quote_mode": "auto",
        "quote_busy_per_minute": 12,
        # 寒暄：认出来就用热情语气回，不走攻击人设
        "greeting_keywords": [
            "早上好", "早安", "大家早", "中午好", "午安", "下午好",
            "晚上好", "晚安", "你好", "您好", "哈喽", "在吗","小家伙"
        ],
        "greeting_tone": (
            "【这一条的语气，优先级高于上面的性格设定】对方在打招呼，你要像正常人一样热情回应，"
            "可以关心一句、接一句寒暄，绝对不要贬低、嘲讽、挑刺、反问。"
            "参考语感：「早啊，今天起这么早」「晚上好，忙完了没」「在呢，啥事你说」"
        ),
        # 出现这些词说明话已经聊完了（笑场/收尾），机器人不要再接
        "quiet_keywords": [
            "破绷", "绷不住", "绷了", "哈哈哈", "笑死", "乐死", "笑不活了", "破防",
        ],
        # 命中关键词直接回固定句子，不走模型（值可以是字符串，也可以是数组随机取）
        "fixed_replies": {
            "华为": ["不敢高声语，恐惊天上人", "这个我不聊，懂的都懂"],
        },
        # 命中关键词就发图 + 一句话；图片相对路径以 bot.py 所在目录为基准
        "image_replies": {},
        "image_reply_per_minute": 3,
        # 组合回复：一条消息同时命中 keywords 里的所有词时，优先走这条
        # [{"keywords": ["华为", "看看鱼尾"], "text": "...", "image": "assets/x.jpg"}]
        "combo_replies": [],
        "combo_reply_per_minute": 3,
        # 睡太久（hours 内没人 @ 也没说话），被 @ 时先赖床一句；多句随机，不连着重复
        "sleep_wake_hours": 5,
        "wake_replies": [
            "干嘛，好不容易睡着了就把人家@醒",
            "几点了还@，你不上班的？",
            "吵什么吵，梦都做到一半了",
            "醒了醒了，别喊了，烦不烦",
            "你最好是有事，不然我接着睡",
            "谁啊，大半夜的@魂呢",
            "说吧，说完我继续睡",
        ],
    },
    "groups": {"whitelist": [], "blacklist": []},
    "users": {"blacklist": []},
    "safety": {
        "blocked_input": ["裸聊", "约炮"],
        "blocked_output": [
            "习近平", "共产党", "六四", "台独", "法轮功", "自杀",
        ],
        "fallback_reply": "君子谨言慎行",
    },
    "runtime": {"dry_run": False, "log_file": "logs/bot.log", "log_level": "INFO"},
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_secret(value: Any) -> str:
    """支持 "env:NAME" 形式，避免把 key 写进文件。"""
    if not isinstance(value, str):
        return ""
    if value.startswith("env:"):
        return os.environ.get(value[4:].strip(), "").strip()
    return value.strip()


def load_config(path: str | Path) -> dict:
    raw: dict = {}
    p = Path(path)
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8-sig"))
    cfg = deep_merge(DEFAULTS, raw)
    cfg["llm"]["api_key"] = resolve_secret(cfg["llm"]["api_key"])
    return cfg


def setup_logging(cfg: dict) -> None:
    level = getattr(logging, str(cfg["runtime"].get("log_level", "INFO")).upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    log_file = cfg["runtime"].get("log_file")
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)


# 消息解析

@dataclass
class Msg:
    speaker: str
    text: str
    is_bot: bool
    ts: float
    quote: str = ""
    # Keep sender identity separate from the message body.
    sender_id: str = ""
    nickname: str = ""
    card: str = ""
    message_id: str = ""
    mentions: tuple[str, ...] = ()
    reply_to_id: str = ""
    reply_to_sender_id: str = ""


def _cq_segments(raw: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    pos = 0
    for m in re.finditer(r"\[CQ:([^\]]+)\]", raw):
        if m.start() > pos:
            segments.append({"type": "text", "data": {"text": raw[pos:m.start()]}})
        body = m.group(1)
        parts = body.split(",")
        data: dict[str, str] = {}
        for item in parts[1:]:
            if "=" in item:
                k, v = item.split("=", 1)
                data[k] = v
        segments.append({"type": parts[0], "data": data})
        pos = m.end()
    if pos < len(raw):
        segments.append({"type": "text", "data": {"text": raw[pos:]}})
    return segments


def parse_message_details(
    message: Any, self_id: int, ignore_at_all: bool = True
) -> ParsedMessage:
    """保留 @ 的实际对象与引用编号；兼容数组与 CQ 字符串。"""
    if isinstance(message, str):
        segments = _cq_segments(message)
    elif isinstance(message, list):
        segments = [s for s in message if isinstance(s, dict)]
    else:
        segments = []

    texts: list[str] = []
    at_self = False
    mentioned_ids: list[str] = []
    media = False
    reply_id = ""
    for seg in segments:
        typ = str(seg.get("type") or "")
        data = seg.get("data") or {}
        if typ == "text":
            texts.append(str(data.get("text") or ""))
        elif typ == "at":
            qq = str(data.get("qq") or "")
            if qq:
                mentioned_ids.append(qq)
            if qq == str(self_id):
                at_self = True
            elif qq == "all":
                if not ignore_at_all:
                    at_self = True
            else:
                texts.append(" ")
        elif typ == "reply":
            reply_id = str(data.get("id") or "")
        elif typ in ("image", "mface", "record", "video", "face", "json", "xml", "forward"):
            media = True
        # rps / dice / file 等忽略
    return ParsedMessage("".join(texts).strip(), at_self, media, reply_id, tuple(dict.fromkeys(mentioned_ids)))


def parse_message(
    message: Any, self_id: int, ignore_at_all: bool = True
) -> tuple[str, bool, bool, str]:
    """保持旧调用的四元组接口；新对话链使用 parse_message_details。"""
    parsed = parse_message_details(message, self_id, ignore_at_all)
    return parsed.text, parsed.at_self, parsed.media, parsed.reply_id


# 上下文 / 限流

class ContextStore:
    def __init__(self, size: int, ttl: float) -> None:
        self.size = max(2, int(size))
        self.ttl = float(ttl)
        self._data: dict[int, deque[Msg]] = {}

    def add(self, group_id: int, msg: Msg) -> None:
        dq = self._data.setdefault(group_id, deque(maxlen=self.size * 3))
        dq.append(msg)
        self._trim(group_id)

    def _trim(self, group_id: int) -> None:
        dq = self._data.get(group_id)
        if not dq:
            return
        now = time.time()
        while dq and now - dq[0].ts > self.ttl:
            dq.popleft()
        while len(dq) > self.size:
            dq.popleft()

    def transcript(self, group_id: int) -> str:
        self._trim(group_id)
        dq = self._data.get(group_id) or deque()
        rows = []
        for m in dq:
            text = re.sub(r"\s+", " ", m.text)[:200]
            if not text:
                continue
            # Each line is JSONL: sender is metadata; content is the actual utterance.
            entry = {
                "role": "assistant" if m.is_bot else "group_member",
                "sender": {
                    "id": m.sender_id or None,
                    "display_name": str(m.speaker)[:64],
                    "nickname": m.nickname or None,
                    "card": m.card or None,
                },
                "content": text,
            }
            if m.message_id:
                entry["message_id"] = m.message_id
            if m.mentions:
                entry["mentions"] = list(m.mentions)
            if m.reply_to_id or m.reply_to_sender_id:
                entry["reply_to"] = {"message_id": m.reply_to_id or None, "sender_id": m.reply_to_sender_id or None}
            if m.quote:
                entry["quote"] = re.sub(r"\s+", " ", m.quote)[:90]
            rows.append(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
        return "\n".join(rows)

    def trailing_bot_count(self, group_id: int) -> int:
        dq = self._data.get(group_id) or deque()
        count = 0
        for m in reversed(dq):
            if not m.is_bot:
                break
            count += 1
        return count


class WindowLimiter:
    """滑动窗口计数：key -> 最近 60s 内的命中时间。"""

    def __init__(self) -> None:
        self._hits: dict[Any, deque[float]] = {}

    def allow(self, key: Any, limit: int, window: float = 60.0) -> bool:
        now = time.monotonic()
        dq = self._hits.setdefault(key, deque())
        while dq and now - dq[0] > window:
            dq.popleft()
        if len(dq) >= limit:
            return False
        dq.append(now)
        return True

    def allow_many(self, limits: list[tuple[Any, int]], window: float = 60.0,
                   *, consume: bool = True) -> bool:
        """Check/reserve several quotas atomically; a denied key charges none."""
        now = time.monotonic()
        queues = []
        for key, limit in limits:
            dq = self._hits.setdefault(key, deque())
            while dq and now - dq[0] > window:
                dq.popleft()
            if len(dq) >= limit:
                return False
            queues.append(dq)
        if consume:
            for dq in queues:
                dq.append(now)
        return True


# OneBot v11 客户端

class OneBotError(RuntimeError):
    pass


class OneBotClient:
    def __init__(self, cfg: dict, on_event) -> None:
        self.cfg = cfg
        self.on_event = on_event
        self.self_id = 0
        self.nickname = ""
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._echo = itertools.count(1)
        self._stop = asyncio.Event()
        self._session: aiohttp.ClientSession | None = None
        self._tasks: set[asyncio.Task] = set()

    async def run(self) -> None:
        self._session = aiohttp.ClientSession()
        delay = float(self.cfg["reconnect_min"])
        try:
            while not self._stop.is_set():
                try:
                    await self._connect_once()
                    delay = float(self.cfg["reconnect_min"])
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - 断线要能自愈
                    log.warning("连接异常：%s", exc)
                if self._stop.is_set():
                    break
                log.info("%.0f 秒后重连…", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, float(self.cfg["reconnect_max"]))
        finally:
            if self._session:
                await self._session.close()

    async def _connect_once(self) -> None:
        assert self._session is not None
        headers = {}
        token = str(self.cfg.get("access_token") or "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        url = str(self.cfg["ws_url"])
        log.info("连接 OneBot：%s", url)
        hb = float(self.cfg.get("heartbeat") or 0)
        ws = await self._session.ws_connect(
            url,
            headers=headers,
            heartbeat=hb or None,
            max_msg_size=0,
        )
        self.ws = ws
        reader = asyncio.create_task(self._read_loop(ws))
        try:
            info = await self._login()
            self.self_id = int(info.get("user_id") or 0)
            self.nickname = str(info.get("nickname") or "")
            log.info("已登录：%s（%s）", info.get("nickname") or "?", self.self_id)
            await reader
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
            self.ws = None
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(OneBotError("连接已断开"))
            self._pending.clear()

    async def _read_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                self._dispatch_raw(msg.data)
            elif msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            ):
                log.warning("OneBot 连接关闭：%s", ws.exception())
                return

    async def _login(self) -> dict:
        last: Exception | None = None
        for _ in range(3):
            try:
                return await self.send_action("get_login_info", {}, timeout=10)
            except Exception as exc:  # noqa: BLE001
                last = exc
                await asyncio.sleep(1.0)
        raise OneBotError(f"get_login_info 失败：{last}")

    def _dispatch_raw(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.debug("忽略非 JSON 数据：%s", raw[:200])
            return
        if not isinstance(data, dict):
            return
        echo = data.get("echo")
        if echo is not None and str(echo) in self._pending:
            fut = self._pending.pop(str(echo))
            if not fut.done():
                fut.set_result(data)
            return
        if data.get("post_type"):
            task = asyncio.create_task(self._safe_event(data))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _safe_event(self, data: dict) -> None:
        try:
            await self.on_event(data)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("处理事件失败")

    async def send_action(
        self, action: str, params: dict | None = None, timeout: float | None = None
    ) -> dict:
        if self.ws is None or self.ws.closed:
            raise OneBotError("尚未连接")
        echo = str(next(self._echo))
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[echo] = fut
        try:
            await self.ws.send_json({"action": action, "params": params or {}, "echo": echo})
            resp = await asyncio.wait_for(
                fut, timeout=float(timeout or self.cfg["request_timeout"])
            )
        finally:
            self._pending.pop(echo, None)
        status = resp.get("status")
        retcode = resp.get("retcode", 0)
        if status not in ("ok", "async") or retcode not in (0, None):
            raise OneBotError(f"{action} 失败：{str(resp)[:300]}")
        return resp.get("data") or {}

    async def stop(self) -> None:
        self._stop.set()
        if self.ws is not None and not self.ws.closed:
            await self.ws.close()


# --------------------------------------------------------------------------
# 大脑
# --------------------------------------------------------------------------

class Brain:
    def __init__(self, cfg: dict, safety: dict, persona: dict, max_chars: int) -> None:
        self.cfg = cfg
        self.safety = safety
        self.persona = persona
        self.max_chars = int(max_chars)
        self._session: aiohttp.ClientSession | None = None
        self._recent: dict[int, deque[str]] = {}
        self._warned = False

    async def reply(self, group_id: int, transcript: str, speaker: str, is_at: bool,
                    tone: str = "", focus_note: str = "") -> str | None:
        result = await self._respond(group_id, transcript, speaker, is_at, tone, focus_note)
        return result if isinstance(result, str) else None

    async def plan_reply(self, group_id: int, transcript: str, speaker: str, is_at: bool,
                         tone: str = "", focus_note: str = "") -> ReplyPlan | None:
        result = await self._respond(group_id, transcript, speaker, is_at, tone, focus_note, plan_mode=True)
        return result if isinstance(result, ReplyPlan) else None

    async def _respond(self, group_id: int, transcript: str, speaker: str, is_at: bool,
                       tone: str = "", focus_note: str = "", *, plan_mode: bool = False
                       ) -> str | ReplyPlan | None:
        key = str(self.cfg.get("api_key") or "")
        if not key:
            if not self._warned:
                log.error("未配置模型 key：设置环境变量 DEEPSEEK_API_KEY 或 llm.api_key")
                self._warned = True
            return None
        if self._session is None:
            self._session = aiohttp.ClientSession()

        messages = [
            {
                "role": "system",
                "content": build_system_prompt(
                    str(self.persona.get("name") or "大肥鱼"),
                    str(self.persona.get("style") or "sharp"),
                    str(self.persona.get("extra") or ""),
                    tone, plan_mode=plan_mode,
                ),
            },
            {"role": "user", "content": build_user_prompt(transcript, speaker, is_at, focus_note, plan_mode=plan_mode)},
        ]
        recent = self._recent.setdefault(group_id, deque(maxlen=5))
        base_temp = float(self.cfg.get("temperature", 1.1))
        attempts = int(self.cfg.get("retries", 1)) + 2
        protocol_failures = 0
        fallback_plan = None
        fallback_source = ""

        for attempt in range(attempts):
            temperature = min(1.6, base_temp + 0.15 * attempt)
            raw = await self._call(messages, temperature, plan_mode=plan_mode)
            if not raw:
                continue
            if plan_mode:
                parse_options = dict(name=str(self.persona.get("name") or ""),
                    blocked=list(self.safety.get("blocked_output") or []),
                    fallback=str(self.safety.get("fallback_reply") or "君子谨言慎行"),
                    max_chars=self.max_chars)
                plan = parse_reply_plan(raw, strict=True, **parse_options)
                if plan is None:
                    protocol_failures += 1
                    compatible = parse_reply_plan(raw, **parse_options)
                    source = "invalid"
                    if compatible and compatible.action == "reply":
                        fallback_plan = compatible
                        source = "missing_next" if raw.lstrip().startswith(("{", "```")) else "plain"
                        fallback_source = source
                    log.warning("[群 %s] 回复规划格式纠正 source=%s attempt=%d",
                                group_id, source, attempt + 1)
                    # One repair retry for old text/missing-next responses, then
                    # keep chat available via an explicit, observable single reply.
                    if protocol_failures >= 2 and fallback_plan:
                        break
                    messages[-1]["content"] += ("\n格式纠正：只输出合法JSON，完整填写action、text、next。"
                        "next必须从reply/wait/stop中选一个；是否继续由内容决定，不要为格式凑条数。")
                    continue
                source = "json" if raw.lstrip().startswith(("{", "```")) else "control"
                log.info("[群 %s] 回复规划 source=%s action=%s next=%s attempt=%d anchor=%s understanding=%s",
                         group_id, source, plan.action, plan.next_action, attempt + 1,
                         bool(plan.anchor), bool(plan.understanding))
                if plan.action != "reply":
                    return plan
                out = plan.text
            else:
                out = sanitize_reply(
                    raw, name=str(self.persona.get("name") or ""),
                    blocked=list(self.safety.get("blocked_output") or []),
                    fallback=str(self.safety.get("fallback_reply") or "君子谨言慎行"),
                    max_chars=self.max_chars)
            if not out:
                continue
            if out in recent:
                log.debug("重复回复，重试：%s", out)
                messages[-1]["content"] += "\n（换一种说法，别和你上一条一样）"
                continue
            recent.append(out)
            return plan if plan_mode else out
        if plan_mode and fallback_plan and fallback_plan.text not in recent:
            log.warning("[群 %s] 回复规划降级 source=%s_fallback action=reply next=wait",
                        group_id, fallback_source)
            recent.append(fallback_plan.text)
            return ReplyPlan("reply", fallback_plan.text, "wait")
        if plan_mode:
            log.warning("[群 %s] 回复规划结束 reason=no_valid_plan", group_id)
        return None

    async def _call(self, messages: list[dict], temperature: float, *, plan_mode: bool = False) -> str | None:
        assert self._session is not None
        url = str(self.cfg["base_url"]).rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg["model"],
            "messages": messages,
            "temperature": round(temperature, 2),
            "max_tokens": max(int(self.cfg.get("max_tokens", 100)),
                              int(self.cfg.get("planning_max_tokens", 192)) if plan_mode else 0),
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.cfg['api_key']}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=float(self.cfg.get("timeout", 30)))
        try:
            async with self._session.post(
                url, json=payload, headers=headers, timeout=timeout
            ) as resp:
                body = await resp.text()
                if resp.status != 200:
                    log.warning("模型返回 %s：%s", resp.status, body[:300])
                    return None
                data = json.loads(body)
            finish = data["choices"][0].get("finish_reason")
            if plan_mode and finish == "length":
                log.warning("回复规划输出被截断 finish_reason=length；检查 planning_max_tokens")
            return str(data["choices"][0]["message"]["content"] or "").strip()
        except asyncio.TimeoutError:
            log.warning("模型请求超时")
        except (aiohttp.ClientError, KeyError, IndexError, json.JSONDecodeError) as exc:
            log.warning("模型请求失败：%s", exc)
        return None

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

# 机器人主体

class Bot(DialogueMixin):
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.beh = cfg["behavior"]
        self.safety = cfg["safety"]
        self.name = str(cfg["persona"]["name"] or "大肥鱼")
        self.store = ContextStore(int(self.beh["history_size"]), float(self.beh["history_ttl"]))
        self.limiter = WindowLimiter()
        self.brain = Brain(cfg["llm"], self.safety, cfg["persona"], int(self.beh["max_chars"]))
        self.client = OneBotClient(cfg["onebot"], self.on_event)
        self._last_group: dict[int, float] = {}
        self._last_user: dict[tuple[int, int], float] = {}
        self._last_reply: dict[int, tuple[int, float]] = {}
        self._muted: set[tuple[int, int]] = set()
        self._msg_index: dict[tuple[int, str], IndexedMessage] = {}
        self._activity: dict[int, deque[float]] = {}
        self._recent_pick: dict[str, str] = {}
        self._image_cache: dict[str, str] = {}
        self._last_image_pick: dict[str, str] = {}
        self._last_awake = time.monotonic()
        self._last_any = 0.0
        self._seen: dict[str, float] = {}
        self._inflight = False
        self._whitelist = {int(g) for g in (cfg["groups"]["whitelist"] or [])}
        self._group_blacklist = {int(g) for g in (cfg["groups"]["blacklist"] or [])}
        self._user_blacklist = {int(u) for u in (cfg["users"]["blacklist"] or [])}
        self._init_dialogue()
        self._delivery_lock = asyncio.Lock()

    async def run(self) -> None:
        try:
            log.info("回复设置 planning=%s burst_limit=%s delay=%s~%ss group_per_minute=%s",
                     bool(self.beh.get("model_reply_planning", True)),
                     self.beh.get("model_burst_max_messages", 3),
                     self.beh.get("second_reply_delay_min", .8),
                     self.beh.get("second_reply_delay_max", .8),
                     self.beh.get("per_group_per_minute"))
            await self.client.run()
        finally:
            await self.close()

    # ---- 过滤 ----

    def _group_allowed(self, group_id: int) -> bool:
        if group_id in self._group_blacklist:
            return False
        if self._whitelist and group_id not in self._whitelist:
            return False
        return True

    @staticmethod
    def _is_safe_name(name: str) -> bool:
        """昵称像一句话时（带句读标点、或长到像短语），不进上下文，避免被当成正文。"""
        if not name or len(name) > 18:
            return False
        if re.search(r"[？?。！!，,、；;：:\n\r]", name):
            return False
        if " " in name and len(name) > 8:
            return False
        return True

    def _is_dup(self, message_id: Any, group_id: int = 0) -> bool:
        if message_id is None:
            return False
        key = f"{group_id}:{message_id}"
        now = time.monotonic()
        for k in [k for k, ts in self._seen.items() if now - ts > 300]:
            self._seen.pop(k, None)
        if key in self._seen:
            return True
        self._seen[key] = now
        return False

    # ---- 事件入口 ----

    async def on_event(self, event: dict) -> None:
        ptype = event.get("post_type")
        if ptype == "notice":
            await self._on_notice(event)
            return
        if ptype != "message" or event.get("message_type") != "group":
            return
        await self._on_group_message(event)

    async def _on_group_message(self, event: dict) -> None:
        """Receive promptly; expensive reply work is serialized by group."""
        group_id = int(event.get("group_id") or 0)
        user_id = int(event.get("user_id") or 0)
        if not group_id or not user_id or user_id == self.client.self_id:
            return
        if not self._group_allowed(group_id) or user_id in self._user_blacklist:
            return
        if self._is_dup(event.get("message_id"), group_id):
            return
        # Even a forward/empty attachment cancels an old pending second reply.
        arrival_revision = self._note_group_activity(group_id, user_id)
        now = time.monotonic()
        load = self._activity.setdefault(group_id, deque())
        load.append(now)
        while load and load[0] < now - 60:
            load.popleft()
        sender = event.get("sender") or {}
        nickname = str(sender.get("nickname") or "")
        card = str(sender.get("card") or "")
        raw_name = str(card or nickname or f"用户{user_id}")
        # Identity and content remain separate; sentence-like names use a neutral label.
        speaker = raw_name if self._is_safe_name(raw_name) else f"用户{user_id}"
        parsed = parse_message_details(event.get("message"), self.client.self_id, bool(self.beh["ignore_at_all"]))
        stored = parsed.text or ("[表情]" if parsed.media else "")
        if not stored and not parsed.at_self:
            return
        message_id = str(event.get("message_id") or "")
        prev = self._msg_index.get((group_id, parsed.reply_id)) if parsed.reply_id else None
        quote = ""
        if parsed.reply_id:
            quote = f"{prev.speaker}：{prev.text[:60]}" if prev else "更早的一条消息（作者与原话未记录，不猜测回复对象）"
        msg = Msg(
            speaker, stored, False, time.time(), quote, sender_id=str(user_id),
            nickname=nickname, card=card, message_id=message_id,
            mentions=parsed.mentioned_ids, reply_to_id=parsed.reply_id,
            reply_to_sender_id=str(prev.user_id) if prev else "",
        )
        if message_id:
            self._msg_index[(group_id, message_id)] = IndexedMessage(
                group_id, user_id, speaker, stored, msg.ts, False)
        for key in [key for key, value in self._msg_index.items() if msg.ts - value.ts > 600]:
            self._msg_index.pop(key, None)
        self.store.add(group_id, msg)
        self._enqueue_turn(PendingTurn(
            group_id, user_id, message_id, msg, parsed.at_self, parsed.mentioned_ids,
            parsed.reply_id, msg.reply_to_sender_id, now, now,
            (message_id,) if message_id else (), arrival_revision=arrival_revision,
        ))

    async def _process_turn(self, turn: PendingTurn) -> None:
        group_id, user_id = turn.group_id, turn.user_id
        text, at_self = turn.message.text, turn.at_self
        speaker = turn.message.speaker
        event = {"message_id": turn.message_id or None}
        # Protect other people's conversations before fixed/image/mute triggers too.
        if self._addressed_elsewhere(turn):
            self._friendly_sessions.pop((group_id, user_id), None)
            return
        if self._superseded(turn):
            return
        # Rechecks reuse rate reservations, never safety decisions for old text.
        if any(word and str(word) in text for word in (self.safety.get("blocked_input") or [])):
            return
        # Freeze this turn's tone before an explicit closing clears the session.
        # Generation/ACK latency must not turn a friendly reply sharp mid-flight.
        self._turn_tones[group_id] = (user_id, self._session_tone(group_id, user_id))
        if self._is_tone_session_end(text):
            self._friendly_sessions.pop((group_id, user_id), None)
        if self._is_tone_session_end(text) or any(marker in text for marker in TOPIC_ENDINGS):
            self._dialogues.pop(group_id, None)
            if not self._is_direct_turn(turn):
                return
        # 被这个人拉黑过：他说什么都不理，除非他再 @ 我一次
        if self._is_muted(group_id, user_id):
            if not at_self:
                return
            self._unmute(group_id, user_id)
            log.info("[群 %s] %s 重新 @ 了，解除静音", group_id, user_id)

        # 「肥鱼闭嘴」：从此不再理这个人，直到他下次 @ 我
        if self._apply_mute(group_id, user_id, text):
            await self._send(
                group_id, self._mute_line(), event.get("message_id"), user_id, "mute"
            )
            return

        # 组合回复：一条消息同时命中多个关键词时，优先走组合规则
        combo = self._combo_reply(text)
        if combo:
            now = time.monotonic()
            gap_ok = now - self._last_any >= float(self.beh["min_gap"])
            if gap_ok and self.limiter.allow(
                ("combo", group_id), int(self.beh.get("combo_reply_per_minute", 3))
            ):
                await self._send_image(
                    group_id, combo[0], combo[1], event.get("message_id"), user_id
                )
            return

        # 图片回复：命中关键词就发图 + 一句话
        pic = self._image_reply(text)
        if pic:
            now = time.monotonic()
            gap_ok = now - self._last_any >= float(self.beh["min_gap"])
            if gap_ok and self.limiter.allow(
                ("image", group_id), int(self.beh.get("image_reply_per_minute", 3))
            ):
                await self._send_image(
                    group_id, pic[0], pic[1], event.get("message_id"), user_id
                )
            return

        # 固定回复：命中关键词就发预设句子
        canned = self._fixed_reply(text)
        if canned:
            now = time.monotonic()
            gap_ok = now - self._last_any >= float(self.beh["min_gap"])
            if gap_ok and self.limiter.allow(("fixed", group_id), int(self.beh.get("fixed_reply_per_minute", 3))):
                await self._send(group_id, canned, event.get("message_id"), user_id, "fixed")
            return

        # 笑场/收尾词：这条不接，让别人把话说完
        if self._is_quiet(text):
            state = self._dialogues.get(group_id)
            if state and state.user_id == user_id:
                self._dialogues.pop(group_id, None)
            return

        # 太久没人 @（也一直没说话），被 @ 时赖床一句
        now = time.monotonic()
        slept = now - self._last_awake
        if at_self:
            self._last_awake = now
            if slept >= float(self.beh.get("sleep_wake_hours", 5)) * 3600:
                await self._send(
                    group_id,
                    self._wake_line(),
                    event.get("message_id"),
                    user_id,
                    "wake",
                )
                return

        reply_to_self = turn.reply_sender_id == str(self.client.self_id)
        reason = turn.reserved_reason or self._decide(group_id, user_id, text, at_self, reply_to_self=reply_to_self, is_poke=turn.is_poke)
        if turn.is_poke and reason:
            reason = "poke"
        if not reason:
            return
        if reason in ("greeting", "friendly"):
            self._turn_tones[group_id] = (user_id, "friendly")
        if reason in ("followup", "taunt") and not await self._wait_followup_gap(turn):
            log.info("[群 %s] 取消隐式续聊：对话对象已变化", group_id)
            return
        if self.store.trailing_bot_count(group_id) >= int(self.beh["max_chain"]):
            log.debug("连续发言过多，跳过")
            return

        self._inflight = True
        try:
            tone = self._tone_for(reason, text, group_id=group_id, user_id=user_id)
            transcript, target_note = self._reply_context(turn)
            focus_note = target_note + "\n" + self._focus_note(reason)
            plan = None
            if self.beh.get("model_reply_planning", True):
                plan = await self._get_reply_plan(group_id, transcript, speaker, at_self,
                                                  tone=tone, focus_note=focus_note)
                answer = plan.text if plan and plan.action == "reply" else None
                if plan and plan.action != "reply":
                    log.info("[群 %s] 模型选择 %s，等待新群消息", group_id, plan.action)
            else:
                answer = await self.brain.reply(group_id, transcript, speaker, at_self,
                                                tone=tone, focus_note=focus_note)
        finally:
            self._inflight = False
        if not answer or self._superseded(turn):
            return
        if self._defer_late_supplement(turn, reason):
            return
        if await self._send(group_id, answer, event.get("message_id"), user_id, reason):
            if self.beh.get("model_reply_planning", True) and plan:
                await self._continue_model_turn(turn, plan, reason)
            else:
                await self._maybe_send_second(turn, answer, reason)

    async def _get_reply_plan(self, group_id: int, transcript: str, speaker: str, is_at: bool,
                              *, tone: str = "", focus_note: str = "") -> ReplyPlan | None:
        method = getattr(self.brain, "plan_reply", None)
        if callable(method):
            result = await method(group_id, transcript, speaker, is_at, tone=tone, focus_note=focus_note)
            return result if isinstance(result, ReplyPlan) else None
        # A legacy/custom brain can still reply once, but cannot create a burst
        # without an explicit structured continuation decision.
        text = await self.brain.reply(group_id, transcript, speaker, is_at, tone=tone, focus_note=focus_note)
        return ReplyPlan("reply", text, "wait") if isinstance(text, str) and text.strip() else None

    async def _continue_model_turn(self, turn: PendingTurn, plan: ReplyPlan, reason: str) -> None:
        """The model chooses 1/2/3 messages; limits are only a safety ceiling."""
        limit = min(max(1, int(self.beh.get("model_burst_max_messages", 3))),
                    max(1, int(self.beh["max_chain"])))
        sent_texts = [plan.text]
        anchor = plan.anchor
        understanding = plan.understanding
        def finish(why: str) -> None:
            log.info("[群 %s] 本轮结束 sent=%d reason=%s", turn.group_id, len(sent_texts), why)
        norm = lambda value: re.sub(r"[\W_]+", "", value, flags=re.UNICODE).casefold()
        seen = {norm(plan.text)}
        while plan.next_action == "reply" and len(sent_texts) < limit:
            if not self._second_reply_ready(turn):
                finish("activity_or_target_changed")
                return
            if plan.text == str(self.safety.get("fallback_reply") or ""):
                finish("safety_fallback")
                return
            if self.store.trailing_bot_count(turn.group_id) >= int(self.beh["max_chain"]):
                finish("chain_limit")
                return
            metadata = {}
            if anchor:
                metadata["anchor"] = anchor
            if understanding is not None:
                metadata["understanding"] = understanding
            plan = await self._plan_continuation_step(turn, sent_texts, reason, **metadata)
            if not plan or plan.action != "reply":
                finish("model_" + plan.action if plan else "continuation_cancelled_or_invalid")
                return
            if anchor and (not plan.anchor or plan.anchor != anchor):
                finish("anchor_missing" if not plan.anchor else "anchor_changed")
                return
            if understanding is not None and plan.understanding != understanding:
                finish("understanding_missing" if plan.understanding is None else "understanding_changed")
                return
            # Older providers may omit anchor. Once one is available, keep it
            # stable for remaining steps; never print this metadata into chat.
            anchor = anchor or plan.anchor
            # These tags describe the selected member's unchanged utterance,
            # not the bot's changing sentence form. A new input starts afresh.
            # Equality cannot prove the initial understanding was correct.
            understanding = understanding or plan.understanding
            if (not norm(plan.text) or norm(plan.text) in seen
                    or plan.text == str(self.safety.get("fallback_reply") or "")):
                finish("duplicate_or_filtered")
                return
            if not self._second_reply_ready(turn):
                finish("activity_or_target_changed")
                return
            if not await self._send(turn.group_id, plan.text, turn.message_id or None,
                                    turn.user_id, "continuation"):
                finish("delivery_rejected")
                return
            seen.add(norm(plan.text))
            sent_texts.append(plan.text)
        # wait/stop end this burst. Never poll the model again without new input.
        finish("message_ceiling" if plan.next_action == "reply" else "next_" + plan.next_action)

    async def _plan_continuation_step(self, turn: PendingTurn, sent_texts: list[str],
                                       reason: str, *, anchor: str = "",
                                       understanding: TurnUnderstanding | None = None) -> ReplyPlan | None:
        low = max(0.0, float(self.beh.get("second_reply_delay_min", 0.8)))
        high = max(low, float(self.beh.get("second_reply_delay_max", 0.8)))
        cancel = asyncio.Event()
        self._second_cancel_events[turn.group_id] = cancel
        generation = watcher = None
        try:
            try:
                await asyncio.wait_for(cancel.wait(), timeout=random.uniform(low, high))
                log.info("[群 %s] 续话取消 stage=delay reason=cancel_event", turn.group_id)
                return None
            except asyncio.TimeoutError:
                pass
            if not self._second_reply_ready(turn):
                log.info("[群 %s] 续话取消 stage=preflight reason=activity_or_target_changed", turn.group_id)
                return None
            if not self.limiter.allow_many(self._second_reply_limits(turn.group_id), consume=False):
                log.info("[群 %s] 续话取消 stage=preflight reason=rate_limit", turn.group_id)
                return None
            transcript, target_note = self._reply_context(turn)
            tone = self._tone_for(reason, turn.message.text,
                                  group_id=turn.group_id, user_id=turn.user_id)
            tone += ("\n【自主话轮规划】不要因为还有额度就凑句数。前面已表达完整、需要对方回应时选wait；"
                     "话轮结束时选stop。仅有针对当前落点的新补充才reply；新角度不是另翻一个旧话题，"
                     "不能把上一条换词再说一遍。保持原人设和临时语气，友好时不突然转攻击；"
                     "原本互怼时可轻度调侃，明确结束、现实威胁等边界仍优先。")
            focus = (target_note + "\n【同一话轮，非新群友发言】你已经成功发出："
                     + json.dumps(sent_texts, ensure_ascii=False)
                     + f"。现在考虑第{len(sent_texts) + 1}条；并非必须发送。"
                     "没有新群消息，不得编造对方回复，不重复已发内容，不自问自答。自行决定reply/wait/stop。")
            if anchor:
                focus += ("\n【本轮已确定回复落点】" + json.dumps({"anchor": anchor}, ensure_ascii=False)
                          + "。若reply，返回的anchor须原样沿用；围绕当前反问/回应补充，"
                          "不要转向没有承接关系的旧背景。没有新内容则wait/stop。")
            if understanding is not None:
                focus += ("\n【本轮当前群友原话的理解标签】"
                          + json.dumps({"understanding": {"act": understanding.act,
                                                         "target": understanding.target}}, ensure_ascii=False)
                          + "。没有新用户发言，这两个标签若reply须原样沿用。它们描述群友原话，"
                          "不是你这条回复的句型；self是当前群友，bot_reply是其针对你的话。"
                          "不要将对你的评价偷换成群友自述。标签并非正确性证明；"
                          "没有把握自然补充时wait/stop，不为沿用标签硬编内容。")
            generation = asyncio.create_task(self._get_reply_plan(
                turn.group_id, transcript, turn.message.speaker, turn.at_self,
                tone=tone, focus_note=focus))
            watcher = asyncio.create_task(cancel.wait())
            await asyncio.wait((generation, watcher), return_when=asyncio.FIRST_COMPLETED)
            if cancel.is_set() or not self._second_reply_ready(turn):
                log.info("[群 %s] 续话取消 stage=generation reason=activity_or_target_changed", turn.group_id)
                return None
            return await generation
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("[群 %s] 自主续话失败：%s", turn.group_id, exc)
            return None
        finally:
            if self._second_cancel_events.get(turn.group_id) is cancel:
                self._second_cancel_events.pop(turn.group_id, None)
            tasks = [task for task in (generation, watcher) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def _second_reply_limits(self, group_id: int) -> list[tuple[Any, int]]:
        return [(("group", group_id), int(self.beh["per_group_per_minute"])),
                (("global",), int(self.beh["global_per_minute"]))]

    async def _maybe_send_second(self, turn: PendingTurn, first_text: str, reason: str) -> None:
        """One optional addendum, never recursive; fresh group activity cancels it."""
        rate = max(0.0, min(1.0, float(self.beh.get("second_reply_rate", 0.60))))
        if (rate <= 0 or not self._second_reply_ready(turn)
                or first_text == str(self.safety.get("fallback_reply") or "")
                or self.store.trailing_bot_count(turn.group_id) >= int(self.beh["max_chain"])):
            return
        if random.random() >= rate:
            return
        low = max(0.0, float(self.beh.get("second_reply_delay_min", 0.8)))
        high = max(low, float(self.beh.get("second_reply_delay_max", 0.8)))
        cancel = asyncio.Event()
        self._second_cancel_events[turn.group_id] = cancel
        generation = watcher = None
        try:
            try:
                await asyncio.wait_for(cancel.wait(), timeout=random.uniform(low, high))
                return
            except asyncio.TimeoutError:
                pass
            if not self._second_reply_ready(turn):
                return
            # Preflight costs no quota. Reserve both counters only at delivery.
            if not self.limiter.allow_many(self._second_reply_limits(turn.group_id), consume=False):
                return
            transcript, target_note = self._reply_context(turn)
            tone = self._tone_for(reason, turn.message.text,
                                  group_id=turn.group_id, user_id=turn.user_id)
            tone += ("\n【本次为程序许可的同轮第二条】只补一句不同角度的自然短话，不要重复首条。"
                     "保持既定人设；首条友好/寒暄则继续友好，不突然攻击。"
                     "原本互怼时可轻度调侃；单个笑脸不自动视作结束，但明确结束/现实威胁等边界仍优先。"
                     "没有合适补充时只输出：不补发。不要解释补发机制。")
            focus = (target_note + "\n【同轮补发第二条】你已成功发出："
                     + json.dumps(first_text, ensure_ascii=False)
                     + "。当前没有新群消息。只针对原对象补一句新意思，不重复首条，不自问自答，不编造对方的新发言。")
            generation = asyncio.create_task(self.brain.reply(
                turn.group_id, transcript, turn.message.speaker, turn.at_self,
                tone=tone, focus_note=focus))
            watcher = asyncio.create_task(cancel.wait())
            await asyncio.wait((generation, watcher), return_when=asyncio.FIRST_COMPLETED)
            if cancel.is_set() or not self._second_reply_ready(turn):
                return
            answer = await generation
            if not answer or answer.strip(" 。.!！\n") == "不补发":
                return
            norm = lambda text: re.sub(r"[\W_]+", "", text, flags=re.UNICODE).casefold()
            if not norm(answer) or norm(answer) == norm(first_text):
                return
            if answer == str(self.safety.get("fallback_reply") or ""):
                return
            if self._second_reply_ready(turn):
                await self._send(turn.group_id, answer, turn.message_id or None, turn.user_id, "second")
        except asyncio.CancelledError:
            raise
        except Exception as exc:                                                        #概率限制
            log.warning("[群 %s] 补发生成失败：%s", turn.group_id, exc)
        finally:
            if self._second_cancel_events.get(turn.group_id) is cancel:
                self._second_cancel_events.pop(turn.group_id, None)
            tasks = [task for task in (generation, watcher) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _on_notice(self, event: dict) -> None:
        if event.get("notice_type") != "notify" or event.get("sub_type") != "poke":
            return
        if str(event.get("target_id")) != str(self.client.self_id):
            return
        group_id, user_id = int(event.get("group_id") or 0), int(event.get("user_id") or 0)
        if not group_id or not user_id or user_id == self.client.self_id:
            return
        if not self._group_allowed(group_id) or user_id in self._user_blacklist or self._is_muted(group_id, user_id):
            return
        arrival_revision = self._note_group_activity(group_id, user_id)
        if not self.beh.get("reply_poke"):
            return
        now = time.monotonic()
        msg = Msg(f"群友{user_id}", "[戳了戳机器人]", False, time.time(), sender_id=str(user_id))
        self._enqueue_turn(PendingTurn(group_id, user_id, "", msg, True, (), "", "", now, now, (), is_poke=True,
                                       arrival_revision=arrival_revision))

    # ---- 触发判定 ----

    def _apply_mute(self, group_id: int, user_id: int, text: str) -> bool:
        """命中「肥鱼闭嘴」→ 永久无视此人，直到他下次 @ 我。返回 True 表示要回一句。"""
        words = self.beh.get("mute_keywords") or []
        if not text or not any(w and str(w) in text for w in words):
            return False
        if self._is_muted(group_id, user_id):
            return False
        self._muted.add((group_id, user_id))
        self._friendly_sessions.pop((group_id, user_id), None)
        log.info("[群 %s] %s 让我闭嘴，已加入无视名单", group_id, user_id)
        return True

    def _is_muted(self, group_id: int, user_id: int) -> bool:
        return (group_id, user_id) in self._muted

    def _unmute(self, group_id: int, user_id: int) -> None:
        self._muted.discard((group_id, user_id))

    def _mute_line(self) -> str:
        lines = self.beh.get("mute_reply") or ["呜呜呜，接下来我再也不会理你了"]
        return self._pick("mute", lines) or "呜呜呜，接下来我再也不会理你了"

    def _is_quiet(self, text: str) -> bool:
        """笑场/收尾词：这条不接，也不触发任何回复。"""
        words = self.beh.get("quiet_keywords") or []
        return bool(text) and any(w and str(w) in text for w in words)

    def _is_greeting(self, text: str) -> bool:
        """寒暄：以问候词开头、整句不长、且不是在骂人。

        「早上好啊小肥鱼」「早上好啊deepseek」都算；「你好菜」不算。
        称呼部分允许是机器人自己的名字（可能很长），先剥掉再判断。
        """
        clean = re.sub(r"[\s，。！？!?,.~～、]+", "", text.strip().lower())
        if not clean or len(clean) > 20:
            return False
        if any(bad in clean for bad in ("菜", "傻", "滚", "废", "垃圾", "闭嘴")):
            return False

        aliases = [str(k).lower() for k in (self.beh.get("keywords") or []) if k]
        aliases += [self.client.nickname.lower(), self.name.lower()]
        aliases += ["宝宝", "小东西", "兄弟", "老哥", "大佬", "哥", "姐", "宝"]
        # 名字里可能带点/空格（deepseek V4.1），按同样的规则归一化再比
        norm = lambda v: re.sub(r"[\s，。！？!?,.~～、]+", "", str(v).lower())  # noqa: E731
        aliases = [norm(a) for a in aliases if a]

        for w in self.beh.get("greeting_keywords") or []:
            word = norm(w)
            if not word or not clean.startswith(word):
                continue
            rest = clean[len(word):]
            if len(rest) <= 6:
                return True
            for a in aliases:
                if a:
                    rest = rest.replace(a, "")
            rest = re.sub(r"[啊呀哇哈哦捏鸭宝哥姐]", "", rest)
            if len(rest) <= 1:
                return True
        return bool(re.fullmatch(r"(hi|hello|morning)", clean))

    def _topic(self, text: str) -> str:
        """粗略分话题：命中游戏词算 game，否则 daily。"""
        lowered = text.lower()
        for w in self.beh.get("game_keywords") or []:
            if w and str(w).lower() in lowered:
                return "game"
        return "daily"

    def _focus_note(self, reason: str) -> str:
        """只有在问候时补一句：这条是新问候，别扯之前的事。"""
        if reason == "greeting":
            return "这是新的问候，直接回应问候，不要提之前聊过的事。"
        return ""

    def _has_topic_rates(self) -> bool:
        return bool(self.beh.get("topic_rates"))

    def _roll_topic_reason(self, text: str) -> str | None:
        """按话题掷骰：先判攻击档，再判友好档。攻击优先。"""
        rates = (self.beh.get("topic_rates") or {}).get(self._topic(text)) or {}
        attack = float(rates.get("attack") or 0)
        friendly = float(rates.get("friendly") or 0)
        if attack <= 0 and friendly <= 0:
            return None
        roll = random.random()
        if roll < attack:
            return "attack"
        if roll < attack + friendly:
            return "friendly"
        return None

    def _is_tone_session_end(self, text: str) -> bool:
        """真正离场结束会话；换话题、简短回答、笑场不等于离场。"""
        clean = re.sub(r"[\s，。！？!?,.~～、]+", "", text.strip().lower())
        # Only short standalone farewells, not mentions/quotes inside a sentence.
        aliases = [self.name, self.client.nickname, *(self.beh.get("keywords") or [])]
        aliases = [prefix + str(a).lower() for a in aliases if a for prefix in ("", "大", "小", "老")]
        for alias in sorted(aliases, key=len, reverse=True):
            alias = re.sub(r"[\s，。！？!?,.~～、]+", "", alias)
            if alias:
                clean = clean.replace(alias, "")
        if re.match(r"^(?:那|行|好|好吧|我|我们|咱们)?(?:不聊了|先不聊|不用回|别回复)", clean):
            return True
        return bool(re.fullmatch(r"(?:晚安|拜拜|再见|回头聊|下次聊|先这样)[啊呀啦吧哦了哈]*", clean))

    def _session_tone(self, group_id: int, user_id: int) -> str:
        """Tone lifetime is independent of implicit routing/topic/turn-count gates."""
        key = (group_id, user_id)
        stamp = self._friendly_sessions.get(key)
        if stamp is None:
            return ""
        window = max(0.0, float(self.beh.get("followup_window", 120)))
        if window <= 0 or time.monotonic() - stamp > window:
            self._friendly_sessions.pop(key, None)
            return ""
        return "friendly"

    def _tone_for(self, reason: str, text: str, *, group_id: int = 0, user_id: int = 0) -> str:
        """会话温和覆盖优先于默认毒舌；无会话时沿用原来的单条规则。"""
        beh = self.beh
        session = self._session_tone(group_id, user_id)
        captured = self._turn_tones.get(group_id)
        friendly = session == "friendly" or captured == (user_id, "friendly")
        if friendly or reason in ("greeting", "friendly"):
            # Greeting is only the opening. Later replies use ordinary friendly
            # conversation, not repeated salutations and greeting-only examples.
            opening = reason == "greeting" and not session
            tone = str(beh.get("greeting_tone" if opening else "friendly_tone") or "")
            return tone + ("\n【当前会话语气：温和，优先于默认毒舌人格、示例及临时攻击档】"
                           "对方以友好方式开始这段对话，整段交流延续温和、自然、平等的群友口吻。"
                           "后续短回答、聊游戏、换话题、@、引用或戳一戳都不是邀请你开骂。"
                           "先接住当前意思，可以关心、询问或轻松接梗，但不贬低、阴阳怪气或攻击人格；"
                           "即使对方一句话听着冲，也可平和设边界，不擅自切回毒舌。"
                           "不必每条重新问好；补发也必须保持这一语气，不为凑条数硬怼。")
        if reason == "attack":
            if self._topic(text) == "game":
                return str(beh.get("attack_game_tone") or "")
            return str(beh.get("attack_tone") or "")
        return ""

    def _pick(self, key: str, value: Any) -> str:
        """值可以是字符串或数组；数组随机取，且不跟上次重复。"""
        choices = value if isinstance(value, (list, tuple)) else [value]
        choices = [str(c) for c in choices if c]
        if not choices:
            return ""
        if len(choices) == 1:
            return choices[0]
        last = self._recent_pick.get(key)
        pool = [c for c in choices if c != last] or choices
        choice = random.choice(pool)
        self._recent_pick[key] = choice
        return choice

    def _fixed_reply(self, text: str) -> str | None:
        """命中预设关键词就返回固定句子（不走模型）。"""
        mapping = self.beh.get("fixed_replies") or {}
        if not text:
            return None
        for key, reply in mapping.items():
            if key and str(key) in text and reply:
                return self._pick(str(key), reply) or None
        return None

    def _image_variant(self, val: Any, key: str = "") -> tuple[str, str] | None:
        """把配置里的一条图片规则解析成 (文字, 图片路径)。

        支持三种写法：
          "a.jpg"                                   → 只有图
          {"text": "...", "image": "a.jpg"}         → 固定一条
          {"variants": [{"text","image","weight"}]} → 按权重随机，且不连着重复
        """
        if isinstance(val, str):
            return ("", val) if val else None
        if isinstance(val, dict):
            variants = val.get("variants")
            if isinstance(variants, list):
                pool = [v for v in variants if isinstance(v, dict) and v.get("image")]
                if not pool:
                    return None
                # 排除上一次发过的那张，避免相邻两次一样
                last = self._last_image_pick.get(key)
                if len(pool) > 1 and last:
                    filtered = [v for v in pool if str(v["image"]) != last]
                    if filtered:
                        pool = filtered
                weights = [max(0.0, float(v.get("weight", 1) or 0)) for v in pool]
                total = sum(weights)
                if total <= 0:
                    chosen = random.choice(pool)
                else:
                    r = random.uniform(0.0, total)
                    acc = 0.0
                    chosen = pool[-1]
                    for v, w in zip(pool, weights):
                        acc += w
                        if r <= acc:
                            chosen = v
                            break
                if key:
                    self._last_image_pick[key] = str(chosen["image"])
                return str(chosen.get("text") or ""), str(chosen["image"])
            if val.get("image"):
                return str(val.get("text") or ""), str(val["image"])
        return None

    def _image_reply(self, text: str) -> tuple[str, str] | None:
        """命中图片关键词，返回一条 (文字, 图片路径)。"""
        mapping = self.beh.get("image_replies") or {}
        if not text:
            return None
        for key, val in mapping.items():
            if key and str(key) in text:
                picked = self._image_variant(val, str(key))
                if picked:
                    return picked
        return None

    def _combo_reply(self, text: str) -> tuple[str, str] | None:
        """组合规则：keywords 全部出现时，返回这条规则的 (文字, 图片)。"""
        rules = self.beh.get("combo_replies") or []
        if not text:
            return None
        for idx, rule in enumerate(rules):
            if not isinstance(rule, dict):
                continue
            keys = [str(k) for k in (rule.get("keywords") or []) if k]
            if not keys or not all(k in text for k in keys):
                continue
            picked = self._image_variant(rule, f"combo:{idx}")
            if picked:
                return picked
        return None

    def _image_uri(self, path: str) -> str:
        """把本地图片转成 base64://，避免 NapCat 读不到相对路径。"""
        if path in self._image_cache:
            return self._image_cache[path]
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        uri = "base64://" + base64.b64encode(p.read_bytes()).decode()
        self._image_cache[path] = uri
        return uri

    async def _send_image(
        self, group_id: int, text: str, image: str, message_id: Any, user_id: int
    ) -> bool:
        """Keep configured image replies; failed uploads never become conversation history."""
        if self.cfg["runtime"].get("dry_run"):
            log.info("[dry-run][群 %s] 图片回复：%s (%s)", group_id, text, image)
            return False
        try:
            segments: list[dict[str, Any]] = []
            if self._should_quote(group_id) and message_id:
                segments.append({"type": "reply", "data": {"id": str(message_id)}})
            if self.beh["prefix_at"]:
                segments.extend([{"type": "at", "data": {"qq": str(user_id)}}, {"type": "text", "data": {"text": " "}}])
            if text:
                segments.append({"type": "text", "data": {"text": text}})
            segments.append({"type": "image", "data": {"file": self._image_uri(image)}})
            async with self._delivery_lock:
                await self._wait_delivery_gap()
                current = self._active_turn.get(group_id)
                if current and self._superseded(current):
                    return False
                data = await self.client.send_action("send_group_msg", {"group_id": group_id, "message": segments})
                self._record_delivery(group_id, user_id, text or "[图片]", (data or {}).get("message_id"), "image")
            log.info("[群 %s] %s：%s [图片]", group_id, self.name, text)
            return True
        except Exception as exc:
            log.warning("图片发送失败：%s", exc)
            return False

    def _decide(self, group_id: int, user_id: int, text: str, at_self: bool, *, reply_to_self: bool = False, is_poke: bool = False) -> str | None:
        beh = self.beh
        if self._is_muted(group_id, user_id):
            return None
        clean = text.strip()
        blocked = self.safety.get("blocked_input") or []
        if any(word and word in clean for word in blocked):
            return None
        if not clean and not at_self:
            return None

        lowered = clean.lower()
        now = time.monotonic()
        # A short uninterrupted session is a continuation; older turns need cues.
        in_banter = self._related_followup(group_id, user_id, text)
        called = self._calls_bot(text)
        direct_evidence = (at_self and beh["at_reply"]) or reply_to_self or called or is_poke
        taunt = any(
            k and str(k).lower() in lowered for k in (beh["taunt_keywords"] or [])
        )

        reason: str | None = None
        direct = False
        if direct_evidence and self._is_greeting(clean):
            reason, direct = "greeting", True
        elif is_poke:
            reason, direct = "poke", True
        elif at_self and beh["at_reply"]:
            reason, direct = "at", True
        elif reply_to_self:
            reason, direct = "reply", True
        elif called:
            reason, direct = "keyword", True
        elif in_banter and taunt:
            reason, direct = "taunt", True
        elif in_banter and random.random() < float(beh["followup_rate"]):
            reason, direct = "followup", True
        elif beh["question_reply"] and re.search(r"[?？]", clean):
            if random.random() < float(beh["question_rate"]):
                reason = "question"
        elif self._has_topic_rates():
            reason = self._roll_topic_reason(clean)
        elif random.random() < float(beh["random_rate"]):
            reason = "random"
        if reason is None:
            return None

        # Confirmed implicit continuations wait in the group worker. Ordinary
        # unsolicited/direct messages keep their existing cooldown policy.
        if reason not in ("followup", "taunt"):
            if now - self._last_any < float(beh["min_gap"]):
                return None
            cooldown = float(beh["direct_cooldown"] if direct else beh["group_cooldown"])
            if now - self._last_group.get(group_id, 0.0) < cooldown:
                return None
            user_cd = float(beh["direct_user_cooldown"] if direct else beh["user_cooldown"])
            if now - self._last_user.get((group_id, user_id), 0.0) < user_cd:
                return None
        if beh.get("model_reply_planning", True):
            if not self.limiter.allow_many(self._second_reply_limits(group_id), consume=False):
                return None
        else:
            if not self.limiter.allow(("group", group_id), int(beh["per_group_per_minute"])):
                return None
            if not self.limiter.allow(("global",), int(beh["global_per_minute"])):
                return None
        return reason

    # ---- 发送 ----

    def _group_load(self, group_id: int) -> int:
        """最近 60 秒这个群有多少条消息。"""
        dq = self._activity.get(group_id)
        if not dq:
            return 0
        cutoff = time.monotonic() - 60
        while dq and dq[0] < cutoff:
            dq.popleft()
        return len(dq)

    def _should_quote(self, group_id: int) -> bool:
        """默认不引用；只有群里刷得很快时才引用，避免插话看不出来是回谁。"""
        mode = str(self.beh.get("quote_mode") or "auto")
        if mode == "always":
            return True
        if mode == "never":
            return False
        threshold = int(self.beh.get("quote_busy_per_minute", 12))
        return self._group_load(group_id) >= threshold

    async def _wait_delivery_gap(self) -> None:
        delay = max(0.0, float(self.beh["min_gap"])) - (time.monotonic() - self._last_any)
        if delay > 0:
            await asyncio.sleep(delay)

    def _record_delivery(self, group_id: int, user_id: int, text: str, new_id: Any, reason: str) -> None:
        """Commit conversation/cooldown state only after OneBot acknowledges delivery."""
        now = time.monotonic()
        self._last_any = self._last_awake = now
        self._last_group[group_id] = now
        self._last_user[(group_id, user_id)] = now
        self._last_reply[group_id] = (user_id, now)
        turn = self._active_turn.get(group_id)
        # Only a successfully delivered MODEL main reply starts/refreshes tone.
        # Fixed/image/wake presets and continuations keep, but never renew it.
        key = (group_id, user_id)
        captured = self._turn_tones.get(group_id)
        if reason == "mute" or (turn and turn.user_id == user_id
                                 and self._is_tone_session_end(turn.message.text)):
            self._friendly_sessions.pop(key, None)
        elif reason not in ("second", "continuation", "fixed", "image", "wake"):
            current = turn is None or not self._superseded(turn)
            if current and (captured == (user_id, "friendly") or reason in ("greeting", "friendly")
                            or self._session_tone(group_id, user_id) == "friendly"):
                self._friendly_sessions[key] = now
                # Bound long-running public-group bookkeeping without persisting IDs.
                if len(self._friendly_sessions) > 4096:
                    window = max(0.0, float(self.beh.get("followup_window", 120)))
                    for old_key, stamp in list(self._friendly_sessions.items()):
                        if now - stamp > window:
                            self._friendly_sessions.pop(old_key, None)
                    while len(self._friendly_sessions) > 4096:
                        self._friendly_sessions.pop(next(iter(self._friendly_sessions)))
        previous = self._dialogues.get(group_id)
        followups = previous.followups + 1 if previous and previous.user_id == user_id and reason in ("followup", "taunt") else 0
        if reason in ("second", "continuation"):
            # An addendum is not a new user turn. Preserve the reply window and
            # followup count, while retaining its words for topic matching.
            if previous and previous.user_id == user_id:
                previous.bot_text += "\n" + text
        elif reason != "mute" and not (turn and self._is_tone_session_end(turn.message.text)):
            self._dialogues[group_id] = DialogueState(
                user_id, turn.message.text if turn and turn.user_id == user_id else "", text, now, followups)
        else:
            self._dialogues.pop(group_id, None)
        bot_name = self.client.nickname or self.name
        message_id = str(new_id) if new_id is not None else ""
        self.store.add(group_id, Msg(
            bot_name, text, True, time.time(), sender_id=str(self.client.self_id),
            nickname=bot_name, message_id=message_id,
            reply_to_id=turn.message_id if turn and turn.user_id == user_id else "",
            reply_to_sender_id=str(user_id),
        ))
        if message_id:
            self._msg_index[(group_id, message_id)] = IndexedMessage(
                group_id, self.client.self_id, bot_name, text, time.time(), True)

    async def _send(
        self, group_id: int, text: str, message_id: Any, user_id: int, reason: str
    ) -> bool:
        if self.cfg["runtime"].get("dry_run"):
            log.info("[dry-run][群 %s] %s：%s  （触发=%s）", group_id, self.name, text, reason)
            return False
        segments: list[dict[str, Any]] = []
        if self._should_quote(group_id) and message_id:
            segments.append({"type": "reply", "data": {"id": str(message_id)}})
        if self.beh["prefix_at"]:
            segments.extend([{"type": "at", "data": {"qq": str(user_id)}}, {"type": "text", "data": {"text": " "}}])
        segments.append({"type": "text", "data": {"text": text}})
        try:
            async with self._delivery_lock:
                await self._wait_delivery_gap()
                current = self._active_turn.get(group_id)
                if current and self._superseded(current):
                    return False
                if reason in ("followup", "taunt") and (not current or not self._followup_is_current(current)):
                    log.info("[群 %s] 取消隐式续聊发送：已有人插话或对象转移", group_id)
                    return False
                if reason in ("second", "continuation"):
                    if not current or not self._second_reply_ready(current):
                        return False
                    if self.store.trailing_bot_count(group_id) >= int(self.beh["max_chain"]):
                        return False
                    if not self.limiter.allow_many(self._second_reply_limits(group_id)):
                        log.info("[群 %s] 续发取消：达到每分钟限流", group_id)
                        return False
                elif self.beh.get("model_reply_planning", True) and reason not in ("fixed", "mute", "wake"):
                    if not self.limiter.allow_many(self._second_reply_limits(group_id)):
                        log.info("[群 %s] 模型回复取消：达到每分钟限流", group_id)
                        return False
                data = await self.client.send_action("send_group_msg", {"group_id": group_id, "message": segments})
                self._record_delivery(group_id, user_id, text, (data or {}).get("message_id"), reason)
            log.info("[群 %s] %s：%s  （触发=%s）", group_id, self.name, text, reason)
            return True
        except Exception as exc:
            log.warning("发送失败：%s", exc)
            return False


    
####入口


def main() -> int:
    parser = argparse.ArgumentParser(description="QQ 群嘴替机器人（OneBot v11 + DeepSeek）")
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只打印不发送")
    parser.add_argument("--debug", action="store_true", help="输出调试日志")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dry_run:
        cfg["runtime"]["dry_run"] = True
    if args.debug:
        cfg["runtime"]["log_level"] = "DEBUG"
    setup_logging(cfg)

    log.info(
        "启动：%s（风格 %s） dry_run=%s",
        cfg["persona"]["name"],
        cfg["persona"]["style"],
        cfg["runtime"]["dry_run"],
    )
    if not cfg["llm"]["api_key"]:
        log.warning("没有模型 key，机器人只会连上但不会说话")

    bot = Bot(cfg)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        log.info("已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
