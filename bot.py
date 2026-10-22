# -*- coding: utf-8 -*-
"""QQ 群「deepseek」机器人。

链路：QQ ← NapCat / LLOneBot(OneBot v11) ←WebSocket→ 本程序 → DeepSeek(OpenAI 兼容)

用法：
    set DEEPSEEK_API_KEY=sk-xxx
    python bot.py --config config.json
"""
from __future__ import annotations

import argparse
import asyncio
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

from persona import build_system_prompt, build_user_prompt, sanitize_reply

log = logging.getLogger("dafish")


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

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
        "timeout": 30,
        "retries": 1,
    },
    "persona": {"name": "聊天机器人", "style": "sharp", "extra": ""},
    "behavior": {
        "at_reply": True,
        "keyword_reply": True,
        "keywords": ["聊天机器人", "机器人","deepseek","Deepseek","ds"],
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
        "per_group_per_minute": 6,
        "global_per_minute": 25,
        "history_size": 14,
        "history_ttl": 900,
        "quote_reply": True,
        "prefix_at": False,
        "reply_poke": True,
        "ignore_at_all": True,
        "max_chain": 3,
        "max_chars": 120,
        "followup_window": 90,
        "followup_rate": 0.85,
        "direct_user_cooldown": 3,
        "taunt_keywords": [
            "你妈", "尼玛", "妈死", "傻逼", "煞笔", "傻b", "废物",
            "垃圾", "滚", "草", "cnm", "sb",
        ],
        "mute_keywords": ["机器人闭嘴"],
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


# --------------------------------------------------------------------------
# 消息解析
# --------------------------------------------------------------------------

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


def parse_message(
    message: Any, self_id: int, ignore_at_all: bool = True
) -> tuple[str, bool, bool, str]:
    """返回 (纯文本, 是否@了我, 是否含图片/表情, 引用的消息id)。兼容数组与 CQ 字符串。"""
    if isinstance(message, str):
        segments = _cq_segments(message)
    elif isinstance(message, list):
        segments = [s for s in message if isinstance(s, dict)]
    else:
        segments = []

    texts: list[str] = []
    at_self = False
    media = False
    reply_id = ""
    for seg in segments:
        typ = str(seg.get("type") or "")
        data = seg.get("data") or {}
        if typ == "text":
            texts.append(str(data.get("text") or ""))
        elif typ == "at":
            qq = str(data.get("qq") or "")
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
    return "".join(texts).strip(), at_self, media, reply_id


# --------------------------------------------------------------------------
# 上下文 / 限流
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# OneBot v11 客户端
# --------------------------------------------------------------------------

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

    async def reply(
        self,
        group_id: int,
        transcript: str,
        speaker: str,
        is_at: bool,
        tone: str = "",
        focus_note: str = "",
    ) -> str | None:
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
                    str(self.persona.get("name") or "聊天机器人"),
                    str(self.persona.get("style") or "sharp"),
                    str(self.persona.get("extra") or ""),
                    tone,
                ),
            },
            {"role": "user", "content": build_user_prompt(transcript, speaker, is_at, focus_note)},
        ]
        recent = self._recent.setdefault(group_id, deque(maxlen=5))
        base_temp = float(self.cfg.get("temperature", 1.1))
        attempts = int(self.cfg.get("retries", 1)) + 2

        for attempt in range(attempts):
            temperature = min(1.6, base_temp + 0.15 * attempt)
            raw = await self._call(messages, temperature)
            if not raw:
                continue
            out = sanitize_reply(
                raw,
                name=str(self.persona.get("name") or ""),
                blocked=list(self.safety.get("blocked_output") or []),
                fallback=str(self.safety.get("fallback_reply") or "君子谨言慎行"),
                max_chars=self.max_chars,
            )
            if not out:
                continue
            if out in recent:
                log.debug("重复回复，重试：%s", out)
                messages[-1]["content"] += "\n（换一种说法，别和你上一条一样）"
                continue
            recent.append(out)
            return out
        return None

    async def _call(self, messages: list[dict], temperature: float) -> str | None:
        assert self._session is not None
        url = str(self.cfg["base_url"]).rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg["model"],
            "messages": messages,
            "temperature": round(temperature, 2),
            "max_tokens": int(self.cfg.get("max_tokens", 100)),
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


# --------------------------------------------------------------------------
# 机器人主体
# --------------------------------------------------------------------------

class Bot:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.beh = cfg["behavior"]
        self.safety = cfg["safety"]
        self.name = str(cfg["persona"]["name"] or "聊天机器人")
        self.store = ContextStore(int(self.beh["history_size"]), float(self.beh["history_ttl"]))
        self.limiter = WindowLimiter()
        self.brain = Brain(cfg["llm"], self.safety, cfg["persona"], int(self.beh["max_chars"]))
        self.client = OneBotClient(cfg["onebot"], self.on_event)
        self._last_group: dict[int, float] = {}
        self._last_user: dict[int, float] = {}
        self._last_reply: dict[int, tuple[int, float]] = {}
        self._muted: set[tuple[int, int]] = set()
        self._msg_index: dict[str, tuple[str, str, float]] = {}
        self._activity: dict[int, deque[float]] = {}
        self._recent_pick: dict[str, str] = {}
        self._last_awake = time.monotonic()
        self._last_any = 0.0
        self._seen: dict[str, float] = {}
        self._inflight = False
        self._whitelist = {int(g) for g in (cfg["groups"]["whitelist"] or [])}
        self._group_blacklist = {int(g) for g in (cfg["groups"]["blacklist"] or [])}
        self._user_blacklist = {int(u) for u in (cfg["users"]["blacklist"] or [])}

    async def run(self) -> None:
        try:
            await self.client.run()
        finally:
            await self.brain.close()

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

    def _is_dup(self, message_id: Any) -> bool:
        if message_id is None:
            return False
        key = str(message_id)
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
        group_id = int(event.get("group_id") or 0)
        user_id = int(event.get("user_id") or 0)
        if not group_id or not user_id or user_id == self.client.self_id:
            return
        if not self._group_allowed(group_id) or user_id in self._user_blacklist:
            return
        if self._is_dup(event.get("message_id")):
            return

        # 记录群活跃度，用来决定要不要用引用回复
        load = self._activity.setdefault(group_id, deque())
        load.append(time.monotonic())
        while load and load[0] < time.monotonic() - 60:
            load.popleft()

        sender = event.get("sender") or {}
        nickname = str(sender.get("nickname") or "")
        card = str(sender.get("card") or "")
        raw_name = str(card or nickname or f"用户{user_id}")
        # 昵称如果本身就是一句话（「这个名字是一句话？」），它会和正文混在一起被模型吃进去。
        # 这种名字不放进上下文，直接换成中性编号。
        speaker = raw_name if self._is_safe_name(raw_name) else f"用户{user_id}"
        text, at_self, media, reply_id = parse_message(
            event.get("message"), self.client.self_id, bool(self.beh["ignore_at_all"])
        )
        stored = text or ("[表情]" if media else "")

        # 引用回复：把被引用的原话和原发言人找出来，别混进当前发言人的正文
        quote = ""
        if reply_id:
            prev = self._msg_index.get(reply_id)
            if prev:
                quote = f"{prev[0]}：{prev[1][:60]}"
            else:
                # 引用的消息太早，本地没缓存到；至少告诉模型这是引用，不是他打的
                quote = "更早的一条消息（原话没记录到）"

        now_ts = time.time()
        self._msg_index[str(event.get("message_id"))] = (speaker, stored, now_ts)
        for k in [k for k, v in self._msg_index.items() if now_ts - v[2] > 600]:
            self._msg_index.pop(k, None)

        self.store.add(
            group_id,
            Msg(
                speaker,
                stored,
                False,
                now_ts,
                quote,
                sender_id=str(user_id),
                nickname=nickname,
                card=card,
            ),
        )

        # 被这个人拉黑过：他说什么都不理，除非他再 @ 我一次
        if self._is_muted(group_id, user_id):
            if not at_self:
                return
            self._unmute(group_id, user_id)
            log.info("[群 %s] %s 重新 @ 了，解除静音", group_id, user_id)

        # 「机器人闭嘴」：从此不再理这个人，直到他下次 @ 我
        if self._apply_mute(group_id, user_id, text):
            await self._send(
                group_id, self._mute_line(), event.get("message_id"), user_id, "mute"
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

        reason = self._decide(group_id, user_id, text, at_self)
        if not reason:
            return
        if self.store.trailing_bot_count(group_id) >= int(self.beh["max_chain"]):
            log.debug("连续发言过多，跳过")
            return

        self._inflight = True
        try:
            tone = self._tone_for(reason, text)
            focus_note = self._focus_note(reason)
            answer = await self.brain.reply(
                group_id,
                self.store.transcript(group_id),
                speaker,
                at_self,
                tone=tone,
                focus_note=focus_note,
            )
        finally:
            self._inflight = False
        if not answer:
            return
        await self._send(group_id, answer, event.get("message_id"), user_id, reason)

    async def _on_notice(self, event: dict) -> None:
        if not self.beh.get("reply_poke"):
            return
        if event.get("notice_type") != "notify" or event.get("sub_type") != "poke":
            return
        if str(event.get("target_id")) != str(self.client.self_id):
            return
        group_id = int(event.get("group_id") or 0)
        user_id = int(event.get("user_id") or 0)
        if not group_id or not user_id or user_id == self.client.self_id:
            return
        if not self._group_allowed(group_id) or user_id in self._user_blacklist:
            return
        if self._is_muted(group_id, user_id):
            return
        now = time.monotonic()
        if now - self._last_any < float(self.beh["min_gap"]):
            return
        if not self.limiter.allow(("group", group_id), int(self.beh["per_group_per_minute"])):
            return
        if self._inflight:
            return
        self._inflight = True
        try:
            answer = await self.brain.reply(
                group_id, self.store.transcript(group_id), f"群友{user_id}", True
            )
        finally:
            self._inflight = False
        if answer:
            await self._send(group_id, answer, None, user_id, "poke")

    # ---- 触发判定 ----

    def _apply_mute(self, group_id: int, user_id: int, text: str) -> bool:
        """命中「机器人闭嘴」→ 永久无视此人，直到他下次 @ 我。返回 True 表示要回一句。"""
        words = self.beh.get("mute_keywords") or []
        if not text or not any(w and str(w) in text for w in words):
            return False
        if self._is_muted(group_id, user_id):
            return False
        self._muted.add((group_id, user_id))
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

        「早上好啊小机器人」「早上好啊deepseek」都算；「你好菜」不算。
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

    def _tone_for(self, reason: str, text: str) -> str:
        """按触发原因给这一条挑语气覆盖，空字符串表示用默认人设。"""
        beh = self.beh
        if reason == "greeting":
            return str(beh.get("greeting_tone") or "")
        if reason == "friendly":
            return str(beh.get("friendly_tone") or "")
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

    def _wake_line(self) -> str:
        lines = self.beh.get("wake_replies") or [self.beh.get("wake_reply") or ""]
        return self._pick("wake", lines) or "醒了醒了，别喊了"

    def _decide(self, group_id: int, user_id: int, text: str, at_self: bool) -> str | None:
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
        last = self._last_reply.get(group_id)
        # 刚接过这个人的话，90 秒内他再说话，视为继续对线
        in_banter = (
            last is not None
            and last[0] == user_id
            and now - last[1] <= float(beh["followup_window"])
        )
        taunt = any(
            k and str(k).lower() in lowered for k in (beh["taunt_keywords"] or [])
        )

        reason: str | None = None
        direct = False
        if self._is_greeting(clean):
            reason, direct = "greeting", True
        elif at_self and beh["at_reply"]:
            reason, direct = "at", True
        elif beh["keyword_reply"] and any(
            k and str(k).lower() in lowered for k in (beh["keywords"] or [])
        ):
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

        if now - self._last_any < float(beh["min_gap"]):
            return None
        cooldown = float(beh["direct_cooldown"] if direct else beh["group_cooldown"])
        if now - self._last_group.get(group_id, 0.0) < cooldown:
            return None
        user_cd = float(beh["direct_user_cooldown"] if direct else beh["user_cooldown"])
        if now - self._last_user.get(user_id, 0.0) < user_cd:
            return None
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

    async def _send(
        self, group_id: int, text: str, message_id: Any, user_id: int, reason: str
    ) -> None:
        now = time.monotonic()
        self._last_any = now
        self._last_awake = now
        self._last_group[group_id] = now
        self._last_user[user_id] = now
        self._last_reply[group_id] = (user_id, now)
        # 群里看到的昵称优先，人设名只在没拿到登录信息时兜底
        bot_name = self.client.nickname or self.name
        self.store.add(
            group_id,
            Msg(
                bot_name,
                text,
                True,
                time.time(),
                sender_id=str(self.client.self_id) if self.client.self_id else "",
                nickname=bot_name,
            ),
        )

        if self.cfg["runtime"].get("dry_run"):
            log.info("[dry-run][群 %s] %s：%s  （触发=%s）", group_id, self.name, text, reason)
            return

        segments: list[dict[str, Any]] = []
        if self._should_quote(group_id) and message_id:
            segments.append({"type": "reply", "data": {"id": str(message_id)}})
        if self.beh["prefix_at"]:
            segments.append({"type": "at", "data": {"qq": str(user_id)}})
            segments.append({"type": "text", "data": {"text": " "}})
        segments.append({"type": "text", "data": {"text": text}})
        try:
            data = await self.client.send_action(
                "send_group_msg", {"group_id": group_id, "message": segments}
            )
            # 自己的消息也要进索引，别人引用它时才能还原原话
            new_id = (data or {}).get("message_id")
            if new_id is not None:
                self._msg_index[str(new_id)] = (bot_name, text, time.time())
            log.info("[群 %s] %s：%s  （触发=%s）", group_id, self.name, text, reason)
        except Exception as exc:  # noqa: BLE001
            log.warning("发送失败：%s", exc)


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------

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
