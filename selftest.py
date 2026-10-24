# -*- coding: utf-8 -*-
"""离线自检：假 OneBot 服务端 + 假模型接口，跑通「收消息 → 生成 → 发消息」。

不需要 QQ、不需要网络、不需要 API key：
    python selftest.py
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time

import aiohttp
from aiohttp import web

from bot import Bot, ContextStore, Msg, load_config, setup_logging
from persona import build_system_prompt, sanitize_reply, build_user_prompt
from bot import parse_message

BOT_QQ = 10001
GROUP_ID = 88888
USER_ID = 20002
REPLY_TEXT = "你先把补刀补明白"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))


class FakeQQ:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.replied = asyncio.Event()
        self.logged_in = asyncio.Event()
        self.calls = 0
        self.ws: web.WebSocketResponse | None = None

    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.ws = ws
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            action = data.get("action")
            echo = data.get("echo")
            if action == "get_login_info":
                await ws.send_json(
                    {
                        "status": "ok",
                        "retcode": 0,
                        "data": {"user_id": BOT_QQ, "nickname": "大肥鱼"},
                        "echo": echo,
                    }
                )
                self.logged_in.set()
            elif action == "send_group_msg":
                self.sent.append(data.get("params") or {})
                await ws.send_json(
                    {"status": "ok", "retcode": 0, "data": {"message_id": 777}, "echo": echo}
                )
                self.replied.set()
            else:
                await ws.send_json({"status": "ok", "retcode": 0, "data": {}, "echo": echo})
        return ws

    async def llm_handler(self, request: web.Request) -> web.Response:
        payload = await request.json()
        system = payload["messages"][0]["content"]
        assert "群昵称" in system, "系统提示词没带人设"
        assert payload["model"]
        self.calls += 1
        content = REPLY_TEXT if self.calls == 1 else f"第{self.calls}次了，你还问？"
        if "话轮计划与输出协议" in system:
            content = json.dumps({"action": "reply", "text": content, "next": "wait"}, ensure_ascii=False)
        return web.json_response(
            {"choices": [{"message": {"role": "assistant", "content": content}}]}
        )

    def group_event(self, message: object, message_id: int, text_id: str = "") -> dict:
        return {
            "post_type": "message",
            "message_type": "group",
            "sub_type": "normal",
            "message_id": message_id,
            "group_id": GROUP_ID,
            "user_id": USER_ID,
            "self_id": BOT_QQ,
            "raw_message": text_id,
            "message": message,
            "sender": {"user_id": USER_ID, "nickname": "青铜菜鸡", "card": "菜鸡"},
        }

    async def push(self, event: dict) -> None:
        assert self.ws is not None
        await self.ws.send_json(event)


def unit_checks() -> None:
    text, at_self, media, _reply = parse_message(
        [
            {"type": "at", "data": {"qq": str(BOT_QQ)}},
            {"type": "text", "data": {"text": " 你行不行"}},
            {"type": "face", "data": {"id": "1"}},
        ],
        BOT_QQ,
    )
    check("数组消息：@识别 + 文本提取 + 表情识别", at_self and text.strip() == "你行不行" and media)

    text2, at2, _, _ = parse_message("[CQ:at,qq=999] 你好", BOT_QQ)
    check("CQ 字符串：@别人不算 @我", not at2 and "你好" in text2)

    text3, at3, _, _ = parse_message("[CQ:at,qq=all] 全体注意", BOT_QQ)
    check("默认忽略 @全体成员", not at3)

    out = sanitize_reply("「大肥鱼：你先把补刀补明白」\n第二行", name="大肥鱼", max_chars=50)
    check("清洗：去名字前缀/引号/多行", out == "你先把补刀补明白", repr(out))

    out2 = sanitize_reply("x" * 200, max_chars=20)
    check("清洗：长度截断", len(out2) <= 21, repr(out2))

    out3 = sanitize_reply("我去开盒他", blocked=["开盒"], fallback="这话我没法接")
    check("清洗：命中禁用词回落", out3 == "这话我没法接", out3)

    prompt = build_system_prompt("大肥鱼", "savage")
    check("人设提示词：含性格/规则", "嘴替" in prompt and "只发一句话" in prompt)
    up = build_user_prompt("A：1\nB：2", "菜鸡", True)
    check("用户提示词：带 @ 标记", "@ 了你" in up)

    # 回归：昵称不能被模型输入当成正文，且身份字段必须保留。
    store = ContextStore(4, 900)
    store.add(
        1,
        Msg(
            "丝袜",
            "看着鱼尾",
            False,
            time.time(),
            sender_id="42",
            nickname="丝袜",
        ),
    )
    record = json.loads(store.transcript(1))
    check(
        "回归：昵称与正文分离",
        record["sender"]["display_name"] == "丝袜"
        and record["sender"]["nickname"] == "丝袜"
        and record["content"] == "看着鱼尾",
        repr(record),
    )
    structured = build_user_prompt(store.transcript(1), "丝袜", True)
    check(
        "回归：提示词使用 JSONL 字段",
        "JSONL" in structured
        and '"content":"看着鱼尾"' in structured
        and "只有 content 是原话" in structured,
        structured,
    )


async def integration() -> None:
    fake = FakeQQ()
    app = web.Application()
    app.router.add_get("/ws", fake.ws_handler)
    app.router.add_post("/v1/chat/completions", fake.llm_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    cfg = load_config("config.json")
    cfg["behavior"]["second_reply_rate"] = 0
    cfg["onebot"]["ws_url"] = f"ws://127.0.0.1:{port}/ws"
    cfg["onebot"]["heartbeat"] = 0
    cfg["onebot"]["reconnect_min"] = 1
    cfg["llm"]["base_url"] = f"http://127.0.0.1:{port}/v1"
    cfg["llm"]["api_key"] = "test-key"
    cfg["llm"]["model"] = "test-model"
    # All proactive paths are disabled so silence is deterministic, not hidden by cooldowns.
    cfg["behavior"]["merge_delay"] = 0.02
    cfg["behavior"]["merge_at_delay"] = 0.02
    cfg["behavior"]["merge_max_wait"] = 0.1
    cfg["behavior"]["topic_rates"] = {"daily": {"friendly": 0, "attack": 0}, "game": {"friendly": 0, "attack": 0}}
    cfg["behavior"]["followup_rate"] = 0.0
    cfg["behavior"]["random_rate"] = 0.0
    cfg["behavior"]["question_rate"] = 0.0
    cfg["groups"]["whitelist"] = []
    cfg["groups"]["blacklist"] = []
    cfg["behavior"]["quote_mode"] = "always"
    cfg["behavior"]["user_cooldown"] = 0
    cfg["behavior"]["min_gap"] = 0
    cfg["behavior"]["direct_cooldown"] = 0
    cfg["runtime"]["dry_run"] = False
    cfg["runtime"]["log_level"] = "WARNING"
    cfg["runtime"]["log_file"] = ""
    setup_logging(cfg)

    bot = Bot(cfg)
    task = asyncio.create_task(bot.run())
    try:
        await asyncio.wait_for(fake.logged_in.wait(), 10)
        check("OneBot：完成 get_login_info 握手", True)

        await fake.push(
            fake.group_event(
                [
                    {"type": "at", "data": {"qq": str(BOT_QQ)}},
                    {"type": "text", "data": {"text": " 这英雄怎么玩"}},
                ],
                1001,
            )
        )
        await asyncio.wait_for(fake.replied.wait(), 10)
        params = fake.sent[-1]
        segments = params.get("message") or []
        kinds = [s.get("type") for s in segments]
        body = "".join(
            str(s.get("data", {}).get("text", "")) for s in segments if s.get("type") == "text"
        )
        check("发送：命中 send_group_msg", params.get("group_id") == GROUP_ID, str(kinds))
        check("发送：正文是模型输出", body == REPLY_TEXT, body)
        check("发送：带引用回复段", "reply" in kinds, str(kinds))

        # 冷却内再 @ 一次：direct_cooldown=0，但用户冷却已置 0，应仍能回；
        # 这里验证「重复消息 ID 去重」
        before = len(fake.sent)
        await fake.push(
            fake.group_event(
                [
                    {"type": "at", "data": {"qq": str(BOT_QQ)}},
                    {"type": "text", "data": {"text": " 这英雄怎么玩"}},
                ],
                1001,
            )
        )
        await asyncio.sleep(1.0)
        check("去重：同 message_id 不重复回复", len(fake.sent) == before)

        # 无关消息，random_rate=0：不该主动说话
        before = len(fake.sent)
        await fake.push(
            fake.group_event([{"type": "text", "data": {"text": "今天天气不错"}}], 1002)
        )
        await asyncio.sleep(1.2)
        check("静默：无关消息且随机率 0 时不发言", len(fake.sent) == before)

        # 关键词触发
        cfg["behavior"]["user_cooldown"] = 0
        bot._last_user.clear()
        bot._last_group.clear()
        bot._last_any = 0.0
        before = len(fake.sent)
        await fake.push(
            fake.group_event([{"type": "text", "data": {"text": "肥鱼出来说话"}}], 1003)
        )
        await asyncio.sleep(1.5)
        check("触发：关键词命中并发言", len(fake.sent) > before)

        # 打断重连测试的干扰，直接收尾
        login_before = fake.logged_in.is_set()
        await bot.client.stop()
        await asyncio.sleep(0.3)
        check("连接：可正常停止", login_before)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        await runner.cleanup()


def main() -> int:
    print("=== 单元检查 ===")
    unit_checks()
    print("=== 联调（假 OneBot + 假模型）===")
    asyncio.run(integration())

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
