# -*- coding: utf-8 -*-
"""不上 QQ 直接试人设。

    python preview.py "这英雄怎么玩"
    python preview.py "你是不是机器人" 小明
"""
from __future__ import annotations

import asyncio
import sys
import time

from bot import Brain, ContextStore, Msg, load_config, setup_logging


async def main() -> int:
    cfg = load_config("config.json")
    setup_logging(cfg)
    if not cfg["llm"]["api_key"]:
        print("请先设置环境变量 DEEPSEEK_API_KEY")
        return 1

    text = sys.argv[1] if len(sys.argv) > 1 else "这英雄怎么玩"
    speaker = sys.argv[2] if len(sys.argv) > 2 else "群友"

    brain = Brain(cfg["llm"], cfg["safety"], cfg["persona"], int(cfg["behavior"]["max_chars"]))
    try:
        store = ContextStore(2, 900)
        store.add(0, Msg(speaker, text, False, time.time(), nickname=speaker))
        transcript = store.transcript(0)
        answer = await brain.reply(0, transcript, speaker, False)
        print(f"{cfg['persona']['name']}：{answer or '（没生成，看上面的日志）'}")
    finally:
        await brain.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
