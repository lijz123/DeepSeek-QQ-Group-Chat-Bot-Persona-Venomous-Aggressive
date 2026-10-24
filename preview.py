# -*- coding: utf-8 -*-
"""不上 QQ 直接试人设。

    python preview.py "这英雄怎么玩"
    python preview.py "你是不是机器人" 小明
"""
from __future__ import annotations

import asyncio
import json
import sys

from bot import Brain, load_config, setup_logging


async def main() -> int:
    cfg = load_config("config.json")
    setup_logging(cfg)
    if not cfg["llm"]["api_key"]:
        print("先设置 DEEPSEEK_API_KEY，或在 config.json 里填 llm.api_key")
        return 1

    text = sys.argv[1] if len(sys.argv) > 1 else "这英雄怎么玩"
    speaker = sys.argv[2] if len(sys.argv) > 2 else "群友"

    brain = Brain(cfg["llm"], cfg["safety"], cfg["persona"], int(cfg["behavior"]["max_chars"]))
    try:
        transcript = json.dumps({"role": "group_member",
            "sender": {"id": "preview-user", "display_name": speaker},
            "content": text}, ensure_ascii=False)
        answer = await brain.reply(0, transcript, speaker, False)
        print(f"{cfg['persona']['name']}：{answer or '（没生成，看上面的日志）'}")
    finally:
        await brain.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
