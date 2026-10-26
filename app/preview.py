# -*- coding: utf-8 -*-
"""不上 QQ 直接试人设。

    python app/preview.py "这英雄怎么玩"
    python app/preview.py "你是不是机器人" 小明
"""
from __future__ import annotations

import asyncio
import argparse
import json
import sys

from bot import Brain, load_config, setup_logging
from project_paths import PROJECT_ROOT


async def main() -> int:
    parser = argparse.ArgumentParser(description="不上 QQ 试回复；会真实调用所选模型并消耗额度")
    parser.add_argument("text", nargs="?", default="这英雄怎么玩")
    parser.add_argument("speaker", nargs="?", default="群友")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.json"))
    parser.add_argument("--provider", choices=("deepseek", "chatgpt_plan"))
    args = parser.parse_args()
    cfg = load_config(args.config, provider=args.provider)
    setup_logging(cfg)
    text, speaker = args.text, args.speaker

    brain = Brain(cfg["llm"], cfg["safety"], cfg["persona"], int(cfg["behavior"]["max_chars"]))
    try:
        if not brain._provider_ready():
            return 1
        transcript = json.dumps({"role": "group_member", "sender": {"id": "preview-user", "display_name": speaker},
                                 "content": text, "message_id": "preview-1", "event_kind": "message"}, ensure_ascii=False)
        answer = await brain.reply(0, transcript, speaker, False)
        print(f"{cfg['persona']['name']}：{answer or '（没生成，看上面的日志）'}")
    finally:
        await brain.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
