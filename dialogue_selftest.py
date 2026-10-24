# -*- coding: utf-8 -*-
"""聊天节奏/对象判断离线回归。不读取 config.json，不连接 QQ/HTTP，不需要密钥。

运行：python dialogue_selftest.py
"""
from __future__ import annotations

import asyncio
import copy
import json
import time
import unittest
from collections import defaultdict
from unittest.mock import patch

from bot import Bot, DEFAULTS, parse_message_details
from unittest.mock import AsyncMock
from bot import Brain
from persona import (ReplyPlan, parse_reply_plan, build_correction_note,
                     build_system_prompt, build_user_prompt, POLICY_CORRECTION,
                     CORE_IDENTITY, EXAMPLES, POLICY_PLAN_FORMAT, build_turn_focus,
                     POLICY_COHERENCE)


BOT_ID = 10001
GROUP = 88888
ALICE = 20002
BOB = 20003


class StubBrain:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.active: dict[int, int] = defaultdict(int)
        self.max_active: dict[int, int] = defaultdict(int)
        self.total_active = 0
        self.max_total = 0
        self.started = asyncio.Event()
        self.release: asyncio.Event | None = None
        self.delay = 0.0
        self.closed = False

    async def reply(self, group_id, transcript, speaker, is_at, **kwargs):
        self.calls.append({"group_id": group_id, "transcript": transcript,
                           "speaker": speaker, "is_at": is_at, **kwargs})
        self.active[group_id] += 1
        self.total_active += 1
        self.max_active[group_id] = max(self.max_active[group_id], self.active[group_id])
        self.max_total = max(self.max_total, self.total_active)
        self.started.set()
        try:
            if self.release is not None:
                await self.release.wait()
            if self.delay:
                await asyncio.sleep(self.delay)
            return "收到，继续说。"
        finally:
            self.active[group_id] -= 1
            self.total_active -= 1

    async def close(self):
        self.closed = True


class GatedSecondBrain(StubBrain):
    """Distinct replies; event gates target the supplementary generation only."""
    def __init__(self):
        super().__init__()
        self.second_started = asyncio.Event()
        self.second_release = None
        self.second_text = "刚才不是挺能问的，现在只剩表情包撑场了？"
        self.second_error = False

    async def reply(self, group_id, transcript, speaker, is_at, **kwargs):
        self.calls.append({"group_id": group_id, "transcript": transcript,
                           "speaker": speaker, "is_at": is_at, **kwargs})
        self.started.set()
        if len(self.calls) == 1:
            return "笑一下就想糊弄过去"
        self.second_started.set()
        if self.second_release is not None:
            await self.second_release.wait()
        if self.second_error:
            raise RuntimeError("offline simulated supplement generation failure")
        return self.second_text

class FakePlanBrain(StubBrain):
    """Scripted decisions, no HTTP; gate a selected model call to test cancellation."""
    def __init__(self, plans):
        super().__init__()
        self.plans = list(plans)
        self.plan_started = asyncio.Event()
        self.plan_release = None
        self.gate_call = 2
        self.raise_call = 0
        self.cancelled = False

    async def plan_reply(self, group_id, transcript, speaker, is_at, **kwargs):
        self.calls.append({"group_id": group_id, "transcript": transcript,
                           "speaker": speaker, "is_at": is_at, **kwargs})
        index = len(self.calls)
        self.started.set()
        if index == self.gate_call:
            self.plan_started.set()
            if self.plan_release is not None:
                try:
                    await self.plan_release.wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
        if index == self.raise_call:
            raise RuntimeError("offline simulated planning failure")
        return self.plans.pop(0) if self.plans else ReplyPlan("wait")

class FakeClient:
    def __init__(self):
        self.self_id = BOT_ID
        self.nickname = "测试机器人"
        self.sent: list[dict] = []
        self.fail = False
        self.next_id = 90000

    async def send_action(self, action, params=None, **kwargs):
        if self.fail:
            raise RuntimeError("offline simulated send failure")
        self.next_id += 1
        self.sent.append({"action": action, **(params or {}), "sent_id": self.next_id,
                          "sent_at": time.monotonic()})
        return {"message_id": self.next_id}


def test_config():
    cfg = copy.deepcopy(DEFAULTS)
    cfg["llm"]["api_key"] = ""
    cfg["groups"] = {"whitelist": [], "blacklist": []}
    cfg["users"] = {"blacklist": []}
    cfg["runtime"].update(dry_run=False, log_file="")
    cfg["safety"]["blocked_input"] = []
    cfg["behavior"].update(
        merge_delay=0.03, merge_at_delay=0.02, merge_max_wait=0.10,
        merge_max_messages=20, pending_message_limit=100,
        group_cooldown=0, direct_cooldown=0, user_cooldown=0,
        direct_user_cooldown=0, min_gap=0, per_group_per_minute=100,
        global_per_minute=100, max_chain=100, history_size=100,
        keywords=["测试机器人"], random_rate=0, question_reply=False,
        question_rate=0, topic_rates={}, followup_rate=1, second_reply_rate=0,
        followup_window=90, followup_max_turns=2,
        fixed_replies={}, image_replies={}, mute_keywords=[],
        quiet_keywords=[], sleep_wake_hours=100000,
        quote_mode="always", prefix_at=False,
    )
    return cfg


class DialogueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bot = Bot(test_config())
        await self.bot.brain.close()
        self.brain = StubBrain()
        self.client = FakeClient()
        self.bot.brain = self.brain
        self.bot.client = self.client
        self.msg_id = 100

    async def asyncTearDown(self):
        await self.bot.close()

    def event(self, text, *, user=ALICE, group=GROUP, at=None, reply=None, message_id=None):
        self.msg_id += 1
        segs = []
        if reply is not None:
            segs.append({"type": "reply", "data": {"id": str(reply)}})
        for uid in (at or []):
            segs.append({"type": "at", "data": {"qq": str(uid)}})
        segs.append({"type": "text", "data": {"text": text}})
        return {"post_type": "message", "message_type": "group",
                "group_id": group, "user_id": user, "self_id": BOT_ID,
                "message_id": self.msg_id if message_id is None else message_id,
                "sender": {"user_id": user, "nickname": f"成员{user}", "card": ""},
                "message": segs}

    async def push(self, text, **kwargs):
        event = self.event(text, **kwargs)
        await self.bot.on_event(event)
        return event["message_id"]

    async def idle(self):
        await asyncio.wait_for(self.bot.wait_idle(), 3)

    def members(self, call=-1):
        return [json.loads(row) for row in self.brain.calls[call]["transcript"].splitlines()
                if json.loads(row).get("role") == "group_member"]

    def reply_id(self, sent=-1):
        return next(seg["data"]["id"] for seg in self.client.sent[sent]["message"]
                    if seg["type"] == "reply")

    async def test_01_same_sender_fragments_merge_once(self):
        await self.push("这件打野装备", at=[BOT_ID])
        await self.push("我不是说不好")
        last = await self.push("想问是否适合这局")
        await self.idle()
        self.assertEqual(len(self.brain.calls), 1)
        self.assertEqual(len(self.client.sent), 1)
        merged = "\n".join(row["content"] for row in self.members())
        for text in ("这件打野装备", "我不是说不好", "想问是否适合这局"):
            self.assertIn(text, merged)
        self.assertEqual(self.reply_id(), str(last))

    async def test_02_different_senders_keep_separate_identity(self):
        await self.push("我在讨论打野装备", at=[BOT_ID])
        await self.push("我说的是今晚吃饭", user=BOB)
        await self.idle()
        rows = self.members(0)
        alice_rows = [row for row in rows if str(row["sender"]["id"]) == str(ALICE)]
        bob_rows = [row for row in rows if str(row["sender"]["id"]) == str(BOB)]
        self.assertTrue(alice_rows)
        self.assertTrue(bob_rows)
        self.assertFalse(any("今晚吃饭" in row["content"] for row in alice_rows))
        self.assertFalse(any("打野装备" in row["content"] for row in bob_rows))

    async def test_03_explicit_at_other_does_not_interrupt(self):
        self.bot.beh["topic_rates"] = {"daily": {"attack": 1, "friendly": 0}}
        await self.push("你觉得怎么样？", at=[BOB])
        await self.idle()
        self.assertEqual(self.brain.calls, [])
        self.assertEqual(self.client.sent, [])

    async def test_04_reply_other_does_not_interrupt(self):
        prior = await self.push("今晚一起吃饭", user=BOB)
        await self.idle()
        self.bot.beh["topic_rates"] = {"daily": {"attack": 1, "friendly": 0}}
        await self.push("那就这么定了", reply=prior)
        await self.idle()
        self.assertEqual(self.brain.calls, [])

    async def test_05_reply_bot_without_at_responds(self):
        await self.push("打野装备怎么选", at=[BOT_ID])
        await self.idle()
        prior = self.client.sent[-1]["sent_id"]
        await self.push("你再解释一下", reply=prior, user=BOB)
        await self.idle()
        self.assertEqual(len(self.brain.calls), 2)
        self.assertEqual(self.brain.calls[-1]["speaker"], f"成员{BOB}")

    async def test_06_unaddressed_greeting_not_forced(self):
        await self.push("大家早")
        await self.idle()
        self.assertEqual(self.client.sent, [])
        await self.push("你好", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)

    async def test_07_same_person_same_topic_can_follow_up(self):
        await self.push("打野装备应该怎么选择", at=[BOT_ID])
        await self.idle()
        await self.push("打野装备是不是还要看对面阵容")
        await self.idle()
        self.assertEqual(len(self.brain.calls), 2)

    async def test_08_changed_topic_after_grace_does_not_follow(self):
        await self.push("打野装备应该怎么选择", at=[BOT_ID])
        await self.idle()
        self.bot._dialogues[GROUP].updated_at -= 31
        await self.push("今晚吃什么饭")
        await self.idle()
        self.assertEqual(len(self.brain.calls), 1)

    async def test_09_same_topic_different_person_not_followup(self):
        await self.push("打野装备应该怎么选择", at=[BOT_ID])
        await self.idle()
        await self.push("打野装备是不是还要看阵容", user=BOB)
        await self.idle()
        self.assertEqual(len(self.brain.calls), 1)

    async def test_10_expired_window_does_not_follow(self):
        self.bot.beh["followup_window"] = 0
        await self.push("打野装备应该怎么选择", at=[BOT_ID])
        await self.idle()
        await self.push("打野装备是不是还要看阵容")
        await self.idle()
        self.assertEqual(len(self.brain.calls), 1)

    async def test_11_groups_independent_same_group_serial(self):
        self.brain.delay = 0.12
        await self.push("第一条问题", at=[BOT_ID])
        await asyncio.wait_for(self.brain.started.wait(), 1)
        await self.push("第二位问题", at=[BOT_ID], user=BOB)
        await self.push("另一群的问题", at=[BOT_ID], group=GROUP + 1)
        await self.idle()
        self.assertEqual(self.brain.max_active[GROUP], 1)
        self.assertEqual(self.brain.max_active[GROUP + 1], 1)
        self.assertGreaterEqual(self.brain.max_total, 2)
        self.assertEqual(len(self.client.sent), 3)

    async def test_12_failed_send_does_not_commit_dialogue(self):
        self.client.fail = True
        await self.push("打野装备怎么选", at=[BOT_ID])
        await self.idle()
        self.assertNotIn(GROUP, self.bot._last_reply)
        self.assertNotIn(GROUP, self.bot._dialogues)
        rows = [json.loads(row) for row in self.bot.store.transcript(GROUP).splitlines()]
        self.assertFalse(any(row["role"] == "assistant" for row in rows))
        self.client.fail = False
        await self.push("打野装备还有别的选择吗")
        await self.idle()
        self.assertEqual(len(self.brain.calls), 1)

    async def test_13_inflight_explicit_target_not_replaced(self):
        self.brain.release = asyncio.Event()
        target = await self.push("打野装备怎么选", at=[BOT_ID])
        await asyncio.wait_for(self.brain.started.wait(), 1)
        await self.push("我的晚饭已经吃完了", user=BOB)
        self.brain.release.set()
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(self.reply_id(), str(target))
        self.assertEqual(self.bot._last_reply[GROUP][0], ALICE)

    async def test_14_active_probabilities_off_really_silent(self):
        for text in ("普通聊天", "你们怎么想？", "大家早", "垃圾装备太多了"):
            await self.push(text)
        await self.idle()
        self.assertEqual(self.brain.calls, [])

    async def test_15_at_self_overrides_other_target(self):
        await self.push("你们一起看下打野装备", at=[BOB, BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)

    async def test_16_no_infinite_postponement_on_busy_group(self):
        self.bot.beh["merge_delay"] = 0.05
        self.bot.beh["merge_at_delay"] = 0.05
        self.bot.beh["merge_max_wait"] = 0.10
        await self.push("打野装备怎么选", at=[BOT_ID])
        for _ in range(8):
            await asyncio.sleep(0.02)
            await self.push("打野装备补充情况")
        self.assertTrue(self.brain.started.is_set(), "持续补充不能无限推迟首批回复")
        await self.idle()

    async def test_17_close_cancels_buffered_work(self):
        self.bot.beh["merge_delay"] = 10
        self.bot.beh["merge_at_delay"] = 10
        self.bot.beh["merge_max_wait"] = 10
        await self.push("稍后处理的问题", at=[BOT_ID])
        await self.bot.close()
        await asyncio.sleep(0.04)
        self.assertEqual(self.brain.calls, [])
        self.assertEqual(self.client.sent, [])
        self.assertTrue(self.brain.closed)

    async def test_18_coarse_topic_category_after_grace_is_not_enough(self):
        await self.push("打野装备应该怎么选择", at=[BOT_ID])
        await self.idle()
        self.bot._dialogues[GROUP].updated_at -= 31
        await self.push("王者皮肤抽奖要多少钱")
        await self.idle()
        self.assertEqual(len(self.brain.calls), 1,
                         "同属游戏大类不代表同一个话题，不能继续追着接话")

    async def test_19_followup_turns_are_bounded(self):
        self.bot.beh["followup_max_turns"] = 1
        await self.push("打野装备应该怎么选择", at=[BOT_ID])
        await self.idle()
        await self.push("打野装备是不是还要看对面阵容")
        await self.idle()
        await self.push("打野装备还需要根据阵容再调整吗")
        await self.idle()
        self.assertEqual(len(self.brain.calls), 2)

    async def test_20_close_cancels_inflight_generation(self):
        self.brain.release = asyncio.Event()
        await self.push("稍后处理的问题", at=[BOT_ID])
        await asyncio.wait_for(self.brain.started.wait(), 1)
        await asyncio.wait_for(self.bot.close(), 1)
        self.assertEqual(self.brain.total_active, 0)
        self.assertEqual(self.client.sent, [])
        self.assertTrue(self.brain.closed)

    async def test_21_late_fragments_replan_once_and_send_merged_once(self):
        first_release = asyncio.Event()
        self.brain.release = first_release
        await self.push("打野装备我想问", at=[BOT_ID])
        await asyncio.wait_for(self.brain.started.wait(), 1)
        await self.push("我不是说不好")
        last = await self.push("主要想知道是否适合这局阵容")
        first_release.set()
        await self.idle()
        self.assertEqual(len(self.brain.calls), 2, "已生成过程中收到补充应仅额外重判一次")
        self.assertEqual(len(self.client.sent), 1, "不能先发旧答案再发合并答案")
        self.assertEqual(self.brain.max_active[GROUP], 1)
        merged = self.members()[-1]["content"]
        for part in ("打野装备我想问", "我不是说不好", "主要想知道是否适合这局阵容"):
            self.assertIn(part, merged)
        self.assertEqual(self.reply_id(), str(last))

    async def test_22_same_batch_switch_to_other_target_drops_old_request(self):
        await self.push("打野装备你觉得呢", at=[BOT_ID])
        await self.push("算了，问你吧", at=[BOB])
        await self.idle()
        self.assertEqual(self.brain.calls, [])
        self.assertEqual(self.client.sent, [])

    async def test_23_poke_and_message_share_group_serial_queue(self):
        self.brain.delay = 0.08
        await self.push("打野装备怎么选", at=[BOT_ID])
        await asyncio.wait_for(self.brain.started.wait(), 1)
        await self.bot.on_event({"post_type": "notice", "notice_type": "notify",
                                 "sub_type": "poke", "group_id": GROUP,
                                 "user_id": BOB, "target_id": BOT_ID})
        await self.idle()
        self.assertEqual(len(self.brain.calls), 2)
        self.assertEqual(self.brain.max_active[GROUP], 1)
        self.assertEqual(len(self.client.sent), 2)

    async def test_24_same_message_id_in_two_groups_not_duplicate(self):
        await self.push("第一群问题", at=[BOT_ID], message_id=4242)
        await self.push("第二群问题", at=[BOT_ID], group=GROUP + 1, message_id=4242)
        await self.idle()
        self.assertEqual(len(self.brain.calls), 2)
        self.assertEqual({call["group_id"] for call in self.brain.calls}, {GROUP, GROUP + 1})

    async def test_25_image_delivery_is_indexed_and_can_be_replied_to(self):
        self.bot.beh["image_replies"] = {"发测试图": {"text": "测试图片", "image": "not-a-real-image.png"}}
        with patch.object(self.bot, "_image_uri", return_value="base64://offline-test-fixture") as uri:
            await self.push("发测试图", at=[BOT_ID])
            await self.idle()
        uri.assert_called_once_with("not-a-real-image.png")
        self.assertEqual(len(self.client.sent), 1)
        self.assertTrue(any(segment["type"] == "image" for segment in self.client.sent[0]["message"]))
        image_message_id = self.client.sent[0]["sent_id"]
        await self.push("这张图是什么意思", user=BOB, reply=image_message_id)
        await self.idle()
        self.assertEqual(len(self.brain.calls), 1, "图片回复也必须被识别为机器人所发消息")
        self.assertEqual(len(self.client.sent), 2)
        self.assertEqual(self.brain.calls[0]["speaker"], f"成员{BOB}")

    async def test_26_failed_image_does_not_commit_or_index(self):
        self.client.fail = True
        self.bot.beh["image_replies"] = {"发测试图": {"text": "测试图片", "image": "not-a-real-image.png"}}
        with patch.object(self.bot, "_image_uri", return_value="base64://offline-test-fixture"):
            await self.push("发测试图", at=[BOT_ID])
            await self.idle()
        self.assertNotIn(GROUP, self.bot._last_reply)
        self.assertNotIn(GROUP, self.bot._dialogues)
        rows = [json.loads(row) for row in self.bot.store.transcript(GROUP).splitlines()]
        self.assertFalse(any(row["role"] == "assistant" for row in rows))
        self.assertFalse(any(item.is_bot for item in self.bot._msg_index.values()))
        self.assertEqual(self.client.sent, [])

    async def test_27_unknown_reference_does_not_guess_addressee(self):
        self.bot.beh["topic_rates"] = {"daily": {"attack": 1, "friendly": 0}}
        await self.push("这是在叫谁？", reply=876543)
        await self.idle()
        self.assertEqual(self.brain.calls, [])
        self.assertEqual(self.client.sent, [])

    async def test_28_late_replan_consumes_only_one_rate_slot(self):
        self.bot.beh.update(per_group_per_minute=1, global_per_minute=1)
        first_release = asyncio.Event()
        self.brain.release = first_release
        await self.push("打野装备我想问", at=[BOT_ID])
        await asyncio.wait_for(self.brain.started.wait(), 1)
        await self.push("主要问是否适合这局阵容")
        first_release.set()
        await self.idle()
        self.assertEqual(len(self.brain.calls), 2,
                         "同一请求的有界重判不能再扣一次限流额度")
        self.assertEqual(len(self.client.sent), 1)

    async def test_29_poke_switch_independent_of_at_reply_switch(self):
        self.bot.beh.update(at_reply=False, reply_poke=True)
        await self.bot.on_event({"post_type": "notice", "notice_type": "notify",
                                 "sub_type": "poke", "group_id": GROUP,
                                 "user_id": ALICE, "target_id": BOT_ID})
        await self.idle()
        self.assertEqual(len(self.brain.calls), 1,
                         "关掉@回复不应连带关掉明确开启的戳一戳回复")
        self.assertEqual(len(self.client.sent), 1)

    async def test_30_unaddressed_suffix_cannot_hide_same_batch_redirect(self):
        await self.push("打野装备你觉得呢", at=[BOT_ID])
        await self.push("算了，问你吧", at=[BOB])
        await self.push("再补充一下阵容情况")
        await self.idle()
        self.assertEqual(self.brain.calls, [],
                         "转向其他人的后续无@补充不能使旧@机器人请求复活")
        self.assertEqual(self.client.sent, [])

    async def test_31_unaddressed_suffix_cannot_hide_inflight_redirect(self):
        self.brain.release = asyncio.Event()
        await self.push("打野装备你觉得呢", at=[BOT_ID])
        await asyncio.wait_for(self.brain.started.wait(), 1)
        await self.push("算了，问你吧", at=[BOB])
        await self.push("再补充一下阵容情况")
        self.brain.release.set()
        await self.idle()
        self.assertEqual(len(self.brain.calls), 1,
                         "已开始的旧请求应废弃，不再为已转向他人的补充重判")
        self.assertEqual(self.client.sent, [])

    async def test_32_batch_limit_leaves_remaining_items_queued(self):
        self.bot.beh["merge_max_messages"] = 2
        self.brain.release = asyncio.Event()
        for offset in range(5):
            await self.push(f"不同成员的问题{offset}", user=ALICE + offset, at=[BOT_ID])
        await asyncio.wait_for(self.brain.started.wait(), 1)
        self.assertEqual(len(self.bot._pending[GROUP]), 3,
                         "一批最多提取两条，不能取空整组队列再声称分批")
        self.brain.release.set()
        await self.idle()
        self.assertEqual(len(self.client.sent), 5)
        self.assertEqual(self.brain.max_active[GROUP], 1)

    async def test_33_cross_group_generation_parallel_delivery_gap_enforced(self):
        gap = 0.03
        self.bot.beh["min_gap"] = gap
        self.brain.delay = 0.05
        await self.push("第一群的问题", at=[BOT_ID])
        await self.push("第二群的问题", at=[BOT_ID], group=GROUP + 1)
        await self.idle()
        self.assertEqual(len(self.client.sent), 2)
        self.assertGreaterEqual(self.brain.max_total, 2,
                                "发送限速不能把不同群的模型生成串行化")
        timestamps = sorted(item["sent_at"] for item in self.client.sent)
        self.assertGreaterEqual(timestamps[1] - timestamps[0], gap - 0.002)

    async def test_34_dry_run_does_not_commit_delivery_state(self):
        self.bot.cfg["runtime"]["dry_run"] = True
        await self.push("打野装备怎么选", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.brain.calls), 1)
        self.assertEqual(self.client.sent, [])
        self.assertNotIn(GROUP, self.bot._last_reply)
        self.assertNotIn(GROUP, self.bot._dialogues)
        rows = [json.loads(row) for row in self.bot.store.transcript(GROUP).splitlines()]
        self.assertFalse(any(row["role"] == "assistant" for row in rows))
        self.assertFalse(any(item.is_bot for item in self.bot._msg_index.values()))

    async def test_35_reply_index_is_scoped_by_group_with_same_id(self):
        await self.push("第一群打野装备问题", at=[BOT_ID])
        await self.idle()
        bot_message_id = self.client.sent[0]["sent_id"]
        # A different group reuses the same ID, but there it belongs to BOB.
        await self.push("第二群成员自己的发言", user=BOB, group=GROUP + 1,
                        message_id=bot_message_id)
        await self.idle()
        await self.push("再解释一下", reply=bot_message_id)
        await self.push("你说的什么意思", reply=bot_message_id, group=GROUP + 1)
        await self.idle()
        self.assertEqual(len(self.brain.calls), 2)
        self.assertEqual({call["group_id"] for call in self.brain.calls}, {GROUP},
                         "引用应按本群消息作者判断，不能跨群借用同ID的机器人身份")

    async def test_36_late_replan_rechecks_blocked_input_in_merged_text(self):
        blocked_word = "离线测试禁止输入标记"
        self.bot.safety["blocked_input"] = [blocked_word]
        self.brain.release = asyncio.Event()
        await self.push("打野装备怎么选", at=[BOT_ID])
        await asyncio.wait_for(self.brain.started.wait(), 1)
        await self.push(f"再补充一条：{blocked_word}")
        self.brain.release.set()
        await self.idle()
        self.assertEqual(len(self.brain.calls), 1,
                         "重判虽复用触发原因，但合并后的全文仍必须重新检查禁止输入")
        self.assertEqual(self.client.sent, [])


    def relaxed_followup_settings(self):
        self.bot.beh.update(followup_window=120, followup_grace_window=30,
                            followup_max_turns=6, followup_rate=1,
                            merge_delay=0.005, merge_at_delay=0.005,
                            merge_max_wait=0.02)

    async def test_37_screenshot_find_me_answer_continues_without_overlap(self):
        self.relaxed_followup_settings()
        await self.push("现在你还在吗大肥鱼", at=[BOT_ID])
        await self.idle()
        self.bot._dialogues[GROUP].bot_text = "找我干嘛，天线宝宝还想管我"
        await self.push("没事找你聊天啊，今天天气不错")
        await self.idle()
        self.assertEqual(len(self.brain.calls), 2)
        self.assertEqual(len(self.client.sent), 2)
        self.assertEqual(self.members()[-1]["content"], "没事找你聊天啊，今天天气不错")

    async def test_38_screenshot_just_one_question_answer_continues(self):
        self.relaxed_followup_settings()
        await self.push("现在你还在吗大肥鱼", at=[BOT_ID])
        await self.idle()
        self.bot._dialogues[GROUP].bot_text = "活着，你就为问这一句?"
        await self.push("没事找你聊天啊，今天天气不错")
        await self.idle()
        self.assertEqual(len(self.brain.calls), 2)
        self.assertEqual(len(self.client.sent), 2)

    async def test_39_grace_window_allows_natural_same_person_topic_change(self):
        self.relaxed_followup_settings()
        await self.push("打野装备怎么选择", at=[BOT_ID])
        await self.idle()
        await self.push("今天天气不错，晚上出去吃火锅")
        await self.idle()
        self.assertEqual(len(self.client.sent), 2,
                         "30秒内同一人无人插话时，不应强制重复旧话题关键词")

    async def test_40_grace_window_does_not_treat_other_member_as_followup(self):
        self.relaxed_followup_settings()
        await self.push("打野装备怎么选择", at=[BOT_ID])
        await self.idle()
        await self.push("打野装备要看阵容", user=BOB)
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)

    async def test_41_other_member_interruption_blocks_even_related_followup(self):
        self.relaxed_followup_settings()
        await self.push("打野装备怎么选择", at=[BOT_ID])
        await self.idle()
        await self.push("我先说一句", user=BOB)
        await self.idle()
        await self.push("打野装备还需要看阵容")
        await self.idle()
        self.assertEqual(len(self.client.sent), 1,
                         "别人插话后不能仅凭词重合恢复隐式续聊")

    async def test_42_outer_window_allows_clear_you_link_without_overlap(self):
        self.relaxed_followup_settings()
        await self.push("现在你还在吗大肥鱼", at=[BOT_ID])
        await self.idle()
        self.bot._dialogues[GROUP].updated_at = time.monotonic() - 45
        self.bot._dialogues[GROUP].bot_text = "活着，你就为问这一句?"
        await self.push("没事找你聊天啊，今天天气不错")
        await self.idle()
        self.assertEqual(len(self.client.sent), 2)

    async def test_43_outer_window_keeps_related_topic(self):
        self.relaxed_followup_settings()
        await self.push("打野装备怎么选择", at=[BOT_ID])
        await self.idle()
        self.bot._dialogues[GROUP].updated_at = time.monotonic() - 45
        await self.push("打野装备还要看什么阵容")
        await self.idle()
        self.assertEqual(len(self.client.sent), 2)

    async def test_44_expired_even_clear_you_link_is_not_implicit_followup(self):
        self.relaxed_followup_settings()
        await self.push("现在你还在吗大肥鱼", at=[BOT_ID])
        await self.idle()
        self.bot._dialogues[GROUP].updated_at = time.monotonic() - 121
        await self.push("没事找你聊天啊，今天天气不错")
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)

    async def test_45_followup_limit_six_replies_not_infinite(self):
        self.relaxed_followup_settings()
        await self.push("现在有空吗", at=[BOT_ID])
        await self.idle()
        for offset in range(7):
            await self.push(f"继续聊第{offset + 1}件事，今天天气不错")
            await self.idle()
        self.assertEqual(len(self.client.sent), 7, "一次明确唤起后只隐式续聊六次")
        self.assertEqual(self.bot._dialogues[GROUP].followups, 6)

    async def test_46_followup_rate_one_does_not_randomly_drop(self):
        self.relaxed_followup_settings()
        await self.push("现在有空吗", at=[BOT_ID])
        await self.idle()
        with patch("bot.random.random", return_value=0.99):
            await self.push("没事找你聊天啊，今天天气不错")
            await self.idle()
        self.assertEqual(len(self.client.sent), 2)

    async def test_47_fast_followup_waits_direct_cooldown_not_dropped(self):
        self.relaxed_followup_settings()
        await self.push("现在有空吗", at=[BOT_ID])
        await self.idle()
        self.bot.beh.update(direct_cooldown=0.06, direct_user_cooldown=0.06)
        await self.push("没事找你聊天啊，今天天气不错")
        await self.idle()
        self.assertEqual(len(self.client.sent), 2)
        self.assertGreaterEqual(self.client.sent[-1]["sent_at"] -
                                self.client.sent[0]["sent_at"], 0.055)

    async def test_48_interruption_during_followup_cooldown_cancels(self):
        self.relaxed_followup_settings()
        await self.push("现在有空吗", at=[BOT_ID])
        await self.idle()
        self.bot.beh.update(direct_cooldown=0.06, direct_user_cooldown=0.06)
        await self.push("没事找你聊天啊，今天天气不错")
        await asyncio.sleep(0.02)
        await self.push("我先说一句", user=BOB)
        await self.idle()
        self.assertEqual(len(self.brain.calls), 1)
        self.assertEqual(len(self.client.sent), 1)

    async def test_49_interruption_during_followup_generation_cancels(self):
        self.relaxed_followup_settings()
        await self.push("现在有空吗", at=[BOT_ID])
        await self.idle()
        self.brain.started.clear()
        self.brain.release = asyncio.Event()
        await self.push("没事找你聊天啊，今天天气不错")
        await asyncio.wait_for(self.brain.started.wait(), 1)
        await self.push("我先说一句", user=BOB)
        self.brain.release.set()
        await self.idle()
        self.assertEqual(len(self.brain.calls), 2)
        self.assertEqual(len(self.client.sent), 1)

    async def test_50_interruption_before_followup_delivery_cancels(self):
        self.relaxed_followup_settings()
        await self.push("现在有空吗", at=[BOT_ID])
        await self.idle()
        self.brain.started.clear()
        await self.bot._delivery_lock.acquire()
        try:
            await self.push("没事找你聊天啊，今天天气不错")
            await asyncio.wait_for(self.brain.started.wait(), 1)
            await asyncio.sleep(0.01)
            await self.push("我先说一句", user=BOB)
        finally:
            self.bot._delivery_lock.release()
        await self.idle()
        self.assertEqual(len(self.brain.calls), 2)
        self.assertEqual(len(self.client.sent), 1)

    async def test_51_redirect_during_followup_cooldown_cancels(self):
        self.relaxed_followup_settings()
        await self.push("现在有空吗", at=[BOT_ID])
        await self.idle()
        self.bot.beh.update(direct_cooldown=0.06, direct_user_cooldown=0.06)
        await self.push("没事找你聊天啊，今天天气不错")
        await asyncio.sleep(0.02)
        await self.push("算了我还是问你吧", at=[BOB])
        await self.idle()
        self.assertEqual(len(self.brain.calls), 1)
        self.assertEqual(len(self.client.sent), 1)

    async def test_52_topic_end_blocks_grace_followup(self):
        self.relaxed_followup_settings()
        await self.push("现在有空吗", at=[BOT_ID])
        await self.idle()
        await self.push("先不聊了，今天天气不错")
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)

    async def test_53_zero_grace_restores_lexical_gate(self):
        self.relaxed_followup_settings()
        self.bot.beh["followup_grace_window"] = 0
        await self.push("打野装备怎么选择", at=[BOT_ID])
        await self.idle()
        await self.push("晚上去吃火锅")
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        await self.push("打野装备还看阵容吗")
        await self.idle()
        self.assertEqual(len(self.client.sent), 2)

    async def test_54_redirect_then_plain_suffix_does_not_resume_dialogue(self):
        self.relaxed_followup_settings()
        await self.push("现在有空吗", at=[BOT_ID])
        await self.idle()
        await self.push("我们聊天吧", at=[BOB])
        await self.push("没事找你聊天啊，今天天气不错")
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)

    async def test_55_relaxed_followup_still_respects_group_quota(self):
        self.relaxed_followup_settings()
        self.bot.beh["per_group_per_minute"] = 1
        await self.push("现在有空吗", at=[BOT_ID])
        await self.idle()
        await self.push("没事找你聊天啊，今天天气不错")
        await self.idle()
        self.assertEqual(len(self.brain.calls), 1)
        self.assertEqual(len(self.client.sent), 1)


    def setup_second_reply(self, rate=1.0):
        self.brain = GatedSecondBrain()
        self.bot.brain = self.brain
        self.bot.beh.update(model_reply_planning=False, second_reply_rate=rate,
                            second_reply_delay_min=0.02,
                            second_reply_delay_max=0.03)

    async def wait_sent_count(self, count):
        async def wait():
            while len(self.client.sent) < count:
                await asyncio.sleep(0.001)
        await asyncio.wait_for(wait(), 1)

    def bot_rows(self, group=GROUP):
        return [json.loads(line) for line in self.bot.store.transcript(group).splitlines()
                if json.loads(line).get("role") == "assistant"]

    async def test_56_second_reply_point59_passes_and_never_thirds(self):
        self.setup_second_reply(0.6)
        with patch("bot.random.random", return_value=0.59) as coin:
            await self.push(":)", at=[BOT_ID])
            await self.idle()
        self.assertEqual(len(self.client.sent), 2)
        self.assertEqual(len(self.brain.calls), 2)
        self.assertEqual(coin.call_count, 1, "每轮只有一次补发抽签，不能递归补发")
        self.assertIn("补", self.brain.calls[1].get("focus_note", ""))
        self.assertIn("笑一下就想糊弄过去", self.brain.calls[1]["transcript"])
        self.assertEqual(len(self.bot_rows()), 2)
        self.assertEqual(self.bot._dialogues[GROUP].followups, 0,
                         "主动补一句不消耗群友的隐式续聊轮数")

    async def test_57_second_reply_point60_rejected_at_60_percent(self):
        self.setup_second_reply(0.6)
        with patch("bot.random.random", return_value=0.60) as coin:
            await self.push("给我说句话", at=[BOT_ID])
            await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.brain.calls), 1)
        self.assertEqual(coin.call_count, 1)

    async def test_58_second_reply_zero_disables_extra_generation(self):
        self.setup_second_reply(0)
        await self.push("给我说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.brain.calls), 1)
        self.assertEqual(len(self.client.sent), 1)

    async def test_59_failed_primary_never_generates_second(self):
        self.setup_second_reply()
        self.client.fail = True
        await self.push("给我说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.brain.calls), 1)
        self.assertEqual(len(self.client.sent), 0)
        self.assertEqual(self.bot_rows(), [])

    async def test_60_new_other_user_during_delay_cancels_second(self):
        self.setup_second_reply()
        self.bot.beh.update(second_reply_delay_min=0.08,
                            second_reply_delay_max=0.08)
        await self.push("给我说句话", at=[BOT_ID])
        await self.wait_sent_count(1)
        await self.push("我先说一句", user=BOB)
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.brain.calls), 1)

    async def test_61_same_user_redirect_during_delay_cancels_second(self):
        self.setup_second_reply()
        self.bot.beh.update(second_reply_delay_min=0.08,
                            second_reply_delay_max=0.08)
        await self.push("给我说句话", at=[BOT_ID])
        await self.wait_sent_count(1)
        await self.push("我现在问别人", at=[BOB])
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.brain.calls), 1)

    async def test_62_other_group_message_does_not_cancel_second(self):
        self.setup_second_reply()
        await self.push("给我说句话", at=[BOT_ID])
        await self.wait_sent_count(1)
        await self.push("我先说一句", group=GROUP + 1, user=BOB)
        await self.idle()
        self.assertEqual(len(self.client.sent), 2)
        self.assertTrue(all(item["group_id"] == GROUP for item in self.client.sent))

    async def test_63_next_turn_already_in_worker_batch_cancels_second(self):
        self.setup_second_reply()
        await self.push("给我说句话", at=[BOT_ID])
        await self.push("我先说一句", user=BOB)
        await self.idle()
        self.assertEqual(len(self.client.sent), 1,
                         "take_batch已把下一条拿走也不能对上一条追发")
        self.assertEqual(len(self.brain.calls), 1)

    async def test_64_new_message_during_second_generation_cancels_send(self):
        self.setup_second_reply()
        self.brain.second_release = asyncio.Event()
        await self.push("给我说句话", at=[BOT_ID])
        await asyncio.wait_for(self.brain.second_started.wait(), 1)
        await self.push("我先说一句", user=BOB)
        self.brain.second_release.set()
        await self.idle()
        self.assertEqual(len(self.brain.calls), 2)
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.bot_rows()), 1)

    async def test_65_new_message_while_second_waits_send_lock_cancels(self):
        self.setup_second_reply()
        self.brain.second_release = asyncio.Event()
        await self.push("给我说句话", at=[BOT_ID])
        await asyncio.wait_for(self.brain.second_started.wait(), 1)
        await self.bot._delivery_lock.acquire()
        try:
            self.brain.second_release.set()
            await asyncio.sleep(0.01)
            await self.push("我先说一句", user=BOB)
        finally:
            self.bot._delivery_lock.release()
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.bot_rows()), 1)

    async def test_66_second_send_failure_does_not_pollute_history(self):
        self.setup_second_reply()
        original_send = self.client.send_action
        async def send(action, params=None, **kwargs):
            if self.client.sent:
                raise RuntimeError("offline simulated second send failure")
            return await original_send(action, params, **kwargs)
        self.client.send_action = send
        await self.push("给我说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.bot_rows()), 1)
        self.assertEqual(self.bot._dialogues[GROUP].bot_text, "笑一下就想糊弄过去")
        self.assertEqual(sum(item.is_bot for item in self.bot._msg_index.values()), 1)

    async def test_67_second_obeys_actual_group_limit_eighteen(self):
        self.setup_second_reply()
        self.bot.beh["per_group_per_minute"] = 18
        for _ in range(17):
            self.assertTrue(self.bot.limiter.allow(("group", GROUP), 18))
        await self.push("给我说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.bot.limiter._hits[("group", GROUP)]), 18)

    async def test_68_second_obeys_actual_global_limit_twentyfive(self):
        self.setup_second_reply()
        self.bot.beh["global_per_minute"] = 25
        for _ in range(24):
            self.assertTrue(self.bot.limiter.allow(("global",), 25))
        await self.push("给我说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.bot.limiter._hits[("global",)]), 25)

    async def test_69_close_cancels_second_without_delivery(self):
        self.setup_second_reply()
        self.bot.beh.update(second_reply_delay_min=0.08,
                            second_reply_delay_max=0.08)
        await self.push("给我说句话", at=[BOT_ID])
        await self.wait_sent_count(1)
        await self.bot.close()
        await asyncio.sleep(0.10)
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.brain.calls), 1)
        self.assertTrue(self.brain.closed)

    async def test_70_normalized_duplicate_second_is_not_sent(self):
        self.setup_second_reply()
        self.brain.second_text = "  笑一下就想糊弄过去！！  "
        await self.push("给我说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.brain.calls), 2)
        self.assertEqual(len(self.client.sent), 1)

    async def test_71_second_failure_is_contained_and_worker_keeps_working(self):
        self.setup_second_reply()
        self.brain.second_error = True
        await self.push("给我说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.brain.second_error = False
        self.bot.beh["second_reply_rate"] = 0
        await self.push("另一个明确的问题", at=[BOT_ID], user=BOB)
        await self.idle()
        self.assertEqual(len(self.client.sent), 2)

    async def test_72_default_second_probability_and_delay_contract(self):
        self.assertEqual(DEFAULTS["behavior"]["second_reply_rate"], 0.6)
        self.assertEqual(DEFAULTS["behavior"]["second_reply_delay_min"], 0.8)
        self.assertEqual(DEFAULTS["behavior"]["second_reply_delay_max"], 0.8)

    async def test_73_new_topic_ending_during_delay_cancels_second(self):
        self.setup_second_reply()
        self.bot.beh.update(second_reply_delay_min=0.08,
                            second_reply_delay_max=0.08)
        await self.push("给我说句话", at=[BOT_ID])
        await self.wait_sent_count(1)
        await self.push("先不聊了")
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.brain.calls), 1)

    async def test_74_supplement_consumes_both_quota_buckets(self):
        self.setup_second_reply()
        await self.push("给我说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 2)
        self.assertEqual(len(self.bot.limiter._hits[("group", GROUP)]), 2)
        self.assertEqual(len(self.bot.limiter._hits[("global",)]), 2)

    async def test_75_second_respects_max_chain_one(self):
        self.setup_second_reply()
        self.bot.beh["max_chain"] = 1
        await self.push("说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.brain.calls), 1)

    async def test_76_same_member_new_message_cancels_old_second(self):
        self.setup_second_reply()
        self.bot.beh.update(second_reply_delay_min=.08, second_reply_delay_max=.08)
        await self.push("说句话", at=[BOT_ID])
        await self.wait_sent_count(1)
        await self.push("我还有一句新话")
        self.bot.beh["second_reply_rate"] = 0
        await self.idle()
        self.assertEqual(len(self.client.sent), 2)
        self.assertEqual(len(self.brain.calls), 2)
        self.assertNotIn("同轮补发第二条", self.brain.calls[-1].get("focus_note", ""))

    async def test_77_empty_forward_cancels_pending_second(self):
        self.setup_second_reply()
        self.bot.beh.update(second_reply_delay_min=.08, second_reply_delay_max=.08)
        await self.push("说句话", at=[BOT_ID])
        await self.wait_sent_count(1)
        e = self.event("", user=BOB)
        e["message"] = [{"type": "forward", "data": {"id": "offline-forward"}}]
        await self.bot.on_event(e)
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.brain.calls), 1)

    async def test_78_second_does_not_restart_dialogue_clock(self):
        self.setup_second_reply()
        await self.push("说句话", at=[BOT_ID])
        await self.wait_sent_count(1)
        state = self.bot._dialogues[GROUP]
        before = (state.updated_at, state.followups, state.user_text)
        await self.idle()
        state = self.bot._dialogues[GROUP]
        self.assertEqual((state.updated_at, state.followups, state.user_text), before)
        self.assertIn("只剩表情包", state.bot_text)

    async def test_79_full_global_does_not_charge_extra_group_slot(self):
        self.setup_second_reply()
        self.bot.beh["global_per_minute"] = 1
        await self.push("说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.bot.limiter._hits[("group", GROUP)]), 1)
        self.assertEqual(len(self.brain.calls), 1)

    async def test_80_new_message_cancels_inflight_generation_without_release(self):
        self.setup_second_reply()
        self.brain.second_release = asyncio.Event()
        await self.push("说句话", at=[BOT_ID])
        await asyncio.wait_for(self.brain.second_started.wait(), 1)
        await self.push("有人插话", user=BOB)
        await asyncio.wait_for(self.bot.wait_idle(), 1)
        self.assertEqual(len(self.client.sent), 1)
        self.assertFalse(self.brain.second_release.is_set())

    async def test_81_interruption_during_primary_prevents_second(self):
        self.setup_second_reply()
        original = self.brain.reply
        started, release = asyncio.Event(), asyncio.Event()
        async def gated(*args, **kwargs):
            if not self.brain.calls:
                started.set()
                await release.wait()
            return await original(*args, **kwargs)
        self.brain.reply = gated
        await self.push("说句话", at=[BOT_ID])
        await asyncio.wait_for(started.wait(), 1)
        await self.push("先说一句", user=BOB)
        release.set()
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.brain.calls), 1)


    async def test_82_poke_cancels_second_even_when_poke_reply_disabled(self):
        self.setup_second_reply()
        self.bot.beh.update(reply_poke=False, second_reply_delay_min=.08, second_reply_delay_max=.08)
        await self.push("说句话", at=[BOT_ID])
        await self.wait_sent_count(1)
        await self.bot.on_event({"post_type": "notice", "notice_type": "notify", "sub_type": "poke",
                                 "group_id": GROUP, "user_id": BOB, "target_id": BOT_ID})
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.brain.calls), 1)


    def setup_model_plans(self, *plans):
        self.brain = FakePlanBrain(plans)
        self.bot.brain = self.brain
        self.bot.beh.update(model_reply_planning=True, model_burst_max_messages=3,
                            second_reply_rate=0.6, second_reply_delay_min=0.01,
                            second_reply_delay_max=0.01)

    async def test_83_model_primary_wait_or_stop_sends_nothing_without_timer_retry(self):
        for action in ("wait", "stop"):
            with self.subTest(action=action):
                self.setup_model_plans(ReplyPlan(action))
                before = len(self.client.sent)
                await self.push("现在聊两句", at=[BOT_ID])
                await self.idle()
                await asyncio.sleep(0.04)
                self.assertEqual(len(self.client.sent), before)
                self.assertEqual(len(self.brain.calls), 1)
                self.assertFalse(self.bot.limiter._hits.get(("group", GROUP)))
                self.assertFalse(self.bot.limiter._hits.get(("global",)))

    async def test_84_model_one_reply_wait_does_not_probe_second(self):
        self.setup_model_plans(ReplyPlan("reply", "笑一下就想糊弄过去", "wait"))
        await self.push(":)", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.brain.calls), 1)

    async def test_85_model_two_replies_not_blocked_by_old_random_gate(self):
        self.setup_model_plans(ReplyPlan("reply", "笑一下就想糊弄过去", "reply"),
                               ReplyPlan("reply", "刚才不是还挺能问的", "wait"))
        with patch("bot.random.random", return_value=0.99):
            await self.push(":)", at=[BOT_ID])
            await self.idle()
        self.assertEqual(len(self.client.sent), 2)
        self.assertEqual(len(self.brain.calls), 2)
        self.assertIn("笑一下就想糊弄过去", self.brain.calls[1]["transcript"])

    async def test_86_model_can_send_three_but_never_four(self):
        self.setup_model_plans(ReplyPlan("reply", "先把问题说清楚", "reply"),
                               ReplyPlan("reply", "别让我替你猜谜", "reply"),
                               ReplyPlan("reply", "你这聊天难度快赶上解题了", "reply"),
                               ReplyPlan("reply", "不应该发出的第四条", "wait"))
        await self.push("你猜猜", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 3)
        self.assertEqual(len(self.brain.calls), 3)
        self.assertEqual(len(self.brain.plans), 1)
        self.assertEqual(len(self.bot_rows()), 3)

    async def test_87_model_secondary_wait_or_stop_ends_burst(self):
        for action in ("wait", "stop"):
            with self.subTest(action=action):
                self.setup_model_plans(ReplyPlan("reply", "先说第一句" + action, "reply"),
                                       ReplyPlan(action),
                                       ReplyPlan("reply", "不应该再生成", "wait"))
                before = len(self.client.sent)
                await self.push("现在继续聊", at=[BOT_ID])
                await self.idle()
                self.assertEqual(len(self.client.sent), before + 1)
                self.assertEqual(len(self.brain.calls), 2)

    async def test_88_model_next_stop_makes_one_call(self):
        self.setup_model_plans(ReplyPlan("reply", "这句话说完就收住", "stop"))
        await self.push("说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.brain.calls), 1)

    async def test_89_model_new_same_group_message_cancels_during_delay(self):
        self.setup_model_plans(ReplyPlan("reply", "先说第一句", "reply"),
                               ReplyPlan("reply", "不该追着发", "wait"))
        self.bot.beh.update(second_reply_delay_min=0.08, second_reply_delay_max=0.08)
        await self.push("说句话", at=[BOT_ID])
        await self.wait_sent_count(1)
        await self.push("有人插话", user=BOB)
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.brain.calls), 1)

    async def test_90_model_other_group_message_keeps_burst(self):
        self.setup_model_plans(ReplyPlan("reply", "先说第一句", "reply"),
                               ReplyPlan("reply", "再补一个新意思", "wait"))
        self.bot.beh.update(second_reply_delay_min=0.05, second_reply_delay_max=0.05)
        await self.push("说句话", at=[BOT_ID])
        await self.wait_sent_count(1)
        await self.push("另外群聊消息", user=BOB, group=GROUP + 1)
        await self.idle()
        self.assertEqual(len(self.client.sent), 2)
        self.assertTrue(all(row["group_id"] == GROUP for row in self.client.sent))

    async def test_91_model_quota_eighteen_prevents_extra_generation(self):
        self.setup_model_plans(ReplyPlan("reply", "第十八条", "reply"),
                               ReplyPlan("reply", "超额不该调用", "wait"))
        self.bot.beh["per_group_per_minute"] = 18
        for _ in range(17):
            self.assertTrue(self.bot.limiter.allow(("group", GROUP), 18))
        await self.push("说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.brain.calls), 1)
        self.assertEqual(len(self.bot.limiter._hits[("group", GROUP)]), 18)

    async def test_92_model_global_twentyfive_prevents_extra_charge(self):
        self.setup_model_plans(ReplyPlan("reply", "第二十五条", "reply"),
                               ReplyPlan("reply", "不该调用", "wait"))
        self.bot.beh["global_per_minute"] = 25
        for _ in range(24):
            self.assertTrue(self.bot.limiter.allow(("global",), 25))
        await self.push("说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.brain.calls), 1)
        self.assertEqual(len(self.bot.limiter._hits[("group", GROUP)]), 1)

    async def test_93_model_max_chain_still_caps_burst(self):
        self.setup_model_plans(ReplyPlan("reply", "第一条", "reply"),
                               ReplyPlan("reply", "第二条", "reply"),
                               ReplyPlan("reply", "不应发的第三条", "wait"))
        self.bot.beh["max_chain"] = 2
        await self.push("说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 2)
        self.assertEqual(len(self.brain.calls), 2)

    async def test_94_model_duplicate_continuation_not_sent(self):
        self.setup_model_plans(ReplyPlan("reply", "笑一下就想糊弄过去", "reply"),
                               ReplyPlan("reply", " 笑一下就想糊弄过去！！ ", "reply"),
                               ReplyPlan("reply", "不应继续生成", "wait"))
        await self.push(":)", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.brain.calls), 2)

    async def test_95_model_failed_primary_does_not_continue_or_commit(self):
        self.setup_model_plans(ReplyPlan("reply", "第一条", "reply"),
                               ReplyPlan("reply", "不该生成", "wait"))
        self.client.fail = True
        await self.push("说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.brain.calls), 1)
        self.assertEqual(self.client.sent, [])
        self.assertEqual(self.bot_rows(), [])

    async def test_96_model_failed_secondary_does_not_commit_or_third(self):
        self.setup_model_plans(ReplyPlan("reply", "第一条", "reply"),
                               ReplyPlan("reply", "失败的第二条", "reply"),
                               ReplyPlan("reply", "不应再生成", "wait"))
        original_send = self.client.send_action
        async def send(action, params=None, **kwargs):
            if self.client.sent:
                raise RuntimeError("offline simulated planned send failure")
            return await original_send(action, params, **kwargs)
        self.client.send_action = send
        await self.push("说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.brain.calls), 2)
        self.assertEqual(len(self.bot_rows()), 1)

    async def test_97_model_new_message_cancels_inflight_continuation(self):
        self.setup_model_plans(ReplyPlan("reply", "第一条", "reply"),
                               ReplyPlan("reply", "不该发送", "reply"))
        self.brain.plan_release = asyncio.Event()
        await self.push("说句话", at=[BOT_ID])
        await asyncio.wait_for(self.brain.plan_started.wait(), 1)
        await self.push("我来说一句", user=BOB)
        await asyncio.wait_for(self.bot.wait_idle(), 1)
        self.assertEqual(len(self.client.sent), 1)
        self.assertTrue(self.brain.cancelled)
        self.assertFalse(self.brain.plan_release.is_set())

    async def test_98_model_rechecks_before_actual_send(self):
        self.setup_model_plans(ReplyPlan("reply", "第一条", "reply"),
                               ReplyPlan("reply", "不该发送的第二条", "wait"))
        self.brain.plan_release = asyncio.Event()
        await self.push("说句话", at=[BOT_ID])
        await asyncio.wait_for(self.brain.plan_started.wait(), 1)
        await self.bot._delivery_lock.acquire()
        try:
            self.brain.plan_release.set()
            await asyncio.sleep(0.01)
            await self.push("我来说一句", user=BOB)
        finally:
            self.bot._delivery_lock.release()
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)

    async def test_99_model_continuations_do_not_restart_dialogue_timer_or_turn_count(self):
        self.setup_model_plans(ReplyPlan("reply", "第一条", "reply"),
                               ReplyPlan("reply", "第二条", "reply"),
                               ReplyPlan("reply", "第三条", "wait"))
        self.bot.beh.update(second_reply_delay_min=0.04, second_reply_delay_max=0.04)
        await self.push("说句话", at=[BOT_ID])
        await self.wait_sent_count(1)
        state = self.bot._dialogues[GROUP]
        before = (state.updated_at, state.followups, state.user_text)
        await self.idle()
        state = self.bot._dialogues[GROUP]
        self.assertEqual((state.updated_at, state.followups, state.user_text), before)
        self.assertTrue(state.bot_text.endswith("第三条"))
        self.assertIn("第一条", state.bot_text)
        self.assertIn("第二条", state.bot_text)
        self.assertEqual(len(self.bot.limiter._hits[("group", GROUP)]), 3)
        self.assertEqual(len(self.bot.limiter._hits[("global",)]), 3)

    async def test_100_model_close_cancels_pending_burst(self):
        self.setup_model_plans(ReplyPlan("reply", "第一条", "reply"),
                               ReplyPlan("reply", "不该生成", "wait"))
        self.bot.beh.update(second_reply_delay_min=0.08, second_reply_delay_max=0.08)
        await self.push("说句话", at=[BOT_ID])
        await self.wait_sent_count(1)
        await self.bot.close()
        await asyncio.sleep(0.1)
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.brain.calls), 1)
        self.assertTrue(self.brain.closed)

    async def test_101_model_zero_legacy_rate_and_no_random_gate_still_allows_three(self):
        self.setup_model_plans(ReplyPlan("reply", "第一条", "reply"),
                               ReplyPlan("reply", "第二条", "reply"),
                               ReplyPlan("reply", "第三条", "wait"))
        self.bot.beh["second_reply_rate"] = 0
        with patch("bot.random.random", side_effect=AssertionError("planning must not roll")):
            await self.push("说句话", at=[BOT_ID])
            await self.idle()
        self.assertEqual(len(self.client.sent), 3)

    async def test_102_model_third_cannot_repeat_first(self):
        self.setup_model_plans(ReplyPlan("reply", "第一条完整内容", "reply"),
                               ReplyPlan("reply", "第二条新内容", "reply"),
                               ReplyPlan("reply", "第一条完整内容！", "reply"))
        await self.push("说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 2)
        self.assertEqual(len(self.brain.calls), 3)

    async def test_103_model_generation_failure_does_not_break_next_turn(self):
        self.setup_model_plans(ReplyPlan("reply", "第一条", "reply"),
                               ReplyPlan("reply", "恢复后继续", "wait"))
        self.brain.raise_call = 2
        await self.push("说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(self.bot._second_cancel_events, {})
        self.brain.raise_call = 0
        await self.push("新的明确问题", user=BOB, at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 2)


    async def test_104_real_brain_json_decisions_drive_three_sends_offline(self):
        cfg = test_config()
        cfg["llm"].update(api_key="offline-fixture", base_url="http://127.0.0.1:1", retries=0)
        brain = Brain(cfg["llm"], cfg["safety"], cfg["persona"], 120)
        brain._call = AsyncMock(side_effect=[
            '{"action":"reply","text":"第一句新内容","next":"reply"}',
            '{"action":"reply","text":"第二句新内容","next":"reply"}',
            '{"action":"reply","text":"第三句新内容","next":"wait"}'])
        self.bot.brain = brain
        self.bot.beh.update(model_reply_planning=True, model_burst_max_messages=3,
                            second_reply_rate=0, second_reply_delay_min=.005, second_reply_delay_max=.005)
        await self.push("给我聊两句", at=[BOT_ID])
        await self.idle()
        self.assertEqual(len(self.client.sent), 3)
        self.assertEqual(brain._call.await_count, 3)
        self.assertTrue(all(call.kwargs.get("plan_mode") for call in brain._call.await_args_list))
        texts = ["".join(seg["data"]["text"] for seg in row["message"] if seg["type"] == "text")
                 for row in self.client.sent]
        self.assertEqual(texts, ["第一句新内容", "第二句新内容", "第三句新内容"])

    async def test_105_real_brain_malformed_json_never_leaks_to_group(self):
        cfg = test_config()
        cfg["llm"].update(api_key="offline-fixture", base_url="http://127.0.0.1:1", retries=0)
        brain = Brain(cfg["llm"], cfg["safety"], cfg["persona"], 120)
        brain._call = AsyncMock(return_value='{"action":"reply","text":')
        self.bot.brain = brain
        self.bot.beh["model_reply_planning"] = True
        await self.push("说句话", at=[BOT_ID])
        await self.idle()
        self.assertEqual(self.client.sent, [])
        self.assertEqual(brain._call.await_count, 2)
        self.assertFalse(self.bot.limiter._hits.get(("group", GROUP)))


class ParseTests(unittest.TestCase):
    def test_details_preserves_explicit_target_ids(self):
        parsed = parse_message_details(
            "[CQ:at,qq=20003][CQ:at,qq=10001] 好吗[CQ:reply,id=42]", BOT_ID)
        self.assertTrue(parsed.at_self)
        self.assertEqual(parsed.reply_id, "42")
        self.assertIn(str(BOB), parsed.mentioned_ids)
        self.assertIn(str(BOT_ID), parsed.mentioned_ids)


class ReplyPlanParseTests(unittest.TestCase):
    def test_plan_parser_bare_control_words_and_wrappers(self):
        for raw, action in (("wait", "wait"), ("WAIT", "wait"), ("stop", "stop")):
            plan = parse_reply_plan(raw)
            self.assertEqual((plan.action, plan.text), (action, ""))
        for raw in ("reply", '解释一下：{"action":"wait"}', '```json\n{bad}\n```'):
            self.assertIsNone(parse_reply_plan(raw))

    def test_plan_parser_json_actions_and_fenced_json(self):
        for raw in ('{"action":"reply","text":"先把问题讲清楚","next":"reply"}',
                    '```json\n{"action":"reply","text":"先把问题讲清楚","next":"reply"}\n```'):
            with self.subTest(raw=raw):
                plan = parse_reply_plan(raw)
                self.assertEqual((plan.action, plan.text, plan.next_action),
                                 ("reply", "先把问题讲清楚", "reply"))
        for action in ("wait", "stop"):
            plan = parse_reply_plan(json.dumps({"action": action}))
            self.assertEqual(plan.action, action)
            self.assertEqual(plan.text, "")

    def test_plan_parser_plaintext_compatibility_and_output_filter(self):
        plan = parse_reply_plan("先把问题讲清楚")
        self.assertEqual((plan.action, plan.text, plan.next_action),
                         ("reply", "先把问题讲清楚", "wait"))
        plan = parse_reply_plan('{"action":"reply","text":"禁止测试词","next":"reply"}',
                                blocked=("禁止测试词",), fallback="换个话题")
        self.assertEqual((plan.text, plan.next_action), ("换个话题", "wait"))

    def test_plan_parser_rejects_malformed_unknown_and_wrong_types(self):
        invalid = ('{"action":', '{broken json}', '[]', 'null', '3',
                   '{"action":"unknown","text":"别发"}',
                   '{"action":"reply","text":["别发"]}',
                   '{"action":"reply","text":23}',
                   '{"action":true,"text":"别发"}',
                   '{"action":"reply","text":"别发","next":"unknown"}',
                   '{"action":"reply","text":"别发","next":true}',
                   '{"action":"reply","text":""}')
        for raw in invalid:
            with self.subTest(raw=raw):
                self.assertIsNone(parse_reply_plan(raw))
        for raw in (None, 123, [], {}):
            with self.subTest(type=type(raw).__name__):
                self.assertIsNone(parse_reply_plan(raw))


class BrainPlanTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_brain_json_once_but_plain_fallback_requires_repair(self):
        cfg = test_config()
        cfg["llm"]["api_key"] = "offline-test-not-a-real-key"
        cfg["llm"]["base_url"] = "http://127.0.0.1:1"
        brain = Brain(cfg["llm"], cfg["safety"], cfg["persona"], 120)
        brain._call = AsyncMock(return_value='{"action":"reply","text":"先把问题讲清楚","next":"reply"}')
        try:
            plan = await brain.plan_reply(GROUP, '{"role":"group_member","content":"你猜猜"}',
                                          "离线测试成员", True)
            self.assertEqual((plan.action, plan.text, plan.next_action),
                             ("reply", "先把问题讲清楚", "reply"))
            self.assertEqual(brain._call.await_count, 1)
            self.assertTrue(brain._call.await_args.kwargs.get("plan_mode"))
            messages = brain._call.await_args.args[0]
            prompts = "\n".join(message["content"] for message in messages)
            for key in ("JSON", "action", "text", "next", "wait", "stop", "reply"):
                self.assertIn(key, prompts)
            brain._call.reset_mock(return_value=True)
            brain._call.return_value = "这次只说这一句"
            plan = await brain.plan_reply(GROUP, '{"role":"group_member","content":"说点新的"}',
                                          "离线测试成员", True)
            self.assertEqual((plan.action, plan.text, plan.next_action),
                             ("reply", "这次只说这一句", "wait"))
            self.assertEqual(brain._call.await_count, 2)
        finally:
            await brain.close()

    async def test_real_brain_repairs_plain_and_missing_next_without_forcing_wait(self):
        for raw in ("这是第一条回应", '{"action":"reply","text":"这是第一条回应"}'):
            with self.subTest(raw=raw):
                cfg = test_config()
                cfg["llm"]["api_key"] = "offline-placeholder"
                brain = Brain(cfg["llm"], cfg["safety"], cfg["persona"], 120)
                brain._call = AsyncMock(side_effect=[raw,
                    '{"action":"reply","text":"这是第一条回应","next":"reply"}'])
                try:
                    plan = await brain.plan_reply(GROUP, "离线输入", "离线成员", True)
                    self.assertEqual((plan.action, plan.next_action), ("reply", "reply"))
                    self.assertEqual(brain._call.await_count, 2)
                finally:
                    await brain.close()

    async def test_strict_parser_does_not_invent_wait_for_missing_protocol(self):
        for raw in ("这是普通回复", '{"action":"reply","text":"这是普通回复"}'):
            with self.subTest(raw=raw):
                self.assertIsNone(parse_reply_plan(raw, strict=True))
                self.assertEqual(parse_reply_plan(raw).next_action, "wait")
        self.assertEqual(parse_reply_plan("wait", strict=True).action, "wait")
        self.assertEqual(parse_reply_plan("stop", strict=True).action, "stop")

    async def test_real_brain_planning_budget_preserves_plain_reply_budget(self):
        class Response:
            status = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def text(self):
                return '{"choices":[{"message":{"content":"offline result"}}]}'
        class Session:
            def __init__(self):
                self.payloads = []
            def post(self, url, **kwargs):
                self.payloads.append(kwargs["json"])
                return Response()
            async def close(self):
                pass
        cfg = test_config()
        cfg["llm"].update(api_key="offline-test-not-a-real-key", base_url="http://127.0.0.1:1",
                           max_tokens=100, planning_max_tokens=192)
        brain = Brain(cfg["llm"], cfg["safety"], cfg["persona"], 120)
        session = Session()
        brain._session = session
        try:
            await brain._call([{"role":"user", "content":"offline"}], 1.0, plan_mode=True)
            await brain._call([{"role":"user", "content":"offline"}], 1.0)
            self.assertEqual([row["max_tokens"] for row in session.payloads], [192, 100])
        finally:
            await brain.close()

class CorrectionPromptTests(unittest.TestCase):
    def prompt(self, text, **fields):
        row = {"role": "group_member", "sender": {"id": "20002", "display_name": "成员"},
               "content": text}
        row.update(fields)
        return build_user_prompt(json.dumps(row, ensure_ascii=False), "成员", False, plan_mode=True)

    def test_correction_cues_are_general_not_memory_or_nickname_specific(self):
        for text in ("我说的是你有记忆了", "我的意思是你刚才听错了", "你理解错了",
                     "不是这个意思", "不是我记性好，是你记得前文", "我指的是他，不是你",
                     "不是说你菜，是说我刚才菜", "我是说你刚才接错话了", "你说反了",
                     "你是不是理解错了", "我不是说你菜", "你指的是我吗"):
            with self.subTest(text=text):
                self.assertTrue(build_correction_note(text))
                self.assertIn(build_correction_note(text), self.prompt(text))

    def test_ordinary_banter_and_omission_are_not_forced_corrections(self):
        for text in ("我不是很想玩", "我刚连跪三把", "欧耶，有记忆了", "天青色等烟雨，而我在等你",
                     "你没理解错，确实是他"):
            with self.subTest(text=text):
                self.assertEqual(build_correction_note(text), "")
                self.assertNotIn("【当前发言含澄清/纠正线索】", self.prompt(text))

    def test_only_current_content_not_names_quotes_or_old_history(self):
        cue = "我说的是你有记忆了"
        self.assertNotIn("【当前发言含澄清/纠正线索】",
                         self.prompt("晚上好", sender={"id": "20002", "display_name": cue}, quote=cue))
        old = json.dumps({"role": "group_member", "content": cue}, ensure_ascii=False)
        now = json.dumps({"role": "group_member", "content": "晚上好"}, ensure_ascii=False)
        user = build_user_prompt(old+"\n"+now, "成员", False, plan_mode=True)
        self.assertNotIn("【当前发言含澄清/纠正线索】", user)
        self.assertNotIn("【当前发言含澄清/纠正线索】", self.prompt(cue, role="assistant"))

    def test_pronouns_and_third_person_original_text_are_not_rewritten(self):
        for text in ("我说的是你有记忆了", "我说的是我自己", "我指的是他，不是你"):
            with self.subTest(text=text):
                self.assertIn(json.dumps(text, ensure_ascii=False), self.prompt(text))
        self.assertIn("不能把所有“你”机械替换成机器人", POLICY_CORRECTION)

    def test_correction_policy_preserves_every_style_and_output_protocol(self):
        for style in CORE_IDENTITY:
            for plan_mode in (False, True):
                with self.subTest(style=style, plan_mode=plan_mode):
                    system = build_system_prompt("测试机器人", style, "保留夸张吐槽", "语气覆盖", plan_mode=plan_mode)
                    self.assertIn(CORE_IDENTITY[style], system)
                    self.assertIn(EXAMPLES[style], system)
                    # Correction is now part of the highest-priority foundation,
                    # not a tail patch appended after an aggressive personality.
                    foundation = system.index("【第一层 · 理解与交互底座】")
                    strategy = system.index("【第二层 · 会话策略")
                    expression = system.index("【第三层 · 表达风格")
                    self.assertLess(foundation, system.index(POLICY_CORRECTION))
                    self.assertLess(system.index(POLICY_CORRECTION), strategy)
                    self.assertLess(strategy, expression)
                    self.assertLess(expression, system.index("语气覆盖"))
                    self.assertIn("不得覆盖第一层的语义、对象、明确纠正", system)
                    if plan_mode:
                        self.assertIn(POLICY_PLAN_FORMAT, system)

    def test_legacy_and_nonstring_inputs_do_not_manufacture_corrections(self):
        for value in (None, 123, [], {}):
            self.assertEqual(build_correction_note(value), "")
        for transcript in ("普通旧文本", "{broken json", "[]", '"我说的是你"'):
            self.assertNotIn("【当前发言含澄清/纠正线索】", build_user_prompt(transcript, "成员", False))


class CoherencePromptTests(unittest.TestCase):
    def focus(self, rows):
        transcript = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        text = build_turn_focus(transcript)
        return json.loads(text.splitlines()[1]), transcript, text

    def row(self, role, sender, content, message_id, **fields):
        return dict(role=role, sender={"id": sender}, content=content, message_id=message_id, **fields)

    def test_anchor_parser_is_optional_private_bounded_metadata(self):
        raw = '{"action":"reply","text":"喊我就得认吗","next":"reply","anchor":"回应当前反问"}'
        result = parse_reply_plan(raw, strict=True)
        self.assertEqual((result.text, result.anchor), ("喊我就得认吗", "回应当前反问"))
        self.assertEqual(parse_reply_plan('{"action":"reply","text":"旧协议","next":"wait"}').anchor, "")
        self.assertEqual(parse_reply_plan('{"action":"wait","anchor":"不应发群"}').anchor, "")
        for anchor in (None, [], {}, 1, "长" * 81):
            self.assertIsNone(parse_reply_plan(json.dumps(dict(action="reply", text="不应发群", next="wait", anchor=anchor))))

    def test_rhetorical_focus_does_not_depend_on_correction_regex(self):
        text = "我喊傻子不也是喊你吗"
        self.assertEqual(build_correction_note(text), "")
        data, transcript, focus = self.focus([self.row("group_member", "A", text, "3")])
        self.assertEqual(data["current"]["content"], text)
        self.assertIn(focus, build_user_prompt(transcript, "成员", True, plan_mode=True))
        self.assertIn(POLICY_COHERENCE, build_system_prompt("机器人", plan_mode=True))

    def test_recent_assistant_is_not_assumed_to_reply_to_nearest_member(self):
        rows = [self.row("group_member", "A", "A的问题", "1"),
                self.row("group_member", "B", "B插话", "2"),
                self.row("assistant", "BOT", "其实在答A", "3", reply_to={"message_id": "1", "sender_id": "A"}),
                self.row("group_member", "B", "B的新问题", "4")]
        data, _, _ = self.focus(rows)
        self.assertNotIn("last_exchange", data)
        rows[-1] = self.row("group_member", "A", "A接着反问", "5")
        data, _, _ = self.focus(rows)
        self.assertEqual([r["content"] for r in data["last_exchange"]], ["A的问题", "其实在答A"])

    def test_unknown_reply_never_falls_back_to_another_exchange(self):
        rows = [self.row("assistant", "BOT", "已知旧回复", "1", reply_to={"sender_id": "A"}),
                self.row("group_member", "A", "引用未记录消息", "2", reply_to={"message_id": "unknown"})]
        data, _, _ = self.focus(rows)
        self.assertNotIn("last_exchange", data)
        self.assertEqual(data["association"], "explicit_reply_not_resolved_do_not_guess")

    def test_explicit_third_person_quote_keeps_original_author(self):
        rows = [self.row("group_member", "B", "B说我记性好", "1"),
                self.row("group_member", "A", "他说的是他自己", "2", reply_to={"message_id": "1", "sender_id": "B"})]
        data, _, _ = self.focus(rows)
        self.assertEqual(data["referenced_message"]["sender_id"], "B")
        self.assertEqual(data["current"]["sender_id"], "A")

    def test_legacy_rows_without_recipient_do_not_manufacture_association(self):
        rows = [self.row("group_member", "A", "旧问题", "1"),
                self.row("assistant", "BOT", "归属未知", "2"),
                self.row("group_member", "A", "当前问题", "3")]
        data, _, _ = self.focus(rows)
        self.assertNotIn("last_exchange", data)
        self.assertEqual(build_turn_focus(json.dumps(rows[-2], ensure_ascii=False)), "")
        for invalid in ("普通旧文本", "[]", "null", "{broken"):
            self.assertEqual(build_turn_focus(invalid), "")


class SessionToneTests(unittest.IsolatedAsyncioTestCase):
    """Session tone/state only; scripted responses are not model quality proof."""
    event, push, idle = DialogueTests.event, DialogueTests.push, DialogueTests.idle

    async def asyncSetUp(self):
        await DialogueTests.asyncSetUp(self)
        self.bot.beh.update(followup_window=120, followup_max_turns=6,
            model_reply_planning=True, model_burst_max_messages=3, max_chain=3,
            second_reply_delay_min=.001, second_reply_delay_max=.001, combo_replies=[])
        self.brain = FakePlanBrain([ReplyPlan("reply", f"离线脚本回复{i}", "wait") for i in range(30)])
        self.bot.brain = self.brain

    async def asyncTearDown(self):
        await self.bot.close()

    async def say(self, text, **kwargs):
        await self.push(text, **kwargs)
        await self.idle()

    async def greet(self):
        await self.say("晚上好啊大肥鱼", at=[BOT_ID])
        self.assertEqual(self.bot._session_tone(GROUP, ALICE), "friendly")

    def assert_friendly(self, index=-1):
        tone = self.brain.calls[index]["tone"]
        self.assertIn(self.bot.beh["friendly_tone"], tone)
        self.assertIn("当前会话语气：温和", tone)

    async def test_demo_greeting_game_short_word_preserve_friendly(self):
        await self.greet()
        await self.say("没呢，还在玩三角洲")
        self.assert_friendly()
        await self.say("打钱")
        self.assert_friendly()
        self.assertEqual(len(self.client.sent), 3)

    async def test_failed_greeting_does_not_seed_session(self):
        self.client.fail = True
        await self.say("晚上好", at=[BOT_ID])
        self.assertEqual(self.bot._session_tone(GROUP, ALICE), "")
        self.assertEqual(self.bot._turn_tones, {})

    async def test_model_wait_does_not_seed_session(self):
        self.brain.plans = [ReplyPlan("wait")]
        await self.say("晚上好", at=[BOT_ID])
        self.assertEqual(self.client.sent, [])
        self.assertEqual(self.bot._session_tone(GROUP, ALICE), "")

    async def test_direct_reply_and_poke_keep_session_tone(self):
        await self.greet()
        await self.say("今天有点忙", at=[BOT_ID])
        self.assert_friendly()
        await self.say("还没忙完呢", reply=self.client.sent[-1]["sent_id"])
        self.assert_friendly()
        await self.bot.on_event({"post_type": "notice", "notice_type": "notify",
            "sub_type": "poke", "group_id": GROUP, "user_id": ALICE, "target_id": BOT_ID})
        await self.idle()
        self.assert_friendly()

    async def test_continuations_preserve_tone_without_refreshing_session(self):
        await self.greet()
        recorded = []
        original = self.bot._record_delivery
        def record(*args, **kwargs):
            value = original(*args, **kwargs)
            recorded.append((self.bot._friendly_sessions.get((GROUP, ALICE)),
                             self.bot._dialogues[GROUP].updated_at))
            return value
        self.bot._record_delivery = record
        self.brain.plans = [ReplyPlan("reply", "离线首条", "reply", "本轮游戏"),
                            ReplyPlan("reply", "离线第二条", "reply", "本轮游戏"),
                            ReplyPlan("reply", "离线第三条", "wait", "本轮游戏")]
        await self.say("没呢，还在玩三角洲")
        self.assertEqual(len(recorded), 3)
        self.assertEqual(len(set(recorded)), 1)
        for index in (-3, -2, -1):
            self.assert_friendly(index)

    async def test_other_group_and_other_member_do_not_inherit(self):
        await self.greet()
        await self.say("傻子出来聊天", user=BOB, at=[BOT_ID])
        self.assertEqual(self.brain.calls[-1]["tone"], "")
        await self.say("傻子出来聊天", group=GROUP + 1, at=[BOT_ID])
        self.assertEqual(self.brain.calls[-1]["tone"], "")
        self.assertEqual(self.bot._session_tone(GROUP, ALICE), "friendly")

    async def test_expired_session_no_longer_overrides_default(self):
        await self.greet()
        self.bot._friendly_sessions[(GROUP, ALICE)] -= 121
        await self.say("重新开始普通对话", at=[BOT_ID])
        self.assertEqual(self.brain.calls[-1]["tone"], "")
        self.assertEqual(self.bot._session_tone(GROUP, ALICE), "")

    async def test_topic_change_keeps_tone_explicit_end_clears_it(self):
        await self.greet()
        await self.say("换个话题，今晚吃啥", at=[BOT_ID])
        self.assert_friendly()
        await self.say("不聊了")
        self.assertEqual(self.bot._session_tone(GROUP, ALICE), "")

    async def test_member_interrupt_blocks_implicit_route_not_saved_tone(self):
        await self.greet()
        await self.say("别人普通插话", user=BOB)
        count = len(self.brain.calls)
        await self.say("普通隐式续话不抢答")
        self.assertEqual(len(self.brain.calls), count)
        await self.say("我明确找你继续聊", at=[BOT_ID])
        self.assert_friendly()

    async def test_implicit_turn_limit_does_not_reset_explicit_session_tone(self):
        await self.greet()
        self.bot._dialogues[GROUP].followups = 6
        count = len(self.brain.calls)
        await self.say("普通下一句")
        self.assertEqual(len(self.brain.calls), count)
        await self.say("我明确问你", at=[BOT_ID])
        self.assert_friendly()


class UnderstandingFirstProtocolTests(unittest.TestCase):
    """Compact act/target metadata and prompt structure, not semantic quality."""
    def raw(self, understanding, **updates):
        value = dict(understanding=understanding, action="reply", text="接住当前评价", next="wait",
                     anchor="回应当前评价")
        value.update(updates)
        return json.dumps(value, ensure_ascii=False)

    def test_legacy_plan_constructors_and_json_remain_compatible(self):
        self.assertIsNone(ReplyPlan("reply", "旧三参数", "wait").understanding)
        self.assertIsNone(ReplyPlan("reply", "旧四参数", "wait", "旧焦点").understanding)
        value = parse_reply_plan('{"action":"reply","text":"旧协议","next":"wait"}', strict=True)
        self.assertIsNotNone(value)
        self.assertIsNone(value.understanding)

    def test_valid_frame_is_preserved_without_rewriting_text(self):
        from persona import TurnUnderstanding
        raw = self.raw(dict(act="evaluate", target="bot_reply"), text="嫌这句不够狠？")
        plan = parse_reply_plan(raw, strict=True)
        self.assertEqual(plan.understanding, TurnUnderstanding("evaluate", "bot_reply"))
        self.assertEqual(plan.text, "嫌这句不够狠？")

    def test_invalid_present_frames_are_rejected_not_guessed(self):
        invalid = (None, [], "evaluate", {}, {"act": "evaluate"}, {"target": "bot_reply"},
                   {"act": "unknown", "target": "bot_reply"}, {"act": "evaluate", "target": "who"},
                   {"act": True, "target": "bot_reply"},
                   {"act": "evaluate", "target": "bot_reply", "reason": "not allowed"})
        for value in invalid:
            with self.subTest(value=value):
                self.assertIsNone(parse_reply_plan(self.raw(value), strict=True))

    def test_controls_and_blocked_fallback_drop_private_frame(self):
        valid = dict(act="evaluate", target="bot_reply")
        for action in ("wait", "stop"):
            result = parse_reply_plan(self.raw(valid, action=action, text=""), strict=True)
            self.assertIsNone(result.understanding)
        result = parse_reply_plan(self.raw(valid, text="禁用样本", next="reply"), strict=True,
                                  blocked=["禁用样本"], fallback="边界回复")
        self.assertEqual((result.text, result.next_action), ("边界回复", "wait"))
        self.assertIsNone(result.understanding)

    def test_frame_is_frozen_and_speaker_self_is_not_rewritten(self):
        from dataclasses import FrozenInstanceError
        from persona import TurnUnderstanding
        frame = TurnUnderstanding("clarify", "self")
        self.assertEqual(frame.target, "self")
        with self.assertRaises(FrozenInstanceError):
            frame.target = "bot"

    def test_understanding_foundation_precedes_expression_for_every_style(self):
        from persona import POLICY_PRIORITY, POLICY_UNDERSTANDING
        for style in CORE_IDENTITY:
            prompt = build_system_prompt("测试机器人", style, tone="测试临时语气", plan_mode=True)
            self.assertIn(POLICY_PRIORITY, prompt)
            self.assertIn(POLICY_UNDERSTANDING, prompt)
            self.assertLess(prompt.index(POLICY_UNDERSTANDING), prompt.index(CORE_IDENTITY[style]))
            self.assertIn("测试临时语气", prompt)
            self.assertIn(EXAMPLES[style], prompt)


class UnderstandingFirstDialogueTests(unittest.IsolatedAsyncioTestCase):
    """Scripted understanding choices test delivery control, not model accuracy."""
    event, push, idle = DialogueTests.event, DialogueTests.push, DialogueTests.idle

    async def asyncSetUp(self):
        await DialogueTests.asyncSetUp(self)
        self.bot.beh.update(model_reply_planning=True, model_burst_max_messages=3, max_chain=3,
                            second_reply_delay_min=.001, second_reply_delay_max=.001,
                            combo_replies=[], followup_max_turns=6)

    async def asyncTearDown(self):
        await self.bot.close()

    def plan(self, text, next_action="wait", frame=("evaluate", "bot_reply")):
        from persona import TurnUnderstanding
        return ReplyPlan("reply", text, next_action, "回应本轮评价",
                         TurnUnderstanding(*frame) if frame else None)

    async def replay(self, plans, text="攻击力就这"):
        self.brain = FakePlanBrain(plans)
        self.bot.brain = self.brain
        await self.push(text, at=[BOT_ID])
        await self.idle()

    async def test_same_frame_can_send_two_and_three_with_one_call_per_step(self):
        await self.replay([self.plan("离线首条", "reply"), self.plan("离线第二条")])
        self.assertEqual((len(self.client.sent), len(self.brain.calls)), (2, 2))
        await self.replay([self.plan("另一轮首条", "reply"), self.plan("另一轮第二条", "reply"),
                           self.plan("另一轮第三条")])
        self.assertEqual((len(self.client.sent), len(self.brain.calls)), (5, 3))

    async def test_changed_understanding_stops_before_sending(self):
        await self.replay([self.plan("首条接评价", "reply"), self.plan("错误切成自述", frame=("answer", "self"))])
        self.assertEqual((len(self.client.sent), len(self.brain.calls)), (1, 2))

    async def test_missing_understanding_stops_after_established_frame(self):
        await self.replay([self.plan("首条接评价", "reply"), self.plan("缺失理解的补发", frame=None)])
        self.assertEqual((len(self.client.sent), len(self.brain.calls)), (1, 2))

    async def test_old_protocol_without_frame_keeps_two_message_compatibility(self):
        await self.replay([self.plan("旧协议首条", "reply", None), self.plan("旧协议第二条", frame=None)])
        self.assertEqual((len(self.client.sent), len(self.brain.calls)), (2, 2))

    async def test_later_frame_becomes_fixed_for_remaining_steps(self):
        await self.replay([self.plan("旧首条", "reply", None), self.plan("第二条建立理解", "reply"),
                           self.plan("第三条改变理解", frame=("answer", "self"))])
        self.assertEqual((len(self.client.sent), len(self.brain.calls)), (2, 3))

    async def test_new_user_turn_can_correct_the_previous_understanding(self):
        await self.replay([self.plan("首条回应评价")])
        await self.replay([self.plan("新一轮接住自述纠正", frame=("clarify", "self"))], text="不是说你，是说我自己")
        self.assertEqual(len(self.client.sent), 2)

    async def test_private_metadata_is_never_public_message_text(self):
        await self.replay([self.plan("只发这句正文")])
        text = "".join(segment["data"]["text"] for segment in self.client.sent[0]["message"]
                       if segment["type"] == "text")
        self.assertEqual(text, "只发这句正文")
        for marker in ("understanding", "evaluate", "bot_reply", "回应本轮评价"):
            self.assertNotIn(marker, text)

    async def test_friendly_session_and_greeting_frame_remain_independent(self):
        await self.replay([self.plan("晚上好啊", "reply", ("greet", "bot")),
                           self.plan("今天忙完没", frame=("greet", "bot"))], text="晚上好")
        self.assertEqual(len(self.client.sent), 2)
        self.assertEqual(self.bot._session_tone(GROUP, ALICE), "friendly")
        await self.replay([self.plan("正常接住后续评价")])
        self.assertIn(self.bot.beh["friendly_tone"], self.brain.calls[-1]["tone"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
