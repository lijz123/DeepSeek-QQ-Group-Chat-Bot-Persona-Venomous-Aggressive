"""Independent OFFLINE session-tone regression. No provider/QQ calls.

python session_tone_tests.py SOURCE_DIR [RESULT_JSON]
Only propagation of program-selected tone/state is tested. Scripted text is not
evidence of a real model's friendliness, comprehension, or semantic quality.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import logging
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
SOURCE = Path(sys.argv[1]).resolve()
RESULT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else ROOT / "session_tone_tests_results.json"
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))

from bot import Bot, DEFAULTS
from persona import ReplyPlan, build_system_prompt
from dialogue_selftest import (BOT_ID, GROUP, ALICE, BOB, DialogueTests,
                              FakeClient, test_config)

GREETING = "TEST_GREETING_OVERRIDE: 温和问候"
FRIENDLY = "TEST_FRIENDLY_OVERRIDE: 温和聊天"
ATTACK = "TEST_ATTACK_OVERRIDE: 原毒舌"
ATTACK_GAME = "TEST_GAME_ATTACK_OVERRIDE: 原游戏毒舌"


class CaptureBrain:
    def __init__(self):
        self.calls = []
        self.plans = []
        self.started = asyncio.Event()
        self.release = None
        self.raise_call = 0

    async def plan_reply(self, group_id, transcript, speaker, is_at, **kw):
        self.calls.append(dict(group_id=group_id, transcript=transcript,
                               speaker=speaker, is_at=is_at, **kw))
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.raise_call == len(self.calls):
            raise RuntimeError("offline scripted generation error")
        return self.plans.pop(0) if self.plans else ReplyPlan(
            "reply", f"离线脚本回复第{len(self.calls)}条", "wait")

    async def reply(self, group_id, transcript, speaker, is_at, **kw):
        plan = await self.plan_reply(group_id, transcript, speaker, is_at, **kw)
        return plan.text if plan.action == "reply" else None

    async def close(self):
        pass


class SessionToneTests(unittest.IsolatedAsyncioTestCase):
    event = DialogueTests.event
    push = DialogueTests.push
    idle = DialogueTests.idle

    async def asyncSetUp(self):
        cfg = test_config()
        cfg["behavior"].update(
            followup_window=120, followup_max_turns=6,
            greeting_tone=GREETING, friendly_tone=FRIENDLY,
            attack_tone=ATTACK, attack_game_tone=ATTACK_GAME,
            model_reply_planning=True, model_burst_max_messages=3,
            max_chain=3, combo_replies=[],
            second_reply_delay_min=.001, second_reply_delay_max=.001,
        )
        self.bot = Bot(cfg)
        await self.bot.brain.close()
        self.brain = CaptureBrain()
        self.client = FakeClient()
        self.bot.brain, self.bot.client = self.brain, self.client
        self.msg_id = 100

    async def asyncTearDown(self):
        await self.bot.close()

    def session(self, group=GROUP, user=ALICE):
        return self.bot._session_tone(group, user)

    def assert_friendly(self, index=-1):
        tone = self.brain.calls[index]["tone"]
        self.assertIn(FRIENDLY, tone)
        self.assertTrue(any(word in tone for word in ("会话", "本轮", "整段", "整轮")), tone)
        self.assertNotIn(ATTACK, tone)
        self.assertNotIn(ATTACK_GAME, tone)

    async def say(self, text, **kwargs):
        mid = await self.push(text, **kwargs)
        await self.idle()
        return mid

    async def greet(self, **kwargs):
        await self.say("晚上好啊大肥鱼", at=[BOT_ID], **kwargs)
        self.assertEqual(self.session(kwargs.get("group", GROUP), kwargs.get("user", ALICE)), "friendly")
        self.assertTrue(GREETING in self.brain.calls[-1]["tone"] or FRIENDLY in self.brain.calls[-1]["tone"])

    async def test_01_opening_success_seeds_identity_scoped_session(self):
        await self.greet()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(self.session(user=BOB), "")
        self.assertEqual(self.session(group=GROUP + 1), "")

    async def test_02_demo_game_then_short_word_keep_friendly(self):
        await self.greet()
        await self.say("没呢，还在玩三角洲")
        self.assert_friendly()
        await self.say("打钱")
        self.assert_friendly()
        self.assertEqual(len(self.brain.calls), 3)
        self.assertEqual(len(self.client.sent), 3)

    async def test_03_direct_at_remains_same_friendly_session(self):
        await self.greet()
        await self.say("今晚打算玩点别的", at=[BOT_ID])
        self.assert_friendly()

    async def test_04_quote_reply_to_bot_keeps_friendly(self):
        await self.greet()
        await self.say("还没忙完呢", reply=self.client.sent[-1]["sent_id"])
        self.assert_friendly()

    async def test_05_keyword_keeps_friendly(self):
        await self.greet()
        await self.say("测试机器人，你觉得呢")
        self.assert_friendly()

    async def test_06_poke_keeps_same_user_friendly(self):
        await self.greet()
        await self.bot.on_event({"post_type": "notice", "notice_type": "notify",
            "sub_type": "poke", "group_id": GROUP, "user_id": ALICE,
            "target_id": BOT_ID})
        await self.idle()
        self.assert_friendly()
        self.assertEqual(len(self.brain.calls), 2)

    async def test_07_taunt_does_not_silently_switch_locked_session(self):
        await self.greet()
        await self.say("你这个垃圾")
        self.assert_friendly()

    async def test_08_different_user_does_not_inherit(self):
        await self.greet()
        await self.say("我在玩游戏", user=BOB, at=[BOT_ID])
        self.assertNotIn(FRIENDLY, self.brain.calls[-1]["tone"])
        self.assertEqual(self.session(user=BOB), "")

    async def test_09_different_group_does_not_inherit(self):
        await self.greet()
        await self.say("我在玩游戏", group=GROUP + 1, at=[BOT_ID])
        self.assertNotIn(FRIENDLY, self.brain.calls[-1]["tone"])
        self.assertEqual(self.session(group=GROUP + 1), "")

    async def test_10_other_group_activity_does_not_clear_original(self):
        await self.greet()
        await self.say("旁边群聊一句", user=BOB, group=GROUP + 1)
        self.assertEqual(self.session(), "friendly")
        await self.say("继续聊刚才的游戏", at=[BOT_ID])
        self.assert_friendly()

    async def test_11_other_member_interrupt_changes_route_not_original_tone(self):
        await self.greet()
        await self.say("另一位群友插句话", user=BOB)
        self.assertTrue(self.bot._dialogues[GROUP].interrupted)
        self.assertEqual(self.session(), "friendly")
        count = len(self.brain.calls)
        await self.say("普通隐式续话不应抢答")
        self.assertEqual(len(self.brain.calls), count)
        await self.say("我明确叫你继续聊", at=[BOT_ID])
        self.assert_friendly()

    async def test_12_explicit_at_after_implicit_turn_cap_still_friendly(self):
        await self.greet()
        self.bot._dialogues[GROUP].followups = 6
        count = len(self.brain.calls)
        await self.say("没有指定对象的普通下一句")
        self.assertEqual(len(self.brain.calls), count)
        await self.say("现在我明确问你", at=[BOT_ID])
        self.assert_friendly()

    async def test_13_timeout_no_longer_inherits(self):
        await self.greet()
        self.bot._friendly_sessions[(GROUP, ALICE)] -= 121
        self.assertEqual(self.session(), "")
        await self.say("重新开始普通聊天", at=[BOT_ID])
        self.assertNotIn(FRIENDLY, self.brain.calls[-1]["tone"])

    async def test_14_actual_delivered_primary_refreshes_window(self):
        await self.greet()
        self.bot._friendly_sessions[(GROUP, ALICE)] -= 80
        previous = self.bot._friendly_sessions[(GROUP, ALICE)]
        await self.say("续聊一条", at=[BOT_ID])
        self.assert_friendly()
        self.assertGreater(self.bot._friendly_sessions[(GROUP, ALICE)], previous + 70)

    async def test_15_explicit_end_clears_friendly(self):
        await self.greet()
        await self.say("不聊了")
        self.assertEqual(self.session(), "")
        await self.say("另开普通一轮", at=[BOT_ID])
        self.assertNotIn(FRIENDLY, self.brain.calls[-1]["tone"])

    async def test_16_topic_change_is_not_end_of_friendly_conversation(self):
        await self.greet()
        await self.say("换个话题，今晚吃啥", at=[BOT_ID])
        self.assert_friendly()
        self.assertEqual(self.session(), "friendly")
        await self.say("说点别的，你今天忙啥", at=[BOT_ID])
        self.assert_friendly()

    async def test_17_explicit_address_to_other_ends_own_session(self):
        await self.greet()
        await self.say("这句话是跟你聊", at=[BOB])
        self.assertEqual(self.session(), "")

    async def test_18_mute_ends_own_session(self):
        await self.greet()
        self.bot.beh["mute_keywords"] = ["肥鱼闭嘴"]
        await self.say("肥鱼闭嘴")
        self.assertEqual(self.session(), "")
        self.assertTrue(self.bot._is_muted(GROUP, ALICE))

    async def test_19_wait_does_not_create_unsent_friendly_session(self):
        self.brain.plans = [ReplyPlan("wait")]
        await self.say("晚上好", at=[BOT_ID])
        self.assertEqual(self.client.sent, [])
        self.assertEqual(self.session(), "")
        self.assertEqual(self.bot._turn_tones, {})

    async def test_20_stop_does_not_create_unsent_friendly_session(self):
        self.brain.plans = [ReplyPlan("stop")]
        await self.say("晚上好", at=[BOT_ID])
        self.assertEqual(self.client.sent, [])
        self.assertEqual(self.session(), "")
        self.assertEqual(self.bot._turn_tones, {})

    async def test_21_failed_send_does_not_create_session(self):
        self.client.fail = True
        await self.say("晚上好", at=[BOT_ID])
        self.assertEqual(self.session(), "")
        self.assertEqual(self.bot._turn_tones, {})

    async def test_22_failed_followup_does_not_extend_session(self):
        await self.greet()
        self.bot._friendly_sessions[(GROUP, ALICE)] -= 70
        previous = self.bot._friendly_sessions[(GROUP, ALICE)]
        self.client.fail = True
        await self.say("还没玩完", at=[BOT_ID])
        self.assertEqual(self.bot._friendly_sessions[(GROUP, ALICE)], previous)

    async def test_23_wait_does_not_extend_existing_session(self):
        await self.greet()
        self.bot._friendly_sessions[(GROUP, ALICE)] -= 70
        previous = self.bot._friendly_sessions[(GROUP, ALICE)]
        self.brain.plans = [ReplyPlan("wait")]
        await self.say("还没玩完", at=[BOT_ID])
        self.assertEqual(self.bot._friendly_sessions[(GROUP, ALICE)], previous)

    async def test_24_two_model_continuations_keep_friendly(self):
        await self.greet()
        self.brain.plans = [ReplyPlan("reply", "离线首条", "reply", "当前游戏"),
                            ReplyPlan("reply", "离线第二条", "reply", "当前游戏"),
                            ReplyPlan("reply", "离线第三条", "wait", "当前游戏")]
        await self.say("没呢，还在玩三角洲")
        self.assertEqual(len(self.client.sent), 4)
        for index in (-3, -2, -1):
            self.assert_friendly(index)
        self.assertEqual(self.bot._turn_tones, {})

    async def test_25_continuations_do_not_refresh_session_timestamp(self):
        await self.greet()
        recorded = []
        original = self.bot._record_delivery
        def record(*args, **kw):
            result = original(*args, **kw)
            recorded.append((args[-1], self.bot._friendly_sessions.get((GROUP, ALICE)),
                             self.bot._dialogues[GROUP].updated_at))
            return result
        self.bot._record_delivery = record
        self.brain.plans = [ReplyPlan("reply", "离线首条", "reply", "当前游戏"),
                            ReplyPlan("reply", "离线第二条", "reply", "当前游戏"),
                            ReplyPlan("reply", "离线第三条", "wait", "当前游戏")]
        await self.say("没呢，还在玩三角洲")
        self.assertEqual(len(recorded), 3)
        self.assertEqual(recorded[1][0], "continuation")
        self.assertEqual(recorded[2][0], "continuation")
        self.assertEqual(len({item[1] for item in recorded}), 1)
        self.assertEqual(len({item[2] for item in recorded}), 1)

    async def test_26_legacy_optional_second_keeps_friendly(self):
        await self.greet()
        self.bot.beh.update(model_reply_planning=False, second_reply_rate=1)
        await self.say("没呢，还在玩三角洲")
        self.assertEqual(len(self.brain.calls), 3)
        self.assert_friendly(-2)
        self.assert_friendly(-1)

    async def test_27_friendly_topic_opening_seeds_without_greeting(self):
        self.bot.beh["topic_rates"] = {"daily": {"attack": 0, "friendly": 1}}
        await self.say("今天心情不错")
        self.assertEqual(self.session(), "friendly")
        await self.say("我准备玩会游戏")
        self.assert_friendly()

    async def test_28_sharp_opening_does_not_create_friendly_session(self):
        await self.say("傻子出来聊天", at=[BOT_ID])
        self.assertEqual(self.session(), "")
        self.assertNotIn(FRIENDLY, self.brain.calls[-1]["tone"])
        await self.say("我说你呢")
        self.assertNotIn(FRIENDLY, self.brain.calls[-1]["tone"])

    async def test_29_no_identity_tone_api_preserves_default_attack(self):
        await self.greet()
        self.assertEqual(self.bot._tone_for("attack", "日常"), ATTACK)
        self.assertEqual(self.bot._tone_for("attack", "打野装备"), ATTACK_GAME)
        self.assertEqual(self.bot._tone_for("at", "普通内容"), "")

    async def test_30_tone_lookup_does_not_seed_state(self):
        for reason in ("greeting", "friendly", "at", "followup", "poke"):
            self.bot._tone_for(reason, "普通内容", group_id=GROUP, user_id=ALICE)
        self.assertEqual(self.session(), "")
        self.assertEqual(self.bot._friendly_sessions, {})

    async def test_31_closing_goodnight_replies_warmly_but_does_not_seed(self):
        await self.say("晚安", at=[BOT_ID])
        self.assertEqual(len(self.client.sent), 1)
        self.assertTrue(GREETING in self.brain.calls[-1]["tone"] or FRIENDLY in self.brain.calls[-1]["tone"])
        self.assertEqual(self.session(), "")

    async def test_32_new_greeting_reopens_after_end(self):
        await self.greet()
        await self.say("不聊了")
        self.assertEqual(self.session(), "")
        await self.greet()
        await self.say("这次聊点别的")
        self.assert_friendly()

    async def test_33_fixed_rule_remains_direct_and_does_not_refresh_friendly(self):
        await self.greet()
        stamp = self.bot._friendly_sessions[(GROUP, ALICE)]
        self.bot.beh["fixed_replies"] = {"固定样本": "离线固定原文"}
        await self.say("固定样本")
        self.assertEqual(len(self.brain.calls), 1)
        self.assertEqual(self.bot._friendly_sessions[(GROUP, ALICE)], stamp)
        self.assertEqual(self.session(), "friendly")

    async def test_34_close_releases_tone_caches(self):
        await self.greet()
        await self.bot.close()
        self.assertEqual(self.bot._friendly_sessions, {})
        self.assertEqual(self.bot._turn_tones, {})

    async def test_35_wait_after_interruption_keeps_existing_cache_without_refresh(self):
        await self.greet()
        stamp = self.bot._friendly_sessions[(GROUP, ALICE)]
        await self.say("另一人插话", user=BOB)
        self.brain.plans = [ReplyPlan("wait")]
        await self.say("我明确找你", at=[BOT_ID])
        self.assert_friendly()
        self.assertEqual(self.bot._friendly_sessions[(GROUP, ALICE)], stamp)

    async def test_36_parallel_groups_capture_tone_independently(self):
        self.brain.release = asyncio.Event()
        await self.push("晚上好", at=[BOT_ID])
        await asyncio.wait_for(self.brain.started.wait(), 1)
        self.assertEqual(self.session(), "")
        await self.push("傻子出来聊天", group=GROUP + 1, user=BOB, at=[BOT_ID])
        for _ in range(50):
            if len(self.brain.calls) == 2:
                break
            await asyncio.sleep(.01)
        self.assertEqual(len(self.brain.calls), 2)
        by_group = {row["group_id"]: row for row in self.brain.calls}
        self.assertTrue(GREETING in by_group[GROUP]["tone"] or FRIENDLY in by_group[GROUP]["tone"])
        self.assertNotIn(FRIENDLY, by_group[GROUP + 1]["tone"])
        self.brain.release.set()
        await self.idle()
        self.assertEqual(self.session(), "friendly")
        self.assertEqual(self.session(group=GROUP + 1, user=BOB), "")
        self.assertEqual(self.bot._turn_tones, {})

    async def test_37_generation_error_does_not_commit_or_leak_captured_tone(self):
        self.brain.raise_call = 1
        await self.say("晚上好", at=[BOT_ID])
        self.assertEqual(self.client.sent, [])
        self.assertEqual(self.session(), "")
        self.assertEqual(self.bot._turn_tones, {})

    async def test_38_generation_and_ack_latency_do_not_flip_captured_friendly(self):
        await self.greet()
        self.bot._friendly_sessions[(GROUP, ALICE)] -= 119
        self.brain.release = asyncio.Event()
        self.brain.started.clear()
        send_started, ack_release = asyncio.Event(), asyncio.Event()
        original = self.client.send_action
        async def gate_send(*args, **kwargs):
            send_started.set()
            await ack_release.wait()
            return await original(*args, **kwargs)
        self.client.send_action = gate_send
        await self.push("继续聊刚才的内容", at=[BOT_ID])
        await asyncio.wait_for(self.brain.started.wait(), 1)
        self.assert_friendly()
        self.bot._friendly_sessions[(GROUP, ALICE)] -= 2
        self.brain.release.set()
        await asyncio.wait_for(send_started.wait(), 1)
        self.assertEqual(self.session(), "")
        ack_release.set()
        await self.idle()
        self.assertEqual(self.session(), "friendly")
        self.assertLess(time.monotonic() - self.bot._friendly_sessions[(GROUP, ALICE)], 1)

    async def pending_closing(self, closing):
        await self.greet()
        self.brain.release = asyncio.Event()
        self.brain.started.clear()
        self.brain.plans = [ReplyPlan("reply", "旧生成不该发送", "reply", "旧落点"),
                            ReplyPlan("reply", "离线温和收尾", "wait", "收尾")]
        await self.push("继续普通聊天", at=[BOT_ID])
        await asyncio.wait_for(self.brain.started.wait(), 1)
        await self.push(closing, at=[BOT_ID])
        self.brain.release.set()
        await self.idle()
        sent_texts = [seg["data"]["text"] for row in self.client.sent
                      for seg in row["message"] if seg["type"] == "text"]
        self.assertNotIn("旧生成不该发送", sent_texts)
        self.assertIn("离线温和收尾", sent_texts)
        self.assertEqual(self.session(), "")
        self.assertEqual(self.bot._turn_tones, {})

    async def test_39_goodnight_during_pending_generation_prevents_resurrection(self):
        await self.pending_closing("晚安")

    async def test_40_bye_during_pending_generation_prevents_resurrection(self):
        await self.pending_closing("再见")


def run():
    logging.disable(logging.CRITICAL)
    output, stdout, stderr = io.StringIO(), io.StringIO(), io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SessionToneTests)
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    ledger = {
        "kind": "OFFLINE_TONE_PROPAGATION_AND_STATE_REGRESSION_NOT_REAL_MODEL_QUALITY",
        "source": str(SOURCE), "real_model_calls": 0, "qq_network_sends": 0,
        "source_hashes": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                          for name in ("bot.py", "dialogue.py", "persona.py")},
        "tests_run": result.testsRun, "successful": result.wasSuccessful(),
        "failures": [{"test": str(test), "traceback": trace} for test, trace in result.failures],
        "errors": [{"test": str(test), "traceback": trace} for test, trace in result.errors],
        "literal_stdout": stdout.getvalue(), "literal_stderr": stderr.getvalue(),
        "literal_test_output": output.getvalue(), "exit_status": 0 if result.wasSuccessful() else 1,
    }
    RESULT.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output.getvalue(), end="")
    print(json.dumps({key: ledger[key] for key in ("kind", "tests_run", "successful", "exit_status")}, ensure_ascii=False))
    return ledger["exit_status"]


if __name__ == "__main__":
    raise SystemExit(run())
