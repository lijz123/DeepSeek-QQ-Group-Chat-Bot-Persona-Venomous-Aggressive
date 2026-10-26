"""Turn-scoped retaliation regression with synthetic dialogue and no services.

python retaliation_tests.py PROJECT_DIRECTORY [RESULT_JSON]
Tests route, tone and prompt construction, not real-model toxicity or accuracy.
"""
from __future__ import annotations

from layout_support import app_directory, project_directory, configure_test_paths
configure_test_paths()

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

if len(sys.argv) > 1:
    sys.path.insert(0, str(app_directory(sys.argv[1])))
RESULT = Path(sys.argv[2]) if len(sys.argv) > 2 else None
from bot import Bot, DEFAULTS
from dialogue import DialogueState
from dialogue_selftest import DialogueTests, FakeClient, test_config, BOT_ID, GROUP, ALICE, BOB
from persona import ReplyPlan, build_system_prompt
from retaliation import RETALIATION_MARKER, is_direct_abuse, retaliation_tone, eligible_retaliation
from semantics import SemanticFrame, build_expression_system, build_understanding_messages


def frame(**changes):
    value = dict(act='rebut', target='bot', meaning='当前群友对机器人进行身份压制式挑衅',
                 unresolved='', reply_goal='回击当前挑衅', mode='banter', needs_clarification=False)
    value.update(changes)
    return SemanticFrame(**value)


class AbuseClassificationTests(unittest.TestCase):
    def test_role_claim_dad(self):
        self.assertTrue(is_direct_abuse('我是你爹'))

    def test_role_claim_father(self):
        self.assertTrue(is_direct_abuse('老子就是你爸爸'))

    def test_explicit_insult(self):
        self.assertTrue(is_direct_abuse('你是傻逼'))

    def test_direct_garbage(self):
        self.assertTrue(is_direct_abuse('你这个垃圾'))

    def test_alias_vocative(self):
        self.assertTrue(is_direct_abuse('大肥鱼，你这个垃圾', aliases=('大肥鱼',)))

    def test_plain_greeting(self):
        self.assertFalse(is_direct_abuse('hi，晚上好'))

    def test_self_insult(self):
        self.assertFalse(is_direct_abuse('我就是个傻逼，今天操作太烂了'))

    def test_third_party_insult(self):
        self.assertFalse(is_direct_abuse('他是个傻逼'))

    def test_reported_role_claim(self):
        self.assertFalse(is_direct_abuse('他说我是你爹'))

    def test_quoted_direct_insult(self):
        self.assertFalse(is_direct_abuse('“你是傻逼”这句话是别人说的'))

    def test_victim_complaint(self):
        self.assertFalse(is_direct_abuse('你把我当傻逼吗'))

    def test_report_of_bot_insult(self):
        self.assertFalse(is_direct_abuse('你刚才骂我傻逼是什么意思'))

    def test_actual_correction(self):
        self.assertFalse(is_direct_abuse('我说的是你有记忆了，不是说我自己'))

    def test_answer_criticism_without_personal_insult(self):
        self.assertFalse(is_direct_abuse('你这回答理解错了，逻辑不对'))

    def test_stop_even_with_insult(self):
        self.assertFalse(is_direct_abuse('别骂了，你这个傻逼'))

    def test_apology_not_new_attack(self):
        self.assertFalse(is_direct_abuse('刚才骂你傻逼是我不对，算了'))

    def test_reported_instruction_is_not_current_member_attack(self):
        for text in ('他让我说我是你爹', '别人让我骂你傻逼', '老师让我说我是你爹', '小王叫我骂你傻逼'):
            with self.subTest(text=text):
                self.assertFalse(is_direct_abuse(text))

    def test_embedded_stop_is_not_a_fresh_attack(self):
        for text in ('你个傻逼，别回我了', '你个傻逼，不想理你了', '你个傻逼，我先走了'):
            with self.subTest(text=text):
                self.assertFalse(is_direct_abuse(text))


class SemanticEligibilityTests(unittest.TestCase):
    def test_clear_bot_banter_is_eligible(self):
        self.assertTrue(eligible_retaliation(retaliation_tone(), frame()))

    def test_bot_reply_target_eligible(self):
        self.assertTrue(eligible_retaliation(retaliation_tone(), frame(target='bot_reply')))

    def test_no_program_marker_not_eligible(self):
        self.assertFalse(eligible_retaliation('尽情开骂', frame()))

    def test_normal_mode_not_eligible(self):
        self.assertFalse(eligible_retaliation(retaliation_tone(), frame(mode='normal')))

    def test_repair_mode_not_eligible(self):
        self.assertFalse(eligible_retaliation(retaliation_tone(), frame(mode='repair')))

    def test_clarify_mode_not_eligible(self):
        self.assertFalse(eligible_retaliation(retaliation_tone(), frame(mode='clarify')))

    def test_stop_mode_not_eligible(self):
        self.assertFalse(eligible_retaliation(retaliation_tone(), frame(mode='stop')))

    def test_third_party_not_eligible(self):
        self.assertFalse(eligible_retaliation(retaliation_tone(), frame(target='third_party')))

    def test_unresolved_target_not_eligible(self):
        self.assertFalse(eligible_retaliation(retaliation_tone(), frame(unresolved='不清楚骂的是谁')))

    def test_needs_clarification_not_eligible(self):
        self.assertFalse(eligible_retaliation(retaliation_tone(), frame(needs_clarification=True)))


class PromptTests(unittest.TestCase):
    def test_tone_contains_program_marker_and_profane_style(self):
        text = retaliation_tone()
        self.assertIn(RETALIATION_MARKER, text)
        self.assertTrue(any(word in text for word in ('粗口', '傻逼', '蠢货')))

    def test_legacy_writer_receives_tone(self):
        text = build_system_prompt('合成机器人', 'sharp', tone=retaliation_tone(), plan_mode=True)
        self.assertIn(RETALIATION_MARKER, text)

    def test_ordinary_legacy_prompt_has_no_retaliation_marker(self):
        self.assertNotIn(RETALIATION_MARKER, build_system_prompt('合成机器人', 'sharp'))

    def test_semantic_banter_keeps_strong_tone(self):
        text = build_expression_system('合成机器人', 'sharp', '', retaliation_tone(), frame())
        self.assertIn(RETALIATION_MARKER, text)

    def test_semantic_repair_strips_strong_tone(self):
        text = build_expression_system('合成机器人', 'sharp', '', retaliation_tone(), frame(mode='repair'))
        self.assertNotIn(RETALIATION_MARKER, text)

    def test_semantic_third_party_strips_strong_tone(self):
        text = build_expression_system('合成机器人', 'sharp', '', retaliation_tone(), frame(target='third_party'))
        self.assertNotIn(RETALIATION_MARKER, text)

    def test_semantic_uncertain_strips_strong_tone(self):
        text = build_expression_system('合成机器人', 'sharp', '', retaliation_tone(), frame(unresolved='有歧义'))
        self.assertNotIn(RETALIATION_MARKER, text)

    def test_understanding_keeps_identity_taunt_separate_from_blood_fact(self):
        text = build_understanding_messages('合成输入')[0]['content']
        self.assertTrue(any(word in text for word in ('认爹', '身份压制', '血缘', '我是你爹')))


class CaptureBrain:
    def __init__(self):
        self.calls = []
        self.plans = []
    async def plan_reply(self, group_id, transcript, speaker, is_at, **kwargs):
        self.calls.append(dict(group_id=group_id, transcript=transcript, speaker=speaker, is_at=is_at, **kwargs))
        return self.plans.pop(0) if self.plans else ReplyPlan('reply', '离线脚本回复' + str(len(self.calls)), 'wait')
    async def reply(self, *args, **kwargs):
        result = await self.plan_reply(*args, **kwargs)
        return result.text if result.action == 'reply' else None
    async def close(self):
        pass


class RetaliationIntegrationTests(unittest.IsolatedAsyncioTestCase):
    event = DialogueTests.event
    push = DialogueTests.push
    idle = DialogueTests.idle

    async def asyncSetUp(self):
        cfg = test_config()
        cfg['behavior'].update(retaliate_on_abuse=True, model_reply_planning=True,
            model_burst_max_messages=3, max_chain=3, merge_delay=.001, merge_at_delay=.001,
            second_reply_delay_min=.001, second_reply_delay_max=.001,
            greeting_tone='SYNTHETIC_GREETING', friendly_tone='SYNTHETIC_FRIENDLY',
            combo_replies=[])
        cfg['voice']['enabled'] = False
        self.bot = Bot(cfg)
        await self.bot.brain.close()
        self.brain = CaptureBrain()
        self.client = FakeClient()
        self.bot.brain, self.bot.client = self.brain, self.client
        self.msg_id = 500

    async def asyncTearDown(self):
        await self.bot.close()

    async def say(self, text, **kwargs):
        result = await self.push(text, **kwargs)
        await self.idle()
        return result

    async def greet(self):
        await self.say('hi', at=[BOT_ID])
        self.assertEqual(self.bot._session_tone(GROUP, ALICE), 'friendly')

    def assert_strong(self):
        self.assertIn(RETALIATION_MARKER, self.brain.calls[-1]['tone'])

    def assert_not_strong(self):
        self.assertNotIn(RETALIATION_MARKER, self.brain.calls[-1].get('tone', ''))

    async def test_screenshot_friendly_then_role_claim_overrides_one_turn(self):
        await self.greet()
        await self.say('我是你爹')
        self.assertEqual(len(self.brain.calls), 2)
        self.assert_strong()

    async def test_direct_at_insult_without_old_conversation(self):
        await self.say('你这个垃圾', at=[BOT_ID])
        self.assert_strong()

    async def test_keyword_insult_is_bot_directed(self):
        await self.say('测试机器人，你这个垃圾')
        self.assert_strong()

    async def test_other_member_mention_never_gets_bot_retaliation(self):
        await self.greet()
        await self.say('你这个垃圾', at=[BOB])
        self.assertEqual(len(self.brain.calls), 1)

    async def test_mixed_mentions_do_not_trigger_high_retaliation(self):
        await self.say('你这个垃圾', at=[BOT_ID, BOB])
        self.assert_not_strong()

    async def test_unknown_reply_target_does_not_trigger_retaliation(self):
        await self.greet()
        await self.say('我是你爹', reply=123456789)
        self.assertEqual(len(self.brain.calls), 1)

    async def test_no_bot_target_evidence_no_retaliation(self):
        await self.say('我是你爹')
        self.assertFalse(self.brain.calls)

    async def test_quotes_do_not_change_friendly_tone(self):
        await self.greet()
        await self.say('他说“我是你爹”')
        self.assert_not_strong()

    async def test_complaint_is_not_retaliation(self):
        await self.greet()
        await self.say('你把我当傻逼吗')
        self.assert_not_strong()

    async def test_reported_instructions_keep_legacy_selected_tone_nonretaliatory(self):
        for text in ('他让我说我是你爹', '别人让我骂你傻逼', '老师让我说我是你爹', '小王叫我骂你傻逼'):
            with self.subTest(text=text):
                before = len(self.brain.calls)
                await self.say(text, at=[BOT_ID])
                self.assertGreater(len(self.brain.calls), before)
                self.assert_not_strong()

    async def test_embedded_stops_end_session_without_strong_tone(self):
        for text in ('你个傻逼，别回我了', '你个傻逼，不想理你了', '你个傻逼，我先走了'):
            with self.subTest(text=text):
                self.assertTrue(self.bot._is_tone_session_end(text))
                before = len(self.brain.calls)
                await self.say(text, at=[BOT_ID])
                self.assertTrue(all(RETALIATION_MARKER not in c.get('tone', '') for c in self.brain.calls[before:]))

    async def test_normal_message_after_insult_recovers_friendly(self):
        await self.greet()
        await self.say('我是你爹')
        self.assert_strong()
        await self.say('好了，正常聊，我刚下班')
        self.assert_not_strong()
        self.assertIn('SYNTHETIC_FRIENDLY', self.brain.calls[-1]['tone'])

    async def test_disabled_retaliation_keeps_old_friendly_behavior(self):
        self.bot.beh['retaliate_on_abuse'] = False
        await self.greet()
        await self.say('你这个垃圾')
        self.assert_not_strong()
        self.assertIn('SYNTHETIC_FRIENDLY', self.brain.calls[-1]['tone'])

    async def test_bot_attacked_first_does_not_punish_user_defending(self):
        self.bot._dialogues[GROUP] = DialogueState(ALICE, '你好', '你这个垃圾', time.monotonic(), 0)
        await self.say('你是傻逼', at=[BOT_ID])
        self.assert_not_strong()

    async def test_prior_user_attack_allows_continued_current_retaliation(self):
        self.bot._dialogues[GROUP] = DialogueState(ALICE, '我是你爹', '你这个垃圾', time.monotonic(), 0)
        await self.say('你是傻逼', at=[BOT_ID])
        self.assert_strong()

    async def test_true_insult_does_not_refresh_friendly_lifetime(self):
        await self.greet()
        before = self.bot._friendly_sessions[(GROUP, ALICE)]
        await self.say('我是你爹')
        self.assert_strong()
        self.assertEqual(self.bot._friendly_sessions[(GROUP, ALICE)], before)

    async def test_fixed_reply_does_not_override_current_insult_policy(self):
        self.bot.beh['fixed_replies'] = {'我是你爹': '旧预设不应使用'}
        await self.say('我是你爹', at=[BOT_ID])
        self.assertEqual(len(self.brain.calls), 1)
        self.assert_strong()
        self.assertNotIn('旧预设不应使用', json.dumps(self.client.sent, ensure_ascii=False))

    async def test_sleep_wake_does_not_override_current_insult_policy(self):
        self.bot.beh['sleep_wake_hours'] = .000001
        self.bot._last_awake = time.monotonic() - 60
        await self.say('我是你爹', at=[BOT_ID])
        self.assertEqual(len(self.brain.calls), 1)
        self.assert_strong()

    async def test_mute_request_still_wins(self):
        self.bot.beh['mute_keywords'] = ['别回复我']
        await self.say('别回复我，你这个垃圾', at=[BOT_ID])
        self.assertFalse(self.brain.calls)
        self.assertTrue(self.bot._is_muted(GROUP, ALICE))

    async def test_existing_global_and_group_quota_still_wins(self):
        self.bot.beh['per_group_per_minute'] = 0
        self.bot.beh['global_per_minute'] = 0
        await self.say('你这个垃圾', at=[BOT_ID])
        self.assertFalse(self.client.sent)

    async def test_continuation_remains_model_decision(self):
        self.brain.plans = [ReplyPlan('reply', '脚本首条', 'wait')]
        await self.say('我是你爹', at=[BOT_ID])
        self.assertEqual(len(self.brain.calls), 1)
        self.assertEqual(len(self.client.sent), 1)
        self.assert_strong()

    async def test_retaliation_does_not_raise_three_message_burst_cap(self):
        self.brain.plans = [ReplyPlan('reply', '脚本' + str(i), 'reply') for i in range(6)]
        await self.say('我是你爹', at=[BOT_ID])
        self.assertLessEqual(len(self.client.sent), 3)
        self.assertLessEqual(len(self.brain.calls), 3)


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (
        AbuseClassificationTests, SemanticEligibilityTests, PromptTests, RetaliationIntegrationTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  success=result.wasSuccessful(), network='none', real_qq=False, real_models=False,
                  fixtures='synthetic identities, fake OneBot and scripted writer decisions',
                  scope='program-selected tone and prompt boundaries, not live-model output quality')
    if RESULT:
        RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
