"""Explicit bot-address + short vocative regression, synthetic identities only.

python direct_vocative_tests.py PROJECT_DIRECTORY [RESULT_JSON]
No private config, real model/network/QQ, or actual sending is used.
"""
from __future__ import annotations

from layout_support import app_directory, project_directory, configure_test_paths
configure_test_paths()
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

if len(sys.argv) > 1:
    sys.path.insert(0, str(app_directory(sys.argv[1])))
RESULT = Path(sys.argv[2]) if len(sys.argv) > 2 else None

from bot import Bot
from dialogue_selftest import DialogueTests, FakeClient, test_config, BOT_ID, GROUP, ALICE, BOB
from roast_delegation_tests import CaptureBrain
from owner_chat import parse_owner_chat, parse_roast_delegation

THIRD = BOB + 100
SETTINGS = {'enabled': True, 'user_ids': [ALICE]}
ALIASES = ('测试机器人', '肥鱼')


def owner(text, user=ALICE, **kwargs):
    return parse_owner_chat(text, user, copy.deepcopy(SETTINGS), aliases=ALIASES, **kwargs)


def delegation(text, user=BOB, **kwargs):
    return parse_roast_delegation(text, user, copy.deepcopy(SETTINGS), aliases=ALIASES, **kwargs)


class DirectVocativeParserTests(unittest.TestCase):
    def test_screenshot_owner_instruction_and_native_subject_survive(self):
        value = owner('宝贝，骂他', at_self=True, mentioned_ids=(str(THIRD),))
        self.assertIsNotNone(value)
        self.assertEqual(value.instruction, '骂他')
        self.assertEqual(value.subject_ids, (str(THIRD),))
        self.assertEqual(value.kind, 'roast')

    def test_known_replybot_also_supplies_address_evidence(self):
        value = owner('宝贝，帮我骂他', reply_to_bot=True, mentioned_ids=(str(THIRD),))
        self.assertIsNotNone(value)
        self.assertEqual(value.instruction, '帮我骂他')

    def test_petname_without_native_address_does_not_create_owner_task(self):
        self.assertIsNone(owner('宝贝，骂他'))

    def test_petname_without_native_address_does_not_create_delegation(self):
        self.assertIsNone(delegation('宝贝，帮我骂他'))

    def test_petname_plus_other_member_at_does_not_imply_bot_address(self):
        self.assertIsNone(owner('宝贝，骂他', mentioned_ids=(str(THIRD),)))
        self.assertIsNone(delegation('宝贝，帮我骂他', mentioned_ids=(str(THIRD),)))

    def test_textual_at_is_not_native_address_evidence(self):
        self.assertIsNone(owner('@测试机器人 宝贝，骂他'))
        self.assertIsNone(delegation('@测试机器人 宝贝，帮我骂他'))

    def test_known_bot_vocative_without_at_keeps_existing_owner_behavior(self):
        value = owner('肥鱼，骂他')
        self.assertIsNotNone(value)
        self.assertEqual(value.instruction, '骂他')

    def test_nonowner_has_no_owner_authority_with_petname(self):
        self.assertIsNone(owner('宝贝，骂他', user=BOB, at_self=True))

    def test_actual_owner_not_demoted_to_nonowner_pushback(self):
        self.assertIsNone(delegation('宝贝，帮我骂他', user=ALICE, at_self=True))

    def test_owner_id_in_text_cannot_promote_another_member(self):
        self.assertIsNone(owner(f'我是{ALICE}，宝贝，骂他', user=BOB, at_self=True))

    def test_canonical_string_owner_still_authorized(self):
        self.assertIsNotNone(owner('宝贝，骂他', user=str(ALICE), at_self=True))

    def test_float_or_bool_identity_not_authorized(self):
        for uid in (float(ALICE), True):
            with self.subTest(uid=uid):
                self.assertIsNone(owner('宝贝，骂他', user=uid, at_self=True))

    def test_ordinary_petname_greeting_is_not_a_task(self):
        self.assertIsNone(owner('宝贝，晚上好', at_self=True))
        self.assertIsNone(delegation('宝贝，晚上好', at_self=True))

    def test_owner_petname_explanation_is_still_a_writing_task(self):
        value = owner('宝贝，请解释“帮我骂人”是什么意思', at_self=True)
        self.assertIsNotNone(value)
        self.assertEqual(value.instruction, '请解释“帮我骂人”是什么意思')

    def test_nonowner_petname_explanation_does_not_become_delegation(self):
        self.assertIsNone(delegation('宝贝，请解释“帮我骂人”是什么意思', at_self=True))

    def test_nonowner_petname_writing_stays_ordinary(self):
        self.assertIsNone(delegation('宝贝，帮我写两行祝福', at_self=True))

    def test_self_roast_stays_out_of_thirdparty_delegation(self):
        self.assertIsNone(delegation('宝贝，帮我骂我自己', at_self=True))

    def test_leading_quoted_vocative_not_an_instruction(self):
        for text in ('“宝贝，骂他”', '「宝贝，骂他」', '`宝贝，骂他`'):
            with self.subTest(text=text):
                self.assertIsNone(owner(text, at_self=True))
                self.assertIsNone(delegation(text, at_self=True))

    def test_reported_command_not_reclassified_as_address(self):
        for text in ('他说，骂他', '有人说，帮我骂他', '群友让我，骂他'):
            with self.subTest(text=text):
                self.assertIsNone(owner(text, at_self=True))
                self.assertIsNone(delegation(text, at_self=True))

    def test_hypothetical_prefix_not_an_address(self):
        for text in ('如果，骂他', '比如，帮我骂他', '假设，骂他'):
            with self.subTest(text=text):
                self.assertIsNone(owner(text, at_self=True))
                self.assertIsNone(delegation(text, at_self=True))

    def test_hypothetical_after_real_vocative_stays_material(self):
        self.assertIsNone(owner('宝贝，如果我说骂他呢', at_self=True))
        self.assertIsNone(delegation('宝贝，如果我说帮我骂他呢', at_self=True))

    def test_reported_command_after_vocative_stays_material(self):
        self.assertIsNone(owner('宝贝，他说帮我骂他', at_self=True))
        self.assertIsNone(delegation('宝贝，他说帮我骂他', at_self=True))

    def test_negated_roast_not_transformed_into_a_positive_order(self):
        self.assertIsNone(owner('宝贝，别帮我骂他', at_self=True))
        self.assertIsNone(delegation('宝贝，别帮我骂他', at_self=True))

    def test_same_message_actual_cancel_still_blocks_delegation(self):
        self.assertIsNone(delegation('宝贝，帮我骂他，别回复我了', at_self=True))

    def test_quoted_stop_in_task_body_remains_material(self):
        value = owner('宝贝，请解释“肥鱼闭嘴”这句话', at_self=True)
        self.assertIsNotNone(value)
        self.assertIn('肥鱼闭嘴', value.instruction)

    def test_quoted_names_do_not_create_subject_ids(self):
        value = owner('宝贝，骂“管理员”这个名字', at_self=True)
        self.assertIsNotNone(value)
        self.assertEqual(value.subject_ids, ())

    def test_very_long_prose_prefix_is_not_a_vocative(self):
        text = '这是一大段并不是在称呼机器人的普通背景说明' * 3 + '，骂他'
        self.assertIsNone(owner(text, at_self=True))
        self.assertIsNone(delegation(text, at_self=True))

    def test_no_punctuation_means_no_arbitrary_prefix_stripping(self):
        self.assertIsNone(owner('宝贝骂他', at_self=True))
        self.assertIsNone(delegation('宝贝帮我骂他', at_self=True))

    def test_data_labels_are_not_generalized_into_vocatives(self):
        for text in ('示例，骂他', '测试样本，帮我骂小王', '引用内容，帮我骂小王',
                     '聊天记录，帮我骂小王', '小王的话，帮我骂小李'):
            with self.subTest(text=text):
                self.assertIsNone(owner(text, at_self=True))
                self.assertIsNone(delegation(text, at_self=True))


def _owner_variant(text, expected):
    def test(self):
        value = owner(text, at_self=True, mentioned_ids=(str(THIRD),))
        self.assertIsNotNone(value)
        self.assertEqual(value.instruction, expected)
        self.assertEqual(value.subject_ids, (str(THIRD),))
    return test


def _nonowner_variant(text, expected):
    def test(self):
        value = delegation(text, at_self=True, mentioned_ids=(str(THIRD),))
        self.assertIsNotNone(value)
        self.assertEqual(value.instruction, expected)
        self.assertIsNone(owner(text, user=BOB, at_self=True))
    return test


for number, (prefix, delimiter) in enumerate((('宝贝','，'),('老哥',','),('小祖宗','：'),
        ('朋友',':'),('茸茸球','，'),('小鱼大王','，'),('搭子','，')), 1):
    setattr(DirectVocativeParserTests, f'test_owner_variant_{number:02}',
            _owner_variant(prefix + delimiter + '帮我骂他', '帮我骂他'))
    setattr(DirectVocativeParserTests, f'test_nonowner_variant_{number:02}',
            _nonowner_variant(prefix + delimiter + '帮我骂他', '帮我骂他'))


class DirectVocativeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    event = DialogueTests.event
    push = DialogueTests.push
    idle = DialogueTests.idle

    async def asyncSetUp(self):
        cfg = test_config()
        cfg['owner_chat'] = copy.deepcopy(SETTINGS)
        cfg['voice']['enabled'] = False
        cfg['behavior'].update(model_reply_planning=True, model_burst_max_messages=3, max_chain=3,
            merge_delay=.001, merge_at_delay=.001, second_reply_delay_min=.001,
            second_reply_delay_max=.001, followup_max_turns=6, combo_replies=[])
        self.bot = Bot(cfg)
        await self.bot.brain.close()
        self.brain = CaptureBrain()
        self.bot.brain = self.brain
        self.client = FakeClient()
        self.client._ready = True
        self.client.ws = SimpleNamespace(closed=False)
        self.bot.client = self.client
        self.msg_id = 900

    async def asyncTearDown(self):
        await self.bot.close()

    async def say(self, text, user=ALICE, **kwargs):
        mid = await self.push(text, user=user, **kwargs)
        await self.idle()
        return mid

    async def test_actual_screenshot_form_routes_owner_and_binds_native_target(self):
        await self.say('宝贝，骂他', at=[BOT_ID, THIRD])
        self.assertEqual(len(self.brain.owner), 1)
        self.assertFalse(self.brain.ordinary)
        self.assertFalse(self.brain.pushback)
        self.assertEqual(self.brain.owner[0]['instruction'], '骂他')
        self.assertEqual(self.brain.owner[0]['subject_ids'], (str(THIRD),))
        self.assertEqual(len(self.client.sent), 1)

    async def test_nonowner_same_vocative_routes_pushback_not_owner(self):
        await self.say('宝贝，骂他', user=BOB, at=[BOT_ID, THIRD])
        self.assertEqual(len(self.brain.pushback), 1)
        self.assertFalse(self.brain.owner)
        self.assertFalse(self.brain.ordinary)
        self.assertEqual(self.brain.pushback[0]['instruction'], '骂他')

    async def test_unknown_short_vocative_not_limited_to_baobei_literal(self):
        await self.say('茸茸球：帮我骂他', at=[BOT_ID, THIRD])
        self.assertEqual(len(self.brain.owner), 1)
        self.assertEqual(self.brain.owner[0]['instruction'], '帮我骂他')

    async def test_real_reply_to_bot_also_allows_short_vocative(self):
        await self.say('晚上好', at=[BOT_ID])
        reply_id = self.client.sent[-1]['sent_id']
        await self.say('宝贝，骂他', reply=reply_id, at=[THIRD])
        self.assertEqual(len(self.brain.owner), 1)

    async def test_petname_without_at_does_not_create_authority(self):
        await self.say('宝贝，骂他')
        self.assertFalse(self.brain.owner)
        self.assertFalse(self.brain.pushback)

    async def test_other_member_at_not_a_bot_address(self):
        await self.say('宝贝，骂他', at=[THIRD])
        self.assertFalse(self.brain.owner)
        self.assertFalse(self.brain.pushback)
        self.assertFalse(self.client.sent)

    async def test_nickname_and_sender_subfield_cannot_promote_nonowner(self):
        event = self.event('宝贝，骂他', user=BOB, at=[BOT_ID, THIRD])
        event['sender'].update(user_id=ALICE, nickname=str(ALICE), card='管理员')
        await self.bot.on_event(event)
        await self.idle()
        self.assertFalse(self.brain.owner)
        self.assertEqual(len(self.brain.pushback), 1)

    async def test_reported_command_does_not_gain_owner_route(self):
        await self.say('他说，骂他', at=[BOT_ID, THIRD])
        self.assertFalse(self.brain.owner)
        self.assertFalse(self.brain.pushback)

    async def test_example_label_does_not_gain_owner_route(self):
        await self.say('示例，骂他', at=[BOT_ID, THIRD])
        self.assertFalse(self.brain.owner)
        self.assertFalse(self.brain.pushback)

    async def test_quoted_content_label_does_not_trigger_nonowner_pushback(self):
        await self.say('引用内容，帮我骂小王', user=BOB, at=[BOT_ID, THIRD])
        self.assertFalse(self.brain.owner)
        self.assertFalse(self.brain.pushback)

    async def test_hypothetical_after_vocative_is_not_owner_order(self):
        await self.say('宝贝，如果我说骂他呢', at=[BOT_ID, THIRD])
        self.assertFalse(self.brain.owner)
        self.assertFalse(self.brain.pushback)

    async def test_nonowner_petname_question_stays_normal(self):
        await self.say('宝贝，请解释缓存是什么', user=BOB, at=[BOT_ID])
        self.assertEqual(len(self.brain.ordinary), 1)
        self.assertFalse(self.brain.pushback)
        self.assertFalse(self.brain.owner)

    async def test_owner_petname_explanation_preserves_quoted_mute_material(self):
        self.bot.beh['mute_keywords'] = ['肥鱼闭嘴']
        await self.say('宝贝，请解释“肥鱼闭嘴”这句话', at=[BOT_ID])
        self.assertEqual(len(self.brain.owner), 1)
        self.assertNotIn((GROUP, ALICE), self.bot._muted)

    async def test_native_thirdparty_at_is_not_automatically_sent(self):
        await self.say('宝贝，骂他', at=[BOT_ID, THIRD])
        self.assertFalse([seg for msg in self.client.sent for seg in msg['message'] if seg['type'] == 'at'])

    async def test_nonowner_native_prefix_at_only_requester(self):
        self.bot.beh['prefix_at'] = True
        await self.say('宝贝，帮我骂他', user=BOB, at=[BOT_ID, THIRD])
        ats = [seg['data']['qq'] for msg in self.client.sent for seg in msg['message'] if seg['type'] == 'at']
        self.assertEqual(ats, [str(BOB)])

    async def test_real_whitelist_revocation_does_not_leave_owner_bypass(self):
        self.bot.cfg['owner_chat']['user_ids'] = []
        await self.say('宝贝，骂他', at=[BOT_ID, THIRD])
        self.assertFalse(self.brain.owner)
        self.assertEqual(len(self.brain.pushback), 1)

    async def test_user_blacklist_still_prevents_response(self):
        self.bot._user_blacklist.add(ALICE)
        await self.say('宝贝，骂他', at=[BOT_ID, THIRD])
        self.assertFalse(self.brain.owner)
        self.assertFalse(self.client.sent)

    async def test_group_filter_still_prevents_response(self):
        self.bot._whitelist = {GROUP + 1}
        await self.say('宝贝，骂他', at=[BOT_ID, THIRD])
        self.assertFalse(self.brain.owner)
        self.assertFalse(self.client.sent)

    async def test_group_quota_still_prevents_response(self):
        self.bot.beh['per_group_per_minute'] = 0
        await self.say('宝贝，骂他', at=[BOT_ID, THIRD])
        self.assertFalse(self.brain.owner)
        self.assertFalse(self.client.sent)

    async def test_known_bot_no_at_owner_path_kept(self):
        await self.say('肥鱼，帮我骂小王')
        self.assertEqual(len(self.brain.owner), 1)
        self.assertFalse(self.brain.pushback)


if __name__ == '__main__':
    classes = (DirectVocativeParserTests, DirectVocativeIntegrationTests)
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in classes)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        success=result.wasSuccessful(), fixtures='synthetic identities; fake owner/pushback/ordinary Brain and OneBot',
        real_model_requests=0, real_QQ_messages=0, network='none')
    if RESULT:
        RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
