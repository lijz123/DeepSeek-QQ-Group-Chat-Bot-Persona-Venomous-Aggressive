"""Offline discussion-vs-command and configuration-grounded capability checks.

python notification_intent_tests.py [project_dir] [result.json]
Synthetic IDs, temporary SQLite, fake OneBot; no keys, model or network requests.
"""
from __future__ import annotations

from layout_support import app_directory, project_directory, configure_test_paths
configure_test_paths()
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

if len(sys.argv) > 1:
    sys.path.insert(0, str(app_directory(sys.argv[1])))
RESULT = Path(sys.argv[2]) if len(sys.argv) > 2 else None
from bot import Bot, Brain, DEFAULTS, parse_message_details
from capabilities import build_runtime_capabilities
from dialogue import DialogueState
from persona import build_system_prompt
from reminders import parse_command
from semantics import SemanticFrame, build_expression_system

BOT, ALICE, BOB, GROUP = 10001, 20002, 20003, 88888
NOW = datetime(2026, 10, 25, 6, tzinfo=timezone(timedelta(hours=8))).timestamp()
SCREENSHOT = '还是说，每次找你都得艾特一下'

def parse(text, mentions=()):
    return parse_command(text, tuple(map(str, mentions)), ALICE, BOT, NOW)

DISCUSSIONS = [
    ('screenshot', SCREENSHOT),
    ('every_time_question', '每次找你都得艾特一下吗'),
    ('every_time_at_you', '是不是每次找你都要@你'),
    ('without_at', '不艾特你也能聊天吗'),
    ('need_not_repeat', '是不是不用每次都艾特你'),
    ('meaning', '艾特是什么意思'),
    ('notification_feature', '通知功能好用吗'),
    ('reminder_rules', '提醒的规则是什么'),
    ('past_action', '我刚才已经艾特你了'),
    ('reported_action', '他说让我艾特你'),
    ('why_reply_only_at', '为什么总得艾特你才回复'),
    ('can_remind', '你能提醒我吗'),
    ('can_notify_others', '可以通知别人吗'),
    ('will_you_remind', '会不会提醒我'),
    ('mention_difference', '艾特和通知有什么区别'),
    ('body_requirement', '通知是不是必须带正文'),
    ('quoted_self_meaning', '提醒我是什么意思'),
    ('at_to_get_reply', '艾特你才能回复吗'),
]

COMMANDS = [
    ('immediate_self', '立即艾特我，测试完成', (), 'immediate', (ALICE,)),
    ('immediate_actual_other', '立即艾特一下，开会啦', (BOT, BOB), 'immediate', (BOB,)),
    ('scheduled_explicit', '明天早上8点艾特我，起床', (), 'schedule', (ALICE,)),
    ('scheduled_natural', '明早9点叫我起床', (), 'schedule', (ALICE,)),
    ('polite_timed', '能不能明早9点叫我起床', (), 'schedule', (ALICE,)),
    ('question_inside_body', '立即艾特我，说：你能准时到吗', (), 'immediate', (ALICE,)),
    ('rules_inside_body', '立即艾特我，说：告诉我规则', (), 'immediate', (ALICE,)),
    ('feature_inside_body_without_separator', '立即艾特我测试通知功能', (), 'immediate', (ALICE,)),
]

class ParserTests(unittest.TestCase):
    def test_true_missing_recipient_still_asks_for_target(self):
        result = parse('立即艾特一下，开会啦')
        self.assertEqual(result.action, 'error')
        self.assertFalse(result.targets)

    def test_nickname_is_not_a_recipient_id(self):
        result = parse('立即艾特小明，开会啦')
        self.assertEqual(result.action, 'error')
        self.assertFalse(result.targets)

    def test_actual_metadata_does_not_turn_discussion_into_command(self):
        self.assertIsNone(parse(SCREENSHOT, (BOT, BOB)))

    def test_actual_mention_alone_is_not_a_notification(self):
        self.assertIsNone(parse('你好呀', (BOT, BOB)))

    def test_missing_time_real_request_remains_clarification(self):
        result = parse('提醒我喝水')
        self.assertEqual(result.action, 'error')
        self.assertIn('时间', result.error)

def discussion_test(text):
    def test(self):
        self.assertIsNone(parse(text), text)
    return test

def command_test(text, mentions, action, targets):
    def test(self):
        result = parse(text, mentions)
        self.assertIsNotNone(result, text)
        self.assertEqual(result.action, action, (text, result))
        self.assertEqual(result.targets, targets)
        self.assertTrue(result.body.strip())
    return test

for label, text in DISCUSSIONS:
    setattr(ParserTests, 'test_discussion_' + label, discussion_test(text))
for label, text, mentions, action, targets in COMMANDS:
    setattr(ParserTests, 'test_command_' + label, command_test(text, mentions, action, targets))

class FakeClient:
    def __init__(self):
        self.self_id = BOT
        self.nickname = '合成机器人'
        self._ready = True
        self.ws = SimpleNamespace(closed=False)
        self.actions = []

    async def close(self):
        pass

    async def send_action(self, action, params=None, **kwargs):
        self.actions.append({'action': action, 'params': copy.deepcopy(params or {})})
        if action != 'send_group_msg':
            raise AssertionError('Unexpected fake API request: ' + action)
        return {'message_id': 70000 + len(self.actions)}

class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = copy.deepcopy(DEFAULTS)
        cfg['llm']['api_key'] = ''
        cfg['runtime'].update(dry_run=False, log_file='')
        cfg['groups'] = {'whitelist': [], 'blacklist': []}
        cfg['users'] = {'blacklist': []}
        cfg['behavior'].update(min_gap=0, per_group_per_minute=100, global_per_minute=100)
        cfg['voice']['enabled'] = False
        cfg['notifications'].update(database=str(Path(self.tmp.name) / 'jobs.sqlite3'))
        self.bot = Bot(cfg)
        self.bot.client = FakeClient()
        self.queued = []
        self.bot._enqueue_turn = self.queued.append
        self.bot.brain.reply = AsyncMock(side_effect=AssertionError('No model requests'))
        self.bot.brain.plan_reply = AsyncMock(side_effect=AssertionError('No model requests'))

    async def asyncTearDown(self):
        await self.bot.close()
        self.tmp.cleanup()

    async def event(self, text, mentions=(BOT,)):
        await self.bot.on_event({'post_type': 'message', 'message_type': 'group',
            'group_id': GROUP, 'user_id': ALICE, 'message_id': 'synthetic-' + str(time.monotonic_ns()),
            'sender': {'nickname': '合成群友'}, 'message': [
                *[{'type': 'at', 'data': {'qq': str(value)}} for value in mentions],
                {'type': 'text', 'data': {'text': text}}]})

    def assert_chat_only(self):
        self.assertEqual(len(self.queued), 1)
        self.assertFalse(self.bot.client.actions)
        self.assertEqual(self.bot._notification_store().list_owned(GROUP, ALICE, bot_id=BOT), [])
        self.bot.brain.reply.assert_not_awaited()
        self.bot.brain.plan_reply.assert_not_awaited()

    async def test_screenshot_addressed_routes_to_chat_not_fixed_error(self):
        await self.event(SCREENSHOT)
        self.assert_chat_only()
        self.assertEqual(self.queued[0].message.text, SCREENSHOT)

    async def test_screenshot_existing_dialogue_without_at_preserves_target(self):
        old = DialogueState(ALICE, '触发机器人', '文字称呼也可以', time.monotonic(), 1)
        self.bot._dialogues[GROUP] = old
        await self.event(SCREENSHOT, ())
        self.assert_chat_only()
        self.assertIs(self.bot._dialogues[GROUP], old)

    async def test_actual_other_mention_does_not_publish_notification(self):
        await self.event('是不是每次找你都得艾特一下', (BOT, BOB))
        self.assert_chat_only()

    async def test_body_requirement_discussion_is_normal_chat(self):
        await self.event('通知是不是必须带正文')
        self.assert_chat_only()

    async def test_real_immediate_request_still_sends_actual_at_body(self):
        await self.event('立即艾特我，测试完成')
        self.assertFalse(self.queued)
        self.assertEqual([a['action'] for a in self.bot.client.actions], ['send_group_msg'])
        message = self.bot.client.actions[0]['params']['message']
        self.assertEqual([s['data']['qq'] for s in message if s['type'] == 'at'], [str(ALICE)])
        self.assertTrue(''.join(s['data']['text'] for s in message if s['type'] == 'text').strip())

    async def test_brain_keeps_actual_runtime_flags_reference(self):
        self.assertIs(self.bot.brain.runtime, self.bot.cfg)
        self.assertFalse(self.bot.brain.runtime['voice']['enabled'])
        self.bot.cfg['voice']['enabled'] = True
        self.assertTrue(self.bot.brain.runtime['voice']['enabled'])

class CapabilityTests(unittest.TestCase):
    def cfg(self):
        value = copy.deepcopy(DEFAULTS)
        value['voice']['enabled'] = False
        return value

    def test_voice_disabled_does_not_advertise_positive_voice_commands(self):
        text = build_runtime_capabilities(self.cfg())
        self.assertIn('未启用', text)
        self.assertIn('音色查询', text)
        self.assertNotIn('用户明确要求时可由程序发送QQ AI语音', text)
        self.assertNotIn('可以建议', text)

    def test_voice_enabled_is_sending_not_voice_recognition(self):
        cfg = self.cfg()
        cfg['voice']['enabled'] = True
        text = build_runtime_capabilities(cfg)
        self.assertIn('语音发送与音色查询/选择已启用', text)
        self.assertIn('没有接入语音识别', text)
        self.assertIn('不能直接承诺支持声控', text)

    def test_missing_voice_runtime_is_conservative(self):
        text = build_runtime_capabilities()
        self.assertIn('功能代码保留但不可用', text)
        self.assertNotIn('语音发送与音色查询/选择已启用', text)

    def test_capability_builder_never_exports_credentials(self):
        cfg = self.cfg()
        marker = 'SYNTHETIC_NEVER_EXPORT_PRIVATE_VALUE'
        cfg['llm']['api_key'] = marker
        cfg['onebot']['access_token'] = marker
        cfg['other_unknown_private_section'] = {'secret': marker}
        self.assertNotIn(marker, build_runtime_capabilities(cfg))

    def test_disabled_notifications_not_promised_as_active(self):
        cfg = self.cfg()
        cfg['notifications']['enabled'] = False
        text = build_runtime_capabilities(cfg)
        self.assertIn('通知与定时提醒已关闭', text)
        self.assertIn('不得说已安排', text)
        self.assertNotIn('当前通知与定时提醒已启用', text)

    def test_trigger_rules_do_not_require_repeated_at(self):
        text = build_runtime_capabilities(self.cfg())
        self.assertIn('可以不重复@直接续聊', text)
        self.assertIn('这是文字识别，不是语音声控', text)
        self.assertIn('不能保证任何一条消息都必回', text)

    def test_disabled_keyword_and_followup_are_not_advertised(self):
        cfg = self.cfg()
        cfg['behavior'].update(keyword_reply=False, keywords=[], followup_rate=0)
        text = build_runtime_capabilities(cfg)
        self.assertIn('没有启用有效的称呼关键词触发', text)
        self.assertIn('普通自动续聊窗口或概率已关闭', text)
        self.assertNotIn('可以不重复@直接续聊', text)

    def test_both_writers_include_identical_runtime_capability_block(self):
        cfg = self.cfg()
        shared = build_runtime_capabilities(cfg)
        frame = SemanticFrame('ask', 'bot', '询问聊天如何触发', '', '解释当前真实入口', 'normal', False)
        self.assertIn(shared, build_system_prompt('合成机器人', runtime=cfg))
        self.assertIn(shared, build_expression_system('合成机器人', 'sharp', '', '', frame, runtime=cfg))

    def test_incoming_record_is_media_not_transcribed_speech(self):
        parsed = parse_message_details([{'type': 'record', 'data': {
            'file': 'synthetic-audio', 'text': 'this field is not an ASR transcript'}}], BOT)
        self.assertTrue(parsed.media)
        self.assertEqual(parsed.text, '')

class WriterWiringTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runtime = copy.deepcopy(DEFAULTS)
        self.runtime['voice']['enabled'] = False
        self.runtime['llm'].update(api_key='', semantic_review_policy='all')
        self.brain = Brain(self.runtime['llm'], self.runtime['safety'], self.runtime['persona'],
                           120, runtime=self.runtime)
        self.brain._provider_ready = lambda: True
        self.brain._session = SimpleNamespace(close=AsyncMock())
        self.shared = build_runtime_capabilities(self.runtime)
        self.transcript = json.dumps({'role': 'group_member', 'sender': {'id': str(ALICE),
            'display_name': '合成群友'}, 'content': SCREENSHOT, 'message_id': 'synthetic',
            'event_kind': 'message'}, ensure_ascii=False)

    async def asyncTearDown(self):
        await self.brain.close()

    def response_data(self):
        return json.dumps(dict(act='ask', target='bot', meaning='询问聊天是否每次都需艾特',
            unresolved='', reply_goal='解释当前实际触发方式', mode='normal',
            needs_clarification=False), ensure_ascii=False)

    async def test_legacy_writer_receives_real_disabled_voice_runtime(self):
        self.brain._call = AsyncMock(return_value='不用每次都艾特，正常接着聊就行。')
        await self.brain._legacy_respond(GROUP, self.transcript, '合成群友', True)
        self.assertIn(self.shared, self.brain._call.await_args_list[0].args[0][0]['content'])

    async def test_initial_semantic_writer_receives_actual_runtime(self):
        answer = json.dumps({'action': 'reply', 'text': '不用每次都艾特，正常接着聊就行', 'next': 'wait'}, ensure_ascii=False)
        self.brain._call = AsyncMock(side_effect=[self.response_data(), answer, '{"ok":true,"issue":""}'])
        result = await self.brain.plan_reply(GROUP, self.transcript, '合成群友', True)
        self.assertEqual(result.action, 'reply')
        expressions = [call.args[0][0]['content'] for call in self.brain._call.await_args_list
                       if call.kwargs.get('plan_mode')]
        self.assertEqual(len(expressions), 1)
        self.assertIn(self.shared, expressions[0])

    async def test_semantic_rewrite_does_not_lose_disabled_runtime(self):
        answer = json.dumps({'action': 'reply', 'text': '正常接着聊就行，不用每次都艾特', 'next': 'wait'}, ensure_ascii=False)
        self.brain._call = AsyncMock(side_effect=[self.response_data(), answer, answer,
                                                   '{"ok":true,"issue":""}'])
        with patch('bot.unconfirmed_relation_issue', side_effect=['synthetic guard rejection', '']):
            await self.brain.plan_reply(GROUP, self.transcript, '合成群友', True)
        expressions = [call.args[0][0]['content'] for call in self.brain._call.await_args_list
                       if call.kwargs.get('plan_mode')]
        self.assertEqual(len(expressions), 2)
        self.assertTrue(all(self.shared in prompt for prompt in expressions))

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  success=result.wasSuccessful(), fixtures='synthetic IDs, temporary SQLite, fake OneBot',
                  real_QQ_messages=0, real_model_requests=0, network=False)
    if RESULT:
        RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
