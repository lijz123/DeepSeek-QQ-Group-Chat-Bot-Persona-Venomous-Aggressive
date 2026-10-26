"""Offline natural-reminder checks: no real QQ, credentials, network or models.

Usage: python natural_reminder_tests.py [project_dir] [result.json]
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

from bot import Bot, DEFAULTS
from dialogue import DialogueState
from reminders import parse_command

BOT, ALICE, BOB, GROUP = 10001, 20002, 20003, 88888
TZ = timezone(timedelta(hours=8))
NOW = datetime(2026, 10, 25, 6, 0, tzinfo=TZ).timestamp()
SCREENSHOT = '明天早上9点叫我起床，我要去上班'


def stamp(day, hour, minute=0):
    return datetime(2026, 10, day, hour, minute, tzinfo=TZ).timestamp()


def parsed(text, mentions=()):
    return parse_command(text, tuple(map(str, mentions)), ALICE, BOT, NOW)


class NaturalParserTests(unittest.TestCase):
    def check_schedule(self, text, due, body=None, mentions=(), targets=(ALICE,)):
        command = parsed(text, mentions)
        self.assertIsNotNone(command, text)
        self.assertEqual(command.action, 'schedule', (text, command))
        self.assertEqual(command.due, due, text)
        self.assertEqual(command.targets, targets, text)
        self.assertTrue(command.body.strip(), text)
        if body is not None:
            self.assertEqual(command.body, body, text)

    def check_no_job(self, text, mentions=()):
        command = parsed(text, mentions)
        self.assertTrue(command is None or command.action == 'error', (text, command))


SCHEDULE_CASES = [
    ('screenshot', SCREENSHOT, stamp(26, 9), '起床', (), (ALICE,)),
    ('wake_before_time', '叫醒我明天早上9点', stamp(26, 9), '起床', (), (ALICE,)),
    ('time_before_wake', '明天早上9点叫醒我', stamp(26, 9), '起床', (), (ALICE,)),
    ('time_after_call', '叫我明天早上9点起床', stamp(26, 9), '起床', (), (ALICE,)),
    ('time_after_remind', '提醒我明早9点去上班', stamp(26, 9), '去上班', (), (ALICE,)),
    ('call_once', '明早9点喊我一声', stamp(26, 9), None, (), (ALICE,)),
    ('tomorrow_evening', '明晚8点催我交作业', stamp(26, 20), '交作业', (), (ALICE,)),
    ('tonight', '今晚8点喊我喝水', stamp(25, 20), '喝水', (), (ALICE,)),
    ('half_hour', '半小时后提醒我休息', NOW + 1800, '休息', (), (ALICE,)),
    ('quarter_hour', '一刻钟后提醒我喝水', NOW + 900, '喝水', (), (ALICE,)),
    ('set_alarm', '帮我设个明早9点的闹钟', stamp(26, 9), None, (), (ALICE,)),
    ('polite_alarm', '请帮我定一个明早9点的闹钟', stamp(26, 9), None, (), (ALICE,)),
    ('polite_request_prefix', '能不能明早9点叫我起床', stamp(26, 9), '起床', (), (ALICE,)),
    ('polite_request_suffix', '明早9点能叫我起床吗', stamp(26, 9), '起床', (), (ALICE,)),
    ('polite_can_remind', '可以明早9点提醒我开会吗', stamp(26, 9), '开会', (), (ALICE,)),
    ('body_time_not_deadline', '明早8点提醒我晚上10点吃药', stamp(26, 8), '晚上10点吃药', (), (ALICE,)),
    ('wake_reason_not_sender_speech', '明早9点叫我起床，因为我要上班', stamp(26, 9), '起床', (), (ALICE,)),
    ('relative_urge', '10分钟后催我交作业', NOW + 600, '交作业', (), (ALICE,)),
    ('actual_third_party', '明早9点喊他起床', stamp(26, 9), '起床', (BOT, BOB), (BOB,)),
    ('self_metadata_dedup', '明早9点叫我起床', stamp(26, 9), '起床', (BOT, ALICE), (ALICE,)),
    ('do_not_forget_prefix', '别忘了明早9点叫我起床', stamp(26, 9), '起床', (), (ALICE,)),
    ('please_do_not_forget', '不要忘了明早9点叫我起床', stamp(26, 9), '起床', (), (ALICE,)),
    ('do_not_forget_after_time', '10分钟后别忘了叫我喝水', NOW + 600, '喝水', (), (ALICE,)),
    ('explicit_negative_body', '明天早上9点叫我，说：不要迟到', stamp(26, 9), '不要迟到', (), (ALICE,)),
    ('explicit_retraction_words_are_body', '明天早上9点叫我，说：算了不用了', stamp(26, 9), '算了不用了', (), (ALICE,)),
    ('remind_once_before_recipient', '明天早上9点提醒一下我喝水', stamp(26, 9), '喝水', (), (ALICE,)),
    ('wake_reason_full_stop', '明天早上9点叫我起床。我要去上班', stamp(26, 9), '起床', (), (ALICE,)),
    ('wake_reason_without_punctuation', '明天早上9点叫我起床我要去上班', stamp(26, 9), '起床', (), (ALICE,)),
]

NO_JOB_CASES = [
    ('statement', '明早9点我要去上班', ()),
    ('nickname_report', '他叫我小王', ()),
    ('nickname_command', '明天起叫我老板', ()),
    ('hypothetical', '如果明早9点叫我起床，我就去上班', ()),
    ('reported', '他说，明早9点叫我起床', ()),
    ('quoted', '“明早9点叫我起床”', ()),
    ('negation_prefix', '不要明早9点叫我起床', ()),
    ('negation_after_time', '明早9点别叫我起床', ()),
    ('no_need', '我不需要明早9点叫我起床', ()),
    ('capability_only', '你能叫醒我吗', ()),
    ('vague_deadline', '到时候叫我起床', ()),
    ('ambiguous_hour', '8点叫我起床', ()),
    ('past_deadline', '今天早上5点叫醒我', ()),
    ('nickname_not_identity', '明早9点叫小王起床', ()),
    ('pronoun_not_identity', '明早9点叫他起床', ()),
    ('all_not_allowed', '明早9点叫我起床', (BOT, 'all')),
    ('wrong_you_target', '明早9点叫你起床', (BOT,)),
    ('recurring_unsupported', '每天早上9点叫我起床', ()),
    ('possessive_not_sender', '明早9点叫我的同事起床', ()),
    ('trailing_request_retracted', '明天早上9点叫我起床，算了不用了', ()),
    ('trailing_do_not_create', '明天早上9点叫我起床，不要真的设置', ()),
]


def schedule_case(case):
    def run(self):
        _, text, due, body, mentions, targets = case
        self.check_schedule(text, due, body, mentions, targets)
    return run


def no_job_case(case):
    def run(self):
        _, text, mentions = case
        self.check_no_job(text, mentions)
    return run


for case in SCHEDULE_CASES:
    setattr(NaturalParserTests, 'test_schedule_' + case[0], schedule_case(case))
for case in NO_JOB_CASES:
    setattr(NaturalParserTests, 'test_no_job_' + case[0], no_job_case(case))


class FakeClient:
    self_id = BOT
    nickname = '合成机器人'
    _ready = True
    ws = SimpleNamespace(closed=False)

    def __init__(self):
        self.sent = []

    async def send_action(self, action, params=None, **kwargs):
        if action != 'send_group_msg':
            raise AssertionError('No external action allowed')
        self.sent.append(params)
        return {'message_id': str(90000 + len(self.sent))}


class NaturalIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = patch('reminders.time.time', return_value=NOW)
        self.clock.start()
        cfg = copy.deepcopy(DEFAULTS)
        cfg['llm']['api_key'] = ''
        cfg['runtime'].update(dry_run=False, log_file='')
        cfg['behavior'].update(min_gap=0, per_group_per_minute=100, global_per_minute=100)
        cfg['groups'] = {'whitelist': [], 'blacklist': []}
        cfg['users'] = {'blacklist': []}
        cfg['notifications'] = {'enabled': True, 'database': str(Path(self.tmp.name) / 'jobs.sqlite3'),
                                'poll_interval': 1, 'grace_seconds': 300}
        self.bot = Bot(cfg)
        self.bot.client = FakeClient()
        self.queued = []
        self.bot._enqueue_turn = lambda turn: self.queued.append(turn)
        self.bot.brain.plan_reply = AsyncMock(side_effect=AssertionError('No model requests'))
        self.bot.brain.reply = AsyncMock(side_effect=AssertionError('No model requests'))

    async def asyncTearDown(self):
        await self.bot.close()
        self.clock.stop()
        self.tmp.cleanup()

    async def event(self, text=SCREENSHOT, mentions=(BOT,), mid='synthetic-request'):
        segments = [{'type': 'at', 'data': {'qq': str(x)}} for x in mentions]
        segments.append({'type': 'text', 'data': {'text': text}})
        await self.bot.on_event({'post_type': 'message', 'message_type': 'group', 'group_id': GROUP,
                                'user_id': ALICE, 'message_id': mid, 'sender': {'nickname': '合成群友'},
                                'message': segments})

    def jobs(self):
        return self.bot._notification_store().list_owned(GROUP, ALICE, bot_id=BOT)

    def reply(self):
        return ''.join(s['data']['text'] for s in self.bot.client.sent[-1]['message'] if s['type'] == 'text')

    async def test_screenshot_creates_saved_job_without_model(self):
        await self.event()
        rows = self.jobs()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['due'], stamp(26, 9))
        self.assertEqual(rows[0]['targets'], (ALICE,))
        self.assertEqual(rows[0]['body'], '起床')
        self.assertEqual(self.queued, [])
        self.bot.brain.plan_reply.assert_not_awaited()
        self.bot.brain.reply.assert_not_awaited()

    async def test_original_user_text_not_rewritten(self):
        await self.event()
        user = next(m for m in self.bot.store._data[GROUP] if not m.is_bot)
        self.assertEqual(user.text, SCREENSHOT)

    async def test_persona_confirmation_not_teach_magic_keyword(self):
        await self.event()
        text = self.reply()
        self.assertIn('明天早上9点', text)
        self.assertIn('起床', text)
        self.assertNotIn('无法', text)
        self.assertNotIn('以程序回执为准', text)
        self.assertNotIn('艾特我', text)
        self.assertFalse(any(s['type'] == 'at' for s in self.bot.client.sent[-1]['message']))

    async def test_active_conversation_no_at_works(self):
        self.bot._dialogues[GROUP] = DialogueState(ALICE, '之前的话', '刚才回答', time.monotonic())
        await self.event(mentions=())
        self.assertEqual(len(self.jobs()), 1)
        self.assertFalse(self.queued)

    async def test_bot_name_without_at_is_parsed_without_rewriting_original(self):
        text = '大肥鱼，明早9点叫我起床'
        await self.event(text, mentions=())
        self.assertEqual(len(self.jobs()), 1)
        self.assertEqual(self.jobs()[0]['due'], stamp(26, 9))
        user = next(m for m in self.bot.store._data[GROUP] if not m.is_bot)
        self.assertEqual(user.text, text)
        self.assertFalse(self.queued)

    async def test_no_address_no_session_does_not_intercept(self):
        await self.event(mentions=())
        self.assertFalse(self.jobs())
        self.assertEqual(len(self.queued), 1)
        self.assertFalse(self.bot.client.sent)

    async def test_other_user_session_does_not_intercept(self):
        self.bot._dialogues[GROUP] = DialogueState(BOB, '别人的话', '给别人的回答', time.monotonic())
        await self.event(mentions=())
        self.assertFalse(self.jobs())
        self.assertEqual(len(self.queued), 1)

    async def test_stale_session_does_not_intercept(self):
        self.bot._dialogues[GROUP] = DialogueState(ALICE, '旧话', '旧回答', time.monotonic() - 999)
        await self.event(mentions=())
        self.assertFalse(self.jobs())
        self.assertEqual(len(self.queued), 1)

    async def test_actual_other_member_metadata_not_nickname(self):
        await self.event('明早9点喊他起床', (BOT, BOB))
        self.assertEqual(self.jobs()[0]['targets'], (BOB,))
        self.assertEqual(self.jobs()[0]['body'], '起床')

    async def test_ambiguous_time_does_not_make_false_saved_confirmation(self):
        await self.event('8点叫我起床')
        self.assertFalse(self.jobs())
        self.assertTrue(self.bot.client.sent)
        self.assertNotIn('小懒虫', self.reply())

    async def test_time_statement_remains_chat_not_job(self):
        await self.event('明早9点我要去上班')
        self.assertFalse(self.jobs())
        self.assertEqual(len(self.queued), 1)
        self.assertFalse(self.bot.client.sent)

    async def test_natural_saved_job_still_delivers_actual_at_and_body(self):
        await self.event()
        await self.bot._notification_tick(stamp(26, 9) + 1)
        message = self.bot.client.sent[-1]['message']
        self.assertEqual([s['data']['qq'] for s in message if s['type'] == 'at'], [str(ALICE)])
        self.assertEqual(''.join(s['data']['text'] for s in message if s['type'] == 'text').strip(), '起床')
        self.assertEqual(self.jobs()[0]['status'], 'sent')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'success': result.wasSuccessful(), 'network': 'none',
              'fixtures': 'synthetic identities; temporary SQLite; fake OneBot; fixed China time'}
    if RESULT:
        RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
