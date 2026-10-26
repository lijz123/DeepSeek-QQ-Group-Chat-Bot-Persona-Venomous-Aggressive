"""Offline persona reminder acknowledgements; no credentials or real sends.

python reminder_ack_tests.py [project_dir] [result.json]
"""
from __future__ import annotations

from layout_support import app_directory, project_directory, configure_test_paths
configure_test_paths()

import copy
import datetime as dt
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

if len(sys.argv) > 1:
    sys.path.insert(0, str(app_directory(sys.argv[1])))
RESULT = Path(sys.argv[2]) if len(sys.argv) > 2 else None

from bot import Bot, DEFAULTS, parse_message_details
from notifications import human_reminder_time
from reminders import LOCAL_TZ

BOT, ALICE, BOB, CAROL, GROUP = 10001, 20002, 20003, 20004, 88888


def stamp(year=2026, month=10, day=25, hour=21, minute=0, second=0):
    return dt.datetime(year, month, day, hour, minute, second, tzinfo=LOCAL_TZ).timestamp()


NOW = stamp()
DUE = stamp(day=26, hour=8)


class TimeWordingTests(unittest.TestCase):
    def test_tomorrow_morning(self):
        self.assertEqual(human_reminder_time(DUE, NOW), '明天早上8点')

    def test_today_evening(self):
        self.assertEqual(human_reminder_time(stamp(hour=22), NOW), '今天晚上10点')

    def test_day_after_tomorrow(self):
        self.assertEqual(human_reminder_time(stamp(day=27, hour=8), NOW), '后天早上8点')

    def test_farther_date_not_tomorrow(self):
        self.assertEqual(human_reminder_time(stamp(day=29, hour=8), NOW), '10月29日早上8点')

    def test_cross_year_keeps_year(self):
        value = human_reminder_time(stamp(year=2027, month=1, day=3, hour=8), NOW)
        self.assertIn('2027年1月3日', value)

    def test_half_past(self):
        self.assertEqual(human_reminder_time(stamp(day=26, hour=8, minute=30), NOW), '明天早上8点半')

    def test_minutes_not_dropped(self):
        self.assertEqual(human_reminder_time(stamp(day=26, hour=8, minute=17), NOW), '明天早上8点17分')

    def test_seconds_not_dropped(self):
        value = human_reminder_time(stamp(day=26, hour=8, minute=17, second=9), NOW)
        self.assertEqual(value, '明天早上8点17分9秒')

    def test_midnight_not_noon(self):
        self.assertEqual(human_reminder_time(stamp(day=26, hour=0), NOW), '明天凌晨0点')

    def test_noon_not_midnight(self):
        self.assertEqual(human_reminder_time(stamp(day=26, hour=12), NOW), '明天中午12点')

    def test_afternoon(self):
        self.assertEqual(human_reminder_time(stamp(day=26, hour=15), NOW), '明天下午3点')


class ConfirmationWordingTests(unittest.TestCase):
    def setUp(self):
        self.bot = Bot(copy.deepcopy(DEFAULTS))

    def job(self, **values):
        args = dict(id='abc123abc123', due=DUE, targets=(ALICE,), body='起床', status='pending')
        args.update(values)
        return SimpleNamespace(**args)

    def reply(self, **values):
        with patch('notifications.time.time', return_value=NOW):
            return self.bot._reminder_confirmation(self.job(**values), ALICE)

    def test_wakeup_fish_persona_and_exact_saved_time(self):
        text = self.reply()
        self.assertIn('明天早上8点', text)
        self.assertIn('本肥鱼', text)
        self.assertIn('起床', text)
        self.assertIn('小懒虫', text)

    def test_no_technical_receipt_in_default_reply(self):
        text = self.reply()
        for forbidden in ('abc123abc123', '已保存提醒', '北京时间', 'NapCat', '取消提醒', '2026-10-26', '需在线'):
            self.assertNotIn(forbidden, text)

    def test_generic_body_preserved_not_wakeup_template(self):
        text = self.reply(body='给绿植浇水')
        self.assertIn('给绿植浇水', text)
        self.assertNotIn('赖床', text)
        self.assertNotIn('小懒虫', text)

    def test_other_member_not_called_you(self):
        text = self.reply(targets=(BOB,), body='参加活动')
        self.assertIn('这位群友', text)
        self.assertIn('参加活动', text)
        self.assertNotIn('提醒你', text)

    def test_multiple_members_preserved(self):
        text = self.reply(targets=(BOB, CAROL), body='参加活动')
        self.assertIn('这几位群友', text)

    def test_sender_and_members_preserved(self):
        text = self.reply(targets=(ALICE, BOB), body='参加活动')
        self.assertIn('你和这几位群友', text)

    def test_custom_name_not_forced_to_fish(self):
        self.bot.name = '小助手'
        text = self.reply(body='喝水')
        self.assertIn('小助手', text)
        self.assertNotIn('本肥鱼', text)

    def test_sent_replay_not_future_promise(self):
        text = self.reply(status='sent')
        self.assertIn('已经', text)
        self.assertNotIn('明天早上8点', text)

    def test_cancelled_replay_not_future_promise(self):
        text = self.reply(status='cancelled')
        self.assertIn('取消', text)
        self.assertNotIn('明天早上8点', text)

    def test_uncertain_replay_not_success_promise(self):
        text = self.reply(status='uncertain')
        self.assertIn('没拿到准信', text)
        self.assertNotIn('叫你起床', text)

    def test_expired_replay_not_future_promise(self):
        text = self.reply(status='expired')
        self.assertIn('错过', text)
        self.assertNotIn('明天早上8点', text)


class FakeClient:
    def __init__(self):
        self.self_id = BOT
        self.nickname = '测试肥鱼'
        self._ready = True
        self.ws = SimpleNamespace(closed=False)
        self.sent = []

    async def send_action(self, action, params=None, **kwargs):
        assert action == 'send_group_msg'
        self.sent.append(params)
        return {'message_id': str(90000 + len(self.sent))}


class SavedBeforeSpeechTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        cfg = copy.deepcopy(DEFAULTS)
        cfg['llm']['api_key'] = ''
        cfg['runtime'].update(dry_run=False, log_file='')
        cfg['groups'] = {'whitelist': [], 'blacklist': []}
        cfg['users'] = {'blacklist': []}
        cfg['behavior'].update(min_gap=0, per_group_per_minute=100, global_per_minute=100)
        cfg['notifications'] = {'enabled': True, 'database': str(Path(self.temp.name) / 'jobs.sqlite3')}
        self.bot = Bot(cfg)
        self.bot.client = FakeClient()
        self.bot.brain.plan_reply = AsyncMock(side_effect=AssertionError('No model call for ack'))
        self.bot.brain.reply = AsyncMock(side_effect=AssertionError('No model call for ack'))

    async def asyncTearDown(self):
        await self.bot.close()
        self.temp.cleanup()

    async def command(self, text='明天早上8点艾特我，提醒我起床', targets=(BOT,), message_id='synthetic-ack'):
        parts = [{'type': 'at', 'data': {'qq': str(n)}} for n in targets]
        parts.append({'type': 'text', 'data': {'text': text}})
        parsed = parse_message_details(parts, BOT)
        with patch('notifications.time.time', return_value=NOW):
            return await self.bot._handle_notification_command(GROUP, ALICE, parsed, message_id)

    def rows(self):
        return self.bot._notification_store().list_owned(GROUP, ALICE, bot_id=BOT)

    def sent_text(self):
        return ''.join(s['data']['text'] for p in self.bot.client.sent for s in p['message'] if s['type'] == 'text')

    async def test_persisted_before_persona_confirmation(self):
        observed = []
        async def capture(group, user, text):
            observed.append((len(self.rows()), self.rows()[0]['due'], text))
            return 'sent'
        self.bot._notification_feedback = capture
        self.assertTrue(await self.command())
        self.assertEqual(observed[0][:2], (1, DUE))
        self.assertIn('明天早上8点', observed[0][2])

    async def test_persona_ack_is_plain_text_not_early_ping(self):
        await self.command()
        message = self.bot.client.sent[-1]['message']
        self.assertTrue(all(s['type'] == 'text' for s in message))
        self.assertIn('本肥鱼', self.sent_text())
        self.assertEqual(self.rows()[0]['targets'], (ALICE,))

    async def test_no_extra_model_calls(self):
        await self.command()
        self.bot.brain.plan_reply.assert_not_awaited()
        self.bot.brain.reply.assert_not_awaited()

    async def test_failed_save_does_not_emit_success(self):
        store = self.bot._notification_store()
        with patch.object(store, 'add', side_effect=ValueError('离线测试：保存失败')):
            await self.command()
        self.assertEqual(self.rows(), [])
        self.assertIn('未保存', self.sent_text())
        self.assertNotIn('叫你起床', self.sent_text())

    async def test_invalid_time_does_not_emit_success(self):
        await self.command('明天8点艾特我，提醒我起床')
        self.assertEqual(self.rows(), [])
        self.assertNotIn('叫你起床', self.sent_text())

    async def test_actual_other_member_saved_not_requester(self):
        await self.command('明天早上8点艾特 说：开会', (BOT, BOB))
        self.assertEqual(self.rows()[0]['targets'], (BOB,))
        self.assertIn('这位群友', self.sent_text())

    async def test_jobs_still_listable_with_id(self):
        await self.command()
        job_id = self.rows()[0]['id']
        self.assertNotIn(job_id, self.sent_text())
        await self.command('查看提醒', message_id='synthetic-list')
        self.assertIn(job_id, self.sent_text())

    async def test_dry_run_never_saves_or_confirms(self):
        self.bot.cfg['runtime']['dry_run'] = True
        await self.command()
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.bot.client.sent, [])

    async def test_blacklisted_target_never_saves_or_confirms(self):
        self.bot._user_blacklist.add(BOB)
        await self.command('明天早上8点艾特 说：开会', (BOT, BOB))
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.bot.client.sent, [])

    async def test_saved_body_not_rewritten_to_persona_text(self):
        await self.command()
        self.assertEqual(self.rows()[0]['body'], '起床')
        self.assertEqual(self.rows()[0]['status'], 'pending')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'success': result.wasSuccessful(), 'network': 'none', 'real_QQ_messages': 0,
              'fixtures': 'synthetic IDs, fixed China time, temporary SQLite, fake OneBot'}
    if RESULT:
        RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
