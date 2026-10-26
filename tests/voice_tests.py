"""Offline voice-message regression: fake OneBot, synthetic users, temporary DB.

python voice_tests.py [project_dir] [result.json]
Never starts QQ, a browser, a model, a TTS process, or audio playback. All native
voice actions are recorded by in-memory fakes; get_ai_record is forbidden because
that seemingly read-only action can itself send a voice message.
"""
from __future__ import annotations

from layout_support import app_directory, project_directory, configure_test_paths
configure_test_paths()

import asyncio
import copy
from dataclasses import replace
import datetime as dt
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

if len(sys.argv) > 1:
    sys.path.insert(0, str(app_directory(sys.argv[1])))
RESULT = Path(sys.argv[2]) if len(sys.argv) > 2 else None

from bot import Bot, DEFAULTS, Msg, OneBotClient
from dialogue import DialogueState, PendingTurn
from reminders import LOCAL_TZ, ReminderStore, parse_command
from voice import QQVoice, VoiceUnavailable, parse_voice_command, wants_voice_reply

BOT, ALICE, BOB, GROUP = 10001, 20002, 20003, 88888
NOW = dt.datetime(2026, 10, 25, 21, tzinfo=LOCAL_TZ).timestamp()
ROLES = [{"type": "合成音色", "characters": [
    {"character_id": "synthetic-fish", "character_name": "合成肥鱼",
     "preview_url": "https://example.invalid/private-preview"},
    {"character_id": "synthetic-bird", "character_name": "合成小鸟"},
]}]


def command(text):
    return parse_command(text, (), ALICE, BOT, NOW)


class VoiceGrammarTests(unittest.TestCase):
    def test_literal_speech_preserves_text(self):
        value = parse_voice_command('用语音说：你好，合成群友。')
        self.assertEqual((value.action, value.text), ('speak', '你好，合成群友。'))

    def test_polite_speech_command(self):
        value = parse_voice_command('请用语音念一下：现在休息一下')
        self.assertEqual((value.action, value.text), ('speak', '现在休息一下'))

    def test_send_one_voice_with_explicit_colon(self):
        self.assertEqual(parse_voice_command('发一条语音：你好').text, '你好')

    def test_list_voices_is_not_speech(self):
        self.assertEqual(parse_voice_command('查看语音音色').action, 'voices')

    def test_empty_and_nonstring_not_commands(self):
        for value in (None, 12, '', '用语音说：', '用语音说：   '):
            with self.subTest(value=value):
                self.assertIsNone(parse_voice_command(value))

    def test_voice_discussion_not_a_command(self):
        for text in ('你会语音吗', '语音功能修好了吗', '我想研究用语音说话', '我的QQ电话坏了',
                     '用语音说话是什么意思', '用语音回复不了吗'):
            with self.subTest(text=text):
                self.assertIsNone(parse_voice_command(text))
                self.assertFalse(wants_voice_reply(text))

    def test_negated_voice_not_opt_in(self):
        for text in ('不要用语音说：你好', '别用语音回复我', '不用语音回答我'):
            with self.subTest(text=text):
                self.assertIsNone(parse_voice_command(text))
                self.assertFalse(wants_voice_reply(text))

    def test_quoted_reported_and_hypothetical_not_opt_in(self):
        for text in ('“用语音说：你好”', '他说用语音回复我', '如果用语音回复我会怎样',
                     '用语音回复我不要', '用语音回复我会怎样', '用语音回复我，不用了',
                     '用语音回复我，这句话是别人说的', '用语音回复我，如果你会的话'):
            with self.subTest(text=text):
                self.assertIsNone(parse_voice_command(text))
                self.assertFalse(wants_voice_reply(text))

    def test_explicit_voice_reply_opt_in(self):
        self.assertTrue(wants_voice_reply('用语音回复我'))
        self.assertTrue(wants_voice_reply('请用语音回答我：今晚吃什么？'))
        self.assertIsNone(parse_voice_command('用语音回复我'))

    def test_future_voice_command_is_not_immediate_speech(self):
        self.assertIsNone(parse_voice_command('明早9点用语音叫我起床'))

    def test_explicit_voice_schedule_variants(self):
        for text in ('明早9点用语音叫我起床', '用语音提醒我明天早上9点开会',
                     '明早9点叫我起床，用语音'):
            with self.subTest(text=text):
                value = command(text)
                self.assertEqual(value.action, 'schedule')
                self.assertEqual(value.delivery, 'voice')
                self.assertNotIn('用语音', value.body)

    def test_voice_word_in_saved_body_remains_text(self):
        for text in ('10分钟后提醒我修语音功能', '10分钟后提醒我，说：用语音练习普通话'):
            with self.subTest(text=text):
                value = command(text)
                self.assertEqual(value.action, 'schedule')
                self.assertEqual(value.delivery, 'text')
                self.assertIn('语音', value.body)


class FakeClient:
    def __init__(self):
        self.self_id = BOT
        self.nickname = '合成机器人'
        self._ready = True
        self.ws = SimpleNamespace(closed=False)
        self.actions = []
        self.roles = copy.deepcopy(ROLES)
        self.voice_ack = {'message_id': 91001}
        self.voice_error = None
        self.text_ack = None
        self.text_error = None
        self.list_error = None
        self.after_roles = None

    async def send_action(self, action, params=None, **kwargs):
        self.actions.append({'action': action, **copy.deepcopy(params or {})})
        if action == 'get_ai_record':
            raise AssertionError('get_ai_record can send: fetch-and-resend is forbidden')
        if action == 'get_ai_characters':
            if self.list_error:
                raise self.list_error
            if self.after_roles:
                self.after_roles()
            return copy.deepcopy(self.roles)
        if action == 'send_group_ai_record':
            if self.voice_error:
                raise self.voice_error
            return copy.deepcopy(self.voice_ack)
        if action == 'send_group_msg':
            if self.text_error:
                raise self.text_error
            if self.text_ack is not None:
                return copy.deepcopy(self.text_ack)
            return {'message_id': 92000 + len(self.actions)}
        raise AssertionError('Unexpected native action: ' + action)

    def of(self, action):
        return [value for value in self.actions if value['action'] == action]


class VoiceProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = FakeClient()
        self.engine = QQVoice({'enabled': True, 'character_id': 'auto', 'max_chars': 200})

    async def test_native_character_shape_and_preview_discard(self):
        roles = await self.engine.characters(self.client, GROUP, BOT)
        self.assertEqual([v.character_id for v in roles], ['synthetic-fish', 'synthetic-bird'])
        self.assertEqual(roles[0].character_name, '合成肥鱼')
        self.assertNotIn('preview', repr(roles))
        self.assertEqual([a['action'] for a in self.client.actions], ['get_ai_characters'])

    async def test_auto_selects_actual_role(self):
        self.assertEqual(await self.engine.character(self.client, GROUP, BOT), 'synthetic-fish')

    async def test_configured_role_must_be_discovered(self):
        engine = QQVoice({'character_id': 'synthetic-bird'})
        self.assertEqual(await engine.character(self.client, GROUP, BOT), 'synthetic-bird')
        engine = QQVoice({'character_id': 'not-returned-by-group'})
        with self.assertRaises(VoiceUnavailable):
            await engine.character(self.client, GROUP, BOT)
        self.assertFalse(self.client.of('send_group_ai_record'))

    async def test_role_cache_is_group_scoped(self):
        await self.engine.character(self.client, GROUP, BOT)
        await self.engine.character(self.client, GROUP, BOT)
        await self.engine.character(self.client, GROUP + 1, BOT)
        self.assertEqual(len(self.client.of('get_ai_characters')), 2)

    async def test_role_cache_is_account_scoped(self):
        await self.engine.character(self.client, GROUP, BOT)
        await self.engine.character(self.client, GROUP, BOT + 1)
        self.assertEqual(len(self.client.of('get_ai_characters')), 2)

    async def test_disabled_never_discovers_or_sends(self):
        engine = QQVoice({'enabled': False})
        with self.assertRaises(VoiceUnavailable):
            await engine.character(self.client, GROUP, BOT)
        with self.assertRaises(VoiceUnavailable):
            await engine.send(self.client, GROUP, 'synthetic-fish', '你好')
        self.assertFalse(self.client.actions)

    async def test_malformed_character_response_no_send(self):
        self.client.roles = {'characters': ROLES}
        with self.assertRaises(VoiceUnavailable):
            await self.engine.character(self.client, GROUP, BOT)
        self.assertFalse(self.client.of('send_group_ai_record'))

    async def test_empty_character_response_no_send(self):
        self.client.roles = []
        with self.assertRaises(VoiceUnavailable):
            await self.engine.character(self.client, GROUP, BOT)
        self.assertFalse(self.client.of('send_group_ai_record'))

    async def test_positive_message_id_is_sent(self):
        result = await self.engine.send(self.client, GROUP, 'synthetic-fish', ' 你好 ')
        self.assertEqual((result.state, result.message_id), ('sent', 91001))
        self.assertEqual(self.client.of('send_group_ai_record')[0]['text'], '你好')
        self.assertEqual(self.client.of('send_group_ai_record')[0]['group_id'], str(GROUP))

    async def test_zero_id_is_submitted_not_fake_failure(self):
        for identity in (0, '0'):
            with self.subTest(identity=identity):
                self.client.voice_ack = {'message_id': identity}
                result = await self.engine.send(self.client, GROUP, 'synthetic-fish', '你好')
                self.assertEqual((result.state, result.message_id), ('submitted', None))

    async def test_missing_id_is_uncertain_without_resend(self):
        self.client.voice_ack = {}
        result = await self.engine.send(self.client, GROUP, 'synthetic-fish', '你好')
        self.assertEqual(result.state, 'uncertain')
        self.assertEqual(len(self.client.of('send_group_ai_record')), 1)

    async def test_invalid_ids_not_quotable_success(self):
        for identity in (-1, True, False, 'garbage', '９１００１', 1.5):
            with self.subTest(identity=identity):
                self.client.voice_ack = {'message_id': identity}
                result = await self.engine.send(self.client, GROUP, 'synthetic-fish', '你好')
                self.assertEqual((result.state, result.message_id), ('uncertain', None))

    async def test_timeout_is_uncertain_with_one_send_no_fetch(self):
        self.client.voice_error = TimeoutError('synthetic lost reply')
        result = await self.engine.send(self.client, GROUP, 'synthetic-fish', '你好')
        self.assertEqual(result.state, 'uncertain')
        self.assertEqual([a['action'] for a in self.client.actions], ['send_group_ai_record'])

    async def test_role_timeout_is_pre_send_unavailable(self):
        self.client.list_error = TimeoutError('synthetic role timeout')
        with self.assertRaises(VoiceUnavailable):
            await self.engine.character(self.client, GROUP, BOT)
        self.assertEqual([a['action'] for a in self.client.actions], ['get_ai_characters'])

    async def test_empty_and_oversized_speech_rejected_before_send(self):
        for text in (' ', '长' * 201):
            with self.subTest(size=len(text)):
                with self.assertRaises(VoiceUnavailable):
                    await self.engine.send(self.client, GROUP, 'synthetic-fish', text)
        self.assertFalse(self.client.actions)

    async def test_settings_cannot_raise_hard_speech_cap(self):
        engine = QQVoice({'max_chars': 1000})
        self.assertLessEqual(engine.max_chars, 200)
        with self.assertRaises(VoiceUnavailable):
            await engine.send(self.client, GROUP, 'synthetic-fish', '长' * 201)
        self.assertFalse(self.client.actions)

    async def test_invalid_group_rejected_before_send(self):
        for group in (0, -1, True, 'all', '１２３'):
            with self.subTest(group=group):
                with self.assertRaises(VoiceUnavailable):
                    await self.engine.send(self.client, group, 'synthetic-fish', '你好')
        self.assertFalse(self.client.actions)

    async def test_disconnected_before_send_is_unavailable(self):
        self.client.ws.closed = True
        with self.assertRaises(VoiceUnavailable):
            await self.engine.send(self.client, GROUP, 'synthetic-fish', '你好')
        self.assertFalse(self.client.actions)

    async def test_cancellation_propagates_without_retry(self):
        self.client.voice_error = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.engine.send(self.client, GROUP, 'synthetic-fish', '你好')
        self.assertEqual(len(self.client.of('send_group_ai_record')), 1)

    async def test_normal_onebot_ack_envelope_and_native_zero(self):
        client = OneBotClient(copy.deepcopy(DEFAULTS['onebot']), AsyncMock())
        sent = []

        async def send_json(request):
            sent.append(copy.deepcopy(request))
            client._pending[request['echo']].set_result({
                'status': 'ok', 'retcode': 0, 'data': {'message_id': 0}, 'echo': request['echo']})

        client.ws = SimpleNamespace(closed=False, send_json=send_json)
        result = await self.engine.send(client, GROUP, 'synthetic-fish', '你好')
        self.assertEqual(result.state, 'submitted')
        self.assertEqual([a['action'] for a in sent], ['send_group_ai_record'])
        self.assertEqual(client._pending, {})


class VoiceStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'voice-jobs.sqlite3'
        self.store = ReminderStore(self.path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def add(self, **kwargs):
        values = dict(group_id=GROUP, requester_id=ALICE, targets=(ALICE,), body='起床',
                      due=NOW + 60, now=NOW, bot_id=BOT, delivery='voice')
        values.update(kwargs)
        return self.store.add(**values)

    def test_voice_delivery_survives_restart(self):
        job = self.add()
        self.store.close()
        self.store = ReminderStore(self.path)
        row = self.store.list_owned(GROUP, ALICE, bot_id=BOT)[0]
        self.assertEqual((row['id'], row['delivery']), (job['id'], 'voice'))

    def test_existing_callers_default_to_text(self):
        row = self.store.add(GROUP, ALICE, (ALICE,), '喝水', NOW + 60, NOW, bot_id=BOT)
        self.assertEqual(row['delivery'], 'text')

    def test_old_schema_migrates_rows_to_text(self):
        self.store.close()
        connection = sqlite3.connect(self.path)
        try:
            connection.execute('''CREATE TABLE reminders (
                id TEXT PRIMARY KEY, bot_id INTEGER NOT NULL, group_id INTEGER NOT NULL,
                requester_id INTEGER NOT NULL, targets TEXT NOT NULL, body TEXT NOT NULL,
                due REAL NOT NULL, created REAL NOT NULL, status TEXT NOT NULL,
                error TEXT NOT NULL DEFAULT '', source_id TEXT NOT NULL DEFAULT '')''')
            connection.execute('INSERT INTO reminders VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                ('synthetic-old', BOT, GROUP, ALICE, json.dumps([ALICE]), '旧提醒',
                 NOW + 60, NOW, 'pending', '', 'old-source'))
            connection.commit()
        finally:
            connection.close()
        self.store = ReminderStore(self.path)
        row = self.store.list_owned(GROUP, ALICE, bot_id=BOT)[0]
        self.assertEqual((row['id'], row['body'], row['delivery']), ('synthetic-old', '旧提醒', 'text'))

    def test_duplicate_source_does_not_change_delivery(self):
        first = self.add(source_id='same-synthetic-request')
        second = self.add(source_id='same-synthetic-request', delivery='text')
        self.assertEqual((first['id'], second['delivery']), (second['id'], 'voice'))

    def test_owner_can_cancel_voice_job(self):
        job = self.add()
        self.assertEqual(self.store.cancel_owned(GROUP, ALICE, job['id'], bot_id=BOT), 1)
        self.assertFalse(self.store.due_candidates(NOW + 61, bot_id=BOT))

    def test_other_owner_cannot_cancel_voice_job(self):
        job = self.add()
        self.assertEqual(self.store.cancel_owned(GROUP, BOB, job['id'], bot_id=BOT), 0)
        self.assertEqual(self.store.list_owned(GROUP, BOB, bot_id=BOT), [])

    def test_other_account_cannot_read_claim_or_cancel_voice_job(self):
        job = self.add()
        self.assertEqual(self.store.list_owned(GROUP, ALICE, bot_id=BOT + 1), [])
        self.assertFalse(self.store.claim(job['id'], NOW + 61, bot_id=BOT + 1))
        self.assertEqual(self.store.cancel_owned(GROUP, ALICE, job['id'], bot_id=BOT + 1), 0)

    def test_submitted_is_terminal_not_retried(self):
        job = self.add()
        self.assertTrue(self.store.claim(job['id'], NOW + 61, bot_id=BOT))
        self.store.finish(job['id'], 'submitted', bot_id=BOT)
        self.assertFalse(self.store.due_candidates(NOW + 62, bot_id=BOT))
        self.assertEqual(self.store.list_owned(GROUP, ALICE, bot_id=BOT)[0]['status'], 'submitted')

    def test_interrupted_voice_send_becomes_uncertain_after_restart(self):
        job = self.add()
        self.store.claim(job['id'], NOW + 61, bot_id=BOT)
        self.store.close()
        self.store = ReminderStore(self.path)
        self.assertFalse(self.store.due_candidates(NOW + 62, bot_id=BOT))
        self.assertEqual(self.store.list_owned(GROUP, ALICE, bot_id=BOT)[0]['status'], 'uncertain')


class VoiceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = copy.deepcopy(DEFAULTS)
        cfg['llm']['api_key'] = ''
        cfg['runtime'].update(dry_run=False, log_file='')
        cfg['groups'] = {'whitelist': [], 'blacklist': []}
        cfg['users'] = {'blacklist': []}
        cfg['behavior'].update(min_gap=0, per_group_per_minute=100, global_per_minute=100)
        cfg['notifications'] = {'enabled': True, 'database': str(Path(self.tmp.name) / 'jobs.sqlite3'),
                                'grace_seconds': 300, 'poll_interval': 1}
        cfg['voice'] = {'enabled': True, 'character_id': 'auto', 'max_chars': 200}
        self.bot = Bot(cfg)
        self.bot.client = FakeClient()
        self.queued = []
        self.bot._enqueue_turn = lambda turn: self.queued.append(turn)
        self.bot.brain.plan_reply = AsyncMock(side_effect=AssertionError('No model request allowed'))
        self.bot.brain.reply = AsyncMock(side_effect=AssertionError('No model request allowed'))

    async def asyncTearDown(self):
        await self.bot.close()
        self.tmp.cleanup()

    async def event(self, text, mentions=(BOT,), mid='synthetic-voice-request'):
        segments = [{'type': 'at', 'data': {'qq': str(value)}} for value in mentions]
        segments.append({'type': 'text', 'data': {'text': text}})
        await self.bot.on_event({'post_type': 'message', 'message_type': 'group', 'group_id': GROUP,
            'user_id': ALICE, 'message_id': mid, 'sender': {'nickname': '合成群友'}, 'message': segments})

    def jobs(self, bot_id=BOT):
        return self.bot._notification_store().list_owned(GROUP, ALICE, bot_id=bot_id)

    def add_due(self, **kwargs):
        values = dict(group_id=GROUP, requester_id=ALICE, targets=(ALICE,), body='起床',
                      due=NOW + 1, now=NOW, bot_id=BOT, delivery='voice')
        values.update(kwargs)
        return self.bot._notification_store().add(**values)

    async def send(self, targets=(), body='合成语音', **kwargs):
        return await self.bot._send_voice_notification(GROUP, ALICE, targets, body, **kwargs)

    def active_turn(self, text='用语音回复我', revision=0):
        message = Msg('合成群友', text, False, NOW, sender_id=str(ALICE), message_id='request')
        turn = PendingTurn(GROUP, ALICE, 'request', message, True, (), '', '',
                           time.monotonic(), time.monotonic(), ('request',), target_revision=revision)
        self.bot._active_turn[GROUP] = turn
        return turn

    async def test_literal_command_sends_native_voice_without_model_or_fetch(self):
        await self.event('用语音说：你好')
        self.assertEqual([a['action'] for a in self.bot.client.actions],
                         ['get_ai_characters', 'send_group_ai_record'])
        self.assertEqual(self.bot.client.of('send_group_ai_record')[0]['text'], '你好')
        self.assertFalse(self.queued)
        self.bot.brain.plan_reply.assert_not_awaited()
        self.bot.brain.reply.assert_not_awaited()

    async def test_character_list_uses_metadata_not_preview_or_voice_send(self):
        await self.event('查看语音音色')
        self.assertEqual(len(self.bot.client.of('get_ai_characters')), 1)
        self.assertFalse(self.bot.client.of('send_group_ai_record'))
        messages = self.bot.client.of('send_group_msg')
        self.assertEqual(len(messages), 1)
        text = ''.join(s['data']['text'] for s in messages[0]['message'] if s['type'] == 'text')
        self.assertIn('合成肥鱼', text)
        self.assertNotIn('preview', text)
        self.assertFalse(self.queued)

    async def test_normal_voice_discussion_remains_chat(self):
        await self.event('语音功能修好了吗')
        self.assertEqual(len(self.queued), 1)
        self.assertFalse(self.bot.client.actions)
        self.assertFalse(self.jobs())

    async def test_scheduled_voice_is_persisted_before_confirmation(self):
        with patch('notifications.time.time', return_value=NOW):
            await self.event('明早9点用语音叫我起床')
        self.assertEqual(len(self.jobs()), 1)
        self.assertEqual((self.jobs()[0]['delivery'], self.jobs()[0]['body']), ('voice', '起床'))
        self.assertFalse(self.bot.client.of('send_group_ai_record'))
        self.assertFalse(self.queued)

    async def test_due_voice_has_real_at_and_exactly_one_native_send(self):
        self.add_due()
        await self.bot._notification_tick(NOW + 2)
        await self.bot._notification_tick(NOW + 3)
        self.assertEqual(len(self.bot.client.of('send_group_ai_record')), 1)
        self.assertEqual(self.bot.client.of('send_group_ai_record')[0]['text'], '起床')
        mentions = [s['data']['qq'] for a in self.bot.client.of('send_group_msg')
                    for s in a['message'] if s['type'] == 'at']
        self.assertEqual(mentions, [str(ALICE)])
        self.assertEqual(self.jobs()[0]['status'], 'sent')

    async def test_zero_ack_submitted_job_not_retried_or_indexed_as_zero(self):
        self.add_due()
        self.bot.client.voice_ack = {'message_id': 0}
        await self.bot._notification_tick(NOW + 2)
        await self.bot._notification_tick(NOW + 3)
        self.assertEqual(self.jobs()[0]['status'], 'submitted')
        self.assertEqual(len(self.bot.client.of('send_group_ai_record')), 1)
        self.assertNotIn((GROUP, '0'), self.bot._msg_index)

    async def test_missing_ack_job_uncertain_no_resend(self):
        self.add_due()
        self.bot.client.voice_ack = {}
        await self.bot._notification_tick(NOW + 2)
        await self.bot._notification_tick(NOW + 3)
        self.assertEqual(self.jobs()[0]['status'], 'uncertain')
        self.assertEqual(len(self.bot.client.of('send_group_ai_record')), 1)

    async def test_send_timeout_job_uncertain_no_resend(self):
        self.add_due()
        self.bot.client.voice_error = TimeoutError('synthetic uncertain voice')
        await self.bot._notification_tick(NOW + 2)
        self.bot.client.voice_error = None
        await self.bot._notification_tick(NOW + 3)
        self.assertEqual(self.jobs()[0]['status'], 'uncertain')
        self.assertEqual(len(self.bot.client.of('send_group_ai_record')), 1)

    async def test_whitelist_blocks_before_native_send(self):
        self.bot._whitelist = {GROUP + 1}
        self.assertEqual(await self.send(), 'blocked')
        self.assertFalse(self.bot.client.actions)

    async def test_requester_mute_blocks_before_native_send(self):
        self.bot._muted.add((GROUP, ALICE))
        self.assertEqual(await self.send(), 'blocked')
        self.assertFalse(self.bot.client.actions)

    async def test_target_mute_blocks_before_native_send(self):
        self.bot._muted.add((GROUP, BOB))
        self.assertEqual(await self.send((BOB,)), 'blocked')
        self.assertFalse(self.bot.client.actions)

    async def test_requester_blacklist_blocks_before_native_send(self):
        self.bot._user_blacklist.add(ALICE)
        self.assertEqual(await self.send(), 'blocked')
        self.assertFalse(self.bot.client.actions)

    async def test_dry_run_never_discovers_sends_or_schedules(self):
        self.bot.cfg['runtime']['dry_run'] = True
        self.assertEqual(await self.send(), 'dry_run')
        await self.event('明早9点用语音叫我起床')
        self.assertFalse(self.bot.client.actions)
        self.assertFalse(self.jobs())

    async def test_exhausted_group_quota_defers_before_native_send(self):
        self.bot.beh['per_group_per_minute'] = 0
        self.assertEqual(await self.send(), 'deferred')
        self.assertFalse(self.bot.client.actions)

    async def test_voice_shares_existing_text_notification_quota(self):
        self.bot.beh['per_group_per_minute'] = 1
        self.assertEqual(await self.bot._send_notification(GROUP, ALICE, (), '合成文字'), 'sent')
        self.assertEqual(await self.send(), 'deferred')
        self.assertFalse(self.bot.client.of('send_group_ai_record'))

    async def test_disabled_voice_never_discovers_or_sends(self):
        self.bot.cfg['voice']['enabled'] = False
        self.assertEqual(await self.send(), 'blocked')
        self.assertFalse(self.bot.client.actions)

    async def test_disconnected_defers_without_claim_or_send(self):
        self.add_due()
        self.bot.client._ready = False
        self.assertEqual(await self.send(), 'deferred')
        await self.bot._notification_tick(NOW + 2)
        self.assertEqual(self.jobs()[0]['status'], 'pending')
        self.assertFalse(self.bot.client.actions)

    async def test_expected_account_mismatch_defers(self):
        self.assertEqual(await self.send(expected_bot_id=BOT + 1), 'deferred')
        self.assertFalse(self.bot.client.actions)

    async def test_account_switch_during_role_lookup_never_sends_old_job(self):
        self.add_due()
        self.bot.client.after_roles = lambda: setattr(self.bot.client, 'self_id', BOT + 1)
        await self.bot._notification_tick(NOW + 2)
        self.assertFalse(self.bot.client.of('send_group_ai_record'))
        self.assertFalse(self.bot.client.of('send_group_msg'))
        self.assertEqual(self.jobs()[0]['status'], 'pending')

    async def test_new_account_cannot_deliver_old_account_job(self):
        self.add_due()
        self.bot.client.self_id = BOT + 1
        await self.bot._notification_tick(NOW + 2)
        self.assertFalse(self.bot.client.actions)
        self.assertEqual(self.jobs()[0]['status'], 'pending')

    async def test_failed_claim_callback_does_not_send(self):
        claim = []

        def fail_claim():
            claim.append(True)
            return False

        self.assertEqual(await self.send(before_send=fail_claim), 'blocked')
        self.assertEqual(claim, [True])
        self.assertFalse(self.bot.client.of('send_group_ai_record'))
        self.assertFalse(self.bot.client.of('send_group_msg'))

    async def test_blocked_words_rejected_before_role_or_send(self):
        self.bot.safety['blocked_output'] = ['synthetic-denied']
        self.assertEqual(await self.send(body='synthetic-denied'), 'blocked')
        self.assertFalse(self.bot.client.actions)

    async def test_oversized_voice_rejected_before_role_or_send(self):
        self.assertEqual(await self.send(body='长' * 201), 'blocked')
        self.assertFalse(self.bot.client.actions)

    async def test_model_voice_reply_is_opt_in_and_uses_existing_delivery_path(self):
        self.active_turn()
        self.assertTrue(await self.bot._send(GROUP, '这是合成回答', 'request', ALICE, 'at'))
        self.assertEqual(len(self.bot.client.of('send_group_ai_record')), 1)
        self.assertEqual(self.bot.client.of('send_group_ai_record')[0]['text'], '这是合成回答')

    async def test_model_ordinary_voice_topic_still_uses_text(self):
        self.active_turn('语音功能修好了吗')
        self.assertTrue(await self.bot._send(GROUP, '这是文字回答', 'request', ALICE, 'at'))
        self.assertEqual([a['action'] for a in self.bot.client.actions], ['send_group_msg'])

    async def test_model_voice_reply_superseded_before_lookup_never_sends(self):
        turn = self.active_turn()
        self.bot._latest_from_user[(GROUP, ALICE)] = replace(turn, target_revision=1)
        self.assertFalse(await self.bot._send(GROUP, '旧回答', 'request', ALICE, 'at'))
        self.assertFalse(self.bot.client.actions)

    async def test_model_voice_reply_superseded_during_lookup_never_sends(self):
        turn = self.active_turn()
        self.bot.client.after_roles = lambda: self.bot._latest_from_user.update({
            (GROUP, ALICE): replace(turn, target_revision=1)})
        self.assertFalse(await self.bot._send(GROUP, '旧回答', 'request', ALICE, 'at'))
        self.assertEqual([a['action'] for a in self.bot.client.actions], ['get_ai_characters'])

    async def test_model_voice_reply_mute_never_sends(self):
        self.active_turn()
        self.bot._muted.add((GROUP, ALICE))
        self.assertFalse(await self.bot._send(GROUP, '不应发送', 'request', ALICE, 'at'))
        self.assertFalse(self.bot.client.actions)

    async def test_model_voice_reply_quota_never_sends(self):
        self.active_turn()
        self.bot.beh['per_group_per_minute'] = 0
        self.assertFalse(await self.bot._send(GROUP, '不应发送', 'request', ALICE, 'at'))
        self.assertFalse(self.bot.client.actions)

    async def test_model_voice_reply_mute_during_lookup_rechecked(self):
        self.active_turn()
        self.bot.client.after_roles = lambda: self.bot._muted.add((GROUP, ALICE))
        self.assertFalse(await self.bot._send(GROUP, '不应发送', 'request', ALICE, 'at'))
        self.assertEqual([a['action'] for a in self.bot.client.actions], ['get_ai_characters'])

    async def test_scheduled_voice_requires_two_available_quota_slots(self):
        self.add_due()
        self.bot.beh['per_group_per_minute'] = 1
        self.bot.beh['global_per_minute'] = 1
        await self.bot._notification_tick(NOW + 2)
        self.assertEqual(self.jobs()[0]['status'], 'pending')
        self.assertFalse(self.bot.client.actions)
        self.assertTrue(self.bot.limiter.allow_many(self.bot._second_reply_limits(GROUP), consume=False))

    async def test_scheduled_voice_exact_two_quota_boundary(self):
        self.add_due()
        self.bot.beh['per_group_per_minute'] = 2
        self.bot.beh['global_per_minute'] = 2
        await self.bot._notification_tick(NOW + 2)
        self.assertEqual(self.jobs()[0]['status'], 'sent')
        self.assertEqual([a['action'] for a in self.bot.client.actions],
                         ['get_ai_characters', 'send_group_msg', 'send_group_ai_record'])
        self.assertEqual(len(self.bot.limiter._hits[('group', GROUP)]), 2)
        self.assertEqual(len(self.bot.limiter._hits[('global',)]), 2)
        self.assertEqual(await self.send(), 'deferred')

    async def test_role_unavailability_before_claim_preserves_pending_and_backs_off(self):
        job = self.add_due()
        self.bot.client.list_error = TimeoutError('synthetic no role metadata')
        await self.bot._notification_tick(NOW + 2)
        self.assertEqual(self.jobs()[0]['status'], 'pending')
        self.assertGreater(self.bot._voice_deferred_until[job['id']], time.monotonic())
        await self.bot._notification_tick(NOW + 3)
        self.assertEqual([a['action'] for a in self.bot.client.actions], ['get_ai_characters'])
        self.bot.client.list_error = None
        self.bot._voice_deferred_until[job['id']] = 0
        await self.bot._notification_tick(NOW + 4)
        self.assertEqual(self.jobs()[0]['status'], 'sent')
        self.assertEqual(len(self.bot.client.of('send_group_ai_record')), 1)

    async def test_scheduled_text_ack_missing_stops_before_ai_send(self):
        self.add_due()
        self.bot.client.text_ack = {}
        await self.bot._notification_tick(NOW + 2)
        await self.bot._notification_tick(NOW + 3)
        self.assertEqual(self.jobs()[0]['status'], 'uncertain')
        self.assertEqual(len(self.bot.client.of('send_group_msg')), 1)
        self.assertFalse(self.bot.client.of('send_group_ai_record'))

    async def test_scheduled_text_ack_timeout_stops_before_ai_send(self):
        self.add_due()
        self.bot.client.text_error = TimeoutError('synthetic lost mention receipt')
        await self.bot._notification_tick(NOW + 2)
        self.bot.client.text_error = None
        await self.bot._notification_tick(NOW + 3)
        self.assertEqual(self.jobs()[0]['status'], 'uncertain')
        self.assertEqual(len(self.bot.client.of('send_group_msg')), 1)
        self.assertFalse(self.bot.client.of('send_group_ai_record'))

    async def test_at_text_requires_strict_positive_id_before_native_ai_send(self):
        for identity in (0, '0', True, False, -1, '-1', 'garbage', '９３００１', 1.5, None, '', {}):
            with self.subTest(identity=identity):
                self.bot.client.text_ack = {'message_id': identity}
                start = len(self.bot.client.actions)
                self.assertEqual(await self.send((ALICE,)), 'uncertain')
                actions = [a['action'] for a in self.bot.client.actions[start:]]
                self.assertEqual(actions.count('send_group_msg'), 1)
                self.assertNotIn('send_group_ai_record', actions)
                self.assertFalse(self.bot._msg_index)
                self.assertFalse([m for m in self.bot.store._data.get(GROUP, ()) if m.is_bot])
        self.bot.client.text_ack = {'message_id': '93001'}
        self.assertEqual(await self.send((ALICE,)), 'sent')
        self.assertEqual(len(self.bot.client.of('send_group_ai_record')), 1)
        self.assertEqual(self.bot._msg_index[(GROUP, '93001')].user_id, BOT)

    async def test_account_switch_during_scheduling_no_saved_job_or_confirmation(self):
        self.bot.client.after_roles = lambda: setattr(self.bot.client, 'self_id', BOT + 1)
        with patch('notifications.time.time', return_value=NOW):
            await self.event('明早9点用语音叫我起床')
        self.assertFalse(self.jobs())
        self.assertFalse(self.jobs(BOT + 1))
        self.assertEqual([a['action'] for a in self.bot.client.actions], ['get_ai_characters'])

    async def test_account_switch_during_character_list_no_false_confirmation(self):
        self.bot.client.after_roles = lambda: setattr(self.bot.client, 'self_id', BOT + 1)
        await self.event('查看语音音色')
        self.assertEqual([a['action'] for a in self.bot.client.actions], ['get_ai_characters'])

    async def test_model_zero_ack_is_submitted_success_without_zero_index(self):
        self.active_turn()
        self.bot.client.voice_ack = {'message_id': 0}
        self.assertTrue(await self.bot._send(GROUP, '合成回答', 'request', ALICE, 'at'))
        self.assertEqual(len(self.bot.client.of('send_group_ai_record')), 1)
        self.assertNotIn((GROUP, '0'), self.bot._msg_index)

    async def test_disabled_character_list_never_queries_or_reveals_voices(self):
        self.bot.cfg['voice']['enabled'] = False
        for alias in ('查看语音音色', '列出语音音色', '看看语音音色'):
            self.bot.client.actions.clear()
            await self.event(alias, mid='disabled-' + alias)
            self.assertEqual([a['action'] for a in self.bot.client.actions], ['send_group_msg'])
            text = ''.join(s['data']['text'] for s in self.bot.client.of('send_group_msg')[0]['message'] if s['type'] == 'text')
            self.assertNotIn('synthetic-fish', text)
            self.assertNotIn('合成肥鱼', text)
            self.assertIsNone(self.bot._qq_voice)

    async def test_disabled_voice_keeps_pending_job_untouched(self):
        self.bot.cfg['voice']['enabled'] = False
        self.add_due()
        await self.bot._notification_tick(NOW + 2)
        self.assertEqual(self.jobs()[0]['status'], 'pending')
        self.assertFalse(self.bot.client.actions)

    async def test_disabled_generated_voice_request_still_replies_as_text(self):
        self.bot.cfg['voice']['enabled'] = False
        self.active_turn()
        self.assertTrue(await self.bot._send(GROUP, '先用文字聊', 'request', ALICE, 'at'))
        self.assertEqual([a['action'] for a in self.bot.client.actions], ['send_group_msg'])

    async def test_disabled_voice_job_does_not_expire_while_paused(self):
        self.bot.cfg['voice']['enabled'] = False
        self.add_due()
        await self.bot._notification_tick(NOW + 1000)
        self.assertEqual(self.jobs()[0]['status'], 'pending')
        self.assertFalse(self.bot.client.actions)
        self.bot.cfg['voice']['enabled'] = True
        await self.bot._notification_tick(NOW + 1000)
        self.assertEqual(self.jobs()[0]['status'], 'expired')
        self.assertFalse(self.bot.client.actions)

    async def test_disabled_voice_backlog_never_starves_text_jobs(self):
        self.bot.cfg['voice']['enabled'] = False
        for index in range(10):
            requester = ALICE + 100 + index
            self.add_due(body='暂停语音' + str(index), requester_id=requester, targets=(requester,))
        text_job = self.add_due(body='文字提醒', due=NOW + 2, delivery='text')
        await self.bot._notification_tick(NOW + 3)
        self.assertEqual([a['action'] for a in self.bot.client.actions], ['send_group_msg'])
        rows = self.jobs()
        self.assertEqual(next(j for j in rows if j['id'] == text_job['id'])['status'], 'sent')
        for index in range(10):
            row = self.bot._notification_store().list_owned(GROUP, ALICE + 100 + index, bot_id=BOT)[0]
            self.assertEqual(row['status'], 'pending')

    async def test_disabled_direct_character_selection_has_no_lookup(self):
        self.bot.cfg['voice']['enabled'] = False
        with self.assertRaises(VoiceUnavailable):
            await self.bot._voice_engine().character(self.bot.client, GROUP, BOT)
        self.assertFalse(self.bot.client.actions)

    async def test_model_uncertain_voice_does_not_fall_back_to_text(self):
        self.active_turn()
        self.bot.client.voice_ack = {}
        self.assertFalse(await self.bot._send(GROUP, '合成回答', 'request', ALICE, 'at'))
        self.assertEqual([a['action'] for a in self.bot.client.actions],
                         ['get_ai_characters', 'send_group_ai_record'])

    async def test_account_switch_during_native_voice_ack_is_uncertain_without_index(self):
        turn = self.active_turn()
        dialogue = DialogueState(BOB, '新账号已有话题', '新账号已有回答', time.monotonic())
        self.bot._dialogues[GROUP] = dialogue
        original_send = self.bot.client.send_action

        async def switching_send(action, params=None, **kwargs):
            data = await original_send(action, params, **kwargs)
            if action == 'send_group_ai_record':
                self.bot.client.self_id = BOT + 1
            return data

        self.bot.client.send_action = switching_send
        self.assertEqual(await self.send(turn=turn), 'uncertain')
        self.assertEqual(len(self.bot.client.of('send_group_ai_record')), 1)
        self.assertFalse(self.bot._msg_index)
        self.assertIs(self.bot._dialogues[GROUP], dialogue)
        self.assertFalse([m for m in self.bot.store._data.get(GROUP, ()) if m.is_bot])

    async def test_account_switch_during_at_text_ack_no_index_or_native_voice(self):
        self.add_due()
        original_send = self.bot.client.send_action

        async def switching_send(action, params=None, **kwargs):
            data = await original_send(action, params, **kwargs)
            if action == 'send_group_msg':
                self.bot.client.self_id = BOT + 1
            return data

        self.bot.client.send_action = switching_send
        await self.bot._notification_tick(NOW + 2)
        await self.bot._notification_tick(NOW + 3)
        self.assertEqual(self.jobs()[0]['status'], 'uncertain')
        self.assertEqual(len(self.bot.client.of('send_group_msg')), 1)
        self.assertFalse(self.bot.client.of('send_group_ai_record'))
        self.assertFalse(self.bot._msg_index)
        self.assertFalse([m for m in self.bot.store._data.get(GROUP, ()) if m.is_bot])

    async def test_superseded_during_native_ack_keeps_new_dialogue_and_transport_history(self):
        turn = self.active_turn()
        newer_dialogue = DialogueState(BOB, '后来群友的话', '后来群友的回答', time.monotonic())
        self.bot._dialogues[GROUP] = newer_dialogue
        self.bot._friendly_sessions[(GROUP, ALICE)] = 123.0
        self.bot._last_group[GROUP] = 234.0
        self.bot._last_user[(GROUP, ALICE)] = 345.0
        self.bot._last_reply[GROUP] = (BOB, 456.0)
        original_send = self.bot.client.send_action

        async def superseding_send(action, params=None, **kwargs):
            data = await original_send(action, params, **kwargs)
            if action == 'send_group_ai_record':
                newer = replace(turn, target_revision=turn.target_revision + 1)
                self.bot._latest_from_user[(GROUP, ALICE)] = newer
                self.bot._active_turn[GROUP] = newer
            return data

        self.bot.client.send_action = superseding_send
        with patch.object(self.bot, '_record_delivery', wraps=self.bot._record_delivery) as record:
            self.assertEqual(await self.send(turn=turn, body='已经在途的旧回答', reason='at'), 'sent')
            record.assert_not_called()
        self.assertIs(self.bot._dialogues[GROUP], newer_dialogue)
        self.assertEqual(self.bot._friendly_sessions[(GROUP, ALICE)], 123.0)
        self.assertEqual(self.bot._last_group[GROUP], 234.0)
        self.assertEqual(self.bot._last_user[(GROUP, ALICE)], 345.0)
        self.assertEqual(self.bot._last_reply[GROUP], (BOB, 456.0))
        history = [m for m in self.bot.store._data[GROUP] if m.is_bot]
        self.assertEqual([m.text for m in history], ['已经在途的旧回答'])
        self.assertEqual(history[0].sender_id, str(BOT))
        self.assertEqual(self.bot._msg_index[(GROUP, '91001')].user_id, BOT)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'success': result.wasSuccessful(), 'network': 'none',
              'fixtures': 'synthetic IDs; temporary SQLite; fake native OneBot actions',
              'real_qq': False, 'audio_playback': False, 'get_ai_record': 'forbidden'}
    if RESULT:
        RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
