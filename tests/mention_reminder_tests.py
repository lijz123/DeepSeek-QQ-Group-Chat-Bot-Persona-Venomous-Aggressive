# -*- coding: utf-8 -*-
"""Offline mention/reminder regression: synthetic IDs, temporary DB, fake OneBot.

python mention_reminder_tests.py [project_dir] [result.json]
No credentials, network, model requests, live QQ, or process restart.
"""
from __future__ import annotations

from layout_support import app_directory, project_directory, configure_test_paths
configure_test_paths()
import asyncio
import copy
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

if len(sys.argv) > 1:
    sys.path.insert(0, str(app_directory(sys.argv[1])))
RESULT = Path(sys.argv[2]) if len(sys.argv) > 2 else None
from reminders import parse_command, ReminderStore
from bot import Bot, DEFAULTS, parse_message_details
from dialogue import DialogueState

BOT, ALICE, BOB, GROUP = 10001, 20002, 20003, 88888
TZ = timezone(timedelta(hours=8))
NOW = datetime(2026, 10, 25, 6, 0, tzinfo=TZ).timestamp()

def command(text, mentions=(), now=NOW):
    return parse_command(text, tuple(str(x) for x in mentions), ALICE, BOT, now)

class ParserTests(unittest.TestCase):
    def test_incidental_time_is_not_command(self):
        self.assertIsNone(command('明天早上8点吃早饭'))
    def test_other_chat_is_not_command(self):
        self.assertIsNone(command('晚上好啊大肥鱼'))
    def test_mention_metadata_is_not_command(self):
        self.assertIsNone(command('你好', (BOB,)))
    def test_immediate_self_actual_sender(self):
        c = command('立即艾特我，测试完成')
        self.assertEqual(c.action, 'immediate'); self.assertEqual(c.targets, (ALICE,))
        self.assertTrue(c.body.strip())
    def test_other_target_from_metadata(self):
        c = command('立即艾特一下，开会啦', (BOT, BOB))
        self.assertEqual(c.action, 'immediate'); self.assertEqual(c.targets, (BOB,))
        self.assertIn('开会', c.body)
    def test_target_deduplicated(self):
        c = command('立即艾特一下，开会啦', (BOB, BOB, BOT))
        self.assertEqual(c.targets, (BOB,))
    def test_model_or_text_id_never_becomes_target(self):
        c = command('艾特123456，开会啦')
        self.assertEqual(c.action, 'error'); self.assertFalse(c.targets)
    def test_nickname_not_guessed(self):
        c = command('艾特小明，开会啦')
        self.assertEqual(c.action, 'error'); self.assertFalse(c.targets)
    def test_at_all_rejected(self):
        self.assertEqual(command('立即艾特一下，开会啦', ('all',)).action, 'error')
    def test_at_bot_not_target(self):
        c = command('立即艾特一下，开会啦', (BOT,))
        self.assertEqual(c.action, 'error'); self.assertNotIn(BOT, c.targets)
    def test_too_many_targets_rejected(self):
        self.assertEqual(command('立即艾特一下，开会啦', range(30001,30007)).action, 'error')
    def test_missing_body_has_nonempty_default(self):
        c = command('立即艾特我')
        self.assertEqual(c.action, 'immediate'); self.assertTrue(c.body.strip())
    def test_relative_minutes(self):
        c = command('10分钟后艾特我，记得休息')
        self.assertEqual(c.action,'schedule'); self.assertAlmostEqual(c.due, NOW+600)
        self.assertIn('休息', c.body)
    def test_relative_seconds(self):
        c = command('30秒后艾特我，喝水')
        self.assertEqual(c.action,'schedule'); self.assertAlmostEqual(c.due,NOW+30)
    def test_relative_hours(self):
        c = command('2小时后提醒我喝水')
        self.assertEqual(c.action,'schedule'); self.assertAlmostEqual(c.due,NOW+7200)
    def test_tomorrow_morning(self):
        c = command('明天早上8点艾特我，提醒我起床')
        self.assertEqual(c.action,'schedule')
        self.assertEqual(datetime.fromtimestamp(c.due,TZ).isoformat(),'2026-10-26T08:00:00+08:00')
        self.assertTrue(c.body.strip())
    def test_afternoon_time(self):
        c = command('今天下午3点30分艾特我，开会')
        self.assertEqual(c.action,'schedule')
        self.assertEqual(datetime.fromtimestamp(c.due,TZ).hour,15)
        self.assertEqual(datetime.fromtimestamp(c.due,TZ).minute,30)
    def test_colon_clock(self):
        c = command('08:00提醒我起床')
        self.assertEqual(c.action,'schedule'); self.assertEqual(c.due,NOW+7200)
    def test_clock_rolls_to_next_day(self):
        c = command('05:00提醒我起床')
        self.assertEqual(c.action,'schedule')
        self.assertEqual(datetime.fromtimestamp(c.due,TZ).day,26)
    def test_explicit_past_today_rejected(self):
        self.assertEqual(command('今天凌晨5点艾特我，起床').action,'error')
    def test_impossible_clock_rejected(self):
        self.assertEqual(command('明天25:99提醒我起床').action,'error')
    def test_ambiguous_bare_hour_rejected(self):
        self.assertEqual(command('8点艾特我，起床').action,'error')
    def test_zero_delay_rejected(self):
        self.assertEqual(command('0分钟后艾特我，喝水').action,'error')
    def test_horizon_rejected(self):
        self.assertEqual(command('900小时后艾特我，喝水').action,'error')
    def test_recurring_rejected(self):
        self.assertEqual(command('每天早上8点艾特我，起床').action,'error')
    def test_list_command(self):
        self.assertEqual(command('查看提醒').action,'list')
    def test_cancel_command(self):
        c = command('取消提醒 abcdef12')
        self.assertEqual(c.action,'cancel'); self.assertEqual(c.job_id,'abcdef12')
    def test_cancel_all_command(self):
        c = command('取消全部提醒')
        self.assertEqual(c.action,'cancel'); self.assertEqual(c.job_id,'all')

    def test_hypothetical_not_command(self):
        c=command('如果10分钟后艾特我，说：喝水')
        self.assertTrue(c is None or c.action=='error')
    def test_reported_command_not_command(self):
        c=command('他说10分钟后提醒我喝水')
        self.assertTrue(c is None or c.action=='error')
    def test_quoted_command_not_command(self):
        c=command('“10分钟后艾特我，喝水”')
        self.assertTrue(c is None or c.action=='error')
    def test_negative_time_never_positive(self):
        c=command('-1分钟后艾特我，喝水')
        self.assertTrue(c is None or c.action=='error')
    def test_decimal_time_never_truncated(self):
        c=command('1.5小时后艾特我，喝水')
        self.assertTrue(c is None or c.action=='error')
    def test_unknown_time_never_immediate(self):
        for text in ('10分后艾特我，喝水','一刻钟后艾特我，喝水'):
            with self.subTest(text=text):
                c=command(text)
                if text.startswith('一刻钟后'):
                    self.assertEqual(c.action,'schedule')
                    self.assertEqual(c.due,NOW+15*60)
                else:
                    self.assertTrue(c is None or c.action=='error')
    def test_negated_command_not_scheduled(self):
        c=command('别在明天早上8点艾特我，起床')
        self.assertTrue(c is None or c.action=='error')
    def test_question_in_body_does_not_cancel_schedule(self):
        c=command('10分钟后艾特我，说：你能准时到吗')
        self.assertEqual(c.action,'schedule'); self.assertIn('你能准时到吗',c.body)
    def test_time_in_immediate_body_preserved(self):
        c=command('立即艾特我，说：明天早上8点开会')
        self.assertEqual(c.action,'immediate'); self.assertIn('明天早上8点',c.body)

    def test_ambiguous_midnight_twelve_requires_24h(self):
        for text in ('晚上12点艾特我，休息','明天早上12点艾特我，休息'):
            with self.subTest(text=text):
                self.assertEqual(command(text).action,'error')
    def test_night_zero_never_noon(self):
        c=command('夜里0点艾特我，休息')
        self.assertEqual(c.action,'schedule')
        self.assertEqual(datetime.fromtimestamp(c.due,TZ).hour,0)
    def test_explicit_noon_twelve(self):
        c=command('今天中午12点艾特我，吃饭')
        self.assertEqual(c.action,'schedule')
        self.assertEqual(datetime.fromtimestamp(c.due,TZ).hour,12)

class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.path=Path(self.tmp.name)/'reminders.sqlite3'
        self.s=ReminderStore(self.path)
    def tearDown(self):
        self.s.close(); self.tmp.cleanup()
    def add(self,**kw):
        vals=dict(group_id=GROUP, requester_id=ALICE, targets=(ALICE,), body='喝水', due=NOW+60, now=NOW, bot_id=BOT)
        vals.update(kw); return self.s.add(**vals)
    def test_add_and_list_scope(self):
        j=self.add(); rows=self.s.list_owned(GROUP,ALICE,bot_id=BOT)
        self.assertEqual(len(rows),1); self.assertEqual(rows[0]['id'],j['id'])
        self.assertEqual(tuple(rows[0]['targets']),(ALICE,)); self.assertEqual(rows[0]['body'],'喝水')
    def test_pending_survives_restart(self):
        j=self.add(); self.s.close(); self.s=ReminderStore(self.path)
        self.assertEqual(self.s.list_owned(GROUP,ALICE,bot_id=BOT)[0]['id'],j['id'])
    def test_wrong_owner_cannot_list(self):
        self.add(); self.assertEqual(self.s.list_owned(GROUP,BOB,bot_id=BOT),[])
    def test_wrong_group_cannot_list(self):
        self.add(); self.assertEqual(self.s.list_owned(GROUP+1,ALICE,bot_id=BOT),[])
    def test_wrong_bot_cannot_list(self):
        self.add(); self.assertEqual(self.s.list_owned(GROUP,ALICE,bot_id=BOT+1),[])
    def test_wrong_owner_cannot_cancel(self):
        j=self.add(); self.assertEqual(self.s.cancel_owned(GROUP,BOB,j['id'],bot_id=BOT),0)
        self.assertEqual(len(self.s.list_owned(GROUP,ALICE,bot_id=BOT)),1)
    def test_wrong_group_cannot_cancel(self):
        j=self.add(); self.assertEqual(self.s.cancel_owned(GROUP+1,ALICE,j['id'],bot_id=BOT),0)
    def test_wrong_bot_cannot_cancel(self):
        j=self.add(); self.assertEqual(self.s.cancel_owned(GROUP,ALICE,j['id'],bot_id=BOT+1),0)
    def test_owned_cancel_suppresses_due(self):
        j=self.add(); self.assertEqual(self.s.cancel_owned(GROUP,ALICE,j['id'],bot_id=BOT),1)
        self.assertEqual(self.s.due_candidates(NOW+61,bot_id=BOT),[])
    def test_cancel_all_only_own(self):
        self.add(); self.add(requester_id=BOB,targets=(BOB,))
        self.assertEqual(self.s.cancel_owned(GROUP,ALICE,'all',bot_id=BOT),1)
        self.assertEqual(len(self.s.list_owned(GROUP,BOB,bot_id=BOT)),1)
    def test_not_due_before_time(self):
        self.add(); self.assertEqual(self.s.due_candidates(NOW+59,bot_id=BOT),[])
    def test_due_job_found(self):
        j=self.add(); rows=self.s.due_candidates(NOW+61,bot_id=BOT)
        self.assertEqual([x['id'] for x in rows],[j['id']])
    def test_due_account_scope(self):
        self.add(); self.assertEqual(self.s.due_candidates(NOW+61,bot_id=BOT+1),[])
    def test_claim_once(self):
        j=self.add(); self.assertTrue(self.s.claim(j['id'],NOW+61,bot_id=BOT))
        self.assertFalse(self.s.claim(j['id'],NOW+61,bot_id=BOT))
    def test_claim_requires_due(self):
        j=self.add(); self.assertFalse(self.s.claim(j['id'],NOW+59,bot_id=BOT))
    def test_claim_wrong_bot(self):
        j=self.add(); self.assertFalse(self.s.claim(j['id'],NOW+61,bot_id=BOT+1))
    def test_sent_not_due_again(self):
        j=self.add(); self.s.claim(j['id'],NOW+61,bot_id=BOT); self.s.finish(j['id'],'sent',bot_id=BOT)
        self.assertEqual(self.s.due_candidates(NOW+62,bot_id=BOT),[])
    def test_uncertain_not_due_again(self):
        j=self.add(); self.s.claim(j['id'],NOW+61,bot_id=BOT); self.s.finish(j['id'],'uncertain',bot_id=BOT)
        self.assertEqual(self.s.due_candidates(NOW+62,bot_id=BOT),[])
    def test_interrupted_sending_never_retries(self):
        j=self.add(); self.s.claim(j['id'],NOW+61,bot_id=BOT); self.s.close(); self.s=ReminderStore(self.path)
        self.assertEqual(self.s.due_candidates(NOW+62,bot_id=BOT),[])
        self.assertFalse(self.s.claim(j['id'],NOW+62,bot_id=BOT))
    def test_expired_job_never_claimed(self):
        j=self.add(); self.assertEqual(self.s.due_candidates(NOW+361,bot_id=BOT,grace_seconds=300),[])
        self.assertFalse(self.s.claim(j['id'],NOW+361,bot_id=BOT,grace_seconds=300))
    def test_requester_pending_limit(self):
        for i in range(5): self.add(due=NOW+60+i)
        with self.assertRaises(ValueError): self.add(due=NOW+100)
    def test_empty_body_rejected(self):
        with self.assertRaises(ValueError): self.add(body='  ')
    def test_no_at_all_in_store(self):
        with self.assertRaises((ValueError,TypeError)): self.add(targets=('all',))
    def test_source_id_deduplication(self):
        a=self.add(source_id='synthetic-123'); b=self.add(source_id='synthetic-123')
        self.assertEqual(a['id'],b['id']); self.assertEqual(len(self.s.list_owned(GROUP,ALICE,bot_id=BOT)),1)

    def test_uncertain_status_visible_for_user_inspection(self):
        j=self.add(); self.s.claim(j['id'],NOW+61,bot_id=BOT)
        self.s.finish(j['id'],'uncertain',bot_id=BOT)
        rows=self.s.list_owned(GROUP,ALICE,bot_id=BOT)
        self.assertEqual(rows[0]['status'],'uncertain')
    def test_two_open_store_claim_only_once(self):
        j=self.add(); other=ReminderStore(self.path)
        try:
            other.list_owned(GROUP,ALICE,bot_id=BOT)
            self.assertTrue(self.s.claim(j['id'],NOW+61,bot_id=BOT))
            self.assertFalse(other.claim(j['id'],NOW+61,bot_id=BOT))
        finally:
            other.close()
    def test_group_pending_limit(self):
        for i in range(30): self.add(requester_id=30000+i)
        with self.assertRaises(ValueError): self.add(requester_id=40000)
    def test_oversized_body_rejected(self):
        with self.assertRaises(ValueError): self.add(body='长'*301)

class FakeClient:
    def __init__(self):
        self.self_id=BOT; self.nickname='测试机器人'; self._ready=True
        self.ws=SimpleNamespace(closed=False); self.sent=[]; self.fail=False; self.no_ack=False
    async def send_action(self,action,params=None,**kw):
        if self.fail: raise TimeoutError('offline simulated uncertain send')
        self.sent.append({'action':action,**(params or {})})
        return {} if self.no_ack else {'message_id':str(90000+len(self.sent))}

class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        cfg=copy.deepcopy(DEFAULTS)
        cfg['llm']['api_key']=''; cfg['runtime'].update(dry_run=False,log_file='')
        cfg['groups']={'whitelist':[],'blacklist':[]}; cfg['users']={'blacklist':[]}
        cfg['behavior'].update(min_gap=0,per_group_per_minute=100,global_per_minute=100)
        cfg['notifications']={'enabled':True,'database':str(Path(self.tmp.name)/'jobs.sqlite3'),'poll_interval':1,'grace_seconds':300}
        self.bot=Bot(cfg); self.bot.client=FakeClient()
        self.bot.brain.plan_reply=AsyncMock(side_effect=AssertionError('commands must not call LLM'))
        self.bot.brain.reply=AsyncMock(side_effect=AssertionError('commands must not call LLM'))
    async def asyncTearDown(self):
        await self.bot.close(); self.tmp.cleanup()
    def parsed(self,text,targets=(BOT,)):
        segments=[{'type':'at','data':{'qq':str(x)}} for x in targets]
        segments.append({'type':'text','data':{'text':text}})
        return parse_message_details(segments,BOT)
    async def handle(self,text,targets=(BOT,),user=ALICE,group=GROUP):
        return await self.bot._handle_notification_command(group,user,self.parsed(text,targets),'synthetic-'+str(time.monotonic_ns()))
    def store(self):
        return self.bot._notification_store()
    def add_due(self,**kwargs):
        args=dict(group_id=GROUP,requester_id=ALICE,targets=(ALICE,),body='到点喝水',due=NOW+1,now=NOW,bot_id=BOT)
        args.update(kwargs); return self.store().add(**args)
    def assertAtBody(self,message,target):
        at=[s['data']['qq'] for s in message if s['type']=='at']
        body=''.join(s['data']['text'] for s in message if s['type']=='text')
        self.assertEqual(at,[str(target)]); self.assertTrue(body.strip())
    async def test_immediate_real_at_plus_body(self):
        self.assertTrue(await self.handle('立即艾特我，测试完成'))
        self.assertAtBody(self.bot.client.sent[-1]['message'],ALICE)
        self.bot.brain.plan_reply.assert_not_awaited(); self.bot.brain.reply.assert_not_awaited()
    async def test_immediate_other_actual_at(self):
        self.assertTrue(await self.handle('立即艾特一下，开会啦',(BOT,BOB)))
        self.assertAtBody(self.bot.client.sent[-1]['message'],BOB)
    async def test_send_notification_no_empty_body(self):
        result=await self.bot._send_notification(GROUP,ALICE,(ALICE,),' ')
        self.assertEqual(result,'blocked'); self.assertFalse(self.bot.client.sent)
    async def test_send_notification_no_all(self):
        result=await self.bot._send_notification(GROUP,ALICE,('all',),'测试')
        self.assertEqual(result,'blocked'); self.assertFalse(self.bot.client.sent)
    async def test_send_notification_no_self_bot(self):
        result=await self.bot._send_notification(GROUP,ALICE,(BOT,),'测试')
        self.assertEqual(result,'blocked'); self.assertFalse(self.bot.client.sent)
    async def test_notification_does_not_hijack_conversation(self):
        old=DialogueState(BOB,'旧话题','旧回答',time.monotonic(),2)
        self.bot._dialogues[GROUP]=old
        self.assertEqual(await self.bot._send_notification(GROUP,ALICE,(ALICE,),'测试'),'sent')
        self.assertIs(self.bot._dialogues[GROUP],old)
    async def test_group_whitelist_enforced(self):
        self.bot._whitelist={GROUP+1}
        self.assertEqual(await self.bot._send_notification(GROUP,ALICE,(ALICE,),'测试'),'blocked')
        self.assertFalse(self.bot.client.sent)
    async def test_user_blacklist_enforced(self):
        self.bot._user_blacklist.add(ALICE)
        self.assertEqual(await self.bot._send_notification(GROUP,ALICE,(ALICE,),'测试'),'blocked')
    async def test_target_blacklist_enforced(self):
        self.bot._user_blacklist.add(BOB)
        self.assertEqual(await self.bot._send_notification(GROUP,ALICE,(BOB,),'测试'),'blocked')
    async def test_mute_enforced(self):
        self.bot._muted.add((GROUP,ALICE))
        self.assertEqual(await self.bot._send_notification(GROUP,ALICE,(ALICE,),'测试'),'blocked')
    async def test_rate_limit_defers(self):
        self.bot.beh['per_group_per_minute']=0
        self.assertEqual(await self.bot._send_notification(GROUP,ALICE,(ALICE,),'测试'),'deferred')
        self.assertFalse(self.bot.client.sent)
    async def test_disconnection_defers_without_send(self):
        self.bot.client._ready=False
        self.assertEqual(await self.bot._send_notification(GROUP,ALICE,(ALICE,),'测试'),'deferred')
        self.assertFalse(self.bot.client.sent)
    async def test_dry_run_no_send_no_schedule(self):
        self.bot.cfg['runtime']['dry_run']=True
        self.assertEqual(await self.bot._send_notification(GROUP,ALICE,(ALICE,),'测试'),'dry_run')
        await self.handle('10分钟后艾特我，喝水')
        self.assertFalse(self.bot.client.sent)
        self.assertFalse(self.store().list_owned(GROUP,ALICE,bot_id=BOT))
    async def test_command_schedule_saved_before_confirmation(self):
        self.assertTrue(await self.handle('10分钟后艾特我，喝水'))
        rows=self.store().list_owned(GROUP,ALICE,bot_id=BOT)
        self.assertEqual(len(rows),1); self.assertTrue(rows[0]['body'])
        self.assertTrue(self.bot.client.sent)
    async def test_due_sends_real_at_and_body_once(self):
        self.add_due(); await self.bot._notification_tick(NOW+2); await self.bot._notification_tick(NOW+3)
        self.assertEqual(len(self.bot.client.sent),1)
        self.assertAtBody(self.bot.client.sent[0]['message'],ALICE)
    async def test_not_due_no_send(self):
        self.add_due(); await self.bot._notification_tick(NOW)
        self.assertFalse(self.bot.client.sent)
    async def test_due_other_bot_not_sent(self):
        self.add_due(bot_id=BOT+1); await self.bot._notification_tick(NOW+2)
        self.assertFalse(self.bot.client.sent)
    async def test_disconnected_job_remains_pending(self):
        self.add_due(); self.bot.client._ready=False; await self.bot._notification_tick(NOW+2)
        self.assertFalse(self.bot.client.sent)
        self.assertEqual(len(self.store().list_owned(GROUP,ALICE,bot_id=BOT)),1)
    async def test_reconnect_delivers_pending_once(self):
        self.add_due(); self.bot.client._ready=False; await self.bot._notification_tick(NOW+2)
        self.bot.client._ready=True; await self.bot._notification_tick(NOW+3)
        self.assertEqual(len(self.bot.client.sent),1)
    async def test_uncertain_delivery_not_retried(self):
        self.add_due(); self.bot.client.fail=True; await self.bot._notification_tick(NOW+2)
        self.bot.client.fail=False; await self.bot._notification_tick(NOW+3)
        self.assertFalse(self.bot.client.sent)
        self.assertEqual([r['status'] for r in self.store().list_owned(GROUP,ALICE,bot_id=BOT)],['uncertain'])
    async def test_missing_ack_not_retried(self):
        self.add_due(); self.bot.client.no_ack=True; await self.bot._notification_tick(NOW+2)
        self.bot.client.no_ack=False; await self.bot._notification_tick(NOW+3)
        self.assertEqual(len(self.bot.client.sent),1)
    async def test_due_late_beyond_grace_not_sent(self):
        self.add_due(); await self.bot._notification_tick(NOW+302)
        self.assertFalse(self.bot.client.sent)
    async def test_rate_deferral_does_not_claim(self):
        self.add_due(); self.bot.beh['per_group_per_minute']=0
        await self.bot._notification_tick(NOW+2)
        self.assertEqual(len(self.store().list_owned(GROUP,ALICE,bot_id=BOT)),1)
    async def test_no_address_no_active_session_not_intercepted(self):
        self.assertFalse(await self.handle('10分钟后艾特我，喝水',()))
        self.assertFalse(self.bot.client.sent)
    async def test_explicit_command_intercepted_before_other_addressee_filter(self):
        event={'post_type':'message','message_type':'group','group_id':GROUP,'user_id':ALICE,'message_id':'offline-123',
               'sender':{'nickname':'合成用户'},'message':[
                  {'type':'at','data':{'qq':str(BOT)}}, {'type':'text','data':{'text':' 立即艾特一下 '}},
                  {'type':'at','data':{'qq':str(BOB)}}, {'type':'text','data':{'text':'，开会啦'}}]}
        await self.bot.on_event(event)
        self.assertEqual(len(self.bot.client.sent),1)
        self.assertAtBody(self.bot.client.sent[0]['message'],BOB)
        self.assertFalse(self.bot._workers)

    async def test_active_conversation_no_at_command(self):
        self.bot._dialogues[GROUP]=DialogueState(ALICE,'之前的话','刚才的回复',time.monotonic())
        self.assertTrue(await self.handle('10分钟后艾特我，喝水',()))
        self.assertEqual(len(self.store().list_owned(GROUP,ALICE,bot_id=BOT)),1)
    async def test_bot_quote_addresses_command(self):
        self.bot._msg_index[(GROUP,'bot-quote')]=SimpleNamespace(is_bot=True)
        parsed=SimpleNamespace(text='立即艾特我，测试完成',at_self=False,mentioned_ids=(),reply_id='bot-quote')
        self.assertTrue(await self.bot._handle_notification_command(GROUP,ALICE,parsed,'quote-source'))
        self.assertAtBody(self.bot.client.sent[-1]['message'],ALICE)
    async def test_stale_conversation_not_addressed(self):
        self.bot._dialogues[GROUP]=DialogueState(ALICE,'旧话','旧答',time.monotonic()-999)
        self.assertFalse(await self.handle('10分钟后艾特我，喝水',()))
        self.assertFalse(self.bot.client.sent)
    async def test_other_user_session_not_addressed(self):
        self.bot._dialogues[GROUP]=DialogueState(BOB,'旧话','旧答',time.monotonic())
        self.assertFalse(await self.handle('10分钟后艾特我，喝水',()))
    async def test_account_change_during_delivery_defers_without_claim(self):
        self.add_due(); entered=asyncio.Event(); release=asyncio.Event()
        async def gap():
            entered.set(); await release.wait()
        self.bot._wait_delivery_gap=gap
        task=asyncio.create_task(self.bot._notification_tick(NOW+2))
        await asyncio.wait_for(entered.wait(),1)
        self.bot.client.self_id=BOT+1; release.set(); await task
        self.assertFalse(self.bot.client.sent)
        self.assertEqual(len(self.store().list_owned(GROUP,ALICE,bot_id=BOT)),1)
    async def test_cancelled_inflight_claim_never_retries_after_store_reopen(self):
        self.add_due(); entered=asyncio.Event(); release=asyncio.Event()
        async def interrupted(action,params=None,**kw):
            entered.set(); await release.wait(); return {'message_id':'synthetic-unreached'}
        old_send=self.bot.client.send_action; self.bot.client.send_action=interrupted
        task=asyncio.create_task(self.bot._notification_tick(NOW+2))
        await asyncio.wait_for(entered.wait(),1); task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.bot._reminder_store.close(); self.bot._reminder_store=None
        self.bot.client.send_action=old_send
        await self.bot._notification_tick(NOW+3)
        self.assertFalse(self.bot.client.sent)
        self.assertEqual([r['status'] for r in self.store().list_owned(GROUP,ALICE,bot_id=BOT)],['uncertain'])
    async def test_user_text_cq_does_not_inject_at_segments(self):
        self.assertEqual(await self.bot._send_notification(GROUP,ALICE,(ALICE,),'原文[CQ:at,qq=all]'),'sent')
        message=self.bot.client.sent[-1]['message']
        self.assertAtBody(message,ALICE)
        self.assertIn('[CQ:at,qq=all]',message[-1]['data']['text'])

    async def test_account_change_between_due_batch_jobs_never_cross_sends(self):
        self.add_due(body='第一条'); self.add_due(targets=(BOB,),body='第二条')
        original=self.bot.client.send_action
        async def switch_after_ack(action,params=None,**kw):
            data=await original(action,params,**kw)
            self.bot.client.self_id=BOT+1
            return data
        self.bot.client.send_action=switch_after_ack
        await self.bot._notification_tick(NOW+2)
        await self.bot._notification_tick(NOW+3)
        self.assertEqual(len(self.bot.client.sent),1)
        rows=self.store().list_owned(GROUP,ALICE,bot_id=BOT)
        self.assertEqual(sorted(r['status'] for r in rows),['pending','sent'])


    async def test_account_switch_does_not_cancel_old_job_targeting_new_account(self):
        self.add_due(due=NOW+1)
        second=self.add_due(due=NOW+2,targets=(BOB,))
        send=self.bot.client.send_action
        async def switch_after_first(action,params=None,**kwargs):
            result=await send(action,params,**kwargs)
            self.bot.client.self_id=BOB
            return result
        self.bot.client.send_action=switch_after_first
        await self.bot._notification_tick(NOW+3)
        rows={j['id']:j for j in self.store().list_owned(GROUP,ALICE,bot_id=BOT)}
        self.assertEqual(rows[second['id']]['status'],'pending')
        self.assertEqual(len(self.bot.client.sent),1)

class CapabilityBoundaryTests(unittest.TestCase):
    def test_legacy_prompt_cannot_fake_schedule_creation(self):
        from persona import build_system_prompt
        text=build_system_prompt('机器人')
        self.assertIn('语言模型不能自行发送@或创建后台任务',text)
        self.assertIn('不得说已安排',text)
    def test_semantic_expression_cannot_fake_schedule_creation(self):
        from semantics import SemanticFrame, build_expression_system
        f=SemanticFrame('ask','bot','询问定时提醒','','说明功能','normal',False)
        text=build_expression_system('机器人','sharp','','',f)
        self.assertIn('语言模型不能自行发送@或创建后台任务',text)
        self.assertIn('程序回执为准',text)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'success':result.wasSuccessful(),'network':'none','fixtures':'synthetic IDs, temporary SQLite, fake OneBot'}
    if RESULT: RESULT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
