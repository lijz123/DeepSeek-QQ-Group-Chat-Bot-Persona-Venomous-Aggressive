"""Offline opt-in complete text, generation budget, and physical delivery tests.

Uses synthetic content, fake HTTP and fake OneBot only. No config.json, secrets,
model billing, external network, or real QQ messages are read or sent.
"""
from __future__ import annotations

from layout_support import app_directory, project_directory, configure_test_paths
configure_test_paths()

import asyncio
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

SOURCE = app_directory(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None)
RESULT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
sys.path.insert(0, str(SOURCE))
sys.dont_write_bytecode = True

from bot import Bot, Brain, Msg
from dialogue import PendingTurn, DialogueState
from dialogue_selftest import test_config, FakeClient, BOT_ID, GROUP, ALICE
from persona import sanitize_reply, parse_reply_plan, build_system_prompt, build_user_prompt, ReplyPlan
from owner_chat import parse_owner_chat_plan, build_owner_chat_messages
from output_policy import wide_output, wide_prompt, preserve_output, split_output
import output_delivery
from search_dialogue import build_search_messages, parse_search_answer, append_search_sources
from semantics import SemanticFrame, build_expression_system
from web_search import SearchOutcome


SAMPLE = "  ```c\n#include <stdint.h>\n#define GPIO_PIN_A2 (1u << 2)\n\n@decorator\nvoid init(void) {\n\tfor (int i = 0; i < 4; ++i) { led[i] = '#'; }\n}\n```\n  "
LONG = SAMPLE * 90
LARGE = SAMPLE * 500
FRAME = dict(act='ask', target='topic', meaning='请求完整程序代码', unresolved='',
             reply_goal='直接提供实际示例', mode='normal', needs_clarification=False)
KEY = 'offline-synthetic-key-NOT-A-CREDENTIAL'
PUBLIC = 'https://docs.python.org/3/library/asyncio.html'


def cfg(enabled=True):
    value = test_config()
    value['output'] = {'unlimited': enabled, 'max_tokens': 393216, 'chunk_chars': 256}
    value['llm'].update(provider='deepseek', base_url='https://api.deepseek.com/v1', api_key=KEY,
        model='deepseek-chat', semantic_pipeline=True, semantic_review_policy='all',
        max_tokens=100, planning_max_tokens=192)
    value['voice']['enabled'] = False
    value['search']['enabled'] = False
    value['owner_chat']['enabled'] = False
    value['behavior'].update(max_chars=120, min_gap=0, direct_cooldown=0, direct_user_cooldown=0,
        quote_mode='always', concurrent_reply_window=0, max_chain=3,
        per_group_per_minute=100, global_per_minute=100)
    value['safety']['blocked_output'] = []
    return value


def raw(text, **changes):
    value = {'action': 'reply', 'text': text, 'next': 'wait'}
    value.update(changes)
    return json.dumps(value, ensure_ascii=False)


def transcript(text='肥鱼，写完整示例代码'):
    return json.dumps({'role': 'group_member', 'sender': {'id': str(ALICE), 'display_name': '合成成员'},
        'message_id': 'wide-source', 'content': text}, ensure_ascii=False)


class TextContracts(unittest.TestCase):
    def test_wide_flag_is_opt_in(self):
        self.assertFalse(wide_output({}))
        self.assertFalse(wide_output({'output': {'unlimited': False}}))

    def test_wide_flag_requires_real_true(self):
        for invalid in (1, 'true', [], {}):
            with self.subTest(invalid=invalid): self.assertFalse(wide_output({'output': {'unlimited': invalid}}))
        self.assertTrue(wide_output({'output': {'unlimited': True}}))

    def test_ordinary_old_120_retains_ellipsis_behavior(self):
        self.assertEqual(sanitize_reply('x'*130, max_chars=120), 'x'*120+'…')

    def test_ordinary_old_multiline_still_first_line(self):
        self.assertEqual(sanitize_reply('line1\nline2', max_chars=120), 'line1')

    def test_zero_limit_retains_complete_large_code(self):
        self.assertGreater(len(LARGE), 50000)
        self.assertEqual(sanitize_reply(LARGE, max_chars=0), LARGE)

    def test_zero_limit_retains_all_lines_and_whitespace(self):
        self.assertEqual(sanitize_reply(SAMPLE, max_chars=0), SAMPLE)

    def test_zero_limit_retains_crlf_exactly(self):
        code = '\r\n\tcode();\r\n'
        self.assertEqual(sanitize_reply(code, max_chars=0), code)

    def test_zero_limit_retains_decorators_and_markdown(self):
        value = '@decorator\n# title\n`text`\n[1] (x) *item*'
        self.assertEqual(sanitize_reply(value, max_chars=0), value)

    def test_literal_cq_is_text_not_stripped(self):
        value = '示例：[CQ:at,qq=12345]\n不实际通知'
        self.assertEqual(sanitize_reply(value, max_chars=0), value)

    def test_unicode_emoji_and_combining_characters_preserved(self):
        value = '🐟汉字e\u0301\n' * 300
        self.assertEqual(sanitize_reply(value, max_chars=0), value)

    def test_blank_wide_reply_rejected(self):
        self.assertEqual(sanitize_reply(' \n\t', max_chars=0), '')

    def test_wide_controls_not_forwarded(self):
        self.assertEqual(sanitize_reply('hello\x00world', max_chars=0), '')

    def test_wide_surrogate_not_forwarded(self):
        self.assertEqual(sanitize_reply('hello\ud800world', max_chars=0), '')

    def test_wide_blocked_output_retains_safeguard(self):
        self.assertEqual(sanitize_reply('BLOCKED', max_chars=0, blocked=['BLOCKED'], fallback='stop'), 'stop')

    def test_wide_blocked_output_nfkc_equivalence(self):
        self.assertEqual(sanitize_reply('ＢＬＯＣＫＥＤ', max_chars=0, blocked=['BLOCKED'], fallback='stop'), 'stop')

    def test_strict_json_text_keeps_large_code(self):
        plan = parse_reply_plan(raw(LONG), max_chars=0, strict=True)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.text, LONG)

    def test_unlimited_does_not_accept_unknown_reply_protocol(self):
        self.assertIsNone(parse_reply_plan(raw(SAMPLE, execute='no'), max_chars=0, strict=True))

    def test_unlimited_does_not_accept_malformed_reply_json(self):
        self.assertIsNone(parse_reply_plan('{"text":', max_chars=0, strict=True))

    def test_owner_old_500_still_accepts(self):
        self.assertIsNotNone(parse_owner_chat_plan(raw('x'*500), max_chars=500))

    def test_owner_old_501_still_rejects(self):
        self.assertIsNone(parse_owner_chat_plan(raw('x'*501), max_chars=500))

    def test_owner_wide_accepts_raw_json_over_50000(self):
        plan = parse_owner_chat_plan(raw(LARGE), max_chars=0)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.text, LARGE)

    def test_owner_wide_preserves_crlf_and_ends(self):
        value = '\n  ' + SAMPLE.replace('\n', '\r\n') + '\r\n'
        self.assertEqual(parse_owner_chat_plan(raw(value), max_chars=0).text, value)

    def test_owner_wide_does_not_weaken_duplicate_keys(self):
        self.assertIsNone(parse_owner_chat_plan('{"action":"reply","text":"a","text":"b","next":"wait"}', max_chars=0))

    def test_owner_wide_does_not_weaken_unknown_fields(self):
        self.assertIsNone(parse_owner_chat_plan(raw(SAMPLE, sudo='no'), max_chars=0))

    def test_owner_wide_blocks_controls(self):
        self.assertIsNone(parse_owner_chat_plan(raw('bad\x00text'), max_chars=0))

    def test_owner_wide_blocks_nfkc_words(self):
        self.assertIsNone(parse_owner_chat_plan(raw('ＢＬＯＣＫＥＤ'), max_chars=0, blocked=['BLOCKED']))

    def test_split_empty(self):
        self.assertEqual(split_output(''), [])

    def test_split_preserves_code_and_every_character(self):
        parts = split_output(LONG, 256)
        self.assertEqual(''.join(parts), LONG)
        self.assertTrue(all(0 < len(part) <= 256 for part in parts))

    def test_split_preserves_large_unbroken_line(self):
        text = 'x' * 100001
        parts = split_output(text, 1500)
        self.assertEqual(''.join(parts), text)
        self.assertEqual(max(map(len, parts)), 1500)

    def test_split_prefers_line_boundary(self):
        parts = split_output('a'*200+'\n'+'b'*200, 256)
        self.assertEqual(parts[0], 'a'*200+'\n')

    def test_split_clamps_transport_max_not_total_length(self):
        parts = split_output('x'*10000, 99999)
        self.assertEqual(''.join(parts), 'x'*10000)
        self.assertEqual(max(map(len, parts)), 2000)

    def test_split_clamps_transport_min(self):
        self.assertEqual(len(split_output('x'*1000, 1)[0]), 256)

    def test_split_invalid_size_uses_default(self):
        self.assertEqual(len(split_output('x'*2000, True)[0]), 1500)

    def test_split_non_text_rejected(self):
        with self.assertRaises(TypeError): split_output(b'bytes')

    def test_prompt_says_deliver_actual_content_not_promises(self):
        prompt = wide_prompt()
        self.assertIn('实际内容', prompt)
        self.assertIn('不能承诺真正无限', prompt)
        self.assertIn('不为用满模型预算', prompt)

    def test_legacy_system_wide_removes_short_sentence_instruction(self):
        prompt = build_system_prompt('大肥鱼','sharp','','',runtime=cfg())
        self.assertNotIn('只发一句话，一般 6~30',prompt)
        self.assertNotIn('绝不分段',prompt)
        self.assertIn('完整内容输出',prompt)

    def test_legacy_system_off_keeps_short_style(self):
        prompt = build_system_prompt('大肥鱼','sharp','','',runtime=cfg(False))
        self.assertIn('6~30',prompt)

    def test_legacy_user_wide_not_one_sentence_only(self):
        prompt = build_user_prompt(transcript(),'合成成员',True,rich_output=True)
        self.assertNotIn('只输出你要发到群里的那一句话',prompt)
        self.assertIn('完整',prompt)

    def test_search_wide_keeps_complete_code_not_only_short_summary(self):
        results=({'title':'公开资料','url':PUBLIC,'snippet':'示例代码','date':''},)
        value=json.dumps({'answer':LONG,'sources':[1]},ensure_ascii=False)
        parsed=parse_search_answer(value,results,max_chars=0)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed[0],LONG)

    def test_search_wide_keeps_crlf_layout(self):
        results=({'title':'公开资料','url':PUBLIC},)
        text='\r\n  code();\r\n'
        parsed=parse_search_answer(json.dumps({'answer':text,'sources':[1]}),results,max_chars=0)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed[0],text)

    def test_search_wide_blocks_surrogate(self):
        self.assertIsNone(parse_search_answer(json.dumps({'answer':'bad\ud800text','sources':[1]}),({'url':PUBLIC},),max_chars=0))

    def test_search_wide_blocks_nfkc_equivalent_words(self):
        self.assertIsNone(parse_search_answer(json.dumps({'answer':'ＢＬＯＣＫＥＤ','sources':[1]}),({'url':PUBLIC},),max_chars=0,blocked=['BLOCKED']))

    def test_search_wide_no_total_source_message_truncation(self):
        results=({'title':'公开资料','url':PUBLIC},)
        text='文档内容\n'*1000
        result=append_search_sources(text,(1,),results,max_total_chars=0)
        self.assertTrue(result.startswith(text))
        self.assertIn(PUBLIC,result)
        self.assertGreater(len(result),3400)

    def test_search_wide_still_rejects_invented_source_index(self):
        self.assertIsNone(parse_search_answer(json.dumps({'answer':'正文','sources':[2]}),({'url':PUBLIC},),max_chars=0))


class FakeResponse:
    status = 200
    def __init__(self, content): self.body=json.dumps({'choices':[{'message':{'content':content},'finish_reason':'stop'}]})
    async def text(self): return self.body
    async def __aenter__(self): return self
    async def __aexit__(self,*a): pass


class ModelSession:
    def __init__(self, text=LONG, search=False):
        self.requests=[]
        self.text=text
        self.search=search
    def post(self,url,**kwargs):
        payload=kwargs['json'];self.requests.append(copy.deepcopy(payload))
        system=payload['messages'][0]['content']
        if '中性语义解析器' in system:
            frame=dict(FRAME)
            if 'need_search' in system:
                frame.update(need_search=self.search, search_query='Python asyncio 官方文档' if self.search else '')
            content=json.dumps(frame,ensure_ascii=False)
        elif '资料依据复核器' in system or '中性的群聊回复语义复核器' in system:
            content='{"ok":true,"issue":""}'
        elif '公开资料回答' in system:
            content=json.dumps({'answer':self.text,'sources':[1]},ensure_ascii=False)
        else:
            content=raw(self.text)
        return FakeResponse(content)
    async def close(self): pass


class BrainContracts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config=cfg();self.brain=Brain(self.config['llm'],self.config['safety'],self.config['persona'],120,runtime=self.config)
        self.session=ModelSession();self.brain._session=self.session
    async def asyncTearDown(self): await self.brain.close()

    async def test_semantic_wide_returns_every_code_character(self):
        plan=await self.brain.plan_reply(GROUP,transcript(),'合成成员',True)
        self.assertEqual(plan.text,LONG)

    async def test_semantic_generation_uses_configured_ceiling(self):
        await self.brain.plan_reply(GROUP,transcript(),'合成成员',True)
        budgets=[p['max_tokens'] for p in self.session.requests]
        self.assertEqual(budgets,[640,393216,256])

    async def test_neutral_phases_are_not_promoted_to_maximum(self):
        await self.brain.plan_reply(GROUP,transcript(),'合成成员',True)
        requests=self.session.requests
        self.assertEqual(requests[0]['max_tokens'],640)
        self.assertEqual(requests[-1]['max_tokens'],256)

    async def test_owner_wide_returns_complete_code(self):
        plan=await self.brain.owner_chat_reply(GROUP,transcript(),'合成成员',True,instruction='写完整示例代码')
        self.assertEqual(plan.text,LONG)

    async def test_owner_generation_uses_ceiling(self):
        await self.brain.owner_chat_reply(GROUP,transcript(),'合成成员',True,instruction='写完整示例代码')
        self.assertEqual(self.session.requests[0]['max_tokens'],393216)

    async def test_wide_does_not_force_output_to_fill_budget(self):
        self.session.text='好呀'
        plan=await self.brain.plan_reply(GROUP,transcript('肥鱼，晚上好'),'合成成员',True)
        self.assertEqual(plan.text,'好呀')

    async def test_ordinary_disabled_retains_truncation(self):
        self.brain.runtime['output']['unlimited']=False
        self.brain.max_chars=120
        self.session.text='x'*130
        plan=await self.brain.plan_reply(GROUP,transcript(),'合成成员',True)
        self.assertEqual(plan.text,'x'*120+'…')
        self.assertEqual([p['max_tokens'] for p in self.session.requests],[640,512,256])

    async def test_semantic_wide_system_no_short_response_rule(self):
        await self.brain.plan_reply(GROUP,transcript(),'合成成员',True)
        system=self.session.requests[1]['messages'][0]['content']
        self.assertNotIn('一般6~30字',system)
        self.assertIn('完整内容输出',system)

    async def test_search_wide_returns_over_600_and_complete_source(self):
        self.config['search']['enabled']=True
        self.session.search=True
        self.session.text='文档说明公开接口。\n'*200
        evidence=SearchOutcome('ok','Python asyncio 官方文档',({'title':'Python 官方资料','url':PUBLIC,'snippet':self.session.text,'date':'2026-10-26'},))
        class FakeSearch:
            async def search(self,*a,**k): return evidence
        self.brain._official_search_client=FakeSearch()
        plan=await self.brain.plan_reply(GROUP,transcript('肥鱼，查一下Python asyncio官方文档'),'合成成员',True)
        self.assertGreater(len(plan.text),600)
        self.assertIn(self.session.text.strip(),plan.text)
        self.assertIn(PUBLIC,plan.text)
        self.assertEqual([p['max_tokens'] for p in self.session.requests],[640,393216,256])

    async def test_legacy_planning_wide_preserves_layout(self):
        self.brain.cfg['semantic_pipeline']=False
        plan=await self.brain.plan_reply(GROUP,transcript(),'合成成员',True)
        self.assertEqual(plan.text,LONG)
        self.assertEqual(self.session.requests[0]['max_tokens'],393216)

    async def test_provider_length_finish_is_not_sent_as_complete_answer(self):
        original=self.session.post
        def limited(url,**kwargs):
            response=original(url,**kwargs)
            if kwargs['json']['max_tokens']==393216:
                value=json.loads(response.body);value['choices'][0]['finish_reason']='length'
                response.body=json.dumps(value)
            return response
        self.session.post=limited
        plan=await self.brain.plan_reply(GROUP,transcript(),'合成成员',True)
        self.assertTrue(plan is None or plan.action!='reply')
        self.assertEqual(len(self.session.requests),2)


class FakeBrain:
    def __init__(self,text): self.text=text;self.calls=0
    async def plan_reply(self,*a,**kw): self.calls+=1;return ReplyPlan('reply',self.text,'reply')
    async def close(self): pass


class DeliveryContracts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config=cfg();self.bot=Bot(self.config);self.bot.client=FakeClient()
        self.bot.brain=FakeBrain(LONG)
        now=time.monotonic()
        message=Msg('合成成员','肥鱼，请给完整代码',False,time.time(),sender_id=str(ALICE),message_id='wide-source')
        self.turn=PendingTurn(GROUP,ALICE,'wide-source',message,True,(),'', '',now,now,('wide-source',),arrival_revision=1)
        self.bot._active_turn[GROUP]=self.turn
        self.bot._latest_from_user[(GROUP,ALICE)]=self.turn
        self.bot._arrival_revisions[GROUP]=1
        self.bot._delivery_owners[GROUP]=(self.turn,BOT_ID,getattr(self.bot.client,'ws',None))
    async def asyncTearDown(self): await self.bot.close()

    def body(self):
        return ''.join(item['data']['text'] for message in self.bot.client.sent for item in message['message'] if item['type']=='text')

    async def send(self,text=LONG):
        return await self.bot._send(GROUP,text,'wide-source',ALICE,'at')

    async def test_long_delivery_concat_is_exact(self):
        self.assertTrue(await self.send())
        self.assertEqual(self.body(),LONG)
        self.assertGreater(len(self.bot.client.sent),3)

    async def test_each_part_has_correct_native_quote(self):
        await self.send()
        for message in self.bot.client.sent:
            self.assertEqual([x['data']['id'] for x in message['message'] if x['type']=='reply'],['wide-source'])

    async def test_code_at_cq_remain_plain_text(self):
        value=('@decorator\n[CQ:at,qq=54321]\n# header\n'*30)
        await self.send(value)
        self.assertEqual(self.body(),value)
        self.assertFalse(any(x['type']=='at' for m in self.bot.client.sent for x in m['message']))

    async def test_every_transport_part_bounded_without_total_cutoff(self):
        await self.send()
        pieces=[''.join(x['data']['text'] for x in m['message'] if x['type']=='text') for m in self.bot.client.sent]
        self.assertTrue(all(0<len(x)<=256 for x in pieces))
        self.assertEqual(''.join(pieces),LONG)

    async def test_receipt_contains_no_plaintext(self):
        await self.send()
        receipt=self.bot._wide_delivery_receipts[GROUP]
        self.assertTrue(receipt['complete'])
        self.assertEqual(receipt['sent_parts'],len(self.bot.client.sent))
        self.assertNotIn('text',receipt)
        self.assertNotIn('正文',json.dumps(receipt,ensure_ascii=False))

    async def test_delivery_context_clears_after_success(self):
        await self.send()
        self.assertNotIn(GROUP,self.bot._wide_delivery_context)

    async def test_disabled_output_preserves_single_transport_message(self):
        self.bot.cfg['output']['unlimited']=False
        self.assertTrue(await self.send())
        self.assertEqual(len(self.bot.client.sent),1)

    async def test_short_wide_output_still_single_message(self):
        self.assertTrue(await self.send('短消息'))
        self.assertEqual(len(self.bot.client.sent),1)

    async def test_blacklist_blocks_before_any_part(self):
        self.bot._user_blacklist.add(ALICE)
        self.assertFalse(await self.send())
        self.assertEqual(self.bot.client.sent,[])

    async def test_muted_blocks_before_any_part(self):
        self.bot._muted.add((GROUP,ALICE))
        self.assertFalse(await self.send())
        self.assertEqual(self.bot.client.sent,[])

    async def test_closed_blocks_before_any_part(self):
        self.bot._closing=True
        self.assertFalse(await self.send())
        self.assertEqual(self.bot.client.sent,[])

    async def test_stop_revision_after_first_part_cancels_rest(self):
        original=self.bot.client.send_action
        async def send(action,params=None,**kw):
            value=await original(action,params,**kw)
            self.bot._latest_from_user[(GROUP,ALICE)]=replace(self.turn,target_revision=1)
            return value
        self.bot.client.send_action=send
        self.assertFalse(await self.send())
        self.assertEqual(len(self.bot.client.sent),1)
        self.assertEqual(self.bot._wide_delivery_receipts[GROUP]['sent_parts'],1)

    async def test_socket_change_after_first_part_cancels_rest(self):
        original=self.bot.client.send_action
        async def send(action,params=None,**kw):
            value=await original(action,params,**kw)
            self.bot.client.ws=object()
            return value
        self.bot.client.send_action=send
        self.assertFalse(await self.send())
        self.assertEqual(len(self.bot.client.sent),1)

    async def test_fail_does_not_retry_physical_send(self):
        self.bot.client.fail=True
        self.assertFalse(await self.send())
        self.assertEqual(self.bot.client.sent,[])
        self.assertFalse(self.bot._wide_delivery_receipts[GROUP]['complete'])

    async def test_uncertain_ack_does_not_continue_or_resend(self):
        calls=[]
        async def uncertain(action,params=None,**kw): calls.append(params);return {}
        self.bot.client.send_action=uncertain
        self.assertFalse(await self.send())
        self.assertEqual(len(calls),1)
        self.assertEqual(self.bot._wide_delivery_receipts[GROUP]['status'],'uncertain')

    async def test_quota_is_consumed_per_physical_part(self):
        text='x'*700
        await self.send(text)
        self.assertEqual(len(self.bot.limiter._hits[('group',GROUP)]),3)
        self.assertEqual(len(self.bot.limiter._hits[('global',)]),3)

    async def test_quota_wait_does_not_bypass_or_retry_sent_chunk(self):
        self.bot.beh['per_group_per_minute']=1
        pauses=[]
        async def release_quota(delay):
            pauses.append(delay)
            for queue in self.bot.limiter._hits.values(): queue.clear()
            await asyncio.sleep(0)
        with patch.object(output_delivery,'_pause',release_quota):
            self.assertTrue(await self.send('x'*700))
        self.assertGreaterEqual(len(pauses),2)
        self.assertEqual(self.body(),'x'*700)
        self.assertEqual(len(self.bot.client.sent),3)

    async def test_waiting_for_quota_can_be_cancelled_without_send(self):
        self.bot.beh['per_group_per_minute']=1
        self.bot.limiter.allow_many(self.bot._second_reply_limits(GROUP))
        async def cancel(delay):
            self.bot._closing=True
            await asyncio.sleep(0)
        with patch.object(output_delivery,'_pause',cancel):
            self.assertFalse(await self.send())
        self.assertEqual(self.bot.client.sent,[])

    async def test_quota_wait_expires_truthfully_without_sending(self):
        self.bot.beh['per_group_per_minute']=1
        self.bot.limiter.allow_many(self.bot._second_reply_limits(GROUP))
        with patch.object(output_delivery,'QUOTA_WAIT_SECONDS',.001):
            self.assertFalse(await self.send())
        self.assertEqual(self.bot.client.sent,[])
        self.assertEqual(self.bot._wide_delivery_receipts[GROUP]['status'],'quota_wait_expired')

    async def test_physical_parts_commit_one_logical_history_entry(self):
        await self.send('x'*700)
        messages=list(self.bot.store._data.get(GROUP,()))
        self.assertEqual(len(messages),1)
        self.assertEqual(messages[0].text,'x'*700)
        self.assertEqual(self.bot.store.trailing_bot_count(GROUP),1)
        self.assertEqual(self.bot._dialogues[GROUP].bot_text,'x'*700)

    async def test_partial_failure_commits_only_acked_prefix_once(self):
        original=self.bot.client.send_action
        async def send(action,params=None,**kw):
            value=await original(action,params,**kw)
            self.bot._latest_from_user[(GROUP,ALICE)]=replace(self.turn,target_revision=1)
            return value
        self.bot.client.send_action=send
        self.assertFalse(await self.send('x'*700))
        messages=list(self.bot.store._data.get(GROUP,()))
        self.assertEqual(len(messages),1)
        self.assertEqual(messages[0].text,self.body())
        self.assertEqual(messages[0].text,'x'*256)

    async def test_uncertain_ack_does_not_commit_unconfirmed_text(self):
        async def uncertain(*a,**kw): return {}
        self.bot.client.send_action=uncertain
        self.assertFalse(await self.send('x'*700))
        self.assertEqual(len(self.bot.store._data.get(GROUP,())),0)

    async def test_source_receipt_binds_exact_turn_identity(self):
        await self.send('x'*700)
        self.assertEqual(self.bot._wide_delivery_receipts[GROUP]['source_turn'],(1,'wide-source'))

    async def test_legacy_planning_off_still_counts_every_part_quota(self):
        self.bot.beh['model_reply_planning']=False
        await self.send('x'*700)
        self.assertEqual(len(self.bot.limiter._hits[('group',GROUP)]),3)
        self.assertEqual(len(self.bot.limiter._hits[('global',)]),3)

    async def test_parts_respect_original_physical_min_gap(self):
        self.bot.beh['min_gap']=.015
        await self.send('x'*700)
        stamps=[message['sent_at'] for message in self.bot.client.sent]
        self.assertEqual(len(stamps),3)
        self.assertTrue(all(b-a>=.010 for a,b in zip(stamps,stamps[1:])))

    async def test_physical_chunks_do_not_start_retaliation_tracking(self):
        tracked=[]
        self.bot._duel_record_delivery=lambda *args:tracked.append(args)
        await self.send('x'*700)
        self.assertEqual(tracked,[])

    async def test_long_logical_followup_increments_turn_counter_only_once(self):
        self.bot._dialogues[GROUP]=DialogueState(ALICE,'原问题','原回答',time.monotonic(),followups=2)
        self.assertTrue(await self.bot._send(GROUP,'x'*700,'wide-source',ALICE,'followup'))
        self.assertEqual(self.bot._dialogues[GROUP].followups,3)
        self.assertEqual(self.bot.store.trailing_bot_count(GROUP),1)

    async def test_stop_during_delivery_gap_prevents_next_physical_send(self):
        async def stop_gap():
            self.bot._latest_from_user[(GROUP,ALICE)]=replace(self.turn,target_revision=1)
        self.bot._wait_delivery_gap=stop_gap
        self.assertFalse(await self.send('x'*700))
        self.assertEqual(self.bot.client.sent,[])

    async def test_permission_change_during_gap_prevents_send(self):
        async def revoke_gap(): self.bot._group_allowed=lambda group:False
        self.bot._wait_delivery_gap=revoke_gap
        self.assertFalse(await self.send('x'*700))
        self.assertEqual(self.bot.client.sent,[])

    async def test_actual_queue_long_reply_does_not_make_extra_model_burst(self):
        self.bot._active_turn.clear()
        self.bot._latest_from_user.clear()
        self.bot._delivery_owners.clear()
        event={'post_type':'message','message_type':'group','group_id':GROUP,'user_id':ALICE,
            'self_id':BOT_ID,'message_id':'wide-event','sender':{'nickname':'合成成员'},
            'message':[{'type':'at','data':{'qq':str(BOT_ID)}},{'type':'text','data':{'text':'请给完整代码'}}]}
        await self.bot.on_event(event)
        await asyncio.wait_for(self.bot.wait_idle(),2)
        self.assertEqual(self.bot.brain.calls,1)
        self.assertEqual(self.body(),LONG)
        self.assertEqual(self.bot.store.trailing_bot_count(GROUP),1)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'success':result.wasSuccessful(),'real_model_requests':0,'real_QQ_messages':0,'real_external_network_requests':0}
    if RESULT: RESULT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('RESULT_JSON',RESULT)
    raise SystemExit(0 if result.wasSuccessful() else 1)
