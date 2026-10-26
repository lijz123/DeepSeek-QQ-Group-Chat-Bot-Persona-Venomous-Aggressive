"""Offline chat-100/task-complete regression tests using synthetic sources only.

This file never loads private config, contacts a model/search service, or sends
real QQ messages. Old tests are not modified; the old wide flag is tested too.
"""
from __future__ import annotations

from layout_support import app_directory, project_directory, configure_test_paths
configure_test_paths()

import asyncio
import copy
from dataclasses import fields, FrozenInstanceError, replace
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

from bot import Bot, Brain
from dialogue_selftest import test_config, FakeClient, BOT_ID, GROUP, ALICE, BOB
from output_policy import daily_limit, daily_prompt, limit_daily_text
from semantics import (SemanticFrame, SearchSemanticFrame, OutputSemanticFrame,
    parse_semantic_frame, frame_data, build_understanding_messages,
    build_expression_system)
from persona import ReplyPlan
from owner_chat import (build_owner_chat_messages, split_owner_response,
    is_owner_daily_task, parse_owner_chat_plan)
from web_search import SearchOutcome
from duel_tests import ReceiptBrain, Client as DuelClient


KEY = 'synthetic-local-key-not-a-real-credential'
PUBLIC = 'https://docs.python.org/3/library/asyncio.html'
CODE = '\r\n```c\r\n#include "main.h"\r\n' + (
    'void led(void) {\r\n\tHAL_GPIO_WritePin(GPIOA, GPIO_PIN_2, GPIO_PIN_SET);\r\n}\r\n' * 180) + '```\r\n'
ARTICLE = '# 示例文章\n\n' + '这是实际交付的长文章内容，不是聊天承诺。\n' * 180
CHAT = '今天天气不错，我们可以正常聊一会儿。' * 20
ROAST = '你这句话倒是挺会重复，认真想一想再说吧。' * 20
CORE = dict(act='ask', target='topic', meaning='当前成员请求聊天', unresolved='',
    reply_goal='直接回答当前意思', mode='normal', needs_clarification=False)


def cfg(*, daily=100, owner=False):
    value = test_config()
    value['output'] = {'unlimited': True, 'max_tokens': 393216, 'chunk_chars': 2000,
        'daily_max_chars': daily}
    value['llm'].update(provider='deepseek', base_url='https://api.deepseek.com/v1',
        api_key=KEY, model='deepseek-chat', semantic_pipeline=True,
        semantic_review_policy='all', max_tokens=100, planning_max_tokens=192)
    value['voice']['enabled'] = False
    value['search']['enabled'] = False
    value['owner_chat'].update(enabled=owner, user_ids=[ALICE] if owner else [])
    value['safety']['blocked_output'] = []
    value['behavior'].update(merge_delay=0, merge_at_delay=0, merge_max_wait=0,
        min_gap=0, direct_cooldown=0, direct_user_cooldown=0, concurrent_reply_window=0,
        quote_mode='always', max_chain=3, model_reply_planning=True,
        model_burst_max_messages=3, second_reply_delay_min=0,
        second_reply_delay_max=0, per_group_per_minute=1000, global_per_minute=1000,
        followup_rate=1.0, random_rate=0.0, question_reply=False)
    return value


def source(text='肥鱼，晚上好', uid=ALICE, mid='daily-source', background=()):
    row = dict(role='group_member', sender={'id': str(uid), 'display_name': '合成成员'+str(uid)},
        message_id=mid, content=text)
    return '\n'.join(json.dumps(x, ensure_ascii=False) for x in (*background, row))


def frame(kind='chat', *, search=False, **changes):
    value = {**CORE, 'response_kind': kind, **changes}
    if search:
        value.update(need_search=True, search_query='Python asyncio 官方文档')
    return value


class PolicyTests(unittest.TestCase):
    def test_daily_cap_is_explicit_not_old_wide_default(self):
        self.assertEqual(daily_limit({}), 0)
        self.assertEqual(daily_limit({'output': {'unlimited': True}}), 0)
        self.assertEqual(daily_limit(cfg()), 100)

    def test_invalid_daily_cap_does_not_become_truthy(self):
        for value in (True, False, '100', None, [], {}):
            with self.subTest(value=value):
                self.assertEqual(daily_limit({'output': {'daily_max_chars': value}}), 0)

    def test_exact_hundred_is_not_changed(self):
        self.assertEqual(limit_daily_text('字'*100, 100), '字'*100)

    def test_daily_overflow_including_ellipsis_stays_at_hundred(self):
        result = limit_daily_text('字'*101, 100)
        self.assertLessEqual(len(result), 100)
        self.assertGreater(len(result), 0)

    def test_remaining_zero_is_empty_not_unlimited(self):
        self.assertEqual(limit_daily_text('不能再发', 0), '')

    def test_small_remaining_never_adds_an_extra_ellipsis(self):
        self.assertLessEqual(len(limit_daily_text('abcdef', 1)), 1)
        self.assertLessEqual(len(limit_daily_text('abcdef', 2)), 2)

    def test_daily_cap_counts_punctuation_spaces_and_ascii(self):
        self.assertLessEqual(len(limit_daily_text('abc, 空格。'*30, 100)), 100)

    def test_daily_does_not_drop_second_line_when_already_short(self):
        self.assertEqual(limit_daily_text('第一行\n第二行', 100), '第一行\n第二行')

    def test_daily_prompt_includes_limit(self):
        self.assertIn('100', daily_prompt(100))
        self.assertIn('30', daily_prompt(30))

    def test_original_frame_dataclass_still_has_exact_seven_fields(self):
        self.assertEqual([x.name for x in fields(SemanticFrame)], list(CORE))

    def test_old_seven_field_parser_and_serializer_still_work(self):
        value = parse_semantic_frame(json.dumps(CORE))
        self.assertEqual(frame_data(value), CORE)

    def test_old_search_frame_still_roundtrips(self):
        value = {**CORE, 'need_search': True, 'search_query': 'Python 文档'}
        parsed = parse_semantic_frame(json.dumps(value))
        self.assertIsInstance(parsed, SearchSemanticFrame)
        self.assertEqual(frame_data(parsed), value)

    def test_new_eight_field_response_kind_roundtrips(self):
        parsed = parse_semantic_frame(json.dumps(frame('task')))
        self.assertIsInstance(parsed, OutputSemanticFrame)
        self.assertEqual(frame_data(parsed)['response_kind'], 'task')

    def test_new_ten_field_search_kind_roundtrips(self):
        value = frame('task', search=True)
        self.assertEqual(frame_data(parse_semantic_frame(json.dumps(value))), value)

    def test_required_kind_does_not_silently_accept_old_frame(self):
        self.assertIsNone(parse_semantic_frame(json.dumps(CORE), require_response_kind=True))

    def test_required_kind_accepts_valid_new_frame(self):
        self.assertIsNotNone(parse_semantic_frame(json.dumps(frame()), require_response_kind=True))

    def test_response_kind_unknown_value_rejected(self):
        self.assertIsNone(parse_semantic_frame(json.dumps(frame('unlimited'))))

    def test_response_kind_must_be_string(self):
        for kind in (True, False, 1, None, [], {}):
            with self.subTest(kind=kind):
                self.assertIsNone(parse_semantic_frame(json.dumps(frame(kind))))

    def test_extended_frame_is_frozen_and_replace_preserves_kind(self):
        parsed = parse_semantic_frame(json.dumps(frame('task')))
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            parsed.response_kind = 'chat'
        self.assertEqual(replace(parsed, reply_goal='继续代码').response_kind, 'task')

    def test_banter_cannot_label_itself_unlimited_task(self):
        parsed = parse_semantic_frame(json.dumps(frame('task', mode='banter', act='rebut', target='bot_reply')))
        self.assertEqual(parsed.response_kind, 'chat')

    def test_understanding_enabled_requests_kind_without_extra_request(self):
        messages = build_understanding_messages(source(), runtime=cfg())
        self.assertEqual(len(messages), 2)
        self.assertIn('response_kind', messages[0]['content'])
        self.assertIn('辱骂', messages[0]['content'])

    def test_understanding_daily_off_keeps_old_protocol(self):
        config = cfg(daily=0)
        self.assertNotIn('response_kind', build_understanding_messages(source(), runtime=config)[0]['content'])

    def test_chat_expression_has_hundred_limit_not_unbounded_prompt(self):
        parsed = parse_semantic_frame(json.dumps(frame()))
        text = build_expression_system('大肥鱼', 'sharp', '', '', parsed, runtime=cfg())
        self.assertIn('100', text)
        self.assertNotIn('本轮没有应用层字符数', text)

    def test_chat_continuation_prompt_uses_only_remaining_budget(self):
        parsed = parse_semantic_frame(json.dumps(frame()))
        text = build_expression_system('大肥鱼', 'sharp', '', '', parsed, runtime=cfg(), remaining_chars=25)
        self.assertIn('25', text)

    def test_task_expression_keeps_complete_output_policy(self):
        parsed = parse_semantic_frame(json.dumps(frame('task')))
        text = build_expression_system('大肥鱼', 'sharp', '', '', parsed, runtime=cfg())
        self.assertIn('完整内容输出', text)

    def test_owner_daily_mode_requests_response_kind_same_generation(self):
        messages = build_owner_chat_messages('大肥鱼', 'sharp', source(), '合成成员', '说句话', runtime=cfg())
        self.assertIn('response_kind', messages[0]['content'])


class OwnerProtocolTests(unittest.TestCase):
    def test_owner_kind_is_stripped_without_changing_complete_code(self):
        value = json.dumps({'action':'reply','text':CODE,'next':'wait','response_kind':'task'})
        stripped, kind = split_owner_response(value)
        self.assertEqual(kind, 'task')
        self.assertEqual(parse_owner_chat_plan(stripped, max_chars=0).text, CODE)

    def test_owner_missing_kind_rejected_in_daily_protocol(self):
        self.assertIsNone(split_owner_response('{"action":"reply","text":"a","next":"wait"}'))

    def test_owner_invalid_kind_rejected(self):
        self.assertIsNone(split_owner_response('{"action":"reply","text":"a","next":"wait","response_kind":"unlimited"}'))

    def test_owner_duplicate_kind_rejected(self):
        self.assertIsNone(split_owner_response('{"action":"reply","text":"a","next":"wait","response_kind":"chat","response_kind":"task"}'))

    def test_owner_unknown_fields_rejected(self):
        self.assertIsNone(split_owner_response('{"action":"reply","text":"a","next":"wait","response_kind":"task","sudo":true}'))

    def test_roast_primary_purpose_stays_daily_even_if_called_an_essay(self):
        self.assertTrue(is_owner_daily_task('帮我写个小作文骂他，火力全开'))

    def test_quoted_insult_word_explanation_not_automatically_a_roast(self):
        self.assertFalse(is_owner_daily_task('解释“骂人”这个词是什么意思'))

    def test_legitimate_code_delivery_not_daily_roast(self):
        self.assertFalse(is_owner_daily_task('写一个Python程序检测文本中是否出现“骂人”两个字'))


class FakeResponse:
    status = 200
    def __init__(self, content):
        self.body = json.dumps({'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]})
    async def text(self): return self.body
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass


class ModelSession:
    def __init__(self, *, kind='chat', text=CHAT, mode='normal', next_action='wait', search=False):
        self.kind, self.text, self.mode, self.next_action, self.search = kind, text, mode, next_action, search
        self.requests = []
        self.generation_count = 0
        self.sequence = None

    def post(self, url, **kwargs):
        payload = kwargs['json']
        self.requests.append(copy.deepcopy(payload))
        system = payload['messages'][0]['content']
        if '中性语义解析器' in system:
            value = {**CORE, 'mode': self.mode}
            if self.kind == 'task': value.update(meaning='当前成员请求实际交付完整内容', reply_goal='提供当前所需的完整内容')
            if self.mode == 'banter': value.update(act='rebut', target='bot_reply')
            if 'response_kind' in system: value['response_kind'] = self.kind
            if 'need_search' in system: value.update(need_search=self.search, search_query='Python asyncio 官方文档' if self.search else '')
            content = json.dumps(value, ensure_ascii=False)
        elif '资料依据复核器' in system or '中性的群聊回复语义复核器' in system or '对象归属复核' in system:
            content = '{"ok":true,"issue":""}'
        elif '公开资料回答' in system:
            self.generation_count += 1
            content = json.dumps({'answer': self.text, 'sources': [1]}, ensure_ascii=False)
        else:
            text = self.sequence[min(self.generation_count, len(self.sequence)-1)] if self.sequence else self.text
            self.generation_count += 1
            value = {'action': 'reply', 'text': text, 'next': self.next_action}
            if 'response_kind' in system and ('聊天管理员明确' in system or '理解与交互底座' in system): value['response_kind'] = self.kind
            content = json.dumps(value, ensure_ascii=False)
        return FakeResponse(content)

    async def close(self): pass


class BrainTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = cfg()
        self.brain = Brain(self.config['llm'], self.config['safety'], self.config['persona'], 120, runtime=self.config)
        self.session = ModelSession()
        self.brain._session = self.session
    async def asyncTearDown(self): await self.brain.close()

    async def plan(self, text='肥鱼，晚上好'):
        return await self.brain.plan_reply(GROUP, source(text), '合成成员', True)

    async def test_actual_semantic_daily_reply_never_exceeds_hundred(self):
        result = await self.plan()
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result.text), 100)

    async def test_task_relation_rewrite_is_capped_as_daily_clarification(self):
        self.session.kind = 'task'
        self.session.text = '这是一段需要澄清未确认关系的回复。' * 20
        with patch('bot.unconfirmed_relation_issue', side_effect=['未确认的因果关系', '']):
            result = await self.plan('肥鱼，帮我分析这段代码的问题')
        self.assertIsNotNone(result)
        self.assertEqual(result.action, 'reply')
        self.assertLessEqual(len(result.text), 100)
        self.assertEqual(self.brain.output_kind_for(GROUP, ALICE, 'daily-source'), 'chat')
        self.assertEqual([x['max_tokens'] for x in self.session.requests], [640, 393216, 512, 256])
        self.assertIn('剩余最多100字符', self.session.requests[-2]['messages'][0]['content'])
        self.assertNotIn('本轮没有应用层字符数', self.session.requests[-2]['messages'][0]['content'])

    async def test_daily_request_keeps_original_model_call_count(self):
        await self.plan()
        self.assertEqual(len(self.session.requests), 3)

    async def test_daily_generation_does_not_use_enormous_task_ceiling(self):
        await self.plan()
        budgets = [x['max_tokens'] for x in self.session.requests]
        self.assertEqual(budgets[0], 640)
        self.assertEqual(budgets[-1], 256)
        self.assertLessEqual(budgets[1], 1024)

    async def test_just_mentioning_code_is_still_daily(self):
        result = await self.plan('肥鱼，这代码也太长了吧，聊聊呗')
        self.assertLessEqual(len(result.text), 100)

    async def test_coding_capability_question_is_daily(self):
        result = await self.plan('肥鱼，你会不会写代码？')
        self.assertLessEqual(len(result.text), 100)

    async def test_actual_code_task_preserves_more_than_ten_thousand_characters(self):
        self.assertGreater(len(CODE), 10000)
        self.session.kind, self.session.text = 'task', CODE
        result = await self.plan('肥鱼，写完整LED代码')
        self.assertEqual(result.text, CODE)

    async def test_code_task_still_uses_verified_provider_ceiling(self):
        self.session.kind, self.session.text = 'task', CODE
        await self.plan('肥鱼，写完整LED代码')
        self.assertEqual([x['max_tokens'] for x in self.session.requests], [640, 393216, 256])

    async def test_article_task_keeps_complete_text(self):
        self.session.kind, self.session.text = 'task', ARTICLE
        self.assertEqual((await self.plan('肥鱼，写一篇完整教程')).text, ARTICLE)

    async def test_contextual_send_it_can_remain_task_without_code_keyword(self):
        self.session.kind, self.session.text = 'task', CODE
        prior = [dict(role='group_member', sender={'id':str(ALICE),'display_name':'合成成员'}, message_id='original-code-request', content='请给完整LED代码'),
            dict(role='assistant', sender={'id':str(BOT_ID),'display_name':'大肥鱼'}, message_id='prior-code-answer', content='需要完整主函数和GPIO初始化吗？', reply_to={'sender_id':str(ALICE),'message_id':'original-code-request'})]
        context = source('发我，我需要', background=prior)
        result = await self.brain.plan_reply(GROUP, context, '合成成员', True)
        self.assertEqual(result.text, CODE)

    async def test_banter_kind_task_is_forced_back_to_chat(self):
        self.session.mode, self.session.kind, self.session.text = 'banter', 'task', ROAST
        result = await self.plan('肥鱼，你就这点本事？')
        self.assertLessEqual(len(result.text), 100)

    async def test_explicit_stop_cannot_become_content_task(self):
        self.session.mode, self.session.kind = 'stop', 'task'
        result = await self.plan('别说了')
        self.assertNotEqual(result.action, 'reply')

    async def test_daily_zero_preserves_previous_complete_mode(self):
        self.config['output']['daily_max_chars'] = 0
        result = await self.plan()
        self.assertEqual(result.text, CHAT)
        self.assertEqual(self.session.requests[1]['max_tokens'], 393216)

    async def test_owner_daily_does_not_get_an_automatic_length_exemption(self):
        result = await self.brain.owner_chat_reply(GROUP, source(), '合成成员', True, instruction='随便说句话')
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result.text), 100)

    async def test_owner_task_keeps_complete_code(self):
        self.session.kind, self.session.text = 'task', CODE
        result = await self.brain.owner_chat_reply(GROUP, source('写完整代码'), '合成成员', True, instruction='写完整代码')
        self.assertEqual(result.text, CODE)

    async def test_owner_roast_cannot_choose_task_to_escape_limit(self):
        self.session.kind, self.session.text = 'task', ROAST
        result = await self.brain.owner_chat_reply(GROUP, source('帮我骂一骂'), '合成成员', True, instruction='帮我骂一骂')
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result.text), 100)

    async def test_search_material_delivery_keeps_complete_answer_and_sources(self):
        self.config['search']['enabled'] = True
        self.session.search, self.session.kind, self.session.text = True, 'task', ARTICLE
        evidence = SearchOutcome('ok', 'Python asyncio 官方文档', ({'title':'Python 官方文档','url':PUBLIC,'snippet':ARTICLE,'date':''},))
        class Search:
            async def search(self, *args, **kwargs): return evidence
        self.brain._official_search_client = Search()
        result = await self.plan('肥鱼，搜索并整理Python asyncio的完整教程')
        self.assertIn(ARTICLE, result.text)
        self.assertIn(PUBLIC, result.text)

    async def test_daily_multiple_generated_plans_share_remaining_budget(self):
        self.session.next_action = 'reply'
        self.session.sequence = ['甲'*60, '乙'*60, '丙'*60]
        first = await self.plan()
        second = await self.brain.plan_reply(GROUP, source(), '合成成员', True, continuation=True)
        self.assertLessEqual(len(first.text) + len(second.text), 100)
        third = await self.brain.plan_reply(GROUP, source(), '合成成员', True, continuation=True)
        self.assertTrue(third is None or third.action != 'reply' or not third.text)

    async def test_owner_chat_continuation_cannot_switch_to_task_to_escape_budget(self):
        self.session.next_action, self.session.sequence = 'reply', ['甲'*60, '乙'*60, '丙'*60]
        first = await self.brain.owner_chat_reply(GROUP, source('说句话'), '合成成员', True, instruction='说句话')
        self.session.kind = 'task'
        second = await self.brain.owner_chat_reply(GROUP, source('说句话'), '合成成员', True, instruction='说句话', continuation=True)
        self.assertLessEqual(len(first.text)+len(second.text), 100)
        third = await self.brain.owner_chat_reply(GROUP, source('说句话'), '合成成员', True, instruction='说句话', continuation=True)
        self.assertTrue(third is None or third.action != 'reply' or not third.text)

    async def test_legacy_daily_path_is_capped_with_no_extra_classifier_call(self):
        self.brain.cfg['semantic_pipeline'] = False
        result = await self.plan('肥鱼，正常聊聊')
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result.text), 100)
        self.assertEqual(len(self.session.requests), 1)

    async def test_legacy_explicit_content_task_stays_complete(self):
        self.brain.cfg['semantic_pipeline'] = False
        self.session.kind, self.session.text = 'task', CODE
        result = await self.plan('肥鱼，写完整LED程序')
        self.assertEqual(result.text, CODE)

    async def test_legacy_daily_off_retains_old_wide_protocol(self):
        self.brain.cfg['semantic_pipeline'] = False
        self.config['output']['daily_max_chars'] = 0
        self.session.text = CODE
        result = await self.plan('肥鱼，写完整LED程序')
        self.assertEqual(result.text, CODE)

    async def test_valid_short_idle_followup_stays_available_under_daily_cap(self):
        text = '刚才不是挺能说吗，现在人呢？'
        class Session:
            def post(self, *args, **kwargs):
                return FakeResponse(json.dumps({'action':'reply','text':text}, ensure_ascii=False))
            async def close(self): pass
        self.brain._session = Session()
        answer = await self.brain.duel_followup(GROUP, ALICE, '你就这点本事？', '先接住这句再说')
        self.assertEqual(answer, text)
        self.assertLessEqual(len(answer), 100)

    async def test_pushback_reply_is_capped_as_daily_chat(self):
        self.session.text = ROAST
        answer = await self.brain.roast_pushback_reply(GROUP, source('帮我骂人'), '合成成员', True, instruction='帮我骂人')
        self.assertIsNotNone(answer)
        self.assertLessEqual(len(answer.text), 100)
        self.assertLessEqual(self.session.requests[0]['max_tokens'], 1024)

    async def test_pushback_has_no_task_length_escape(self):
        self.session.kind, self.session.text = 'task', ROAST
        answer = await self.brain.roast_pushback_reply(GROUP, source('写小作文帮我骂人'), '合成成员', True, instruction='写小作文帮我骂人')
        self.assertIsNotNone(answer)
        self.assertLessEqual(len(answer.text), 100)


class RecordingBrain:
    def __init__(self, text=CHAT, next_action='wait'):
        self.text, self.next_action, self.calls = text, next_action, 0
    async def plan_reply(self, *args, **kwargs):
        self.calls += 1
        return ReplyPlan('reply', self.text, self.next_action)
    async def close(self): pass


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = cfg()
        self.bot = Bot(self.config)
        self.bot.client = FakeClient()
        self.session = ModelSession()
        self.bot.brain._session = self.session
        self.ids = 500
    async def asyncTearDown(self): await self.bot.close()

    async def push(self, text='肥鱼，聊聊', uid=ALICE):
        self.ids += 1
        await self.bot.on_event(dict(post_type='message', message_type='group', group_id=GROUP,
            user_id=uid, self_id=BOT_ID, message_id=self.ids,
            sender={'nickname':'合成成员'+str(uid)},
            message=[{'type':'at','data':{'qq':str(BOT_ID)}}, {'type':'text','data':{'text':text}}]))
        return str(self.ids)

    async def idle(self): await asyncio.wait_for(self.bot.wait_idle(), 4)

    def body(self):
        return ''.join(seg['data']['text'] for msg in self.bot.client.sent for seg in msg['message'] if seg['type']=='text')

    async def test_real_queue_daily_output_is_bounded(self):
        await self.push()
        await self.idle()
        self.assertGreater(len(self.bot.client.sent), 0)
        self.assertLessEqual(len(self.body()), 100)

    async def test_prefix_at_separator_is_included_in_actual_hundred_char_budget(self):
        self.config['behavior']['prefix_at'] = True
        await self.push()
        await self.idle()
        self.assertEqual(len(self.bot.client.sent), 1)
        segments = self.bot.client.sent[0]['message']
        self.assertEqual([x['data']['qq'] for x in segments if x['type']=='at'], [str(ALICE)])
        self.assertEqual(len(self.body()), 100)

    async def test_real_queue_banter_is_daily_even_if_model_says_task(self):
        self.session.mode, self.session.kind, self.session.text = 'banter', 'task', ROAST
        await self.push('肥鱼，攻击力就这？')
        await self.idle()
        self.assertLessEqual(len(self.body()), 100)

    async def test_real_queue_code_task_is_lossless_and_not_hundred_truncated(self):
        self.session.kind, self.session.text = 'task', CODE
        await self.push('肥鱼，写完整LED代码')
        await self.idle()
        self.assertEqual(self.body(), CODE)
        self.assertGreater(len(self.bot.client.sent), 1)

    async def test_actual_task_transport_quote_binds_its_source(self):
        self.session.kind, self.session.text = 'task', CODE
        mid = await self.push('肥鱼，写完整LED代码')
        await self.idle()
        for message in self.bot.client.sent:
            self.assertEqual([x['data']['id'] for x in message['message'] if x['type']=='reply'], [mid])

    async def test_generated_burst_cannot_multiply_hundred_limit(self):
        self.session.next_action, self.session.sequence = 'reply', ['甲'*60, '乙'*60, '丙'*60]
        await self.push()
        await self.idle()
        self.assertLessEqual(len(self.body()), 100)
        self.assertLessEqual(len(self.bot.client.sent), 2)

    async def test_unknown_custom_brain_defaults_to_daily_final_gate(self):
        self.bot.brain = RecordingBrain('字'*400)
        await self.push()
        await self.idle()
        self.assertLessEqual(len(self.body()), 100)

    async def test_custom_brain_burst_still_cannot_bypass_final_gate(self):
        self.bot.brain = RecordingBrain('字'*70, 'reply')
        await self.push()
        await self.idle()
        self.assertLessEqual(len(self.body()), 100)

    async def test_each_new_user_turn_gets_its_own_daily_budget(self):
        await self.push()
        await self.idle()
        first_count = len(self.bot.client.sent)
        self.session.text = '换了新话题，我就接着回应这一轮。' * 20
        await self.push('肥鱼，换个话题正常聊')
        await self.idle()
        self.assertGreater(len(self.bot.client.sent), first_count)
        second = ''.join(x['data']['text'] for m in self.bot.client.sent[first_count:] for x in m['message'] if x['type']=='text')
        self.assertLessEqual(len(second), 100)
        self.assertGreater(len(second), 0)

    async def test_second_member_does_not_reuse_first_members_remaining_budget(self):
        self.config['behavior']['concurrent_reply_window'] = .03
        self.config['behavior']['merge_max_wait'] = .04
        self.session.sequence = ['甲'*150, '乙'*150]
        first = await self.push(uid=ALICE)
        second = await self.push(uid=BOB)
        await self.idle()
        self.assertEqual(len(self.bot.client.sent), 2)
        self.assertEqual([[x['data']['id'] for x in m['message'] if x['type']=='reply'] for m in self.bot.client.sent], [[first], [second]])
        for m in self.bot.client.sent:
            self.assertLessEqual(sum(len(x['data']['text']) for x in m['message'] if x['type']=='text'), 100)

    async def test_response_kind_metadata_is_not_posted_as_group_text(self):
        await self.push()
        await self.idle()
        self.assertNotIn('response_kind', self.body())
        self.assertNotIn('"action"', self.body())

    async def test_task_profile_is_not_inherited_by_next_daily_turn(self):
        self.session.kind, self.session.text = 'task', CODE
        await self.push('肥鱼，写完整LED程序')
        await self.idle()
        start = len(self.bot.client.sent)
        self.session.kind, self.session.text = 'chat', CHAT
        await self.push('肥鱼，聊点日常吧')
        await self.idle()
        tail = ''.join(x['data']['text'] for m in self.bot.client.sent[start:] for x in m['message'] if x['type']=='text')
        self.assertGreater(len(tail), 0)
        self.assertLessEqual(len(tail), 100)

    async def test_blocked_member_still_gets_nothing(self):
        self.bot._user_blacklist.add(ALICE)
        await self.push()
        await self.idle()
        self.assertEqual(self.bot.client.sent, [])

    async def test_daily_off_keeps_previous_wide_queue_behavior(self):
        self.config['output']['daily_max_chars'] = 0
        await self.push()
        await self.idle()
        self.assertEqual(self.body(), CHAT)


class IdleBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = cfg()
        self.config['duel'] = dict(enabled=True, idle_seconds=60, expiry_seconds=180,
            max_groups=256, poll_interval=1)
        self.bot = Bot(self.config)
        self.brain = ReceiptBrain(self.bot)
        self.bot.brain = self.brain
        self.bot.client = DuelClient()

    async def asyncTearDown(self): await self.bot.close()

    async def arm(self, primary):
        self.brain.plans = [ReplyPlan('reply', primary, 'wait')]
        await self.bot.on_event(dict(post_type='message', message_type='group', group_id=GROUP,
            user_id=ALICE, self_id=BOT_ID, message_id='idle-budget-source',
            sender={'nickname':'合成成员'}, message=[{'type':'at','data':{'qq':str(BOT_ID)}},
                {'type':'text','data':{'text':'我是你爹'}}]))
        await asyncio.wait_for(self.bot.wait_idle(), 3)
        return self.bot._duels.current(GROUP)

    def texts(self):
        return [''.join(x['data']['text'] for x in m['message'] if x['type']=='text') for m in self.bot.client.sent]

    async def test_primary_hundred_does_not_generate_extra_idle_text(self):
        session = await self.arm('甲'*100)
        if session is not None:
            await self.bot._duel_tick(session.due_at)
        self.assertEqual(len(self.bot.client.sent), 1)
        self.assertEqual(len(''.join(self.texts())), 100)
        self.assertEqual(self.brain.followup_calls, [])

    async def test_primary_ninety_nine_does_not_send_only_mention_separator(self):
        session = await self.arm('甲'*99)
        self.assertIsNotNone(session)
        await self.bot._duel_tick(session.due_at)
        self.assertEqual(len(self.bot.client.sent), 1)
        self.assertEqual(len(''.join(self.texts())), 99)
        self.assertEqual(self.brain.followup_calls, [])

    async def test_primary_sixty_leaves_at_most_forty_for_idle_reply(self):
        session = await self.arm('甲'*60)
        self.assertIsNotNone(session)
        self.brain.followup_text = '追'*80
        await self.bot._duel_tick(session.due_at)
        self.assertEqual(len(self.bot.client.sent), 2)
        self.assertLessEqual(len(self.texts()[1]), 40)
        self.assertLessEqual(len(''.join(self.texts())), 100)
        self.assertEqual(len(self.brain.followup_calls), 1)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        success=result.wasSuccessful(), real_model_requests=0, real_QQ_messages=0,
        real_external_network_requests=0)
    if RESULT: RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('RESULT_JSON', RESULT)
    raise SystemExit(0 if result.wasSuccessful() else 1)
