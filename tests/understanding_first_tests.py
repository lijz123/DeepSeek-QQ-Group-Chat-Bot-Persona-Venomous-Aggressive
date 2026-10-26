"""Independent OFFLINE prompt/schema/continuation checks, not semantic accuracy.

python understanding_first_tests.py SOURCE_DIR [RESULT_JSON]
Uses synthetic messages, model plans and QQ client. Does not read config.json,
contact a provider, or send a real QQ message. Human contrast labels are input
fixtures, never counted as the model understanding them correctly.
"""
from __future__ import annotations

from layout_support import app_directory, project_directory, configure_test_paths
configure_test_paths()
import asyncio
import contextlib
from dataclasses import FrozenInstanceError, fields
import hashlib
import io
import json
import logging
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parent
SOURCE = app_directory(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None)
RESULT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else ROOT/'understanding_first_results.json'
sys.dont_write_bytecode = True
sys.path.insert(0,str(SOURCE))

import persona
from persona import (ReplyPlan, TurnUnderstanding, parse_reply_plan,
    build_system_prompt, build_user_prompt, build_turn_focus,
    POLICY_UNDERSTANDING, POLICY_PRIORITY, CORE_IDENTITY, EXAMPLES,
    POLICY_ENGAGEMENT, POLICY_COHERENCE, POLICY_CORRECTION)
from bot import Bot, Brain, DEFAULTS
from dialogue_selftest import (BOT_ID,GROUP,ALICE,BOB,DialogueTests,
                               FakeClient,FakePlanBrain,test_config)

ACTS = ('greet','ask','answer','evaluate','rebut','clarify','joke','close','other','uncertain')
TARGETS = ('bot_reply','bot','self','third_party','topic','unclear')
U = TurnUnderstanding('evaluate','bot_reply')
V = TurnUnderstanding('evaluate','self')

def raw(frame=U, **kw):
    value = dict(action='reply',text='离线测试正文',next='wait',anchor='回应当前评价')
    if frame is not None:
        value['understanding'] = dict(act=frame.act,target=frame.target)
    value.update(kw)
    return json.dumps(value,ensure_ascii=False)

def member(text,mid='m1',sender='A',**kw):
    return dict(role='group_member',sender={'id':sender,'display_name':'测试昵称'},
                content=text,message_id=mid,**kw)

def assistant(text,mid='b1',source='m1',recipient='A'):
    return dict(role='assistant',sender={'id':'BOT'},content=text,message_id=mid,
                reply_to={'message_id':source,'sender_id':recipient})

def transcript(rows):
    return '\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)


class SchemaTests(unittest.TestCase):
    def test_01_frame_is_immutable_compact_not_free_reasoning(self):
        self.assertEqual([f.name for f in fields(TurnUnderstanding)],['act','target','meaning','unresolved'])
        with self.assertRaises((FrozenInstanceError,AttributeError)):
            U.target='self'

    def test_02_parse_valid_frame_preserves_text(self):
        p=parse_reply_plan(raw(),strict=True)
        self.assertEqual(p.understanding,U)
        self.assertEqual(p.text,'离线测试正文')

    def test_03_every_documented_enum_value_supported(self):
        for act in ACTS:
            for target in TARGETS:
                with self.subTest(act=act,target=target):
                    frame=TurnUnderstanding(act,target)
                    self.assertEqual(parse_reply_plan(raw(frame),strict=True).understanding,frame)

    def test_04_old_three_and_four_argument_replyplans(self):
        self.assertIsNone(ReplyPlan('reply','旧正文','reply').understanding)
        self.assertIsNone(ReplyPlan('reply','旧正文','reply','原有焦点').understanding)
        self.assertEqual(ReplyPlan('reply','新正文','wait','焦点',U).understanding,U)

    def test_05_old_json_without_frame_is_compatible(self):
        p=parse_reply_plan(raw(None),strict=True)
        self.assertIsNone(p.understanding)
        self.assertEqual((p.text,p.next_action),('离线测试正文','wait'))

    def test_06_wrong_container_types_are_rejected(self):
        for value in (None,[],True,1,'evaluate',{}):
            with self.subTest(value=value):
                self.assertIsNone(parse_reply_plan(raw(understanding=value),strict=True))

    def test_07_missing_extra_or_free_reasoning_keys_rejected(self):
        for value in ({'act':'evaluate'},{'target':'bot_reply'},
                      {'act':'evaluate','target':'bot_reply','reasoning':'不该接受自由推理'},
                      {'act':'evaluate','target':'bot_reply','sender_id':'A'}):
            with self.subTest(value=value):
                self.assertIsNone(parse_reply_plan(raw(understanding=value),strict=True))

    def test_08_bad_enum_types_or_unknown_values_rejected(self):
        for bad in (None,True,1,[],{},'EVALUATE','other-person','x'*200):
            for key,other in (('act',{'target':'bot_reply'}),('target',{'act':'evaluate'})):
                with self.subTest(bad=bad,key=key):
                    self.assertIsNone(parse_reply_plan(raw(understanding={key:bad,**other}),strict=True))

    def test_09_wait_and_stop_drop_understanding_and_anchor(self):
        for action in ('wait','stop'):
            p=parse_reply_plan(raw(action=action,text='',next=action),strict=True)
            self.assertEqual(p.action,action)
            self.assertIsNone(p.understanding)
            self.assertEqual(p.anchor,'')

    def test_10_safety_fallback_drops_frame_and_next(self):
        p=parse_reply_plan(raw(text='包含离线禁用词',next='reply'),strict=True,
                           blocked=['离线禁用词'],fallback='离线安全回落')
        self.assertEqual((p.text,p.next_action,p.anchor),('离线安全回落','wait',''))
        self.assertIsNone(p.understanding)

    def test_11_fenced_json_and_plain_compatibility(self):
        self.assertEqual(parse_reply_plan('```json\n'+raw()+'\n```',strict=True).understanding,U)
        p=parse_reply_plan('旧纯文本正文')
        self.assertIsNone(p.understanding)
        self.assertEqual(p.text,'旧纯文本正文')

    def test_12_frame_does_not_create_a_reply_from_control_word(self):
        for control in ('wait','stop'):
            self.assertEqual(parse_reply_plan(control,strict=True).action,control)
            self.assertIsNone(parse_reply_plan(control,strict=True).understanding)

    def test_13_uncertain_is_not_forced_into_claimed_certainty(self):
        uncertain=TurnUnderstanding('uncertain','unclear')
        p=parse_reply_plan(raw(uncertain,text='你是在说刚才哪一句？'),strict=True)
        self.assertEqual((p.text,p.understanding),('你是在说刚才哪一句？',uncertain))


class PromptTests(unittest.TestCase):
    def test_14_shared_understanding_and_priority_in_every_style(self):
        for style in ('mild','sharp','savage'):
            prompt=build_system_prompt('测试机器人',style,plan_mode=True)
            self.assertIn(POLICY_UNDERSTANDING,prompt)
            self.assertIn(POLICY_PRIORITY,prompt)
            self.assertIn(CORE_IDENTITY[style],prompt)
            self.assertIn(EXAMPLES[style],prompt)

    def test_15_understanding_precedes_strategy_and_style(self):
        prompt=build_system_prompt('测试机器人','sharp',plan_mode=True)
        self.assertLess(prompt.index(POLICY_UNDERSTANDING),prompt.index(POLICY_ENGAGEMENT))
        self.assertLess(prompt.index(POLICY_ENGAGEMENT),prompt.index(CORE_IDENTITY['sharp']))

    def test_16_unconditional_negative_strategy_removed(self):
        self.assertNotIn('别人问你，先否定再反问',POLICY_ENGAGEMENT)
        self.assertNotIn('不先肯定',CORE_IDENTITY['sharp'])

    def test_17_tone_overrides_style_not_foundation(self):
        prompt=build_system_prompt('测试机器人',tone='TEST_FRIENDLY_TONE',plan_mode=True)
        self.assertIn('TEST_FRIENDLY_TONE',prompt)
        self.assertNotIn('本条临时语气 · 优先级高于第一、二层',prompt)
        self.assertTrue('不覆盖' in prompt and '理解' in prompt and '协议' in prompt)
        self.assertIn(POLICY_PRIORITY,prompt)

    def test_18_coherence_and_correction_not_lost(self):
        for plan in (False,True):
            prompt=build_system_prompt('测试机器人',plan_mode=plan)
            self.assertIn(POLICY_COHERENCE,prompt)
            self.assertIn(POLICY_CORRECTION,prompt)

    def test_19_planning_prompt_requests_private_act_target_fields(self):
        prompt=build_system_prompt('测试机器人',plan_mode=True)
        for key in ('understanding','act','target','bot_reply','uncertain','unclear'):
            self.assertIn(key,prompt)

    def test_20_public_legacy_prompt_still_asks_for_text(self):
        rows=[member('普通问题')]
        prompt=build_user_prompt(transcript(rows),'测试昵称',True,plan_mode=False)
        self.assertIn('只输出',prompt)
        self.assertIn('普通问题',prompt)

    def test_21_identity_and_boundaries_not_removed_by_extra_or_tone(self):
        prompt=build_system_prompt('测试机器人','sharp','TEST_EXTRA','TEST_TONE',plan_mode=True)
        for value in ('测试机器人','TEST_EXTRA','TEST_TONE',POLICY_UNDERSTANDING,POLICY_PRIORITY):
            self.assertIn(value,prompt)
        self.assertTrue('安全' in prompt or '边界' in prompt)

    def test_22_default_rate_delay_and_maximum_preserved(self):
        beh=DEFAULTS['behavior']
        self.assertEqual(beh['per_group_per_minute'],18)
        self.assertEqual(beh['second_reply_delay_min'],.8)
        self.assertEqual(beh['second_reply_delay_max'],.8)
        self.assertEqual(beh['model_burst_max_messages'],3)


CONTRAST_FIXTURES = [
    ('challenge_bot','除了这句你还能整点别的活吗','就这','evaluate','bot_reply'),
    ('literal_menu','还要加别的菜吗','就这','answer','topic'),
    ('literal_selection','你只买这一件吗','就这','answer','topic'),
    ('game_weapon','这把武器的伤害面板是多少','攻击力就这','evaluate','topic'),
    ('explicit_clarification','你是在夸自己记性好吗','我说的是你有记忆了','clarify','bot'),
]


class ContrastContextTests(unittest.TestCase):
    def test_23_same_phrase_contexts_preserved_not_hard_rewritten(self):
        for label,prior,text,act,target in CONTRAST_FIXTURES:
            with self.subTest(label=label):
                rows=[member('前一条问题'),assistant(prior),member(text,'m2')]
                focus=json.loads(build_turn_focus(transcript(rows)).splitlines()[1])
                prompt=build_user_prompt(transcript(rows),'测试昵称',False,plan_mode=True)
                self.assertEqual(focus['current']['content'],text)
                self.assertEqual(focus['last_exchange'][-1]['content'],prior)
                self.assertIn(text,prompt)
                self.assertIn(prior,prompt)
                # These frames are human fixtures, not inferred by this test.
                self.assertEqual(parse_reply_plan(raw(TurnUnderstanding(act,target)),strict=True).understanding,
                                 TurnUnderstanding(act,target))

    def test_24_explicit_third_party_quote_keeps_its_author(self):
        rows=[member('攻击力就这','q1','B'),
              assistant('另一段无关回复','b1','m0','A'),
              member('他说的就这是在嫌谁','m2','A',reply_to={'message_id':'q1','sender_id':'B'})]
        data=json.loads(build_turn_focus(transcript(rows)).splitlines()[1])
        self.assertEqual(data['referenced_message']['sender_id'],'B')
        self.assertEqual(data['referenced_message']['content'],'攻击力就这')
        self.assertEqual(data['current']['sender_id'],'A')

    def test_25_unknown_reference_not_filled_with_other_member(self):
        rows=[member('我说就这','q1','B'),
              member('就这','m2','A',reply_to={'message_id':'missing','sender_id':'B'})]
        data=json.loads(build_turn_focus(transcript(rows)).splitlines()[1])
        self.assertNotIn('referenced_message',data)
        self.assertNotIn('last_exchange',data)

    def test_26_names_are_not_semantic_content(self):
        row=member('正常正文')
        row['sender']['display_name']='就这攻击力低'
        data=json.loads(build_turn_focus(transcript([row])).splitlines()[1])
        self.assertEqual(data['current']['content'],'正常正文')

    def test_27_fresh_current_clarification_not_mutated_by_history(self):
        rows=[member('就这'),assistant('就这就没了？'),member('我是说你的攻击力','m2')]
        data=json.loads(build_turn_focus(transcript(rows)).splitlines()[1])
        self.assertEqual(data['current']['content'],'我是说你的攻击力')
        self.assertIn('我是说你的攻击力',build_user_prompt(transcript(rows),'测试昵称',True,plan_mode=True))


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    event,push,idle=DialogueTests.event,DialogueTests.push,DialogueTests.idle

    async def asyncSetUp(self):
        cfg=test_config()
        cfg['behavior'].update(model_reply_planning=True,model_burst_max_messages=3,
            max_chain=3,combo_replies=[],second_reply_delay_min=.001,second_reply_delay_max=.001,
            followup_window=120,followup_max_turns=6)
        self.bot=Bot(cfg)
        await self.bot.brain.close()
        self.brain=FakePlanBrain([]);self.client=FakeClient()
        self.bot.brain,self.bot.client=self.brain,self.client
        self.msg_id=100

    async def asyncTearDown(self):
        await self.bot.close()

    async def say(self,text='就这',**kw):
        await self.push(text,at=kw.pop('at',[BOT_ID]),**kw)
        await self.idle()

    def plans(self,items):
        self.brain.plans=[ReplyPlan('reply',f'离线正文{i+1}',n,a,f) for i,(n,a,f) in enumerate(items)]

    def public(self):
        return [seg['data']['text'] for msg in self.client.sent for seg in msg['message'] if seg['type']=='text']

    async def test_28_same_frame_allows_two_without_extra_judge_call(self):
        self.plans([('reply','共同落点',U),('wait','共同落点',U)])
        await self.say()
        self.assertEqual(len(self.client.sent),2)
        self.assertEqual(len(self.brain.calls),2)

    async def test_29_same_frame_allows_three(self):
        self.plans([('reply','共同落点',U),('reply','共同落点',U),('wait','共同落点',U)])
        await self.say()
        self.assertEqual(len(self.client.sent),3)
        self.assertEqual(len(self.brain.calls),3)

    async def test_30_changed_target_stops_before_public_send(self):
        self.plans([('reply','共同落点',U),('wait','共同落点',V)])
        await self.say()
        self.assertEqual(len(self.brain.calls),2)
        self.assertEqual(len(self.client.sent),1)

    async def test_31_changed_act_stops_before_public_send(self):
        self.plans([('reply','共同落点',U),('wait','共同落点',TurnUnderstanding('answer','bot_reply'))])
        await self.say()
        self.assertEqual(len(self.client.sent),1)

    async def test_32_missing_established_frame_stops(self):
        self.plans([('reply','共同落点',U),('wait','共同落点',None)])
        await self.say()
        self.assertEqual(len(self.client.sent),1)

    async def test_33_legacy_absent_frame_remains_compatible(self):
        self.plans([('reply','共同落点',None),('wait','共同落点',None)])
        await self.say()
        self.assertEqual(len(self.client.sent),2)

    async def test_34_late_frame_establishes_lock(self):
        self.plans([('reply','共同落点',None),('reply','共同落点',U),('wait','共同落点',V)])
        await self.say()
        self.assertEqual(len(self.brain.calls),3)
        self.assertEqual(len(self.client.sent),2)

    async def test_35_late_frame_then_missing_stops(self):
        self.plans([('reply','共同落点',None),('reply','共同落点',U),('wait','共同落点',None)])
        await self.say()
        self.assertEqual(len(self.client.sent),2)

    async def test_36_late_frame_same_next_allows_three(self):
        self.plans([('reply','共同落点',None),('reply','共同落点',U),('wait','共同落点',U)])
        await self.say()
        self.assertEqual(len(self.client.sent),3)

    async def test_37_frame_does_not_bypass_anchor_guard(self):
        self.plans([('reply','原落点',U),('wait','别的落点',U)])
        await self.say()
        self.assertEqual(len(self.client.sent),1)

    async def test_38_metadata_never_enters_public_text(self):
        self.plans([('reply','共同落点',U),('wait','共同落点',U)])
        await self.say()
        self.assertEqual(self.public(),['离线正文1','离线正文2'])
        for word in ('understanding','evaluate','bot_reply','共同落点'):
            self.assertNotIn(word,json.dumps(self.client.sent,ensure_ascii=False))

    async def test_39_frozen_frame_passed_to_continuation_prompt(self):
        self.plans([('reply','共同落点',U),('wait','共同落点',U)])
        await self.say()
        self.assertEqual(len(self.brain.calls),2)
        focus=self.brain.calls[1]['focus_note']
        self.assertIn('evaluate',focus)
        self.assertIn('bot_reply',focus)

    async def test_40_new_user_message_can_update_understanding(self):
        self.plans([('wait','第一轮',V)])
        await self.say()
        self.plans([('wait','第二轮',TurnUnderstanding('clarify','bot'))])
        await self.say('我说的是你的攻击力')
        self.assertEqual(len(self.client.sent),2)

    async def test_41_different_member_does_not_inherit_frame(self):
        self.plans([('wait','第一轮',U)])
        await self.say()
        self.plans([('wait','第二轮',TurnUnderstanding('ask','topic'))])
        await self.say('另一位成员的问题',user=BOB)
        self.assertEqual(len(self.client.sent),2)

    async def test_42_different_group_does_not_inherit_frame(self):
        self.plans([('wait','第一轮',U)])
        await self.say()
        self.plans([('wait','第二轮',TurnUnderstanding('ask','topic'))])
        await self.say('另一个群的问题',group=GROUP+1)
        self.assertEqual(len(self.client.sent),2)

    async def test_43_friendly_session_kept_with_new_frame(self):
        self.plans([('wait','问候',TurnUnderstanding('greet','bot'))])
        await self.say('晚上好啊大肥鱼')
        self.plans([('reply','游戏',TurnUnderstanding('answer','topic')),
                    ('wait','游戏',TurnUnderstanding('answer','topic'))])
        await self.say('没呢，还在玩三角洲')
        self.assertEqual(len(self.client.sent),3)
        self.assertEqual(self.bot._session_tone(GROUP,ALICE),'friendly')
        for call in self.brain.calls[-2:]:
            self.assertIn('当前会话语气：温和',call['tone'])

    async def test_44_failed_first_send_never_generates_continuation(self):
        self.client.fail=True
        self.plans([('reply','共同落点',U),('wait','共同落点',U)])
        await self.say()
        self.assertEqual(len(self.client.sent),0)
        self.assertEqual(len(self.brain.calls),1)

    async def test_45_activity_still_cancels_pending_frame_continuation(self):
        self.plans([('reply','共同落点',U),('wait','共同落点',U)])
        self.brain.plan_release=asyncio.Event()
        await self.push('就这',at=[BOT_ID])
        await asyncio.wait_for(self.brain.plan_started.wait(),1)
        await self.push('另一位成员普通插话',user=BOB)
        await self.idle()
        self.assertEqual(len(self.client.sent),1)

    async def test_46_model_wait_still_sends_nothing(self):
        self.brain.plans=[ReplyPlan('wait')]
        await self.say()
        self.assertEqual(self.client.sent,[])
        self.assertEqual(len(self.brain.calls),1)


class RealBrainOfflineTests(unittest.IsolatedAsyncioTestCase):
    async def test_47_valid_frame_uses_one_existing_generation_request(self):
        cfg=test_config();cfg['llm']['api_key']='offline-placeholder-not-a-real-key'
        brain=Brain(cfg['llm'],cfg['safety'],cfg['persona'],120)
        class Session:
            async def close(self):pass
        brain._session=Session()
        brain._call=AsyncMock(return_value=raw(next='reply'))
        try:
            rows=[member('前一条问题'),assistant('除了这句还能整别的活吗'),member('就这','m2')]
            p=await brain.plan_reply(GROUP,transcript(rows),'测试昵称',True)
            self.assertEqual(p.understanding,U)
            self.assertEqual(brain._call.await_count,1)
            messages=brain._call.await_args.args[0]
            self.assertIn(POLICY_UNDERSTANDING,messages[0]['content'])
            self.assertIn('就这',messages[1]['content'])
        finally:await brain.close()

    async def test_48_legacy_valid_json_does_not_add_classifier_call(self):
        cfg=test_config();cfg['llm']['api_key']='offline-placeholder-not-a-real-key'
        brain=Brain(cfg['llm'],cfg['safety'],cfg['persona'],120)
        class Session:
            async def close(self):pass
        brain._session=Session()
        brain._call=AsyncMock(return_value=raw(None))
        try:
            p=await brain.plan_reply(GROUP,transcript([member('普通问题')]),'测试昵称',True)
            self.assertIsNone(p.understanding)
            self.assertEqual(brain._call.await_count,1)
        finally:await brain.close()


def run():
    logging.disable(logging.CRITICAL)
    output,stdout,stderr=io.StringIO(),io.StringIO(),io.StringIO()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c)
        for c in (SchemaTests,PromptTests,ContrastContextTests,RuntimeTests,RealBrainOfflineTests))
    with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
        result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    ledger=dict(kind='OFFLINE_STRUCTURAL_PROTOCOL_CONTROL_TESTS_NOT_REAL_MODEL_SEMANTIC_ACCURACY',
        source=str(SOURCE),real_model_calls=0,qq_network_sends=0,
        source_hashes={n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest()
                       for n in ('persona.py','bot.py','dialogue.py')},
        contrast_fixture_labels_are_human_inputs_not_model_predictions=True,
        contrast_fixtures=CONTRAST_FIXTURES,
        tests_run=result.testsRun,successful=result.wasSuccessful(),
        failures=[dict(test=str(test),traceback=trace) for test,trace in result.failures],
        errors=[dict(test=str(test),traceback=trace) for test,trace in result.errors],
        literal_stdout=stdout.getvalue(),literal_stderr=stderr.getvalue(),
        literal_test_output=output.getvalue(),exit_status=0 if result.wasSuccessful() else 1)
    RESULT.write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
    print(output.getvalue(),end='')
    print(json.dumps({k:ledger[k] for k in ('kind','tests_run','successful','exit_status')},ensure_ascii=False))
    return ledger['exit_status']

if __name__=='__main__':raise SystemExit(run())
