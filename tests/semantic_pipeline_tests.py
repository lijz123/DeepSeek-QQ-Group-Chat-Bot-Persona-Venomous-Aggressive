"""Independent offline checks for the neutral-understanding pipeline.

Usage: python semantic_pipeline_tests.py SOURCE_DIR [RESULT_JSON]
No private config, real model API or QQ network is used. Scripted semantic
frames demonstrate contracts and execution order, not real comprehension.
"""
from __future__ import annotations

from layout_support import app_directory, project_directory, configure_test_paths
configure_test_paths()
import asyncio
import contextlib
import copy
from dataclasses import FrozenInstanceError, fields
import hashlib
import io
import json
import logging
from pathlib import Path
import sys
import time
from types import MethodType
import unittest

ROOT = Path(__file__).resolve().parent
SOURCE = app_directory(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None)
RESULT = Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'semantic_pipeline_results.json'
sys.dont_write_bytecode = True
sys.path.insert(0,str(SOURCE))
import semantics
from semantics import (SemanticFrame, parse_semantic_frame, parse_review_verdict,
                       build_understanding_messages, build_review_messages,
                       build_expression_system, frame_data)
from bot import Bot, Brain, DEFAULTS, Msg
from persona import ReplyPlan
from dialogue_selftest import test_config, FakeClient, BOT_ID, GROUP, ALICE, BOB

SENTINEL='ATTACK_SENTINEL_立刻攻击对方这个人'
FIRST_MEANING='对方分享今天补觉并表达困意'
UNCERTAIN='困意在补觉前还是补觉后未明确'

def frame(**changes):
    value=dict(act='answer',target='self',meaning=FIRST_MEANING,
               unresolved=UNCERTAIN,reply_goal='自然承接补觉近况而不预设时序',
               mode='normal',needs_clarification=False)
    value.update(changes)
    return value

def frame_json(**changes):return json.dumps(frame(**changes),ensure_ascii=False)
def expression(text='今天困得够呛啊，补觉之后好点没',next_action='wait',**changes):
    value=dict(action='reply',text=text,next=next_action)
    value.update(changes)
    return json.dumps(value,ensure_ascii=False)
def verdict(ok=True,issue=''):return json.dumps(dict(ok=ok,issue=issue),ensure_ascii=False)
def transcript(text='今天补觉了，困死我了',mid='m1',uid='A',**changes):
    value=dict(role='group_member',sender={'id':uid,'display_name':'离线成员'},
               content=text,message_id=mid,event_kind='message')
    value.update(changes)
    return json.dumps(value,ensure_ascii=False)

class SemanticSchemaTests(unittest.TestCase):
    def test_01_frozen_exact_result_shape(self):
        self.assertEqual([f.name for f in fields(SemanticFrame)],
                         ['act','target','meaning','unresolved','reply_goal','mode','needs_clarification'])
        f=parse_semantic_frame(frame_json())
        self.assertIsInstance(f,SemanticFrame)
        with self.assertRaises((FrozenInstanceError,AttributeError)):f.mode='banter'

    def test_02_roundtrip_result_data(self):
        self.assertEqual(frame_data(parse_semantic_frame(frame_json())),frame())

    def test_03_all_declared_modes(self):
        for mode in ['normal','banter','repair','clarify','stop']:
            with self.subTest(mode=mode):self.assertEqual(parse_semantic_frame(frame_json(mode=mode)).mode,mode)

    def test_04_unknown_mode_or_labels_rejected(self):
        for changes in [dict(mode='attack_everyone'),dict(act='invented'),dict(target='everyone')]:
            self.assertIsNone(parse_semantic_frame(frame_json(**changes)))

    def test_05_missing_and_unknown_fields_rejected(self):
        for key in frame():
            data=frame();del data[key]
            with self.subTest(key=key):self.assertIsNone(parse_semantic_frame(json.dumps(data)))
        data=frame();data['reasoning']='not part of the contract'
        self.assertIsNone(parse_semantic_frame(json.dumps(data)))

    def test_06_boolean_is_not_coerced(self):
        for bad in [None,0,1,'false','true',[],{}]:
            self.assertIsNone(parse_semantic_frame(frame_json(needs_clarification=bad)))

    def test_07_semantic_text_types_are_strict(self):
        for key in ['meaning','unresolved','reply_goal']:
            for bad in [None,True,1,[],{}]:
                with self.subTest(key=key,bad=bad):self.assertIsNone(parse_semantic_frame(frame_json(**{key:bad})))

    def test_08_semantic_bounds(self):
        self.assertIsNotNone(parse_semantic_frame(frame_json(meaning='义'*160,unresolved='疑'*160,reply_goal='回'*120)))
        for key,size in [('meaning',161),('unresolved',161),('reply_goal',121)]:
            self.assertIsNone(parse_semantic_frame(frame_json(**{key:'长'*size})))

    def test_09_nonobjects_and_partial_json_rejected(self):
        for raw in ['',None,'[]','null','true','{"act":','explanation before '+frame_json()]:
            self.assertIsNone(parse_semantic_frame(raw))

    def test_10_verdict_shape_and_boolean_strictness(self):
        self.assertTrue(parse_review_verdict(verdict()).ok)
        self.assertFalse(parse_review_verdict(verdict(False,'未接住纠正')).ok)
        for value in [dict(ok='true',issue=''),dict(ok=True),dict(ok=True,issue='',reason='extra'),
                      dict(ok=False,issue=None),dict(ok=False,issue='长'*161)]:
            self.assertIsNone(parse_review_verdict(json.dumps(value)))

    def test_11_neutral_messages_have_raw_input_not_persona(self):
        raw=transcript('我说的是你记住了，不是我记性好')
        messages=build_understanding_messages(raw)
        self.assertEqual(messages[0]['role'],'system')
        self.assertIn(raw,'\n'.join(m['content'] for m in messages))
        self.assertNotIn(SENTINEL,str(messages))

    def test_12_review_is_separate_from_expression_style(self):
        f=parse_semantic_frame(frame_json())
        messages=build_review_messages(transcript(),f,'睡了个假觉',['上一句'])
        rendered='\n'.join(m['content'] for m in messages)
        self.assertIn('睡了个假觉',rendered)
        self.assertIn(FIRST_MEANING,rendered)
        self.assertNotIn(SENTINEL,rendered)

    def test_13_normal_repair_clarify_exclude_attack_extra(self):
        for mode in ['normal','repair','clarify']:
            f=parse_semantic_frame(frame_json(mode=mode))
            prompt=build_expression_system('离线机器人','sharp',SENTINEL,'温和会话',f)
            self.assertNotIn(SENTINEL,prompt)

    def test_14_same_input_meaning_is_not_rewritten_by_expression_builder(self):
        f=parse_semantic_frame(frame_json())
        before=frame_data(f)
        build_expression_system('离线机器人','sharp',SENTINEL,'',f)
        self.assertEqual(frame_data(f),before)

class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        cfg=copy.deepcopy(DEFAULTS['llm'])
        cfg.update(api_key='offline-fixture-not-a-real-key',semantic_pipeline=True,
                   semantic_review=True,semantic_temperature=.2,temperature=.8)
        persona=copy.deepcopy(DEFAULTS['persona']);persona.update(name='离线机器人',style='sharp',extra=SENTINEL)
        self.brain=Brain(cfg,copy.deepcopy(DEFAULTS['safety']),persona,120)
        self.calls=[];self.responses=[]
        async def fake_call(_self,messages,temperature,*,plan_mode=False,token_budget=0,**kwargs):
            self.calls.append(dict(messages=copy.deepcopy(messages),temperature=temperature,
                                   plan_mode=plan_mode,token_budget=token_budget,**kwargs))
            return self.responses.pop(0) if self.responses else None
        self.brain._call=MethodType(fake_call,self.brain)

    async def asyncTearDown(self):await self.brain.close()
    async def run_plan(self,responses,*,text=None,group=1,continuation=False,tone=''):
        self.responses=list(responses)
        return await self.brain.plan_reply(group,text or transcript(),'离线成员',True,
                                          tone=tone,continuation=continuation)

    async def test_15_one_reply_has_three_separate_calls(self):
        p=await self.run_plan([frame_json(),expression(),verdict()])
        self.assertEqual(p.action,'reply');self.assertEqual(len(self.calls),3)
        self.assertNotIn(SENTINEL,str(self.calls[0]['messages']))
        self.assertNotIn(SENTINEL,str(self.calls[2]['messages']))

    async def test_16_canonical_semantics_cannot_be_overridden_by_writer(self):
        malicious=dict(act='evaluate',target='bot',meaning='WRONG_MEANING',unresolved='')
        p=await self.run_plan([frame_json(),expression(anchor='WRONG_ANCHOR',understanding=malicious),verdict()])
        self.assertEqual((p.understanding.act,p.understanding.target),('answer','self'))
        self.assertEqual(p.understanding.meaning,FIRST_MEANING)
        self.assertEqual(p.understanding.unresolved,UNCERTAIN)
        self.assertEqual(p.anchor,frame()['reply_goal'][:80])

    async def test_17_neutral_result_reaches_writer_and_review(self):
        await self.run_plan([frame_json(),expression(),verdict()])
        for call in self.calls[1:]:
            self.assertIn(FIRST_MEANING,str(call['messages']))
            self.assertIn(UNCERTAIN,str(call['messages']))

    async def test_18_review_rejection_rewrites_once(self):
        issue='REVIEW_ISSUE_SENTINEL 未承接明确纠正'
        p=await self.run_plan([frame_json(mode='repair',act='clarify',unresolved=''),
                              expression('错误草稿'),verdict(False,issue),
                              expression('哦，你说的是我记住了'),verdict()])
        self.assertEqual(p.text,'哦，你说的是我记住了')
        self.assertEqual(len(self.calls),5)
        self.assertIn(issue,str(self.calls[3]['messages']))

    async def test_19_twice_rejected_never_sends_unreviewed_text(self):
        p=await self.run_plan([frame_json(),expression('错误草稿'),verdict(False,'偏题'),
                              expression('再次错误草稿'),verdict(False,'仍然偏题')])
        self.assertTrue(p is None or p.action!='reply')
        self.assertLessEqual(len(self.calls),5)

    async def test_20_invalid_understanding_fails_closed(self):
        p=await self.run_plan(['not-json','not-json','not-json'])
        self.assertTrue(p is None or p.action!='reply')
        self.assertNotIn(SENTINEL,str([c['messages'] for c in self.calls]))

    async def test_21_invalid_review_cannot_release_draft(self):
        p=await self.run_plan([frame_json(),expression('未复核草稿'),'not-json',
                              expression('另一条未复核草稿'),'not-json'])
        self.assertTrue(p is None or p.action!='reply')

    async def test_22_continuation_reuses_frame_not_another_understanding_call(self):
        await self.run_plan([frame_json(),expression(next_action='reply'),verdict()])
        p=await self.run_plan([expression('第二条新补充'),verdict()],continuation=True)
        self.assertEqual(p.text,'第二条新补充')
        self.assertEqual(len(self.calls),5)
        self.assertEqual(p.understanding.meaning,FIRST_MEANING)

    async def test_23_continuation_without_prior_frame_waits(self):
        p=await self.run_plan([],continuation=True)
        self.assertTrue(p is None or p.action!='reply')
        self.assertEqual(len(self.calls),0)

    async def test_24_continuation_cannot_reuse_other_message_frame(self):
        await self.run_plan([frame_json(),expression(next_action='reply'),verdict()])
        p=await self.run_plan([],text=transcript('另一句话','m2'),continuation=True)
        self.assertTrue(p is None or p.action!='reply')
        self.assertEqual(len(self.calls),3)

    async def test_25_continuation_cannot_reuse_other_group_frame(self):
        await self.run_plan([frame_json(),expression(next_action='reply'),verdict()])
        p=await self.run_plan([],group=2,continuation=True)
        self.assertTrue(p is None or p.action!='reply')
        self.assertEqual(len(self.calls),3)

    async def test_26_new_main_input_reanalyzes_explicit_correction(self):
        await self.run_plan([frame_json(),expression(),verdict()])
        new=frame_json(act='clarify',target='bot',meaning='明确纠正为机器人记住了前文',
                       unresolved='',reply_goal='承接对机器人的记忆评价',mode='repair')
        p=await self.run_plan([new,expression('哦，说我记住了啊'),verdict()],
                              text=transcript('我说的是你记住了，不是我记性好','m2'))
        self.assertEqual(len(self.calls),6)
        self.assertEqual(p.understanding.target,'bot')
        self.assertIn('机器人记住了',p.understanding.meaning)

    async def test_27_question_mark_frame_uses_repair_expression(self):
        f=frame_json(act='clarify',target='bot_reply',meaning='对刚才回答表示困惑',
                     unresolved='',reply_goal='重新检查自己上一条是否答偏',mode='repair')
        p=await self.run_plan([f,expression('刚才那句没接准，我重新说'),verdict()],text=transcript('?'))
        self.assertEqual(p.understanding.target,'bot_reply')
        self.assertNotIn(SENTINEL,str(self.calls[1]['messages']))

    async def test_28_third_party_report_does_not_become_reporter_self_claim(self):
        f=frame_json(act='other',target='third_party',meaning='转述第三人对机器人的挑衅',
                     unresolved='未明确第三人的具体身份',reply_goal='回应转述内容而不偷换成传话者自述',mode='banter')
        p=await self.run_plan([f,expression('让他自己来，我又不收传话费'),verdict()],text=transcript('他说要挑战你'))
        self.assertEqual(p.understanding.target,'third_party')

    async def test_29_explicit_causal_and_after_relations_stay_distinct(self):
        for i,(meaning,raw) in enumerate([('因为困才去补觉','因为困所以今天去补觉'),
                                        ('补完觉之后仍然困','补完觉还是很困')]):
            f=frame_json(meaning=meaning,unresolved='',reply_goal='回应明确的状态')
            p=await self.run_plan([f,expression('这句按明确关系接住'),verdict()],
                                  text=transcript(raw,'m'+str(i+1)),group=i+1)
            self.assertEqual(p.understanding.meaning,meaning)

    async def test_30_no_idle_poll_after_wait(self):
        p=await self.run_plan([frame_json(),expression(),verdict()])
        before=len(self.calls);await asyncio.sleep(.01)
        self.assertEqual(p.next_action,'wait');self.assertEqual(len(self.calls),before)

    async def test_31_semantic_metadata_is_not_reply_text(self):
        p=await self.run_plan([frame_json(),expression('只发这句正文'),verdict()])
        self.assertEqual(p.text,'只发这句正文')
        for marker in [FIRST_MEANING,UNCERTAIN,'understanding','needs_clarification']:
            self.assertNotIn(marker,p.text)

    async def test_32_retry_does_not_raise_expression_temperature(self):
        await self.run_plan([frame_json(),expression('错误草稿'),verdict(False,'答偏'),
                             expression('换一种说法'),verdict()])
        self.assertLessEqual(self.calls[0]['temperature'],.3)
        self.assertEqual(self.calls[1]['temperature'],self.calls[3]['temperature'])
        self.assertLessEqual(self.calls[1]['temperature'],.8)

    async def test_33_nonplanning_reply_also_uses_pipeline(self):
        self.responses=[frame_json(),expression('普通接口也先理解'),verdict()]
        result=await self.brain.reply(1,transcript(),'离线成员',True)
        self.assertEqual(result,'普通接口也先理解')
        self.assertEqual(len(self.calls),3)

    async def test_46_local_unknown_relation_guard_precedes_permissive_reviewer(self):
        bad='补了觉还困，你这是睡了个假觉'
        good='今天困得够呛啊，补觉之后好点没'
        biased_meaning='补完觉以后仍然困'
        drafts=[bad,good];reviewed=[]
        async def routed_call(_self,messages,temperature,*,plan_mode=False,token_budget=0,**kwargs):
            self.calls.append(dict(messages=copy.deepcopy(messages),temperature=temperature,
                                   plan_mode=plan_mode,token_budget=token_budget,**kwargs))
            if token_budget==640:
                # The first-stage model may contradict its own uncertainty.
                # Rejecting the draft must quarantine this biased summary too.
                return frame_json(meaning=biased_meaning,reply_goal='调侃补完觉还困的反差')
            if token_budget==512:
                return expression(drafts.pop(0))
            if token_budget==256:
                # This reviewer would approve any draft. The local guard must
                # prevent a contradictory draft from reaching it or the user.
                reviewed.append(json.loads(messages[-1]['content'])['draft'])
                return verdict(True,'')
            raise AssertionError('Unexpected offline phase budget')
        self.brain._call=MethodType(routed_call,self.brain)
        p=await self.brain.plan_reply(1,transcript(),'离线成员',True)
        self.assertEqual(p.text,good)
        self.assertNotIn(bad,reviewed)
        self.assertIn(good,reviewed)
        self.assertEqual(drafts,[])
        self.assertLessEqual(len(self.calls),5)
        self.assertEqual(p.understanding.meaning,'今天补觉了，困死我了')
        self.assertEqual(p.understanding.unresolved,UNCERTAIN)
        rewrite=[c for c in self.calls if c['token_budget']==512][1]
        rewrite_data=json.JSONDecoder().raw_decode(rewrite['messages'][-1]['content'])[0]
        # Expression sees an independent literal clause, while canonical
        # metadata and the independent reviewer retain the complete original.
        self.assertEqual(rewrite_data['semantic_frame']['meaning'],'困死我了')
        review_calls=[c for c in self.calls if c['token_budget']==256]
        self.assertIn('今天补觉了，困死我了',review_calls[0]['messages'][-1]['content'])
        self.assertEqual(p.next_action,'wait')
        self.assertNotEqual(rewrite_data['semantic_frame']['reply_goal'],'调侃补完觉还困的反差')

    async def test_49_report_role_guard_preserves_third_party_even_with_permissive_review(self):
        data=frame(target='third_party',meaning='转述第三人想揍机器人',
                   unresolved='第三人的具体身份没有明确',reply_goal='回应第三人的挑衅',mode='banter')
        canonical=parse_semantic_frame(json.dumps(data,ensure_ascii=False))
        helper=getattr(semantics,'reported_role_issue',None)
        self.assertTrue(callable(helper))
        bad='我没实体让你揍'
        good='让他自己来说，我又不收传话费'
        self.assertTrue(helper(canonical,bad))
        self.assertFalse(helper(canonical,good))
        direct=parse_semantic_frame(frame_json(target='self',meaning='群友直接挑战机器人',unresolved=''))
        self.assertFalse(helper(direct,bad))
        drafts=[bad,good];reviewed=[]
        async def routed_call(_self,messages,temperature,*,plan_mode=False,token_budget=0,**kwargs):
            self.calls.append(dict(messages=copy.deepcopy(messages),temperature=temperature,
                                   plan_mode=plan_mode,token_budget=token_budget,**kwargs))
            if token_budget==640:return json.dumps(data,ensure_ascii=False)
            if token_budget==512:return expression(drafts.pop(0))
            if token_budget==256:
                reviewed.append(json.loads(messages[-1]['content'])['draft'])
                return verdict(True,'')
            raise AssertionError('Unexpected offline phase budget')
        self.brain._call=MethodType(routed_call,self.brain)
        p=await self.brain.plan_reply(1,transcript('他说想揍你'),'离线成员',True)
        self.assertEqual(p.text,good)
        self.assertNotIn(bad,reviewed)
        self.assertEqual((p.understanding.target,p.understanding.meaning),('third_party',data['meaning']))

    async def test_50_quote_reply_and_mentions_are_part_of_continuation_identity(self):
        original=dict(quote='原引用',reply_to={'message_id':'r1','sender_id':'B'},mentions=['BOT'])
        changes=[dict(quote='不同引用'),dict(reply_to={'message_id':'r2','sender_id':'C'}),
                 dict(mentions=['OTHER'])]
        for group,change in enumerate(changes,10):
            before=transcript('我指的是这件事',**original)
            await self.run_plan([frame_json(),expression(next_action='reply'),verdict()],text=before,group=group)
            count=len(self.calls)
            changed=dict(original);changed.update(change)
            p=await self.run_plan([],text=transcript('我指的是这件事',**changed),group=group,continuation=True)
            self.assertTrue(p is None or p.action!='reply')
            self.assertEqual(len(self.calls),count)

    async def test_51_needs_clarification_overrides_banter_mode(self):
        data=frame_json(mode='banter',needs_clarification=True,
                        unresolved='关键对象尚未说明',reply_goal='先问清关键对象')
        p=await self.run_plan([data,expression('这句具体指哪件事？'),verdict()])
        self.assertEqual(p.action,'reply')
        writer=self.calls[1]['messages']
        self.assertNotIn(SENTINEL,writer[0]['content'])
        self.assertEqual(json.loads(writer[-1]['content'])['semantic_frame']['mode'],'clarify')


class UnconfirmedRelationTests(unittest.TestCase):
    def guard(self,data,draft):
        helper=getattr(semantics,'unconfirmed_relation_issue',None)
        self.assertTrue(callable(helper),'Neutral semantics must expose the local relation guard')
        return helper(parse_semantic_frame(json.dumps(data,ensure_ascii=False)),draft)

    def test_41_ambiguous_sleep_does_not_license_asserted_after_state(self):
        self.assertTrue(self.guard(frame(),'补了觉还困，你这是睡了个假觉'))

    def test_42_same_guard_generalizes_to_repair_and_error_relation(self):
        data=frame(meaning='今天处理过问题，并抱怨报错烦人',
                   unresolved='报错出现在处理之前还是之后未明确',
                   reply_goal='回应处理问题的近况')
        self.assertTrue(self.guard(data,'修过了还报错，你这修了个寂寞'))

    def test_43_explicit_after_state_does_not_trigger_unknown_relation_guard(self):
        data=frame(meaning='补完觉以后仍然困',unresolved='',reply_goal='承接已明确的补后困意')
        self.assertFalse(self.guard(data,'补了觉还困，今天真是困得够呛'))

    def test_44_unknown_identity_is_not_mistaken_for_unknown_time(self):
        data=frame(target='third_party',meaning='转述第三人的话',
                   unresolved='第三人的具体身份没有明确',reply_goal='回应转述内容')
        self.assertFalse(self.guard(data,'他说完还不肯自己来，那就先等着'))

    def test_45_genuine_questions_do_not_assert_the_unknown_relation(self):
        self.assertFalse(self.guard(frame(),'补觉以后还困吗？'))
        data=frame(meaning='处理过问题并表达对报错的烦恼',
                   unresolved='报错发生在处理前还是之后没有说明')
        self.assertFalse(self.guard(data,'处理之后还报错吗？'))

class PokeContextTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        cfg=test_config();cfg['behavior'].update(reply_poke=True,poke_context_window=45)
        self.bot=Bot(cfg);await self.bot.brain.close();self.calls=[]
        self.next_plan=ReplyPlan('wait')
        owner=self
        class Capture:
            async def plan_reply(self,*args,**kwargs):
                owner.calls.append(dict(transcript=args[1],**kwargs));return owner.next_plan
            async def close(self):pass
        self.bot.brain=Capture();self.bot.client=FakeClient()
    async def asyncTearDown(self):await self.bot.close()
    def member(self,text='还没回答的原话',uid=ALICE,mid='m1',age=1):
        self.bot.store.add(GROUP,Msg('离线成员',text,False,time.time()-age,sender_id=str(uid),message_id=mid))
    async def poke(self):
        await self.bot._on_notice(dict(notice_type='notify',sub_type='poke',group_id=GROUP,user_id=ALICE,target_id=BOT_ID))
        await asyncio.wait_for(self.bot.wait_idle(),5)
        return json.loads(self.calls[-1]['transcript'].splitlines()[-1])

    async def test_34_poke_is_explicit_event_not_human_words(self):
        row=await self.poke()
        self.assertEqual(row['event_kind'],'poke')

    async def test_35_recent_same_sender_unanswered_text_is_linked(self):
        self.member();row=await self.poke()
        self.assertEqual(row['related_message']['content'],'还没回答的原话')
        self.assertEqual(str(row['related_message']['sender']['id']),str(ALICE))

    async def test_36_another_members_message_is_not_attached(self):
        self.member();self.member('另一个人的话',uid=BOB,mid='m2')
        self.assertNotIn('related_message',await self.poke())

    async def test_37_expired_old_topic_is_not_attached(self):
        self.member(age=46)
        self.assertNotIn('related_message',await self.poke())

    async def test_38_already_answered_text_is_not_attached(self):
        self.member()
        self.bot.store.add(GROUP,Msg('离线机器人','已经回复',True,time.time(),sender_id=str(BOT_ID),
                                    message_id='b1',reply_to_id='m1',reply_to_sender_id=str(ALICE)))
        self.assertNotIn('related_message',await self.poke())

    async def test_39_identityless_text_is_not_guessed_as_target(self):
        self.member(mid='')
        self.assertNotIn('related_message',await self.poke())

    async def test_40_preceding_text_remains_history(self):
        self.member();await self.poke()
        rows=[json.loads(line) for line in self.calls[-1]['transcript'].splitlines()]
        self.assertTrue(any(r.get('content')=='还没回答的原话' for r in rows[:-1]))

    async def test_47_successful_poke_reply_acknowledges_related_original_once(self):
        self.member()
        self.next_plan=ReplyPlan('reply','这就接上刚才那句话','wait')
        first=await self.poke()
        self.assertEqual(first['related_message']['message_id'],'m1')
        self.assertEqual(len(self.bot.client.sent),1)
        delivered=[m for m in self.bot.store._data[GROUP] if m.is_bot][-1]
        self.assertEqual(delivered.reply_to_id,'m1')
        self.assertEqual(self.bot._dialogues[GROUP].user_text,'还没回答的原话')
        self.next_plan=ReplyPlan('wait')
        self.assertNotIn('related_message',await self.poke())

    async def test_48_failed_poke_send_does_not_acknowledge_related_original(self):
        self.member()
        self.bot.client.fail=True
        self.next_plan=ReplyPlan('reply','发送失败的离线草稿','wait')
        await self.poke()
        self.assertEqual(len(self.bot.client.sent),0)
        self.assertFalse(any(m.is_bot for m in self.bot.store._data[GROUP]))
        self.bot.client.fail=False
        self.next_plan=ReplyPlan('wait')
        again=await self.poke()
        self.assertEqual(again['related_message']['message_id'],'m1')

def main():
    logging.disable(logging.CRITICAL)
    buffer=io.StringIO()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                             for cls in [SemanticSchemaTests,PipelineTests,UnconfirmedRelationTests,PokeContextTests])
    with contextlib.redirect_stdout(buffer),contextlib.redirect_stderr(buffer):
        result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(suite)
    data=dict(kind='OFFLINE_NEUTRAL_SEMANTIC_PIPELINE_CONTRACT_NOT_REAL_ACCURACY',
              source=str(SOURCE),successful=result.wasSuccessful(),tests_run=result.testsRun,
              failures=len(result.failures),errors=len(result.errors),real_model_calls=0,qq_network_sends=0,
              output=buffer.getvalue())
    RESULT.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(buffer.getvalue(),end='')
    print('RESULT_JSON '+str(RESULT))
    raise SystemExit(0 if result.wasSuccessful() else 1)
if __name__=='__main__':main()
