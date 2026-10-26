"""Independent offline contracts for selective semantic review.

Usage: python adaptive_review_tests.py SOURCE_DIR [RESULT_JSON]
All model replies are synthetic. No real configuration, profiles, network,
process restarts, QQ sends or model calls occur. Passing does not prove model
comprehension accuracy or a real-world latency improvement.
"""
from __future__ import annotations

from layout_support import app_directory, project_directory, configure_test_paths
configure_test_paths()
import asyncio, contextlib, copy, hashlib, io, json, logging, re, sys, time, unittest
from pathlib import Path
from types import MethodType

ROOT=Path(__file__).resolve().parent
SOURCE = app_directory(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None)
RESULT=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'logs'/'tests'/('adaptive-tests-'+str(time.time_ns())+'.json')
sys.dont_write_bytecode=True
sys.path.insert(0,str(SOURCE))
import bot as bot_module
import semantics
from bot import Brain,DEFAULTS

def frame(**changes):
    data=dict(act='answer',target='self',meaning='群友分享今天外出散步的近况',unresolved='',
              reply_goal='自然回应已经明确的当前近况',mode='normal',needs_clarification=False)
    data.update(changes)
    return data

def fjson(**changes):return json.dumps(frame(**changes),ensure_ascii=False)
def expression(text='出去走走挺舒服的',next_action='wait',**extra):
    return json.dumps(dict(action='reply',text=text,next=next_action,**extra),ensure_ascii=False)
def verdict(ok=True,issue=''):return json.dumps(dict(ok=ok,issue=issue),ensure_ascii=False)
def row(text='我今天去公园散步了',mid='m1',uid='A',**changes):
    data=dict(role='group_member',sender={'id':uid,'display_name':'离线成员'},
              content=text,message_id=mid,event_kind='message')
    data.update(changes);return data
def transcript(text='我今天去公园散步了',mid='m1',uid='A',**changes):
    return json.dumps(row(text,mid,uid,**changes),ensure_ascii=False)

class AdaptivePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.cfg=copy.deepcopy(DEFAULTS['llm'])
        self.cfg.update(api_key='SYNTHETIC_NOT_A_KEY',semantic_pipeline=True,semantic_review_policy='adaptive')
        self.brain=Brain(self.cfg,copy.deepcopy(DEFAULTS['safety']),copy.deepcopy(DEFAULTS['persona']),120)
        self.responses=[];self.calls=[]
        async def fake_call(_self,messages,temperature,*,plan_mode=False,token_budget=0,**kwargs):
            self.calls.append({'messages':copy.deepcopy(messages),'temperature':temperature,
                               'plan_mode':plan_mode,'token_budget':token_budget})
            return self.responses.pop(0) if self.responses else None
        self.brain._call=MethodType(fake_call,self.brain)
    async def asyncTearDown(self):await self.brain.close()
    async def plan(self,responses,*,raw=None,continuation=False,group=1,at_self=True):
        self.responses=list(responses)
        return await self.brain.plan_reply(group,raw or transcript(),'离线成员',at_self,continuation=continuation)
    def trace(self,group=1):return self.brain._semantic_last[group]
    def assert_reply(self,p,text=None):
        self.assertIsNotNone(p);self.assertEqual(p.action,'reply')
        if text is not None:self.assertEqual(p.text,text)

    async def test_01_clear_self_sharing_uses_two_calls(self):
        p=await self.plan([fjson(),expression()])
        self.assert_reply(p,'出去走走挺舒服的');self.assertEqual(len(self.calls),2)
        self.assertEqual(self.trace()['review_source'],'local')
        self.assertEqual([p['phase'] for p in self.trace()['phases']],['understanding','expression'])

    async def test_02_clear_greeting_uses_two_calls(self):
        p=await self.plan([fjson(act='greet',target='bot',meaning='群友友好问候机器人',reply_goal='自然回应问候'),
                           expression('晚上好，今天过得怎么样')],raw=transcript('晚上好啊大肥鱼'))
        self.assert_reply(p);self.assertEqual(len(self.calls),2)
        self.assertEqual(self.trace()['review_source'],'local')

    async def test_03_clear_plan_sharing_uses_two_calls(self):
        p=await self.plan([fjson(meaning='群友计划今晚早点休息'),expression('早点休息也挺好的')],
                          raw=transcript('今晚打算早点睡'))
        self.assert_reply(p);self.assertEqual(len(self.calls),2)

    async def test_04_all_policy_preserves_three_calls_for_simple_chat(self):
        self.cfg['semantic_review_policy']='all'
        p=await self.plan([fjson(),expression(),verdict()])
        self.assert_reply(p);self.assertEqual(len(self.calls),3)
        self.assertEqual(self.trace()['review_source'],'model')

    async def test_05_unknown_policy_fails_closed_to_model_review(self):
        self.cfg['semantic_review_policy']='unrecognized-policy'
        p=await self.plan([fjson(),expression(),verdict()])
        self.assert_reply(p);self.assertEqual(len(self.calls),3)
        self.assertEqual(self.trace()['review_source'],'model')

    async def test_06_explicit_correction_three_calls_even_wrong_normal_frame(self):
        p=await self.plan([fjson(),expression('哦，你说的是我记住了前文'),verdict()],
                          raw=transcript('我说的是你有记忆了，不是我记性好'))
        self.assert_reply(p);self.assertEqual(len(self.calls),3)
        self.assertEqual(self.trace()['review_source'],'model')

    async def test_07_question_mark_requires_review_even_wrong_normal_frame(self):
        p=await self.plan([fjson(),expression('刚才是不是我没接准'),verdict()],raw=transcript('?'))
        self.assert_reply(p);self.assertEqual(len(self.calls),3)

    async def test_08_elliptical_rebuttal_requires_review(self):
        p=await self.plan([fjson(),expression('嫌我这句不够有劲啊'),verdict()],raw=transcript('就这'))
        self.assert_reply(p);self.assertEqual(len(self.calls),3)

    async def test_09_ambiguous_temporal_clause_requires_review_even_bad_frame(self):
        p=await self.plan([fjson(),expression('今天困得够呛啊'),verdict()],
                          raw=transcript('今天补觉了，困死我了'))
        self.assert_reply(p);self.assertEqual(len(self.calls),3)

    async def test_10_third_party_report_requires_review_even_bad_frame(self):
        p=await self.plan([fjson(),expression('让他自己来聊呗'),verdict()],raw=transcript('他说要挑战你'))
        self.assert_reply(p);self.assertEqual(len(self.calls),3)

    async def test_11_quote_requires_review(self):
        raw=transcript(quote={'sender':{'id':'B'},'content':'先前的话'})
        p=await self.plan([fjson(),expression(),verdict()],raw=raw)
        self.assert_reply(p);self.assertEqual(len(self.calls),3)

    async def test_12_explicit_reply_requires_review(self):
        p=await self.plan([fjson(),expression(),verdict()],raw=transcript(reply_to={'message_id':'old','sender_id':'B'}))
        self.assert_reply(p);self.assertEqual(len(self.calls),3)

    async def test_13_other_member_mention_requires_review(self):
        p=await self.plan([fjson(),expression(),verdict()],raw=transcript(mentions=['B']),at_self=False)
        self.assert_reply(p);self.assertEqual(len(self.calls),3)

    async def test_14_poke_requires_review(self):
        p=await self.plan([fjson(),expression(),verdict()],raw=transcript('戳一戳',event_kind='poke',related_message=row()))
        self.assert_reply(p);self.assertEqual(len(self.calls),3)

    async def test_15_recent_multiuser_interleaving_requires_review(self):
        raw='\n'.join(json.dumps(x,ensure_ascii=False) for x in [row('先说个近况','a0','A'),row('我也来插一句','b0','B'),row()])
        p=await self.plan([fjson(),expression(),verdict()],raw=raw)
        self.assert_reply(p);self.assertEqual(len(self.calls),3)

    async def test_16_known_unresolved_frame_requires_review(self):
        p=await self.plan([fjson(unresolved='发生在之前还是之后尚不明确'),expression('今天挺累的啊'),verdict()])
        self.assert_reply(p);self.assertEqual(len(self.calls),3)

    async def test_17_repair_mode_always_model_reviews(self):
        p=await self.plan([fjson(mode='repair'),expression('刚才那句我答偏了'),verdict()])
        self.assert_reply(p);self.assertEqual(len(self.calls),3)

    async def test_18_banter_mode_always_model_reviews(self):
        p=await self.plan([fjson(mode='banter'),expression('你这问法还挺会挑事'),verdict()])
        self.assert_reply(p);self.assertEqual(len(self.calls),3)

    async def test_19_normal_mode_aggressive_draft_still_model_reviews(self):
        p=await self.plan([fjson(),expression('你脑子是不是进水了'),verdict()])
        self.assert_reply(p);self.assertEqual(len(self.calls),3)
        self.assertEqual(self.trace()['review_source'],'model')

    async def test_20_bad_model_review_never_delivers_risky_draft(self):
        p=await self.plan([fjson(mode='repair'),expression('错误草稿'),'not-json'])
        self.assertTrue(p is None or p.action!='reply');self.assertEqual(len(self.calls),3)

    async def test_21_understanding_format_repair_cannot_fast_pass(self):
        p=await self.plan(['not-json',fjson(),expression(),verdict()])
        self.assert_reply(p);self.assertEqual(len(self.calls),4)
        self.assertEqual(self.trace()['review_source'],'model')

    async def test_22_expression_format_repair_cannot_fast_pass(self):
        p=await self.plan([fjson(),'not-json',expression(),verdict()])
        self.assert_reply(p);self.assertEqual(len(self.calls),4)
        self.assertEqual(self.trace()['review_source'],'model')

    async def test_23_rewrite_after_rejection_is_always_reviewed(self):
        p=await self.plan([fjson(mode='repair'),expression('错误草稿'),verdict(False,'指代错误'),
                           expression('哦，你说的是我记住了'),verdict()])
        self.assert_reply(p,'哦，你说的是我记住了');self.assertEqual(len(self.calls),5)

    async def test_24_duplicate_fast_draft_is_not_locally_released(self):
        await self.plan([fjson(),expression()])
        before=len(self.calls)
        p=await self.plan([fjson(),expression(),verdict(),expression('那就慢慢走，放松一下'),verdict()],
                          raw=transcript('我今天也去公园散步了','m2'))
        self.assert_reply(p,'那就慢慢走，放松一下')
        self.assertEqual(len(self.calls)-before,5)
        self.assertEqual(self.trace()['review_source'],'model')

    async def test_25_continuation_keeps_model_review_and_cached_understanding(self):
        await self.plan([fjson(),expression(next_action='reply')])
        before=len(self.calls)
        p=await self.plan([expression('顺便看看路边的风景'),verdict()],continuation=True)
        self.assert_reply(p);self.assertEqual(len(self.calls)-before,2)
        self.assertEqual([x['phase'] for x in self.trace()['phases']],['expression','review'])
        self.assertEqual(self.trace()['review_source'],'model')

    async def test_26_transport_unavailable_never_adds_review_or_retry(self):
        p=await self.plan([None])
        self.assertTrue(p is None or p.action!='reply');self.assertEqual(len(self.calls),1)

    async def test_27_writer_transport_unavailable_never_retries(self):
        p=await self.plan([fjson(),None])
        self.assertTrue(p is None or p.action!='reply');self.assertEqual(len(self.calls),2)

    async def test_28_trace_and_logs_include_phase_timing_without_fixture_text(self):
        marker='PRIVATE_SYNTHETIC_TEXT_DO_NOT_LOG'
        with self.assertLogs(bot_module.log.name,level='INFO') as records:
            p=await self.plan([fjson(),expression()],raw=transcript('我今天去公园散步了',display_data=marker))
        self.assert_reply(p)
        self.assertTrue(all(isinstance(x['ms'],int) and x['ms']>=0 for x in self.trace()['phases']))
        text='\n'.join(records.output)
        self.assertIn('phase=understanding',text);self.assertIn('phase=expression',text)
        self.assertNotIn(marker,text);self.assertNotIn('SYNTHETIC_NOT_A_KEY',text)

    async def test_32_old_unrelated_history_does_not_block_clear_current_chat(self):
        history=[row('另一个很早的话题','b-old','B'),
                 row('晚上好','bot1','BOT',role='assistant'),
                 row('我想出门走走','a-old','A'),
                 row('那就放松一下','bot2','BOT',role='assistant'),row()]
        raw='\n'.join(json.dumps(x,ensure_ascii=False) for x in history)
        p=await self.plan([fjson(),expression()],raw=raw)
        self.assert_reply(p);self.assertEqual(len(self.calls),2)
        self.assertEqual(self.trace()['review_source'],'local')

    async def test_33_known_bot_mention_does_not_force_extra_review(self):
        history=[row('晚上好','bot1','BOT',role='assistant'),row(mentions=['BOT'])]
        raw='\n'.join(json.dumps(x,ensure_ascii=False) for x in history)
        p=await self.plan([fjson(),expression()],raw=raw)
        self.assert_reply(p);self.assertEqual(len(self.calls),2)

    async def test_34_unknown_string_mention_keeps_model_review(self):
        p=await self.plan([fjson(),expression(),verdict()],raw=transcript(mentions=['UNKNOWN']),at_self=False)
        self.assert_reply(p);self.assertEqual(len(self.calls),3)

    async def test_35_distinct_ordinary_topic_also_uses_two_calls(self):
        p=await self.plan([fjson(meaning='群友正在看电影'),expression('看的是哪部电影呀')],raw=transcript('我今天正在看电影'))
        self.assert_reply(p);self.assertEqual(len(self.calls),2)

    async def test_36_first_unique_bot_at_uses_trusted_at_self_without_history(self):
        p=await self.plan([fjson(act='greet',target='bot',meaning='群友友好问候机器人'),
                           expression('晚上好，今天过得怎么样')],
                          raw=transcript('晚上好啊大肥鱼',mentions=['BOT']),at_self=True)
        self.assert_reply(p);self.assertEqual(len(self.calls),2)
        self.assertEqual(self.trace()['review_source'],'local')

    async def test_37_multiple_mentions_still_review_even_when_bot_is_at(self):
        p=await self.plan([fjson(act='greet',target='bot'),expression('晚上好呀'),verdict()],
                          raw=transcript('晚上好啊大肥鱼',mentions=['BOT','B']),at_self=True)
        self.assert_reply(p);self.assertEqual(len(self.calls),3)
        self.assertEqual(self.trace()['review_source'],'model')

    async def test_38_attack_challenge_cannot_use_misclassified_greeting_fast_path(self):
        p=await self.plan([fjson(act='greet',target='bot',meaning='群友向机器人问好'),
                           expression('嫌我这句不够有劲啊'),verdict()],raw=transcript('攻击力就这'))
        self.assert_reply(p);self.assertEqual(len(self.calls),3)
        self.assertEqual(self.trace()['review_source'],'model')

    async def test_39_raw_insult_cannot_use_misclassified_greeting_fast_path(self):
        p=await self.plan([fjson(act='greet',target='bot',meaning='群友向机器人问好'),
                           expression('你这是叫我出来啊'),verdict()],raw=transcript('死肥鱼'))
        self.assert_reply(p);self.assertEqual(len(self.calls),3)
        self.assertEqual(self.trace()['review_source'],'model')

    async def test_40_bad_normal_draft_is_reviewed_and_rejected_not_directly_released(self):
        bad_drafts=('你玩得很烂','你水平真的很低','混了一整天吧')
        for i,bad in enumerate(bad_drafts):
            with self.subTest(draft=bad):
                before=len(self.calls)
                p=await self.plan([fjson(meaning='群友正在玩三角洲',reply_goal='正常回应游戏近况'),
                                   expression(bad),verdict(False,'没有依据的贬低或动机归因'),
                                   expression('今天打算玩多久呀'),verdict()],
                                  raw=transcript('我在玩三角洲',mid='game'+str(i)),group=10+i)
                self.assert_reply(p,'今天打算玩多久呀')
                self.assertNotIn(p.text,bad_drafts)
                self.assertEqual(len(self.calls)-before,5)
                self.assertEqual(self.trace(10+i)['review_source'],'model')

    async def test_41_ordinary_warm_question_draft_still_uses_two_calls(self):
        p=await self.plan([fjson(),expression('今天公园人多吗')])
        self.assert_reply(p,'今天公园人多吗');self.assertEqual(len(self.calls),2)
        self.assertEqual(self.trace()['review_source'],'local')

class LocalPolicyTests(unittest.TestCase):
    def reason(self,raw=None,**flags):
        f=semantics.parse_semantic_frame(fjson())
        return semantics.review_risk_reason(raw or transcript(),f,'出去走走挺舒服的',**flags)
    def test_29_malformed_context_requires_review_not_fast_pass(self):
        self.assertTrue(self.reason('not-json'))
    def test_30_display_name_does_not_become_member_speech(self):
        a=self.reason(transcript());b=self.reason(transcript(sender={'id':'A','display_name':'丝袜'}))
        self.assertEqual(a,b)
    def test_31_risk_flags_cannot_be_ignored(self):
        for flag in ('continuation','understanding_repaired','expression_repaired','rewriting'):
            with self.subTest(flag=flag):self.assertTrue(self.reason(**{flag:True}))

def main():
    stream=io.StringIO();stdout=io.StringIO();stderr=io.StringIO()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (AdaptivePipelineTests,LocalPolicyTests))
    with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    ledger={'kind':'OFFLINE_SYNTHETIC_ADAPTIVE_REVIEW_CONTRACT_NOT_LIVE_MODEL_QUALITY',
        'source':str(SOURCE),'tests_run':result.testsRun,'successful':result.wasSuccessful(),
        'failures':[{'test':str(t),'traceback':trace} for t,trace in result.failures],
        'errors':[{'test':str(t),'traceback':trace} for t,trace in result.errors],
        'literal_stdout':stdout.getvalue(),'literal_stderr':stderr.getvalue(),'literal_test_output':stream.getvalue(),
        'source_hashes':{n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in ('bot.py','semantics.py')},
        'real_model_requests':0,'real_qq_sends':0,'profile_reads':0,'exit_status':0 if result.wasSuccessful() else 1}
    RESULT.parent.mkdir(parents=True,exist_ok=True)
    RESULT.write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
    (RESULT.parent/'latest-test-result.json').write_text(json.dumps({'result':str(RESULT)},indent=2),encoding='utf-8')
    print(stream.getvalue(),end='');print(json.dumps({k:ledger[k] for k in ('tests_run','successful','exit_status')},ensure_ascii=False))
    print('RESULT_JSON '+str(RESULT));return ledger['exit_status']

if __name__=='__main__':raise SystemExit(main())
