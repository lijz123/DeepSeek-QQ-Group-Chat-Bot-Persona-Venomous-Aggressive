"""Offline same-second member routing, native reply, and safety regression checks.

python simultaneous_reply_tests.py SOURCE_DIRECTORY [RESULT_JSON]
Only actual Bot queues with synthetic events, fake models and a fake OneBot are
used. This module never reads config.json, credentials, or opens a connection.
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

SOURCE = app_directory(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None)
RESULT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
sys.path.insert(0, str(SOURCE))
sys.dont_write_bytecode = True

from bot import Bot, Msg
from dialogue import DialogueState, PendingTurn
from dialogue_selftest import test_config, BOT_ID, GROUP, ALICE, BOB, FakeClient
from persona import ReplyPlan

CAROL = BOB + 1


def config(*, enabled=True, quote='auto', gap=.025):
    cfg = test_config()
    # Scale the local collection window for fast integration runs; a separate
    # case below exercises the actual one-second user-facing configuration.
    cfg['behavior'].update(concurrent_reply_window=.06 if enabled else 0,
        merge_delay=.025, merge_at_delay=.015, merge_max_wait=.08,
        quote_mode=quote, quote_busy_per_minute=9999,
        min_gap=gap, direct_cooldown=gap, direct_user_cooldown=gap,
        keywords=['大肥鱼', '肥鱼'], model_reply_planning=True,
        model_burst_max_messages=3, max_chain=100)
    cfg['voice']['enabled'] = False
    cfg['search']['enabled'] = False
    cfg['owner_chat']['enabled'] = False
    cfg['safety']['blocked_output'] = []
    return cfg


class RecordingBrain:
    def __init__(self):
        self.calls = []
        self.started = asyncio.Event()
        self.release = None
        self.delay = 0
        self.closed = False
        self.active = 0
        self.max_active = 0
        self.next_action = 'wait'

    async def plan_reply(self, group_id, transcript, speaker, is_at, **kwargs):
        selected = json.loads(transcript.splitlines()[-1])
        self.calls.append(dict(group_id=group_id, selected=selected,
            transcript=transcript, speaker=speaker, is_at=is_at, **kwargs))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if self.release is not None:
                await self.release.wait()
            if self.delay:
                await asyncio.sleep(self.delay)
            return ReplyPlan('reply', '回答' + str(selected['sender']['id']) + '：' + selected['content'], self.next_action)
        finally:
            self.active -= 1

    async def reply(self, *args, **kwargs):
        plan = await self.plan_reply(*args, **kwargs)
        return plan.text

    async def close(self):
        self.closed = True


class Harness:
    def __init__(self, settings=None):
        self.bot = Bot(settings or config())
        self.bot.brain = RecordingBrain()
        self.bot.client = FakeClient()
        self.ids = 100

    @property
    def brain(self): return self.bot.brain

    @property
    def client(self): return self.bot.client

    def event(self, text, *, user=ALICE, group=GROUP, at=None, reply=None, message_id=None):
        self.ids += 1
        segs = []
        if reply is not None: segs.append({'type': 'reply', 'data': {'id': str(reply)}})
        for user_id in at or []: segs.append({'type': 'at', 'data': {'qq': str(user_id)}})
        segs.append({'type': 'text', 'data': {'text': text}})
        return dict(post_type='message', message_type='group', group_id=group,
            user_id=user, self_id=BOT_ID, message_id=self.ids if message_id is None else message_id,
            sender={'user_id': user, 'nickname': '合成成员' + str(user), 'card': ''}, message=segs)

    async def push(self, text, **kwargs):
        event = self.event(text, **kwargs)
        await self.bot.on_event(event)
        return str(event['message_id'])

    async def idle(self):
        await asyncio.wait_for(self.bot.wait_idle(), 2.5)

    async def close(self):
        await self.bot.close()

    def quote(self, index):
        return [item['data']['id'] for item in self.client.sent[index]['message'] if item['type'] == 'reply']

    def content(self, index):
        return ''.join(item['data']['text'] for item in self.client.sent[index]['message'] if item['type'] == 'text')


class ConcurrentQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self): self.h = Harness()
    async def asyncTearDown(self): await self.h.close()

    async def two(self, *, at=True):
        first = await self.h.push('第一位的问题', at=[BOT_ID] if at else None)
        second = await self.h.push('第二位的问题', user=BOB, at=[BOT_ID] if at else None)
        await self.h.idle()
        return first, second

    async def test_two_direct_members_both_receive_answers_despite_cooldown(self):
        await self.two()
        self.assertEqual(len(self.h.client.sent), 2)
        self.assertEqual(len(self.h.brain.calls), 2)

    async def test_two_direct_members_each_get_native_quote_of_their_own_source(self):
        ids = await self.two()
        self.assertEqual([self.h.quote(i) for i in range(len(self.h.client.sent))], [[x] for x in ids])

    async def test_selected_sender_does_not_change_to_latest_member(self):
        await self.two()
        self.assertEqual([call['selected']['sender']['id'] for call in self.h.brain.calls], [str(ALICE), str(BOB)])

    async def test_selected_text_stays_associated_with_its_member(self):
        await self.two()
        self.assertEqual([call['selected']['content'] for call in self.h.brain.calls], ['第一位的问题', '第二位的问题'])

    async def test_first_answer_and_second_answer_keep_original_order(self):
        await self.two()
        self.assertIn(str(ALICE), self.h.content(0))
        self.assertIn(str(BOB), self.h.content(1))

    async def test_three_direct_members_receive_three_distinct_quoted_answers(self):
        ids = []
        for uid, text in [(ALICE, '甲的问题'), (BOB, '乙的问题'), (CAROL, '丙的问题')]:
            ids.append(await self.h.push(text, user=uid, at=[BOT_ID]))
        await self.h.idle()
        self.assertEqual(len(self.h.client.sent), 3)
        self.assertEqual([self.h.quote(i) for i in range(3)], [[x] for x in ids])

    async def test_same_group_model_work_remains_serial(self):
        self.h.brain.delay = .015
        await self.two()
        self.assertEqual(self.h.brain.max_active, 1)

    async def test_direct_cooldown_is_waited_not_bypassed(self):
        self.h.bot.beh.update(min_gap=.055, direct_cooldown=.055)
        await self.two()
        self.assertEqual(len(self.h.client.sent), 2)
        self.assertGreaterEqual(self.h.client.sent[1]['sent_at']-self.h.client.sent[0]['sent_at'], .045)

    async def test_delivery_rechecks_cooldown_changed_while_model_was_generating(self):
        self.h.brain.release = asyncio.Event()
        self.h.bot.beh.update(min_gap=.075, direct_cooldown=.075)
        await self.h.push('第一位的问题', at=[BOT_ID]); await self.h.push('第二位的问题', user=BOB, at=[BOT_ID])
        await asyncio.wait_for(self.h.brain.started.wait(), .5)
        changed = time.monotonic()
        self.h.bot._last_any = changed
        self.h.bot._last_group[GROUP] = changed
        self.h.brain.release.set()
        await self.h.idle()
        self.assertEqual(len(self.h.client.sent), 2)
        self.assertGreaterEqual(self.h.client.sent[0]['sent_at']-changed, .060)

    async def test_quote_mode_never_is_overridden_only_for_competing_requests(self):
        self.h.bot.beh['quote_mode'] = 'never'
        ids = await self.two()
        self.assertEqual(len(self.h.client.sent), 2)
        self.assertEqual([self.h.quote(i) for i in range(2)], [[x] for x in ids])

    async def test_quote_auto_does_not_require_a_busy_group_for_overlap(self):
        self.h.bot.beh['quote_busy_per_minute'] = 999999
        await self.two()
        self.assertEqual(len(self.h.client.sent), 2)
        self.assertTrue(all(self.h.quote(i) for i in range(2)))

    async def test_keyword_addressed_members_are_also_kept(self):
        first = await self.h.push('肥鱼，第一位的问题')
        second = await self.h.push('大肥鱼，第二位的问题', user=BOB)
        await self.h.idle()
        self.assertEqual(len(self.h.client.sent), 2)
        self.assertEqual([self.h.quote(i) for i in range(2)], [[first], [second]])

    async def test_known_reply_to_bot_can_compete_with_new_direct_request(self):
        self.h.bot.beh.update(min_gap=0, direct_cooldown=0)
        await self.h.push('先问一个', at=[BOT_ID]); await self.h.idle()
        bot_id = self.h.client.sent[0]['sent_id']
        self.h.bot.beh.update(min_gap=.025, direct_cooldown=.025)
        first = await self.h.push('继续解释一下', reply=bot_id)
        second = await self.h.push('我也有问题', user=BOB, at=[BOT_ID])
        await self.h.idle()
        self.assertEqual(len(self.h.client.sent), 3)
        self.assertEqual([self.h.quote(i) for i in (1, 2)], [[first], [second]])

    async def test_existing_member_implicit_followup_is_retained_before_other_direct_arrival(self):
        self.h.bot.beh.update(min_gap=0, direct_cooldown=0)
        await self.h.push('晚上聊点什么', at=[BOT_ID]); await self.h.idle()
        self.h.bot.beh.update(min_gap=.025, direct_cooldown=.025)
        first = await self.h.push('你还挺天真的')
        second = await self.h.push('你知道是什么类型的群聊吗', user=BOB, at=[BOT_ID])
        await self.h.idle()
        self.assertEqual(len(self.h.client.sent), 3)
        self.assertEqual([self.h.quote(i) for i in (1, 2)], [[first], [second]])

    async def test_existing_member_followup_is_retained_when_other_direct_arrives_first(self):
        self.h.bot.beh.update(min_gap=0, direct_cooldown=0)
        await self.h.push('晚上聊点什么', at=[BOT_ID]); await self.h.idle()
        self.h.bot.beh.update(min_gap=.025, direct_cooldown=.025)
        first = await self.h.push('我也想问一件事', user=BOB, at=[BOT_ID])
        second = await self.h.push('你还挺天真的', user=ALICE)
        await self.h.idle()
        self.assertEqual(len(self.h.client.sent), 3)
        self.assertEqual([self.h.quote(i) for i in (1, 2)], [[first], [second]])

    async def test_actual_one_second_window_includes_member_arriving_after_point_three_seconds(self):
        self.h.bot.beh['concurrent_reply_window'] = 1.0
        first = await self.h.push('甲的问题', at=[BOT_ID])
        await asyncio.sleep(.30)
        second = await self.h.push('乙的问题', user=BOB, at=[BOT_ID])
        await self.h.idle()
        self.assertEqual(len(self.h.client.sent), 2)
        self.assertEqual([self.h.quote(i) for i in (0, 1)], [[first], [second]])

    async def test_four_member_primary_replies_are_not_self_supplement_chain(self):
        self.h.bot.beh['max_chain'] = 3
        ids = []
        for uid in (ALICE, BOB, CAROL, CAROL+1):
            ids.append(await self.h.push('各自的问题'+str(uid), user=uid, at=[BOT_ID]))
        await self.h.idle()
        self.assertEqual(len(self.h.client.sent), 4)
        self.assertEqual([self.h.quote(i) for i in range(4)], [[x] for x in ids])

    async def test_same_sender_consecutive_fragments_still_merge(self):
        await self.h.push('肥鱼，第一段')
        last = await self.h.push('第二段')
        await self.h.idle()
        self.assertEqual(len(self.h.brain.calls), 1)
        self.assertEqual(self.h.brain.calls[0]['selected']['content'], '肥鱼，第一段\n第二段')
        self.assertEqual(self.h.brain.calls[0]['selected']['message_id'], last)

    async def test_different_senders_are_never_merged(self):
        await self.two()
        self.assertTrue(all(len(call['selected']['source_message_ids']) == 1 for call in self.h.brain.calls))

    async def test_aba_interleaving_does_not_assign_a_suffix_to_b(self):
        first = await self.h.push('肥鱼，甲第一段')
        second = await self.h.push('肥鱼，乙的问题', user=BOB)
        suffix = await self.h.push('甲第二段')
        await self.h.idle()
        self.assertEqual(len(self.h.brain.calls), 2)
        self.assertEqual(len(self.h.client.sent), 2)
        alice = next(call['selected'] for call in self.h.brain.calls if call['selected']['sender']['id'] == str(ALICE))
        self.assertEqual(alice['content'], '肥鱼，甲第一段\n甲第二段')
        self.assertEqual(alice['source_message_ids'], [first, suffix])
        self.assertEqual(alice['message_id'], suffix)
        self.assertEqual({tuple(self.h.quote(i)) for i in range(2)}, {(suffix,), (second,)})
        for call in self.h.brain.calls:
            if call['selected']['sender']['id'] == str(BOB):
                self.assertNotIn('甲第二段', call['selected']['content'])

    async def test_no_unrequested_native_at_is_added(self):
        await self.two()
        self.assertFalse(any(item['type'] == 'at' for sent in self.h.client.sent for item in sent['message']))

    async def test_competing_primary_replies_do_not_launch_extra_model_bursts(self):
        self.h.brain.next_action = 'reply'
        await self.two()
        self.assertEqual(len(self.h.brain.calls), 2)
        self.assertEqual(len(self.h.client.sent), 2)

    async def test_legacy_single_reply_path_also_preserves_each_competing_member(self):
        self.h.bot.beh['model_reply_planning'] = False
        ids = await self.two()
        self.assertEqual(len(self.h.brain.calls), 2)
        self.assertEqual(len(self.h.client.sent), 2)
        self.assertEqual([self.h.quote(i) for i in range(2)], [[x] for x in ids])

    async def test_legacy_random_addendum_does_not_take_over_competing_primary_queue(self):
        self.h.bot.beh.update(model_reply_planning=False, second_reply_rate=1,
            second_reply_delay_min=.001, second_reply_delay_max=.001)
        await self.two()
        self.assertEqual(len(self.h.brain.calls), 2)
        self.assertEqual(len(self.h.client.sent), 2)


class IsolationAndSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self): self.h = Harness()
    async def asyncTearDown(self): await self.h.close()

    async def test_single_member_auto_quote_behavior_stays_unchanged(self):
        await self.h.push('一个人的问题', at=[BOT_ID]); await self.h.idle()
        self.assertEqual(len(self.h.client.sent), 1)
        self.assertEqual(self.h.quote(0), [])

    async def test_single_member_never_quote_behavior_stays_unchanged(self):
        self.h.bot.beh['quote_mode'] = 'never'
        await self.h.push('一个人的问题', at=[BOT_ID]); await self.h.idle()
        self.assertEqual(self.h.quote(0), [])

    async def test_single_member_always_quote_behavior_stays_unchanged(self):
        self.h.bot.beh['quote_mode'] = 'always'
        source = await self.h.push('一个人的问题', at=[BOT_ID]); await self.h.idle()
        self.assertEqual(self.h.quote(0), [source])

    async def test_disabled_feature_preserves_old_cooldown_drop(self):
        self.h.bot.beh['concurrent_reply_window'] = 0
        await self.h.push('第一位的问题', at=[BOT_ID]); await self.h.push('第二位的问题', user=BOB, at=[BOT_ID])
        await self.h.idle()
        self.assertEqual(len(self.h.client.sent), 1)
        self.assertEqual(self.h.quote(0), [])

    async def test_other_member_chatter_is_not_promoted_to_request(self):
        await self.h.push('第一位的问题', at=[BOT_ID]); await self.h.push('我是在跟别人聊天', user=BOB)
        await self.h.idle()
        self.assertEqual(len(self.h.client.sent), 1)
        self.assertEqual(self.h.quote(0), [])

    async def test_at_other_member_does_not_gain_reply_eligibility(self):
        await self.h.push('第一位的问题', at=[BOT_ID]); await self.h.push('这是给第三位的问题', user=BOB, at=[CAROL])
        await self.h.idle()
        self.assertEqual(len(self.h.brain.calls), 1)

    async def test_unknown_quote_is_not_guessed_as_bot_reply(self):
        await self.h.push('第一位的问题', at=[BOT_ID]); await self.h.push('回应某条旧话', user=BOB, reply='unknown')
        await self.h.idle()
        self.assertEqual(len(self.h.brain.calls), 1)

    async def test_blacklisted_member_never_enters_competing_queue(self):
        self.h.bot._user_blacklist.add(BOB)
        await self.h.push('第一位的问题', at=[BOT_ID]); await self.h.push('第二位的问题', user=BOB, at=[BOT_ID])
        await self.h.idle()
        self.assertEqual(len(self.h.brain.calls), 1)
        self.assertEqual(self.h.quote(0), [])

    async def test_blacklisted_group_remains_silent(self):
        self.h.bot._group_blacklist.add(GROUP)
        await self.h.push('第一位的问题', at=[BOT_ID]); await self.h.push('第二位的问题', user=BOB, at=[BOT_ID])
        await self.h.idle()
        self.assertEqual(self.h.client.sent, [])

    async def test_disabled_at_entry_is_not_reenabled_by_competing_members(self):
        self.h.bot.beh['at_reply'] = False
        await self.h.push('第一位的问题', at=[BOT_ID]); await self.h.push('第二位的问题', user=BOB, at=[BOT_ID])
        await self.h.idle()
        self.assertEqual(self.h.brain.calls, [])

    async def test_disabled_keyword_entry_is_not_reenabled_by_competing_members(self):
        self.h.bot.beh['keyword_reply'] = False
        await self.h.push('肥鱼，第一位的问题'); await self.h.push('肥鱼，第二位的问题', user=BOB)
        await self.h.idle()
        self.assertEqual(self.h.brain.calls, [])

    async def test_model_wait_decision_is_not_overridden_by_overlap(self):
        async def wait_plan(*args, **kwargs):
            return ReplyPlan('wait')
        self.h.brain.plan_reply = wait_plan
        await self.h.push('第一位的问题', at=[BOT_ID]); await self.h.push('第二位的问题', user=BOB, at=[BOT_ID])
        await self.h.idle()
        self.assertEqual(self.h.client.sent, [])

    async def test_whitelist_exclusion_remains_silent(self):
        self.h.bot._whitelist = {GROUP + 1}
        await self.h.push('第一位的问题', at=[BOT_ID]); await self.h.push('第二位的问题', user=BOB, at=[BOT_ID])
        await self.h.idle()
        self.assertEqual(self.h.brain.calls, [])

    async def test_blocked_input_never_becomes_an_overlap_request(self):
        self.h.bot.safety['blocked_input'] = ['BLOCKED_SYNTHETIC']
        await self.h.push('第一位的问题', at=[BOT_ID]); await self.h.push('BLOCKED_SYNTHETIC', user=BOB, at=[BOT_ID])
        await self.h.idle()
        self.assertEqual(len(self.h.brain.calls), 1)

    async def test_muted_member_keyword_does_not_bypass_mute(self):
        self.h.bot._muted.add((GROUP, BOB))
        await self.h.push('第一位的问题', at=[BOT_ID]); await self.h.push('肥鱼，第二位的问题', user=BOB)
        await self.h.idle()
        self.assertEqual(len(self.h.brain.calls), 1)
        self.assertEqual(self.h.quote(0), [])

    async def test_expired_overlap_does_not_restore_previously_interrupted_member(self):
        self.h.bot.beh.update(min_gap=0, direct_cooldown=0)
        await self.h.push('先和甲聊一句', at=[BOT_ID]); await self.h.idle()
        await self.h.push('乙单独的问题', user=BOB, at=[BOT_ID])
        await asyncio.sleep(.09)
        await self.h.push('甲的普通续话', user=ALICE)
        await self.h.idle()
        self.assertEqual(len(self.h.brain.calls), 2)
        self.assertFalse(any(call['selected']['content'] == '甲的普通续话' for call in self.h.brain.calls))

    async def test_same_sender_stop_cancels_its_pending_old_question(self):
        await self.h.push('甲的旧问题', at=[BOT_ID]); await self.h.push('乙的问题', user=BOB, at=[BOT_ID])
        await self.h.push('不用回我了', user=ALICE)
        await self.h.idle()
        self.assertFalse(any('甲的旧问题' in self.h.content(i) for i in range(len(self.h.client.sent))))

    async def test_same_sender_redirect_to_other_member_cancels_old_question(self):
        await self.h.push('甲的旧问题', at=[BOT_ID]); await self.h.push('乙的问题', user=BOB, at=[BOT_ID])
        await self.h.push('改成问第三位', user=ALICE, at=[CAROL])
        await self.h.idle()
        self.assertFalse(any(call['selected']['content'] == '甲的旧问题' for call in self.h.brain.calls))

    async def test_muting_command_during_generation_invalidates_its_earlier_answer(self):
        self.h.bot.beh['mute_keywords'] = ['肥鱼闭嘴']
        self.h.brain.release = asyncio.Event()
        await self.h.push('甲的旧问题', at=[BOT_ID]); await self.h.push('乙的问题', user=BOB, at=[BOT_ID])
        await asyncio.wait_for(self.h.brain.started.wait(), .5)
        await self.h.push('肥鱼闭嘴', user=ALICE)
        self.h.brain.release.set()
        await self.h.idle()
        self.assertFalse(any('甲的旧问题' in self.h.content(i) for i in range(len(self.h.client.sent))))

    async def test_expired_ticket_cannot_resurrect_as_ordinary_direct_reply_after_pruning(self):
        self.h.brain.release = asyncio.Event()
        await self.h.push('甲的过期问题', at=[BOT_ID]); await self.h.push('乙的过期问题', user=BOB, at=[BOT_ID])
        await asyncio.wait_for(self.h.brain.started.wait(), .5)
        for ticket in self.h.bot._overlap_tickets.values(): ticket['created'] -= 121
        self.h.bot._overlap_prune()
        self.h.brain.release.set()
        await self.h.idle()
        self.assertEqual(self.h.client.sent, [])

    async def test_overlap_bookkeeping_is_cleared_when_group_queue_finishes(self):
        await self.h.push('第一位的问题', at=[BOT_ID]); await self.h.push('第二位的问题', user=BOB, at=[BOT_ID])
        await self.h.idle()
        for name in ('_overlap_tickets', '_overlap_cohorts', '_overlap_open', '_overlap_interrupted', '_overlap_activity'):
            self.assertEqual(getattr(self.h.bot, name), {}, name)

    async def test_group_per_minute_limit_is_not_relaxed(self):
        self.h.bot.beh['per_group_per_minute'] = 1
        await self.h.push('第一位的问题', at=[BOT_ID]); await self.h.push('第二位的问题', user=BOB, at=[BOT_ID])
        await self.h.idle()
        self.assertEqual(len(self.h.client.sent), 1)

    async def test_global_per_minute_limit_is_not_relaxed(self):
        self.h.bot.beh['global_per_minute'] = 1
        await self.h.push('第一位的问题', at=[BOT_ID]); await self.h.push('第二位的问题', user=BOB, at=[BOT_ID])
        await self.h.idle()
        self.assertEqual(len(self.h.client.sent), 1)

    async def test_close_cancels_waiting_generation_and_pending_requests(self):
        self.h.brain.release = asyncio.Event()
        await self.h.push('第一位的问题', at=[BOT_ID]); await self.h.push('第二位的问题', user=BOB, at=[BOT_ID])
        await asyncio.wait_for(self.h.brain.started.wait(), .5)
        await self.h.bot.close()
        self.assertEqual(self.h.client.sent, [])
        self.assertEqual(self.h.bot._pending, {})

    async def test_connection_identity_change_during_generation_withholds_old_output(self):
        self.h.brain.release = asyncio.Event()
        await self.h.push('第一位的问题', at=[BOT_ID]); await self.h.push('第二位的问题', user=BOB, at=[BOT_ID])
        await asyncio.wait_for(self.h.brain.started.wait(), .5)
        self.h.bot.client.self_id += 1
        self.h.brain.release.set()
        await self.h.idle()
        self.assertFalse(any('第一位的问题' in self.h.content(i) for i in range(len(self.h.client.sent))))

    async def test_connection_object_change_invalidates_all_old_cohort_tickets(self):
        self.h.brain.release = asyncio.Event()
        await self.h.push('第一位的问题', at=[BOT_ID]); await self.h.push('第二位的问题', user=BOB, at=[BOT_ID])
        await asyncio.wait_for(self.h.brain.started.wait(), .5)
        self.h.bot.client.ws = object()
        self.h.brain.release.set()
        await self.h.idle()
        self.assertEqual(self.h.client.sent, [])

    async def test_senders_in_different_groups_do_not_force_cross_group_quotes(self):
        self.h.bot.beh.update(min_gap=0, direct_cooldown=0)
        await self.h.push('甲群的问题', at=[BOT_ID]); await self.h.push('乙群的问题', user=BOB, group=GROUP+1, at=[BOT_ID])
        await self.h.idle()
        self.assertEqual(len(self.h.client.sent), 2)
        self.assertTrue(all(not self.h.quote(i) for i in range(2)))

    async def test_duplicate_message_is_not_queued_twice(self):
        await self.h.push('第一位的问题', at=[BOT_ID], message_id=501)
        await self.h.push('第一位的问题', at=[BOT_ID], message_id=501)
        await self.h.push('第二位的问题', user=BOB, at=[BOT_ID], message_id=502)
        await self.h.idle()
        self.assertEqual(len(self.h.brain.calls), 2)


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
        for cls in (ConcurrentQueueTests, IsolationAndSafetyTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        success=result.wasSuccessful(), fixtures='actual Bot queue, synthetic members, fake Brain and OneBot',
        real_model_requests=0, real_search_requests=0, real_QQ_messages=0, network='none')
    if RESULT: RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
