"""Identity-scoped duel/idle-followup tests; synthetic identities and fake time.

python duel_tests.py PROJECT_DIRECTORY [RESULT_JSON]
No model, network, live QQ, private configuration, or real sixty-second wait.
"""
from __future__ import annotations

from layout_support import app_directory, project_directory, configure_test_paths
configure_test_paths()

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

if len(sys.argv) > 1:
    sys.path.insert(0, str(app_directory(sys.argv[1])))
RESULT = Path(sys.argv[2]) if len(sys.argv) > 2 else None
from bot import Bot, Brain
from dialogue import DialogueState
from dialogue_selftest import DialogueTests, test_config, BOT_ID, GROUP, ALICE, BOB
from duel import DuelTracker
from persona import ReplyPlan
from retaliation import RETALIATION_MARKER, retaliation_tone


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = DuelTracker(max_groups=3)

    def arm(self, **changes):
        values = dict(group_id=GROUP, user_id=ALICE, bot_id=BOT_ID, revision=1,
                      user_text='我是你爹', bot_text='脚本回击', now=1000,
                      idle_seconds=60, expiry_seconds=180)
        values.update(changes)
        return self.tracker.arm(**values)

    def test_initial_tracker_is_empty(self):
        self.assertEqual(len(self.tracker), 0)

    def test_arm_retains_exact_participant_identity(self):
        session = self.arm()
        self.assertEqual((session.group_id, session.user_id, session.bot_id, session.revision),
                         (GROUP, ALICE, BOT_ID, 1))

    def test_arm_sets_sixty_second_deadline(self):
        session = self.arm()
        self.assertEqual((session.opened_at, session.due_at, session.expires_at), (1000, 1060, 1180))

    def test_before_deadline_not_due(self):
        self.arm()
        self.assertEqual(self.tracker.due(1059.99), ())

    def test_exact_deadline_is_due(self):
        session = self.arm()
        self.assertEqual(self.tracker.due(1060), (session,))

    def test_claim_is_exactly_once(self):
        session = self.arm()
        self.assertTrue(self.tracker.claim(session, 1060))
        self.assertFalse(self.tracker.claim(session, 1061))
        self.assertIsNone(self.tracker.current(GROUP))

    def test_claim_before_deadline_is_rejected(self):
        session = self.arm()
        self.assertFalse(self.tracker.claim(session, 1059))

    def test_cancel_only_removes_selected_group(self):
        self.arm()
        other = self.arm(group_id=GROUP + 1)
        self.tracker.cancel(GROUP)
        self.assertIs(self.tracker.current(GROUP + 1), other)
        self.assertIsNone(self.tracker.current(GROUP))

    def test_clear_forgets_all_ephemeral_participants(self):
        self.arm()
        self.arm(group_id=GROUP + 1)
        self.tracker.clear()
        self.assertEqual(len(self.tracker), 0)

    def test_old_snapshot_cannot_claim_replacement(self):
        first = self.arm()
        second = self.arm(now=1001, bot_text='新的脚本回击')
        self.assertIsNot(first, second)
        self.assertFalse(self.tracker.claim(first, 1061))
        self.assertTrue(self.tracker.claim(second, 1061))

    def test_later_ack_resets_idle_deadline_not_original_expiry(self):
        self.arm()
        latest = self.arm(now=1010, bot_text='第二条脚本回击')
        self.assertEqual((latest.opened_at, latest.due_at, latest.expires_at), (1000, 1070, 1180))

    def test_replaced_user_does_not_inherit_first_user_session(self):
        self.arm()
        second = self.arm(user_id=BOB, now=1001)
        self.assertEqual((second.user_id, second.opened_at), (BOB, 1001))

    def test_account_identity_change_is_a_new_session(self):
        self.arm()
        second = self.arm(bot_id=BOT_ID + 1, now=1001)
        self.assertEqual((second.bot_id, second.opened_at), (BOT_ID + 1, 1001))

    def test_expired_session_is_removed_not_due(self):
        self.arm()
        self.assertEqual(self.tracker.due(1181), ())
        self.assertIsNone(self.tracker.current(GROUP))

    def test_expired_session_cannot_be_claimed(self):
        session = self.arm()
        self.assertFalse(self.tracker.claim(session, 1181))

    def test_capacity_is_bounded(self):
        for index in range(4):
            self.arm(group_id=GROUP + index, now=1000 + index)
        self.assertEqual(len(self.tracker), 3)
        self.assertIsNone(self.tracker.current(GROUP))

    def test_update_cannot_extend_past_original_expiry(self):
        self.arm()
        self.assertIsNone(self.arm(now=1150))
        self.assertIsNone(self.tracker.current(GROUP))


class ReceiptBrain:
    """Scripted certificate issuer, never an actual model or network client."""
    def __init__(self, bot):
        self.bot = bot
        self.calls = []
        self.receipt = None
        self.approve = True
        self.plans = []
        self.certificate_reply_override = None
        self.before_return = None
        self.followup_calls = []
        self.followup_text = '合成模型想出的追问'
        self.followup_hook = None

    async def duel_followup(self, group_id, user_id, user_text, bot_text, *, recent_texts=()):
        self.followup_calls.append(dict(group_id=group_id, user_id=user_id,
            user_text=user_text, bot_text=bot_text, recent_texts=tuple(recent_texts)))
        if self.followup_hook is not None:
            await self.followup_hook()
        return self.followup_text

    async def plan_reply(self, group_id, transcript, speaker, is_at, **kwargs):
        self.calls.append(dict(group_id=group_id, transcript=transcript, **kwargs))
        self.receipt = None
        plan = self.plans.pop(0) if self.plans else ReplyPlan('reply', '脚本回应' + str(len(self.calls)), 'wait')
        turn = self.bot._active_turn.get(group_id)
        if (self.approve and turn and plan.action == 'reply'
                and RETALIATION_MARKER in kwargs.get('tone', '')):
            certified_text = self.certificate_reply_override or plan.text
            self.receipt = (group_id, turn.user_id, str(turn.message_id), turn.message.text, certified_text)
        if self.before_return:
            self.before_return()
        return plan

    def retaliation_confirmed(self, group_id, user_id, message_id, text, delivered_text):
        return self.receipt == (group_id, user_id, str(message_id), text, delivered_text)

    async def reply(self, *args, **kwargs):
        plan = await self.plan_reply(*args, **kwargs)
        return plan.text if plan.action == 'reply' else None

    async def close(self):
        pass


class Client:
    def __init__(self):
        self.self_id = BOT_ID
        self.nickname = '合成机器人'
        self._ready = True
        self.ws = SimpleNamespace(closed=False)
        self.sent = []
        self.ack = 'positive'
        self.error = False
        self.after_send = None

    async def send_action(self, action, params=None, **kwargs):
        if action != 'send_group_msg':
            raise AssertionError('Unexpected voice, metadata, or external action: ' + action)
        self.sent.append(dict(action=action, **(params or {})))
        if self.after_send:
            self.after_send()
        if self.error:
            raise TimeoutError('synthetic send outcome unknown')
        if self.ack == 'positive':
            return {'message_id': 94000 + len(self.sent)}
        if self.ack == 'missing':
            return {}
        return {'message_id': self.ack}


class BrainReceiptTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        cfg = test_config()
        cfg['llm'].update(semantic_pipeline=True, semantic_review=True, semantic_review_policy='all')
        self.brain = Brain(cfg['llm'], cfg['safety'], cfg['persona'], 100, runtime=cfg)
        self.brain._session = SimpleNamespace(close=AsyncMock())
        self.brain._provider_ready = lambda: True
        self.transcript = json.dumps(dict(role='group_member', sender={'id': str(ALICE)},
            message_id='receipt-1', content='我是你爹'), ensure_ascii=False)
        self.calls = []

    async def asyncTearDown(self):
        await self.brain.close()

    async def generate(self, mode='banter'):
        frame = dict(act='rebut', target='bot', meaning='当前群友对机器人进行身份压制式挑衅',
            unresolved='', reply_goal='回应当前说法', mode=mode, needs_clarification=False)
        outputs = [json.dumps(frame, ensure_ascii=False),
                   json.dumps({'action': 'reply', 'text': '脚本回击', 'next': 'wait'}, ensure_ascii=False),
                   '{"ok":true,"issue":""}']
        async def fake_call(messages, temperature, **kwargs):
            self.calls.append(messages)
            if not outputs:
                raise AssertionError('Unexpected additional model phase')
            return outputs.pop(0)
        self.brain._call = fake_call
        return await self.brain.plan_reply(GROUP, self.transcript, '合成群友', True, tone=retaliation_tone())

    async def test_real_brain_certificate_binds_exact_current_reply_tuple(self):
        plan = await self.generate()
        args = [GROUP, ALICE, 'receipt-1', '我是你爹', plan.text]
        self.assertTrue(self.brain.retaliation_confirmed(*args))
        for index, value in enumerate((GROUP + 1, BOB, 'receipt-old', '另一条原话', '另一条回复')):
            changed = list(args)
            changed[index] = value
            self.assertFalse(self.brain.retaliation_confirmed(*changed))
        self.assertEqual(len(self.calls), 3)

    async def test_real_brain_repair_mode_never_certifies_attack(self):
        plan = await self.generate(mode='repair')
        self.assertEqual(plan.action, 'reply')
        self.assertFalse(self.brain.retaliation_confirmed(GROUP, ALICE, 'receipt-1', '我是你爹', plan.text))

    async def test_new_failed_generation_clears_old_certificate(self):
        plan = await self.generate()
        self.assertTrue(self.brain.retaliation_confirmed(GROUP, ALICE, 'receipt-1', '我是你爹', plan.text))
        self.brain._provider_ready = lambda: False
        self.assertIsNone(await self.brain.plan_reply(GROUP, self.transcript, '合成群友', True, tone=retaliation_tone()))
        self.assertFalse(self.brain.retaliation_confirmed(GROUP, ALICE, 'receipt-1', '我是你爹', plan.text))


class DuelIntegrationTests(unittest.IsolatedAsyncioTestCase):
    event = DialogueTests.event
    push = DialogueTests.push
    idle = DialogueTests.idle

    async def asyncSetUp(self):
        cfg = test_config()
        cfg['behavior'].update(model_reply_planning=True, model_burst_max_messages=3,
            max_chain=3, combo_replies=[], merge_delay=.001, merge_at_delay=.001,
            second_reply_delay_min=.001, second_reply_delay_max=.001)
        cfg['voice']['enabled'] = False
        cfg['duel'] = dict(enabled=True, idle_seconds=60, expiry_seconds=180,
            max_groups=256, poll_interval=1)
        self.bot = Bot(cfg)
        await self.bot.brain.close()
        self.brain = ReceiptBrain(self.bot)
        self.client = Client()
        self.bot.brain, self.bot.client = self.brain, self.client
        self.msg_id = 700

    async def asyncTearDown(self):
        await self.bot.close()

    async def say(self, text, **kwargs):
        result = await self.push(text, **kwargs)
        await self.idle()
        return result

    async def arm(self):
        await self.say('我是你爹', at=[BOT_ID])
        session = self.bot._duels.current(GROUP)
        self.assertIsNotNone(session)
        return session

    async def tick(self, session, offset=0):
        await self.bot._duel_tick(session.due_at + offset)

    def followups(self):
        return [message for message in self.client.sent
                if any(part['type'] == 'at' for part in message['message'])]

    async def test_init_does_not_start_background_or_read_database(self):
        self.assertIsNone(self.bot._duel_task)
        self.assertIsNone(self.bot._reminder_store)

    async def test_normal_chat_never_arms_idle_followup(self):
        await self.say('hi', at=[BOT_ID])
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_positive_ack_arms_exact_identity(self):
        session = await self.arm()
        self.assertEqual((session.bot_id, session.group_id, session.user_id), (BOT_ID, GROUP, ALICE))
        self.assertAlmostEqual(session.due_at - session.opened_at, 60, delta=.05)

    async def test_semantic_veto_never_arms_from_preliminary_tone(self):
        self.brain.approve = False
        await self.say('我是你爹', at=[BOT_ID])
        self.assertIn(RETALIATION_MARKER, self.brain.calls[-1]['tone'])
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_custom_brain_without_certificate_does_not_arm(self):
        self.brain.retaliation_confirmed = None
        await self.say('我是你爹', at=[BOT_ID])
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_zero_ack_does_not_arm(self):
        self.client.ack = 0
        await self.say('我是你爹', at=[BOT_ID])
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_missing_ack_does_not_arm(self):
        self.client.ack = 'missing'
        await self.say('我是你爹', at=[BOT_ID])
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_certificate_for_different_reply_cannot_arm(self):
        self.brain.certificate_reply_override = '这是另一条旧模型输出'
        await self.say('我是你爹', at=[BOT_ID])
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_boolean_ack_is_not_positive_message_identity(self):
        self.client.ack = True
        await self.say('我是你爹', at=[BOT_ID])
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_failed_original_reply_never_arms(self):
        self.client.error = True
        await self.say('我是你爹', at=[BOT_ID])
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_exact_599_seconds_is_too_early(self):
        session = await self.arm()
        await self.tick(session, -.1)
        self.assertEqual(len(self.client.sent), 1)

    async def test_sixty_second_followup_is_one_real_at_plus_text(self):
        session = await self.arm()
        await self.tick(session)
        self.assertEqual(len(self.client.sent), 2)
        message = self.client.sent[-1]['message']
        self.assertEqual([s['data']['qq'] for s in message if s['type'] == 'at'], [str(ALICE)])
        self.assertTrue(''.join(s['data']['text'] for s in message if s['type'] == 'text').strip())

    async def test_silence_never_becomes_a_recurring_send_loop(self):
        session = await self.arm()
        await self.tick(session)
        await self.tick(session, 60)
        await self.tick(session, 120)
        self.assertEqual(len(self.client.sent), 2)
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_two_concurrent_ticks_claim_only_one_send(self):
        session = await self.arm()
        await asyncio.gather(self.tick(session), self.tick(session))
        self.assertEqual(len(self.client.sent), 2)
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_idle_followup_uses_one_dedicated_model_call_not_ordinary_dialogue(self):
        session = await self.arm()
        count = len(self.brain.calls)
        await self.tick(session)
        self.assertEqual(len(self.brain.calls), count)
        self.assertEqual(len(self.brain.followup_calls), 1)

    async def test_idle_followup_does_not_hijack_dialogue_state(self):
        session = await self.arm()
        previous = self.bot._dialogues[GROUP]
        before = (previous.user_text, previous.bot_text, previous.updated_at)
        await self.tick(session)
        self.assertIs(self.bot._dialogues[GROUP], previous)
        self.assertEqual((previous.user_text, previous.bot_text, previous.updated_at), before)

    async def test_idle_followup_does_not_renew_friendly_session(self):
        await self.say('hi', at=[BOT_ID])
        stamp = self.bot._friendly_sessions[(GROUP, ALICE)]
        session = await self.arm()
        await self.tick(session)
        self.assertEqual(self.bot._friendly_sessions[(GROUP, ALICE)], stamp)

    async def test_same_user_normal_message_cancels_old_timer(self):
        session = await self.arm()
        await self.say('今天天气不错')
        count = len(self.client.sent)
        await self.tick(session)
        self.assertEqual(len(self.client.sent), count)
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_apology_cancels_old_timer(self):
        session = await self.arm()
        await self.say('对不起，刚才是我嘴臭')
        count = len(self.client.sent)
        await self.tick(session)
        self.assertEqual(len(self.client.sent), count)

    async def test_explicit_stop_cancels_old_timer(self):
        session = await self.arm()
        await self.say('别回我了')
        count = len(self.client.sent)
        await self.tick(session)
        self.assertEqual(len(self.client.sent), count)
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_other_group_member_immediately_cancels_old_timer(self):
        session = await self.arm()
        await self.say('我插句话', user=BOB)
        count = len(self.client.sent)
        await self.tick(session)
        self.assertEqual(len(self.client.sent), count)

    async def test_other_group_activity_does_not_cancel_this_group(self):
        session = await self.arm()
        await self.say('我在另一个群聊天', user=BOB, group=GROUP + 1)
        await self.tick(session)
        self.assertEqual(len(self.followups()), 1)

    async def test_new_hostile_turn_replaces_old_revision(self):
        first = await self.arm()
        await self.say('你这个垃圾', at=[BOT_ID])
        second = self.bot._duels.current(GROUP)
        self.assertIsNotNone(second)
        self.assertIsNot(first, second)
        self.assertGreater(second.revision, first.revision)

    async def test_config_disabled_cancels_without_sending(self):
        session = await self.arm()
        self.bot.cfg['duel']['enabled'] = False
        await self.tick(session)
        self.assertEqual(len(self.client.sent), 1)
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_dry_run_cannot_send_background_followup(self):
        session = await self.arm()
        self.bot.cfg['runtime']['dry_run'] = True
        await self.tick(session)
        self.assertEqual(len(self.client.sent), 1)

    async def test_disconnected_client_cancels_without_delayed_send(self):
        session = await self.arm()
        self.client._ready = False
        await self.tick(session)
        self.assertEqual(len(self.client.sent), 1)
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_changed_account_cancels_old_account_session(self):
        session = await self.arm()
        self.client.self_id = BOT_ID + 1
        await self.tick(session)
        self.assertEqual(len(self.client.sent), 1)
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_same_account_new_connection_cancels_stale_timer(self):
        session = await self.arm()
        self.client.ws = SimpleNamespace(closed=False)
        await self.tick(session)
        self.assertEqual(len(self.client.sent), 1)
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_identity_or_connection_changed_during_original_ack_does_not_arm(self):
        for mode in ('account', 'connection'):
            with self.subTest(mode=mode):
                self.bot._duels.clear()
                self.client.self_id = BOT_ID
                self.client.ws = SimpleNamespace(closed=False)
                if mode == 'account':
                    self.client.after_send = lambda: setattr(self.client, 'self_id', BOT_ID + 1)
                else:
                    self.client.after_send = lambda: setattr(self.client, 'ws', SimpleNamespace(closed=False))
                await self.say('我是你爹', at=[BOT_ID])
                self.assertIsNone(self.bot._duels.current(GROUP))
                self.client.after_send = None

    async def test_owner_changed_during_generation_never_sends_or_arms(self):
        for mode in ('account', 'connection'):
            with self.subTest(mode=mode):
                self.bot._duels.clear()
                self.client.self_id = BOT_ID
                self.client.ws = SimpleNamespace(closed=False)
                if mode == 'account':
                    self.brain.before_return = lambda: setattr(self.client, 'self_id', BOT_ID + 1)
                else:
                    self.brain.before_return = lambda: setattr(self.client, 'ws', SimpleNamespace(closed=False))
                count = len(self.client.sent)
                await self.say('我是你爹', at=[BOT_ID])
                self.assertEqual(len(self.client.sent), count)
                self.assertIsNone(self.bot._duels.current(GROUP))
                self.brain.before_return = None

    async def test_muted_requester_cannot_receive_idle_callout(self):
        session = await self.arm()
        self.bot._muted.add((GROUP, ALICE))
        await self.tick(session)
        self.assertEqual(len(self.client.sent), 1)

    async def test_blacklisted_requester_cannot_receive_idle_callout(self):
        session = await self.arm()
        self.bot._user_blacklist.add(ALICE)
        await self.tick(session)
        self.assertEqual(len(self.client.sent), 1)

    async def test_group_whitelist_change_prevents_idle_callout(self):
        session = await self.arm()
        self.bot._whitelist = {GROUP + 1}
        await self.tick(session)
        self.assertEqual(len(self.client.sent), 1)

    async def test_blocked_output_text_never_sent(self):
        session = await self.arm()
        self.bot.safety['blocked_output'] = ['合成模型']
        await self.tick(session)
        self.assertEqual(len(self.client.sent), 1)

    async def test_group_quota_applies_to_idle_followup(self):
        session = await self.arm()
        self.bot.beh['per_group_per_minute'] = 0
        await self.tick(session)
        self.assertEqual(len(self.client.sent), 1)

    async def test_global_quota_applies_to_idle_followup(self):
        session = await self.arm()
        self.bot.beh['global_per_minute'] = 0
        await self.tick(session)
        self.assertEqual(len(self.client.sent), 1)

    async def test_deferred_quota_can_send_once_before_session_expires(self):
        session = await self.arm()
        self.bot.beh['per_group_per_minute'] = 0
        await self.tick(session)
        self.bot.beh['per_group_per_minute'] = 100
        await self.tick(session, .1)
        await self.tick(session, .2)
        self.assertEqual(len(self.client.sent), 2)

    async def test_original_chain_ceiling_is_not_removed(self):
        session = await self.arm()
        self.bot.beh['max_chain'] = 1
        await self.tick(session)
        self.assertEqual(len(self.client.sent), 1)

    async def test_expired_session_never_sends_late(self):
        session = await self.arm()
        await self.bot._duel_tick(session.expires_at + 1)
        self.assertEqual(len(self.client.sent), 1)
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_uncertain_timeout_after_claim_never_retries(self):
        session = await self.arm()
        self.client.error = True
        await self.tick(session)
        self.client.error = False
        await self.tick(session, 1)
        self.assertEqual(len(self.client.sent), 2)
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_missing_idle_ack_never_retries(self):
        session = await self.arm()
        self.client.ack = 'missing'
        await self.tick(session)
        self.client.ack = 'positive'
        await self.tick(session, 1)
        self.assertEqual(len(self.client.sent), 2)

    async def test_new_activity_during_delivery_gap_cancels_before_send(self):
        session = await self.arm()
        async def interrupted_gap():
            self.bot._note_group_activity(GROUP, BOB)
        self.bot._wait_delivery_gap = interrupted_gap
        await self.tick(session)
        self.assertEqual(len(self.client.sent), 1)

    async def test_account_switch_during_delivery_gap_cancels_before_send(self):
        session = await self.arm()
        async def changed_gap():
            self.client.self_id = BOT_ID + 1
        self.bot._wait_delivery_gap = changed_gap
        await self.tick(session)
        self.assertEqual(len(self.client.sent), 1)

    async def test_identity_change_during_idle_ack_never_indexes_or_retries(self):
        for mode in ('account', 'connection'):
            with self.subTest(mode=mode):
                self.client.self_id = BOT_ID
                self.client.ws = SimpleNamespace(closed=False)
                self.client.after_send = None
                session = await self.arm()
                indexed = set(self.bot._msg_index)
                if mode == 'account':
                    self.client.after_send = lambda: setattr(self.client, 'self_id', BOT_ID + 1)
                else:
                    self.client.after_send = lambda: setattr(self.client, 'ws', SimpleNamespace(closed=False))
                await self.tick(session)
                self.assertEqual(set(self.bot._msg_index), indexed)
                self.assertIsNone(self.bot._duels.current(GROUP))
                self.client.after_send = None
                self.client.self_id = BOT_ID
                count = len(self.client.sent)
                await self.tick(session, 1)
                self.assertEqual(len(self.client.sent), count)

    async def test_active_generation_is_not_mistaken_for_user_silence(self):
        session = await self.arm()
        self.bot._active_turn[GROUP] = SimpleNamespace()
        await self.tick(session)
        self.assertEqual(len(self.client.sent), 1)
        self.bot._active_turn.pop(GROUP)

    async def test_shutdown_clears_ephemeral_sessions(self):
        await self.arm()
        await self.bot.close()
        self.assertEqual(len(self.bot._duels), 0)
        self.assertIsNone(self.bot._duel_task)

    async def test_quote_of_other_person_does_not_arm(self):
        await self.say('他说我是你爹', at=[BOT_ID])
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_direct_third_party_abuse_does_not_arm(self):
        await self.say('你这个垃圾', at=[BOB])
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_mixed_mentions_does_not_assume_bot_is_target(self):
        await self.say('你这个垃圾', at=[BOT_ID, BOB])
        self.assertIsNone(self.bot._duels.current(GROUP))

    async def test_voice_feature_stays_off(self):
        await self.arm()
        self.assertFalse(self.bot.cfg['voice']['enabled'])
        self.assertTrue(all(m['action'] == 'send_group_msg' for m in self.client.sent))


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                               for cls in (TrackerTests, BrainReceiptTests, DuelIntegrationTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  success=result.wasSuccessful(), network='none', real_models=False, real_qq=False,
                  fixtures='synthetic identities, fake OneBot/scripted certificates, manual monotonic deadlines')
    if RESULT:
        RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
