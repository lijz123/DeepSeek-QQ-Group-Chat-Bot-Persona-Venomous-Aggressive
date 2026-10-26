"""Offline file generation, selection, storage, and acknowledged upload tests.

Synthetic prompts and fake model/OneBot only: no private config, external HTTP,
real QQ upload, or execution of generated Python/C code occurs here.
"""
from __future__ import annotations

from layout_support import app_directory, project_directory, configure_test_paths
configure_test_paths()

import asyncio
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = app_directory(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None)
RESULT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
sys.path.insert(0, str(SOURCE))
sys.dont_write_bytecode = True

from file_delivery import (FileRequest, GeneratedFile, FileResult, parse_file_request,
    build_file_messages, parse_file_result, GeneratedFileStore)
from file_delivery_bot import FileDeliveryMixin
from bot import Brain, Bot, OneBotClient, OneBotError
from dialogue_selftest import test_config

PYTHON = '# UTF-8 中文\r\n' + ''.join(f'x_{i} = {i}\r\n' for i in range(60)) + 'print(x_1)\r\n'
C = '#include <stdio.h>\r\nint main(void) {\r\n    puts("hello");\r\n    return 0;\r\n}\r\n'
TEXT = '这是一份真实文字说明。\r\n' * 30
GROUP, USER, OTHER, BOT_ID = 88888, 20002, 20003, 10001


def request(text='肥鱼，用 d2l 写个简单 CNN，做成文件发给我'):
    return parse_file_request(text, addressed=True)


def raw_file(content=PYTHON, name='cnn.py', language='python', text=''):
    return json.dumps({'action': 'files', 'text': text,
        'files': [{'name': name, 'language': language, 'content': content}]}, ensure_ascii=False)


def config():
    value = test_config()
    value['file_delivery'] = {'enabled': True}
    value['output'] = {'unlimited': True, 'daily_max_chars': 100,
        'max_tokens': 393216, 'chunk_chars': 1500}
    value['llm'].update(provider='deepseek', api_key='synthetic-not-a-real-key',
        base_url='https://api.deepseek.com/v1', model='deepseek-chat')
    value['voice']['enabled'] = False
    value['search']['enabled'] = False
    value['safety']['blocked_output'] = []
    value['behavior'].update(min_gap=0, direct_cooldown=0, direct_user_cooldown=0,
        concurrent_reply_window=0, quote_mode='always',
        per_group_per_minute=1000, global_per_minute=1000)
    return value


class RequestTests(unittest.TestCase):
    def test_d2l_infers_python_without_suffix(self):
        value = request()
        self.assertIsInstance(value, FileRequest)
        self.assertEqual(value.extension, 'py')

    def test_pytorch_infers_python(self):
        self.assertEqual(request('用PyTorch写一个分类器，生成文件发我').extension, 'py')

    def test_python_explicit_filename(self):
        value = request('写Python示例，保存成 example.py 发给我')
        self.assertEqual(value.filename, 'example.py')
        self.assertEqual(value.extension, 'py')

    def test_c_language_selects_c(self):
        self.assertEqual(request('帮我写C语言点亮LED程序，作为文件发我').extension, 'c')

    def test_plain_text_selects_txt(self):
        self.assertEqual(request('整理一份学习计划，做成文件发给我').extension, 'txt')

    def test_explicit_txt_wins_for_python_content(self):
        value = request('写一段Python代码，保存成 example.txt 发给我')
        self.assertEqual(value.filename, 'example.txt')
        self.assertEqual(value.extension, 'txt')

    def test_explicit_use_txt_wins_over_python_language(self):
        value=request('帮我写Python脚本，用txt文件发我')
        self.assertEqual(value.extension,'txt')

    def test_unknown_language_uses_text_not_c(self):
        self.assertEqual(request('写JavaScript示例，做成文件发我').extension, 'txt')

    def test_plain_code_request_does_not_force_file(self):
        self.assertIsNone(request('帮我写个简单CNN代码'))

    def test_discussing_file_capability_does_not_upload(self):
        self.assertIsNone(request('你能发送py文件吗'))

    def test_quoted_request_is_not_a_current_command(self):
        self.assertIsNone(request('他说“用d2l写个CNN，做成文件发我”，这是什么意思'))

    def test_cancelled_file_request_is_not_executed(self):
        self.assertIsNone(request('不用生成文件了，取消'))

    def test_existing_private_config_request_not_generated_file(self):
        self.assertIsNone(request('把本机config.json文件发给我'))

    def test_code_language_unknown_keeps_need_for_clarification(self):
        value = request('帮我写一个程序，做成文件发我')
        self.assertIsNotNone(value)
        self.assertEqual(value.extension, '')

    def test_no_file_creation_on_casual_chat(self):
        self.assertIsNone(request('今天晚上天气不错，我们聊聊天'))

    def test_abusive_essay_cannot_bypass_daily_limit_as_file(self):
        self.assertIsNone(request('帮我写骂人小作文，做成文件发我'))

    def test_default_requires_bot_address(self):
        self.assertIsNone(parse_file_request('写个Python程序，做成文件发我'))
        self.assertIsNotNone(parse_file_request('肥鱼，写个Python程序，做成文件发我'))


class ProtocolTests(unittest.TestCase):
    def test_long_python_payload_is_not_chat_truncated(self):
        value = parse_file_result(raw_file(), request())
        self.assertIsInstance(value, FileResult)
        self.assertEqual(value.files[0].content, PYTHON)
        self.assertGreater(len(value.files[0].content), 100)

    def test_utf8_chinese_and_crlf_retained(self):
        value = parse_file_result(raw_file(TEXT, 'notes.txt', 'text'), request('整理一份文字资料，做成文件发我'))
        self.assertEqual(value.files[0].content.encode('utf-8'), TEXT.encode('utf-8'))

    def test_outer_code_fence_removed_without_lone_carriage_return(self):
        inner='x = 1\r\nprint(x)'
        value=parse_file_result(raw_file('```python\r\n'+inner+'\r\n```'),request())
        self.assertIsNotNone(value)
        self.assertEqual(value.files[0].content,inner)
        self.assertFalse(value.files[0].content.endswith('\r'))

    def test_c_payload_accepted(self):
        value = parse_file_result(raw_file(C, 'main.c', 'c'), request('写C语言示例，做成文件发我'))
        self.assertEqual(value.files[0].name, 'main.c')
        self.assertEqual(value.files[0].content, C)

    def test_txt_explicit_format_preserves_python_text(self):
        req = request('写Python程序，保存成 code.txt 发我')
        value = parse_file_result(raw_file(PYTHON, 'code.txt', 'python'), req)
        self.assertIsNotNone(value)
        self.assertEqual(value.files[0].name, 'code.txt')
        self.assertEqual(value.files[0].content, PYTHON)

    def test_invalid_python_does_not_become_successful_file(self):
        self.assertIsNone(parse_file_result(raw_file('def broken(:\n pass\n'), request()))

    def test_python_is_parsed_not_executed(self):
        value = parse_file_result(raw_file('raise RuntimeError("must not run")\n'), request())
        self.assertIsNotNone(value)

    def test_promise_is_not_python_payload(self):
        self.assertIsNone(parse_file_result(raw_file('代码马上来，稍等一下'), request()))

    def test_legitimate_document_can_explain_delayed_reply(self):
        value = parse_file_result(raw_file('当接口稍后返回时，应按回执判断是否成功。', 'notes.txt', 'text'), request('写一份接口说明作为txt文件发我'))
        self.assertIsNotNone(value)
        self.assertEqual(value.files[0].content, '当接口稍后返回时，应按回执判断是否成功。')

    def test_screenshot_style_promise_is_not_actual_text_file(self):
        self.assertIsNone(parse_file_result(raw_file('来了来了，别催，代码这不就端上来了嘛', 'notes.txt', 'text'), request('整理资料作为txt文件发我')))

    def test_cpp_code_is_text_not_c_source(self):
        value = parse_file_result(raw_file('#include <iostream>\nint main(){std::cout << "hi";}\n', 'main.txt', 'cpp'), request('写C++程序做成文件发我'))
        self.assertIsNotNone(value)
        self.assertEqual(value.files[0].name, 'main.txt')

    def test_comments_only_are_not_delivered_code(self):
        self.assertIsNone(parse_file_result(raw_file('# 代码马上来\n# TODO\n'), request()))

    def test_promise_is_not_text_file_deliverable(self):
        self.assertIsNone(parse_file_result(raw_file('代码马上来，稍等一下', 'notes.txt', 'text'), request('整理资料成txt文件发我')))

    def test_empty_payload_rejected(self):
        self.assertIsNone(parse_file_result(raw_file('  \r\n'), request()))

    def test_parent_path_from_model_rejected(self):
        self.assertIsNone(parse_file_result(raw_file(name='../cnn.py'), request()))

    def test_windows_absolute_path_from_model_rejected(self):
        self.assertIsNone(parse_file_result(raw_file(name=r'C:\private\cnn.py'), request()))

    def test_windows_reserved_filename_rejected(self):
        self.assertIsNone(parse_file_result(raw_file(name='CON.py'), request()))

    def test_nonwhitelist_extension_rejected(self):
        self.assertIsNone(parse_file_result(raw_file(name='run.exe'), request()))

    def test_python_not_mislabeled_c(self):
        self.assertIsNone(parse_file_result(raw_file(PYTHON, 'main.c', 'c'), request('写C语言代码，做成文件发我')))

    def test_blocked_output_is_not_hidden_in_file(self):
        self.assertIsNone(parse_file_result(raw_file('# forbidden\nprint(1)\n'), request(), blocked=['forbidden']))

    def test_missing_fields_rejected(self):
        self.assertIsNone(parse_file_result('{"action":"files","files":[]}', request()))

    def test_unknown_action_rejected(self):
        self.assertIsNone(parse_file_result('{"action":"upload_anywhere","text":"","files":[]}', request()))

    def test_clarification_contains_no_files(self):
        value = parse_file_result('{"action":"clarify","text":"你希望用哪种编程语言？","files":[]}', FileRequest('写个程序作为文件发我'))
        self.assertEqual(value.action, 'clarify')
        self.assertEqual(value.files, ())

    def test_notification_is_at_most_hundred_characters(self):
        value = parse_file_result(raw_file(text='说明'*100), request())
        self.assertIsNotNone(value)
        self.assertLessEqual(len(value.text), 100)

    def test_prompt_requires_actual_content_not_future_promise(self):
        messages = build_file_messages(request(), name='大肥鱼', runtime=config())
        content = '\n'.join(x['content'] for x in messages)
        self.assertIn('content', content)
        self.assertIn('language', content)
        self.assertTrue('实际' in content or '完整' in content)
        self.assertTrue('不' in content and ('稍后' in content or '承诺' in content))

    def test_extra_top_level_fields_rejected(self):
        value = json.loads(raw_file()); value['destination'] = 'another-group'
        self.assertIsNone(parse_file_result(json.dumps(value), request()))

    def test_primary_python_can_have_secondary_readme_text(self):
        value = json.loads(raw_file())
        value['files'].append({'name':'README.txt','language':'text','content':'安装依赖后再运行示例；本文件未实际执行。'})
        parsed=parse_file_result(json.dumps(value,ensure_ascii=False),request())
        self.assertIsNotNone(parsed)
        self.assertEqual([item.name for item in parsed.files],['cnn.py','README.txt'])

    def test_nonstring_action_is_invalid_not_exception(self):
        value=json.loads(raw_file());value['action']=[]
        self.assertIsNone(parse_file_result(json.dumps(value),request()))

    def test_duplicate_json_field_is_invalid(self):
        self.assertIsNone(parse_file_result('{"action":"wait","action":"files","text":"","files":[]}',request()))

    def test_repeated_promise_does_not_become_document_by_length(self):
        self.assertIsNone(parse_file_result(raw_file('代码马上来。'*100,'notes.txt','text'),request('写一份说明txt文件发我')))

    def test_three_file_ceiling_is_not_unbounded_upload(self):
        value = json.loads(raw_file())
        value['files'] = [dict(value['files'][0], name=f'file{i}.py') for i in range(4)]
        self.assertIsNone(parse_file_result(json.dumps(value), request()))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'generated'
        self.store = GeneratedFileStore(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_actual_utf8_file_and_hash(self):
        item, = self.store.create((GeneratedFile('sample.py', PYTHON),))
        self.assertEqual(item.path.read_bytes(), PYTHON.encode('utf-8'))
        self.assertEqual(item.sha256, hashlib.sha256(PYTHON.encode('utf-8')).hexdigest())
        self.assertEqual(item.size, len(PYTHON.encode('utf-8')))
        self.assertTrue(self.store.verify(item))

    def test_files_are_confined_to_generated_root(self):
        item, = self.store.create((GeneratedFile('sample.py', PYTHON),))
        self.assertTrue(item.path.resolve().is_relative_to(self.root.resolve()))
        self.assertNotEqual(item.path.parent, self.root)

    def test_repeated_filename_uses_separate_task_directories(self):
        first, = self.store.create((GeneratedFile('same.txt', TEXT),))
        second, = self.store.create((GeneratedFile('same.txt', 'different'),))
        self.assertNotEqual(first.path, second.path)
        self.assertEqual(first.path.read_text(encoding='utf-8'), TEXT.replace('\r\n','\n'))
        self.assertEqual(second.path.read_text(encoding='utf-8'), 'different')

    def test_tampered_file_fails_verification(self):
        item, = self.store.create((GeneratedFile('same.txt', TEXT),))
        item.path.write_text('tampered', encoding='utf-8')
        self.assertFalse(self.store.verify(item))

    def test_external_path_cannot_be_verified(self):
        item, = self.store.create((GeneratedFile('same.txt', TEXT),))
        outside = Path(self.temp.name) / 'private.txt'; outside.write_text('secret', encoding='utf-8')
        forged = replace(item, path=outside)
        self.assertFalse(self.store.verify(forged))

    def test_create_rejects_traversal(self):
        with self.assertRaises((ValueError, OSError, TypeError)):
            self.store.create((GeneratedFile('../private.txt', TEXT),))
        self.assertFalse((Path(self.temp.name) / 'private.txt').exists())

    def test_create_rejects_reserved_filename(self):
        with self.assertRaises((ValueError, OSError, TypeError)):
            self.store.create((GeneratedFile('NUL.txt', TEXT),))

    def test_store_rejects_none_instead_of_iterating_host_state(self):
        with self.assertRaises((ValueError,TypeError)): self.store.create(None)

    def test_store_rejects_payload_larger_than_safety_buffer(self):
        import file_delivery
        with self.assertRaises(ValueError):
            self.store.create((GeneratedFile('oversize.txt','x'*(file_delivery.MAX_CONTENT_BYTES+1)),))

    def test_cleanup_removes_only_known_generated_file(self):
        item, = self.store.create((GeneratedFile('same.txt', TEXT),))
        unknown = item.path.parent / 'unknown.txt'; unknown.write_text('retain', encoding='utf-8')
        self.store.cleanup((item,))
        self.assertFalse(item.path.exists())
        self.assertEqual(unknown.read_text(encoding='utf-8'), 'retain')

    def test_cleanup_does_not_delete_external_forged_path(self):
        item, = self.store.create((GeneratedFile('same.txt', TEXT),))
        outside = Path(self.temp.name) / 'private.txt'; outside.write_text('retain', encoding='utf-8')
        self.store.cleanup((replace(item, path=outside),))
        self.assertEqual(outside.read_text(encoding='utf-8'), 'retain')
        self.assertTrue(item.path.exists())


class BrainTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.cfg = config()
        self.brain = Brain(self.cfg['llm'], self.cfg['safety'], self.cfg['persona'], 120, runtime=self.cfg)
        self.brain._session = object()
        self.calls = []

    async def asyncTearDown(self):
        self.brain._session = None

    def script(self, responses):
        replies = iter(responses)
        async def fake(messages, temperature, **kwargs):
            self.calls.append(dict(messages=copy.deepcopy(messages), temperature=temperature, **kwargs))
            self.brain._llm_transport_failed = False
            return next(replies, None)
        self.brain._call = fake

    async def test_successful_file_uses_one_existing_generation_not_classifier(self):
        self.script([raw_file()])
        result = await self.brain.generate_files(request())
        self.assertEqual(result.files[0].content, PYTHON)
        self.assertEqual(len(self.calls), 1)

    async def test_file_generation_keeps_complete_task_budget(self):
        self.script([raw_file()])
        await self.brain.generate_files(request())
        self.assertEqual(self.calls[0]['token_budget'], 393216)
        self.assertTrue(self.calls[0]['wide_generation'])

    async def test_promise_is_internally_rewritten_before_delivery(self):
        self.script([raw_file('代码马上来'), raw_file()])
        result = await self.brain.generate_files(request())
        self.assertEqual(result.files[0].content, PYTHON)
        self.assertEqual(len(self.calls), 2)

    async def test_second_invalid_response_stops_not_infinite_generation(self):
        self.script([raw_file('代码马上来'), raw_file('代码稍后给')])
        self.assertIsNone(await self.brain.generate_files(request()))
        self.assertEqual(len(self.calls), 2)

    async def test_empty_transport_response_is_not_expensively_retried(self):
        self.script([None, raw_file()])
        self.assertIsNone(await self.brain.generate_files(request()))
        self.assertEqual(len(self.calls), 1)

    async def test_clarification_can_return_without_upload_payload(self):
        self.script(['{"action":"clarify","text":"你希望使用哪种编程语言？","files":[]}'])
        result = await self.brain.generate_files(FileRequest('写个程序作为文件发我'))
        self.assertEqual(result.action, 'clarify')
        self.assertFalse(result.files)


class ActionProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def make_client(self, status='ok', retcode=0, include_retcode=True):
        client = OneBotClient({'request_timeout': 1}, lambda event: None)
        async def send_json(data):
            response = {'status': status, 'echo': data['echo'], 'data': {'file_id': 'offline-id'}}
            if include_retcode: response['retcode'] = retcode
            client._dispatch_raw(json.dumps(response))
        client.ws = SimpleNamespace(closed=False, send_json=send_json)
        return client

    async def test_sync_upload_ack_accepted(self):
        client = await self.make_client()
        self.assertEqual(await client.send_action('upload_group_file', {}, require_sync=True), {'file_id':'offline-id'})

    async def test_async_ack_is_not_confirmed_delivery(self):
        client = await self.make_client(status='async')
        with self.assertRaises(OneBotError):
            await client.send_action('upload_group_file', {}, require_sync=True)

    async def test_missing_retcode_not_confirmed_delivery(self):
        client = await self.make_client(include_retcode=False)
        with self.assertRaises(OneBotError):
            await client.send_action('upload_group_file', {}, require_sync=True)

    async def test_boolean_retcode_not_confirmed_delivery(self):
        client = await self.make_client(retcode=False)
        with self.assertRaises(OneBotError):
            await client.send_action('upload_group_file', {}, require_sync=True)

    async def test_legacy_action_semantics_remain_compatible(self):
        client = await self.make_client(status='async', include_retcode=False)
        self.assertEqual(await client.send_action('old_action', {}), {'file_id':'offline-id'})


class FileBrain:
    def __init__(self, result=None):
        self.result = result if result is not None else FileResult('files', '', (GeneratedFile('cnn.py', PYTHON),))
        self.calls = []
        self.started = asyncio.Event()
        self.release = None
        self.after_generate = None

    async def generate_files(self, req, **kwargs):
        self.calls.append((req, kwargs))
        self.started.set()
        if self.release is not None: await self.release.wait()
        if self.after_generate is not None: self.after_generate()
        return self.result


class UploadClient:
    def __init__(self, events):
        self.self_id = BOT_ID
        self.nickname = '离线文件机器人'
        self._ready = True
        self.ws = SimpleNamespace(closed=False)
        self.events = events
        self.uploads = []
        self.responses = []
        self.after_upload = None

    async def send_action(self, action, params=None, **kwargs):
        if action != 'upload_group_file': raise AssertionError('Unexpected external action')
        if kwargs.get('require_sync') is not True: raise AssertionError('Upload must require synchronous ACK')
        params = dict(params or {})
        item = dict(params, body=Path(params['file']).read_bytes(), kwargs=kwargs)
        self.uploads.append(item)
        self.events.append(('upload', item))
        response = self.responses.pop(0) if self.responses else {'file_id':'offline-file-id'}
        if self.after_upload is not None: self.after_upload()
        if isinstance(response, BaseException): raise response
        return response


class FakeLimiter:
    def __init__(self): self.allowed = True; self.consumes = 0
    def allow_many(self, limits, consume=True):
        if self.allowed and consume: self.consumes += 1
        return self.allowed


class UploadBot(FileDeliveryMixin):
    def __init__(self):
        self.cfg = config(); self.safety = self.cfg['safety']
        self._closing = False
        self._active_turn = {}; self._delivery_owners = {}; self._turn_tones = {}
        self._user_blacklist = set()
        self.allowed = True; self.muted = False; self.superseded = False; self.elsewhere = False
        self._last_group = {}; self._last_user = {}; self._last_any = 0.; self._last_awake = 0.
        self.events = []; self.feedback = []; self.receipts = []
        self.client = UploadClient(self.events)
        self.brain = FileBrain()
        self.limiter = FakeLimiter(); self._delivery_lock = asyncio.Lock()
        self.gap_hook = None

    def turn(self, group=GROUP, user=USER, message_id='file-source'):
        turn = SimpleNamespace(group_id=group,user_id=user,message_id=message_id,
            message=SimpleNamespace(text='肥鱼，用d2l写一个CNN，做成文件发我'),
            arrival_revision=1,target_revision=0,source_ids=(message_id,),is_poke=False)
        self._active_turn[group] = turn
        self._delivery_owners[group] = (turn,self.client.self_id,self.client.ws)
        return turn

    def _group_allowed(self, group): return self.allowed
    def _is_muted(self, group, user): return self.muted
    def _superseded(self, turn): return self.superseded
    def _addressed_elsewhere(self, turn): return self.elsewhere
    def _owner_chat_request(self, turn): return object()
    def _second_reply_limits(self, group): return [('group', group)]
    async def _wait_delivery_gap(self):
        if self.gap_hook is not None: self.gap_hook()
    def _record_delivery(self, group, user, text, message_id, reason):
        self.receipts.append(dict(group=group,user=user,text=text,reason=reason))
        self.events.append(('receipt',self.receipts[-1]))
    async def _send(self, group, text, message_id, user, reason):
        self.feedback.append(dict(group=group,user=user,text=text,message_id=message_id,reason=reason))
        self.events.append(('feedback',self.feedback[-1]))
        return True


class UploadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.generated = Path(self.temp.name) / 'generated'
        real_store = GeneratedFileStore
        self.store_patch = patch('file_delivery_bot.GeneratedFileStore', side_effect=lambda ignored:real_store(self.generated))
        self.store_patch.start()
        self.bot = UploadBot(); self.turn = self.bot.turn()

    async def asyncTearDown(self):
        self.store_patch.stop(); self.temp.cleanup()

    async def handle(self):
        await self.bot._handle_file_request(self.turn, request())

    def successful_texts(self):
        return [x['text'] for x in self.bot.feedback if '已上传 ' in x['text']]

    async def test_uploads_real_long_file_to_current_group(self):
        await self.handle()
        self.assertEqual(len(self.bot.client.uploads), 1)
        sent = self.bot.client.uploads[0]
        self.assertEqual(sent['group_id'], GROUP)
        self.assertEqual(sent['name'], 'cnn.py')
        self.assertEqual(sent['body'], PYTHON.encode('utf-8'))
        self.assertTrue(Path(sent['file']).is_absolute())

    async def test_success_feedback_occurs_after_positive_upload_ack(self):
        await self.handle()
        self.assertTrue(self.successful_texts())
        self.assertEqual([name for name,_ in self.bot.events], ['upload','receipt','feedback'])
        self.assertEqual(self.bot.feedback[0]['user'], USER)
        self.assertEqual(self.bot.feedback[0]['message_id'], self.turn.message_id)

    async def test_upload_receipt_stores_metadata_not_long_code(self):
        await self.handle()
        self.assertEqual(len(self.bot.receipts), 1)
        self.assertEqual(self.bot.receipts[0]['reason'], 'file_upload')
        self.assertIn('cnn.py', self.bot.receipts[0]['text'])
        self.assertNotIn(PYTHON, self.bot.receipts[0]['text'])

    async def test_short_confirmation_is_at_most_hundred(self):
        await self.handle()
        self.assertTrue(all(len(item['text'])<=100 for item in self.bot.feedback))
        self.assertTrue(all(item['reason']=='file' for item in self.bot.feedback))

    async def test_successfully_consumed_file_can_be_cleaned(self):
        await self.handle()
        self.assertFalse(Path(self.bot.client.uploads[0]['file']).exists())

    async def test_empty_ack_not_success_and_not_retried(self):
        self.bot.client.responses = [{}]
        await self.handle()
        self.assertEqual(len(self.bot.client.uploads), 1)
        self.assertFalse(self.successful_texts()); self.assertFalse(self.bot.receipts)
        self.assertTrue(Path(self.bot.client.uploads[0]['file']).exists())
        self.assertTrue(any('不确定' in item['text'] for item in self.bot.feedback))

    async def test_timeout_not_success_and_source_retained(self):
        self.bot.client.responses = [asyncio.TimeoutError()]
        await self.handle()
        self.assertEqual(len(self.bot.client.uploads), 1)
        self.assertFalse(self.successful_texts())
        self.assertTrue(Path(self.bot.client.uploads[0]['file']).exists())

    async def test_error_not_blindly_retried(self):
        self.bot.client.responses = [OneBotError('synthetic async ACK rejected')]
        await self.handle()
        self.assertEqual(len(self.bot.client.uploads), 1)
        self.assertFalse(self.successful_texts())

    async def test_invalid_boolean_file_id_not_success(self):
        self.bot.client.responses = [{'file_id':True}]
        await self.handle()
        self.assertFalse(self.successful_texts()); self.assertFalse(self.bot.receipts)

    async def test_disabled_feature_does_not_generate_or_upload(self):
        self.bot.cfg['file_delivery']['enabled'] = False
        await self.handle()
        self.assertFalse(self.bot.brain.calls); self.assertFalse(self.bot.client.uploads)

    async def test_dry_run_does_not_generate_write_or_upload(self):
        self.bot.cfg['runtime']['dry_run'] = True
        await self.handle()
        self.assertFalse(self.bot.brain.calls); self.assertFalse(self.bot.client.uploads)
        self.assertFalse(self.generated.exists())

    async def test_blacklisted_user_cannot_upload(self):
        self.bot._user_blacklist.add(USER)
        await self.handle()
        self.assertFalse(self.bot.brain.calls); self.assertFalse(self.bot.client.uploads)

    async def test_disallowed_group_cannot_upload(self):
        self.bot.allowed = False
        await self.handle()
        self.assertFalse(self.bot.client.uploads)

    async def test_muted_conversation_cannot_upload(self):
        self.bot.muted = True
        await self.handle()
        self.assertFalse(self.bot.client.uploads)

    async def test_closing_bot_cannot_upload(self):
        self.bot._closing = True
        await self.handle()
        self.assertFalse(self.bot.client.uploads)

    async def test_elsewhere_addressed_input_cannot_upload(self):
        self.bot.elsewhere = True
        await self.handle()
        self.assertFalse(self.bot.client.uploads)

    async def test_blocked_current_input_cannot_upload(self):
        self.bot.safety['blocked_input'] = ['d2l']
        await self.handle()
        self.assertFalse(self.bot.client.uploads)

    async def test_quota_exhaustion_does_not_bypass_send_limit(self):
        self.bot.limiter.allowed = False
        with patch('file_delivery_bot.FILE_QUOTA_WAIT_SECONDS', 0): await self.handle()
        self.assertFalse(self.bot.client.uploads)
        self.assertFalse(self.successful_texts())

    async def test_each_upload_consumes_existing_quota(self):
        self.bot.brain.result = FileResult('files','',(GeneratedFile('first.txt',TEXT),GeneratedFile('second.txt','actual second file')))
        await self.handle()
        self.assertEqual(len(self.bot.client.uploads), 2)
        self.assertEqual(self.bot.limiter.consumes, 2)

    async def test_partial_failure_only_confirms_successful_count(self):
        self.bot.brain.result = FileResult('files','',(GeneratedFile('first.txt',TEXT),GeneratedFile('second.txt','actual second file')))
        self.bot.client.responses = [{'file_id':'first-id'},{}]
        await self.handle()
        self.assertEqual(len(self.bot.client.uploads), 2)
        self.assertEqual(len(self.bot.receipts), 1)
        self.assertTrue(any('1/2' in x['text'] for x in self.bot.feedback))
        self.assertFalse(any('已上传 first.txt、second.txt' in x['text'] for x in self.bot.feedback))

    async def test_new_source_during_generation_cancels_old_upload(self):
        self.bot.brain.after_generate = lambda:self.bot.turn(user=OTHER,message_id='new-source')
        await self.handle()
        self.assertFalse(self.bot.client.uploads); self.assertFalse(self.bot.feedback)

    async def test_account_change_before_upload_cancels(self):
        self.bot.brain.after_generate = lambda:setattr(self.bot.client,'self_id',BOT_ID+1)
        await self.handle()
        self.assertFalse(self.bot.client.uploads)

    async def test_connection_change_before_upload_cancels(self):
        self.bot.brain.after_generate = lambda:setattr(self.bot.client,'ws',SimpleNamespace(closed=False))
        await self.handle()
        self.assertFalse(self.bot.client.uploads)

    async def test_new_input_while_waiting_for_delivery_lock_cancels(self):
        self.bot.gap_hook = lambda:self.bot.turn(user=OTHER,message_id='lock-change')
        await self.handle()
        self.assertFalse(self.bot.client.uploads)

    async def test_source_change_after_ack_does_not_notify_new_member(self):
        self.bot.client.after_upload = lambda:self.bot.turn(user=OTHER,message_id='after-ack')
        await self.handle()
        self.assertEqual(len(self.bot.client.uploads), 1)
        self.assertFalse(self.bot.feedback)
        self.assertFalse(self.bot.receipts)

    async def test_two_groups_keep_own_file_destination(self):
        other = self.bot.turn(group=GROUP+1,user=OTHER,message_id='other-group')
        await asyncio.gather(self.handle(),self.bot._handle_file_request(other,request()))
        self.assertEqual(sorted(item['group_id'] for item in self.bot.client.uploads),[GROUP,GROUP+1])
        self.assertEqual({(x['group'],x['user']) for x in self.bot.feedback},{(GROUP,USER),(GROUP+1,OTHER)})

    async def test_clarification_no_file_created_or_upload_claimed(self):
        self.bot.brain.result = FileResult('clarify','你要哪种编程语言？',())
        await self.handle()
        self.assertFalse(self.bot.client.uploads); self.assertFalse(self.generated.exists())
        self.assertEqual(self.bot.feedback[0]['text'],'你要哪种编程语言？')

    async def test_fake_clarification_cannot_claim_already_uploaded(self):
        self.bot.brain.result = FileResult('clarify','文件已发送成功',())
        await self.handle()
        self.assertFalse(self.bot.client.uploads)
        self.assertNotEqual(self.bot.feedback[0]['text'],'文件已发送成功')

    async def test_no_valid_generation_never_claims_success(self):
        self.bot.brain.result = None
        await self.handle()
        self.assertFalse(self.bot.client.uploads); self.assertFalse(self.successful_texts())


class QueueFileBrain(FileBrain):
    def __init__(self, result=None):
        super().__init__(result); self.plans = []
    async def close(self): pass
    async def plan_reply(self, *args, **kwargs):
        from persona import ReplyPlan
        self.plans.append((args,kwargs))
        return ReplyPlan('reply','普通正文回复','wait')


class QueueFileClient(UploadClient):
    def __init__(self, events):
        super().__init__(events); self.sent=[]; self.next_id=90000
    async def stop(self): pass
    async def send_action(self, action, params=None, **kwargs):
        if action=='upload_group_file':
            return await super().send_action(action,params,**kwargs)
        if action=='send_group_msg':
            self.next_id+=1
            self.sent.append(dict(params or {}))
            self.events.append(('send_group_msg',self.sent[-1]))
            return {'message_id':self.next_id}
        raise AssertionError('Unexpected real transport action '+action)


class ActualBotRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.generated=Path(self.temp.name)/'generated'
        real_store=GeneratedFileStore
        self.store_patch=patch('file_delivery_bot.GeneratedFileStore',side_effect=lambda ignored:real_store(self.generated))
        self.store_patch.start()
        self.cfg=config(); self.bot=Bot(self.cfg)
        await self.bot.brain.close()
        self.brain=QueueFileBrain(); self.events=[]; self.client=QueueFileClient(self.events)
        self.bot.brain=self.brain; self.bot.client=self.client
        self.ident=1000

    async def asyncTearDown(self):
        await self.bot.close()
        self.store_patch.stop(); self.temp.cleanup()

    async def push(self,text,user=USER,group=GROUP,at_bot=True,reply=None):
        self.ident+=1
        message=[]
        if reply is not None: message.append({'type':'reply','data':{'id':str(reply)}})
        if at_bot: message.append({'type':'at','data':{'qq':str(BOT_ID)}})
        message.append({'type':'text','data':{'text':text}})
        await self.bot.on_event({'post_type':'message','message_type':'group','group_id':group,
            'user_id':user,'self_id':BOT_ID,'message_id':self.ident,
            'sender':{'nickname':'合成成员'+str(user)},'message':message})
        return str(self.ident)

    async def idle(self): await asyncio.wait_for(self.bot.wait_idle(),5)

    def bodies(self):
        return [''.join(str(x.get('data',{}).get('text','')) for x in item.get('message',[]) if x.get('type')=='text') for item in self.client.sent]

    async def test_full_event_to_upload_and_native_quoted_confirmation(self):
        mid=await self.push('用d2l写一个简单CNN，做成文件发给我')
        await self.idle()
        self.assertEqual(len(self.client.uploads),1)
        self.assertEqual(self.client.uploads[0]['body'],PYTHON.encode('utf-8'))
        self.assertEqual(len(self.brain.calls),1)
        self.assertFalse(self.brain.plans)
        self.assertTrue(any('已上传' in body for body in self.bodies()))
        quote=[seg['data']['id'] for seg in self.client.sent[0]['message'] if seg['type']=='reply']
        self.assertEqual(quote,[mid])
        self.assertTrue(all(len(body)<=100 for body in self.bodies()))

    async def test_plain_code_still_goes_to_normal_body_path(self):
        await self.push('帮我写个简单CNN代码')
        await self.idle()
        self.assertFalse(self.client.uploads); self.assertFalse(self.brain.calls)
        self.assertEqual(len(self.brain.plans),1)
        self.assertIn('普通正文回复',self.bodies())

    async def test_actual_bot_rejects_other_person_reference_as_file_request(self):
        await self.push('用d2l写CNN文件发我',at_bot=False,reply='unknown-other-source')
        await self.idle()
        self.assertFalse(self.client.uploads); self.assertFalse(self.brain.calls)

    async def test_actual_bot_feature_off_preserves_old_reply_path(self):
        self.bot.cfg['file_delivery']['enabled']=False
        await self.push('用d2l写一个CNN，做成文件发我')
        await self.idle()
        self.assertFalse(self.client.uploads); self.assertFalse(self.brain.calls)
        self.assertTrue(self.brain.plans)

    async def test_actual_bot_local_blacklist_blocks_file_request(self):
        self.bot._user_blacklist.add(USER)
        await self.push('用d2l写一个CNN，做成文件发我')
        await self.idle()
        self.assertFalse(self.client.uploads); self.assertFalse(self.brain.calls)

    async def test_actual_bot_dryrun_never_creates_generated_file(self):
        self.bot.cfg['runtime']['dry_run']=True
        await self.push('用d2l写一个CNN，做成文件发我')
        await self.idle()
        self.assertFalse(self.client.uploads); self.assertFalse(self.generated.exists())

    async def begin_clarification(self):
        self.brain.result=FileResult('clarify','你希望使用哪种编程语言？',())
        await self.push('帮我写一个程序，做成文件发我')
        await self.idle()
        self.assertFalse(self.client.uploads)
        self.assertEqual(len(self.brain.calls),1)
        self.brain.result=FileResult('files','',(GeneratedFile('cnn.py',PYTHON),))

    async def test_same_member_language_answer_resumes_pending_file_task(self):
        await self.begin_clarification()
        await self.push('Python',at_bot=False)
        await self.idle()
        self.assertEqual(len(self.brain.calls),2)
        self.assertEqual(len(self.client.uploads),1)
        self.assertEqual(self.brain.calls[-1][0].extension,'py')
        self.assertIn('程序',self.brain.calls[-1][0].instruction)
        self.assertEqual(self.client.uploads[0]['body'],PYTHON.encode('utf-8'))

    async def test_native_quote_of_current_bot_question_resumes_pending_task(self):
        await self.begin_clarification()
        question_id=str(self.client.next_id)
        self.assertIn((GROUP,question_id),self.bot._msg_index)
        await self.push('Python',at_bot=False,reply=question_id)
        await self.idle()
        self.assertEqual(len(self.brain.calls),2)
        self.assertEqual(len(self.client.uploads),1)
        self.assertEqual(self.brain.calls[-1][0].extension,'py')

    async def test_native_quote_of_old_unrelated_bot_message_does_not_resume(self):
        await self.begin_clarification()
        from dialogue import IndexedMessage
        self.bot._msg_index[(GROUP,'old-bot-message')]=IndexedMessage(GROUP,BOT_ID,'离线机器人','以前的无关问候',time.time()-30,True)
        await self.push('Python',at_bot=False,reply='old-bot-message')
        await self.idle()
        self.assertEqual(len(self.brain.calls),1)
        self.assertFalse(self.client.uploads)

    async def test_native_quote_of_other_member_does_not_resume(self):
        await self.begin_clarification()
        from dialogue import IndexedMessage
        self.bot._msg_index[(GROUP,'other-member-message')]=IndexedMessage(GROUP,OTHER,'另一合成成员','另一人的问题',time.time()-20,False)
        await self.push('Python',at_bot=False,reply='other-member-message')
        await self.idle()
        self.assertEqual(len(self.brain.calls),1)
        self.assertFalse(self.client.uploads)

    async def test_another_member_language_answer_cannot_take_pending_task(self):
        await self.begin_clarification()
        await self.push('Python',user=OTHER,at_bot=True)
        await self.idle()
        self.assertEqual(len(self.brain.calls),1)
        self.assertFalse(self.client.uploads)

    async def test_cancel_discards_pending_file_task(self):
        await self.begin_clarification()
        await self.push('算了，不用了',at_bot=False)
        await self.idle()
        await self.push('Python',at_bot=True)
        await self.idle()
        self.assertEqual(len(self.brain.calls),1)
        self.assertFalse(self.client.uploads)

    async def test_new_topic_discards_pending_file_task(self):
        await self.begin_clarification()
        await self.push('今天晚饭吃啥',at_bot=False)
        await self.idle()
        await self.push('Python',at_bot=True)
        await self.idle()
        self.assertEqual(len(self.brain.calls),1)
        self.assertFalse(self.client.uploads)

    async def test_expired_pending_task_not_resumed(self):
        await self.begin_clarification()
        from dialogue import PendingTurn
        from bot import Msg
        turn=PendingTurn(GROUP,USER,'expired-answer',Msg('合成成员','Python',False,time.time(),sender_id=str(USER)),
            True,(),'', '',time.monotonic(),time.monotonic(),('expired-answer',))
        advanced=time.monotonic()+121
        with patch('file_delivery_bot.time.monotonic',return_value=advanced):
            self.assertIsNone(self.bot._resume_file_request(turn))
        self.assertFalse(self.client.uploads)

    async def test_two_members_in_one_second_get_own_quoted_receipts(self):
        self.bot.beh['concurrent_reply_window']=1.0
        self.bot.beh['merge_delay']=0; self.bot.beh['merge_at_delay']=0
        a=await self.push('用d2l写个CNN文件发我',user=USER)
        b=await self.push('用Python写个程序文件发我',user=OTHER)
        await self.idle()
        self.assertEqual(len(self.client.uploads),2)
        quotes=[[seg['data']['id'] for seg in item['message'] if seg['type']=='reply'] for item in self.client.sent]
        self.assertEqual(quotes,[[a],[b]])
        self.assertEqual(len(self.brain.calls),2)


class LinkBoundaryTests(unittest.TestCase):
    def setUp(self): self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def symlink(self,path,target,**kwargs):
        try: path.symlink_to(target,**kwargs)
        except (OSError,NotImplementedError) as exc: self.skipTest('Local OS cannot create test symlink: '+type(exc).__name__)

    def test_generated_root_symlink_is_rejected(self):
        actual=self.root/'actual';actual.mkdir()
        sentinel=actual/'private.txt';sentinel.write_text('untouched',encoding='utf-8')
        linked=self.root/'linked';self.symlink(linked,actual,target_is_directory=True)
        with self.assertRaises((ValueError,OSError)): GeneratedFileStore(linked)
        self.assertEqual(sentinel.read_text(encoding='utf-8'),'untouched')

    def test_generated_file_replaced_by_link_is_not_uploaded_or_deleted(self):
        store=GeneratedFileStore(self.root/'generated')
        item,=store.create((GeneratedFile('sample.txt',TEXT),))
        outside=self.root/'private.txt';outside.write_text('untouched',encoding='utf-8')
        item.path.unlink();self.symlink(item.path,outside)
        self.assertFalse(store.verify(item))
        store.cleanup((item,))
        self.assertTrue(item.path.is_symlink())
        self.assertEqual(outside.read_text(encoding='utf-8'),'untouched')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        success=result.wasSuccessful(), skipped=len(result.skipped), real_model_requests=0, real_QQ_messages=0,
        real_external_network_requests=0)
    if RESULT: RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('RESULT_JSON', RESULT)
    raise SystemExit(0 if result.wasSuccessful() else 1)
