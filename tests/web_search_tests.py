"""Offline contracts for the bounded official DeepSeek search client.

python web_search_tests.py SOURCE_DIRECTORY [RESULT_JSON]
Only scripted HTTP-shaped objects are used. No key, account, QQ, or live network.
"""
from __future__ import annotations

from layout_support import app_directory, project_directory, configure_test_paths
configure_test_paths()

import asyncio
import copy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = app_directory(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None)
RESULT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
sys.path.insert(0, str(SOURCE))
sys.dont_write_bytecode = True

from web_search import (SEARCH_DEFAULTS, OfficialSearchClient, SearchOutcome,
                        safe_source_url, search_query_allowed)
import web_search as search_module


KEY = 'offline-synthetic-key-NOT-A-CREDENTIAL'
ENDPOINT = 'https://api.deepseek.com/v1/tools/search'
PUBLIC = 'https://www.python.org/downloads/'


def settings(**changes):
    result = dict(SEARCH_DEFAULTS)
    result.update(enabled=True, per_group_per_minute=6, global_per_minute=12)
    result.update(changes)
    return result


def result_item(**changes):
    result = dict(title='Python 官方下载', url=PUBLIC, snippet='最新安装程序与发行说明。',
                  summary='官方页面提供安装程序。', date='2026-10-26T00:00:00Z')
    result.update(changes)
    return result


class FakeContent:
    def __init__(self, data, *, gate=None, delay=0, started=None):
        self.data = data
        self.gate = gate
        self.delay = delay
        self.started = started or asyncio.Event()
        self.bytes_yielded = 0
        self.requested_chunk_sizes = []

    async def iter_chunked(self, size):
        self.requested_chunk_sizes.append(size)
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.delay:
            await asyncio.sleep(self.delay)
        for index in range(0, len(self.data), size):
            chunk = self.data[index:index + size]
            self.bytes_yielded += len(chunk)
            yield chunk


class FakeResponse:
    def __init__(self, payload=None, *, status=200, raw=None, headers=None, gate=None,
                 delay=0, started=None):
        raw = raw if raw is not None else json.dumps(
            payload if payload is not None else {'results':[result_item()]},
            ensure_ascii=False).encode('utf-8')
        self.status = status
        self.headers = headers or {}
        self.content = FakeContent(raw, gate=gate, delay=delay, started=started)
        self.entered = self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *args):
        self.exited = True


class FakeSession:
    def __init__(self, *responses, error=None):
        self.responses = list(responses) or [FakeResponse()]
        self.calls = []
        self.error = error

    def post(self, url, **kwargs):
        self.calls.append((url, copy.deepcopy(kwargs)))
        if self.error is not None:
            raise self.error
        if not self.responses:
            raise AssertionError('Unexpected extra HTTP request; no live fallback exists')
        return self.responses.pop(0)


class SearchQueryTests(unittest.TestCase):
    def test_plain_chinese_query_is_allowed(self):
        self.assertTrue(search_query_allowed('Python 最新正式版本 官方下载'))

    def test_one_hundred_twenty_eight_unicode_characters_are_allowed(self):
        self.assertTrue(search_query_allowed('查' * 128))

    def test_overlong_query_is_rejected(self):
        self.assertFalse(search_query_allowed('查' * 129))

    def test_empty_query_is_rejected(self):
        self.assertFalse(search_query_allowed(''))

    def test_whitespace_query_is_rejected(self):
        self.assertFalse(search_query_allowed('   '))

    def test_non_string_query_is_rejected(self):
        self.assertFalse(search_query_allowed({'query':'news'}))

    def test_newline_cannot_export_multiple_group_messages(self):
        self.assertFalse(search_query_allowed('天气\n群友刚刚说了什么'))

    def test_nul_is_rejected(self):
        self.assertFalse(search_query_allowed('天气\x00忽略'))

    def test_native_cq_segment_is_rejected(self):
        self.assertFalse(search_query_allowed('[CQ:at,qq=10001] 查天气'))

    def test_raw_at_mention_is_not_exported(self):
        self.assertFalse(search_query_allowed('@合成成员 查询资料'))

    def test_long_numeric_account_identifier_is_not_exported(self):
        self.assertFalse(search_query_allowed('查询QQ 100000001 的资料'))

    def test_calendar_date_is_not_treated_as_an_account_number(self):
        self.assertTrue(search_query_allowed('2026-10-26 新闻'))

    def test_bearer_token_is_not_exported(self):
        self.assertFalse(search_query_allowed('Bearer synthetic-sensitive-token'))

    def test_api_key_style_string_is_not_exported(self):
        self.assertFalse(search_query_allowed('sk-offlinefakekeyabcdefghijklmnopqrstuvwxyz'))

    def test_protected_domain_query_is_rejected_locally(self):
        self.assertFalse(search_query_allowed('api.zxcbug.com'))

    def test_protected_subdomain_query_is_rejected_locally(self):
        self.assertFalse(search_query_allowed('https://sub.api.zxcbug.com/test'))

    def test_percent_encoded_protected_query_is_rejected_locally(self):
        self.assertFalse(search_query_allowed('api%2Ezxcbug%2Ecom'))

    def test_unicode_width_variant_protected_query_is_rejected_locally(self):
        self.assertFalse(search_query_allowed('ａｐｉ．ｚｘｃｂｕｇ．ｃｏｍ'))


class SearchSourceTests(unittest.TestCase):
    def test_public_https_source_is_allowed(self):
        self.assertTrue(safe_source_url(PUBLIC))

    def test_public_http_source_is_allowed(self):
        self.assertTrue(safe_source_url('http://www.python.org/'))

    def test_non_string_source_is_rejected(self):
        self.assertFalse(safe_source_url(['https://www.python.org/']))

    def test_file_url_is_rejected(self):
        self.assertFalse(safe_source_url('file:///C:/Users/secret.txt'))

    def test_javascript_url_is_rejected(self):
        self.assertFalse(safe_source_url('javascript:alert(1)'))

    def test_localhost_is_rejected(self):
        self.assertFalse(safe_source_url('http://localhost:6099/'))

    def test_ipv4_loopback_is_rejected(self):
        self.assertFalse(safe_source_url('http://127.0.0.1/'))

    def test_ipv6_loopback_is_rejected(self):
        self.assertFalse(safe_source_url('http://[::1]/'))

    def test_private_network_source_is_rejected(self):
        self.assertFalse(safe_source_url('http://192.168.1.2/'))

    def test_link_local_metadata_source_is_rejected(self):
        self.assertFalse(safe_source_url('http://169.254.169.254/latest/meta-data/'))

    def test_username_or_password_in_source_is_rejected(self):
        self.assertFalse(safe_source_url('https://person:password@www.python.org/'))

    def test_url_with_control_character_is_rejected(self):
        self.assertFalse(safe_source_url('https://www.python.org/\nheader'))

    def test_protected_domain_source_is_rejected_locally(self):
        self.assertFalse(safe_source_url('https://api.zxcbug.com/'))

    def test_protected_subdomain_source_is_rejected_locally(self):
        self.assertFalse(safe_source_url('https://foo.api.zxcbug.com/'))

    def test_nested_encoded_protected_source_is_rejected_locally(self):
        self.assertFalse(safe_source_url('https://www.python.org/?next=api%252Ezxcbug%252Ecom'))

    def test_overlong_source_is_rejected(self):
        self.assertFalse(safe_source_url('https://www.python.org/' + 'a' * 2000))


class SearchDataTests(unittest.TestCase):
    def test_public_default_is_disabled(self):
        self.assertIs(SEARCH_DEFAULTS['enabled'], False)

    def test_default_bounds_are_conservative(self):
        self.assertEqual(SEARCH_DEFAULTS['timeout'], 8)
        self.assertEqual(SEARCH_DEFAULTS['count'], 3)
        self.assertEqual(SEARCH_DEFAULTS['cache_ttl'], 300)
        self.assertEqual(SEARCH_DEFAULTS['per_group_per_minute'], 6)
        self.assertEqual(SEARCH_DEFAULTS['global_per_minute'], 12)

    def test_outcome_is_frozen(self):
        value = SearchOutcome('empty', '天气')
        with self.assertRaises(FrozenInstanceError):
            value.status = 'ok'

    def test_outcome_default_has_no_results_or_claimed_cache(self):
        value = SearchOutcome('empty', '天气')
        self.assertEqual(value.results, ())
        self.assertIs(value.cached, False)

    def test_constructor_does_not_mutate_settings(self):
        supplied = settings(); before = copy.deepcopy(supplied)
        OfficialSearchClient(supplied)
        self.assertEqual(supplied, before)


class OfficialSearchAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def search(self, session, query='Python 最新正式版', group_id=88888, **changes):
        client = OfficialSearchClient(settings(**changes))
        outcome = await client.search(query, group_id=group_id, session=session, api_key=KEY)
        return client, outcome

    async def test_success_uses_only_fixed_official_endpoint(self):
        session = FakeSession()
        _, outcome = await self.search(session, endpoint='https://not-an-endpoint.invalid/')
        self.assertEqual(outcome.status, 'ok')
        self.assertEqual(session.calls[0][0], ENDPOINT)

    async def test_request_contains_no_group_history_or_account_metadata(self):
        session = FakeSession(); await self.search(session)
        body = session.calls[0][1]['json']
        self.assertEqual(set(body), {'query','count'})
        self.assertEqual(body['query'], 'Python 最新正式版')

    async def test_authorization_uses_explicit_key_and_disables_redirects(self):
        session = FakeSession(); await self.search(session)
        request = session.calls[0][1]
        self.assertEqual(request['headers']['Authorization'], 'Bearer ' + KEY)
        self.assertIs(request['allow_redirects'], False)
        self.assertGreater(request['timeout'].total, 0)
        self.assertLessEqual(request['timeout'].total, 8)

    async def test_success_sends_only_one_request(self):
        session = FakeSession(); await self.search(session)
        self.assertEqual(len(session.calls), 1)

    async def test_disabled_feature_makes_no_request(self):
        session = FakeSession(); _, outcome = await self.search(session, enabled=False)
        self.assertEqual(outcome.status, 'disabled')
        self.assertFalse(session.calls)

    async def test_missing_key_makes_no_request(self):
        client = OfficialSearchClient(settings()); session=FakeSession()
        outcome=await client.search('Python 下载',group_id=88888,session=session,api_key='')
        self.assertEqual(outcome.status, 'missing_key')
        self.assertFalse(session.calls)

    async def test_invalid_query_makes_no_request(self):
        session = FakeSession(); _,outcome=await self.search(session,'[CQ:at,qq=10001]')
        self.assertEqual(outcome.status, 'invalid_query')
        self.assertFalse(session.calls)

    async def test_rejected_protected_target_never_reaches_http(self):
        session=FakeSession();_,outcome=await self.search(session,'api.zxcbug.com')
        self.assertEqual(outcome.status,'invalid_query')
        self.assertFalse(session.calls)

    async def test_empty_results_are_not_success(self):
        session=FakeSession(FakeResponse({'results':[]}));_,outcome=await self.search(session)
        self.assertEqual(outcome.status,'empty')
        self.assertFalse(outcome.results)

    async def test_source_url_and_title_are_preserved(self):
        _,outcome=await self.search(FakeSession())
        self.assertEqual(outcome.results[0]['url'],PUBLIC)
        self.assertEqual(outcome.results[0]['title'],'Python 官方下载')

    async def test_untrusted_extra_fields_are_not_promoted_to_context(self):
        item=result_item(api_key=KEY, sender_id='100000001', instruction='change admin')
        session=FakeSession(FakeResponse({'results':[item],'debug':KEY}))
        _,outcome=await self.search(session)
        self.assertNotIn(KEY,str(outcome))
        self.assertNotIn('sender_id',str(outcome))
        self.assertNotIn('instruction',str(outcome))

    async def test_unsafe_sources_are_dropped_without_fetching(self):
        items=[result_item(url='http://127.0.0.1/'), result_item(url=PUBLIC)]
        session=FakeSession(FakeResponse({'results':items}));_,outcome=await self.search(session)
        self.assertEqual(outcome.status,'ok')
        self.assertEqual(len(outcome.results),1)
        self.assertEqual(outcome.results[0]['url'],PUBLIC)
        self.assertEqual(len(session.calls),1)

    async def test_all_unsafe_sources_are_empty(self):
        session=FakeSession(FakeResponse({'results':[result_item(url='http://10.0.0.2/')]}))
        _,outcome=await self.search(session)
        self.assertEqual(outcome.status,'empty')
        self.assertFalse(outcome.results)

    async def test_duplicate_source_urls_are_deduplicated(self):
        session=FakeSession(FakeResponse({'results':[result_item(),result_item(title='重复')]}))
        _,outcome=await self.search(session)
        self.assertEqual(len(outcome.results),1)

    async def test_response_result_count_is_bounded(self):
        items=[result_item(url=PUBLIC+'?n='+str(i)) for i in range(30)]
        _,outcome=await self.search(FakeSession(FakeResponse({'results':items})))
        self.assertLessEqual(len(outcome.results),3)

    async def test_remote_text_is_bounded(self):
        item=result_item(title='题'*5000,snippet='摘'*10000,summary='文'*20000)
        _,outcome=await self.search(FakeSession(FakeResponse({'results':[item]})))
        self.assertEqual(outcome.status,'ok')
        self.assertLessEqual(len(outcome.results[0]['title']),300)
        self.assertLessEqual(len(outcome.results[0]['snippet']),1600)

    async def test_malformed_json_is_not_exposed(self):
        session=FakeSession(FakeResponse(raw=b'private-error '+KEY.encode()))
        _,outcome=await self.search(session)
        self.assertEqual(outcome.status,'invalid_response')
        self.assertNotIn(KEY,str(outcome))

    async def test_non_object_json_is_invalid_response(self):
        _,outcome=await self.search(FakeSession(FakeResponse(payload=['not','a','result'])))
        self.assertEqual(outcome.status,'invalid_response')

    async def test_invalid_result_array_is_not_success(self):
        _,outcome=await self.search(FakeSession(FakeResponse({'results':{'title':'not list'}})))
        self.assertNotEqual(outcome.status,'ok')
        self.assertFalse(outcome.results)

    async def test_large_stream_is_aborted_at_body_limit(self):
        response=FakeResponse(raw=b'a'*(300*1024))
        _,outcome=await self.search(FakeSession(response))
        self.assertEqual(outcome.status,'too_large')
        self.assertLessEqual(response.content.bytes_yielded,256*1024+16384)

    async def test_authentication_failure_is_distinct(self):
        _,outcome=await self.search(FakeSession(FakeResponse(status=401,raw=KEY.encode())))
        self.assertEqual(outcome.status,'auth_error')
        self.assertNotIn(KEY,str(outcome))

    async def test_insufficient_balance_is_distinct(self):
        _,outcome=await self.search(FakeSession(FakeResponse(status=402)))
        self.assertEqual(outcome.status,'balance_error')

    async def test_forbidden_is_distinct(self):
        _,outcome=await self.search(FakeSession(FakeResponse(status=403)))
        self.assertEqual(outcome.status,'forbidden')

    async def test_remote_rate_limit_is_distinct(self):
        session=FakeSession(FakeResponse(status=429));_,outcome=await self.search(session)
        self.assertEqual(outcome.status,'rate_limited')
        self.assertEqual(len(session.calls),1)

    async def test_server_error_has_no_automatic_retry(self):
        session=FakeSession(FakeResponse(status=503));_,outcome=await self.search(session)
        self.assertEqual(outcome.status,'service_error')
        self.assertEqual(len(session.calls),1)

    async def test_redirect_is_not_followed(self):
        session=FakeSession(FakeResponse(status=302,headers={'Location':'http://127.0.0.1/'}))
        _,outcome=await self.search(session)
        self.assertEqual(outcome.status,'http_error')
        self.assertEqual(len(session.calls),1)

    async def test_network_error_does_not_leak_exception_text(self):
        session=FakeSession(error=OSError('offline error '+KEY));_,outcome=await self.search(session)
        self.assertEqual(outcome.status,'unavailable')
        self.assertNotIn(KEY,str(outcome))

    async def test_timeout_releases_response_context(self):
        response=FakeResponse(delay=.1)
        _,outcome=await self.search(FakeSession(response),timeout=.01)
        self.assertEqual(outcome.status,'timeout')
        self.assertTrue(response.exited)

    async def test_cancellation_propagates_and_releases_response_context(self):
        gate=asyncio.Event();started=asyncio.Event();response=FakeResponse(gate=gate,started=started)
        client=OfficialSearchClient(settings());session=FakeSession(response)
        task=asyncio.create_task(client.search('Python 下载',group_id=88888,session=session,api_key=KEY))
        await asyncio.wait_for(started.wait(),1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(response.exited)

    async def test_cache_hit_uses_no_second_request(self):
        client=OfficialSearchClient(settings());session=FakeSession()
        first=await client.search('Python 下载',group_id=88888,session=session,api_key=KEY)
        second=await client.search('Python 下载',group_id=88888,session=session,api_key=KEY)
        self.assertEqual(first.status,'ok')
        self.assertEqual(second.results,first.results)
        self.assertIs(second.cached,True)
        self.assertEqual(len(session.calls),1)

    async def test_group_search_quota_blocks_new_query_locally(self):
        client=OfficialSearchClient(settings(per_group_per_minute=1));session=FakeSession()
        await client.search('Python 下载',group_id=88888,session=session,api_key=KEY)
        second=await client.search('Python 文档',group_id=88888,session=session,api_key=KEY)
        self.assertEqual(second.status,'limited')
        self.assertEqual(len(session.calls),1)

    async def test_global_search_quota_is_shared_across_groups(self):
        client=OfficialSearchClient(settings(global_per_minute=1));session=FakeSession()
        await client.search('Python 下载',group_id=88888,session=session,api_key=KEY)
        second=await client.search('Python 文档',group_id=99999,session=session,api_key=KEY)
        self.assertEqual(second.status,'limited')
        self.assertEqual(len(session.calls),1)

    async def test_failed_request_still_consumes_local_attempt_budget(self):
        client=OfficialSearchClient(settings(per_group_per_minute=1));session=FakeSession(FakeResponse(status=503))
        await client.search('Python 下载',group_id=88888,session=session,api_key=KEY)
        second=await client.search('Python 文档',group_id=88888,session=session,api_key=KEY)
        self.assertEqual(second.status,'limited')
        self.assertEqual(len(session.calls),1)

    async def test_cached_rows_cannot_be_mutated_by_previous_caller(self):
        client=OfficialSearchClient(settings());session=FakeSession()
        first=await client.search('Python 下载',group_id=88888,session=session,api_key=KEY)
        first.results[0]['title']='CALLER_MUTATION'
        second=await client.search('Python 下载',group_id=99999,session=session,api_key=KEY)
        self.assertEqual(second.results[0]['title'],'Python 官方下载')
        self.assertEqual(len(session.calls),1)

    async def test_unresolved_environment_key_never_reaches_http(self):
        client=OfficialSearchClient(settings());session=FakeSession()
        outcome=await client.search('Python 下载',group_id=88888,session=session,api_key='env:DEEPSEEK_API_KEY')
        self.assertEqual(outcome.status,'missing_key')
        self.assertFalse(session.calls)

    async def test_whitespace_wrapped_environment_key_never_reaches_http(self):
        client=OfficialSearchClient(settings());session=FakeSession()
        outcome=await client.search('Python 下载',group_id=88888,session=session,api_key=' env:DEEPSEEK_API_KEY ')
        self.assertEqual(outcome.status,'missing_key')
        self.assertFalse(session.calls)

    async def test_header_control_character_is_rejected_before_http(self):
        client=OfficialSearchClient(settings());session=FakeSession()
        outcome=await client.search('Python 下载',group_id=88888,session=session,api_key=KEY+'\nheader')
        self.assertEqual(outcome.status,'missing_key')
        self.assertFalse(session.calls)

    async def test_boolean_group_id_is_not_an_account_or_group(self):
        session=FakeSession();_,outcome=await self.search(session,group_id=True)
        self.assertEqual(outcome.status,'invalid_query')
        self.assertFalse(session.calls)

    async def test_zero_group_id_is_rejected_before_http(self):
        session=FakeSession();_,outcome=await self.search(session,group_id=0)
        self.assertEqual(outcome.status,'invalid_query')
        self.assertFalse(session.calls)

    async def test_requested_result_count_is_hard_capped(self):
        session=FakeSession();await self.search(session,count=1000000)
        self.assertGreaterEqual(session.calls[0][1]['json']['count'],1)
        self.assertLessEqual(session.calls[0][1]['json']['count'],5)

    async def test_known_credential_reflected_by_remote_page_is_removed(self):
        item=result_item(summary='Remote text '+KEY)
        _,outcome=await self.search(FakeSession(FakeResponse({'results':[item]})))
        self.assertEqual(outcome.status,'ok')
        self.assertNotIn(KEY,str(outcome))

    async def test_key_whitespace_normalization_also_applies_to_redaction(self):
        client=OfficialSearchClient(settings());session=FakeSession(FakeResponse(
            {'results':[result_item(summary='Remote text '+KEY)]}))
        outcome=await client.search('Python 下载',group_id=88888,session=session,api_key=' '+KEY+' ')
        self.assertEqual(outcome.status,'ok')
        self.assertNotIn(KEY,str(outcome))

    async def test_empty_results_are_not_cached_as_success(self):
        client=OfficialSearchClient(settings());session=FakeSession(FakeResponse({'results':[]}),FakeResponse())
        first=await client.search('Python 下载',group_id=88888,session=session,api_key=KEY)
        second=await client.search('Python 下载',group_id=88888,session=session,api_key=KEY)
        self.assertEqual(first.status,'empty')
        self.assertEqual(second.status,'ok')
        self.assertFalse(second.cached)
        self.assertEqual(len(session.calls),2)

    async def test_cache_expiration_and_search_rate_window_are_independent(self):
        clock=[1000.0]
        client=OfficialSearchClient(settings(cache_ttl=1,per_group_per_minute=1))
        session=FakeSession(FakeResponse(),FakeResponse())
        with patch.object(search_module,'time',SimpleNamespace(monotonic=lambda:clock[0])):
            first=await client.search('Python 下载',group_id=88888,session=session,api_key=KEY)
            clock[0]+=2
            expired_but_limited=await client.search('Python 下载',group_id=88888,session=session,api_key=KEY)
            clock[0]+=60
            recovered=await client.search('Python 下载',group_id=88888,session=session,api_key=KEY)
        self.assertEqual(first.status,'ok')
        self.assertEqual(expired_but_limited.status,'limited')
        self.assertEqual(recovered.status,'ok')
        self.assertFalse(recovered.cached)
        self.assertEqual(len(session.calls),2)

    async def test_at_most_two_search_streams_execute_concurrently(self):
        gate=asyncio.Event();starts=[asyncio.Event() for _ in range(3)]
        responses=[FakeResponse(gate=gate,started=start) for start in starts]
        session=FakeSession(*responses);client=OfficialSearchClient(settings())
        tasks=[asyncio.create_task(client.search('Python '+str(index),group_id=88888,
                   session=session,api_key=KEY)) for index in range(3)]
        try:
            await asyncio.wait_for(asyncio.gather(starts[0].wait(),starts[1].wait()),1)
            await asyncio.sleep(.01)
            self.assertEqual(len(session.calls),2)
            self.assertFalse(starts[2].is_set())
            gate.set()
            outcomes=await asyncio.wait_for(asyncio.gather(*tasks),1)
            self.assertTrue(all(value.status=='ok' for value in outcomes))
        finally:
            gate.set()
            for task in tasks:
                if not task.done():task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)

    async def test_total_timeout_includes_waiting_for_concurrency_slot(self):
        gate=asyncio.Event();starts=[asyncio.Event() for _ in range(2)]
        responses=[FakeResponse(gate=gate,started=start) for start in starts]
        session=FakeSession(*responses);client=OfficialSearchClient(settings(timeout=1))
        tasks=[asyncio.create_task(client.search('Python '+str(index),group_id=88888,
                   session=session,api_key=KEY)) for index in range(2)]
        try:
            await asyncio.wait_for(asyncio.gather(starts[0].wait(),starts[1].wait()),.5)
            # The two existing calls already hold a one-second wall-time context.
            # Give the new queued call a shorter budget to avoid a simultaneous
            # expiry/release race which says nothing about the waiting budget.
            client.timeout=.02
            loop=asyncio.get_running_loop();began=loop.time()
            outcome=await asyncio.wait_for(client.search('Python waiting',group_id=88888,
                session=session,api_key=KEY),.5)
            self.assertEqual(outcome.status,'timeout')
            self.assertLess(loop.time()-began,.3)
            self.assertEqual(len(session.calls),2)
        finally:
            gate.set()
            for task in tasks:
                if not task.done():task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)


if __name__ == '__main__':
    classes=(SearchQueryTests,SearchSourceTests,SearchDataTests,OfficialSearchAsyncTests)
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in classes)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                success=result.wasSuccessful(),fixtures='synthetic official-search client and fake HTTP stream',
                real_model_requests=0,real_search_requests=0,real_QQ_messages=0,network='none')
    if RESULT:RESULT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
