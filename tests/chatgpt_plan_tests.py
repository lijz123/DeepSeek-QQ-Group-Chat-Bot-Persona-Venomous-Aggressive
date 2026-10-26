"""Offline contract tests for ChatGPT-plan OAuth and the QQ bot adapter.

Usage: python chatgpt_plan_tests.py [SOURCE_DIR] [RESULT_JSON]
All HTTP and model responses are in-memory fixtures. These tests never open a
browser, connect a socket, read real credentials, or consume a model allowance.
They demonstrate protocol handling, not real-model semantic understanding.
"""
from __future__ import annotations

from layout_support import app_directory, project_directory, configure_test_paths
configure_test_paths()

import asyncio
import copy
import io
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
import importlib
import contextlib
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from collections import deque
from unittest.mock import patch, AsyncMock
from contextlib import asynccontextmanager

SOURCE = app_directory(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None)
RESULT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))

import bot
import chatgpt_auth as auth
import chatgpt_provider as provider

BEARER = "OFFLINE_FAKE_BEARER_DO_NOT_LOG"
REFRESH = "OFFLINE_FAKE_REFRESH_DO_NOT_LOG"
VISIBLE = {"models": [
    {"slug": "private-hidden", "visibility": "hidden", "display_name": "Hidden"},
    {"slug": "fixture-first", "visibility": "list", "display_name": "First"},
    {"slug": "fixture-second", "visibility": "list", "display_name": "Second"},
]}


def sse(*events, ending=True):
    result = "".join("data: " + json.dumps(e, ensure_ascii=False) + "\r\n\r\n" for e in events)
    if ending:
        result += 'data: {"type":"response.completed","response":{"status":"completed"}}\r\n\r\n'
    return result.encode("utf-8")


def delta(text):
    return {"type": "response.output_text.delta", "delta": text}


class FakeContent:
    def __init__(self, body: bytes, fragment=7):
        self.body = body
        self.pos = 0
        self.fragment = fragment

    async def read(self, n=-1):
        if n is None or n < 0:
            n = len(self.body) - self.pos
        out = self.body[self.pos:self.pos+n]
        self.pos += len(out)
        return out

    async def readline(self):
        end = self.body.find(b"\n", self.pos)
        return await self.read(len(self.body)-self.pos if end < 0 else end+1-self.pos)

    async def iter_any(self):
        while self.pos < len(self.body):
            yield await self.read(self.fragment)

    async def iter_chunked(self, n):
        while self.pos < len(self.body):
            yield await self.read(min(n, self.fragment))

    def __aiter__(self):
        return self.iter_any()


class FakeResponse:
    def __init__(self, body=b"", status=200, content_type="text/event-stream", fragment=7, headers=None):
        if isinstance(body, dict):
            body = json.dumps(body).encode()
            content_type = "application/json"
        if isinstance(body, str):
            body = body.encode()
        self._body = body
        self.status = status
        self.headers = {"Content-Type": content_type, "x-request-id": "fixture-request-123"}
        self.headers.update(headers or {})
        self.content = FakeContent(body, fragment)
        self.content_type = content_type
        self.charset = "utf-8"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def read(self):
        return await self.content.read()

    async def text(self, **kwargs):
        return (await self.read()).decode("utf-8")

    async def json(self, **kwargs):
        return json.loads((await self.read()).decode("utf-8"))


class FakeSession:
    def __init__(self, get=(), post=()):
        self.get_responses = deque(get)
        self.post_responses = deque(post)
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if not self.get_responses:
            raise AssertionError("Unexpected GET; all traffic must be scripted offline")
        return self.get_responses.popleft()

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if not self.post_responses:
            raise AssertionError("Unexpected POST/retry; all traffic must be scripted offline")
        return self.post_responses.popleft()

    async def close(self):
        self.closed = True


class FakeProfile:
    identity_key = ("fixture-client", "fixture-subject")
    def exists(self):
        return True

    async def access_token(self, session):
        return BEARER


def make_provider(**cfg):
    p = provider.ChatGPTPlanProvider({"chatgpt_model": "auto", "timeout": 5, **cfg})
    p.profile = FakeProfile()
    # Keep test diagnostics out of any real application's per-user profile.
    # This path is never created, so provider._record remains in-memory only.
    p._diagnostic_dir = SOURCE.parent / "chatgpt-plan-run" / "offline-unused-diagnostic-dir"
    return p


class PayloadTests(unittest.TestCase):
    def test_01_system_moved_to_instructions(self):
        data = provider.build_payload([{"role":"system","content":"system rules"},
                                       {"role":"user","content":"hello"}], "fixture-first")
        self.assertEqual(data["instructions"], "system rules")
        self.assertFalse(any(x.get("role") == "system" for x in data["input"]))

    def test_02_history_order_and_roles_preserved(self):
        messages = [{"role":r,"content":r+" text"} for r in ("developer","user","assistant","user")]
        data = provider.build_payload(messages, "fixture-first")
        self.assertEqual([x["role"] for x in data["input"]], ["developer","user","assistant","user"])

    def test_03_required_stream_and_no_store(self):
        data = provider.build_payload([{"role":"user","content":"hello"}], "fixture-first")
        self.assertIs(data["store"], False)
        self.assertIs(data["stream"], True)
        self.assertEqual(data["model"], "fixture-first")

    def test_04_unsupported_parameters_never_added(self):
        data = provider.build_payload([{"role":"user","content":"hello"}], "fixture-first")
        self.assertEqual(set(data), {"model","input","store","stream"})

    def test_05_no_mutation_of_input(self):
        rows = [{"role":"system","content":"s"},{"role":"user","content":"u"}]
        original = copy.deepcopy(rows)
        provider.build_payload(rows, "fixture-first")
        self.assertEqual(rows, original)

    def test_06_unknown_role_fails_closed(self):
        with self.assertRaises(provider.PlanProviderError):
            provider.build_payload([{"role":"tool","content":"u"}], "fixture-first")

    def test_07_nontext_is_not_stringified(self):
        for value in (None, 12, {"instruction":"pretend text"}, [{"type":"image_url"}]):
            with self.subTest(value=value), self.assertRaises(provider.PlanProviderError):
                provider.build_payload([{"role":"user","content":value}], "fixture-first")


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_08_unicode_fragmented_stream(self):
        response = FakeResponse(sse(delta("你好"),delta("，世界🙂")), fragment=1)
        self.assertEqual(await provider.consume_response(response, {}), "你好，世界🙂")

    async def test_09_only_visible_delta_is_returned(self):
        response = FakeResponse(sse({"type":"response.reasoning_text.delta","delta":"private reasoning"},
                                    delta("visible"), {"type":"response.output_text.done","text":"duplicated"}))
        self.assertEqual(await provider.consume_response(response, {}), "visible")

    async def test_10_eof_without_completed_fails(self):
        with self.assertRaises(provider.PlanProviderError):
            await provider.consume_response(FakeResponse(sse(delta("partial"), ending=False)), {})

    async def test_11_failed_terminal_fails(self):
        raw = sse(delta("partial"), {"type":"response.failed","response":{"error":{"code":"quota"}}}, ending=False)
        with self.assertRaises(provider.PlanProviderError):
            await provider.consume_response(FakeResponse(raw), {})

    async def test_12_incomplete_terminal_fails(self):
        with self.assertRaises(provider.PlanProviderError):
            await provider.consume_response(FakeResponse(sse({"type":"response.incomplete"},ending=False)), {})

    async def test_13_json_200_is_not_completed_inference(self):
        with self.assertRaises(provider.PlanProviderError):
            await provider.consume_response(FakeResponse({"output_text":"not an SSE completion"}), {})

    async def test_14_html_200_is_not_inference(self):
        diagnostics = {}
        with self.assertRaises(provider.PlanProviderError):
            await provider.consume_response(FakeResponse("<html>"+BEARER+"</html>",content_type="text/html"), diagnostics)
        self.assertNotIn(BEARER, json.dumps(diagnostics))

    async def test_15_mislabeled_genuine_sse_requires_completed(self):
        self.assertEqual(await provider.consume_response(FakeResponse(sse(delta("ok")),content_type="application/octet-stream"), {}), "ok")

    async def test_16_malformed_data_fails(self):
        with self.assertRaises(provider.PlanProviderError):
            await provider.consume_response(FakeResponse(b"data: {bad}\n\n"), {})

    async def test_17_done_marker_is_not_completion(self):
        with self.assertRaises(provider.PlanProviderError):
            await provider.consume_response(FakeResponse(b"data: [DONE]\n\n"), {})

    async def test_18_empty_completed_not_a_reply(self):
        with self.assertRaises(provider.PlanProviderError):
            await provider.consume_response(FakeResponse(sse()), {})


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_19_models_filter_visibility_and_preserve_order(self):
        p = make_provider()
        rows = await p.list_models(FakeSession(get=[FakeResponse(VISIBLE)]))
        self.assertEqual([x["slug"] for x in rows], ["fixture-first","fixture-second"])

    async def test_20_auto_model_and_official_endpoint(self):
        p = make_provider(base_url="https://example.invalid", api_key="must_not_use")
        session = FakeSession(get=[FakeResponse(VISIBLE)], post=[FakeResponse(sse(delta("ok")))])
        self.assertEqual(await p.complete(session,[{"role":"user","content":"hi"}]), "ok")
        post = [x for x in session.calls if x[0] == "POST"]
        self.assertEqual(len(post),1)
        self.assertEqual(post[0][1], "https://api.openai.com/v1/responses")
        self.assertEqual(post[0][2]["json"]["model"], "fixture-first")
        self.assertEqual(post[0][2]["headers"]["Authorization"], "Bearer "+BEARER)
        self.assertFalse(post[0][2].get("allow_redirects", True))
        self.assertNotIn("must_not_use", repr(session.calls))

    async def test_21_explicit_visible_model_is_selected(self):
        p = make_provider(chatgpt_model="fixture-second")
        session=FakeSession(get=[FakeResponse(VISIBLE)],post=[FakeResponse(sse(delta("ok")))])
        await p.complete(session,[{"role":"user","content":"hi"}])
        self.assertEqual([x for x in session.calls if x[0]=="POST"][0][2]["json"]["model"],"fixture-second")

    async def test_22_unknown_selected_model_prevents_post(self):
        p=make_provider(chatgpt_model="private-hidden")
        session=FakeSession(get=[FakeResponse(VISIBLE)])
        with self.assertRaises(provider.PlanProviderError):
            await p.complete(session,[{"role":"user","content":"hi"}])
        self.assertFalse(any(x[0]=="POST" for x in session.calls))

    async def test_23_http_401_is_not_retried(self):
        await self.assert_status_stops(401)

    async def test_24_http_429_is_not_retried(self):
        await self.assert_status_stops(429)

    async def test_25_http_503_is_not_retried(self):
        await self.assert_status_stops(503)

    async def assert_status_stops(self,status):
        p=make_provider()
        session=FakeSession(get=[FakeResponse(VISIBLE)],post=[FakeResponse({"error":{"code":"fixture_error","message":BEARER}},status=status)])
        with self.assertRaises(provider.PlanProviderError) as raised:
            await p.complete(session,[{"role":"user","content":"hi"}])
        self.assertEqual(sum(x[0]=="POST" for x in session.calls),1)
        self.assertNotIn(BEARER,str(raised.exception))
        self.assertNotIn(BEARER,json.dumps(p.last_diagnostics))
        self.assertEqual(p.last_diagnostics.get("http_status"),status)

    async def test_26_non_sse_diagnostic_retained_no_retry(self):
        p=make_provider()
        session=FakeSession(get=[FakeResponse(VISIBLE)],post=[FakeResponse("<html>not SSE</html>",content_type="text/html")])
        with self.assertRaises(provider.PlanProviderError):
            await p.complete(session,[{"role":"user","content":"hi"}])
        self.assertEqual(sum(x[0]=="POST" for x in session.calls),1)
        self.assertEqual(p.last_diagnostics.get("http_status"),200)
        self.assertIn("text/html",p.last_diagnostics.get("content_type",""))
        self.assertTrue(p.last_diagnostics.get("body_shape"))

    async def test_27_empty_catalog_prevents_post(self):
        p=make_provider();session=FakeSession(get=[FakeResponse({"models":[]})])
        with self.assertRaises(provider.PlanProviderError):
            await p.complete(session,[{"role":"user","content":"hi"}])
        self.assertFalse(any(x[0]=="POST" for x in session.calls))

    async def test_57_catalog_cache_same_account_and_refresh_on_switch(self):
        p=make_provider()
        session=FakeSession(get=[FakeResponse(VISIBLE),FakeResponse(VISIBLE)],
                            post=[FakeResponse(sse(delta("a"))),FakeResponse(sse(delta("b"))),FakeResponse(sse(delta("c")))])
        await p.complete(session,[{"role":"user","content":"first"}])
        await p.complete(session,[{"role":"user","content":"second"}])
        self.assertEqual(sum(x[0]=="GET" for x in session.calls),1)
        p.profile.identity_key=("fixture-client","other-subject")
        await p.complete(session,[{"role":"user","content":"third"}])
        self.assertEqual(sum(x[0]=="GET" for x in session.calls),2)

    async def test_58_invalid_input_does_not_refresh_or_post(self):
        class NoCredentialRead:
            async def access_token(self,session):
                raise AssertionError("Invalid input must fail before credential access")
        p=make_provider();p.profile=NoCredentialRead();session=FakeSession()
        with self.assertRaises(provider.PlanProviderError):
            await p.complete(session,[{"role":"unknown","content":"x"}])
        self.assertEqual(session.calls,[])

    async def test_59_partial_stream_never_returned_or_retried(self):
        p=make_provider()
        session=FakeSession(get=[FakeResponse(VISIBLE)],post=[FakeResponse(sse(delta("unverified draft"),ending=False))])
        with self.assertRaises(provider.PlanProviderError):
            await p.complete(session,[{"role":"user","content":"hi"}])
        self.assertEqual(sum(x[0]=="POST" for x in session.calls),1)
        self.assertFalse(p.last_diagnostics.get("completed_event_seen"))

    async def test_64_invalid_next_input_cannot_reuse_completed_diagnostics(self):
        p=make_provider()
        session=FakeSession(get=[FakeResponse(VISIBLE)],post=[FakeResponse(sse(delta("ok")))])
        self.assertEqual(await p.complete(session,[{"role":"user","content":"valid"}]),"ok")
        self.assertTrue(p.last_diagnostics["completed_event_seen"])
        old_calls=len(session.calls)
        with self.assertRaises(provider.PlanProviderError):
            await p.complete(session,[{"role":"unknown","content":"invalid"}])
        self.assertFalse(p.last_diagnostics["inference_attempted"])
        self.assertFalse(p.last_diagnostics["completed_event_seen"])
        self.assertEqual(p.last_diagnostics["error_code"],"UNSUPPORTED_MESSAGE_ROLE")
        self.assertEqual(len(session.calls),old_calls)


class BotIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def brain(self,semantic=True):
        cfg=copy.deepcopy(bot.DEFAULTS["llm"])
        cfg.update(provider="chatgpt_plan",api_key="",semantic_pipeline=semantic)
        brain=bot.Brain(cfg,copy.deepcopy(bot.DEFAULTS["safety"]),copy.deepcopy(bot.DEFAULTS["persona"]),80)
        brain._session=FakeSession()
        return brain

    async def test_28_direct_call_uses_plan_without_api_key(self):
        class P:
            profile=FakeProfile()
            calls=0
            async def complete(self,session,messages):
                self.calls+=1;return "fixture"
        brain=self.brain();p=P();brain._plan_provider=p
        self.assertTrue(brain._provider_ready())
        self.assertEqual(await brain._call([{"role":"user","content":"hi"}],0.2,token_budget=640),"fixture")
        self.assertEqual(p.calls,1)
        self.assertEqual(brain._session.calls,[])

    async def test_29_understand_expression_review_all_use_plan(self):
        class P:
            profile=FakeProfile()
            def __init__(self):
                self.calls=[]
                self.outputs=deque([
                    json.dumps(dict(act="answer",target="self",meaning="用户分享困意",unresolved="补觉先后不明确",reply_goal="承接确定的困意",mode="normal",needs_clarification=False),ensure_ascii=False),
                    json.dumps(dict(action="reply",text="今天困得够呛啊",next="wait"),ensure_ascii=False),
                    json.dumps(dict(ok=True,issue="")),
                ])
            async def complete(self,session,messages):
                self.calls.append(copy.deepcopy(messages));return self.outputs.popleft()
        brain=self.brain();p=P();brain._plan_provider=p
        transcript=json.dumps(dict(role="group_member",sender={"id":"fixture-user","display_name":"离线成员"},content="今天补觉了，困死我了",message_id="fixture-m1",event_kind="message"),ensure_ascii=False)
        reply=await brain.plan_reply(100,transcript,"离线成员",True)
        self.assertIsNotNone(reply)
        self.assertEqual(reply.text,"今天困得够呛啊")
        self.assertEqual(len(p.calls),3)
        self.assertTrue(brain._semantic_last[100]["review_ok"])
        self.assertEqual(brain._session.calls,[])

    async def test_30_plan_exception_is_sanitized_and_not_retried(self):
        class E(Exception):
            code="FIXTURE_FAILURE"
        class P:
            profile=FakeProfile()
            last_diagnostics={}
            calls=0
            async def complete(self,session,messages):
                self.calls+=1;raise E(BEARER)
        brain=self.brain(False);p=P();brain._plan_provider=p
        capture=io.StringIO();handler=logging.StreamHandler(capture);bot.log.addHandler(handler)
        try:
            result=await brain.plan_reply(100,"[]","离线成员",True)
        finally:
            bot.log.removeHandler(handler)
        self.assertIsNone(result)
        self.assertEqual(p.calls,1)
        self.assertNotIn(BEARER,capture.getvalue())
        self.assertIn("FIXTURE_FAILURE",capture.getvalue())

    async def test_31_missing_plan_profile_fails_without_api_fallback(self):
        class Missing:
            def exists(self):return False
        brain=self.brain();brain.cfg["api_key"]="must_not_fallback"
        brain._plan_provider=type("P",(),{"profile":Missing()})()
        self.assertFalse(brain._provider_ready())
        self.assertEqual(brain._session.calls,[])

    async def test_32_unknown_provider_is_not_deepseek(self):
        brain=self.brain();brain.cfg.update(provider="unknown",api_key="anything")
        self.assertFalse(brain._provider_ready())
        self.assertEqual(brain._session.calls,[])

    async def test_33_close_keeps_transport_resource_cleanup(self):
        brain=self.brain();session=brain._session
        await brain.close()
        self.assertTrue(session.closed)
        self.assertIsNone(brain._session)


class ConfigTests(unittest.TestCase):
    def test_34_plan_does_not_resolve_billing_key(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"config.json";path.write_text(json.dumps({"llm":{"provider":"chatgpt_plan","api_key":"env:DO_NOT_READ"}}))
            with patch.object(bot,"resolve_secret",side_effect=AssertionError("must not resolve")):
                cfg=bot.load_config(path)
            self.assertEqual(cfg["llm"]["api_key"],"")

    def test_35_cli_provider_override_avoids_billing_key(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"config.json";path.write_text("{}")
            with patch.object(bot,"resolve_secret",side_effect=AssertionError("must not resolve")):
                cfg=bot.load_config(path,provider="chatgpt_plan")
            self.assertEqual(cfg["llm"]["provider"],"chatgpt_plan")
            self.assertEqual(cfg["llm"]["api_key"],"")

    def test_36_deepseek_existing_env_resolution_preserved(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{"DEEPSEEK_API_KEY":"fake-deepseek-key"}):
            cfg=bot.load_config(Path(d)/"missing.json")
            self.assertEqual(cfg["llm"]["provider"],"deepseek")
            self.assertEqual(cfg["llm"]["api_key"],"fake-deepseek-key")

    def test_60_start_gpt_selects_oauth_cli_without_secret(self):
        text=(project_directory(SOURCE)/"start_gpt.bat").read_text(encoding="utf-8-sig")
        self.assertIn("--provider chatgpt_plan",text)
        self.assertNotIn(BEARER,text)
        self.assertNotIn("set DEEPSEEK_API_KEY",text)

    def test_61_login_launcher_no_inference(self):
        text=(project_directory(SOURCE)/"login_chatgpt.bat").read_text(encoding="utf-8-sig")
        self.assertIn("chatgpt_auth.py",text)
        self.assertIn("--login",text)
        self.assertNotIn("preview.py",text)
        self.assertNotIn("bot.py",text)

    def test_62_default_start_has_no_exclusive_deepseek_env_gate(self):
        text=(project_directory(SOURCE)/"start.bat").read_text(encoding="utf-8-sig")
        self.assertNotIn('if "%DEEPSEEK_API_KEY%"==""',text)
        self.assertIn("bot.py",text)


class PreviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_63_preview_sends_structured_text_and_provider_override(self):
        preview=importlib.import_module("preview")
        cfg=copy.deepcopy(bot.DEFAULTS);cfg["llm"].update(provider="chatgpt_plan",api_key="")
        captured={}
        class PreviewBrain:
            def __init__(self,*args):pass
            def _provider_ready(self):return True
            async def reply(self,gid,transcript,speaker,is_at):
                captured.update(gid=gid,row=json.loads(transcript),speaker=speaker,is_at=is_at)
                return "fixture answer"
            async def close(self):captured["closed"]=True
        with patch.object(sys,"argv",["preview.py","测试文字","成员甲","--provider","chatgpt_plan"]), \
             patch.object(preview,"Brain",PreviewBrain),patch.object(preview,"load_config",return_value=cfg) as load, \
             patch.object(preview,"setup_logging"),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(await preview.main(),0)
        self.assertEqual(load.call_args.kwargs["provider"],"chatgpt_plan")
        self.assertEqual(captured["row"]["content"],"测试文字")
        self.assertEqual(captured["row"]["sender"]["display_name"],"成员甲")
        self.assertEqual(captured["row"]["event_kind"],"message")
        self.assertTrue(captured["closed"])


def record(**changes):
    value=dict(app="QQ Roast Bot",issuer="https://auth.openai.com",client_id="oaiapp_fixture",ext_agent_host_id="urn:uuid:00000000-0000-4000-8000-000000000000",subject="fixture-subject",access_token=BEARER,refresh_token=REFRESH,id_token="fixture-retained-id",scopes=["chatgpt.tokens.use.direct","resource.invoke","offline_access"],saved_at=time.time(),expires_in=3600)
    value.update(changes)
    return value


class AuthRecordTests(unittest.TestCase):
    def test_37_valid_own_record(self):
        value=auth.validate_record(record())
        self.assertEqual(value["subject"],"fixture-subject")

    def test_38_dynamic_client_is_not_issued_client(self):
        with self.assertRaises(auth.PlanAuthError):auth.validate_record(record(client_id="dynamic_agent_client"))

    def test_39_wrong_issuer_rejected(self):
        with self.assertRaises(auth.PlanAuthError):auth.validate_record(record(issuer="https://example.invalid"))

    def test_40_scope_grant_required(self):
        for scopes in ([],["openid"],["resource.invoke"],["chatgpt.tokens.use.direct"]):
            with self.subTest(scopes=scopes),self.assertRaises(auth.PlanAuthError):auth.validate_record(record(scopes=scopes))

    def test_41_identity_and_tokens_required(self):
        for field in ("subject","access_token","refresh_token"):
            with self.subTest(field=field),self.assertRaises(auth.PlanAuthError):auth.validate_record(record(**{field:""}))

    def test_42_invalid_expiry_rejected(self):
        for changes in (dict(expires_in=-1),dict(expires_in="3600"),dict(saved_at="now")):
            with self.subTest(changes=changes),self.assertRaises(auth.PlanAuthError):auth.validate_record(record(**changes))

    @unittest.skipUnless(os.name=="nt","Windows DPAPI only")
    def test_43_dpapi_fake_record_roundtrip_no_plaintext(self):
        original=record();encrypted=auth.encrypt_record(original)
        self.assertNotIn(BEARER.encode(),encrypted)
        self.assertNotIn(REFRESH.encode(),encrypted)
        self.assertEqual(auth.decrypt_record(encrypted),original)

    def test_44_other_application_profile_rejected(self):
        with self.assertRaises(auth.PlanAuthError):auth.validate_record(record(app="Other App"))

    def test_45_non_ascii_callback_state_safely_rejected(self):
        with self.assertRaises(auth.PlanAuthError):
            auth.validate_callback({"state":"不同","code":"fake","client_id":"oaiapp_fixture"},"expected",None)

    def test_46_callback_state_verified_before_error(self):
        with self.assertRaises(auth.PlanAuthError) as raised:
            auth.validate_callback({"state":"wrong","error":"access_denied"},"expected",None)
        self.assertEqual(raised.exception.code,"STATE_MISMATCH")

    def test_47_returning_client_cannot_change(self):
        with self.assertRaises(auth.PlanAuthError):
            auth.validate_callback({"state":"expected","code":"fake","client_id":"oaiapp_other"},"expected","oaiapp_fixture")

    def test_48_pkce_reference_vector(self):
        self.assertEqual(auth.pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),"E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM")


class MemoryProfile(auth.PlanProfile):
    """Synthetic store only; does not call ACL helpers or read a real profile."""
    def __init__(self,path,state):
        super().__init__(path)
        self.state=state
        self.writes=0

    @asynccontextmanager
    async def _process_lock(self,timeout=20):
        yield

    def _read(self):
        return copy.deepcopy(self.state["record"])

    def _write(self,value):
        auth.validate_record(value)
        self.state["record"]=copy.deepcopy(value)
        self.writes+=1


class RefreshTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Public-key retrieval is mocked too; no fake refresh can reach a socket.
        self.public_keys_patch=patch.object(auth,"_public_keys",new=AsyncMock(return_value={"keys":[]}))
        self.public_keys_patch.start()

    async def asyncTearDown(self):
        self.public_keys_patch.stop()

    async def test_49_valid_token_no_http(self):
        with tempfile.TemporaryDirectory() as d:
            p=MemoryProfile(d,{"record":record()});session=FakeSession()
            self.assertEqual(await p.access_token(session),BEARER)
            self.assertEqual(session.calls,[])
            self.assertEqual(p.identity_key,("https://auth.openai.com","oaiapp_fixture","fixture-subject"))

    async def test_50_refresh_rotates_and_saves_new_token(self):
        with tempfile.TemporaryDirectory() as d:
            state={"record":record(saved_at=time.time()-3600)};p=MemoryProfile(d,state)
            response=dict(access_token="rotated-access",refresh_token="rotated-refresh",token_type="Bearer",expires_in=3600,scope="resource.invoke chatgpt.tokens.use.direct")
            session=FakeSession(post=[FakeResponse(response)])
            with patch.object(auth,"getproxies",return_value={}):
                self.assertEqual(await p.access_token(session),"rotated-access")
            self.assertEqual(p.writes,1)
            self.assertEqual(state["record"]["refresh_token"],"rotated-refresh")
            call=session.calls[0]
            self.assertEqual(call[1],"https://auth.openai.com/api/accounts/oauth/token")
            self.assertEqual(call[2]["data"]["client_id"],"oaiapp_fixture")
            self.assertEqual(call[2]["data"]["resource"],"https://api.openai.com/v1")
            self.assertNotIn("scope",call[2]["data"])
            self.assertFalse(call[2]["allow_redirects"])

    async def test_51_refresh_scope_omission_retains_grant(self):
        with tempfile.TemporaryDirectory() as d:
            state={"record":record(saved_at=time.time()-3600)};p=MemoryProfile(d,state)
            session=FakeSession(post=[FakeResponse(dict(access_token="new",refresh_token="new-refresh",token_type="Bearer",expires_in=3600))])
            with patch.object(auth,"getproxies",return_value={}):
                self.assertEqual(await p.access_token(session),"new")
            self.assertIn("chatgpt.tokens.use.direct",state["record"]["scopes"])

    async def test_52_concurrent_refreshes_share_serialized_record(self):
        with tempfile.TemporaryDirectory() as d:
            state={"record":record(saved_at=time.time()-3600)};a=MemoryProfile(d,state);b=MemoryProfile(d,state)
            session=FakeSession(post=[FakeResponse(dict(access_token="new",refresh_token="new-refresh",token_type="Bearer",expires_in=3600))])
            with patch.object(auth,"getproxies",return_value={}):
                outputs=await asyncio.gather(a.access_token(session),b.access_token(session))
            self.assertEqual(outputs,["new","new"])
            self.assertEqual(sum(x[0]=="POST" for x in session.calls),1)
            self.assertEqual(a.writes+b.writes,1)

    async def test_53_terminal_invalid_grant_clears_fake_file_and_stops(self):
        with tempfile.TemporaryDirectory() as d:
            credential=Path(d)/"credentials.dpapi";credential.write_bytes(b"synthetic-test-not-a-token")
            p=MemoryProfile(d,{"record":record(saved_at=time.time()-3600)})
            session=FakeSession(post=[FakeResponse({"error":"invalid_grant"},status=400)])
            with patch.object(auth,"getproxies",return_value={}),self.assertRaises(auth.PlanAuthError) as raised:
                await p.access_token(session)
            self.assertIn("SIGN_IN_REQUIRED",raised.exception.code)
            self.assertFalse(credential.exists())
            self.assertEqual(len(session.calls),1)

    async def test_54_temporary_refresh_failure_keeps_fake_file(self):
        with tempfile.TemporaryDirectory() as d:
            credential=Path(d)/"credentials.dpapi";credential.write_bytes(b"synthetic-test-not-a-token")
            p=MemoryProfile(d,{"record":record(saved_at=time.time()-3600)})
            session=FakeSession(post=[FakeResponse({"error":{"code":"temporary_failure","message":REFRESH}},status=503)])
            with patch.object(auth,"getproxies",return_value={}),self.assertRaises(auth.PlanAuthError) as raised:
                await p.access_token(session)
            self.assertTrue(credential.exists())
            self.assertEqual(len(session.calls),1)
            self.assertNotIn(REFRESH,str(raised.exception))

    async def test_55_reduced_scopes_never_saved(self):
        with tempfile.TemporaryDirectory() as d:
            state={"record":record(saved_at=time.time()-3600)};p=MemoryProfile(d,state)
            session=FakeSession(post=[FakeResponse(dict(access_token="new",refresh_token="new-refresh",token_type="Bearer",expires_in=3600,scope="openid profile"))])
            with patch.object(auth,"getproxies",return_value={}),self.assertRaises(auth.PlanAuthError):
                await p.access_token(session)
            self.assertEqual(p.writes,0)
            self.assertEqual(state["record"]["refresh_token"],REFRESH)

    async def test_56_earliest_refresh_prevents_early_refresh(self):
        with tempfile.TemporaryDirectory() as d:
            state={"record":record(saved_at=time.time()-3550,earliest_refresh_at=time.time()+20)}
            p=MemoryProfile(d,state);session=FakeSession()
            self.assertEqual(await p.access_token(session),BEARER)
            self.assertEqual(session.calls,[])


class ProductionAuthJWTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
        public=jwt.algorithms.RSAAlgorithm.to_jwk(cls.key.public_key(),as_dict=True)
        public.update(kid="fixture-signing-key",use="sig",alg="RS256")
        cls.jwks={"keys":[public]}

    def claims(self,**changes):
        now=int(time.time())
        value=dict(iss="https://auth.openai.com",aud="oaiapp_fixture",sub="fixture-subject",nonce="fixture-nonce",iat=now-5,exp=now+300)
        value.update(changes);return value

    def signed(self,claims=None,key=None,**changes):
        return jwt.encode(claims or self.claims(**changes),key or self.key,algorithm="RS256",headers={"kid":"fixture-signing-key"})

    def validate(self,token):
        return auth.validate_id_token(token,self.jwks,"oaiapp_fixture","fixture-nonce")

    def test_65_valid_signed_new_module_identity(self):
        self.assertEqual(self.validate(self.signed())["sub"],"fixture-subject")

    def test_66_wrong_issuer_rejected(self):
        with self.assertRaises(auth.PlanAuthError):self.validate(self.signed(iss="https://example.invalid"))

    def test_67_wrong_audience_rejected(self):
        with self.assertRaises(auth.PlanAuthError):self.validate(self.signed(aud="oaiapp_other"))

    def test_68_wrong_or_non_ascii_nonce_rejected(self):
        for nonce in ("another-nonce","不相同"):
            with self.subTest(nonce=nonce),self.assertRaises(auth.PlanAuthError):self.validate(self.signed(nonce=nonce))

    def test_69_expired_or_future_identity_rejected(self):
        for changes in (dict(exp=int(time.time())-300),dict(iat=int(time.time())+600)):
            with self.subTest(changes=changes),self.assertRaises(auth.PlanAuthError):self.validate(self.signed(**changes))

    def test_70_invalid_signature_rejected(self):
        other=rsa.generate_private_key(public_exponent=65537,key_size=2048)
        with self.assertRaises(auth.PlanAuthError):self.validate(self.signed(key=other))

    def test_71_unsigned_or_symmetric_identity_rejected(self):
        for algorithm,key in (("none",""),("HS256","offline-symmetric-fixture-not-a-real-key-123456")):
            token=jwt.encode(self.claims(),key,algorithm=algorithm,headers={"kid":"fixture-signing-key"})
            with self.subTest(algorithm=algorithm),self.assertRaises(auth.PlanAuthError):self.validate(token)

    def test_72_required_claims_not_inferred(self):
        for name in ("exp","iat","iss","aud","sub","nonce"):
            claims=self.claims();del claims[name]
            with self.subTest(name=name),self.assertRaises(auth.PlanAuthError):self.validate(self.signed(claims=claims))

    def test_73_multiaudience_requires_matching_azp(self):
        with self.assertRaises(auth.PlanAuthError):self.validate(self.signed(aud=["oaiapp_fixture","other"]))
        good=self.signed(aud=["oaiapp_fixture","other"],azp="oaiapp_fixture")
        self.assertEqual(self.validate(good)["sub"],"fixture-subject")

    def test_74_refresh_identity_without_nonce_can_still_verify_signature(self):
        claims=self.claims();del claims["nonce"]
        result=auth.validate_id_token(self.signed(claims=claims),self.jwks,"oaiapp_fixture")
        self.assertEqual(result["sub"],"fixture-subject")


class ProfileDirectoryTests(unittest.TestCase):
    def test_75_unrelated_directory_rejected_before_acl_change(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/"unrelated-user-document.txt").write_text("do not change ACL")
            p=auth.PlanProfile(d)
            with patch.object(auth,"_private_acl") as acl,self.assertRaises(auth.PlanAuthError) as raised:p._prepare()
            self.assertEqual(raised.exception.code,"PROFILE_DIRECTORY_NOT_DEDICATED")
            acl.assert_not_called()

    def test_76_repository_directory_never_becomes_credential_directory(self):
        p=auth.PlanProfile(SOURCE)
        with patch.object(auth,"_private_acl") as acl,self.assertRaises(auth.PlanAuthError) as raised:p._prepare()
        self.assertEqual(raised.exception.code,"PROFILE_MUST_BE_OUTSIDE_SOURCE")
        acl.assert_not_called()

    def test_77_own_dedicated_directory_can_be_prepared(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/"registration.json").write_text("{}")
            p=auth.PlanProfile(d)
            with patch.object(auth,"_private_acl") as acl:p._prepare()
            acl.assert_called_once_with(Path(d))
            self.assertTrue(p._prepared)


class ExplicitImportTests(unittest.IsolatedAsyncioTestCase):
    def source(self,path,**changes):
        value=record();reg=dict(app="QQ Roast Bot",client_id=value["client_id"],ext_agent_host_id=value["ext_agent_host_id"],identity_validated=True)
        reg.update(changes)
        (path/"registration.json").write_text(json.dumps(reg))
        (path/"status.json").write_text(json.dumps(dict(identity_verified=True,plan_permission_verified=True)))
        (path/"credentials.dpapi").write_bytes(auth.encrypt_record(value))
        return value

    async def test_78_explicit_import_checks_and_writes_inside_process_lock(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/"synthetic-source";source.mkdir();self.source(source)
            target=Path(d)/"synthetic-target";events=[]
            class Target:
                path=target
                @asynccontextmanager
                async def _process_lock(self,timeout=20):
                    events.append("lock_enter")
                    try:yield
                    finally:events.append("lock_exit")
                def exists(self):
                    self.assert_lock();events.append("exists");return False
                def assert_lock(self):
                    if not events or events[0]!="lock_enter" or events[-1]=="lock_exit":raise AssertionError("not locked")
                def _write(self,value):
                    self.assert_lock();events.append("write")
                    if value["subject"]!="fixture-subject":raise AssertionError("wrong synthetic identity")
            with patch.object(auth,"PlanProfile",return_value=Target()):
                outcome=await auth.import_verified_profile_async(source,target)
            self.assertTrue(outcome["imported"])
            self.assertEqual(outcome["inference_requests"],0)
            self.assertEqual(events,["lock_enter","exists","write","lock_exit"])

    async def test_79_unverified_source_not_imported(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d);self.source(source,identity_validated="true")
            with patch.object(auth,"PlanProfile") as target,self.assertRaises(auth.PlanAuthError):
                await auth.import_verified_profile_async(source,source/"unused-target")
            target.assert_not_called()

    async def test_80_different_existing_account_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/"synthetic-source";source.mkdir();self.source(source)
            target=Path(d)/"synthetic-target"
            class Target:
                path=target
                @asynccontextmanager
                async def _process_lock(self,timeout=20):yield
                def exists(self):return True
                def _read(self):return record(subject="different-account")
                def _write(self,value):raise AssertionError("Must not overwrite another account")
            with patch.object(auth,"PlanProfile",return_value=Target()),self.assertRaises(auth.PlanAuthError) as raised:
                await auth.import_verified_profile_async(source,target)
            self.assertEqual(raised.exception.code,"TARGET_HAS_DIFFERENT_ACCOUNT")


if __name__=="__main__":
    output=io.StringIO()
    errors=io.StringIO()
    with contextlib.redirect_stderr(errors):
        result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    ledger={"kind":"OFFLINE_CHATGPT_PLAN_CONTRACT_TESTS_NOT_LIVE_MODEL_ACCURACY","source":str(SOURCE),"checks":result.testsRun,"passed":result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),"real_model_calls":0,"real_network_calls":0,"browser_opened":False,"real_credentials_read":False,"exit_status":0 if result.wasSuccessful() else 1}
    summary=json.dumps({k:v for k,v in ledger.items() if k!="source"},ensure_ascii=False)+"\n"
    ledger.update(stdout=output.getvalue()+summary,stderr=errors.getvalue())
    if RESULT:RESULT.write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding="utf-8")
    sys.stdout.write(ledger["stdout"])
    sys.stderr.write(ledger["stderr"])
    raise SystemExit(ledger["exit_status"])
