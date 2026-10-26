"""Offline login-progress tests: fake OneBot, clocks, UI and temporary CMD fixture.

Native CMD runs a disposable copy beside a synthetic app/bot.py, never the real
launcher, QQ, credentials or model client. No source files outside this test are
edited. Usage: python launcher_progress_tests.py [project_dir] [result.json]
"""
from __future__ import annotations

from layout_support import app_directory, project_directory, configure_test_paths
configure_test_paths()

import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

PROJECT = project_directory(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None)
RESULT = Path(sys.argv[2]) if len(sys.argv) > 2 else None
sys.path.insert(0, str(PROJECT))
import aiohttp
import launcher

PNG = b'\x89PNG\r\n\x1a\n' + b'offline synthetic image bytes'
FIXED_TIME = 1792879200.0


def backend(base: Path):
    value = launcher.RealBackend.__new__(launcher.RealBackend)
    value.project = base / 'project'
    value.napcat = base / 'napcat'
    value.runtime = value.project / 'logs' / 'oneclick'
    value.settings = {'wait_seconds': 25}
    value.cfg = {'onebot': {'ws_url': 'ws://127.0.0.1:3001', 'access_token': ''}}
    return value


class FakeClock:
    def __init__(self):
        self.now = 0.0
    def monotonic(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.b = backend(Path(self.temp.name))
        self.clock = FakeClock()

    def wait(self, *, ready_after=None, running=True, issue='not_connected'):
        calls = []
        async def ready():
            calls.append(1)
            self.b._last_onebot_issue = issue
            return ready_after is not None and len(calls) >= ready_after
        output = io.StringIO()
        with mock.patch.object(self.b, 'napcat_running', return_value=running), \
             mock.patch.object(self.b, '_onebot_ready', side_effect=ready) as check, \
             mock.patch.object(self.b, '_maybe_show_login_qr', return_value=False) as qr, \
             mock.patch.object(launcher.time, 'monotonic', side_effect=self.clock.monotonic), \
             mock.patch.object(launcher.time, 'sleep', side_effect=self.clock.sleep), \
             contextlib.redirect_stdout(output):
            result = self.b.wait_napcat()
        return result, output.getvalue(), check, qr

    def test_ready_immediately_returns_without_qr(self):
        result, text, check, qr = self.wait(ready_after=1)
        self.assertTrue(result); self.assertIn('已就绪', text)
        self.assertEqual(check.call_count, 1); qr.assert_not_called()

    def test_pending_login_prints_progress_and_times_out(self):
        result, text, check, qr = self.wait()
        self.assertFalse(result)
        self.assertIn('已等待 0 秒', text); self.assertIn('已等待 10 秒', text)
        self.assertIn('已等待 20 秒', text); self.assertIn('还剩约', text)
        self.assertIn('扫码', text); self.assertIn('等待结束', text)
        self.assertEqual(self.clock.now, 25); self.assertGreater(check.call_count, 1)

    def test_eventual_login_returns_before_timeout(self):
        result, text, check, qr = self.wait(ready_after=3, issue='login_pending')
        self.assertTrue(result); self.assertEqual(check.call_count, 3)
        self.assertIn('请扫码', text); self.assertEqual(self.clock.now, 2)

    def test_missing_process_does_not_probe_onebot(self):
        result, text, check, qr = self.wait(running=False)
        self.assertFalse(result); check.assert_not_called(); qr.assert_not_called()
        self.assertIn('主进程', text)

    def test_auth_mismatch_has_actionable_explanation(self):
        result, text, _, _ = self.wait(issue='auth_failed')
        self.assertFalse(result); self.assertIn('access_token', text)
        self.assertNotIn('API key', text)

    def test_offline_has_login_explanation(self):
        result, text, _, _ = self.wait(issue='offline')
        self.assertFalse(result); self.assertIn('不在线', text)

    def test_invalid_response_explanation(self):
        _, text, _, _ = self.wait(issue='invalid_response')
        self.assertIn('不是有效的 OneBot', text)

    def test_timeout_explanation(self):
        _, text, _, _ = self.wait(issue='timeout')
        self.assertIn('没有及时返回', text)

    def test_wait_cap_prevents_unbounded_loop(self):
        self.b.settings['wait_seconds'] = 10000
        self.wait(running=False)
        self.assertEqual(self.clock.now, 600)

    def test_wait_minimum_one_second(self):
        self.b.settings['wait_seconds'] = 0
        self.wait(running=False)
        self.assertEqual(self.clock.now, 1)

    def test_starter_failure_raises_without_waiting_full_deadline(self):
        self.b.napcat_starter = SimpleNamespace(poll=lambda: 1)
        with self.assertRaisesRegex(launcher.LauncherError, 'NAPCAT_START_FAILED'):
            self.wait(running=False)
        self.assertEqual(self.clock.now, 0)

    def test_wait_itself_never_stops_or_starts_bot(self):
        self.b.stop_bot = mock.Mock(side_effect=AssertionError('must not stop'))
        self.b.start_bot = mock.Mock(side_effect=AssertionError('must not start'))
        self.wait()
        self.b.stop_bot.assert_not_called(); self.b.start_bot.assert_not_called()


class QrDisplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.b = backend(Path(self.temp.name))
        self.b._last_onebot_issue = 'login_pending'
        self.image = self.b.napcat / 'runtime' / 'cache' / 'qrcode.png'
        self.image.parent.mkdir(parents=True)
        self.image.write_bytes(PNG)
        os.utime(self.image, (FIXED_TIME-1, FIXED_TIME-1))

    def show(self, *, opens=True):
        text = io.StringIO()
        with mock.patch.object(launcher.time, 'time', return_value=FIXED_TIME), \
             mock.patch.object(launcher, '_open_login_qr', return_value=opens) as call, \
             contextlib.redirect_stdout(text):
            result = self.b._maybe_show_login_qr()
        return result, text.getvalue(), call

    def test_fresh_local_png_is_shown(self):
        result, text, call = self.show()
        self.assertTrue(result); call.assert_called_once_with(self.image)
        self.assertIn('手机', text); self.assertIn('确认登录', text)

    def test_same_run_opens_qr_only_once(self):
        self.show()
        result, _, call = self.show()
        self.assertFalse(result); call.assert_not_called()

    def test_expired_png_is_not_shown(self):
        os.utime(self.image, (FIXED_TIME-200, FIXED_TIME-200))
        result, _, call = self.show()
        self.assertFalse(result); call.assert_not_called()

    def test_future_timestamp_is_not_shown(self):
        os.utime(self.image, (FIXED_TIME+100, FIXED_TIME+100))
        result, _, call = self.show()
        self.assertFalse(result); call.assert_not_called()

    def test_wrong_signature_is_not_shown(self):
        self.image.write_bytes(b'NOT-A-PNG' + b'a'*20)
        result, _, call = self.show()
        self.assertFalse(result); call.assert_not_called()

    def test_empty_file_is_not_shown(self):
        self.image.write_bytes(b'')
        result, _, call = self.show()
        self.assertFalse(result); call.assert_not_called()

    def test_missing_image_is_not_shown(self):
        self.image.unlink()
        result, _, call = self.show()
        self.assertFalse(result); call.assert_not_called()

    def test_auth_error_does_not_show_stale_login_image(self):
        self.b._last_onebot_issue = 'auth_failed'
        result, _, call = self.show()
        self.assertFalse(result); call.assert_not_called()

    def test_ready_state_does_not_show_qr(self):
        self.b._last_onebot_issue = 'ready'
        result, _, call = self.show()
        self.assertFalse(result); call.assert_not_called()

    def test_image_open_failure_has_manual_fallback(self):
        result, text, call = self.show(opens=False)
        self.assertFalse(result); call.assert_called_once()
        self.assertIn('无法自动显示', text); self.assertIn(str(self.image), text)

    def test_link_guard_refusal_prevents_open(self):
        with mock.patch.object(launcher, 'reject_links', side_effect=launcher.LauncherError('LINKED_RUNTIME_PATH_REFUSED')):
            result, _, call = self.show()
        self.assertFalse(result); call.assert_not_called()

    def test_os_open_error_is_caught(self):
        text = io.StringIO()
        with mock.patch.object(launcher.time, 'time', return_value=FIXED_TIME), \
             mock.patch.object(launcher, '_open_login_qr', side_effect=OSError('synthetic viewer failure')), \
             contextlib.redirect_stdout(text):
            self.assertFalse(self.b._maybe_show_login_qr())
        self.assertIn('无法自动显示', text.getvalue())


class FakeWebSocket:
    def __init__(self, *, login=True, online=True, malformed=None, timeout=False, bad_detail=False):
        self.sent = []; self.login = login; self.online = online
        self.malformed = malformed; self.timeout = timeout; self.bad_detail = bad_detail
        self.received = 0
    async def __aenter__(self): return self
    async def __aexit__(self, *args): return False
    async def send_json(self, value): self.sent.append(value)
    async def receive(self, timeout=None):
        if self.timeout: raise asyncio.TimeoutError()
        if self.malformed: return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=self.malformed)
        item = self.sent[self.received]; self.received += 1
        detail = ({'user_id': 10001 if self.login else 0} if item['action']=='get_login_info'
                  else {'online': self.online})
        if self.bad_detail: detail = ['not a mapping']
        return SimpleNamespace(type=aiohttp.WSMsgType.TEXT,
            data=json.dumps({'status':'ok','retcode':0,'echo':item['echo'],'data':detail}))


class FakeSession:
    def __init__(self, ws, error=None): self.ws = ws; self.error = error; self.kwargs = None
    async def __aenter__(self): return self
    async def __aexit__(self, *args): return False
    def ws_connect(self, url, **kwargs):
        self.kwargs = kwargs
        if self.error: raise self.error
        return self.ws


class ReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def run_ready(self, *, error=None, **kwargs):
        b = backend(Path(tempfile.gettempdir()) / 'offline-launcher-tests-no-files')
        ws = FakeWebSocket(**kwargs); session = FakeSession(ws, error)
        with mock.patch.object(aiohttp, 'ClientSession', return_value=session):
            result = await b._onebot_ready()
        return result, b._last_onebot_issue, session, ws

    async def test_valid_login_and_online_ready(self):
        ready, issue, session, ws = await self.run_ready()
        self.assertTrue(ready); self.assertEqual(issue, 'ready')
        self.assertEqual({x['action'] for x in ws.sent}, {'get_login_info','get_status'})
        self.assertIsNone(session.kwargs['proxy'])

    async def test_no_user_id_is_login_pending(self):
        ready, issue, _, _ = await self.run_ready(login=False)
        self.assertFalse(ready); self.assertEqual(issue, 'login_pending')

    async def test_offline_not_ready(self):
        ready, issue, _, _ = await self.run_ready(online=False)
        self.assertFalse(ready); self.assertEqual(issue, 'offline')

    async def test_online_must_be_true_not_integer(self):
        ready, issue, _, _ = await self.run_ready(online=1)
        self.assertFalse(ready); self.assertEqual(issue, 'offline')

    async def test_timeout_is_classified(self):
        ready, issue, _, _ = await self.run_ready(timeout=True)
        self.assertFalse(ready); self.assertEqual(issue, 'timeout')

    async def test_connection_failure_is_classified(self):
        ready, issue, _, _ = await self.run_ready(error=aiohttp.ClientError('synthetic-connect-failure'))
        self.assertFalse(ready); self.assertEqual(issue, 'not_connected')

    async def test_auth_rejection_is_classified(self):
        error = aiohttp.WSServerHandshakeError(None, (), status=401, message='synthetic auth failure')
        ready, issue, _, _ = await self.run_ready(error=error)
        self.assertFalse(ready); self.assertEqual(issue, 'auth_failed')

    async def test_other_http_failure_is_not_connected(self):
        error = aiohttp.WSServerHandshakeError(None, (), status=503, message='synthetic unavailable')
        ready, issue, _, _ = await self.run_ready(error=error)
        self.assertFalse(ready); self.assertEqual(issue, 'not_connected')

    async def test_invalid_json_is_classified(self):
        ready, issue, _, _ = await self.run_ready(malformed='not-json')
        self.assertFalse(ready); self.assertEqual(issue, 'invalid_response')

    async def test_nonmapping_detail_is_classified(self):
        ready, issue, _, _ = await self.run_ready(bad_detail=True)
        self.assertFalse(ready); self.assertEqual(issue, 'invalid_response')

    async def test_cancellation_is_not_swallowed(self):
        with self.assertRaises(asyncio.CancelledError):
            await self.run_ready(error=asyncio.CancelledError())


class WrapperTests(unittest.TestCase):
    def test_wrapper_is_ascii(self):
        data = (PROJECT / 'start.bat').read_bytes()
        self.assertTrue(data.isascii())

    def test_wrapper_preserves_native_crlf(self):
        data = (PROJECT / 'start.bat').read_bytes()
        self.assertIn(b'\r\n', data)
        self.assertNotIn(b'\n', data.replace(b'\r\n', b''))

    def test_wrapper_uses_unbuffered_launcher(self):
        text = (PROJECT / 'start.bat').read_text(encoding='ascii')
        self.assertIn('python -u app/bot.py --config config.json', text)
        self.assertNotIn('taskkill', text.lower())

    def native(self, status):
        with tempfile.TemporaryDirectory(prefix='offline_cmd_wrapper_') as temp:
            base = Path(temp)
            source = (PROJECT / 'start.bat').read_bytes()
            # Fixed output path and harmless synthetic Python module ensure this
            # subprocess can never reach the actual launcher implementation.
            (base/'under-test.bat').write_bytes(source)
            # This fixture tests an already-configured launch, not first-run setup.
            (base/'config.json').write_text('{}', encoding='ascii')
            (base/'launcher.local.json').write_text('{"provider":"deepseek"}', encoding='ascii')
            (base/'app').mkdir()
            (base/'app'/'bot.py').write_text(
                "import os,sys\nprint('OFFLINE_STUB_ONLY',flush=True)\n"
                "raise SystemExit(int(os.environ['OFFLINE_STUB_EXIT']))\n", encoding='ascii')
            env = os.environ.copy(); env['OFFLINE_STUB_EXIT'] = str(status)
            env['PATH'] = str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
            env.pop('DEEPSEEK_API_KEY',None); env.pop('OPENAI_API_KEY',None)
            cmd = Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32'/'cmd.exe'
            result = subprocess.run([str(cmd),'/d','/c','under-test.bat'],cwd=base,env=env,
                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                timeout=15,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            return result.returncode,(result.stdout+result.stderr).decode('utf-8',errors='replace')

    @unittest.skipUnless(os.name=='nt','Native CMD only')
    def test_native_success_executes_only_synthetic_launcher(self):
        status,text = self.native(0)
        self.assertEqual(status,0); self.assertIn('OFFLINE_STUB_ONLY',text)
        self.assertIn('Done.',text); self.assertNotIn('not recognized',text)

    @unittest.skipUnless(os.name=='nt','Native CMD only')
    def test_native_failure_preserves_error_and_no_mangled_commands(self):
        status,text = self.native(7)
        self.assertEqual(status,7); self.assertIn('OFFLINE_STUB_ONLY',text)
        self.assertIn('Startup did not complete.',text); self.assertNotIn('not recognized',text)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
              'skipped':len(result.skipped),'success':result.wasSuccessful(),
              'network_calls':0,'real_QQ_actions':0,'real_launcher_runs':0,
              'native_cmd':'temporary wrapper + synthetic launcher.py only'}
    if RESULT: RESULT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
