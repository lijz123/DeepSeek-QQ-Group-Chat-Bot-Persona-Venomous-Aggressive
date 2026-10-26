"""Offline launcher safety tests: no real QQ/process inventory, credentials or APIs.

All process descriptions and backend actions below are synthetic. Stop-file
tests run only in-process fake coroutines and use a dedicated temporary folder.
"""
from __future__ import annotations

from layout_support import app_directory, project_directory, configure_test_paths
configure_test_paths()

import asyncio
import contextlib
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import launcher


OWNER = r"OFFLINE-PC\test-user"
PROJECT = Path(r"C:\offline-launcher-fixtures\QQ Bot")
NAPCAT = Path(r"C:\offline-launcher-fixtures\napcat")
PYTHON = r"C:\offline-python\python.exe"


def process(pid=10, ppid=1, *, name="python.exe", exe=PYTHON,
            cwd=PROJECT, argv=None, created=1234.5, owner=OWNER):
    return launcher.ProcessInfo(pid=pid, ppid=ppid, name=name, exe=str(exe),
        cwd=str(cwd), argv=tuple(argv or (PYTHON, str(PROJECT / "bot.py"))),
        created=created, owner=owner)


class BotIdentityTests(unittest.TestCase):
    def match(self, value):
        return launcher.is_bot_process(value, PROJECT, OWNER)

    def test_01_absolute_project_script(self):
        self.assertTrue(self.match(process()))

    def test_02_relative_script_requires_correct_cwd(self):
        self.assertTrue(self.match(process(argv=(PYTHON, "bot.py", "--config", "config.json"))))

    def test_03_relative_script_in_other_project_not_owned(self):
        self.assertFalse(self.match(process(cwd=PROJECT.parent / "another", argv=(PYTHON, "bot.py"))))

    def test_04_absolute_other_bot_not_owned(self):
        self.assertFalse(self.match(process(argv=(PYTHON, str(PROJECT.parent / "another" / "bot.py")))))

    def test_05_project_name_substring_does_not_match(self):
        self.assertFalse(self.match(process(argv=(PYTHON, str(Path(str(PROJECT) + "-copy") / "bot.py")))))

    def test_06_not_just_any_python(self):
        self.assertFalse(self.match(process(argv=(PYTHON, str(PROJECT / "other.py")))))

    def test_07_bot_script_as_argument_is_not_executed_script(self):
        self.assertFalse(self.match(process(argv=(PYTHON, str(PROJECT / "other.py"), str(PROJECT / "bot.py")))))

    def test_08_python_command_string_never_matches(self):
        self.assertFalse(self.match(process(argv=(PYTHON, "-c", "print('bot.py')", str(PROJECT / "bot.py")))))

    def test_09_python_module_never_matches(self):
        self.assertFalse(self.match(process(argv=(PYTHON, "-m", "bot.py"))))

    def test_10_wrong_windows_account_not_owned(self):
        self.assertFalse(self.match(process(owner=r"OFFLINE-PC\someone-else")))

    def test_11_qq_never_matches_even_with_script_argument(self):
        self.assertFalse(self.match(process(name="QQ.exe", exe=str(NAPCAT / "qq_extract" / "Files" / "QQ.exe"))))

    def test_12_python_name_spoof_without_python_executable_rejected(self):
        self.assertFalse(self.match(process(exe=str(PROJECT / "other.exe"))))

    def test_13_absolute_target_script_may_have_different_cwd(self):
        self.assertTrue(self.match(process(cwd=PROJECT.parent,
            argv=(PYTHON, str(PROJECT / "bot.py"), "--config", str(PROJECT / "config.json")))))

    def test_14_empty_commandline_not_owned(self):
        value = dataclasses.replace(process(), argv=())
        self.assertFalse(self.match(value))

    def test_15_unknown_cwd_for_relative_script_not_owned(self):
        self.assertFalse(self.match(process(cwd="", argv=(PYTHON, "bot.py"))))

    def test_16_unbuffered_python_script_supported(self):
        self.assertTrue(self.match(process(argv=(PYTHON, "-u", str(PROJECT / "bot.py")))))

    def test_17_bot_py_backup_filename_not_owned(self):
        self.assertFalse(self.match(process(argv=(PYTHON, str(PROJECT / "bot.py.bak")))))

    def test_18_wrong_pid_zero_not_owned(self):
        self.assertFalse(self.match(process(pid=0)))


class NapCatIdentityTests(unittest.TestCase):
    def napcat(self, **values):
        defaults = dict(name="QQ.exe", exe=str(NAPCAT / "qq_extract" / "Files" / "QQ.exe"),
                        cwd=NAPCAT, argv=(str(NAPCAT / "qq_extract" / "Files" / "QQ.exe"), "--no-sandbox"))
        defaults.update(values)
        return process(**defaults)

    def match(self, value):
        return launcher.is_napcat_process(value, NAPCAT, OWNER)

    def test_19_exact_napcat_installation(self):
        self.assertTrue(self.match(self.napcat()))

    def test_20_regular_desktop_qq_is_not_napcat(self):
        self.assertFalse(self.match(self.napcat(exe=r"D:\QQ.exe")))

    def test_21_qq_installation_prefix_lookalike_rejected(self):
        self.assertFalse(self.match(self.napcat(exe=str(Path(str(NAPCAT) + "-other") / "qq_extract" / "Files" / "QQ.exe"))))

    def test_22_renderer_not_a_second_napcat(self):
        self.assertFalse(self.match(self.napcat(argv=("QQ.exe", "--type=renderer"))))

    def test_23_gpu_child_not_a_second_napcat(self):
        self.assertFalse(self.match(self.napcat(argv=("QQ.exe", "--type=gpu-process"))))

    def test_24_other_account_napcat_not_owned(self):
        self.assertFalse(self.match(self.napcat(owner=r"OFFLINE-PC\another")))


class FakeBackend:
    def __init__(self, *, bots=(), napcat=False, ready=True, preflight_error=False,
                 retain_stopped=False, start_error=False):
        self.current = list(bots)
        self.napcat = napcat
        self.ready = ready
        self.preflight_error = preflight_error
        self.retain_stopped = retain_stopped
        self.start_error = start_error
        self.events = []
        self.started = process(pid=900, created=2000)

    def preflight(self):
        self.events.append("preflight")
        if self.preflight_error:
            raise launcher.LauncherError("OFFLINE_PREFLIGHT_FAILURE")

    def bots(self):
        self.events.append("scan_bots")
        return list(self.current)

    def napcat_running(self):
        self.events.append("napcat_running")
        return self.napcat

    def start_napcat(self):
        self.events.append("start_napcat")
        self.napcat = True

    def wait_napcat(self):
        self.events.append("wait_napcat")
        return self.ready

    def stop_bot(self, info):
        self.events.append(("stop_bot", info.pid))
        if not self.retain_stopped:
            self.current = [p for p in self.current if p.pid != info.pid]

    def start_bot(self):
        self.events.append("start_bot")
        if self.start_error:
            raise launcher.LauncherError("OFFLINE_START_FAILURE")
        self.current.append(self.started)
        return self.started


class OrchestrationTests(unittest.TestCase):
    def run_case(self, **options):
        backend = FakeBackend(**options)
        result = launcher.operate(backend, action="restart")
        self.assertIsInstance(result, dict)
        return backend, result

    def test_25_neither_started_starts_napcat_then_one_bot(self):
        backend, _ = self.run_case()
        self.assertEqual(backend.events.count("start_napcat"), 1)
        self.assertEqual(backend.events.count("start_bot"), 1)
        self.assertLess(backend.events.index("start_napcat"), backend.events.index("start_bot"))

    def test_26_both_started_keep_napcat_restart_only_bot(self):
        backend, _ = self.run_case(napcat=True, bots=[process(pid=100)])
        self.assertNotIn("start_napcat", backend.events)
        self.assertEqual([e for e in backend.events if isinstance(e, tuple)], [("stop_bot", 100)])
        self.assertEqual(backend.events.count("start_bot"), 1)

    def test_27_napcat_only_starts_bot_without_napcat_duplicate(self):
        backend, _ = self.run_case(napcat=True)
        self.assertNotIn("start_napcat", backend.events)
        self.assertEqual(backend.events.count("start_bot"), 1)

    def test_28_bot_only_starts_napcat_before_stopping_bot(self):
        backend, _ = self.run_case(bots=[process(pid=100)])
        self.assertLess(backend.events.index("start_napcat"), backend.events.index(("stop_bot", 100)))

    def test_29_preflight_failure_never_stops_current_bot(self):
        backend = FakeBackend(napcat=True, bots=[process(pid=100)], preflight_error=True)
        with self.assertRaises(launcher.LauncherError):
            launcher.operate(backend)
        self.assertEqual(backend.events, ["preflight"])

    def test_30_napcat_not_ready_never_stops_old_bot(self):
        backend = FakeBackend(napcat=True, bots=[process(pid=100)], ready=False)
        with self.assertRaises(launcher.LauncherError):
            launcher.operate(backend)
        self.assertNotIn(("stop_bot", 100), backend.events)
        self.assertNotIn("start_bot", backend.events)

    def test_31_core_python_stopped_before_manager_wrapper(self):
        wrapper, core = process(pid=100), process(pid=101, ppid=100)
        backend, _ = self.run_case(napcat=True, bots=[wrapper, core])
        self.assertLess(backend.events.index(("stop_bot", 101)), backend.events.index(("stop_bot", 100)))

    def test_32_multiple_prior_instances_become_one(self):
        backend, _ = self.run_case(napcat=True, bots=[process(pid=100), process(pid=110)])
        self.assertEqual(len([e for e in backend.events if isinstance(e, tuple)]), 2)
        self.assertEqual(backend.events.count("start_bot"), 1)
        self.assertEqual([p.pid for p in backend.current], [900])

    def test_33_stale_bot_remaining_blocks_duplicate_launch(self):
        backend = FakeBackend(napcat=True, bots=[process(pid=100)], retain_stopped=True)
        with self.assertRaises(launcher.LauncherError):
            launcher.operate(backend)
        self.assertNotIn("start_bot", backend.events)

    def test_34_wait_ready_precedes_stop(self):
        backend, _ = self.run_case(napcat=True, bots=[process(pid=100)])
        self.assertLess(backend.events.index("wait_napcat"), backend.events.index(("stop_bot", 100)))

    def test_35_preflight_is_first_action(self):
        backend, _ = self.run_case(napcat=True)
        self.assertEqual(backend.events[0], "preflight")

    def test_36_start_failure_is_observed_not_reported_success(self):
        backend = FakeBackend(napcat=True, start_error=True)
        with self.assertRaises(launcher.LauncherError):
            launcher.operate(backend)
        self.assertEqual(backend.events.count("start_bot"), 1)


class FakeBot:
    def __init__(self, *, finish=False, fail=False):
        self.started = asyncio.Event()
        self.stopped = False
        self.finish = finish
        self.fail = fail

    async def run(self):
        self.started.set()
        try:
            if self.fail:
                raise RuntimeError("synthetic fake-bot failure")
            if self.finish:
                return
            await asyncio.Event().wait()
        finally:
            self.stopped = True


class ManagedStopTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qq-launcher-offline-")
        self.folder = Path(self.temp.name).resolve()
        # Verify the cleanup target before TemporaryDirectory's recursive cleanup.
        self.assertEqual(self.folder.parent, Path(tempfile.gettempdir()).resolve())
        self.stop = self.folder / "owned-stop-file"
        self.guard = mock.patch.object(launcher, "_safe_stop_path", side_effect=lambda project, path: Path(path))
        self.guard.start()

    async def asyncTearDown(self):
        self.guard.stop()
        self.temp.cleanup()

    async def test_37_stop_file_cancels_fake_bot_and_runs_cleanup(self):
        bot = FakeBot()
        task = asyncio.create_task(launcher.run_bot_until_stop(bot, self.stop))
        try:
            await asyncio.wait_for(bot.started.wait(), timeout=2)
            self.stop.write_text("stop", encoding="ascii")
            await asyncio.wait_for(task, timeout=3)
            self.assertTrue(bot.stopped)
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def test_38_normal_bot_exit_returns_without_waiting_for_stop_file(self):
        bot = FakeBot(finish=True)
        await asyncio.wait_for(launcher.run_bot_until_stop(bot, self.stop), timeout=2)
        self.assertTrue(bot.stopped)

    async def test_39_bot_failure_is_not_claimed_as_successful_start(self):
        bot = FakeBot(fail=True)
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(launcher.run_bot_until_stop(bot, self.stop), timeout=2)
        self.assertTrue(bot.stopped)

    async def test_40_parent_cancellation_cleans_up_fake_bot(self):
        bot = FakeBot()
        task = asyncio.create_task(launcher.run_bot_until_stop(bot, self.stop))
        await asyncio.wait_for(bot.started.wait(), timeout=2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(bot.stopped)


class StopPathTests(unittest.TestCase):
    def test_41_control_path_is_scoped_to_exact_project(self):
        path = PROJECT / "logs" / "oneclick" / ("stop-" + "a" * 32 + ".flag")
        self.assertEqual(launcher._safe_stop_path(PROJECT, path), path.resolve())

    def test_42_arbitrary_file_cannot_be_used_as_stop_signal(self):
        with self.assertRaises(launcher.LauncherError):
            launcher._safe_stop_path(PROJECT, PROJECT / "config.json")

    def test_43_other_projects_stop_file_is_refused(self):
        with self.assertRaises(launcher.LauncherError):
            launcher._safe_stop_path(PROJECT, PROJECT.parent / "other" / "logs" / "oneclick" / ("stop-" + "a" * 32 + ".flag"))

    def test_44_wrong_stop_filename_is_refused(self):
        for filename in ("stop.flag", "stop-abc.flag", "stop-" + "g" * 32 + ".flag", "stop-" + "a" * 32 + ".txt"):
            with self.subTest(filename=filename), self.assertRaises(launcher.LauncherError):
                launcher._safe_stop_path(PROJECT, PROJECT / "logs" / "oneclick" / filename)

    def test_45_parent_traversal_is_refused(self):
        with self.assertRaises(launcher.LauncherError):
            launcher._safe_stop_path(PROJECT, PROJECT / "logs" / "oneclick" / ".." / ("stop-" + "a" * 32 + ".flag"))


class MockNoSuchProcess(Exception):
    pass


class MockAccessDenied(Exception):
    pass


class FreshIdentityTests(unittest.TestCase):
    def setUp(self):
        self.backend = object.__new__(launcher.RealBackend)
        self.backend.project = PROJECT
        self.backend.owner = OWNER
        self.expected = process()

    @staticmethod
    def fake_process(info):
        value = mock.Mock()
        value.pid = info.pid
        value.ppid.return_value = info.ppid
        value.name.return_value = info.name
        value.exe.return_value = info.exe
        value.cwd.return_value = info.cwd
        value.cmdline.return_value = list(info.argv)
        value.create_time.return_value = info.created
        value.username.return_value = info.owner
        return value

    def fresh(self, info=None, error=None):
        value = self.fake_process(info or self.expected)
        psutil = mock.Mock(NoSuchProcess=MockNoSuchProcess, AccessDenied=MockAccessDenied)
        psutil.Process.side_effect = error
        if error is None:
            psutil.Process.return_value = value
        with mock.patch.object(launcher, "psutil", psutil):
            result = self.backend._fresh_target(self.expected)
        return result, value

    def test_46_matching_pid_creation_executable_and_target_permits_exact_handle(self):
        observed, value = self.fresh()
        self.assertIs(observed, value)
        value.terminate.assert_not_called()

    def test_47_reused_pid_with_new_creation_time_is_not_stopped(self):
        with self.assertRaises(launcher.LauncherError):
            self.fresh(dataclasses.replace(self.expected, created=self.expected.created + 10))

    def test_48_changed_account_is_not_stopped(self):
        with self.assertRaises(launcher.LauncherError):
            self.fresh(dataclasses.replace(self.expected, owner=r"OFFLINE-PC\someone-else"))

    def test_49_changed_script_is_not_stopped(self):
        with self.assertRaises(launcher.LauncherError):
            self.fresh(dataclasses.replace(self.expected, argv=(PYTHON, str(PROJECT / "other.py"))))

    def test_50_changed_interpreter_is_not_stopped(self):
        with self.assertRaises(launcher.LauncherError):
            self.fresh(dataclasses.replace(self.expected, exe=r"C:\other-python\python.exe"))

    def test_51_process_already_exited_is_a_noop(self):
        observed, value = self.fresh(error=MockNoSuchProcess())
        self.assertIsNone(observed)
        value.terminate.assert_not_called()

    def test_52_permission_failure_is_not_treated_as_safe_to_stop(self):
        with self.assertRaises(launcher.LauncherError):
            self.fresh(error=MockAccessDenied())

    def test_53_nonfinite_expected_creation_time_is_refused(self):
        self.expected = dataclasses.replace(self.expected, created=float("nan"))
        with self.assertRaises(launcher.LauncherError):
            self.fresh()

    def test_54_nonfinite_observed_creation_time_is_refused(self):
        with self.assertRaises(launcher.LauncherError):
            self.fresh(dataclasses.replace(self.expected, created=float("nan")))


class ConfigIdentityTests(unittest.TestCase):
    def match(self, argv, cwd=PROJECT):
        return launcher.is_bot_process(process(argv=argv, cwd=cwd), PROJECT, OWNER)

    def test_55_different_config_is_not_owned_bot(self):
        self.assertFalse(self.match((PYTHON, str(PROJECT / "bot.py"), "--config", str(PROJECT / "different.json"))))

    def test_56_relative_config_resolves_against_process_cwd(self):
        self.assertTrue(self.match((PYTHON, "bot.py", "--config", "config.json")))

    def test_57_absolute_script_with_default_other_cwd_config_is_refused(self):
        self.assertFalse(self.match((PYTHON, str(PROJECT / "bot.py")), PROJECT.parent))

    def test_58_equals_style_config_is_supported(self):
        self.assertTrue(self.match((PYTHON, str(PROJECT / "bot.py"), "--config=" + str(PROJECT / "config.json"))))

    def test_59_equals_style_other_config_is_refused(self):
        self.assertFalse(self.match((PYTHON, str(PROJECT / "bot.py"), "--config=" + str(PROJECT.parent / "other.json"))))

    def test_60_repeated_config_argument_is_ambiguous_and_refused(self):
        self.assertFalse(self.match((PYTHON, str(PROJECT / "bot.py"), "--config", "config.json", "--config", "different.json")))

    def test_61_missing_config_operand_is_refused(self):
        self.assertFalse(self.match((PYTHON, str(PROJECT / "bot.py"), "--config")))


class LauncherLockTests(unittest.TestCase):
    def test_62_second_click_cannot_acquire_same_live_lock(self):
        with tempfile.TemporaryDirectory(prefix="qq-launcher-lock-offline-") as raw:
            folder = Path(raw).resolve()
            self.assertEqual(folder.parent, Path(tempfile.gettempdir()).resolve())
            path = folder / "launcher.lock"
            with launcher.ExclusiveLauncherLock(path):
                with self.assertRaises(launcher.LauncherError):
                    with launcher.ExclusiveLauncherLock(path):
                        self.fail("Second launcher entered a held lock")

    def test_63_released_lock_allows_next_click(self):
        with tempfile.TemporaryDirectory(prefix="qq-launcher-lock-offline-") as raw:
            folder = Path(raw).resolve()
            self.assertEqual(folder.parent, Path(tempfile.gettempdir()).resolve())
            path = folder / "launcher.lock"
            with launcher.ExclusiveLauncherLock(path):
                pass
            with launcher.ExclusiveLauncherLock(path):
                pass


class StopMutationTests(unittest.TestCase):
    @contextlib.contextmanager
    def backend(self):
        with tempfile.TemporaryDirectory(prefix="qq-launcher-stop-offline-") as raw:
            folder = Path(raw).resolve()
            self.assertEqual(folder.parent, Path(tempfile.gettempdir()).resolve())
            backend = object.__new__(launcher.RealBackend)
            backend.project = folder
            backend.runtime = folder / "logs" / "oneclick"
            backend.owner = OWNER
            yield backend

    def test_64_legacy_stop_rechecks_identity_immediately_before_terminate(self):
        with self.backend() as backend:
            fake_process = mock.Mock()
            with mock.patch.object(backend, "_fresh_target", return_value=fake_process) as fresh:
                backend.stop_bot(process())
            self.assertEqual(fresh.call_count, 2)
            fake_process.terminate.assert_called_once_with()
            fake_process.wait.assert_called_once_with(timeout=5)

    def test_65_identity_changed_between_first_read_and_stop_never_terminates(self):
        with self.backend() as backend:
            fake_process = mock.Mock()
            with mock.patch.object(backend, "_fresh_target", side_effect=[fake_process,
                    launcher.LauncherError("PROCESS_IDENTITY_CHANGED_NO_STOP")]):
                with self.assertRaises(launcher.LauncherError):
                    backend.stop_bot(process())
            fake_process.terminate.assert_not_called()

    def test_66_already_exited_bot_does_not_mutate_other_processes(self):
        with self.backend() as backend:
            with mock.patch.object(backend, "_fresh_target", return_value=None) as fresh:
                backend.stop_bot(process())
            fresh.assert_called_once()
            self.assertFalse(backend.runtime.exists())

    def test_67_managed_stop_uses_owned_flag_and_avoids_terminate_on_clean_exit(self):
        with self.backend() as backend:
            backend.runtime.mkdir(parents=True)
            stop = backend.runtime / ("stop-" + "a" * 32 + ".flag")
            expected = process()
            state = {"pid": expected.pid, "created": expected.created,
                     "project": str(backend.project), "stop_file": str(stop)}
            (backend.runtime / "managed-bot.json").write_text(json.dumps(state), encoding="utf-8")
            fake_process = mock.Mock()
            with mock.patch.object(backend, "_fresh_target", return_value=fake_process) as fresh:
                backend.stop_bot(expected)
            self.assertEqual(stop.read_text(encoding="ascii"), "stop\n")
            self.assertEqual(fresh.call_count, 1)
            fake_process.wait.assert_called_once_with(timeout=8)
            fake_process.terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
