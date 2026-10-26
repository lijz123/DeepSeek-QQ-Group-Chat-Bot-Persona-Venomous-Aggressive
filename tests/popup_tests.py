"""Offline popup contracts. No real dialogs, processes, credentials or network."""

from layout_support import app_directory, project_directory, configure_test_paths
configure_test_paths()
from pathlib import Path
import contextlib, ctypes, importlib.util, io, sys, types, unittest
from unittest.mock import Mock, patch

SOURCE = app_directory(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None)
sys.path.insert(0,str(SOURCE))
spec=importlib.util.spec_from_file_location('popup_launcher_under_test',SOURCE/'launcher.py')
launcher=importlib.util.module_from_spec(spec);sys.modules[spec.name]=launcher;spec.loader.exec_module(launcher)
sys.argv=[sys.argv[0]]

SUCCESS={'bot_started':True,'old_bot_count':0,'napcat_retained':True,'bot_pid':123,'model_test_requests':0}

class PopupTests(unittest.TestCase):
    def test_01_no_success_flag_does_not_show(self):
        with patch.object(launcher,'_show_message_box') as box:
            self.assertFalse(launcher.show_success_popup({'bot_started':False}))
            box.assert_not_called()

    def test_02_missing_success_flag_does_not_show(self):
        with patch.object(launcher,'_show_message_box') as box:
            self.assertFalse(launcher.show_success_popup({}))
            box.assert_not_called()

    def test_03_success_unicode_message_has_no_payload_secrets(self):
        value=dict(SUCCESS,api_key='SYNTHETIC_SECRET',arbitrary='SYNTHETIC_SECRET')
        with patch.object(launcher.os,'name','nt'),patch.object(launcher,'_show_message_box',return_value=1) as box:
            self.assertTrue(launcher.show_success_popup(value))
            title,text=box.call_args.args
            self.assertEqual(title,'启动成功');self.assertIn('机器人已启动成功',text)
            self.assertIn('NapCat 保持在线',text);self.assertNotIn('SYNTHETIC_SECRET',text)
            box.assert_called_once()

    def test_04_restart_message_is_distinct(self):
        with patch.object(launcher.os,'name','nt'),patch.object(launcher,'_show_message_box',return_value=1) as box:
            self.assertTrue(launcher.show_success_popup(dict(SUCCESS,old_bot_count=2)))
            self.assertIn('机器人已重启成功',box.call_args.args[1])

    def test_05_native_failure_is_false(self):
        with patch.object(launcher.os,'name','nt'),patch.object(launcher,'_show_message_box',return_value=0):
            self.assertFalse(launcher.show_success_popup(SUCCESS))

    def test_06_non_windows_does_not_show(self):
        with patch.object(launcher.os,'name','posix'),patch.object(launcher,'_show_message_box') as box:
            self.assertFalse(launcher.show_success_popup(SUCCESS));box.assert_not_called()

    def test_07_native_adapter_uses_unicode_messagebox_without_launching_process(self):
        box=Mock(return_value=1)
        windll=types.SimpleNamespace(user32=types.SimpleNamespace(MessageBoxW=box))
        with patch.object(ctypes,'windll',windll,create=True):
            self.assertEqual(launcher._show_message_box('启动成功','中文提示'),1)
        box.assert_called_once_with(None,'中文提示','启动成功',0x40|0x10000|0x40000)
        self.assertEqual(len(box.argtypes),4);self.assertIs(box.restype,ctypes.c_int)

    def invoke_main(self,*,status=False,operate_error=None,popup_value=True,popup_error=None):
        released=[]
        class Lock:
            def __enter__(self):released.append(False)
            def __exit__(self,*args):released[-1]=True
        backend=types.SimpleNamespace(runtime=Path('unused-runtime'),settings={'provider':'chatgpt_plan'},
            napcat_running=lambda:True,bots=lambda:[])
        output=io.StringIO()
        def popup(result):
            self.assertEqual(released,[True],'Popup must not hold the launch lock')
            if popup_error:raise popup_error
            return popup_value
        with patch.object(sys,'argv',['launcher.py']+(['--status'] if status else [])),\
             patch.object(launcher,'RealBackend',return_value=backend),\
             patch.object(launcher,'ExclusiveLauncherLock',return_value=Lock()) as lock,\
             patch.object(launcher,'operate',return_value=dict(SUCCESS),side_effect=operate_error) as operation,\
             patch.object(launcher,'show_success_popup',side_effect=popup) as notice,\
             contextlib.redirect_stdout(output):
            result=launcher.main()
        return result,output.getvalue(),operation,notice,lock

    def test_08_main_success_once_after_lock_release(self):
        result,text,operation,notice,_=self.invoke_main()
        self.assertEqual(result,0);operation.assert_called_once();notice.assert_called_once_with(SUCCESS)
        self.assertNotIn('无法显示',text)

    def test_09_status_does_not_operate_or_show(self):
        result,text,operation,notice,lock=self.invoke_main(status=True)
        self.assertEqual(result,0);operation.assert_not_called();notice.assert_not_called();lock.assert_not_called()
        self.assertIn('read_only',text)

    def test_10_launcher_failure_never_shows_success(self):
        for error in (launcher.LauncherError('NAPCAT_NOT_READY'),RuntimeError('SYNTHETIC_SECRET')):
            with self.subTest(error=type(error).__name__):
                result,text,_,notice,_=self.invoke_main(operate_error=error)
                self.assertEqual(result,1);notice.assert_not_called();self.assertNotIn('SYNTHETIC_SECRET',text)

    def test_11_popup_false_does_not_undo_success(self):
        result,text,operation,notice,_=self.invoke_main(popup_value=False)
        self.assertEqual(result,0);operation.assert_called_once();notice.assert_called_once()
        self.assertIn('不影响后台运行',text)

    def test_12_popup_exception_does_not_leak_or_mark_bot_failed(self):
        result,text,operation,notice,_=self.invoke_main(popup_error=RuntimeError('SYNTHETIC_SECRET'))
        self.assertEqual(result,0);operation.assert_called_once();notice.assert_called_once()
        self.assertIn('不影响后台运行',text);self.assertNotIn('SYNTHETIC_SECRET',text)

if __name__=='__main__':unittest.main(verbosity=2)
