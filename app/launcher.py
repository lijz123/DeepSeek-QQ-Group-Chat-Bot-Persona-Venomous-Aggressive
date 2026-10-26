"""Windows one-click launcher: retain NapCat, restart only this project's bot.

Import and --status are read-only. There is no taskkill-by-name, QQ stop,
registry edit, model test, browser-token access, or API-key fallback here.
"""
from __future__ import annotations
import argparse, asyncio, contextlib, json, math, os, re, secrets, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from project_paths import PROJECT_ROOT, bot_script_path

try:
    import psutil
except ImportError:
    psutil = None

class LauncherError(Exception):
    def __init__(self, code):
        self.code = str(code)
        super().__init__(self.code)

def _open_login_qr(path):
    """Display only the local NapCat login image; the user confirms on the phone."""
    if os.name != 'nt':return False
    os.startfile(str(path))
    return True

def reject_links(path):
    raw=Path(path)
    for part in (raw,*raw.parents):
        if part.is_symlink() or (hasattr(part,'is_junction') and part.is_junction()):
            raise LauncherError('LINKED_RUNTIME_PATH_REFUSED')

@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    name: str
    exe: str
    cwd: str
    argv: tuple[str, ...]
    created: float
    owner: str

def same_path(a, b):
    try:
        return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))
    except (OSError, ValueError, TypeError):
        return False

def script_argument(argv):
    """The actual script operand, not any occurrence of bot.py inside arguments."""
    args = list(argv)[1:]
    i = 0
    while i < len(args):
        value = args[i]
        if value in ('-c', '-m', '-') or value.startswith(('-c', '-m')):
            return None
        if value == '--':
            return args[i+1] if i+1 < len(args) else None
        if value in ('-X', '-W'):
            i += 2
            continue
        if value.startswith('-'):
            if value in ('-u','-B','-E','-I','-s','-S','-O','-OO','-q','-P') or value.startswith(('-X','-W')):
                i += 1
                continue
            return None
        return value
    return None

def is_bot_process(info: ProcessInfo, project: Path, owner: str) -> bool:
    if (isinstance(info.pid,bool) or not isinstance(info.pid,int) or info.pid<=0
            or not isinstance(info.created,(int,float)) or not math.isfinite(info.created) or info.created<=0):
        return False
    if info.owner.casefold() != owner.casefold() or not owner:
        return False
    if Path(info.exe).name.lower() not in {'python.exe','pythonw.exe','python','python3'}:
        return False
    operand = script_argument(info.argv)
    if not operand:
        return False
    script = Path(operand)
    if not script.is_absolute():
        if not info.cwd:
            return False
        script = Path(info.cwd) / script
    if not same_path(script, bot_script_path(project)):
        return False
    # A different config in the same source folder can be a separate bot.
    arguments=list(info.argv)
    configs=[]
    for index,value in enumerate(arguments):
        if value=='--config':
            if index+1>=len(arguments):return False
            configs.append(arguments[index+1])
        elif value.startswith('--config='):configs.append(value[len('--config='):])
    if len(configs)>1:return False
    config=Path(configs[0]) if configs else Path('config.json')
    if not config.is_absolute():
        if not info.cwd:return False
        config=Path(info.cwd)/config
    return same_path(config,Path(project)/'config.json')

def is_napcat_process(info: ProcessInfo, napcat_dir: Path, owner: str) -> bool:
    return (bool(owner) and info.owner.casefold() == owner.casefold()
            and same_path(info.exe, Path(napcat_dir)/'qq_extract'/'Files'/'QQ.exe')
            and not any(a.startswith('--type=') or a == '--type' for a in info.argv))

def deepest_first(bots):
    parents = {p.pid:p.ppid for p in bots}
    def depth(p):
        seen=set();pid=p.pid
        while pid in parents and pid not in seen:
            seen.add(pid);pid=parents[pid]
        return len(seen)
    return sorted(bots,key=depth,reverse=True)

def operate(backend, *, action='restart'):
    if action != 'restart':
        raise LauncherError('UNSUPPORTED_ACTION')
    backend.preflight()  # Never stop a working bot if the replacement cannot run.
    retained = backend.napcat_running()
    if not retained:
        backend.start_napcat()
    if not backend.wait_napcat():
        raise LauncherError('NAPCAT_NOT_READY_OLD_BOT_RETAINED')
    old = backend.bots()
    for process in deepest_first(old):
        backend.stop_bot(process)
    if backend.bots():
        raise LauncherError('OLD_BOT_STILL_PRESENT_NO_DUPLICATE_STARTED')
    new = backend.start_bot()
    return {'napcat_retained':retained,'napcat_started':not retained,
            'old_bot_count':len(old),'bot_pid':new.pid,'bot_started':True,
            'model_test_requests':0}

class ExclusiveLauncherLock:
    def __init__(self,path):self.path=Path(path);self.file=None
    def __enter__(self):
        reject_links(self.path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.is_symlink():raise LauncherError('UNSAFE_CONTROL_PATH')
        self.file=self.path.open('a+b')
        try:
            import msvcrt
            self.file.seek(0);msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            if self.file.seek(0,os.SEEK_END)==0:self.file.write(b'0');self.file.flush()
            return self
        except OSError:
            self.file.close();self.file=None
            raise LauncherError('ANOTHER_LAUNCH_IS_RUNNING') from None
    def __exit__(self,*args):
        if self.file:
            import msvcrt
            self.file.seek(0);msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1)
            self.file.close()

def _safe_stop_path(project, path):
    reject_links(path)
    p=Path(path).resolve();base=Path(project).resolve()/'logs'/'oneclick'
    if p.parent!=base or not re.fullmatch(r'stop-[0-9a-f]{32}\.flag',p.name) or p.is_symlink():
        raise LauncherError('UNSAFE_STOP_REQUEST_PATH')
    return p

async def run_bot_until_stop(bot, stop_file):
    """Cancellation executes Bot.run's existing finally/close cleanup."""
    path=_safe_stop_path(PROJECT_ROOT,stop_file)
    if path.exists():raise LauncherError('STOP_REQUEST_ALREADY_PRESENT')
    running=asyncio.create_task(bot.run())
    try:
        while not running.done():
            if path.is_file():
                running.cancel()
                with contextlib.suppress(asyncio.CancelledError):await running
                return
            await asyncio.sleep(0.2)
        await running
    finally:
        if not running.done():
            running.cancel()
            with contextlib.suppress(asyncio.CancelledError):await running

class RealBackend:
    def __init__(self, project=None, settings=None):
        if psutil is None:raise LauncherError('INSTALL_PSUTIL_REQUIREMENTS_FIRST')
        self.project=Path(project or PROJECT_ROOT).resolve()
        self.owner=psutil.Process().username()
        self.runtime=self.project/'logs'/'oneclick'
        self.settings={'provider':'chatgpt_plan','napcat_dir':'../napcat',
                       'napcat_user_data':'qqappdata2/QQ','wait_seconds':120}
        settings_path=self.project/'launcher.local.json'
        if settings_path.is_file():
            value=json.loads(settings_path.read_text(encoding='utf-8-sig'))
            if not isinstance(value,dict):raise LauncherError('INVALID_LAUNCHER_SETTINGS')
            self.settings.update(value)
        if settings:self.settings.update(settings)
        self.napcat=(self.project/str(self.settings['napcat_dir'])).resolve()
        data=Path(str(self.settings['napcat_user_data']))
        self.user_data=(data if data.is_absolute() else self.napcat/data).resolve()
        self.cfg=None
        self._last_onebot_issue='not_checked'
        self._login_qr_attempted=False

    @staticmethod
    def info(process):
        return ProcessInfo(process.pid,process.ppid(),process.name(),process.exe(),process.cwd(),
                           tuple(process.cmdline()),process.create_time(),process.username())

    def processes(self):
        result=[]
        for process in psutil.process_iter():
            try:
                if process.name().lower() not in {'python.exe','pythonw.exe','python','python3','qq.exe'}:continue
                result.append(self.info(process))
            except psutil.NoSuchProcess:continue
            except (psutil.AccessDenied,OSError):
                # Cannot prove a Python process belongs elsewhere: fail closed.
                try:python=process.name().lower().startswith('python')
                except psutil.Error:python=False
                if python:raise LauncherError('PROCESS_IDENTITY_PERMISSION_DENIED') from None
        return result

    def bots(self):
        return [p for p in self.processes() if is_bot_process(p,self.project,self.owner)]

    def napcat_running(self):
        return any(is_napcat_process(p,self.napcat,self.owner) for p in self.processes())

    def preflight(self):
        if os.name!='nt':raise LauncherError('WINDOWS_ONLY')
        from bot import load_config, Brain
        self.cfg=load_config(self.project/'config.json',provider=self.settings['provider'])
        if not bot_script_path(self.project).is_file():raise LauncherError('BOT_FILE_MISSING')
        reject_links(bot_script_path(self.project))
        brain=Brain(self.cfg['llm'],self.cfg['safety'],self.cfg['persona'],self.cfg['behavior']['max_chars'])
        if not brain._provider_ready():raise LauncherError('MODEL_AUTH_NOT_READY_OLD_BOT_RETAINED')
        if self.settings['provider']=='chatgpt_plan':
            from chatgpt_auth import PlanProfile
            PlanProfile(self.cfg['llm'].get('chatgpt_profile') or None)._read() # local DPAPI only; no network/refresh.
        uri=urlparse(str(self.cfg['onebot']['ws_url']))
        if uri.scheme not in ('ws','wss') or uri.hostname not in ('127.0.0.1','localhost','::1'):
            raise LauncherError('LOCAL_ONEBOT_REQUIRED')
        if not self.user_data.is_relative_to(self.napcat):raise LauncherError('NAPCAT_DATA_OUTSIDE_INSTALL')
        for path in (self.project,self.runtime,self.napcat,self.user_data):
            reject_links(path)
        if not self.napcat_running():
            for name in ('launch.ps1','runtime/NapCatWinBootHook.dll','runtime/napcat.mjs','qq_extract/Files/QQ.exe'):
                if not (self.napcat/name).is_file():raise LauncherError('NAPCAT_INSTALLATION_NOT_FOUND')

    def start_napcat(self):
        if self.napcat_running():return
        self.runtime.mkdir(parents=True,exist_ok=True)
        executable=Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32/WindowsPowerShell/v1.0/powershell.exe'
        env=os.environ.copy()
        # Avoid PowerShell 7 module paths breaking Windows PowerShell 5.1 helpers.
        env['PSModulePath']=str(executable.parent/'Modules')+os.pathsep+str(Path(os.environ.get('ProgramFiles',r'C:\Program Files'))/'WindowsPowerShell/Modules')
        with (self.runtime/'napcat-start.log').open('ab') as output:
            p=subprocess.Popen([str(executable),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',
                    str(self.project/'start_napcat_safe.ps1'),'-NapCatDir',str(self.napcat),'-UserDataDir',str(self.user_data)],
                    cwd=self.project,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
        self.napcat_starter=p
        print('正在启动 NapCat；如需要登录，请在 QQ/NapCat 界面扫码。',flush=True)

    async def _onebot_ready(self):
        import aiohttp
        from bot import resolve_secret
        headers={};token=resolve_secret(self.cfg['onebot'].get('access_token',''))
        if token:headers['Authorization']='Bearer '+token
        try:
            self._last_onebot_issue='not_connected'
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3),cookie_jar=aiohttp.DummyCookieJar()) as session:
                async with session.ws_connect(self.cfg['onebot']['ws_url'],headers=headers,proxy=None) as ws:
                    login_echo=secrets.token_hex(8);status_echo=secrets.token_hex(8)
                    await ws.send_json({'action':'get_login_info','params':{},'echo':login_echo})
                    await ws.send_json({'action':'get_status','params':{},'echo':status_echo})
                    ready={};deadline=time.monotonic()+2
                    for _ in range(100):
                        remaining=deadline-time.monotonic()
                        if remaining<=0:break
                        message=await ws.receive(timeout=remaining)
                        if message.type!=aiohttp.WSMsgType.TEXT:continue
                        data=json.loads(message.data)
                        if not isinstance(data,dict):continue
                        echo=data.get('echo')
                        if echo in (login_echo,status_echo):
                            ok=data.get('status')=='ok' and data.get('retcode',0)==0
                            detail=data.get('data') or {}
                            if not isinstance(detail,dict):
                                self._last_onebot_issue='invalid_response';return False
                            ready[echo]=ok and (bool(detail.get('user_id')) if echo==login_echo else detail.get('online') is True)
                            if not ready[echo]:
                                self._last_onebot_issue='login_pending' if echo==login_echo else 'offline'
                            if len(ready)==2:
                                if all(ready.values()):self._last_onebot_issue='ready';return True
                                return False
                    self._last_onebot_issue='timeout'
        except aiohttp.WSServerHandshakeError as error:
            self._last_onebot_issue='auth_failed' if error.status in (401,403) else 'not_connected'
        except asyncio.TimeoutError:self._last_onebot_issue='timeout'
        except (ValueError,TypeError):self._last_onebot_issue='invalid_response'
        except aiohttp.ClientError:self._last_onebot_issue='not_connected'
        return False

    def _maybe_show_login_qr(self):
        if getattr(self,'_login_qr_attempted',False) or os.name!='nt':return False
        if getattr(self,'_last_onebot_issue','') not in ('not_connected','login_pending','offline'):return False
        image=self.napcat/'runtime'/'cache'/'qrcode.png'
        try:
            reject_links(image)
            stat=image.stat()
            if not image.is_file() or not 0<=time.time()-stat.st_mtime<=90 or not 8<=stat.st_size<=8*1024*1024:return False
            with image.open('rb') as source:
                if source.read(8)!=b'\x89PNG\r\n\x1a\n':return False
            self._login_qr_attempted=True
            print('NapCat 正在等待手机扫码。请用机器人 QQ 扫码，并在手机上确认登录。',flush=True)
            print('二维码图片：'+str(image),flush=True)
            if _open_login_qr(image):return True
        except (OSError,LauncherError):pass
        if getattr(self,'_login_qr_attempted',False):
            print('无法自动显示二维码，请打开上面的图片；过期后重新运行本脚本。',flush=True)
        return False

    def wait_napcat(self):
        seconds=max(1,min(600,float(self.settings.get('wait_seconds',120))))
        started=time.monotonic();deadline=started+seconds;next_report=started
        print('等待 NapCat 登录和 OneBot 接口就绪；未就绪前不会停止旧机器人。',flush=True)
        reasons={'not_connected':'本地 OneBot 接口尚未开启，可能还在等扫码登录',
                 'login_pending':'QQ 尚未完成登录，请扫码并在手机上确认',
                 'offline':'QQ 当前不在线，请检查 NapCat 登录状态',
                 'auth_failed':'OneBot 访问令牌不匹配，请检查两端的 access_token',
                 'invalid_response':'本地接口返回的内容不是有效的 OneBot 状态',
                 'timeout':'本地接口没有及时返回登录/在线状态',
                 'process_missing':'尚未找到本次配置的 NapCat 主进程'}
        while time.monotonic()<deadline:
            if self.napcat_running():
                if asyncio.run(self._onebot_ready()):
                    print('NapCat 已登录，OneBot 接口已就绪。',flush=True);return True
                self._maybe_show_login_qr()
            else:self._last_onebot_issue='process_missing'
            starter=getattr(self,'napcat_starter',None)
            if starter is not None and starter.poll() not in (None,0):raise LauncherError('NAPCAT_START_FAILED_CHECK_LOG')
            now=time.monotonic()
            if now>=next_report:
                print('已等待 '+str(int(now-started))+' 秒，还剩约 '+str(max(0,int(deadline-now)))+' 秒：'+
                      reasons.get(getattr(self,'_last_onebot_issue',''),'正在检查登录状态')+'。',flush=True)
                next_report=now+10
            time.sleep(1)
        print('等待结束：'+reasons.get(getattr(self,'_last_onebot_issue',''),'NapCat/OneBot 尚未就绪')+'。',flush=True)
        return False

    def _fresh_target(self, expected):
        try:
            p=psutil.Process(expected.pid);info=self.info(p)
            if (not math.isfinite(expected.created) or info.created!=expected.created
                    or not same_path(info.exe,expected.exe)
                    or not is_bot_process(info,self.project,self.owner)):
                raise LauncherError('PROCESS_IDENTITY_CHANGED_NO_STOP')
            return p
        except psutil.NoSuchProcess:return None
        except psutil.AccessDenied:raise LauncherError('PROCESS_STOP_PERMISSION_DENIED') from None

    def stop_bot(self, expected):
        p=self._fresh_target(expected)
        if p is None:return
        state_path=self.runtime/'managed-bot.json'
        try:
            reject_links(state_path)
            state=json.loads(state_path.read_text(encoding='utf-8'))
        except (OSError,ValueError):state={}
        if (state.get('pid')==expected.pid and state.get('created')==expected.created
                and same_path(state.get('project',''),self.project)):
            stop=_safe_stop_path(self.project,state.get('stop_file',''))
            stop.parent.mkdir(parents=True,exist_ok=True);stop.write_text('stop\n',encoding='ascii')
            try:p.wait(timeout=8);return
            except psutil.TimeoutExpired:pass
        p=self._fresh_target(expected)
        if p is None:return
        # Legacy Python processes have no cooperative stop endpoint. This is
        # an exact-PID forced stop, never a process-name or descendant-tree kill.
        p.terminate()
        try:p.wait(timeout=5)
        except psutil.TimeoutExpired:raise LauncherError('OLD_BOT_DID_NOT_STOP_NO_NEW_INSTANCE') from None

    def start_bot(self):
        self.runtime.mkdir(parents=True,exist_ok=True)
        stop=self.runtime/('stop-'+secrets.token_hex(16)+'.flag')
        command=[sys.executable,str(bot_script_path(self.project)),'--config',str(self.project/'config.json'),
                 '--provider',self.settings['provider'],'--managed-stop-file',str(stop)]
        env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
        with (self.runtime/'bot-console.log').open('ab') as output:
            p=subprocess.Popen(command,cwd=self.project,env=env,stdin=subprocess.DEVNULL,
                    stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            time.sleep(2)
            if p.poll() is not None:raise LauncherError('NEW_BOT_EXITED_CHECK_LOG')
            info=self.info(psutil.Process(p.pid))
            if not is_bot_process(info,self.project,self.owner):raise LauncherError('NEW_BOT_IDENTITY_UNCONFIRMED')
            state={'pid':info.pid,'created':info.created,'project':str(self.project),'stop_file':str(stop),'provider':self.settings['provider']}
            temp=self.runtime/('managed-bot-'+secrets.token_hex(8)+'.tmp')
            reject_links(temp);reject_links(self.runtime/'managed-bot.json')
            with temp.open('x',encoding='utf-8') as f:json.dump(state,f,indent=2)
            os.replace(temp,self.runtime/'managed-bot.json')
            return info
        except BaseException:
            # Cleanup only the process handle we just created, not a stale PID.
            if p.poll() is None:
                p.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):p.wait(timeout=5)
            raise

def _show_message_box(title, text):
    import ctypes
    from ctypes import wintypes
    message_box=ctypes.windll.user32.MessageBoxW
    message_box.argtypes=[wintypes.HWND,wintypes.LPCWSTR,wintypes.LPCWSTR,wintypes.UINT]
    message_box.restype=ctypes.c_int
    # Native Unicode dialog, information icon, foreground; no extra UI dependency.
    return message_box(None,text,title,0x40 | 0x10000 | 0x40000)

def show_success_popup(result):
    if not result.get('bot_started') or os.name!='nt':return False
    restarted=bool(result.get('old_bot_count'))
    text=('机器人已重启成功！' if restarted else '机器人已启动成功！')
    text+='\n\nNapCat 保持在线，机器人在后台运行。\n点击“确定”即可关闭提示。'
    return _show_message_box('启动成功',text)==1

def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            with contextlib.suppress(Exception):stream.reconfigure(encoding='utf-8',errors='replace')
    parser=argparse.ArgumentParser(description='一键启动 NapCat 和 GPT 机器人；再次运行只重启机器人')
    parser.add_argument('--status',action='store_true',help='只读检查，不登录、不停止、不启动')
    args=parser.parse_args()
    try:
        backend=RealBackend()
        if args.status:
            print(json.dumps({'napcat_running':backend.napcat_running(),'bot_pids':[p.pid for p in backend.bots()],
                              'default_provider':backend.settings['provider'],'read_only':True},ensure_ascii=False));return 0
        print('一键启动或重启机器人（GPT 套餐）',flush=True)
        print('NapCat 已启动时会保留；只处理本目录的机器人。正在检查启动条件……',flush=True)
        with ExclusiveLauncherLock(backend.runtime/'launcher.lock'):
            result=operate(backend)
        print('完成：NapCat 保持在线，机器人已启动/重启（GPT 套餐模式）。')
        print('机器人在后台运行，可以关闭此窗口。日志：logs\\oneclick\\bot-console.log')
        try:
            if not show_success_popup(result):
                print('机器人已启动，但成功提示窗口未显示；不影响后台运行。')
        except Exception:
            print('机器人已启动，但成功提示窗口无法显示；不影响后台运行。')
        print(json.dumps(result,ensure_ascii=False));return 0
    except LauncherError as error:
        print('操作停止：'+error.code)
        if error.code=='NAPCAT_NOT_READY_OLD_BOT_RETAINED':
            print('请先完成 NapCat 的 QQ 扫码登录，确认本地 OneBot 接口开启，再双击本脚本。')
            print('这次没有执行停止旧机器人的步骤；这个错误不表示 GPT 授权或依赖需要重新安装。')
        print('未按名称批量关闭进程，也不会为重启机器人关闭 NapCat。请查看一键启动说明.md。');return 1
    except Exception:
        print('操作未完成：请检查依赖、授权及 logs\\oneclick 日志；不会自动改用其他模型计费。');return 1

if __name__=='__main__':raise SystemExit(main())
