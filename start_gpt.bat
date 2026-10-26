@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" set "PATH=%~dp0.venv\Scripts;%PATH%"
echo ChatGPT 套餐模式：使用本机官方登录授权，不使用 API Key。
echo 若尚未授权，请先关闭本窗口并双击 login_chatgpt.bat。
echo 请勿与另一个机器人实例同时运行，以免重复回复。
python app/bot.py --config config.json --provider chatgpt_plan %*
echo.
echo 进程已退出，按任意键关闭窗口。
pause >nul
