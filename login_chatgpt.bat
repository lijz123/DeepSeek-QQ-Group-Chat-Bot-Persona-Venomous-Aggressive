@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" set "PATH=%~dp0.venv\Scripts;%PATH%"
echo 仅在 OpenAI 官方浏览器页面登录并授权；不要将密码或验证码发给他人。
echo 此步骤只登录，不发送模型测试请求，也不会登录 QQ。
python app/chatgpt_auth.py --login %*
echo.
pause
