@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist config.json copy /y config.example.json config.json >nul
if "%DEEPSEEK_API_KEY%"=="" (
    echo [!] 请先在当前窗口设置 DEEPSEEK_API_KEY 环境变量，再启动。
    pause
    exit /b 1
)
python bot.py --config config.json
pause
