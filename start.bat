@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" set "PATH=%~dp0.venv\Scripts;%PATH%"
if not exist "config.json" (
    if not exist "config.example.json" (
        echo Missing config.example.json. Extract the complete release first.
        pause
        exit /b 1
    )
    copy /y "config.example.json" "config.json" >nul
    if errorlevel 1 exit /b 1
    echo Created config.json. Configure your API key and NapCat settings first.
    echo Read README.md, then run this file again. No login or bot was started.
    pause
    exit /b 1
)
python -u app/bot.py --config config.json %*
set "BOT_EXIT=%ERRORLEVEL%"
if not "%BOT_EXIT%"=="0" (
    echo Startup did not complete. See the message above.
    pause
    exit /b %BOT_EXIT%
)
echo Done. NapCat was not stopped by this script.
pause
exit /b 0
