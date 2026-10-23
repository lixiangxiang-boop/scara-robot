@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
"C:\Python312\python.exe" "%~dp0platform\menu.py" %*
set "SCARA_EXIT=%ERRORLEVEL%"
if not "%SCARA_EXIT%"=="0" pause
exit /b %SCARA_EXIT%
