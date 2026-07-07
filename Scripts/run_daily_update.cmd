@echo off
REM Daily data refresh + model retrain, run by Windows Task Scheduler.
REM Logs each run to Logs\update_YYYY-MM-DD.log. Safe to run by hand too.

set "ROOT=%~dp0.."
set "PY=C:\Users\gdsak\AppData\Local\Programs\Python\Python313\python.exe"
if not exist "%ROOT%\Logs" mkdir "%ROOT%\Logs"

REM locale-independent date (%date% includes the weekday on some locales,
REM which produced misnamed logs like update_06-Mon-07.log)
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set "TODAY=%%d"
set "LOG=%ROOT%\Logs\update_%TODAY%.log"

echo ==================================================================>> "%LOG%"
echo Run started %date% %time% >> "%LOG%"
"%PY%" "%ROOT%\Scripts\update_all.py" --retrain >> "%LOG%" 2>&1
echo Run finished %date% %time% (exit %ERRORLEVEL%) >> "%LOG%"
exit /b %ERRORLEVEL%
