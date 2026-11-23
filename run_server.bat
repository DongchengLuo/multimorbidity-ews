@echo off
REM ===========================================================================
REM  Multimorbidity Early Warning System - Windows launcher
REM  Usage:  run_server.bat [PORT]
REM  Interpreter priority: %PYTHON% > py -3 > python
REM ===========================================================================
setlocal
cd /d "%~dp0"

if "%~1"=="" (set PORT=5001) else (set PORT=%~1)

REM Force UTF-8 so Chinese log output does not crash on cp936 consoles
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
if "%JEV_DECIDE_CONCURRENCY%"=="" set JEV_DECIDE_CONCURRENCY=8
if "%JEV_DECIDE_TIMEOUT%"=="" set JEV_DECIDE_TIMEOUT=600
if "%JEV_DECIDE_MAX_RETRIES%"=="" set JEV_DECIDE_MAX_RETRIES=0

set "PY="
if not "%PYTHON%"=="" (
    set "PY=%PYTHON%"
) else (
    py -3 -c "import sys" >nul 2>&1 && set "PY=py -3"
)
if "%PY%"=="" (
    python -c "import sys" >nul 2>&1 && set "PY=python"
)
if "%PY%"=="" (
    echo [ERROR] Python 3.10+ not found. Install it or set the PYTHON variable.
    exit /b 1
)

echo Using interpreter: %PY%
echo Starting on http://0.0.0.0:%PORT%
%PY% -u -c "from app import app; app.run(host='0.0.0.0', port=%PORT%, debug=False, threaded=True)"
endlocal
