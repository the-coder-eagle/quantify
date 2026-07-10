@echo off
title Quantify AI

set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "PORT=57921"
set "VENV=%PROJECT_DIR%\.venv"
set "STREAMLIT=%VENV%\Scripts\streamlit.exe"

cd /d "%PROJECT_DIR%"

echo.
echo ============================================================
echo   Quantify AI - Trading Decision System
echo ============================================================
echo.

:: Step 1: Clean zombie port
echo [1/4] Checking port %PORT% ...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT% " 2^>nul') do (
    echo   Killing process %%a on port %PORT%
    taskkill /F /PID %%a 2>nul
)
echo   Port ready

:: Step 2: Check venv
echo [2/4] Checking Python environment ...
if not exist "%VENV%\Scripts\python.exe" (
    echo   ERROR: .venv not found at %VENV%
    echo   Please create venv: python -m venv .venv
    pause
    exit /b 1
)
call "%VENV%\Scripts\activate.bat"

:: Step 3: Install / update dependencies
echo [3/4] Installing dependencies ...
pip install -r "%PROJECT_DIR%\requirements.txt" -q
if %errorlevel% neq 0 (
    echo   WARNING: Some packages failed to install
    echo   Trying to install core packages...
    pip install streamlit plotly requests akshare pandas numpy loguru pyyaml openai -q
)
echo   Dependencies OK

:: Step 4: Launch
echo [4/4] Starting dashboard ...
echo.
echo   URL: http://localhost:%PORT%
echo   Press Ctrl+C to stop
echo.
start "" http://localhost:%PORT%
"%VENV%\Scripts\streamlit.exe" run "%PROJECT_DIR%\app.py" --server.port %PORT% --server.headless true

pause
