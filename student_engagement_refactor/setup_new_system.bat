@echo off
:: ============================================================
:: setup_new_system.bat
:: Double-click this file to set up the project on a new Windows PC.
:: Requirements: Python 3.8+ and Git must be installed first.
::   Python: https://www.python.org/downloads/   (check "Add to PATH")
::   Git:    https://git-scm.com/download/win
:: ============================================================

echo.
echo ============================================================
echo  Student Engagement Detection System  -  System Setup
echo ============================================================
echo.

:: Check PowerShell is available (it always is on Windows 7+)
where powershell >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PowerShell not found. Cannot continue.
    pause
    exit /b 1
)

:: Run the PowerShell script, bypassing execution policy for this session only
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_new_system.ps1"

if errorlevel 1 (
    echo.
    echo [ERROR] Setup encountered errors. Review the output above.
    pause
    exit /b 1
)

echo.
echo Setup finished. Press any key to exit.
pause >nul
