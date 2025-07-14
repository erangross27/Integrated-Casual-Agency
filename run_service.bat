@echo off
REM Continuous Learning Service for Windows
REM Keeps the ICA Framework running indefinitely

echo ========================================
echo ICA Framework - Continuous Learning Service
echo ========================================
echo.

:LOOP
echo [%date% %time%] Starting learning session...

REM Run the learning script
python run_continuous.py

REM Check if we should restart (exit code 0 means restart)
if %errorlevel% equ 0 (
    echo [%date% %time%] Session completed, restarting in 3 seconds...
    timeout /t 3 /nobreak >nul
    goto LOOP
) else (
    echo [%date% %time%] Learning stopped with code %errorlevel%
    goto END
)

:END
echo [%date% %time%] Continuous learning service stopped
pause
