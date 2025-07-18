@echo off
REM AGI Daily Monitoring Script
REM Run this daily to check your AGI's learning progress

echo ========================================
echo TRUE AGI DAILY MONITORING REPORT
echo ========================================
echo.

REM Quick status check
echo Running AGI Monitor Dashboard...
python scripts\agi_monitor_dashboard.py

echo.
echo ========================================
echo Additional monitoring options:
echo.
echo 1. For detailed analysis: python scripts\learning_analyzer.py  
echo 2. For physics discoveries: python scripts\physics_dashboard.py
echo 3. For live system check: python scripts\live_agi_check.py
echo 4. For intelligence test: python scripts\simple_intelligence_test.py
echo 5. For continuous monitoring: python scripts\agi_monitor_dashboard.py --continuous
echo.
echo ========================================

pause
