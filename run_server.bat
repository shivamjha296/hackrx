@echo off
echo =====================================================
echo HackRX 5.0 - Intelligent Query-Retrieval System
echo =====================================================
echo Starting server...
echo.

REM Activate virtual environment
call myenv\Scripts\activate.bat

REM Start the FastAPI server
python start_server.py

pause
