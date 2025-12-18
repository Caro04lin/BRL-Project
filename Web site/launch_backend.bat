@echo off
REM launch_backend.bat
REM This batch file runs the Flask backend with the virtual environment Python

REM Go to the folder of this batch file
cd /d "%~dp0"

REM Path to the AI project folder (relative to this script)
SET AI_PROJECT=%~dp0..\Multimodal-Human-Intention-Detection-for-Upper-Limb-Exoskeleton-Assistance-in-Construction-Work-main

REM Path to Python inside the virtual environment
SET VENV_PYTHON=%AI_PROJECT%\.venv\Scripts\python.exe

REM Check if the virtual env Python exists
IF NOT EXIST "%VENV_PYTHON%" (
    ECHO Virtual environment Python not found. Using system Python.
    SET VENV_PYTHON=python
)

REM Run the Flask backend
"%VENV_PYTHON%" "%~dp0backend\server.py"


PAUSE


