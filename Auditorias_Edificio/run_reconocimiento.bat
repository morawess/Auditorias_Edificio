@echo off
cd /d "%~dp0"
echo ================================================================
echo   Iniciando Comparacion Facial 1:1
echo ================================================================
call .venv\Scripts\python.exe reconocimiento_facial.py
pause
