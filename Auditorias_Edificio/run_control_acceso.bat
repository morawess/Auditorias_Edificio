@echo off
cd /d "%~dp0"
echo ================================================================
echo   Iniciando Control de Acceso y Auditoria (Camara Web)
echo ================================================================
call .venv\Scripts\python.exe control_acceso_residentes.py
pause
