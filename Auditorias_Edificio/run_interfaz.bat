@echo off
cd /d "%~dp0"
echo ================================================================
echo   Iniciando Interfaz Grafica del Sistema
echo ================================================================
call .venv\Scripts\python.exe interfaz_grafica.py
if errorlevel 1 (
    echo.
    echo Ocurrio un error al ejecutar la interfaz grafica.
    pause
)
