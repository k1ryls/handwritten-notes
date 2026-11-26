@echo off
chcp 65001 >nul
cd /d "%~dp0"
".venv\Scripts\python.exe" -c "import cv2, numpy, PIL" >nul 2>&1
if errorlevel 1 (
    echo Первый запуск: ставлю библиотеки, это займёт минуту...
    if exist ".venv" rmdir /s /q ".venv"
    py -3.13 -m venv .venv 2>nul || py -3 -m venv .venv 2>nul || python -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo Не найден Python. Установите Python 3.10+ с python.org и запустите снова.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
)
start "" ".venv\Scripts\pythonw.exe" app.py
