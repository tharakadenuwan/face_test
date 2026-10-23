@echo off
set SCRIPT_DIR=%~dp0
if exist "E:\face_venv\Scripts\python.exe" (
    "E:\face_venv\Scripts\python.exe" "%SCRIPT_DIR%predict.py" %*
) else (
    python "%SCRIPT_DIR%predict.py" %*
)
