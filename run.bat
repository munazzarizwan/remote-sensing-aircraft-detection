@echo off
cd /d %~dp0
call yolo_env\Scripts\activate.bat
python predict_script.py
pause
