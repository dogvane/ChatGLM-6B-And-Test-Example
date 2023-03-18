@echo off

setlocal

call conda.bat activate chatgpt
python web_demo.py

endlocal

pause


