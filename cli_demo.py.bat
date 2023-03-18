@echo off

setlocal

call conda.bat activate chatgpt
python cli_demo.py

endlocal

pause


